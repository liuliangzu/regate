import os
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# -----------------------------
# Data Loading (Bitcoin OTC)
# -----------------------------
def load_btc_data(csv_path, batch_size=100, mode='absolute',
                  pos_thresh=6, neg_thresh=-1, pos_q=0.9, neg_q=0.1):
    edge_cols = ['src_raw', 'dst_raw', 'rating', 'unix_ts']
    edge_df = pd.read_csv(csv_path, header=None, names=edge_cols)

    all_ids = pd.unique(pd.concat([edge_df['src_raw'], edge_df['dst_raw']], ignore_index=True))
    id2idx = pd.Series(np.arange(len(all_ids), dtype=np.int64), index=all_ids).to_dict()
    edge_df['src'] = edge_df['src_raw'].map(id2idx)
    edge_df['dst'] = edge_df['dst_raw'].map(id2idx)
    edge_df = edge_df.dropna(subset=['src', 'dst']).reset_index(drop=True)
    edge_df['src'] = edge_df['src'].astype('int64')
    edge_df['dst'] = edge_df['dst'].astype('int64')
    num_nodes = len(all_ids)

    min_ts = int(edge_df['unix_ts'].min())
    sec_per_day = 24 * 3600
    edge_df['t'] = ((edge_df['unix_ts'].astype('int64') - min_ts) // sec_per_day).astype('int64')

    t_np = edge_df['t'].values
    t_q70, t_q75, t_q80 = np.quantile(t_np, [0.7, 0.75, 0.8]).astype(int)
    t_end = int(t_np.max())

    train_mask_time = edge_df['t'] <= t_q75
    train_ratings = edge_df.loc[train_mask_time, 'rating'].values

    if mode == 'absolute':
        pos_thr, neg_thr = pos_thresh, neg_thresh
    elif mode == 'quantile':
        if len(train_ratings) == 0:
            raise ValueError("No samples in training window for quantile thresholding.")
        pos_thr = math.ceil(np.quantile(train_ratings, pos_q))
        neg_thr = math.floor(np.quantile(train_ratings, neg_q))
    else:
        raise ValueError("mode must be 'absolute' or 'quantile'")

    mask_keep = (edge_df['rating'] >= pos_thr) | (edge_df['rating'] <= neg_thr)
    edge_df = edge_df[mask_keep].reset_index(drop=True)
    y_arr = (edge_df['rating'].values >= pos_thr).astype('int64')

    data = TemporalData(
        src=torch.from_numpy(edge_df['src'].values),
        dst=torch.from_numpy(edge_df['dst'].values),
        t=torch.from_numpy(edge_df['t'].values),
        msg=torch.zeros(len(edge_df), 1),
        y=torch.from_numpy(y_arr)
    )

    # OTC split: train <= t_q75, val: (t_q75, t_q80], test: (t_q80, t_end]
    train_data = data[data.t <= t_q75]
    val_data = data[(data.t > t_q75) & (data.t <= t_q80)]
    test_data = data[(data.t > t_q80) & (data.t <= t_end)]

    return num_nodes, train_data, val_data, test_data, batch_size, min_ts

# -----------------------------
# Evaluation & Training Utils
# -----------------------------
@torch.no_grad()
def warmup_memory(model, loaders, policy_mgr=None, policy_stream=None, device='cuda'):
    model.eval()
    model.reset_memory()
    if policy_mgr is not None:
        policy_mgr.eval()
    loaders = loaders if isinstance(loaders, (list, tuple)) else [loaders]
    for loader in loaders:
        for batch in loader:
            cur_t = int(batch.t.max().item())
            p_agg = None
            if policy_mgr is not None and policy_stream is not None:
                policies = policy_stream.get_policies_until(cur_t)
                policy_mgr.sync_policies_until(policies)
                p_agg = policy_mgr.get_node_policy(model.memory.last_update, device, cur_t=cur_t, window=45)
            model(batch.to(device), p_agg=p_agg)

@torch.no_grad()
def evaluate(model, eval_loader, criterion, device='cuda',
             policy_mgr=None, policy_stream=None, pre_loader=None):
    model.eval()
    model.reset_memory()
    if policy_mgr is not None:
        policy_mgr.eval()
    if pre_loader is not None:
        warmup_memory(model, pre_loader, policy_mgr, policy_stream, device)

    y_true, y_score, losses = [], [], []
    for batch in eval_loader:
        cur_t = int(batch.t.max().item())
        p_agg = None
        if policy_mgr is not None and policy_stream is not None:
            policies = policy_stream.get_policies_until(cur_t)
            policy_mgr.sync_policies_until(policies)
            p_agg = policy_mgr.get_node_policy(model.memory.last_update, device, cur_t=cur_t, window=45)

        batch = batch.to(device)
        logits = model(batch, p_agg=p_agg)
        labels = batch.y
        mask = labels >= 0
        if not mask.any():
            continue

        loss = criterion(logits[mask], labels[mask])
        losses.append(loss.item())
        y_true.append(labels[mask].cpu())
        y_score.append(logits[mask][:, 1].cpu())

    if not y_true:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(y_true).numpy()
    y_score = torch.cat(y_score).numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return float(np.mean(losses)), auc, ap

def train_epoch(model, train_loader, criterion, optimizer1, device='cuda',
                policy_mgr=None, optimizer2=None, policy_stream=None):
    model.train()
    if policy_mgr is not None:
        policy_mgr.train()
    model.reset_memory()

    total_loss = 0.0
    y_true_all, y_score_all = [], []

    for batch in tqdm(train_loader, desc='Train', leave=False):
        cur_t = int(batch.t.max().item())
        p_agg, aux = None, None

        if policy_mgr is not None and policy_stream is not None:
            policies = policy_stream.get_policies_until(cur_t)
            policy_mgr.sync_policies_until(policies)

        batch = batch.to(device)

        if policy_mgr is not None:
            sub_idx = torch.unique(torch.cat([batch.src, batch.dst], dim=0))
            mem = model.memory.memory
            x0_base = torch.cat([mem, model.feats], dim=1)
            p_agg, aux = policy_mgr.get_node_policy(
                model.memory.last_update, device, cur_t=cur_t, window=45,
                x0_base=x0_base, return_aux=True, subset_idx=sub_idx, policy_dropout=0.0
            )

        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.zero_grad()

        logits = model(batch, p_agg=p_agg)
        labels = batch.y
        mask = labels >= 0

        ce_loss = criterion(logits[mask], labels[mask]) if mask.any() else torch.tensor(0.0, device=device)

        bce_loss = ent_loss = cons_loss = torch.tensor(0.0, device=device)
        if policy_mgr is not None and aux is not None and 'sim_raw' in aux:
            has_label = aux['has_label']
            if has_label.any():
                sim_sub = aux['sim_raw']
                labels_sub = aux['masks_sub']
                idx_mask = torch.nonzero(has_label).squeeze(-1)
                sim_sel = sim_sub[:, idx_mask]
                labels_sel = labels_sub[idx_mask].T

                pos = labels_sel.sum(dim=0)
                neg = labels_sel.shape[0] - pos
                pos_w = ((neg + 1.0) / (pos + 1.0)).clamp(max=10.0)
                logit_scale = getattr(policy_mgr, 'bce_temp', 1.0)
                bce_loss = F.binary_cross_entropy_with_logits(
                    logit_scale * sim_sel, labels_sel, pos_weight=pos_w
                )

            probs = torch.softmax(aux['sim_raw'], dim=1)
            ent = -(probs * (probs.clamp_min(1e-8)).log()).sum(1)
            ent_loss = ent.mean()

            mem = model.memory.memory
            x0_base = torch.cat([mem, model.feats], dim=1)
            with torch.no_grad():
                p_teacher = policy_mgr.get_node_policy(
                    model.memory.last_update, device, cur_t=cur_t, window=45,
                    x0_base=x0_base, return_aux=False, subset_idx=None, policy_dropout=0.0
                )
            p_student = policy_mgr.get_node_policy(
                model.memory.last_update, device, cur_t=cur_t, window=45,
                x0_base=x0_base.detach(), return_aux=False, subset_idx=None, policy_dropout=0.3
            )
            cons_loss = F.mse_loss(p_teacher[sub_idx], p_student[sub_idx])

        loss = ce_loss + 0.3 * bce_loss - 1e-3 * ent_loss + 1e-2 * cons_loss
        loss.backward()
        optimizer1.step()
        if optimizer2 is not None:
            optimizer2.step()

        total_loss += loss.item()
        if mask.any():
            y_true_all.append(labels[mask].cpu())
            y_score_all.append(logits[mask][:, 1].cpu())

    if not y_true_all:
        return total_loss / max(1, len(train_loader)), 0.0, 0.0

    y_true = torch.cat(y_true_all).numpy()
    y_score = torch.cat(y_score_all).numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return total_loss / max(1, len(train_loader)), auc, ap

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_policy', action='store_true', default=True)
    parser.add_argument('--data_path', type=str, default='./soc-sign-bitcoinotc.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_nodes, train_data, val_data, test_data, batch_size, min_ts = load_btc_data(
        args.data_path, batch_size=args.batch_size
    )

    train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
    val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
    test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

    # Model: CensNetEdge (GeneralDyG)
    from model_policy import CensNetEdge
    model = CensNetEdge(
        n_nodes=num_nodes,
        emb_dim=8,
        mem_dim=128,
        d_model=128,
        n_layers=2,
        time_dim=64,
        drop=0.1,
        policy_scale_init=0.1
    ).to(device)
    model_name = 'generaldyg'

    counts = torch.bincount(train_data.y, minlength=2).float()
    class_weight = (counts.sum() / (counts + 1e-8))
    class_weight = (class_weight / class_weight.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    policy_mgr, optimizer2, policy_stream = None, None, None
    if args.use_policy:
        from policy import PolicyManager_nn, PolicyStream
        policy_mgr = PolicyManager_nn(
            num_nodes, node_dim=model.emb_dim + 128, policy_vec_dim=133, decay_lambda=0.05
        ).to(device)
        optimizer2 = torch.optim.Adam(policy_mgr.parameters(), lr=1e-2, weight_decay=0)
        policy_stream = PolicyStream('./llm_injection_information2.jsonl', min_ts, transform=None)

    # Determine early stopping mode
    if model_name in ['tgat', 'tgn'] and args.use_policy:
        mode = 'ap_mode'
        patience = 10
    else:
        mode = 'auc_mode'
        patience = 5

    best_val_auc = best_val_ap = 0.0
    best_path = f'best_{model_name}_btcotc_{"policy" if args.use_policy else "nopolicy"}.pt'
    bad = 0

    for epoch in range(1, args.epochs + 1):
        if hasattr(model, 'policy_scale'):
            print(f'policy_scale = {model.policy_scale.item():.4f}')

        tr_loss, tr_auc, tr_ap = train_epoch(
            model, train_loader, criterion, optimizer1, device,
            policy_mgr, optimizer2, policy_stream
        )
        val_loss, val_auc, val_ap = evaluate(
            model, val_loader, criterion, device,
            policy_mgr, policy_stream, pre_loader=train_loader
        )

        print(f'E{epoch:02d} | TR loss {tr_loss:.4f} AUC {tr_auc:.4f} AP {tr_ap:.4f} | '
              f'VAL loss {val_loss:.4f} AUC {val_auc:.4f} AP {val_ap:.4f}')

        improved = False
        if mode == 'auc_mode' and val_auc > best_val_auc and epoch > 1:
            best_val_auc = val_auc
            improved = True
        elif mode == 'ap_mode' and val_ap > best_val_ap and epoch > 1:
            best_val_ap = val_ap
            improved = True

        if improved:
            ckpt = {'model': model.state_dict()}
            if args.use_policy and policy_mgr is not None:
                ckpt['policy_mgr'] = policy_mgr.state_dict()
            torch.save(ckpt, best_path)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping.')
                break

    # Final test evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if args.use_policy and policy_mgr is not None and 'policy_mgr' in ckpt:
        policy_mgr.load_state_dict(ckpt['policy_mgr'])

    pre_loaders = [train_loader, val_loader]
    test_loss, test_auc, test_ap = evaluate(
        model, test_loader, criterion, device,
        policy_mgr, policy_stream, pre_loader=pre_loaders
    )
    print(f'Test loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}')

if __name__ == '__main__':
    main()
