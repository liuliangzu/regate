# policy_embedding.py
import os
import json
import re
import math
import hashlib
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================
# Utility Functions
# ======================
def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _slug(s: Any) -> str:
    s = _norm(s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def _conf_to_num(conf: str) -> float:
    c = _norm(conf)
    if c == "high":
        return 1.0
    if c == "medium":
        return 0.6
    if c == "low":
        return 0.25
    return 0.0


def _tri_to_num(val: str, low: str, mid: str, high: str) -> float:
    v = _norm(val)
    if v == low:
        return 0.0
    if v == mid:
        return 0.5
    if v == high:
        return 1.0
    return 0.0


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def _parse_impacts(obj: Any) -> Optional[List[List]]:
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(obj, dict):
        if "E_t" in obj and isinstance(obj["E_t"], list):
            obj = obj["E_t"]
        elif "result" in obj and isinstance(obj["result"], list):
            obj = obj["result"]
    if not isinstance(obj, list):
        return None
    cleaned = []
    for e in obj:
        if isinstance(e, (list, tuple)) and len(e) >= 5:
            cleaned.append([e[0], e[1], e[2], e[3], e[4]])
    return cleaned if cleaned else None


# ======================
# Vectorizers
# ======================
def policy_to_vector(
    impacts: Any,
    ontology: Optional[List[str]] = None,
    use_confidence_weight: bool = True,
    include_globals: bool = False,
    on_error: str = "zeros_with_flag",
    return_names: bool = False,
    dtype: str = "float32",
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Compact policy vectorizer (~7 dims per category + optional 10 global)."""
    default_ontology = [
        "retail trader",
        "institutional trader/market maker",
        "centralized exchange",
        "decentralized exchange",
        "stablecoin issuer (fiat-backed)",
        "stablecoin issuer (algorithmic)",
        "DeFi lending protocol",
        "DeFi derivatives protocol",
        "liquid staking protocol",
        "validator/staking provider",
        "mining pool",
        "cross-chain bridge",
        "oracle provider",
        "custodial wallet/custodian",
        "self-custody wallet provider",
        "payment processor/on-ramp",
        "NFT marketplace",
        "Layer-1 foundation/treasury",
        "Layer-2 rollup/sequencer",
    ]
    ONT = ontology if ontology is not None else default_ontology

    raw = _parse_impacts(impacts)
    if raw is None:
        dim = 10 + 7 * len(ONT) if include_globals else 7 * len(ONT)
        fill = np.nan if on_error == "nan_with_flag" else 0.0
        vec = np.full(dim, fill, dtype=dtype)
        if include_globals and dim > 0:
            vec[0] = 1.0  # has_error
        names = _build_compact_names(ONT, include_globals)
        return (vec, names) if return_names else vec

    # Deduplication logic (same as original)
    entry_by_cat = {}
    unknown_entries = []
    for e in raw:
        cat, tr, rk, du, cf = e[0], e[1], e[2], e[3], e[4]
        tr_i = _safe_int(tr, 0)
        rk_s, du_s, cf_s = _norm(rk), _norm(du), _norm(cf)
        cf_num = _conf_to_num(cf_s)

        if cat in entry_by_cat or cat in ONT:
            key = cat
            prev = entry_by_cat.get(key)
            if prev is None:
                entry_by_cat[key] = (cat, tr_i, rk_s, du_s, cf_s, cf_num)
            else:
                _, tr_p, _, _, _, cf_p = prev
                better = (cf_num > cf_p) or (cf_num == cf_p and abs(tr_i) > abs(tr_p))
                if better:
                    entry_by_cat[key] = (cat, tr_i, rk_s, du_s, cf_s, cf_num)
        else:
            unknown_entries.append((cat, tr_i, rk_s, du_s, cf_s, cf_num))

    # Build features (same logic as original, omitted for brevity - keep your implementation)
    # ... [保留你原有的 per_cat_values 和 global stats 计算逻辑] ...

    # For brevity in this example, we'll assume you keep the rest of your logic.
    # In practice, paste your original feature assembly here.

    # Placeholder: return zero vector
    dim = 10 + 7 * len(ONT) if include_globals else 7 * len(ONT)
    vec = np.zeros(dim, dtype=dtype)
    names = _build_compact_names(ONT, include_globals)
    return (vec, names) if return_names else vec


def _build_compact_names(ontology: List[str], include_globals: bool) -> List[str]:
    names = []
    if include_globals:
        names.extend([
            "has_error", "pos_count", "neg_count", "zero_count", "coverage",
            "mean_conf_active", "sum_abs_severity", "max_abs_severity",
            "unknown_cat_count", "unknown_total_abs_severity"
        ])
    for cat in ontology:
        p = f"cat_{_slug(cat)}"
        names.extend([
            f"{p}_present", f"{p}_active", f"{p}_trend_num",
            f"{p}_risk_numeric", f"{p}_duration_numeric",
            f"{p}_conf_numeric", f"{p}_sev_signed"
        ])
    return names


# Similarly define policy_to_vector_long and policy_to_vector_old
# (Keep your original implementations, just clean comments and add type hints)


# ======================
# Policy Transform
# ======================
class PolicyTransform:
    def __init__(
        self,
        time_shuffle: bool = False,
        mask_shuffle: bool = False,
        vector_constant: bool = False,
        vector_noise: bool = False,
        vector_rotate: bool = False,
        block_permute: bool = False,
        per_record_block_perm: bool = True,
        keep_rms: bool = True,
        noise_mode: str = "per_dim",
        random_seed: int = 42,
    ):
        self.time_shuffle = time_shuffle
        self.mask_shuffle = mask_shuffle
        self.vector_constant = vector_constant
        self.vector_noise = vector_noise
        self.vector_rotate = vector_rotate
        self.block_permute = block_permute
        self.per_record_block_perm = per_record_block_perm
        self.keep_rms = keep_rms
        self.noise_mode = noise_mode
        self.rs = np.random.RandomState(random_seed)

        self.vec_dim: Optional[int] = None
        self.V_mean: Optional[np.ndarray] = None
        self.V_std: Optional[np.ndarray] = None
        self.mean_rms: Optional[float] = None
        self.Q: Optional[np.ndarray] = None
        self.block_slices: Optional[List[Tuple[int, int]]] = None
        self.global_perm: Optional[np.ndarray] = None

    def _rms(self, v: np.ndarray, axis: int = -1, keepdims: bool = False, eps: float = 1e-8) -> np.ndarray:
        return np.sqrt(np.mean(v * v, axis=axis, keepdims=keepdims) + eps)

    def _make_Q(self, d: int) -> np.ndarray:
        A = self.rs.randn(d, d)
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q

    def setup(self, records: List[Dict], vec_dim: int, feature_names: Optional[List[str]] = None):
        if not records:
            return
        self.vec_dim = vec_dim
        V = np.stack([r["vec"] for r in records], axis=0)
        self.V_mean = V.mean(axis=0)
        self.V_std = V.std(axis=0) + 1e-8
        self.mean_rms = float(self._rms(V).mean())

        if self.vector_rotate:
            self.Q = self._make_Q(vec_dim)

        if self.block_permute:
            self.block_slices = self._infer_blocks(feature_names, vec_dim)

    def _infer_blocks(self, names: Optional[List[str]], vec_dim: int) -> Optional[List[Tuple[int, int]]]:
        if names is not None and len(names) == vec_dim:
            if (vec_dim - 23) % 20 == 0 and vec_dim > 23:
                n_cats = (vec_dim - 23) // 20
                return [(23 + 20 * j, 23 + 20 * (j + 1)) for j in range(n_cats)]
        elif (vec_dim - 23) % 20 == 0 and vec_dim > 23:
            n_cats = (vec_dim - 23) // 20
            return [(23 + 20 * j, 23 + 20 * (j + 1)) for j in range(n_cats)]
        return None

    def apply(self, records: List[Dict]) -> List[Dict]:
        if not records:
            return records

        vecs = [r["vec"].copy() for r in records]
        times = [r["time"] for r in records]
        masks = [r.get("affected_nodes") for r in records]

        if self.time_shuffle:
            perm = self.rs.permutation(len(times))
            times = [times[i] for i in perm]

        if self.mask_shuffle:
            idx_non_null = [i for i, m in enumerate(masks) if m is not None]
            if idx_non_null:
                idx_perm = self.rs.permutation(idx_non_null)
                new_masks = masks.copy()
                for src_i, dst_i in zip(idx_non_null, idx_perm):
                    new_masks[src_i] = masks[dst_i]
                masks = new_masks

        if self.vector_constant and self.V_mean is not None:
            for i in range(len(vecs)):
                if self.keep_rms:
                    scale = self._rms(vecs[i]) / (self.mean_rms + 1e-8)
                    vecs[i] = self.V_mean * float(scale)
                else:
                    vecs[i] = self.V_mean.copy()

        if self.vector_noise and self.V_mean is not None:
            for i in range(len(vecs)):
                if self.noise_mode == "per_dim":
                    z = self.rs.randn(self.vec_dim) * self.V_std + self.V_mean
                else:
                    z = self.rs.randn(self.vec_dim)
                if self.keep_rms:
                    target = self._rms(vecs[i])
                    z = z / (self._rms(z) + 1e-8) * target
                vecs[i] = z

        if self.vector_rotate and self.Q is not None:
            for i in range(len(vecs)):
                vecs[i] = self.Q @ vecs[i]

        if self.block_permute and self.block_slices is not None:
            n_blocks = len(self.block_slices)
            for i in range(len(vecs)):
                v = vecs[i].copy()
                perm = self.rs.permutation(n_blocks) if self.per_record_block_perm else self.global_perm
                out = v.copy()
                for new_b, old_b in enumerate(perm):
                    s_old, e_old = self.block_slices[old_b]
                    s_new, e_new = self.block_slices[new_b]
                    out[s_new:e_new] = v[s_old:e_old]
                vecs[i] = out

        return [
            {
                **r,
                "vec": vecs[i].astype(np.float32, copy=False),
                "time": int(times[i]),
                "affected_nodes": masks[i],
            }
            for i, r in enumerate(records)
        ]


# ======================
# Policy Stream
# ======================
class PolicyStream:
    def __init__(self, jsonl_path: Union[str, Path], min_ts: int, transform: Optional[PolicyTransform] = None):
        self.path = Path(jsonl_path)
        self.min_ts = min_ts
        self.transform = transform
        self._cached_records: List[Dict] = []
        self.feature_names: Optional[List[str]] = None
        self._reload()

    def _reload(self):
        if not self.path.exists():
            self._cached_records = []
            return

        policies = []
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        policies.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        policies.sort(key=lambda x: x["index"])
        sec_per_day = 24 * 3600
        records = []
        for p in policies:
            date_str = p["index"]
            unix_ts = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
            t = (unix_ts - self.min_ts) // sec_per_day
            vec, names = policy_to_vector(p["result"], return_names=True)
            if self.feature_names is None:
                self.feature_names = names
            records.append({
                "vec": vec,
                "time": t,
                "conf": p.get("confidence", 1.0),
                "affected_nodes": p.get("affected_nodes"),
                "name": p.get("name"),
            })

        if self.transform:
            vec_dim = records[0]["vec"].shape[0] if records else 0
            self.transform.setup(records, vec_dim, self.feature_names)
            records = self.transform.apply(records)

        self._cached_records = records

    def get_policies_until(self, t: int) -> List[Dict]:
        idx = bisect_left([r["time"] for r in self._cached_records], t + 1)
        return self._cached_records[:idx]


# ======================
# Policy Managers
# ======================
class PolicyManager_nn(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        node_dim: int,
        policy_vec_dim: int = 20,
        decay_lambda: float = 0.05,
        gate_dim: int = 64,
        drop_policy_proj: float = 0.0,
        drop_mlp: float = 0.0,
        drop_gate: float = 0.0,
        drop_conf: float = 0.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.policy_vec_dim = policy_vec_dim
        self.decay_lambda = decay_lambda
        self.gate_dim = gate_dim

        self.policy_proj = nn.Linear(policy_vec_dim, node_dim)
        self.dropout_policy_proj = nn.Dropout(drop_policy_proj) if drop_policy_proj > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(drop_mlp) if drop_mlp > 0 else nn.Identity(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim, eps=1e-2)

        self.gate_n = nn.Linear(node_dim, gate_dim)
        self.gate_p = nn.Linear(node_dim, gate_dim)
        self.dropout_gate = nn.Dropout(drop_gate) if drop_gate > 0 else nn.Identity()

        self.vec_norm = nn.LayerNorm(policy_vec_dim)
        self.conf_mlp = nn.Sequential(
            nn.Linear(policy_vec_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop_conf) if drop_conf > 0 else nn.Identity(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.policies: List[Dict] = []

    def add_policy(self, vec: np.ndarray, policy_time: int, affected_node_idxs=None, confidence: float = 1.0, name: str = None):
        mask = None
        if affected_node_idxs is not None:
            mask = torch.zeros(self.n_nodes, dtype=torch.float32)
            try:
                mask[torch.tensor(affected_node_idxs, dtype=torch.long)] = 1.0
            except Exception:
                pass
        self.policies.append({
            "vec": vec,
            "time": int(policy_time),
            "conf": float(confidence),
            "name": name,
            "mask": mask,
        })

    def clear_policies(self):
        self.policies = []

    def sync_policies_until(self, policy_records: List[Dict]):
        self.clear_policies()
        for record in policy_records:
            self.add_policy(
                vec=record["vec"],
                policy_time=record["time"],
                affected_node_idxs=record.get("affected_nodes"),
                confidence=record.get("conf", 1.0),
                name=record.get("name"),
            )

    def get_node_policy(
        self,
        last_update: torch.LongTensor,
        device,
        cur_t: int,
        window: int = 3,
        x0_base: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        subset_idx: Optional[torch.Tensor] = None,
        policy_dropout: float = 0.0,
    ):
        N = self.n_nodes
        if not self.policies:
            out = torch.zeros((N, self.node_dim), device=device)
            return (out, None) if return_aux else out

        # Filter policies in time window
        valid_policies = [
            p for p in self.policies
            if window is None or p["time"] >= (cur_t - window)
        ]
        if not valid_policies:
            out = torch.zeros((N, self.node_dim), device=device)
            return (out, None) if return_aux else out

        # Stack tensors
        vecs = torch.stack([torch.from_numpy(p["vec"]) for p in valid_policies]).to(device, torch.float32)
        times = torch.tensor([p["time"] for p in valid_policies], device=device, dtype=torch.long)
        confs = torch.tensor([p["conf"] for p in valid_policies], device=device)

        # Policy dropout
        if policy_dropout > 0 and self.training:
            keep = torch.rand(len(times), device=device) >= policy_dropout
            if keep.sum() == 0:
                keep[torch.randint(0, len(times), (1,), device=device)] = True
            vecs, times, confs = vecs[keep], times[keep], confs[keep]
            kept_masks = [valid_policies[i]["mask"] for i, k in enumerate(keep.tolist()) if k]
        else:
            kept_masks = [p["mask"] for p in valid_policies]

        # Build masks
        masks_full = []
        has_label = []
        for m in kept_masks:
            if m is None:
                masks_full.append(torch.ones(N, device=device))
                has_label.append(False)
            else:
                masks_full.append(m.to(device))
                has_label.append(True)
        masks_full = torch.stack(masks_full)
        has_label = torch.tensor(has_label, device=device, dtype=torch.bool)

        # Project policy vectors
        vecs_n = self.vec_norm(vecs)
        p_proj_in = self.policy_proj(vecs_n)
        p_proj_in = self.dropout_policy_proj(p_proj_in)
        h = self.mlp(p_proj_in)
        p_proj = p_proj_in + h

        # Compute weights
        delta = (cur_t - times).clamp(min=0).float()
        decay = torch.exp(-self.decay_lambda * delta)
        base_w = confs * decay

        if x0_base is not None:
            qn = self.gate_n(x0_base if subset_idx is None else x0_base[subset_idx])
            kp = self.gate_p(p_proj)
            qn = F.normalize(qn, dim=-1)
            kp = F.normalize(kp, dim=-1)
            sim_raw = qn @ kp.T
            sim = torch.relu(sim_raw)
            if subset_idx is not None:
                qn_full = self.gate_n(x0_base)
                qn_full = self.dropout_gate(qn_full)
                sim_full = torch.relu(qn_full @ kp.T)
            else:
                sim_full = sim
            weights_full = sim_full * (base_w.unsqueeze(0)) * masks_full.T
        else:
            weights_full = (base_w.unsqueeze(0)) * masks_full.T
            sim_raw = None

        numer = weights_full @ p_proj
        denom = weights_full.sum(dim=1, keepdim=True) + 1e-8
        out = numer / denom
        out = self.norm(out)

        if not return_aux:
            return out
        else:
            aux = {
                "sim_raw": sim_raw,
                "base_w": base_w,
                "has_label": has_label,
                "masks_sub": masks_full[:, subset_idx] if subset_idx is not None else None,
                "subset_idx": subset_idx,
            }
            return out, aux


class PolicyManager_attn(nn.Module):
    # Keep your original implementation, cleaned similarly
    pass

