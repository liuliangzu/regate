# Regate

This is the initial open-source release of **Regate**, provided solely to reproduce the experimental results reported in our paper.

## Datasets

To run the experiments, you will need to download the following datasets:

- **Bitcoin OTC** (`btc-otc`)
- **Bitcoin Alpha** (`btc-alpha`)

### Download Links

- [Bitcoin OTC Dataset](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- [Bitcoin Alpha Dataset](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)

### Preparation

1. Download both dataset files from the links above.
2. Extract the downloaded archives using the following command (replace `your_path` with the actual file path):

```bash
tar -xvf your_path
```

Place the extracted files in a directory of your choice—ensure the paths are correctly referenced when running the code.

## Source Code

The main executable script is located in the `src/` directory.

To run the experiments, simply execute:

```bash
cd src
python main.py
```

Make sure all dependencies are installed and the dataset paths are properly configured in the code or via command-line arguments (if supported).

---

> **Note**: This release is intended for reproducibility purposes only. Future versions may include additional features, documentation, and usability improvements.


