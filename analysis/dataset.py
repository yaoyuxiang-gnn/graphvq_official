"""Dataset loaders for graph generation: QM9 (SMILES -> RDKit) and TUDataset."""
import os
from pathlib import Path

import numpy as np

from benchmarks.data import load_tudataset

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent / "data")


def load_qm9(n: int, cache_dir: str = None):
    """Load a subset of QM9 as molecular graphs (SMILES -> RDKit)."""
    if cache_dir is None:
        cache_dir = DEFAULT_DATA_DIR
    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("rdkit not installed; cannot load QM9")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "qm9_smiles.txt")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    if not os.path.exists(cache):
        import urllib.request
        import csv
        print("downloading qm9.csv (~150MB)...", flush=True)
        urllib.request.urlretrieve(url, os.path.join(cache_dir, "qm9.csv.tmp"))
        smiles = []
        with open(os.path.join(cache_dir, "qm9.csv.tmp"), newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            si = header.index("smiles")
            for row in reader:
                smiles.append(row[si])
        with open(cache, "w") as f:
            f.write("\n".join(smiles))
    smiles = open(cache).read().splitlines()[:n]

    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F
    n_types = 6
    graphs = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        m = mol.GetNumAtoms()
        feat = np.zeros((m, n_types), dtype=np.float32)
        adj = np.zeros((m, m), dtype=np.float32)
        for atom in mol.GetAtoms():
            feat[atom.GetIdx(), atom_map.get(atom.GetAtomicNum(), 5)] = 1.0
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1.0
        graphs.append((adj, feat, 0))
    return graphs


def load_dataset(name, n_qm9=5000):
    if name == "QM9":
        return load_qm9(n_qm9)
    if name in ("SYN_RING", "SYN_RING_ROLE", "SYN_COMM"):
        from .synthetic import register
        return register()[name]
    graphs, _, _ = load_tudataset(name)
    return graphs


def split_indices(n, seed=0, split=(0.8, 0.1, 0.1)):
    """Deterministic 80/10/10 split indices for held-out NLL evaluations."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(n * split[0])
    b = int(n * (split[0] + split[1]))
    return idx[:a], idx[a:b], idx[b:]
