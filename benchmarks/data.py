"""TUDataset downloader + parser (pure numpy, no DGL/PyG dependency).

Graph-kernel datasets are downloaded from https://www.chrsmrrs.com/graphkerneldatasets
and parsed into per-graph (adjacency, one-hot node features, label) tuples.
"""
import os
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_URL = "https://www.chrsmrrs.com/graphkerneldatasets"
DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent / "data")


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _download(name: str, data_dir: str) -> str:
    """Download and extract a dataset, returning its directory path."""
    os.makedirs(data_dir, exist_ok=True)
    zpath = os.path.join(data_dir, f"{name}.zip")
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(f"{BASE_URL}/{name}.zip", zpath)
    target = os.path.join(data_dir, name)
    if not os.path.exists(target):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data_dir)
    return target


def load_tudataset(name: str, data_dir: str = None):
    """Load a TUDataset graph classification dataset.

    Args:
        name: dataset name (e.g. ``MUTAG``).
        data_dir: directory where datasets are downloaded/read; defaults to the
            ``benchmarks/data`` directory next to this module.

    Returns:
        ``(graphs, num_classes, in_dim)`` where ``graphs`` is a list of
        ``(adjacency, node_features, label)`` with ``adjacency`` a float32 ``(n, n)``
        matrix (with self-loops), ``node_features`` a one-hot ``(n, in_dim)`` matrix
        (or degree feature if node labels are absent), and ``label`` an int.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    d = _download(name, data_dir)
    edges = []
    for line in _read_lines(os.path.join(d, f"{name}_A.txt")):
        a, b = line.replace(",", " ").split()
        edges.append((int(a) - 1, int(b) - 1))
    indicator = [int(v) - 1 for v in _read_lines(os.path.join(d, f"{name}_graph_indicator.txt"))]
    n_nodes = len(indicator)
    n_graphs = max(indicator) + 1

    labels = [int(v) for v in _read_lines(os.path.join(d, f"{name}_graph_labels.txt"))]
    label_map = {v: i for i, v in enumerate(sorted(set(labels)))}
    labels = [label_map[v] for v in labels]
    num_classes = len(label_map)

    node_labels_path = os.path.join(d, f"{name}_node_labels.txt")
    if os.path.exists(node_labels_path):
        node_labels = [int(v) for v in _read_lines(node_labels_path)]
        uniq = sorted(set(node_labels))
        m = {v: i for i, v in enumerate(uniq)}
        node_feat = np.zeros((n_nodes, len(uniq)), dtype=np.float32)
        for i, v in enumerate(node_labels):
            node_feat[i, m[v]] = 1.0
        in_dim = len(uniq)
    else:
        node_feat = np.ones((n_nodes, 1), dtype=np.float32)
        in_dim = 1

    # group edges by graph (both endpoints share the same graph id)
    edges_of = defaultdict(list)
    for (a, b) in edges:
        edges_of[indicator[a]].append((a, b))

    graphs = []
    nodes_of = [[] for _ in range(n_graphs)]
    for i, g in enumerate(indicator):
        nodes_of[g].append(i)

    for g in range(n_graphs):
        idx = nodes_of[g]
        local = {old: k for k, old in enumerate(idx)}
        m = len(idx)
        a = np.eye(m, dtype=np.float32)  # self-loops
        for (u, v) in edges_of[g]:
            a[local[u], local[v]] = 1.0
            a[local[v], local[u]] = 1.0
        graphs.append((a, node_feat[idx], labels[g]))
    return graphs, num_classes, in_dim


if __name__ == "__main__":
    for ds in ["MUTAG", "ENZYMES", "PROTEINS", "NCI1"]:
        gs, nc, ind = load_tudataset(ds)
        sizes = [g[0].shape[0] for g in gs]
        print(f"{ds}: {len(gs)} graphs, {nc} classes, in_dim={ind}, avg_nodes={np.mean(sizes):.1f}")
