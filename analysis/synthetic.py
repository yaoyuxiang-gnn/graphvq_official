"""Controlled synthetic graph distributions for P0/P1 (reviewer 3.2).

- P0 "high-ring" stress test: mixture of cycles / cliques / trees / grids with
  one-hot node labels, cycle+clique weight >= 0.6 so triangles are common.
- P1 datasets: stochastic block model (community structure), plus the P0 mixture
  with adjustable composition, sizes and label noise.

Every generator returns (adjacency, one-hot features, label) tuples so the
output can be fed directly to ``load_dataset``-style pipelines.
"""
import numpy as np


def _adj(n, edges):
    a = np.zeros((n, n), dtype=np.float32)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    return a


def make_cycle(n, rng):
    edges = [(i, (i + 1) % n) for i in range(n)]
    return _adj(n, edges)


def make_clique(n, rng, density=1.0):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                edges.append((i, j))
    return _adj(n, edges)


def make_tree(n, rng):
    edges = []
    for v in range(1, n):
        parent = rng.integers(0, v)  # recursive random tree
        edges.append((v, parent))
    return _adj(n, edges)


def make_grid(n, rng):
    s = int(np.ceil(np.sqrt(n)))
    edges = []
    for i in range(n):
        r, c = divmod(i, s)
        if c + 1 < s and i + 1 < n:
            edges.append((i, i + 1))
        if r + 1 < s and i + s < n:
            edges.append((i, i + s))
    return _adj(n, edges)


def make_sbm(n, rng, n_blocks=3, p_in=0.4, p_out=0.03):
    blocks = np.sort(rng.integers(0, n_blocks, n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if blocks[i] == blocks[j] else p_out
            if rng.random() < p:
                edges.append((i, j))
    a = _adj(n, edges)
    x = np.eye(n_blocks, dtype=np.float32)[blocks]
    return a, x


def _labels(n, rng, n_classes, role_correlated=False, kind=None):
    """One-hot node labels. ``role_correlated`` plants a label shared by all nodes."""
    if role_correlated and kind is not None:
        lab = {"cycle": 0, "clique": 1, "tree": 2, "grid": 3}.get(kind, 0) % n_classes
        x = np.zeros((n, n_classes), dtype=np.float32)
        x[:, lab] = 1.0
        return x
    return np.eye(n_classes, dtype=np.float32)[rng.integers(0, n_classes, n)]


def make_dataset(n_graphs, kinds=("cycle", "clique", "tree", "grid"), weights=None,
                 n_range=(10, 30), n_classes=4, seed=0, role_correlated=False,
                 kind_props=None, sbm_blocks=3):
    """Generate a synthetic graph dataset.

    ``weights``: mixture weights over ``kinds`` (defaults to 0.4 cycle / 0.4 clique
    / 0.1 tree / 0.1 grid when all four kinds are present — a "high-ring" mixture).
    ``kind_props``: optional per-kind extra kwargs (e.g. {"clique": {"density": 0.8}}).
    """
    rng = np.random.default_rng(seed)
    if weights is None:
        if set(kinds) == {"cycle", "clique", "tree", "grid"}:
            weights = [0.4, 0.4, 0.1, 0.1]
        else:
            weights = [1.0 / len(kinds)] * len(kinds)
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    kind_props = kind_props or {}
    graphs = []
    for _ in range(n_graphs):
        kind = kinds[int(rng.choice(len(kinds), p=weights))]
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        props = kind_props.get(kind, {})
        if kind == "cycle":
            a = make_cycle(n, rng)
        elif kind == "clique":
            a = make_clique(n, rng, props.get("density", 1.0))
        elif kind == "tree":
            a = make_tree(n, rng)
        elif kind == "grid":
            a = make_grid(n, rng)
        elif kind == "sbm":
            a, x = make_sbm(n, rng, n_blocks=props.get("n_blocks", sbm_blocks),
                            p_in=props.get("p_in", 0.4), p_out=props.get("p_out", 0.03))
            graphs.append((a, x, 0))
            continue
        else:
            raise ValueError(f"unknown kind: {kind}")
        x = _labels(n, rng, n_classes, role_correlated, kind)
        graphs.append((a, x, 0))
    return graphs


def register():
    """Build a {name: graphs} registry used by the P0/P1 runners."""
    return {
        "SYN_RING": make_dataset(400, seed=0),
        "SYN_RING_ROLE": make_dataset(400, seed=0, role_correlated=True),
        "SYN_COMM": make_dataset(400, kinds=("sbm",), weights=[1.0], n_range=(15, 40),
                                 seed=0, kind_props={"sbm": {"p_in": 0.35, "p_out": 0.02}}),
    }
