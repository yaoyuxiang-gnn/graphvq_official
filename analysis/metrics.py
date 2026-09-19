"""Graph generation metrics shared by the single-stage and two-stage pipelines."""
import statistics

import numpy as np


def is_connected(adj):
    n = adj.shape[0]
    if n == 0:
        return False
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u, v] and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def graph_stats(adj):
    n = adj.shape[0]
    deg = adj.sum(1)
    tri = 0.0
    closed = 0.0
    for i in range(n):
        nb = np.where(adj[i] > 0)[0]
        k = len(nb)
        if k >= 2:
            sub = adj[np.ix_(nb, nb)]
            closed += sub.sum() / 2.0
            tri += k * (k - 1) / 2.0
    triangles = float((adj @ adj @ adj).trace()) / 6.0
    clustering = closed / max(tri, 1.0)
    return {"deg": float(deg.mean()), "tri": triangles, "cluster": float(clustering), "conn": is_connected(adj)}


def degree_hist(adj, bins=8):
    deg = adj.sum(1).astype(int)
    hist, _ = np.histogram(deg, bins=range(bins + 1))
    return hist / max(hist.sum(), 1)


def graph_signature(adj, feat):
    """Legacy cheap signature (node count + edge count + raw bytes).

    Kept for compatibility; novelty/uniqueness now use :func:`wl_signature`
    (canonical-style color refinement) to match the paper's isomorphism wording.
    """
    return (adj.shape[0], int(adj.sum()), adj.tobytes() + feat.tobytes())


def wl_signature(adj, feat, n_iter=3):
    """Weisfeiler-Lehman color-refinement signature (canonical-style graph hash).

    Two graphs with the same signature are provably non-distinguishable by
    ``n_iter`` rounds of 1-WL refinement; in practice this approximates
    isomorphism well for the small graphs used here while staying O(n * d * iter).
    """
    n = adj.shape[0]
    if n == 0:
        return ()
    adj_bin = (adj > 0).astype(np.int64)
    if feat.ndim == 2:
        init = feat.argmax(1).astype(np.int64) + 1
    else:
        init = feat.astype(np.int64).ravel() + 1
    colors = init.tolist()
    deg = adj_bin.sum(1).tolist()
    for _ in range(n_iter):
        new = []
        for i in range(n):
            neigh = tuple(sorted(colors[j] for j in range(n) if adj_bin[i, j] and j != i))
            new.append(hash((colors[i], neigh, deg[i])))
        colors = new
    return tuple(sorted(colors))


def _atom_types(feat):
    return [int(feat[i].argmax()) if feat.ndim == 2 else int(feat[i])
            for i in range(feat.shape[0])]


def _mol_signature(adj, feat, mode):
    """RDKit canonical-SMILES signature of an (adjacency, features) molecular graph.

    ``mode == "qm9"`` maps feature indices to atomic numbers (H, C, N, O, F);
    ``mode == "isotope"`` labels a carbon scaffold with isotopes equal to the
    feature index (for non-element node labels such as MUTAG's). Both sides
    (training and generated graphs) are converted identically, so the signature
    is a canonical graph hash within our single-bond protocol.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return wl_signature(adj, feat)
    n = adj.shape[0]
    if n == 0:
        return ""
    atomic = [1, 6, 7, 8, 9, 0]  # dataset.py atom_map: H, C, N, O, F, other
    types = _atom_types(feat)
    m = Chem.RWMol()
    for i in range(n):
        if mode == "qm9":
            a = Chem.Atom(atomic[types[i]] if types[i] < len(atomic) else 0)
        else:
            a = Chem.Atom(6)
            a.SetIsotope(types[i] + 1)
        m.AddAtom(a)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                m.AddBond(i, j, Chem.BondType.SINGLE)
    return Chem.MolToSmiles(m.GetMol())


SIGNATURE_FNS = {
    "wl": lambda a, x: wl_signature(a, x),
    "qm9": lambda a, x: _mol_signature(a, x, "qm9"),
    "isotope": lambda a, x: _mol_signature(a, x, "isotope"),
}


def summarize_graphs(gen_graphs, train_graphs, include_hist_and_conn=True, signature="wl"):
    """Compute validity / novelty / uniqueness plus graph-stat MAEs.

    ``gen_graphs`` is a list of ``(adjacency, node_features)``; ``train_graphs`` is a
    list of ``(adjacency, node_features, label)``. When ``include_hist_and_conn`` is
    False the degree-histogram and connectivity fields are omitted (two-stage output).
    ``signature`` selects the canonicalization: ``"wl"`` (any graph), ``"qm9"``
    (RDKit canonical SMILES, atom-type features), ``"isotope"`` (isotope-labeled
    carbon scaffold, e.g. MUTAG node labels).
    """
    sig_fn = SIGNATURE_FNS.get(signature, SIGNATURE_FNS["wl"])
    valid = novel = unique = 0
    train_sigs = set()
    train_stats = []
    train_hist = []
    for (a, x, _) in train_graphs:
        train_sigs.add(sig_fn(a, x))
        train_stats.append(graph_stats(a))
        if include_hist_and_conn:
            train_hist.append(degree_hist(a))

    seen = set()
    gen_stats = []
    gen_hist = []
    for adj, feat in gen_graphs:
        if adj.shape[0] > 1 and adj.sum() > 0:
            valid += 1
        sig = sig_fn(adj, feat)
        if sig not in train_sigs:
            novel += 1
        if sig not in seen:
            unique += 1
            seen.add(sig)
        gen_stats.append(graph_stats(adj))
        if include_hist_and_conn:
            gen_hist.append(degree_hist(adj))

    n = max(len(gen_graphs), 1)
    t = {k: statistics.mean([s[k] for s in train_stats]) for k in ("deg", "tri", "cluster")}
    g = {k: statistics.mean([s[k] for s in gen_stats]) for k in ("deg", "tri", "cluster")}

    result = {
        "validity": valid / n,
        "novelty": novel / n,
        "uniqueness": unique / n,
        "signature": signature,
        "mean_degree": {"gen": round(g["deg"], 3), "train": round(t["deg"], 3), "mae": round(abs(g["deg"] - t["deg"]), 3)},
    }
    if include_hist_and_conn:
        t_hist = np.mean(train_hist, 0)
        g_hist = np.mean(gen_hist, 0)
        result["degree_hist_l1"] = round(float(np.abs(g_hist - t_hist).sum()), 3)
    result["triangle"] = {"gen": round(g["tri"], 3), "train": round(t["tri"], 3), "mae": round(abs(g["tri"] - t["tri"]), 3)}
    result["clustering"] = {"gen": round(g["cluster"], 3), "train": round(t["cluster"], 3), "mae": round(abs(g["cluster"] - t["cluster"]), 3)}
    if include_hist_and_conn:
        result["connectivity"] = {"gen": round(statistics.mean([s["conn"] for s in gen_stats]), 3),
                                  "train": round(statistics.mean([s["conn"] for s in train_stats]), 3)}
    result["n_generated"] = len(gen_graphs)
    return result
