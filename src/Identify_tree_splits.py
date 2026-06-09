"""Identify_tree_splits.py

For each Up_k axis in a sequence-projection table, find the branch in a
phylogenetic guide tree whose removal best separates leaves into two clades
along that axis, judged by Mann-Whitney U two-sided p-value.

Algorithm (guaranteed optimum, O(N*K)):

  1. Parse the Newick guide tree iteratively into flat numpy arrays.
  2. Re-root at an arbitrary leaf so that every non-root, non-unifurcation
     node maps 1:1 to a unique unrooted bipartition (2N-3 total).
  3. Join leaves to projections by UniProt accession; rank values per axis.
  4. Single iterative post-order pass accumulates per-subtree validity counts
     and sums of (values, ranks).
  5. Vectorized MWU evaluation across all candidate splits and axes at once,
     using running sums and the standard tie-corrected variance.
  6. Benjamini-Hochberg FDR across (splits * axes). Optionally permutation
     min-p p-values (--permutation N).
  7. Per-axis best split printed to stdout and written to TSV.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Newick tokenizer + iterative parser
# ---------------------------------------------------------------------------

_NEWICK_DELIMS = set("(),:;")


def tokenize_newick(text):
    """Yield Newick tokens. Tokens are single delimiters '(' ')' ',' ':' ';'
    or runs of name/number characters. All whitespace (including newlines) is
    skipped.
    """
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in _NEWICK_DELIMS:
            yield c
            i += 1
            continue
        j = i
        while j < n and text[j] not in _NEWICK_DELIMS and not text[j].isspace():
            j += 1
        yield text[i:j]
        i = j


def parse_newick_iterative(text):
    """Parse a Newick string into flat arrays. No recursion.

    Returns a dict with keys:
        parent       int32[n_nodes]   -1 at root
        children     list[list[int]]
        branch_len   float64[n_nodes] NaN where unspecified
        name         object[n_nodes]  '' for unnamed internal nodes
        is_leaf      bool[n_nodes]
        leaf_ids     int32[n_leaves]  node ids of leaves
        root         int
        n_nodes      int
    """
    parent = []
    children = []
    branch_len = []
    name = []

    def new_node():
        nid = len(parent)
        parent.append(-1)
        children.append([])
        branch_len.append(float("nan"))
        name.append("")
        return nid

    stack = []           # open parents
    current = None       # most recently created/closed node (target of ':length' / post-')' label)
    expecting_child = False

    tokens = tokenize_newick(text)
    for tok in tokens:
        if tok == "(":
            nid = new_node()
            if stack:
                p = stack[-1]
                parent[nid] = p
                children[p].append(nid)
            stack.append(nid)
            current = nid
            expecting_child = True
        elif tok == ",":
            expecting_child = True
            current = None
        elif tok == ")":
            current = stack.pop()
            expecting_child = False
        elif tok == ":":
            length_tok = next(tokens)
            branch_len[current] = float(length_tok)
        elif tok == ";":
            break
        else:
            # NAME / NUMBER token: either a leaf name or an internal label
            if expecting_child:
                nid = new_node()
                p = stack[-1]
                parent[nid] = p
                children[p].append(nid)
                name[nid] = tok
                current = nid
                expecting_child = False
            else:
                if current is not None:
                    name[current] = tok

    n_nodes = len(parent)
    parent_arr = np.asarray(parent, dtype=np.int32)
    branch_len_arr = np.asarray(branch_len, dtype=np.float64)
    name_arr = np.asarray(name, dtype=object)
    is_leaf = np.array([len(c) == 0 for c in children], dtype=bool)
    leaf_ids = np.where(is_leaf)[0].astype(np.int32)

    root_candidates = np.where(parent_arr == -1)[0]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected 1 root, found {len(root_candidates)}")
    root = int(root_candidates[0])

    return {
        "parent": parent_arr,
        "children": children,
        "branch_len": branch_len_arr,
        "name": name_arr,
        "is_leaf": is_leaf,
        "leaf_ids": leaf_ids,
        "root": root,
        "n_nodes": n_nodes,
    }


# ---------------------------------------------------------------------------
# Re-rooting (so every non-root, non-unifurcation node = unique bipartition)
# ---------------------------------------------------------------------------

def reroot_at_leaf(tree, new_root_id):
    """In-place reroot. Flips parent pointers along the path from new_root_id
    to the old root and rotates branch lengths accordingly. The old root
    becomes a unifurcation (one child) — we leave it in place and let the
    split-enumeration step skip nodes with #children == 1.
    """
    parent = tree["parent"]
    children = tree["children"]
    branch_len = tree["branch_len"]

    # Path from new_root to old root
    path = []
    v = new_root_id
    while v != -1:
        path.append(int(v))
        v = int(parent[v])

    # Save old branch lengths so the flip can read consistent values
    old_lens = branch_len.copy()

    for i in range(len(path) - 1):
        child = path[i]            # becomes the parent in the new tree
        old_par = path[i + 1]      # becomes the child in the new tree
        children[old_par].remove(child)
        children[child].append(old_par)
        parent[old_par] = child
        branch_len[old_par] = old_lens[child]

    parent[new_root_id] = -1
    branch_len[new_root_id] = float("nan")
    tree["root"] = int(new_root_id)
    return tree


# ---------------------------------------------------------------------------
# Iterative post-order
# ---------------------------------------------------------------------------

def iterative_postorder(tree):
    root = tree["root"]
    children = tree["children"]
    n = tree["n_nodes"]
    order = np.empty(n, dtype=np.int32)
    out_idx = 0
    stack = [(root, False)]
    while stack:
        v, visited = stack.pop()
        if visited:
            order[out_idx] = v
            out_idx += 1
        else:
            stack.append((v, True))
            for c in children[v]:
                stack.append((c, False))
    return order[:out_idx]


# ---------------------------------------------------------------------------
# Projections + accession join
# ---------------------------------------------------------------------------

def extract_accession_from_leaf(leaf_name):
    parts = leaf_name.split("_", 2)
    return parts[1] if len(parts) >= 2 else leaf_name


def extract_accession_from_seq_id(seq_id):
    return seq_id.split("|", 1)[0]


def load_projections(path):
    df = pd.read_csv(path, sep="\t")
    axis_cols = [c for c in df.columns if c.startswith("Up_")]
    if not axis_cols:
        raise ValueError(f"No 'Up_*' columns found in {path}")
    axis_cols = sorted(axis_cols, key=lambda c: int(c.split("_")[1]))
    if "seq_id" not in df.columns:
        raise ValueError(f"'seq_id' column not found in {path}")
    accessions = df["seq_id"].map(extract_accession_from_seq_id)
    if accessions.duplicated().any():
        dups = accessions[accessions.duplicated()].unique()[:5]
        raise ValueError(f"Duplicate accessions in projections: {list(dups)} ...")
    vals = df[axis_cols].to_numpy(dtype=np.float64)
    acc_to_row = dict(zip(accessions.tolist(), vals))
    return acc_to_row, axis_cols


def build_leaf_value_matrix(tree, acc_to_row, n_axes):
    leaf_ids = tree["leaf_ids"]
    names = tree["name"]
    n_leaves = len(leaf_ids)
    X = np.zeros((n_leaves, n_axes), dtype=np.float64)
    valid = np.zeros((n_leaves, n_axes), dtype=bool)
    tree_accs = np.empty(n_leaves, dtype=object)
    matched = 0
    seen_acc = set()
    dup_acc = []
    for i, leaf_id in enumerate(leaf_ids):
        acc = extract_accession_from_leaf(names[leaf_id])
        tree_accs[i] = acc
        if acc in seen_acc:
            dup_acc.append(acc)
        else:
            seen_acc.add(acc)
        row = acc_to_row.get(acc)
        if row is not None:
            X[i] = row
            valid[i] = True
            matched += 1
    if dup_acc:
        raise ValueError(f"Duplicate accessions among tree leaves (first 5): {dup_acc[:5]}")
    return X, valid, tree_accs, matched


# ---------------------------------------------------------------------------
# Ranking + tie correction
# ---------------------------------------------------------------------------

def rank_per_axis(X, valid):
    """Per-axis average-rank of valid values. Invalid rows get rank 0.
    Returns (R, T_k, N_valid) where T_k = sum(t_i**3 - t_i) for tie groups."""
    n_leaves, K = X.shape
    R = np.zeros_like(X)
    T = np.zeros(K, dtype=np.float64)
    N_valid = np.zeros(K, dtype=np.int64)
    for k in range(K):
        m = valid[:, k]
        vals = X[m, k]
        if vals.size:
            R[m, k] = stats.rankdata(vals, method="average")
            _, counts = np.unique(vals, return_counts=True)
            T[k] = np.sum(counts ** 3 - counts)
        N_valid[k] = m.sum()
    return R, T, N_valid


# ---------------------------------------------------------------------------
# Subtree accumulation
# ---------------------------------------------------------------------------

def accumulate_subtree_stats(tree, postorder, X, R, valid):
    n_nodes = tree["n_nodes"]
    children = tree["children"]
    leaf_ids = tree["leaf_ids"]
    K = X.shape[1]

    n_val = np.zeros((n_nodes, K), dtype=np.int64)
    sum_x = np.zeros((n_nodes, K), dtype=np.float64)
    sum_r = np.zeros((n_nodes, K), dtype=np.float64)

    leaf_idx_of_node = -np.ones(n_nodes, dtype=np.int64)
    leaf_idx_of_node[leaf_ids] = np.arange(len(leaf_ids))

    # Seed leaf rows
    leaf_mask = leaf_idx_of_node >= 0
    n_val[leaf_mask] = valid[leaf_idx_of_node[leaf_mask]].astype(np.int64)
    sum_x[leaf_mask] = np.where(valid[leaf_idx_of_node[leaf_mask]],
                                X[leaf_idx_of_node[leaf_mask]], 0.0)
    sum_r[leaf_mask] = R[leaf_idx_of_node[leaf_mask]]  # already zeroed for invalid

    for v in postorder:
        if leaf_idx_of_node[v] >= 0:
            continue
        for c in children[v]:
            n_val[v] += n_val[c]
            sum_x[v] += sum_x[c]
            sum_r[v] += sum_r[c]
    return n_val, sum_x, sum_r


# ---------------------------------------------------------------------------
# Vectorized MWU across all candidate splits
# ---------------------------------------------------------------------------

def candidate_nodes(tree):
    """Non-root nodes that are not unifurcations. Each maps to a unique
    unrooted bipartition. Count is 2*n_leaves - 3 for a binary tree."""
    new_root = tree["root"]
    children = tree["children"]
    n = tree["n_nodes"]
    cand = [v for v in range(n) if v != new_root and len(children[v]) != 1]
    return np.asarray(cand, dtype=np.int32)


def evaluate_all_splits(node_ids, n_val, sum_x, sum_r,
                         N_total, sum_x_total, T_k, min_side_size):
    """Compute MWU statistic and two-sided asymptotic p-value for every
    (candidate node, axis) pair.

    Returns a dict of (m, K) arrays plus node_ids.
    """
    nA = n_val[node_ids].astype(np.float64)               # (m, K)
    sxA = sum_x[node_ids]
    srA = sum_r[node_ids]

    nB = N_total.astype(np.float64) - nA                   # (m, K)
    nT = N_total.astype(np.float64)                        # (K,)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_A = sxA / np.where(nA > 0, nA, np.nan)
        mean_B = (sum_x_total - sxA) / np.where(nB > 0, nB, np.nan)

        U_A = srA - nA * (nA + 1.0) / 2.0
        # tie-corrected variance of U
        var_U = (nA * nB / 12.0) * ((nT + 1.0) - T_k / (nT * (nT - 1.0)))
        mean_U = nA * nB / 2.0
        Z = (U_A - mean_U) / np.sqrt(var_U)
        p_mwu = 2.0 * stats.norm.sf(np.abs(Z))

    valid_split = (nA >= min_side_size) & (nB >= min_side_size)
    p_mwu = np.where(valid_split, p_mwu, np.nan)
    Z = np.where(valid_split, Z, np.nan)
    U_A = np.where(valid_split, U_A, np.nan)
    return {
        "node_ids": node_ids,
        "nA": nA, "nB": nB,
        "mean_A": mean_A, "mean_B": mean_B,
        "U_A": U_A, "Z": Z, "p_mwu": p_mwu,
        "valid": valid_split,
    }


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def bh_fdr(p_flat):
    p = np.asarray(p_flat, dtype=np.float64)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if mask.sum():
        q[mask] = stats.false_discovery_control(p[mask])
    return q


def permutation_minp(tree, postorder, X, R, valid, T_k, N_total, sum_x_total,
                      cand_node_ids, min_side_size, observed_minp,
                      n_perm, seed, log_every=100):
    """Empirical min-p p-value per axis under leaf-label permutation."""
    rng = np.random.default_rng(seed)
    n_leaves, K = X.shape
    counts = np.zeros(K, dtype=np.int64)
    for b in range(n_perm):
        perm = rng.permutation(n_leaves)
        X_p = X[perm]
        R_p = R[perm]
        valid_p = valid[perm]
        n_val, sum_x, sum_r = accumulate_subtree_stats(tree, postorder, X_p, R_p, valid_p)
        res = evaluate_all_splits(cand_node_ids, n_val, sum_x, sum_r,
                                  N_total, sum_x_total, T_k, min_side_size)
        with np.errstate(invalid="ignore"):
            per_axis_min = np.nanmin(res["p_mwu"], axis=0)
        # treat all-NaN axes as p=1
        per_axis_min = np.where(np.isfinite(per_axis_min), per_axis_min, 1.0)
        counts += (per_axis_min <= observed_minp).astype(np.int64)
        if (b + 1) % log_every == 0:
            print(f"  permutation {b + 1}/{n_perm}", file=sys.stderr)
    return (counts + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def descendants_of(tree, node_id):
    """Set of leaf node ids descended from node_id (or {node_id} if it's a leaf)."""
    out = []
    stack = [node_id]
    children = tree["children"]
    while stack:
        v = stack.pop()
        if len(children[v]) == 0:
            out.append(v)
        else:
            stack.extend(children[v])
    return out


def sample_accessions(tree, leaf_node_ids, k, rng):
    names = tree["name"]
    if len(leaf_node_ids) <= k:
        chosen = list(leaf_node_ids)
    else:
        chosen = rng.choice(leaf_node_ids, size=k, replace=False).tolist()
    return [extract_accession_from_leaf(names[n]) for n in chosen]


def plot_split_histograms(out_df, X, valid, tree, axis_cols, output_path):
    """For each axis, overlay histograms of clade vs complement values along
    that axis using the best split for that axis. Layout: 2 columns,
    ceil(K/2) rows; unused panels removed."""
    K = len(axis_cols)
    n_cols = 2
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3, 1 * n_rows),
                              constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    leaf_ids = tree["leaf_ids"]
    leaf_id_to_idx = {int(l): i for i, l in enumerate(leaf_ids)}
    row_by_axis = {row["axis"]: row for _, row in out_df.iterrows()}

    for k, axname in enumerate(axis_cols):
        ax = axes_flat[k]
        ax.tick_params(labelsize=4)
        if axname not in row_by_axis:
            ax.set_title(f"{axname}: no valid split", fontsize=7)
            continue
        row = row_by_axis[axname]
        node_id = int(row["node_id"])
        under = set(int(l) for l in descendants_of(tree, node_id))
        a_idx = np.array([leaf_id_to_idx[l] for l in under
                          if l in leaf_id_to_idx], dtype=np.int64)
        b_mask = np.ones(len(leaf_ids), dtype=bool)
        b_mask[a_idx] = False
        b_idx = np.where(b_mask)[0]
        a_vals = X[a_idx[valid[a_idx, k]], k]
        b_vals = X[b_idx[valid[b_idx, k]], k]
        if a_vals.size == 0 or b_vals.size == 0:
            ax.set_title(f"{axname}: empty side", fontsize=7)
            continue
        # Smaller side is the "clade" — match the reporting convention
        if a_vals.size <= b_vals.size:
            clade_vals, comp_vals = a_vals, b_vals
        else:
            clade_vals, comp_vals = b_vals, a_vals
        all_vals = np.concatenate([clade_vals, comp_vals])
        bins = np.linspace(all_vals.min(), all_vals.max(), 50)
        ax.hist(comp_vals, bins=bins, color="tab:gray", alpha=0.6,
                label=f"complement (n={len(comp_vals)})")
        ax.hist(clade_vals, bins=bins, color="tab:red", alpha=0.7,
                label=f"clade (n={len(clade_vals)})")
        ax.set_title(
            f"{axname}, "
            f"δ={row['cliffs_delta']:.2f}",
            fontsize=6,
        )
        #ax.legend(fontsize=4, frameon=False)

    for j in range(K, n_rows * n_cols):
        fig.delaxes(axes_flat[j])

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def bipartition_id(tree, smaller_side_leaf_node_ids):
    """A content-based ID for a bipartition: MD5 of the comma-joined sorted
    accessions on the smaller side, truncated to 12 hex chars. Two axes whose
    optimal splits separate the same set of sequences share the same ID even
    if the picked node id happens to differ."""
    names = tree["name"]
    acc = sorted(extract_accession_from_leaf(names[int(n)])
                 for n in smaller_side_leaf_node_ids)
    digest = hashlib.md5(",".join(acc).encode("utf-8")).hexdigest()
    return digest[:12]


def verify_split_reference(node_ids, X, valid, tree, rng, p_mwu, min_side_size):
    """Sanity check: pick a random valid split and compare vectorized MWU
    against scipy.stats.mannwhitneyu on the raw values."""
    mask = np.isfinite(p_mwu).all(axis=1) if p_mwu.ndim == 2 else np.isfinite(p_mwu)
    valid_rows = np.where(mask)[0]
    if valid_rows.size == 0:
        return
    pick = int(rng.choice(valid_rows))
    node_id = int(node_ids[pick])
    side_a_leaves = set(descendants_of(tree, node_id))
    leaf_ids = tree["leaf_ids"]
    side_a_idx = np.array([i for i, l in enumerate(leaf_ids) if l in side_a_leaves], dtype=np.int64)
    side_b_idx = np.array([i for i, l in enumerate(leaf_ids) if l not in side_a_leaves], dtype=np.int64)
    for k in range(X.shape[1]):
        va = valid[side_a_idx, k]
        vb = valid[side_b_idx, k]
        a = X[side_a_idx, k][va]
        b = X[side_b_idx, k][vb]
        if min(len(a), len(b)) < min_side_size:
            continue
        _u, p_scipy = stats.mannwhitneyu(a, b, alternative="two-sided")
        p_vec = p_mwu[pick, k]
        if not np.isfinite(p_vec):
            continue
        # Compare on a log scale to tolerate tiny absolute differences for huge tests
        rel_err = abs(np.log10(max(p_scipy, 1e-300)) - np.log10(max(p_vec, 1e-300)))
        ok = "OK" if rel_err < 0.05 else "MISMATCH"
        print(f"  verify axis {k}: vectorized p={p_vec:.3e}  scipy p={p_scipy:.3e}  {ok}",
              file=sys.stderr)
        return  # one axis is enough


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", default="./PF01316_core/protein-matching-PF01316_sanitized.fasta.tree")
    p.add_argument("--projections", default="./PF01316_core/seq_projections.tsv")
    p.add_argument("--output-dir", default="./PF01316_core/tree_splits")
    p.add_argument("--min-side-size", type=int, default=10,
                    help="Skip splits where either side has fewer valid leaves (default 10)")
    p.add_argument("--permutation", type=int, default=0,
                    help="Number of leaf-label permutations for empirical min-p (default 0 = skip)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- Parse tree ------------------------------------------------------
    print(f"Parsing tree: {args.tree}", file=sys.stderr)
    with open(args.tree) as fh:
        tree_text = fh.read()
    # Sanity precheck: leaf count = commas + 1 (only true for balanced Newick
    # with one leaf per leaf-token; correct here)
    comma_count = tree_text.count(",")

    tree = parse_newick_iterative(tree_text)
    n_leaves = len(tree["leaf_ids"])
    print(f"  n_nodes={tree['n_nodes']} n_leaves={n_leaves} (commas+1={comma_count + 1})",
          file=sys.stderr)

    # ---- Load projections ------------------------------------------------
    print(f"Loading projections: {args.projections}", file=sys.stderr)
    acc_to_row, axis_cols = load_projections(args.projections)
    K = len(axis_cols)
    print(f"  n_proj={len(acc_to_row)} axes={axis_cols}", file=sys.stderr)

    # ---- Build leaf-value matrix ----------------------------------------
    X, valid, _tree_accs, matched = build_leaf_value_matrix(tree, acc_to_row, K)
    tree_only = n_leaves - matched
    proj_only = len(acc_to_row) - matched
    print(f"  matched={matched}  tree_only={tree_only}  proj_only={proj_only}",
          file=sys.stderr)

    # ---- Re-root at first matched leaf -----------------------------------
    matched_leaf_idx = int(np.argmax(valid.any(axis=1)))
    new_root_node = int(tree["leaf_ids"][matched_leaf_idx])
    print(f"Re-rooting at leaf node {new_root_node} "
          f"({tree['name'][new_root_node]})", file=sys.stderr)
    reroot_at_leaf(tree, new_root_node)

    # ---- Rank values per axis -------------------------------------------
    R, T_k, N_valid = rank_per_axis(X, valid)
    sum_x_total = (X * valid).sum(axis=0)
    print(f"  N_valid per axis: {N_valid.tolist()}", file=sys.stderr)

    # ---- Post-order + subtree accumulation -------------------------------
    print("Post-order accumulation", file=sys.stderr)
    postorder = iterative_postorder(tree)
    n_val, sum_x, sum_r = accumulate_subtree_stats(tree, postorder, X, R, valid)

    if not np.array_equal(N_valid, valid.sum(axis=0)):
        raise AssertionError("N_valid mismatch")

    # ---- Enumerate candidate splits --------------------------------------
    cand_node_ids = candidate_nodes(tree)
    expected = 2 * n_leaves - 3
    print(f"  candidate splits: {len(cand_node_ids)} (expected 2N-3 = {expected})",
          file=sys.stderr)

    # ---- Vectorized MWU --------------------------------------------------
    print("Evaluating splits (vectorized MWU)", file=sys.stderr)
    res = evaluate_all_splits(cand_node_ids, n_val, sum_x, sum_r,
                              N_valid, sum_x_total, T_k, args.min_side_size)
    p_mwu = res["p_mwu"]  # (m, K)

    # ---- Reference check -------------------------------------------------
    verify_split_reference(cand_node_ids, X, valid, tree, rng, p_mwu,
                            args.min_side_size)

    # ---- BH FDR over (splits * axes) -------------------------------------
    q_bh = bh_fdr(p_mwu.ravel()).reshape(p_mwu.shape)

    # ---- Per-axis best ---------------------------------------------------
    best_rows = []
    observed_minp = np.full(K, np.nan)
    for k in range(K):
        col = p_mwu[:, k]
        if not np.any(np.isfinite(col)):
            print(f"  axis {axis_cols[k]}: no valid splits", file=sys.stderr)
            best_rows.append(None)
            continue
        idx = int(np.nanargmin(col))
        observed_minp[k] = col[idx]
        best_rows.append(idx)

    # ---- Permutation min-p (optional) ------------------------------------
    perm_p = None
    if args.permutation > 0:
        print(f"Permutation min-p ({args.permutation} reps)", file=sys.stderr)
        perm_p = permutation_minp(tree, postorder, X, R, valid, T_k, N_valid,
                                   sum_x_total, cand_node_ids,
                                   args.min_side_size, observed_minp,
                                   args.permutation, args.seed)

    # ---- Build output table ----------------------------------------------
    n_tests = int(np.isfinite(p_mwu).sum())
    rows = []
    for k, idx in enumerate(best_rows):
        if idx is None:
            continue
        node_id = int(cand_node_ids[idx])
        nA = int(res["nA"][idx, k])
        nB = int(res["nB"][idx, k])
        mA = float(res["mean_A"][idx, k])
        mB = float(res["mean_B"][idx, k])
        U = float(res["U_A"][idx, k])
        Z = float(res["Z"][idx, k])
        p = float(res["p_mwu"][idx, k])
        q = float(q_bh[idx, k])
        # Cliff's delta from side-A's perspective: 2*U_A/(nA*nB) - 1, in [-1, 1].
        # Re-express so "clade" is the smaller side and the sign reflects
        # clade-vs-complement.
        cliffs_delta_A = 2.0 * U / (nA * nB) - 1.0
        under_set = set(descendants_of(tree, node_id))
        all_leaves = [int(l) for l in tree["leaf_ids"]]
        complement_leaves = [l for l in all_leaves if l not in under_set]
        if nA <= nB:
            clade_leaves = list(under_set)
            clade_size = nA
            complement_leaves_n = nB
            mean_clade, mean_complement = mA, mB
            cliffs_delta = cliffs_delta_A
        else:
            clade_leaves = complement_leaves
            complement_leaves = list(under_set)
            clade_size = nB
            complement_leaves_n = nA
            mean_clade, mean_complement = mB, mA
            cliffs_delta = -cliffs_delta_A
        rep_clade = sample_accessions(tree, clade_leaves, 5, rng)
        rep_complement = sample_accessions(tree, complement_leaves, 5, rng)
        bip_id = bipartition_id(tree, clade_leaves)
        row = {
            "axis": axis_cols[k],
            "node_id": node_id,
            "bipartition_id": bip_id,
            "clade_size": clade_size,
            "complement_size": complement_leaves_n,
            "mean_clade": mean_clade,
            "mean_complement": mean_complement,
            "cliffs_delta": cliffs_delta,
            "U_stat": U,
            "Z": Z,
            "p_mwu_raw": p,
            "p_bonferroni": min(1.0, p * n_tests),
            "q_bh": q,
            "representative_leaves_clade": ";".join(rep_clade),
            "representative_leaves_complement": ";".join(rep_complement),
        }
        if perm_p is not None:
            row["p_permutation_minp"] = float(perm_p[k])
        rows.append(row)
    out_df = pd.DataFrame(rows)

    # ---- Print summary ---------------------------------------------------
    print("\n=== Best split per axis ===", file=sys.stdout)
    summary_cols = ["axis", "bipartition_id", "clade_size", "complement_size",
                    "mean_clade", "mean_complement", "cliffs_delta",
                    "p_mwu_raw", "q_bh"]
    if perm_p is not None:
        summary_cols.append("p_permutation_minp")
    with pd.option_context("display.float_format", "{:.4g}".format,
                            "display.max_columns", None,
                            "display.width", 200):
        print(out_df[summary_cols].to_string(index=False), file=sys.stdout)

    # ---- Write TSV -------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_split_per_axis.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}", file=sys.stdout)

    # ---- Plot histograms of each axis split ------------------------------
    fig_path = out_dir / "Up_splits.png"
    plot_split_histograms(out_df, X, valid, tree, axis_cols, fig_path)
    print(f"Wrote {fig_path}", file=sys.stdout)

    print(f"Total runtime: {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
