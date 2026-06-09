"""Pagel_lambda_phylo_signal.py

Test whether each ICA component (Up_k axis) of a sequence-projection table
reflects *phylogeny* rather than some aspect of function.

For each component we measure Pagel's lambda, the phylogenetic signal of the
projection values treated as a continuous trait on a guide tree:

  * lambda ~ 1  -> the trait varies as expected under Brownian-motion descent
                   along the tree, i.e. it largely tracks phylogeny.
  * lambda ~ 0  -> the trait is independent of the tree, consistent with a
                   non-phylogenetic (e.g. functional) signal.

Pipeline:

  1. Parse the Newick guide tree; join leaves to the projection table by
     UniProt accession; keep only sequences present in BOTH files.
  2. (default on) Van der Waerden / normal-scores transform of each component
     to make its marginal ~N(0,1), the normality the Gaussian lambda model
     assumes.  Disable with --no-transform.
  3. Fit Pagel's lambda per component by maximum likelihood, using Felsenstein's
     independent contrasts on the lambda-rescaled tree (O(N) per evaluation, no
     N*N covariance matrix is ever formed -> scales to 10k+ tips).
  4. Likelihood-ratio test of lambda_hat vs the lambda = 0 (no-signal) model;
     boundary-corrected p-value; Benjamini-Hochberg FDR across components.
  5. Report lambda and p per component to stdout and a TSV.

The Newick parser, accession helpers and BH-FDR are reused from
Identify_tree_splits.py (proven on this exact tree file).
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Newick tokenizer + iterative parser  (from Identify_tree_splits.py)
# ---------------------------------------------------------------------------

_NEWICK_DELIMS = set("(),:;")


def tokenize_newick(text):
    """Yield Newick tokens (single delimiters or runs of name/number chars).
    All whitespace, including the newlines this tree file uses between a label
    and its branch length, is skipped.
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
    """Parse a Newick string into flat arrays. No recursion."""
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

    stack = []
    current = None
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


def iterative_postorder(tree):
    """Children-before-parents ordering of reachable nodes, from the root."""
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
# Accession join  (from Identify_tree_splits.py)
# ---------------------------------------------------------------------------

def extract_accession_from_leaf(leaf_name):
    """Tree label '1_A0A010Z1T0_unreviewed_..._taxID_927661' -> 'A0A010Z1T0'."""
    parts = leaf_name.split("_", 2)
    return parts[1] if len(parts) >= 2 else leaf_name


def extract_accession_from_seq_id(seq_id):
    """TSV seq_id 'A0A010Z1T0|unreviewed|...|taxID:927661' -> 'A0A010Z1T0'."""
    return seq_id.split("|", 1)[0]


def load_projections(path):
    """Return (acc -> projection-row dict, sorted Up_* column names)."""
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


def bh_fdr(p_flat):
    """Benjamini-Hochberg FDR (repo convention)."""
    p = np.asarray(p_flat, dtype=np.float64)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if mask.sum():
        q[mask] = stats.false_discovery_control(p[mask])
    return q


# ---------------------------------------------------------------------------
# Prune the tree to a set of accessions (keep relatives' topology)
# ---------------------------------------------------------------------------

def prune_to_accessions(tree, keep_acc):
    """Drop every leaf whose accession is not in ``keep_acc`` and every internal
    node that loses all descendants. Unifurcations are left in place (they
    contribute no contrast and preserve root-to-tip distances). Mutates and
    returns ``tree`` with refreshed ``children`` / ``is_leaf``.
    """
    children = tree["children"]
    name = tree["name"]
    is_leaf = tree["is_leaf"]
    n = tree["n_nodes"]
    root = tree["root"]

    post = iterative_postorder(tree)
    keep = np.zeros(n, dtype=bool)
    for node in post:  # children before parents
        if is_leaf[node]:
            keep[node] = extract_accession_from_leaf(name[node]) in keep_acc
        else:
            keep[node] = any(keep[c] for c in children[node])

    if not keep[root]:
        raise ValueError("Pruning removed the root: no kept leaves under it.")

    new_children = [[] for _ in range(n)]
    for node in range(n):
        if keep[node] and not is_leaf[node]:
            new_children[node] = [c for c in children[node] if keep[c]]

    tree["children"] = new_children
    tree["is_leaf"] = np.array(
        [keep[i] and len(new_children[i]) == 0 for i in range(n)], dtype=bool
    )
    tree["leaf_ids"] = np.where(tree["is_leaf"])[0].astype(np.int32)
    return tree


# ---------------------------------------------------------------------------
# Geometry precompute (postorder, child lists, root-to-node distances)
# ---------------------------------------------------------------------------

def build_geometry(tree):
    """Precompute everything the per-lambda traversal needs, once."""
    children = tree["children"]
    is_leaf = tree["is_leaf"]
    root = tree["root"]
    n = tree["n_nodes"]

    blen = np.array(tree["branch_len"], dtype=np.float64)
    blen[~np.isfinite(blen)] = 0.0          # root + any missing lengths -> 0

    post = iterative_postorder(tree)        # children before parents

    # Root-to-node distance h[node] via preorder (parents before children).
    h = np.zeros(n, dtype=np.float64)
    stack = [root]
    while stack:
        v = stack.pop()
        for c in children[v]:
            h[c] = h[v] + blen[c]
            stack.append(c)

    return {
        "post": post,
        "children": children,
        "is_leaf": is_leaf,
        "blen": blen,
        "h": h,
        "root": root,
        "n_nodes": n,
        "leaf_ids": tree["leaf_ids"],
    }


# ---------------------------------------------------------------------------
# Brownian-motion log-likelihood on the lambda-rescaled tree (contrasts)
# ---------------------------------------------------------------------------

_EPS = 1e-12


def bm_loglik(lam, geom, y_by_node, n_tips):
    """Maximum-likelihood log-likelihood of the data under Brownian motion on
    the Pagel-lambda transformed tree, computed by Felsenstein's independent
    contrasts in a single post-order pass.

    Pagel's lambda scales off-diagonal phylogenetic covariances by ``lam`` while
    keeping tip variances (root-to-tip distances) fixed.  Equivalent per-edge
    transform used here:
        internal edge length  -> lam * original
        terminal edge length  -> original + (1 - lam) * h(parent)
    so each tip's root-to-tip total is preserved and shared paths shrink by lam.
    """
    post = geom["post"]
    children = geom["children"]
    is_leaf = geom["is_leaf"]
    blen = geom["blen"]
    h = geom["h"]

    Xhat = np.empty(geom["n_nodes"], dtype=np.float64)
    Vacc = np.empty(geom["n_nodes"], dtype=np.float64)

    SS = 0.0          # sum of squared standardized contrasts
    logdet = 0.0      # log|C| accumulator

    for node in post:
        if is_leaf[node]:
            Xhat[node] = y_by_node[node]
            Vacc[node] = 0.0
            continue

        ch = children[node]
        hp = h[node]                       # root-to-parent distance for leaves

        def eff_len(c):
            # lambda-transformed length of the edge entering child c
            if is_leaf[c]:
                return blen[c] + (1.0 - lam) * hp
            return lam * blen[c]

        first = ch[0]
        Vr = eff_len(first) + Vacc[first]
        Xr = Xhat[first]

        for c in ch[1:]:
            tr = Vr if Vr > _EPS else _EPS
            tc = eff_len(c) + Vacc[c]
            tc = tc if tc > _EPS else _EPS
            contrast = Xr - Xhat[c]
            vcon = tr + tc
            SS += contrast * contrast / vcon
            logdet += math.log(vcon)
            wr = 1.0 / tr
            wc = 1.0 / tc
            Xr = (wr * Xr + wc * Xhat[c]) / (wr + wc)
            Vr = 1.0 / (wr + wc)

        Xhat[node] = Xr
        Vacc[node] = Vr

    # root variance term completes log|C|
    logdet += math.log(Vacc[geom["root"]] if Vacc[geom["root"]] > _EPS else _EPS)

    sigma2 = SS / n_tips
    if sigma2 < _EPS:
        sigma2 = _EPS
    return -0.5 * (n_tips * math.log(2.0 * math.pi * sigma2) + logdet + n_tips)


def fit_lambda(geom, y_by_node, n_tips):
    """ML estimate of Pagel's lambda in [0, 1], with logLik at the optimum and
    at the lambda = 0 (no phylogenetic signal) null."""
    res = optimize.minimize_scalar(
        lambda lam: -bm_loglik(lam, geom, y_by_node, n_tips),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    lam_hat = float(res.x)
    ll_hat = bm_loglik(lam_hat, geom, y_by_node, n_tips)
    ll_null = bm_loglik(0.0, geom, y_by_node, n_tips)
    # Guard: the optimizer can stop just shy of a boundary optimum.
    ll_one = bm_loglik(1.0, geom, y_by_node, n_tips)
    if ll_one > ll_hat:
        lam_hat, ll_hat = 1.0, ll_one
    if ll_null > ll_hat:
        lam_hat, ll_hat = 0.0, ll_null
    return lam_hat, ll_hat, ll_null


# ---------------------------------------------------------------------------
# Van der Waerden / normal-scores transform
# ---------------------------------------------------------------------------

def van_der_waerden(x):
    """Rank-based inverse-normal transform: norm.ppf(rank / (n + 1))."""
    n = len(x)
    ranks = stats.rankdata(x, method="average")
    return stats.norm.ppf(ranks / (n + 1.0))


# ---------------------------------------------------------------------------
# Random-projection null  (is an IC's lambda higher than a generic projection?)
# ---------------------------------------------------------------------------
#
# A sequence projection onto ANY direction in alignment space inherits the
# family's overall phylogenetic autocorrelation, so a large lambda is the
# expected baseline rather than evidence specific to a given IC.  To calibrate
# this we project the (one-hot encoded) alignment onto many random directions in
# the same position x residue feature space SCA operates in, fit lambda to each,
# and compare each IC's observed lambda to that null distribution.

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"


def build_code_matrix(seqs):
    """Encode aligned sequences as an int8 matrix (n x L) of symbol indices over
    `_ALPHABET` (+1 catch-all). A random projection of the implied one-hot tensor
    is then `sum_pos R[pos, code[:, pos]]` for a random R of shape (L, n_symbols)."""
    n = len(seqs)
    L = len(seqs[0])
    for s in seqs:
        if len(s) != L:
            raise ValueError("Aligned sequences differ in length; not an alignment.")
    other = len(_ALPHABET)                      # index for symbols not in alphabet
    lut = np.full(256, other, dtype=np.uint8)
    for idx, ch in enumerate(_ALPHABET):
        lut[ord(ch)] = idx
        lut[ord(ch.lower())] = idx
    lut[ord(".")] = _ALPHABET.index("-")        # '.' is a gap/insert variant
    code = np.empty((n, L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        code[i] = lut[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
    return code, other + 1


def null_lambda_distribution(geom, leaf_ids, code, n_symbols, n_tips,
                             n_null, transform, rng):
    """ML lambda for `n_null` random linear projections of the encoded alignment."""
    L = code.shape[1]
    col_idx = np.arange(L)
    y_by_node = np.zeros(geom["n_nodes"], dtype=np.float64)
    lams = np.empty(n_null, dtype=np.float64)
    for d in range(n_null):
        R = rng.standard_normal((L, n_symbols))
        proj = R[col_idx, code].sum(axis=1)     # (n_tips,) random projection
        vals = van_der_waerden(proj) if transform else proj
        y_by_node[leaf_ids] = vals
        lams[d] = fit_lambda(geom, y_by_node, n_tips)[0]
        if (d + 1) % 25 == 0 or (d + 1) == n_null:
            print(f"    null draw {d + 1}/{n_null}", file=sys.stderr)
    return lams


def load_aligned_sequences(path):
    """acc -> aligned_sequence string."""
    df = pd.read_csv(path, sep="\t", usecols=["seq_id", "aligned_sequence"])
    acc = df["seq_id"].map(extract_accession_from_seq_id)
    return dict(zip(acc.tolist(), df["aligned_sequence"].tolist()))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tree",
                   default="./PF01316_core/protein-matching-PF01316_sanitized.fasta.tree")
    p.add_argument("--projections", default="./PF01316_core/seq_projections.tsv")
    p.add_argument("--output", default="./PF01316_core/pagel_lambda_results.tsv")
    p.add_argument("--no-transform", action="store_true",
                   help="Skip the van der Waerden transform; fit lambda on raw "
                        "projection values.")
    p.add_argument("--no-boundary-correction", action="store_true",
                   help="Use a plain chi-square(1) LRT p-value instead of the "
                        "0.5*chi2 boundary mixture (lambda=0 is on the boundary).")
    p.add_argument("--null-projections", type=int, default=0, metavar="N",
                   help="If >0, build N random linear projections of the "
                        "alignment, fit lambda to each, and report every IC's "
                        "lambda as a z-score / empirical p against that null. "
                        "Tells you whether an IC's signal exceeds the generic "
                        "phylogenetic autocorrelation of any sequence projection.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # --- load -------------------------------------------------------------
    print(f"Reading tree:        {args.tree}", file=sys.stderr)
    tree = parse_newick_iterative(open(args.tree).read())
    n_tree = int(tree["is_leaf"].sum())

    print(f"Reading projections: {args.projections}", file=sys.stderr)
    acc_to_row, axis_cols = load_projections(args.projections)
    n_tsv = len(acc_to_row)

    tree_acc = {extract_accession_from_leaf(tree["name"][i])
                for i in tree["leaf_ids"]}
    common = tree_acc & set(acc_to_row)
    print(f"  sequences in TSV:   {n_tsv}", file=sys.stderr)
    print(f"  leaves in tree:     {n_tree}", file=sys.stderr)
    print(f"  in both (kept):     {len(common)}", file=sys.stderr)
    print(f"  tree-only (dropped):{n_tree - len(common)}", file=sys.stderr)
    if len(common) < 3:
        raise ValueError("Need at least 3 shared sequences to fit lambda.")

    # --- prune + geometry -------------------------------------------------
    prune_to_accessions(tree, common)
    geom = build_geometry(tree)
    leaf_ids = geom["leaf_ids"]
    n_tips = len(leaf_ids)

    # leaf node -> its row of projection values
    leaf_acc = [extract_accession_from_leaf(tree["name"][i]) for i in leaf_ids]
    proj = np.array([acc_to_row[a] for a in leaf_acc], dtype=np.float64)  # n_tips x K

    transform = not args.no_transform
    print(f"\nFitting Pagel's lambda for {len(axis_cols)} components on "
          f"{n_tips} sequences "
          f"({'van der Waerden transform' if transform else 'raw values'}).\n",
          file=sys.stderr)

    # --- per-component fit ------------------------------------------------
    rows = []
    y_by_node = np.zeros(geom["n_nodes"], dtype=np.float64)
    for k, col in enumerate(axis_cols):
        vals = proj[:, k]
        if transform:
            vals = van_der_waerden(vals)
        y_by_node[leaf_ids] = vals

        lam, ll_hat, ll_null = fit_lambda(geom, y_by_node, n_tips)
        lrt = 2.0 * (ll_hat - ll_null)
        if lrt <= 0:
            pval = 1.0
        elif args.no_boundary_correction:
            pval = float(stats.chi2.sf(lrt, df=1))
        else:
            pval = 0.5 * float(stats.chi2.sf(lrt, df=1))
        rows.append({
            "component": col,
            "lambda": lam,
            "logLik": ll_hat,
            "logLik_null": ll_null,
            "LRT": lrt,
            "p_value": pval,
            "n": n_tips,
        })
        print(f"  {col:>7}: lambda={lam:6.4f}  LRT={lrt:9.2f}  p={pval:.3e}",
              file=sys.stderr)

    out = pd.DataFrame(rows)
    out["p_adj_BH"] = bh_fdr(out["p_value"].to_numpy())
    out = out[["component", "lambda", "logLik", "logLik_null",
               "LRT", "p_value", "p_adj_BH", "n"]]

    # --- random-projection null (optional) --------------------------------
    null_summary = None
    if args.null_projections > 0:
        seq_map = load_aligned_sequences(args.projections)
        missing = [a for a in leaf_acc if a not in seq_map]
        if missing:
            raise ValueError(f"{len(missing)} sequences lack aligned_sequence, "
                             f"e.g. {missing[:3]}")
        code, n_symbols = build_code_matrix([seq_map[a] for a in leaf_acc])
        rng = np.random.default_rng(args.seed)
        print(f"\nCalibrating against {args.null_projections} random projections "
              f"of the alignment (L={code.shape[1]}, {n_symbols} symbols)...",
              file=sys.stderr)
        null_lams = null_lambda_distribution(geom, leaf_ids, code, n_symbols,
                                             n_tips, args.null_projections,
                                             transform, rng)
        nm = float(null_lams.mean())
        ns = float(null_lams.std(ddof=1)) if args.null_projections > 1 else 0.0
        q = np.quantile(null_lams, [0.5, 0.95, 0.99])
        null_summary = (f"null lambda from {args.null_projections} random "
                        f"projections: mean={nm:.4f} sd={ns:.4f} "
                        f"median={q[0]:.4f} p95={q[1]:.4f} p99={q[2]:.4f} "
                        f"range=[{null_lams.min():.4f}, {null_lams.max():.4f}]")
        print("  " + null_summary, file=sys.stderr)

        lam_obs = out["lambda"].to_numpy()
        out["lambda_null_mean"] = nm
        out["lambda_null_sd"] = ns
        out["lambda_z"] = (lam_obs - nm) / ns if ns > 0 else np.nan
        # one-sided empirical p: is the IC's lambda above the null?
        out["p_vs_null"] = [(1 + int(np.sum(null_lams >= lo)))
                            / (args.null_projections + 1) for lo in lam_obs]
        out["p_vs_null_BH"] = bh_fdr(out["p_vs_null"].to_numpy())

    out.to_csv(args.output, sep="\t", index=False, float_format="%.6g")

    # --- report -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("Pagel's lambda phylogenetic-signal test "
          f"({'van der Waerden' if transform else 'raw'}, n={n_tips})")
    print("=" * 72)
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print(out.to_string(index=False))
    print(f"\nWrote {args.output}")
    print("Interpretation: lambda near 1 with small p_adj_BH => the component "
          "tracks phylogeny;\n  lambda near 0 => signal is independent of the "
          "tree (consistent with function).")
    if null_summary is not None:
        print("\n" + null_summary)
        print("vs-null: lambda_z / p_vs_null ask whether an IC's lambda EXCEEDS "
              "the\n  phylogenetic autocorrelation of a generic random projection."
              " Components\n  within the null band carry no more phylogenetic "
              "signal than any axis; only\n  those well above it (large positive "
              "z, small p_vs_null_BH) show excess phylogeny.")


if __name__ == "__main__":
    main()
