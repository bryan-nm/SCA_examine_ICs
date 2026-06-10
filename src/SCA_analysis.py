"""SCA_analysis.py

End-to-end phylogenetic-signal / clade-split analysis of the SCA independent
components (ICs) of a single protein family, starting from a finished SCA
results folder (e.g. ``PF00001/``).

Pipeline
--------
  1. Load the integerized, preprocessed MSA from
     ``<family>/preprocessing/preprocessing_results.npz`` ('msa' key) plus the
     parallel ``retained_sequence_ids``.
  2. Randomly subsample to a fixed number of sequences (default 1000). The
     processed alignment has already been redundancy-reduced (clustered to
     ~80% ID upstream), so uniform subsampling is reasonable here.
  3. Project the subsample onto the family's IC sequence-space coordinates
     (Rivoire et al. 2016 Uᵖ scores) via ``mysca.SCAResults.project_sequences``
     using the eigendecomposition / ICA already stored in ``<family>/scacore``.
  4. Build a guide tree over the subsample with FAMSA (``-gt_export``).
  5. For every IC (Up_k):
       * Pagel's lambda phylogenetic-signal test (does the IC track the tree?)
       * Best clade split by Mann-Whitney U + Cliff's delta (does some clade
         separate cleanly along the IC?)
     The numeric kernels are imported and reused verbatim from
     ``Pagel_lambda_phylo_signal.py`` and ``Identify_tree_splits.py``.

Nothing is ever written into the SCA results folder. All intermediates and
outputs go under a ``tmp/<family>/`` directory created in the current working
directory.

Leaf-label trick: each subsampled sequence is given a synthetic, delimiter-free
label ``s<idx>``. The reused helpers' accession extractors (which split on
``_`` / ``|``) then act as the identity on these labels, so the tree<->Up join
is exact and decoupled from any UniProt naming convention.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the proven kernels from the two sibling scripts. Importing by module
# name requires this script's own directory on sys.path regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Pagel_lambda_phylo_signal as pg   # noqa: E402
import Identify_tree_splits as sp        # noqa: E402


# ---------------------------------------------------------------------------
# Loading + subsampling + projection
# ---------------------------------------------------------------------------

def load_processed_msa(preprocess_dir):
    """Return (msa int array M x L, seq_ids M, int2char lookup array)."""
    npz = os.path.join(preprocess_dir, "preprocessing_results.npz")
    ali = np.load(npz, allow_pickle=True)
    msa = np.asarray(ali["msa"])
    seq_ids = np.asarray(ali["retained_sequence_ids"]).astype(str)
    if msa.shape[0] != seq_ids.shape[0]:
        raise ValueError(
            f"msa rows ({msa.shape[0]}) != retained_sequence_ids "
            f"({seq_ids.shape[0]}) in {npz}"
        )

    sym2int = json.load(open(os.path.join(preprocess_dir, "sym2int.json")))
    n_sym = max(sym2int.values()) + 1
    int2char = np.empty(n_sym, dtype="<U1")
    for ch, idx in sym2int.items():
        int2char[idx] = ch
    return msa, seq_ids, int2char


def subsample_indices(n_rows, n_target, rng):
    """Row indices for the subsample (all rows if n_rows <= n_target)."""
    if n_rows <= n_target:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=n_target, replace=False))


def onehot_from_int_msa(msa_int, n_aa=20):
    """(M, L) ints (0 = gap, 1..n_aa = residues, matching sym2int) ->
    (M, L, n_aa) bool one-hot with all-zero rows at gaps. Mirrors the
    onehot_without_gap convention SCAResults.project_sequences expects."""
    M, L = msa_int.shape
    xmsa = np.zeros((M, L, n_aa), dtype=bool)
    rows, cols = np.where(msa_int > 0)
    xmsa[rows, cols, msa_int[rows, cols] - 1] = True
    return xmsa


def project_subsample(sca, msa_int_sub):
    """Uᵖ scores (n_sub x K) for the subsampled integerized MSA."""
    n_aa = sca.fia.shape[1]
    xmsa = onehot_from_int_msa(msa_int_sub, n_aa=n_aa)
    return sca.project_sequences(xmsa)


# ---------------------------------------------------------------------------
# FAMSA guide tree
# ---------------------------------------------------------------------------

def write_fasta(path, labels, msa_int_sub, int2char):
    """Write ungapped sequences (FAMSA strips gaps anyway) under synthetic
    delimiter-free labels."""
    gap_idx = int(np.where(int2char == "-")[0][0])
    with open(path, "w") as fh:
        for lab, row in zip(labels, msa_int_sub):
            seq = "".join(int2char[v] for v in row if v != gap_idx)
            fh.write(f">{lab}\n{seq}\n")


def famsa_guide_tree(famsa_bin, fasta_path, tree_path, gt_method, threads):
    """Run FAMSA to export only a Newick guide tree (no full alignment)."""
    cmd = [famsa_bin, "-gt", gt_method, "-gt_export",
           "-t", str(threads), fasta_path, tree_path]
    print("  $ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(tree_path) or os.path.getsize(tree_path) == 0:
        raise RuntimeError(f"FAMSA produced no guide tree at {tree_path}")


def resolve_famsa(famsa_bin):
    if famsa_bin and os.path.exists(famsa_bin):
        return famsa_bin
    found = shutil.which("famsa") or shutil.which("FAMSA")
    if found:
        return found
    raise FileNotFoundError(
        "FAMSA binary not found on PATH. Install FAMSA "
        "(https://github.com/refresh-bio/FAMSA) or pass --famsa-bin /path/to/famsa.")


# ---------------------------------------------------------------------------
# Pagel's lambda over every IC (reuses Pagel_lambda_phylo_signal kernels)
# ---------------------------------------------------------------------------

def null_lambda_lrt_distribution(geom, leaf_ids, code, n_symbols, n_tips,
                                 n_null, transform, rng):
    """Per random alignment projection, fit lambda AND record the lambda>0
    likelihood-ratio statistic.

    Mirrors ``pg.null_lambda_distribution`` but also returns the LRT so an
    IC's observed LRT can be calibrated against the generic-projection null.
    Unlike lambda (which saturates at 1.0 and loses per-IC resolution at large
    n), the LRT keeps scaling with how strongly a trait tracks the tree.
    """
    L = code.shape[1]
    col_idx = np.arange(L)
    y_by_node = np.zeros(geom["n_nodes"], dtype=np.float64)
    lams = np.empty(n_null, dtype=np.float64)
    lrts = np.empty(n_null, dtype=np.float64)
    for d in range(n_null):
        R = rng.standard_normal((L, n_symbols))
        proj = R[col_idx, code].sum(axis=1)
        vals = pg.van_der_waerden(proj) if transform else proj
        y_by_node[leaf_ids] = vals
        _lam, ll_hat, ll_null = pg.fit_lambda(geom, y_by_node, n_tips)
        lams[d] = _lam
        lrts[d] = max(0.0, 2.0 * (ll_hat - ll_null))
        if (d + 1) % 25 == 0 or (d + 1) == n_null:
            print(f"    null draw {d + 1}/{n_null}", file=sys.stderr)
    return lams, lrts


def run_pagel(tree_text, acc_to_row, comp_names, transform=True,
              boundary_correction=True, aligned_seqs=None, n_null=0, seed=0):
    tree = pg.parse_newick_iterative(tree_text)
    tree_acc = {pg.extract_accession_from_leaf(tree["name"][i])
                for i in tree["leaf_ids"]}
    common = tree_acc & set(acc_to_row)
    if len(common) < 3:
        raise ValueError("Need >=3 shared sequences for Pagel's lambda.")

    pg.prune_to_accessions(tree, common)
    geom = pg.build_geometry(tree)
    leaf_ids = geom["leaf_ids"]
    n_tips = len(leaf_ids)
    leaf_acc = [pg.extract_accession_from_leaf(tree["name"][i]) for i in leaf_ids]
    proj = np.array([acc_to_row[a] for a in leaf_acc], dtype=np.float64)

    rows = []
    y_by_node = np.zeros(geom["n_nodes"], dtype=np.float64)
    for k, col in enumerate(comp_names):
        vals = proj[:, k]
        if transform:
            vals = pg.van_der_waerden(vals)
        y_by_node[leaf_ids] = vals
        lam, ll_hat, ll_null = pg.fit_lambda(geom, y_by_node, n_tips)
        lrt = 2.0 * (ll_hat - ll_null)
        if lrt <= 0:
            pval = 1.0
        elif boundary_correction:
            pval = 0.5 * float(stats.chi2.sf(lrt, df=1))
        else:
            pval = float(stats.chi2.sf(lrt, df=1))
        # lamP = Pagel's lambda (phylogenetic signal), named to avoid clashing
        # with the SCA component magnitude "lambda" reported elsewhere.
        rows.append({"component": col, "lamP": lam, "LRT": lrt,
                     "p_value": pval, "n": n_tips})
        print(f"    {col}: lamP={lam:6.6f}  LRT={lrt:9.2f}  p={pval:.3e}",
              file=sys.stderr)

    out = pd.DataFrame(rows)
    out["p_adj_BH"] = pg.bh_fdr(out["p_value"].to_numpy())

    # Random-projection null: a sequence projection onto ANY direction in
    # alignment space inherits the family's phylogenetic autocorrelation, so a
    # large lambda is the baseline, not evidence specific to an IC. Calibrate
    # each IC's lambda against the lambda of many random alignment projections.
    if n_null > 0:
        if aligned_seqs is None:
            raise ValueError("null projections requested but aligned_seqs is None")
        seqs = [aligned_seqs[a] for a in leaf_acc]   # leaf_ids order
        code, n_symbols = pg.build_code_matrix(seqs)
        rng = np.random.default_rng(seed)
        print(f"  calibrating against {n_null} random projections "
              f"(L={code.shape[1]}, {n_symbols} symbols)...", file=sys.stderr)
        null_lams, null_lrts = null_lambda_lrt_distribution(
            geom, leaf_ids, code, n_symbols, n_tips, n_null, transform, rng)

        # --- lambda vs null (saturating; kept as a diagnostic) ----------
        nm = float(null_lams.mean())
        ns = float(null_lams.std(ddof=1)) if n_null > 1 else 0.0
        qd = np.quantile(null_lams, [0.5, 0.95, 0.99])
        print(f"  null lamP: mean={nm:.4f} sd={ns:.4f} median={qd[0]:.4f} "
              f"p95={qd[1]:.4f} p99={qd[2]:.4f}", file=sys.stderr)
        lam_obs = out["lamP"].to_numpy()
        out["lamP_null_mean"] = nm
        out["lamP_null_sd"] = ns
        out["lamP_z"] = (lam_obs - nm) / ns if ns > 0 else np.nan
        out["p_vs_null"] = [(1 + int(np.sum(null_lams >= lo))) / (n_null + 1)
                            for lo in lam_obs]
        out["p_vs_null_BH"] = pg.bh_fdr(out["p_vs_null"].to_numpy())

        # --- LRT vs null (non-saturating; the discriminating statistic) --
        lm = float(null_lrts.mean())
        ls = float(null_lrts.std(ddof=1)) if n_null > 1 else 0.0
        ql = np.quantile(null_lrts, [0.5, 0.95, 0.99])
        print(f"  null LRT:    mean={lm:.2f} sd={ls:.2f} median={ql[0]:.2f} "
              f"p95={ql[1]:.2f} p99={ql[2]:.2f} max={null_lrts.max():.2f}",
              file=sys.stderr)
        lrt_obs = out["LRT"].to_numpy()
        out["lrt_null_mean"] = lm
        out["lrt_null_sd"] = ls
        out["lrt_z"] = (lrt_obs - lm) / ls if ls > 0 else np.nan
        # one-sided empirical p: does the IC's LRT exceed the null?
        out["p_lrt_vs_null"] = [(1 + int(np.sum(null_lrts >= lo))) / (n_null + 1)
                                for lo in lrt_obs]
        out["p_lrt_vs_null_BH"] = pg.bh_fdr(out["p_lrt_vs_null"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Best clade split over every IC (reuses Identify_tree_splits kernels)
# ---------------------------------------------------------------------------

def run_splits(tree_text, acc_to_row, comp_names, min_side_size, seed):
    rng = np.random.default_rng(seed)
    K = len(comp_names)
    tree = sp.parse_newick_iterative(tree_text)

    X, valid, _accs, matched = sp.build_leaf_value_matrix(tree, acc_to_row, K)
    if matched < 3:
        raise ValueError("Need >=3 shared sequences for split analysis.")

    matched_leaf_idx = int(np.argmax(valid.any(axis=1)))
    sp.reroot_at_leaf(tree, int(tree["leaf_ids"][matched_leaf_idx]))

    R, T_k, N_valid = sp.rank_per_axis(X, valid)
    sum_x_total = (X * valid).sum(axis=0)
    postorder = sp.iterative_postorder(tree)
    n_val, sum_x, sum_r = sp.accumulate_subtree_stats(tree, postorder, X, R, valid)
    cand = sp.candidate_nodes(tree)
    res = sp.evaluate_all_splits(cand, n_val, sum_x, sum_r,
                                 N_valid, sum_x_total, T_k, min_side_size)
    p_mwu = res["p_mwu"]                                   # (m, K)
    q_bh = sp.bh_fdr(p_mwu.ravel()).reshape(p_mwu.shape)

    rows = []
    for k in range(K):
        col = p_mwu[:, k]
        if not np.any(np.isfinite(col)):
            print(f"    {comp_names[k]}: no valid split", file=sys.stderr)
            continue
        idx = int(np.nanargmin(col))
        node_id = int(cand[idx])
        nA = int(res["nA"][idx, k]); nB = int(res["nB"][idx, k])
        mA = float(res["mean_A"][idx, k]); mB = float(res["mean_B"][idx, k])
        U = float(res["U_A"][idx, k]); Z = float(res["Z"][idx, k])
        cliffs_delta_A = 2.0 * U / (nA * nB) - 1.0
        under = set(sp.descendants_of(tree, node_id))
        complement = [l for l in (int(x) for x in tree["leaf_ids"]) if l not in under]
        if nA <= nB:
            clade_leaves, clade_size, comp_size = list(under), nA, nB
            mean_clade, mean_comp, cliffs = mA, mB, cliffs_delta_A
        else:
            clade_leaves, clade_size, comp_size = complement, nB, nA
            mean_clade, mean_comp, cliffs = mB, mA, -cliffs_delta_A
        rows.append({
            "component": comp_names[k],
            "bipartition_id": sp.bipartition_id(tree, clade_leaves),
            "clade_size": clade_size,
            "complement_size": comp_size,
            "mean_clade": mean_clade,
            "mean_complement": mean_comp,
            # Sign of Cliff's delta only reflects the arbitrary clade vs
            # complement orientation, so report magnitude (mean_clade /
            # mean_complement still carry the direction if needed).
            "abs_cliffs_delta": abs(cliffs),
            "Z": Z,
            "p_mwu": float(res["p_mwu"][idx, k]),
            "q_bh": float(q_bh[idx, k]),
            "rep_clade": ";".join(sp.sample_accessions(tree, clade_leaves, 5, rng)),
        })
        print(f"    {comp_names[k]}: clade={clade_size:5d}/{comp_size:<5d} "
              f"|delta|={abs(cliffs):.3f} p={res['p_mwu'][idx, k]:.3e}",
              file=sys.stderr)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="./PF00001",
                   help="SCA results folder containing preprocessing/ and scacore/.")
    p.add_argument("--n-subsample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gt-method", default="nj", choices=["sl", "upgma", "nj"],
                   help="FAMSA guide-tree method (default nj).")
    p.add_argument("--famsa-bin", default=None,
                   help="Path to the FAMSA binary (auto-detected if omitted).")
    p.add_argument("--threads", type=int, default=0,
                   help="FAMSA threads (0 = half the logical cores).")
    p.add_argument("--min-side-size", type=int, default=10,
                   help="Min valid leaves on each side of a candidate split.")
    p.add_argument("--no-transform", action="store_true",
                   help="Skip the van der Waerden transform for Pagel's lambda.")
    p.add_argument("--null-projections", type=int, default=200, metavar="N",
                   help="Random alignment projections used to calibrate each "
                        "IC's lambda against the generic phylogenetic "
                        "autocorrelation of any sequence projection "
                        "(default 200; 0 disables).")
    p.add_argument("--tmp-dir", default="./tmp",
                   help="Scratch/output root (created in the cwd).")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    results_dir = os.path.abspath(args.results_dir)
    preprocess_dir = os.path.join(results_dir, "preprocessing")
    sca_dir = os.path.join(results_dir, "scacore")
    family = os.path.basename(results_dir.rstrip("/"))
    out_dir = os.path.join(os.path.abspath(args.tmp_dir), family)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Family:        {family}", file=sys.stderr)
    print(f"Results dir:   {results_dir}  (read-only)", file=sys.stderr)
    print(f"Output dir:    {out_dir}", file=sys.stderr)

    # --- load + subsample ------------------------------------------------
    msa, seq_ids, int2char = load_processed_msa(preprocess_dir)
    gap_idx = int(np.where(int2char == "-")[0][0])
    nonempty = (msa != gap_idx).any(axis=1)        # drop any all-gap rows
    keep = np.where(nonempty)[0]
    print(f"Sequences:     {msa.shape[0]} total, {len(keep)} non-empty, "
          f"L={msa.shape[1]}", file=sys.stderr)

    sel = keep[subsample_indices(len(keep), args.n_subsample, rng)]
    msa_sub = msa[sel]
    ids_sub = seq_ids[sel]
    labels = [f"s{i}" for i in range(len(sel))]
    print(f"Subsampled:    {len(sel)} sequences (seed={args.seed})",
          file=sys.stderr)

    # --- project onto ICs ------------------------------------------------
    print("Projecting subsample onto IC sequence space (Uᵖ)...", file=sys.stderr)
    from mysca.results import SCAResults
    sca = SCAResults.load(sca_dir)
    up = project_subsample(sca, msa_sub)               # (n_sub, K)
    K = up.shape[1]
    axis_cols = [f"Up_{k}" for k in range(K)]
    comp_names = [f"{family}_{c}" for c in axis_cols]  # e.g. PF00001_Up_0
    acc_to_row = {lab: up[i] for i, lab in enumerate(labels)}
    print(f"  {K} ICs projected.", file=sys.stderr)

    # Per-IC SCA quantities: component magnitude (lambda = i-th SCA eigenvalue)
    # and the position-wise correlation of the IC loading with conservation.
    conservation = np.asarray(sca.conservation, dtype=np.float64)
    v_ica = np.asarray(sca.v_ica, dtype=np.float64)        # (L, K) loadings
    evals_sca = np.asarray(sca.evals_sca, dtype=np.float64)
    ic_rows = []
    for k in range(K):
        r, p = stats.pearsonr(v_ica[:, k], conservation)
        ic_rows.append({"component": comp_names[k],
                        "lambda": float(evals_sca[k]),
                        "cons_corr_r": float(r), "cons_corr_p": float(p)})
    ic_df = pd.DataFrame(ic_rows)
    # Benjamini-Hochberg FDR across this family's K ICs (within-family
    # multiple-testing correction; matches notebook cell 3).
    ic_df["cons_corr_p_adj"] = stats.false_discovery_control(
        ic_df["cons_corr_p"].to_numpy())

    # Persist the label<->id<->Up mapping for traceability.
    proj_df = pd.DataFrame({"label": labels, "seq_id": ids_sub})
    for k in range(K):
        proj_df[axis_cols[k]] = up[:, k]
    proj_df.to_csv(os.path.join(out_dir, "subsample_projections.tsv"),
                   sep="\t", index=False, float_format="%.6g")

    # --- guide tree ------------------------------------------------------
    fasta_path = os.path.join(out_dir, "subsample.fasta")
    tree_path = os.path.join(out_dir, "guide_tree.nwk")
    write_fasta(fasta_path, labels, msa_sub, int2char)
    print(f"Building FAMSA guide tree ({args.gt_method})...", file=sys.stderr)
    famsa_bin = resolve_famsa(args.famsa_bin)
    famsa_guide_tree(famsa_bin, fasta_path, tree_path, args.gt_method, args.threads)
    tree_text = open(tree_path).read()

    # Gapped, processed-coordinate aligned sequences (leaf label -> string)
    # feed the random-projection null in run_pagel.
    aligned_seqs = {lab: "".join(int2char[v] for v in row)
                    for lab, row in zip(labels, msa_sub)}

    # --- per-IC analyses -------------------------------------------------
    print("\nPagel's lambda per IC:", file=sys.stderr)
    pagel_df = run_pagel(tree_text, acc_to_row, comp_names,
                         transform=not args.no_transform,
                         aligned_seqs=aligned_seqs,
                         n_null=args.null_projections, seed=args.seed)
    # Prepend the SCA magnitude + conservation-correlation columns.
    pagel_df = ic_df.merge(pagel_df, on="component", how="right")
    print("\nBest clade split per IC:", file=sys.stderr)
    splits_df = run_splits(tree_text, acc_to_row, comp_names,
                           args.min_side_size, args.seed)

    # --- write + report --------------------------------------------------
    pagel_path = os.path.join(out_dir, "pagel_lambda_results.tsv")
    splits_path = os.path.join(out_dir, "best_split_per_ic.tsv")
    pagel_df.to_csv(pagel_path, sep="\t", index=False, float_format="%.6g")
    splits_df.to_csv(splits_path, sep="\t", index=False, float_format="%.6g")

    combined = pagel_df.merge(
        splits_df[["component", "clade_size", "abs_cliffs_delta", "p_mwu", "q_bh"]],
        on="component", how="left", suffixes=("_pagel", "_split"))

    # Focused console view; full detail is in the TSVs.
    show = ["component", "lambda", "cons_corr_r", "cons_corr_p_adj",
            "lamP", "lrt_z", "p_lrt_vs_null_BH",
            "clade_size", "abs_cliffs_delta", "q_bh"]
    show = [c for c in show if c in combined.columns]

    print("\n" + "=" * 78)
    print(f"{family}: per-IC phylogenetic signal vs. null and best clade split "
          f"(n={int(pagel_df['n'].iloc[0])})")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 200, "display.max_columns", None):
        print(combined[show].to_string(index=False))
    print(f"\nWrote {pagel_path}")
    print(f"Wrote {splits_path}")
    print(f"Wrote {os.path.join(out_dir, 'subsample_projections.tsv')}")
    print(f"Wrote {tree_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
