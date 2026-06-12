"""structure_analysis.py

Spatial-clustering analysis of SCA independent components (ICs) on a downloaded
AlphaFold structure, replicating the wild-type analysis in AR_SCA_analysis.ipynb
for an arbitrary family + randomly chosen reference sequence.

Prereq: run ``fetch_structure.py`` first to download a PDB + PAE for a random
sequence and write ``tmp/<family>/structure_source.json``.

Pipeline
--------
  1. Build the residue-residue minimum heavy-atom distance matrix from the PDB
     (diagonal = NaN, so a residue is never its own nearest neighbour).
  2. Mask it by the AlphaFold predicted aligned error (PAE): a residue pair is
     usable only where PAE < max_err (default 8 A; the notebook used 5 A, but
     that leaves the masked matrix quite sparse).
  3. Map each IC's processed-MSA positions onto the reference sequence's
     structure residues (``ic_residues``). PF00001's SCA run stored no
     per-sequence IC residues (no reference was set), so we recompute them the
     same way mysca's projection does, then shift by the domain's start offset
     in the full-length AlphaFold model.
  4. Per IC: mean nearest-neighbour distance among its residues, compared to a
     size-matched null drawn from the pool of all IC positions -> z-score and
     empirical p (small distance = spatially clustered).
  5. Cross-IC: mean nearest-neighbour distance from every IC to every other IC;
     report IC pairs whose mean cross-neighbour distance < cross_thresh (5 A),
     flagged reciprocal (both directions) or one-sided (one direction).

Nothing is written into the SCA results folder; outputs go to tmp/<family>/.
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist


_GAPS = set("-.")


# ---------------------------------------------------------------------------
# Structure: min heavy-atom distance matrix
# ---------------------------------------------------------------------------

def load_pdb_residues(pdb_path):
    """Return (sequence str, residue_ids list, list of per-residue atom-coord
    arrays) for the modelled chain, in structure order. Sequence/residue_ids
    come from mysca's PDBStructure (the same ordering used to map IC residues);
    coordinates are pulled by residue id so they stay aligned. AlphaFold models
    are single-chain, all-heavy-atom (no hydrogens)."""
    from mysca.structure.pdb import PDBStructure
    pdb = PDBStructure.from_file(pdb_path)
    struct = PDBParser(QUIET=True).get_structure("m", pdb_path)
    chain = next(iter(next(iter(struct))))           # first chain of first model
    coord_by_resid = {
        int(res.id[1]): np.array([a.coord for a in res.get_atoms()], dtype=np.float64)
        for res in chain if res.id[0] == " "
    }
    coords = [coord_by_resid[r] for r in pdb.residue_ids]
    return pdb.sequence, list(pdb.residue_ids), coords


def min_atom_distance_matrix(coords):
    """Symmetric (N, N) matrix of minimum inter-residue heavy-atom distances,
    NaN on the diagonal (a residue is not its own neighbour)."""
    n = len(coords)
    D = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        ci = coords[i]
        for j in range(i + 1, n):
            d = float(cdist(ci, coords[j]).min())
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# IC residues mapped onto the structure
# ---------------------------------------------------------------------------

def reference_ic_residues(results_dir, ref_seq_id, groups, pdb_seq):
    """Per-IC arrays of 0-based structure residue indices for the reference
    sequence.

    Mirrors mysca's in-sample projection: count raw residues along the
    reference's ORIGINAL-MSA aligned row, restrict to retained (processed)
    columns, take each IC's columns, then shift by the domain's offset within
    the full-length AlphaFold model.
    """
    preprocess_dir = os.path.join(results_dir, "preprocessing")
    ali = np.load(os.path.join(preprocess_dir, "preprocessing_results.npz"),
                  allow_pickle=True)
    retained_positions = np.asarray(ali["retained_positions"], dtype=int)

    idx = SeqIO.index(os.path.join(preprocess_dir, "msa_orig.fasta-aln"), "fasta")
    aligned = str(idx[ref_seq_id].seq)
    is_res = np.array([c not in _GAPS for c in aligned])
    resi_by_orig = np.cumsum(is_res) - 1             # 0-based residue idx per col
    resi_by_orig[~is_res] = -1                       # gaps -> -1
    resi_by_proc = resi_by_orig[retained_positions]

    raw = "".join(c for c in aligned if c not in _GAPS)   # ungapped domain
    offset = pdb_seq.find(raw)
    if offset < 0:
        raise ValueError(
            "Reference domain sequence is not a substring of the PDB sequence; "
            "cannot map IC residues onto the structure.")

    ic_residues = []
    for g in groups:
        g = np.asarray(g, dtype=int)
        resi = resi_by_proc[g]
        resi = resi[resi >= 0]
        struct_idx = offset + resi
        # sanity: structure residue must match the domain residue identity
        bad = [k for k in struct_idx if pdb_seq[k] != raw[k - offset]]
        if bad:
            raise AssertionError("IC residue identity mismatch vs structure.")
        ic_residues.append(np.sort(struct_idx))
    return ic_residues, offset


# ---------------------------------------------------------------------------
# Neighbour-distance kernels (from the notebook)
# ---------------------------------------------------------------------------

def make_dist_list(set_pos, dist_mat, ali_err, max_err=5.0):
    """For each position in ``set_pos``, distance to its closest neighbour
    within the set (PAE-masked, symmetrised). NaN where no usable neighbour."""
    masked = np.where(ali_err < max_err, dist_mat, np.nan)
    sub = masked[:, set_pos][set_pos, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sub = np.nanmean(np.array([sub, sub.T]), axis=0)     # symmetrise
        return np.nanmin(sub, axis=0)


def make_dist_list_compare(set1, set2, dist_mat, ali_err, max_err=5.0):
    """For each position in ``set2``, distance to its closest neighbour in
    ``set1`` (PAE-masked). Directional: set1 is the target, set2 the query."""
    masked = np.where(ali_err < max_err, dist_mat, np.nan)
    sub = masked[:, set1][set2, :]                            # (len2, len1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(sub, axis=1)


def mean_neighbor_distance(set_pos, dist_mat, ali_err, max_err=5.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(make_dist_list(set_pos, dist_mat, ali_err, max_err)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="./PF00001")
    p.add_argument("--tmp-dir", default="./tmp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-err", type=float, default=8.0,
                   help="PAE mask: keep residue pairs with PAE < this (A).")
    p.add_argument("--cross-thresh", type=float, default=5.0,
                   help="Mean cross-neighbour distance below this flags an IC "
                        "pair as touching (A).")
    p.add_argument("--n-null", type=int, default=1000,
                   help="Resamples per IC for the size-matched null.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results_dir = os.path.abspath(args.results_dir)
    family = os.path.basename(results_dir.rstrip("/"))
    out_dir = os.path.join(os.path.abspath(args.tmp_dir), family)

    src_path = os.path.join(out_dir, "structure_source.json")
    if not os.path.exists(src_path):
        sys.exit(f"No structure_source.json in {out_dir}; run fetch_structure.py first.")
    meta = json.load(open(src_path))
    ref_id = meta["seq_id"]
    print(f"Family:    {family}", file=sys.stderr)
    print(f"Reference: {ref_id} ({meta['entryId']})", file=sys.stderr)

    # --- distance matrix + PAE ------------------------------------------
    print("Building min heavy-atom distance matrix from PDB...", file=sys.stderr)
    pdb_seq, _resids, coords = load_pdb_residues(meta["pdb_path"])
    dist_mat = min_atom_distance_matrix(coords)
    N = len(pdb_seq)

    pae = json.load(open(meta["pae_path"]))
    pae = pae[0] if isinstance(pae, list) else pae
    ali_err = np.array(pae["predicted_aligned_error"], dtype=np.float64)
    if ali_err.shape != (N, N):
        sys.exit(f"PAE shape {ali_err.shape} != structure length ({N}).")

    masked_dist = np.where(ali_err < args.max_err, dist_mat, np.nan)
    np.save(os.path.join(out_dir, "masked_dist_mat.npy"), masked_dist)

    # --- IC residues on the structure -----------------------------------
    from mysca.results import SCAResults
    sca = SCAResults.load(os.path.join(results_dir, "scacore"))
    # Restrict to the kstar significant ICs (first kstar by descending SCA
    # eigenvalue) so the per-IC contiguity rows and the cross-IC reciprocal-
    # contact fraction (computed over the other K-1 ICs) both count only
    # significant ICs — matching SCA_analysis.py's Up_0..Up_{kstar-1}.
    groups = sca.ic_positions
    kstar = int(sca.kstar) if sca.kstar is not None else len(groups)
    kstar = max(1, min(kstar, len(groups)))
    groups = groups[:kstar]
    ic_residues, offset = reference_ic_residues(results_dir, ref_id, groups, pdb_seq)
    K = len(ic_residues)
    sizes = [len(p) for p in ic_residues]
    pool = np.unique(np.concatenate(ic_residues))     # all-IC positions
    print(f"ICs: {K}  residues/IC: {sizes}  union pool: {len(pool)}  "
          f"(domain offset {offset})", file=sys.stderr)

    # --- per-IC mean neighbour distance vs size-matched null ------------
    rows = []
    for k in range(K):
        obs = mean_neighbor_distance(ic_residues[k], dist_mat, ali_err, args.max_err)
        nk = sizes[k]
        null = np.empty(args.n_null, dtype=np.float64)
        for b in range(args.n_null):
            samp = rng.choice(pool, size=nk, replace=False)
            null[b] = mean_neighbor_distance(samp, dist_mat, ali_err, args.max_err)
        nm, ns = float(np.nanmean(null)), float(np.nanstd(null, ddof=1))
        z = (obs - nm) / ns if ns > 0 else np.nan
        p_emp = (1 + int(np.sum(null <= obs))) / (args.n_null + 1)   # clustered = small
        rows.append({"IC": f"IC{k}", "n_res": nk, "mean_nbr_dist": obs,
                     "null_mean": nm, "null_sd": ns, "z": z, "p_clustered": p_emp})
        print(f"  IC{k}: n={nk:2d} obs={obs:5.2f} null={nm:5.2f}+/-{ns:4.2f} "
              f"z={z:+.2f} p={p_emp:.3f}", file=sys.stderr)

    import pandas as pd
    nbr_df = pd.DataFrame(rows)
    nbr_df.to_csv(os.path.join(out_dir, "ic_neighbor_distance.tsv"),
                  sep="\t", index=False, float_format="%.4g")

    # --- cross-IC neighbour distances -----------------------------------
    cross = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            l = make_dist_list_compare(ic_residues[i], ic_residues[j],
                                       dist_mat, ali_err, args.max_err)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                cross[i, j] = np.nanmean(l)        # mean over IC_j of nearest IC_i
    np.savetxt(os.path.join(out_dir, "ic_cross_neighbor_distance.tsv"),
               cross, delimiter="\t", fmt="%.4g")

    # pairs that touch: cross[i,j] = how close IC_j sits to IC_i
    touching = []
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = cross[i, j]      # IC_j -> IC_i
            d_ji = cross[j, i]      # IC_i -> IC_j
            below = [d < args.cross_thresh for d in (d_ij, d_ji)]
            if any(below):
                kind = "reciprocal" if all(below) else "one-sided"
                touching.append((f"IC{i}", f"IC{j}", d_ij, d_ji, kind))

    # --- plots (mirror the notebook outputs) ----------------------------
    _plot_masked(masked_dist, os.path.join(out_dir, "masked_dist_mat.png"))
    _plot_cross(cross, K, os.path.join(out_dir, "IC_cross_dist_mat.png"))

    # mask density + overall clustering of all IC positions (sparsity check)
    usable = np.isfinite(np.where(ali_err < args.max_err, dist_mat, np.nan))
    frac_usable = usable.sum() / (N * (N - 1))
    pool_mnd = mean_neighbor_distance(pool, dist_mat, ali_err, args.max_err)

    # --- report ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{family} / {ref_id}: IC spatial clustering (PAE<{args.max_err} A)")
    print("=" * 70)
    print(f"Masked matrix density: {frac_usable:.1%} of residue pairs usable")
    print("\nMean neighbor distance (A):")
    for r in rows:
        print(f"  {r['IC']}: {r['mean_nbr_dist']:.2f}  (z={r['z']:+.2f}, "
              f"p_clustered={r['p_clustered']:.3f})")
    print(f"  all IC positions (pool, n={len(pool)}): {pool_mnd:.2f}")
    print()
    with pd.option_context("display.float_format", "{:.3g}".format):
        print(nbr_df.to_string(index=False))

    print(f"\nCross-IC pairs with mean cross-neighbour distance "
          f"< {args.cross_thresh} A:")
    if not touching:
        print("  (none)")
    for a, b, d_ab, d_ba, kind in sorted(touching, key=lambda t: min(t[2], t[3])):
        print(f"  {a}-{b}: {a}<-{b}={d_ab:.2f}  {b}<-{a}={d_ba:.2f}  [{kind}]")

    print(f"\nWrote {os.path.join(out_dir, 'ic_neighbor_distance.tsv')}")
    print(f"Wrote {os.path.join(out_dir, 'ic_cross_neighbor_distance.tsv')}")
    print(f"Wrote {os.path.join(out_dir, 'masked_dist_mat.npy')} (+ .png, cross .png)")


def _plot_masked(masked_dist, path):
    plt.figure(figsize=(3, 3))
    plt.imshow(masked_dist)
    plt.xlabel("position 1"); plt.ylabel("position 2")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def _plot_cross(cross, K, path):
    plt.figure(figsize=(3, 3))
    plt.imshow(cross, cmap="gray")
    plt.xticks(range(K), [f"IC{k}" for k in range(K)], fontsize=6, rotation=90)
    plt.yticks(range(K), [f"IC{k}" for k in range(K)], fontsize=6)
    plt.colorbar(label="mean cross-neighbour dist (A)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
