#!/usr/bin/env bash
# aggregate_results.sh — concatenate the per-family SCA_examine_ICs results into
# one master per-IC table. Standalone re-run of the aggregation step in
# run_all_pfam.sh; safe to run any time after the analysis has produced
# <OUTDIR>/<family>/pagel_lambda_results.tsv files.
#
#   bash aggregate_results.sh [OUTDIR] [OUTFILE]
#
# OUTDIR defaults to $OUTDIR env or ./pfam_run/results (the run_all_pfam.sh
# default). OUTFILE defaults to <OUTDIR>/all_families_ic_results.tsv.
#
# Robust by design: each family is read in a try/except, so a single empty or
# half-written TSV is skipped (with a warning) instead of aborting the whole
# aggregation.
set -uo pipefail

OUTDIR="${1:-${OUTDIR:-$PWD/pfam_run/results}}"
AGG="${2:-${AGG:-$OUTDIR/all_families_ic_results.tsv}}"
PYTHON="${PYTHON:-python}"
CROSS_THRESH="${CROSS_THRESH:-5.0}"   # Å; IC pair "in contact" below this (matches structure_analysis default)

[ -d "$OUTDIR" ] || { echo "OUTDIR not found: $OUTDIR" >&2; exit 1; }
echo "aggregating per-family results under $OUTDIR -> $AGG" >&2

"$PYTHON" - "$OUTDIR" "$AGG" "$CROSS_THRESH" <<'PY'
import glob, os, sys
import numpy as np
import pandas as pd

outdir, agg_path, cross_thresh = sys.argv[1], sys.argv[2], float(sys.argv[3])

# Recursive so it works regardless of how deep <family>/ sits under OUTDIR.
pagels = sorted(glob.glob(os.path.join(outdir, "**", "pagel_lambda_results.tsv"),
                          recursive=True))
print(f"found {len(pagels)} family pagel tables", file=sys.stderr)

frames, ok, skipped, no_split, no_struct, no_cross = [], 0, 0, 0, 0, 0
for pagel in pagels:
    fam_dir = os.path.dirname(pagel)
    fam = os.path.basename(fam_dir)
    try:
        df = pd.read_csv(pagel, sep="\t")
        if df.empty:
            raise ValueError("empty pagel table")
        if "component" not in df.columns:
            raise ValueError("no 'component' column (malformed/partial table)")
        # An empty best_split_per_ic.tsv just means no IC had a valid clade
        # split (common for small families) — benign, keep pagel columns only.
        split = os.path.join(fam_dir, "best_split_per_ic.tsv")
        if os.path.exists(split) and os.path.getsize(split) > 1:
            try:
                sdf = pd.read_csv(split, sep="\t")
                if not sdf.empty and "component" in sdf.columns:
                    df = df.merge(sdf, on="component", how="left")
                else:
                    no_split += 1
            except pd.errors.EmptyDataError:
                no_split += 1
            except Exception as e:
                print(f"WARN {fam}: split unreadable ({e}); pagel columns only",
                      file=sys.stderr)
        else:
            no_split += 1

        # Structural contiguity (structure_analysis.py): per-IC mean neighbour
        # distance + size-matched-null z-score + empirical p. Its "IC<k>" key
        # maps to this family's "<fam>_Up_<k>" component. Often absent
        # (DO_STRUCTURE=0, or AlphaFold fetch failed) — benign, keep going.
        nbr = os.path.join(fam_dir, "ic_neighbor_distance.tsv")
        if os.path.exists(nbr) and os.path.getsize(nbr) > 1:
            try:
                ndf = pd.read_csv(nbr, sep="\t")
                if not ndf.empty and "IC" in ndf.columns:
                    k = ndf["IC"].astype(str).str.extract(r"(\d+)")[0]
                    ndf = ndf.assign(component=fam + "_Up_" + k)[
                        ["component", "mean_nbr_dist", "z", "p_clustered"]].rename(
                        columns={"mean_nbr_dist": "struct_mean_nbr_dist",
                                 "z": "struct_z", "p_clustered": "struct_p_clustered"})
                    df = df.merge(ndf, on="component", how="left")
                else:
                    no_struct += 1
            except pd.errors.EmptyDataError:
                no_struct += 1
            except Exception as e:
                print(f"WARN {fam}: structure unreadable ({e}); no contiguity cols",
                      file=sys.stderr)
        else:
            no_struct += 1

        # Reciprocal contacts: from the K x K cross-IC neighbour-distance matrix
        # (cross[i, j] = mean nearest-IC_i distance over IC_j's residues), IC i
        # and IC j touch reciprocally iff BOTH directions are < cross_thresh.
        # Report, per IC, the fraction of the other K-1 ICs it reciprocally
        # contacts.
        cross_f = os.path.join(fam_dir, "ic_cross_neighbor_distance.tsv")
        if os.path.exists(cross_f) and os.path.getsize(cross_f) > 1:
            try:
                C = pd.read_csv(cross_f, sep="\t", header=None).to_numpy(dtype=float)
                Kc = C.shape[0]
                if C.ndim == 2 and C.shape[1] == Kc and Kc >= 1:
                    with np.errstate(invalid="ignore"):
                        contact = C < cross_thresh          # NaN -> False
                    recip = contact & contact.T
                    np.fill_diagonal(recip, False)          # exclude self
                    frac = (recip.sum(axis=1) / (Kc - 1)
                            if Kc > 1 else np.full(Kc, np.nan))
                    df = df.merge(pd.DataFrame({
                        "component": [f"{fam}_Up_{i}" for i in range(Kc)],
                        "struct_recip_contact_frac": frac,
                    }), on="component", how="left")
                else:
                    no_cross += 1
            except Exception as e:
                print(f"WARN {fam}: cross matrix unreadable ({e})", file=sys.stderr)
                no_cross += 1
        else:
            no_cross += 1

        df.insert(0, "family", fam)
        frames.append(df)
        ok += 1
    except Exception as e:
        print(f"WARN {fam}: skipped ({e})", file=sys.stderr)
        skipped += 1

if not frames:
    print("no usable per-family results found; nothing written", file=sys.stderr)
    sys.exit(1)

# sort=False -> union of columns across families (any missing cols become NaN).
master = pd.concat(frames, ignore_index=True, sort=False)
os.makedirs(os.path.dirname(os.path.abspath(agg_path)), exist_ok=True)
master.to_csv(agg_path, sep="\t", index=False, float_format="%.6g")
print(f"wrote {agg_path}: {len(master)} IC rows from {ok} families "
      f"({skipped} skipped; {no_split} without clade splits; "
      f"{no_struct} without structure data; {no_cross} without cross-IC matrix; "
      f"{master['family'].nunique()} unique families)", file=sys.stderr)
PY
