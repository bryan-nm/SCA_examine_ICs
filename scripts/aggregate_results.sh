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

[ -d "$OUTDIR" ] || { echo "OUTDIR not found: $OUTDIR" >&2; exit 1; }
echo "aggregating per-family results under $OUTDIR -> $AGG" >&2

"$PYTHON" - "$OUTDIR" "$AGG" <<'PY'
import glob, os, sys
import pandas as pd

outdir, agg_path = sys.argv[1], sys.argv[2]

# Recursive so it works regardless of how deep <family>/ sits under OUTDIR.
pagels = sorted(glob.glob(os.path.join(outdir, "**", "pagel_lambda_results.tsv"),
                          recursive=True))
print(f"found {len(pagels)} family pagel tables", file=sys.stderr)

frames, ok, skipped, no_split = [], 0, 0, 0
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
      f"({skipped} skipped, {no_split} pagel-only/no clade splits, "
      f"{master['family'].nunique()} unique families)", file=sys.stderr)
PY
