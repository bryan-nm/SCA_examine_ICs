#!/usr/bin/env bash
# run_all_pfam.sh — run SCA_examine_ICs over every PFAM family in a directory of
# compressed mysca result tarballs.
#
# For each tarball it: extracts to scratch, finds every family results folder
# inside (a dir with preprocessing/preprocessing_results.npz + scacore/), runs
# the three analysis scripts, captures their stdout to one log per family, then
# deletes the scratch copy. Resumable (per-family + per-tarball .done markers),
# continues past per-family failures.
#
#   module load famsa mysca
#   bash run_all_pfam.sh                 # all tarballs
#   bash run_all_pfam.sh 'PF00*.tar.gz'  # a subset (shard across nodes)
#
# Override any config via env, e.g.  OUTDIR=/scratch/out DO_STRUCTURE=0 bash run_all_pfam.sh
set -uo pipefail
module load mysca famsa

# ---- config -----------------------------------------------------------------
COMPRESSED_DIR="${COMPRESSED_DIR:-/media/nm-data/data/pfam-sca/results/compressed}"
REPO="${REPO:-$HOME/code/SCA_examine_ICs}"          # clone location
SRC="$REPO/src"
WORKROOT="${WORKROOT:-$PWD/pfam_run}"
SCRATCH="${SCRATCH:-$WORKROOT/scratch}"        # tarball extraction (deleted per tarball)
OUTDIR="${OUTDIR:-$WORKROOT/results}"          # tmp/<family> outputs (tsvs, structures, plots)
LOGDIR="${LOGDIR:-$WORKROOT/logs}"             # per-family stdout/stderr + .done markers
PYTHON="${PYTHON:-python}"
N_SUBSAMPLE="${N_SUBSAMPLE:-1000}"
NULL_PROJ="${NULL_PROJ:-200}"
SEED="${SEED:-0}"
DO_STRUCTURE="${DO_STRUCTURE:-1}"              # 0 to skip AlphaFold fetch + structure step
NJOBS="${NJOBS:-12}"                           # tarballs processed in parallel
FAMSA_THREADS="${FAMSA_THREADS:-1}"            # FAMSA threads per family (keep low under parallelism)
TARBALL_GLOB="${1:-*.tar.gz}"

mkdir -p "$SCRATCH" "$OUTDIR" "$LOGDIR"
log() { echo "[$(date '+%F %T')] $*" >&2; }

# Pin per-process linear algebra to a single thread so the NJOBS tarball workers
# map to NJOBS cores instead of each spawning a BLAS/OpenMP thread pool.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---- per-family pipeline ----------------------------------------------------
run_family() {
  local famdir="$1" base="${2:-}"
  local fam; fam="$(basename "$famdir")"
  local done_marker="$LOGDIR/$fam.done"
  [ -f "$done_marker" ] && { log "  [$base] skip $fam (done)"; return; }

  local out="$LOGDIR/$fam.log" err="$LOGDIR/$fam.err"
  : > "$out"; : > "$err"
  log "  [$base] family $fam"

  echo "### family=$fam dir=$famdir ###"      >> "$out"
  echo "### SCA_analysis ###"                  >> "$out"
  "$PYTHON" "$SRC/SCA_analysis.py" --results-dir "$famdir" --tmp-dir "$OUTDIR" \
      --n-subsample "$N_SUBSAMPLE" --null-projections "$NULL_PROJ" --seed "$SEED" \
      --threads "$FAMSA_THREADS" \
      >> "$out" 2>> "$err" || log "    SCA_analysis FAILED ($fam) — see $err"

  if [ "$DO_STRUCTURE" = "1" ]; then
    echo "### fetch_structure ###"            >> "$out"
    "$PYTHON" "$SRC/fetch_structure.py" --results-dir "$famdir" --tmp-dir "$OUTDIR" \
        --seed "$SEED" >> "$out" 2>> "$err" || log "    fetch_structure FAILED ($fam)"
    echo "### structure_analysis ###"         >> "$out"
    "$PYTHON" "$SRC/structure_analysis.py" --results-dir "$famdir" --tmp-dir "$OUTDIR" \
        --seed "$SEED" >> "$out" 2>> "$err" || log "    structure_analysis FAILED ($fam)"
  fi

  touch "$done_marker"
}

# ---- per-tarball worker (extract -> all families -> clean up) ---------------
process_tarball() {
  local tb="$1"
  local base; base="$(basename "$tb" .tar.gz)"
  local tb_done="$LOGDIR/$base.tarball.done"
  [ -f "$tb_done" ] && { log "skip tarball $base (done)"; return; }

  local workdir="$SCRATCH/$base"
  log "tarball $base — extracting to $workdir"
  rm -rf "$workdir"; mkdir -p "$workdir"
  if ! tar -xzf "$tb" -C "$workdir"; then
    log "  ERROR extracting $tb; skipping"; rm -rf "$workdir"; return
  fi

  local prep famdir
  while IFS= read -r prep; do
    famdir="$(dirname "$(dirname "$prep")")"
    [ -d "$famdir/scacore" ] || { log "  [$base] ($(basename "$famdir")) no scacore/, skipping"; continue; }
    run_family "$famdir" "$base"
  done < <(find "$workdir" -type f -path '*/preprocessing/preprocessing_results.npz' | sort)

  log "  finished $base — cleaning scratch"
  rm -rf "$workdir"
  touch "$tb_done"
}

# ---- dispatch: up to NJOBS tarballs in parallel -----------------------------
shopt -s nullglob
tarballs=( "$COMPRESSED_DIR"/$TARBALL_GLOB )
log "found ${#tarballs[@]} tarball(s) in $COMPRESSED_DIR matching '$TARBALL_GLOB'; "\
"running up to $NJOBS in parallel"

# Portable pool (no `wait -n`, which needs bash >=4.3): keep at most NJOBS
# workers alive; when full, block on the OLDEST one before launching the next.
pids=()
for tb in "${tarballs[@]}"; do
  base="$(basename "$tb" .tar.gz)"
  [ -f "$LOGDIR/$base.tarball.done" ] && { log "skip tarball $base (done)"; continue; }
  process_tarball "$tb" &
  pids+=("$!")
  if (( ${#pids[@]} >= NJOBS )); then
    wait "${pids[0]}"            # block on oldest worker, then free its slot
    pids=("${pids[@]:1}")
  fi
done
wait                             # drain the remaining workers

# ---- aggregate one master per-IC table across all families -----------------
# Merges each family's pagel + split TSVs (one row per IC) and concatenates.
# Idempotent: rebuilt from whatever per-family results currently exist, so a
# resumed/partial run still yields an up-to-date table.
AGG="${AGG:-$OUTDIR/all_families_ic_results.tsv}"
log "aggregating per-family results -> $AGG"
"$PYTHON" - "$OUTDIR" "$AGG" <<'PY'
import glob, os, sys
import pandas as pd

outdir, agg_path = sys.argv[1], sys.argv[2]
frames = []
for pagel in sorted(glob.glob(os.path.join(outdir, "*", "pagel_lambda_results.tsv"))):
    fam_dir = os.path.dirname(pagel)
    df = pd.read_csv(pagel, sep="\t")
    split = os.path.join(fam_dir, "best_split_per_ic.tsv")
    if os.path.exists(split):
        df = df.merge(pd.read_csv(split, sep="\t"), on="component", how="left")
    df.insert(0, "family", os.path.basename(fam_dir))
    frames.append(df)

if not frames:
    print("no per-family results found; nothing to aggregate", file=sys.stderr)
    sys.exit(0)
master = pd.concat(frames, ignore_index=True)
master.to_csv(agg_path, sep="\t", index=False, float_format="%.6g")
print(f"wrote {agg_path}: {len(master)} IC rows from {len(frames)} families",
      file=sys.stderr)
PY
log "ALL DONE"
