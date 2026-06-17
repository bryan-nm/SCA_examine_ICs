#!/usr/bin/env bash
# merge_sectors.sh — like aggregate_results.sh, but instead of emitting one row
# per IC it first CATEGORISES, DISCARDS, and MERGES the per-family ICs into
# higher-level "features" (sectors, co-sectors, pseudo-sectors) and then writes
# one row per feature. The pipeline per family is:
#
#   0. pre-filter      : drop families with n_seqs < 50 or align_length < 50.
#   1. conservation    : cons_corr_r > 0.5  -> tag "conservation-associated" and
#                        ALWAYS keep (preempts the masking in steps 2-3).
#   2. pseudo (subclade): abs_cliffs_delta > 0.95 -> "pseudo-sector: subclade
#                        origin", masked from the merge graph.
#   3. pseudo (phylo)  : lrt_z > 7 -> "pseudo-sector: phylogeny origin", masked.
#   4. merge           : among unmasked ICs, if IC_x and IC_y are RECIPROCALLY
#                        closer-than-self (cross_dist < within_dist both ways),
#                        union them; transitive closure -> clusters -> sectors.
#   5. co-sectors      : if IC_x reaches IC_y (cross<within) but NOT vice versa,
#                        IC_y stays a sector and IC_x is tagged a "co-sector" of
#                        IC_y's sector.
#   6. sectors         : every unmasked IC not made a co-sector — singletons and
#                        merged clusters alike — is a "sector".
#   7. report          : one big table, one row per feature.
#
#   bash merge_sectors.sh [OUTDIR] [OUTFILE]
#
# OUTDIR defaults to $OUTDIR env or ./pfam_run/results. OUTFILE defaults to
# <OUTDIR>/all_families_sector_features.tsv.
#
# Thresholds are env-overridable (see below). Robust by design: each family is
# read in a try/except so one empty/half-written TSV is skipped with a warning
# rather than aborting the whole run.
set -uo pipefail

OUTDIR="${1:-${OUTDIR:-$PWD/pfam_run/results}}"
OUT="${2:-${OUT:-$OUTDIR/all_families_sector_features.tsv}}"
PYTHON="${PYTHON:-python}"
LOGDIR="${LOGDIR:-$(dirname "$OUTDIR")/logs}"   # per-family .err logs; backfills align_length / n_seqs

# --- categorisation thresholds (override via env) ----------------------------
MIN_SEQS="${MIN_SEQS:-50}"            # step 0: minimum sequences in the family
MIN_ALIGN="${MIN_ALIGN:-50}"         # step 0: minimum processed alignment length
CONS_R="${CONS_R:-0.5}"              # step 1: cons_corr_r above this -> keep (conservation)
CLIFFS="${CLIFFS:-0.95}"            # step 2: abs_cliffs_delta above this -> subclade pseudo-sector
LRT_Z="${LRT_Z:-7.0}"               # step 3: lrt_z above this -> phylogeny pseudo-sector

[ -d "$OUTDIR" ] || { echo "OUTDIR not found: $OUTDIR" >&2; exit 1; }
echo "merging ICs into sector features under $OUTDIR -> $OUT" >&2

"$PYTHON" - "$OUTDIR" "$OUT" "$LOGDIR" \
           "$MIN_SEQS" "$MIN_ALIGN" "$CONS_R" "$CLIFFS" "$LRT_Z" <<'PY'
import glob, os, re, sys
import numpy as np
import pandas as pd

(outdir, out_path, logdir,
 min_seqs, min_align, cons_r_thr, cliffs_thr, lrt_z_thr) = (
    sys.argv[1], sys.argv[2], sys.argv[3],
    int(sys.argv[4]), int(sys.argv[5]),
    float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))

# "Sequences:   232620 total, 232620 non-empty, L=230" (SCA_analysis stderr) —
# lets runs whose pagel TSV predates the n_seqs/align_length columns still be
# pre-filtered (step 0) without re-running.
_SEQ_RE = re.compile(r"Sequences:\s+(\d+)\s+total,\s+\d+\s+non-empty,\s+L=(\d+)")

def seq_line_fields(fam):
    for ext in (".err", ".log"):
        path = os.path.join(logdir, fam + ext)
        if not os.path.exists(path):
            continue
        try:
            with open(path, errors="replace") as fh:
                for line in fh:
                    m = _SEQ_RE.search(line)
                    if m:
                        return int(m.group(1)), int(m.group(2))   # n_seqs, L
        except OSError:
            pass
    return None, None

def comp_index(component):
    """Trailing integer of '<fam>_Up_<k>' -> k, the row/col index used by the
    structure neighbour table (IC<k>) and the K x K cross-distance matrix."""
    m = re.search(r"_Up_(\d+)$", str(component))
    return int(m.group(1)) if m else None


class UnionFind:
    """Plain union-find for the reciprocal-merge transitive closure (step 4)."""
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path-halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def process_family(fam, fam_dir):
    """Return (feature_rows, status) for one family. status is a short reason
    string when the family yields no features (pre-filtered / unreadable)."""
    pagel = os.path.join(fam_dir, "pagel_lambda_results.tsv")
    df = pd.read_csv(pagel, sep="\t")
    if df.empty or "component" not in df.columns:
        raise ValueError("empty/malformed pagel table")
    df["component"] = df["component"].astype(str)

    # kstar = number of significant ICs SCA_analysis kept (one pagel row per IC).
    kstar = len(df)

    # --- step 0: pre-filter on family size -----------------------------------
    # Prefer the per-IC columns (constant within a family); backfill from the
    # run log for older TSVs. If size stays unknown we keep the family but warn,
    # rather than silently discarding it.
    n_seqs = align_length = None
    if "n_seqs" in df.columns and df["n_seqs"].notna().any():
        n_seqs = int(df["n_seqs"].dropna().iloc[0])
    if "align_length" in df.columns and df["align_length"].notna().any():
        align_length = int(df["align_length"].dropna().iloc[0])
    if n_seqs is None or align_length is None:
        ns, L = seq_line_fields(fam)
        n_seqs = n_seqs if n_seqs is not None else ns
        align_length = align_length if align_length is not None else L
    if n_seqs is not None and n_seqs < min_seqs:
        return [], f"pre-filtered: n_seqs={n_seqs} < {min_seqs}"
    if align_length is not None and align_length < min_align:
        return [], f"pre-filtered: align_length={align_length} < {min_align}"
    if n_seqs is None or align_length is None:
        print(f"WARN {fam}: family size unknown; pre-filter not applied",
              file=sys.stderr)

    # --- bring in abs_cliffs_delta (best clade split) ------------------------
    df["abs_cliffs_delta"] = np.nan
    split = os.path.join(fam_dir, "best_split_per_ic.tsv")
    if os.path.exists(split) and os.path.getsize(split) > 1:
        try:
            sdf = pd.read_csv(split, sep="\t")
            if not sdf.empty and {"component", "abs_cliffs_delta"} <= set(sdf.columns):
                sdf["component"] = sdf["component"].astype(str)
                df = df.drop(columns=["abs_cliffs_delta"]).merge(
                    sdf[["component", "abs_cliffs_delta"]], on="component", how="left")
        except Exception as e:
            print(f"WARN {fam}: split unreadable ({e}); no abs_cliffs_delta",
                  file=sys.stderr)

    # --- within-IC mean neighbour distance (structure contiguity) ------------
    # ic_neighbor_distance.tsv keys ICs as 'IC<k>' -> map to '<fam>_Up_<k>'.
    df["within_dist"] = np.nan
    nbr = os.path.join(fam_dir, "ic_neighbor_distance.tsv")
    if os.path.exists(nbr) and os.path.getsize(nbr) > 1:
        try:
            ndf = pd.read_csv(nbr, sep="\t")
            if not ndf.empty and {"IC", "mean_nbr_dist"} <= set(ndf.columns):
                k = ndf["IC"].astype(str).str.extract(r"(\d+)")[0]
                ndf = ndf.assign(component=fam + "_Up_" + k)[
                    ["component", "mean_nbr_dist"]].rename(
                    columns={"mean_nbr_dist": "within_dist"})
                df = df.drop(columns=["within_dist"]).merge(
                    ndf, on="component", how="left")
        except Exception as e:
            print(f"WARN {fam}: neighbour table unreadable ({e})", file=sys.stderr)

    # --- cross-IC neighbour-distance matrix ----------------------------------
    # cross[i, j] = mean nearest distance to IC_i over IC_j's residues. So the
    # distance measured OVER IC_x's own residues toward IC_y is cross[y, x]
    # (column x) — same residue basis as IC_x's within_dist, hence comparable.
    cross = None
    cross_f = os.path.join(fam_dir, "ic_cross_neighbor_distance.tsv")
    if os.path.exists(cross_f) and os.path.getsize(cross_f) > 1:
        try:
            C = pd.read_csv(cross_f, sep="\t", header=None).to_numpy(dtype=float)
            if C.ndim == 2 and C.shape[0] == C.shape[1] and C.shape[0] >= 1:
                cross = C
        except Exception as e:
            print(f"WARN {fam}: cross matrix unreadable ({e})", file=sys.stderr)

    # --- per-IC categorisation (steps 1-3) -----------------------------------
    # Build per-component records carrying the flags, labels, and the within-IC
    # distance / matrix index needed by the merge graph.
    cons = df.get("cons_corr_r", pd.Series(np.nan, index=df.index))
    cliffs = df.get("abs_cliffs_delta", pd.Series(np.nan, index=df.index))
    lrtz = df.get("lrt_z", pd.Series(np.nan, index=df.index))
    nres = df.get("ic_n_residues", pd.Series(np.nan, index=df.index))

    ics = {}
    for i, row in df.iterrows():
        comp = row["component"]
        labels = []
        conservation = bool(cons.iloc[i] > cons_r_thr) if pd.notna(cons.iloc[i]) else False
        if conservation:
            labels.append("conservation-associated")
        # Masking only applies when conservation does NOT preempt it (step 1).
        masked, ptype = False, None
        if not conservation:
            if pd.notna(cliffs.iloc[i]) and cliffs.iloc[i] > cliffs_thr:
                masked, ptype = True, "subclade origin"
            elif pd.notna(lrtz.iloc[i]) and lrtz.iloc[i] > lrt_z_thr:
                masked, ptype = True, "phylogeny origin"
        if masked:
            labels.append(ptype)
        ics[comp] = {
            "component": comp,
            "idx": comp_index(comp),
            "n_positions": int(nres.iloc[i]) if pd.notna(nres.iloc[i]) else None,
            "within_dist": float(row["within_dist"]) if pd.notna(row["within_dist"]) else np.nan,
            "conservation": conservation,
            "masked": masked,
            "pseudo_type": ptype,
            "labels": labels,
        }

    unmasked = [c for c, v in ics.items() if not v["masked"]]

    # --- directed reachability among unmasked ICs ----------------------------
    # edge x -> y  iff  cross[idx_y, idx_x] < within_dist[x]
    # ("IC_y sits closer to IC_x's residues than IC_x's own internal spread").
    def cross_dist(x, y):
        ix, iy = ics[x]["idx"], ics[y]["idx"]
        if cross is None or ix is None or iy is None:
            return np.nan
        if not (0 <= iy < cross.shape[0] and 0 <= ix < cross.shape[1]):
            return np.nan
        return cross[iy, ix]

    def reaches(x, y):
        d, w = cross_dist(x, y), ics[x]["within_dist"]
        return np.isfinite(d) and np.isfinite(w) and d < w

    edges = {x: set() for x in unmasked}        # x -> {y it reaches}
    for x in unmasked:
        for y in unmasked:
            if x != y and reaches(x, y):
                edges[x].add(y)

    # --- step 4: reciprocal merge -> clusters --------------------------------
    uf = UnionFind(unmasked)
    for x in unmasked:
        for y in edges[x]:
            if x < y and x in edges[y]:           # reciprocal (x<-->y) -> union
                uf.union(x, y)
    clusters = {}
    for c in unmasked:
        clusters.setdefault(uf.find(c), []).append(c)
    cluster_of = {c: uf.find(c) for c in unmasked}

    # --- step 5: classify each unmasked IC -----------------------------------
    # In a merged cluster (>=2) -> sector member. Otherwise a lone IC that
    # reaches some OTHER IC (necessarily one-directionally, else it'd be merged)
    # is a co-sector; a lone IC that reaches nobody is a standalone sector.
    is_cosector = {}
    for c in unmasked:
        if len(clusters[cluster_of[c]]) >= 2:
            is_cosector[c] = False                # part of a merged sector
        else:
            is_cosector[c] = len(edges[c]) > 0    # reaches something one-way

    # Sectors = merged clusters (>=2) + standalone singletons (reach nobody).
    sector_clusters = [members for root, members in clusters.items()
                       if len(members) >= 2 or not is_cosector[members[0]]]
    # Deterministic order: by smallest matrix index (then name) in the cluster.
    def cluster_key(members):
        idxs = [ics[m]["idx"] for m in members if ics[m]["idx"] is not None]
        return (min(idxs) if idxs else 10**9, min(members))
    sector_clusters.sort(key=cluster_key)

    rows = []
    sector_name_of_member = {}                    # component -> sector feature name
    for n, members in enumerate(sector_clusters, start=1):
        members = sorted(members, key=lambda m: (ics[m]["idx"] if ics[m]["idx"] is not None else 10**9, m))
        fname = f"{fam}_sector_{n}"
        for m in members:
            sector_name_of_member[m] = fname
        labels = sorted({l for m in members for l in ics[m]["labels"]})
        npos = [ics[m]["n_positions"] for m in members if ics[m]["n_positions"] is not None]
        rows.append({
            "family": fam,
            "feature_name": fname,
            "feature_type": "sector",
            "n_ics": len(members),
            "n_positions": int(sum(npos)) if npos else np.nan,
            "labels": ";".join(labels),
            "associated_sector": "",
            "member_components": ",".join(members),
            "kstar": kstar,
            "n_seqs": n_seqs if n_seqs is not None else np.nan,
            "align_length": align_length if align_length is not None else np.nan,
        })

    # --- step 5b: resolve each co-sector to the sector it attaches to --------
    # Follow the strongest outgoing edge (smallest cross_dist); walk the chain
    # until it lands on a sector member (a co-sector may point at another
    # co-sector). A cycle / dead-end with no sector demotes the IC to its own
    # standalone sector so nothing is silently dropped.
    cosectors = [c for c in unmasked if is_cosector[c]]

    def best_target(x):
        cands = [(cross_dist(x, y), y) for y in edges[x] if np.isfinite(cross_dist(x, y))]
        return min(cands)[1] if cands else None

    def resolve_sector(x):
        seen, cur = set(), x
        while cur is not None and cur not in seen:
            seen.add(cur)
            tgt = best_target(cur)
            if tgt is None:
                return None
            if tgt in sector_name_of_member:
                return sector_name_of_member[tgt]
            cur = tgt                              # target is itself a co-sector
        return None

    co_n = 0
    for c in sorted(cosectors, key=lambda m: (ics[m]["idx"] if ics[m]["idx"] is not None else 10**9, m)):
        assoc = resolve_sector(c)
        if assoc is None:
            # Unresolvable chain -> treat as its own standalone sector.
            co_n_sector = sum(1 for r in rows if r["feature_type"] == "sector") + 1
            fname = f"{fam}_sector_{co_n_sector}"
            sector_name_of_member[c] = fname
            rows.append({
                "family": fam, "feature_name": fname, "feature_type": "sector",
                "n_ics": 1,
                "n_positions": ics[c]["n_positions"] if ics[c]["n_positions"] is not None else np.nan,
                "labels": ";".join(sorted(set(ics[c]["labels"]))),
                "associated_sector": "", "member_components": c, "kstar": kstar,
                "n_seqs": n_seqs if n_seqs is not None else np.nan,
                "align_length": align_length if align_length is not None else np.nan,
            })
            continue
        co_n += 1
        rows.append({
            "family": fam,
            "feature_name": f"{fam}_cosector_{co_n}",
            "feature_type": "co-sector",
            "n_ics": 1,
            "n_positions": ics[c]["n_positions"] if ics[c]["n_positions"] is not None else np.nan,
            "labels": ";".join(sorted(set(ics[c]["labels"]))),
            "associated_sector": assoc,
            "member_components": c,
            "kstar": kstar,
            "n_seqs": n_seqs if n_seqs is not None else np.nan,
            "align_length": align_length if align_length is not None else np.nan,
        })

    # --- step 2-3 output: pseudo-sectors (one feature per masked IC) ---------
    ps_n = 0
    for c, v in ics.items():
        if not v["masked"]:
            continue
        ps_n += 1
        rows.append({
            "family": fam,
            "feature_name": f"{fam}_pseudosector_{ps_n}",
            "feature_type": "pseudo-sector",
            "n_ics": 1,
            "n_positions": v["n_positions"] if v["n_positions"] is not None else np.nan,
            "labels": ";".join(sorted(set(v["labels"]))),
            "associated_sector": "",
            "member_components": c,
            "kstar": kstar,
            "n_seqs": n_seqs if n_seqs is not None else np.nan,
            "align_length": align_length if align_length is not None else np.nan,
        })

    return rows, None


# Recursive so it works regardless of how deep <family>/ sits under OUTDIR.
pagels = sorted(glob.glob(os.path.join(outdir, "**", "pagel_lambda_results.tsv"),
                          recursive=True))
print(f"found {len(pagels)} family pagel tables", file=sys.stderr)

all_rows, ok, prefiltered, skipped = [], 0, 0, 0
for pagel in pagels:
    fam_dir = os.path.dirname(pagel)
    fam = os.path.basename(fam_dir)
    try:
        rows, status = process_family(fam, fam_dir)
        if status:
            print(f"SKIP {fam}: {status}", file=sys.stderr)
            prefiltered += 1
            continue
        all_rows.extend(rows)
        ok += 1
    except Exception as e:
        print(f"WARN {fam}: skipped ({e})", file=sys.stderr)
        skipped += 1

if not all_rows:
    print("no usable features produced; nothing written", file=sys.stderr)
    sys.exit(1)

cols = ["family", "feature_name", "feature_type", "n_ics", "n_positions",
        "labels", "associated_sector", "member_components",
        "kstar", "n_seqs", "align_length"]
master = pd.DataFrame(all_rows)[cols]
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
master.to_csv(out_path, sep="\t", index=False, float_format="%.6g")

by_type = master["feature_type"].value_counts().to_dict()
print(f"wrote {out_path}: {len(master)} features from {ok} families "
      f"({prefiltered} pre-filtered; {skipped} skipped; "
      f"{master['family'].nunique()} unique families); "
      f"by type: {by_type}", file=sys.stderr)
PY
