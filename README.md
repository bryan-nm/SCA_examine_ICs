# SCA_examine_ICs

Tools for examining the **independent components (ICs)** produced by a
[Statistical Coupling Analysis](https://en.wikipedia.org/wiki/Statistical_coupling_analysis)
run on a protein family. Given a finished SCA results folder, these scripts ask
two complementary questions about each IC:

1. **Is the IC just phylogeny?** — Does a sequence's projection onto the IC
   track the family tree (Pagel's lambda), and does it do so any more than a
   generic random projection of the alignment?
2. **Is the IC a structural unit?** — Do the IC's positions cluster in 3D on an
   AlphaFold model, and which ICs sit next to each other?

All scripts are read-only on the input SCA folder and write everything to
`tmp/<family>/` in the current working directory.

## Pipeline

| Script | What it does | Key outputs (in `tmp/<family>/`) |
|--------|--------------|-----------------------------------|
| `src/SCA_analysis.py` | Subsample the processed alignment, build a FAMSA guide tree, and for every IC fit **Pagel's lambda** (phylogenetic signal) + an **LRT-vs-random-projection null**, and find the **best clade split** (Mann-Whitney U + Cliff's delta). | `pagel_lambda_results.tsv`, `best_split_per_ic.tsv`, `subsample_projections.tsv`, `guide_tree.nwk` |
| `src/fetch_structure.py` | Pick a random sequence, resolve its UniProt accession, and download the **AlphaFold** PDB + predicted aligned error (PAE). Retries other sequences if one isn't in the DB. | `AF-<acc>-F1.pdb`, `AF-<acc>-F1_predicted_aligned_error.json`, `structure_source.json` |
| `src/structure_analysis.py` | Build a PAE-masked min-heavy-atom **distance matrix**, map each IC's positions onto the structure, and measure **per-IC spatial clustering vs a size-matched null** (z-score) and **cross-IC neighbour distances** (reciprocal / one-sided contacts). | `ic_neighbor_distance.tsv`, `ic_cross_neighbor_distance.tsv`, `masked_dist_mat.npy/.png`, `IC_cross_dist_mat.png` |

`src/Pagel_lambda_phylo_signal.py` and `src/Identify_tree_splits.py` are the
underlying kernels reused by `SCA_analysis.py`. They can also be run standalone
on a Newick tree + a `seq_projections.tsv` (columns `seq_id`, `aligned_sequence`,
`Up_0…Up_k`); see each file's module docstring.

## Dependencies

Python (see `requirements.txt`):

```
pip install -r requirements.txt
```

External, not on PyPI — install separately:

- **[mysca](https://github.com/)** — the SCA package that produced the results
  folder; imported for `SCAResults` (sequence projection) and `PDBStructure`.
  Must be importable (`pip install -e /path/to/mysca`, or put it on
  `PYTHONPATH`).
- **[FAMSA](https://github.com/refresh-bio/FAMSA)** — guide-tree construction
  (`SCA_analysis.py`). Must be on `PATH` as `famsa`, or pass `--famsa-bin`.

## Input layout

The scripts expect an SCA results folder written by `mysca` containing:

```
<family>/
  preprocessing/
    preprocessing_results.npz   # msa, retained_sequence_ids, retained_positions
    sym2int.json                # residue alphabet
    msa_orig.fasta-aln          # original alignment (for IC->structure mapping)
  scacore/                      # SCAResults.load() target: phi_ia, fia,
                                # evecs_sca, evals_sca, w_ica, v_ica, ic_positions
```

## Usage

```bash
# 1. Per-IC phylogenetic-signal + clade-split analysis
python src/SCA_analysis.py --results-dir /path/to/PF00001

# 2. Download an AlphaFold structure for a random family member
python src/fetch_structure.py --results-dir /path/to/PF00001

# 3. Per-IC + cross-IC spatial clustering on that structure
python src/structure_analysis.py --results-dir /path/to/PF00001
```

Common flags: `--n-subsample` (default 1000), `--null-projections` (default
200), `--seed`, `--max-err` (PAE mask, default 8 Å), `--cross-thresh` (default
5 Å). Steps 2–3 read `structure_source.json`, so run `fetch_structure.py`
before `structure_analysis.py`.
