"""fetch_structure.py

Pick a random sequence from a family's processed alignment, resolve its
UniProt accession, and download the corresponding AlphaFold structure (PDB)
and predicted aligned error (PAE JSON) into the family's tmp folder.

The AlphaFold DB should cover essentially all UniProt accessions in these
families, but a given accession can occasionally be missing (obsolete entry,
fragment, etc.), so we draw a few sequences at random and use the first one
that resolves.

API: https://alphafold.ebi.ac.uk/api/prediction/{accession}  -> JSON list;
each entry exposes ``pdbUrl`` and ``paeDocUrl`` (among others). See
https://alphafold.ebi.ac.uk/api-docs.

Nothing is written into the SCA results folder; downloads go to
``<tmp-dir>/<family>/``.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

ALPHAFOLD_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
_USER_AGENT = "mysca-structure-fetch/1.0 (https://alphafold.ebi.ac.uk/api-docs)"


def extract_accession(seq_id):
    """'A0A6J1W0I0_9SAUR/25-273' -> 'A0A6J1W0I0'.

    Strips any alignment range after '/' and the organism mnemonic after the
    first '_' (UniProt entry-name convention ``ACCESSION_ORGANISM`` for the
    TrEMBL-style IDs in these alignments).
    """
    token = seq_id.split("/", 1)[0]
    return token.split("_", 1)[0]


def load_sequence_ids(preprocess_dir):
    """Return the family's retained_sequence_ids (read-only, ids only)."""
    npz = os.path.join(preprocess_dir, "preprocessing_results.npz")
    with np.load(npz, allow_pickle=True) as ali:
        return np.asarray(ali["retained_sequence_ids"]).astype(str)


def alphafold_prediction(accession, timeout=30):
    """Return the AlphaFold prediction list for an accession, or None if the
    accession is absent (HTTP 404 or empty list)."""
    url = ALPHAFOLD_PREDICTION_URL.format(accession)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return data or None


def download(url, dest, timeout=120):
    """Stream a URL to a local path."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    return dest


def fetch_structure(seq_ids, out_dir, rng, max_tries=5):
    """Draw random sequences until one resolves in AlphaFold; download its
    PDB + PAE JSON into ``out_dir``.

    Returns a metadata dict (seq_id, accession, entryId, version, file paths).
    Raises RuntimeError if no drawn accession resolves within ``max_tries``.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(seq_ids)
    tried = []
    for attempt in range(1, max_tries + 1):
        seq_id = str(seq_ids[int(rng.integers(n))])
        accession = extract_accession(seq_id)
        if accession in tried:
            continue
        tried.append(accession)
        print(f"  [{attempt}/{max_tries}] {seq_id} -> accession {accession}",
              file=sys.stderr)
        try:
            pred = alphafold_prediction(accession)
        except urllib.error.URLError as e:
            print(f"      network error: {e}; trying another.", file=sys.stderr)
            continue
        if not pred:
            print("      not in AlphaFold DB; trying another.", file=sys.stderr)
            continue

        entry = pred[0]
        entry_id = entry.get("entryId", f"AF-{accession}-F1")
        pdb_url = entry.get("pdbUrl")
        pae_url = entry.get("paeDocUrl")
        if not pdb_url or not pae_url:
            print("      entry lacks pdbUrl/paeDocUrl; trying another.",
                  file=sys.stderr)
            continue

        pdb_path = os.path.join(out_dir, f"{entry_id}.pdb")
        pae_path = os.path.join(out_dir, f"{entry_id}_predicted_aligned_error.json")
        print(f"      downloading {entry_id} (v{entry.get('latestVersion')})...",
              file=sys.stderr)
        download(pdb_url, pdb_path)
        download(pae_url, pae_path)

        meta = {
            "seq_id": seq_id,
            "accession": accession,
            "entryId": entry_id,
            "uniprotAccession": entry.get("uniprotAccession"),
            "latestVersion": entry.get("latestVersion"),
            "pdbUrl": pdb_url,
            "paeDocUrl": pae_url,
            "pdb_path": pdb_path,
            "pae_path": pae_path,
            "n_attempts": attempt,
        }
        meta_path = os.path.join(out_dir, "structure_source.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"      wrote {pdb_path}", file=sys.stderr)
        print(f"      wrote {pae_path}", file=sys.stderr)
        print(f"      wrote {meta_path}", file=sys.stderr)
        return meta

    raise RuntimeError(
        f"No AlphaFold structure found after {max_tries} attempts "
        f"(accessions tried: {', '.join(tried)}).")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="./PF00001",
                   help="SCA results folder containing preprocessing/.")
    p.add_argument("--tmp-dir", default="./tmp",
                   help="Scratch/output root (created in the cwd).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tries", type=int, default=5,
                   help="Random sequences to try before giving up.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results_dir = os.path.abspath(args.results_dir)
    family = os.path.basename(results_dir.rstrip("/"))
    preprocess_dir = os.path.join(results_dir, "preprocessing")
    out_dir = os.path.join(os.path.abspath(args.tmp_dir), family)

    print(f"Family:      {family}", file=sys.stderr)
    print(f"Output dir:  {out_dir}", file=sys.stderr)
    seq_ids = load_sequence_ids(preprocess_dir)
    print(f"Alignment:   {len(seq_ids)} sequences", file=sys.stderr)
    print("Fetching an AlphaFold structure for a random sequence...",
          file=sys.stderr)
    meta = fetch_structure(seq_ids, out_dir, rng, max_tries=args.max_tries)

    print(f"\n{family}: {meta['accession']} ({meta['entryId']}, "
          f"v{meta['latestVersion']})")
    print(f"  PDB: {meta['pdb_path']}")
    print(f"  PAE: {meta['pae_path']}")


if __name__ == "__main__":
    main()
