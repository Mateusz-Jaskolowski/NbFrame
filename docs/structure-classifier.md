# Structure Classifier

[← Back to README](../README.md)

The structure classifier predicts CDR3 conformation from 3D structure using geometric and contact-based features computed from AHo-numbered coordinates. Both PDB and mmCIF file formats are supported.

---

## Command-Line Interface

```bash
nbframe classify-structure [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-p, --pdb FILE` | Single PDB/mmCIF file |
| `-d, --pdb-dir DIR` | Directory of PDB/mmCIF files |
| `-c, --chain ID` | Specific chain ID(s), comma-separated |
| `--output-json FILE` | Save full results as JSON |
| `--output-csv FILE` | Save summary as CSV |
| `--output-aho-pdb DIR` | Save AHo-numbered structures; multi-character chain IDs use mmCIF |
| `-v, --verbose` | Show detailed output |
| `--recursive` | Search PDB directory recursively |
| `--summary-only` | Exclude per-structure features from JSON and CSV output |
| `--no-rmsd-filter` | Disable framework RMSD quality filter |
| `--rmsd-threshold FLOAT` | Maximum framework RMSD (default: 2.0 Å) |
| `--progress-interval INT` | Positive number of structures between progress updates (default: 50) |
| `--kinked-threshold FLOAT` | Threshold for kinked classification (default: 0.55) |
| `--extended-threshold FLOAT` | Threshold for extended classification (default: 0.25) |

### Output Format

For a single chain, the classifier shows detailed output:

```
Chains identified as VHH in the provided PDB:

Chain:       A
Prediction:  KINKED
P(kinked):   0.9502
Thresholds:  kinked >0.55, extended <0.25
```

For multiple chains, a compact format is used:

```
Chains identified as VHH in the provided PDB:

  Chain A:
    Prediction:  KINKED
    P(kinked):   0.9502

  Chain B:
    Prediction:  EXTENDED
    P(kinked):   0.0234

Thresholds: kinked >0.55, extended <0.25
```

---

## Python API

### Single Structure

```python
from nbframe import classify_structure

result = classify_structure("nanobody.pdb", chain_id="H")

print(result['label'])        # 'kinked', 'extended', or 'uncertain'
print(result['probabilities']) # {'kinked': 0.95, 'extended': 0.05}
print(result['features'])      # Computed structural features
```

### All Chains in a PDB

```python
from nbframe import classify_all_nanobodies_in_pdb

results = classify_all_nanobodies_in_pdb("complex.pdb")

for chain_id, result in results.items():
    print(f"Chain {chain_id}: {result['label']}")
```

Automatic discovery considers unpaired heavy chains of 90–150 residues as
VHH-like candidates. In a mixed Fab/nanobody complex, light chains only exclude
the heavy domains with which they form a local interface. The pairing heuristic
requires at least three heavy FR2 Cα atoms within 8 Å of a numbered light domain
and matches the strongest interfaces first. Chain identity is still heuristic;
use explicit chain IDs for ambiguous complexes or tagged/fused domains outside
the automatic length range. The bundled 9bt8 example now detects nanobody A
while excluding the Fab heavy chain H paired with light chain L.

### Files, residue identities, and quality filtering

Directory mode recognises `.pdb`, `.ent`, `.cif`, and `.mmcif`, case-insensitively.
Multi-character chain IDs are preserved in mmCIF intermediates and exports.
Saved filenames include a digest of the absolute input path so that structures
with identical basenames in different directories cannot silently replace one
another in ordinary use.

`compute_features_for_pdb_directory` uses paths relative to the input directory,
including the extension, as `Structure_ID` and dictionary keys. Its DataFrame
and CSV outputs contain both cosine features required by the current model,
alongside the raw angles. Passing an explicit `pattern` still restricts file
discovery to that glob.

The low-level residue mapping keeps ordinary AHo positions as integer keys and
insertions as string keys, for example `123`, `"123A"`, and `"123H"`. Contact and
length calculations include inserted residues independently; framework RMSD
matches the base framework positions. Hydrogens and deuterium are excluded from
contacts and RSA using their element, with atom-name fallback when it is absent.

Angle calculations require CA atoms and connected peptide neighbours. Missing
backbone atoms or broken C–N/CA connectivity return missing angles and prevent
classification, rather than allowing later resolved residues to substitute for
missing neighbours. Framework-filter warnings distinguish excessive RMSD from
insufficient matching framework coverage.

### Coordinate completeness and withheld predictions

Classification also checks the coordinates used by the structural features.
The current conservative policy requires all expected standard-residue heavy
atoms in the resolved CDR3 and FR2 regions, including the RSA key residue at
AHo 44. Hydrogens and terminal OXT are optional. Missing atoms, non-finite
coordinates, and atoms with non-positive or non-finite occupancy are unusable. Unknown
occupancy is accepted when coordinates are present.

All base FR2 positions 44–55 must be present. CDR3 numbering gaps are allowed:
they are not interpreted as missing residues. Observed neighbours spanning
CDR3 and FR2 must have usable backbone atoms, C–N distances of 0.8–2.0 Å,
and CA–CA distances of 2.5–4.5 Å. Angle anchors are checked separately.

Missing heavy atoms or broken backbone connectivity in these regions
conservatively invalidate contact and RSA measurements. Invalidated features
are returned as `None`; the classifier returns `status="insufficient_quality"`,
with `label`, confidence, and probabilities set to `None`. The `quality` report
lists affected features, residue identities (including insertions), atom
completeness, required positions, and backbone breaks. This status describes
insufficient input data; it is distinct from the model's `uncertain` label.

Complete inputs return `status="classified"`. Their feature definitions and
model coefficients are unchanged. Some deposited structures omit disordered
sidechains and will now have predictions withheld, even when they pass the
framework RMSD filter. `--no-rmsd-filter` disables only that framework filter;
it does not disable coordinate-completeness checks.

```python
result = classify_structure("nanobody.pdb", chain_id="B")
if result is None:
    print("No result: framework filtering or no eligible chain")
elif result["status"] == "insufficient_quality":
    print(result["error"])
    print(result["quality"]["regions"])
else:
    print(result["label"], result["prob_kinked"])
```

Multi-chain classification retains withheld results alongside successful ones.
The CLI writes these reports to JSON/CSV, including with `--summary-only`, and
exits with code 1 if none of the reported chains produced a prediction. Mixed
runs with at least one prediction exit successfully. Existing no-chain and
framework-filter behavior remains unchanged.

The report's atom-completeness fractions use the expected atoms of **observed
residues**, not a full-sequence coverage estimate. `sequence_coverage` is
`"not_assessed"`: the package does not reconstruct unresolved sequence from
SEQRES or mmCIF polymer records. These checks also do not assess experimental
resolution, B factors, atoms outside the checked regions (except non-finite
coordinates affecting SASA), or all possible alternate conformers. Alternate
locations generate a warning; features use the selected conformers. A passed
report is a coordinate check, not a guarantee of biological accuracy.

Feature-only calls to `predict_structure_from_features` retain their existing
numeric-input API and report quality as `"not_assessed"` when no coordinate
report is supplied. Feature tables carry the report as a JSON `quality` column;
CSV round trips preserve withheld status. Non-finite model inputs are rejected.

Direct callers of `renumber_structure_to_aho` who omit `temp_dir` own the returned
temporary directory and must remove it when finished. For example, after using
the file, call `shutil.rmtree(output_path.parent)`. High-level classification and
batch APIs continue to clean up their temporary directories automatically.

---

## Classification Thresholds

| Probability Range | Label | Meaning |
|-------------------|-------|---------|
| P > 0.55 | **kinked** | High confidence kinked conformation |
| P < 0.25 | **extended** | High confidence extended conformation |
| 0.25 ≤ P ≤ 0.55 | **uncertain** | Classifier is uncertain |

Thresholds can be adjusted via `--kinked-threshold` and `--extended-threshold` options.

---

## Performance

- **Model**: Logistic Regression with 6 structural features
- **Features**: cos(α_N), τ_N, cos(α_C), τ_C, contact_nres, FR2 RSA(44)
- **Training data**: 100 original expert labels + 46 additional expert labels
  (146 structures; 135 with a clear kinked/extended call)
- **Performance (v0.3.0)**: held-out pool (138 structures, no overlap with
  training) ROC-AUC 0.997, Accuracy 97.8%; 5-fold CV on the training set
  ROC-AUC 0.92, Accuracy 0.86 (CV is lower than v0.2.0's because the training
  set is now enriched with the hardest, most ambiguous cases).

---

## Structural Features

The classifier uses six features computed from AHo-numbered structures:

| Feature | Description |
|---------|-------------|
| cos(α_N) | Cosine of the CDR3 N-terminal dihedral |
| τ_N | Bond angle at CDR3 N-terminus |
| cos(α_C) | Cosine of the CDR3 C-terminal dihedral |
| τ_C | Bond angle at CDR3 C-terminus |
| contact_nres | Number of CDR3 residues contacting FR2 (soft, length-independent) |
| fr2_rsa_key | FR2 key residue (AHo 44) solvent accessibility |

The raw dihedral angles `alpha_N` and `alpha_C` are still reported in the output
features for interpretability; the classifier consumes their cosines (see below).
The legacy length-normalised `contact_density` is also still reported in the
output features, but the v0.3.0 classifier consumes `contact_nres` instead.

### Contact count (`contact_nres`, v0.3.0)

`contact_nres` counts **how many CDR3 residues touch framework region 2 (FR2)**.
For each non-stem CDR3 residue we take its single closest heavy-atom approach to
any FR2 contact residue (AHo 44–55) and map it through the same logistic switch
used for soft contacts, then sum over CDR3 residues:

```
contact_weight(d) = 1 / (1 + exp((d - 4.5) / 0.4))
contact_nres      = Σ_residues  contact_weight( min distance of that residue to FR2 )
```

Crucially this is **not** normalised by loop length. The expert definition of a
kinked conformation is "any part of CDR3 contacting FR2" — even a single side
chain reaching across in an otherwise extended loop. The v0.2.0 feature
`contact_density` divided the contact sum by CDR3 length, which diluted exactly
these localised contacts in long loops and caused such structures to be called
extended. Against a contamination-free set of 46 expert labels, `contact_nres`
separated kinked from extended markedly better than `contact_density`
(univariate ROC-AUC ≈ 0.95 vs ≈ 0.91), which is the main driver of the v0.3.0
improvement. The soft (logistic) weighting keeps the count continuous and robust
to small coordinate changes across MD frames and refinement ensembles. To
reproduce the legacy binary-contact behaviour, set `USE_SOFT_CONTACTS = False`
in `nbframe.structure_config`.

### FR2 RSA at position 44 (v0.3.0)

`fr2_rsa_key` is the relative solvent accessibility of the FR2 hallmark
residue(s) in `FR2_KEY_RSA_AHOS`. In v0.3.0 this is narrowed to **AHo position 44
only** (v0.2.0 used 44 + 54). A feature bake-off against the expert labels showed
position 44 carries almost all of the RSA signal — when CDR3 folds back onto FR2
it buries position 44 — while AHo 54 was near-uninformative and only diluted the
feature.

### Dihedral encoding (cosine, v0.2.0)

`alpha_N` and `alpha_C` are *dihedral* angles in (−180°, 180°]. Feeding raw
degrees to a linear model is unstable near the ±180° branch cut: a ~2° geometric
change can flip the value between +179° and −179° (a 358° jump), which the model
misreads as a large conformational change. This caused spurious kinked/extended
flips across MD frames and X-ray refinement ensembles, and — because a large
fraction of structures sit near the cut — it even degraded training.

v0.2.0 therefore feeds the classifier `cos(alpha_N)` and `cos(alpha_C)` instead
of raw degrees. The cosine is wraparound-safe (cos(+179°) ≈ cos(−179°)) and
captures the physically meaningful trans/gauche character of the dihedral. The
bond angles `tau_N` and `tau_C` are bounded in [0°, 180°] with no wraparound and
are kept linear. This change improves 5-fold CV (ROC-AUC 0.962 → 0.970, recall
0.83 → 0.90) and substantially stabilises MD/ensemble predictions, with only a
marginal change on held-out crystal structures.

---

See [Output Formats](output-formats.md) for details on CSV and JSON output schemas.
