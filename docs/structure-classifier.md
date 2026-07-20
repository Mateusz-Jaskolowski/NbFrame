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
| `--output-aho-pdb DIR` | Save AHo-numbered PDBs |
| `-v, --verbose` | Show detailed output |
| `--recursive` | Search PDB directory recursively |
| `--summary-only` | Exclude per-structure features from JSON and CSV output |
| `--no-rmsd-filter` | Disable framework RMSD quality filter |
| `--rmsd-threshold FLOAT` | Maximum framework RMSD (default: 2.0 Å) |
| `--progress-interval INT` | Number of PDBs processed between progress updates (default: 50) |
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
