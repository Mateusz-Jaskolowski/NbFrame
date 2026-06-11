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
- **Features**: cos(α_N), τ_N, cos(α_C), τ_C, contact density, FR2 RSA
- **Test Performance**: ROC-AUC 0.992, Accuracy 94.0% (5-fold CV ROC-AUC 0.970)

---

## Structural Features

The classifier uses six features computed from AHo-numbered structures:

| Feature | Description |
|---------|-------------|
| cos(α_N) | Cosine of the CDR3 N-terminal dihedral |
| τ_N | Bond angle at CDR3 N-terminus |
| cos(α_C) | Cosine of the CDR3 C-terminal dihedral |
| τ_C | Bond angle at CDR3 C-terminus |
| contact_density | CDR3–FR2 soft (logistic) contact density |
| fr2_rsa_key | FR2 key residue solvent accessibility |

The raw dihedral angles `alpha_N` and `alpha_C` are still reported in the output
features for interpretability; the classifier consumes their cosines (see below).

### Contact density (soft contacts, v0.2.0)

`contact_density` summarises how tightly CDR3 packs against framework region 2
(FR2). For each CDR3 non-stem residue and each FR2 residue (AHo 44–55), the
minimum heavy-atom distance `d` is mapped to a contact weight via a logistic
switching function and summed, then normalised by the number of non-stem CDR3
residues present:

```
contact_weight(d) = 1 / (1 + exp((d - 4.5) / 0.4))
contact_density   = Σ contact_weight(d) / (non-stem CDR3 length present)
```

The midpoint (4.5 Å) matches the legacy hard cutoff, and the 0.4 Å width gives a
10–90% transition over roughly 3.6–5.4 Å. Replacing the original hard 4.5 Å step
with this smooth switch removes the discontinuity that caused single-frame label
flips when a contact drifted across the cutoff in MD/ensembles, while slightly
improving K/E discrimination. The classifier is trained to match this definition.
To reproduce the legacy v0.1.1 binary-contact behaviour, set
`USE_SOFT_CONTACTS = False` in `nbframe.structure_config`.

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
