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
- **Features**: α_N, τ_N, α_C, τ_C, contact density, FR2 RSA
- **Test Performance**: ROC-AUC 0.994, AP 0.995, Accuracy 94.0%

---

## Structural Features

The classifier uses six features computed from AHo-numbered structures:

| Feature | Description |
|---------|-------------|
| α_N | Dihedral angle at CDR3 N-terminus |
| τ_N | Bond angle at CDR3 N-terminus |
| α_C | Dihedral angle at CDR3 C-terminus |
| τ_C | Bond angle at CDR3 C-terminus |
| contact_density | CDR3–FR2 contact density |
| fr2_rsa_key | FR2 key residue solvent accessibility |

---

See [Output Formats](output-formats.md) for details on CSV and JSON output schemas.
