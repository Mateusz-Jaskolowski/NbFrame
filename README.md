# NbFrame

**Classify nanobody structure or predict nanobody CDR3 conformation as kinked or extended from sequence alone.**

NbFrame is a Python package that predicts whether a nanobody (VHH) has a kinked or extended CDR3 loop conformation. It provides both sequence-based and structure-based classifiers with high accuracy.

## Features

- **Sequence-based classification** using a logistic regression model trained on 20 hallmark features (ROC-AUC: 0.94)
- **Structure-based classification** from PDB files using geometric and contact features (ROC-AUC: 0.99)
- **Three-class output**: kinked, extended, or uncertain (with transparent probability thresholds)
- **Batch processing** for high-throughput analysis (millions of sequences in minutes)
- **CLI and Python API** for flexible integration

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/Mateusz-Jaskolowski/NbFrame.git
cd NbFrame

# Create and activate environment
conda env create -f environment.yml
conda activate nbframe

# Install the package
pip install -e .

# Verify installation
nbframe --help
```

### Dependencies

NbFrame requires HMMER and ANARCI for sequence alignment. The Conda environment handles these automatically.

---

## Quick Start

### Sequence Classifier

Classify a single sequence:

```bash
nbframe classify-sequence -s EVQLVESGGGLVQAGGSLRLSCAASGRIEDINAMGWFRQAPGKEREFVALISHIGSNRVYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGGQVFATRVKSYAYWGKGTPVTVSS
```

Output:
```
Prediction:  KINKED
P(kinked):   0.8647
Thresholds:  kinked >0.70, extended <0.40
```

Classify from a FASTA file:

```bash
nbframe classify-sequence -f sequences.fasta -o results.csv
```

### Structure Classifier

Classify a single PDB (auto-detects VHH chains):

```bash
nbframe classify-structure -p nanobody.pdb
```

Classify a directory of PDBs:

```bash
nbframe classify-structure -d pdb_folder/ -o results.csv
```

---

# Sequence Classifier

The sequence classifier predicts CDR3 conformation from amino acid sequence alone using a logistic regression model trained on 20 position-specific hallmark features.

## Command-Line Interface

```bash
nbframe classify-sequence [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-s, --sequence TEXT` | Single VHH sequence string |
| `-f, --fasta FILE` | FASTA file with one or more sequences |
| `-o, --output-csv FILE` | Save results to CSV |
| `-v, --verbose` | Show detailed output |
| `--label/--no-label` | Include/exclude classification label (default: label) |
| `--no-align` | Skip alignment (use for pre-aligned sequences) |
| `--fix-cdr1/--no-fix-cdr1` | Fix CDR-H1 gaps during alignment (default: fix) |
| `--batch-size INT` | Number of sequences per alignment batch (default: 100) |
| `--kinked-threshold FLOAT` | Threshold for kinked classification (default: 0.70) |
| `--extended-threshold FLOAT` | Threshold for extended classification (default: 0.40) |

**Output Format:**

For single sequences, the classifier shows the prediction with the raw probability and thresholds:

```
Prediction:  KINKED
P(kinked):   0.8647
Thresholds:  kinked >0.70, extended <0.40
```

For uncertain predictions, an explanation is provided:

```
Prediction:  UNCERTAIN
P(kinked):   0.5234
Thresholds:  kinked >0.70, extended <0.40

Note: Falls within uncertainty zone (0.40–0.70). Consider structural analysis for higher confidence.
```

For batch processing (FASTA files), results are shown in a color-coded table:

```
       NbFrame Sequence Predictions
┌──────────┬────────────┬──────────┐
│ Name     │ Prediction │ P(kinked)│
├──────────┼────────────┼──────────┤
│ seq_001  │ KINKED     │ 0.8647   │
│ seq_002  │ EXTENDED   │ 0.1234   │
│ seq_003  │ UNCERTAIN  │ 0.5234   │
└──────────┴────────────┴──────────┘

Thresholds: kinked >0.70, extended <0.40
```

## Python API

### Single Sequence

```python
from nbframe import classify_sequence

result = classify_sequence(
    "EVQLVESGGGLVQAGGSLRLSCAASGRIEDINAMGWFRQAPGKEREFVALISHIGSNRVYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGGQVFATRVKSYAYWGKGTPVTVSS"
)

print(result['label'])       # 'kinked', 'extended', or 'uncertain'
print(result['probability']) # P(kinked), e.g., 0.865
print(result['thresholds'])  # {'kinked': 0.70, 'extended': 0.40}
```

### Batch Processing

```python
from nbframe import classify_sequences, predict_from_fasta

# From a list of sequences
results = classify_sequences(
    ["SEQ1...", "SEQ2...", "SEQ3..."],
    batch_size=1000,
    verbose=True
)

# From a FASTA file
df = predict_from_fasta(
    "sequences.fasta",
    batch_size=1000,
    output_csv="results.csv"
)
```

### Working with DataFrames

```python
from nbframe import predict_dataframe

# Add predictions to an existing DataFrame
df = predict_dataframe(
    your_df,
    sequence_column='sequence',
    verbose=True
)
# Adds columns: 'nbframe_score', 'raw_score', 'aligned_sequence'
```

## Classification Thresholds

| Probability Range | Label | Meaning |
|-------------------|-------|---------|
| P > 0.70 | **kinked** | High confidence kinked conformation |
| P < 0.40 | **extended** | High confidence extended conformation |
| 0.40 ≤ P ≤ 0.70 | **uncertain** | Classifier is uncertain |

Thresholds can be adjusted via `--kinked-threshold` and `--extended-threshold` options.

## Performance

- **Model**: Logistic Regression with 20 hallmark features
- **Test Performance**: ROC-AUC 0.941, Accuracy 88.0%
- **Speed**: ~3.5 million sequences/minute on M3 MacBook Pro (pre-aligned)

### Batch Size Recommendations

| Dataset Size | Recommended Batch Size |
|--------------|------------------------|
| < 1,000 | 500–1,000 |
| 1,000–10,000 | 1,000–5,000 |
| 10,000–100,000 | 5,000–10,000 |
| > 100,000 | 10,000–50,000 |

### Pre-aligned Sequences

If your sequences are already aligned to the AHo numbering scheme, skip alignment for faster processing:

```bash
nbframe classify-sequence -f aligned.fasta --no-align
```

```python
results = classify_sequences(sequences, do_alignment=False)
```

---

# Structure Classifier

The structure classifier predicts CDR3 conformation from 3D structure (PDB files) using geometric and contact-based features computed from AHo-numbered coordinates.

## Command-Line Interface

```bash
nbframe classify-structure [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-p, --pdb FILE` | Single PDB file |
| `-d, --pdb-dir DIR` | Directory of PDB files |
| `-c, --chain ID` | Specific chain ID(s), comma-separated |
| `--output-json FILE` | Save full results as JSON |
| `--output-csv FILE` | Save summary as CSV |
| `--output-aho-pdb DIR` | Save AHo-numbered PDBs |
| `-v, --verbose` | Show detailed output |
| `--recursive` | Search PDB directory recursively |
| `--summary-only` | Exclude per-residue features from JSON output |
| `--no-rmsd-filter` | Disable framework RMSD quality filter |
| `--rmsd-threshold FLOAT` | Maximum framework RMSD (default: 2.0 Å) |
| `--progress-interval INT` | Seconds between progress updates for directory mode |
| `--kinked-threshold FLOAT` | Threshold for kinked classification (default: 0.55) |
| `--extended-threshold FLOAT` | Threshold for extended classification (default: 0.25) |

**Output Format:**

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

## Classification Thresholds

| Probability Range | Label | Meaning |
|-------------------|-------|---------|
| P > 0.55 | **kinked** | High confidence kinked conformation |
| P < 0.25 | **extended** | High confidence extended conformation |
| 0.25 ≤ P ≤ 0.55 | **uncertain** | Classifier is uncertain |

Thresholds can be adjusted via `--kinked-threshold` and `--extended-threshold` options.

## Performance

- **Model**: Logistic Regression with 6 structural features
- **Features**: α_N, τ_N, α_C, τ_C, contact density, FR2 RSA
- **Test Performance**: ROC-AUC 0.994, Accuracy 94.0%

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

## Output Files

### CSV Output

The CSV contains:
- `name` – Sequence/structure identifier
- `sequence` – Input sequence (sequence classifier)
- `nbframe_score` / `prob_kinked` – Probability of kinked conformation
- `label` – Classification label
- `raw_score` – Model logit score
- `aligned_sequence` – AHo-aligned sequence (if alignment was performed)

### JSON Output (Structure Classifier)

```json
{
  "label": "kinked",
  "probabilities": {"kinked": 0.95, "extended": 0.05},
  "features": {
    "alpha_N": 1.23,
    "tau_N": 0.45,
    "contact_density": 2.1
  }
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use NbFrame in your research, please cite:

> [Citation information to be added upon publication]

## Acknowledgments

Developed with high affinity for caffeine and a low dissociation constant for long coding sessions in [Sormanni Lab](https://www-sormanni.ch.cam.ac.uk/) at the University of Cambridge, Department of Chemistry.

No actual llamas were harmed (or even consulted) during the making of this classifier.
