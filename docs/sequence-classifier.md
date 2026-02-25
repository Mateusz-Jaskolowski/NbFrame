# Sequence Classifier

[← Back to README](../README.md)

The sequence classifier predicts CDR3 conformation from amino acid sequence alone using a logistic regression model trained on 20 position-specific hallmark features.

---

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
| `--align/--no-align` | Perform AHo alignment (disable with `--no-align` for pre-aligned sequences) |
| `--fix-cdr1/--no-fix-cdr1` | Fix CDR-H1 gaps during alignment (default: fix) |
| `--batch-size INT` | Number of sequences per alignment batch (default: 100) |
| `--kinked-threshold FLOAT` | Threshold for kinked classification (default: 0.70) |
| `--extended-threshold FLOAT` | Threshold for extended classification (default: 0.40) |

### Output Format

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

---

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
# Adds columns: 'nbframe_score', 'raw_score', 'aligned_sequence', 'error'
```

---

## Classification Thresholds

| Probability Range | Label | Meaning |
|-------------------|-------|---------|
| P > 0.70 | **kinked** | High confidence kinked conformation |
| P < 0.40 | **extended** | High confidence extended conformation |
| 0.40 ≤ P ≤ 0.70 | **uncertain** | Classifier is uncertain |

Thresholds can be adjusted via `--kinked-threshold` and `--extended-threshold` options.

---

## Performance

- **Model**: Logistic Regression with 20 hallmark features
- **Test Performance**: ROC-AUC 0.939, AP 0.956, Accuracy 86.0%
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

See [Output Formats](output-formats.md) for details on CSV output columns.
