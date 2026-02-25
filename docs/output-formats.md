# Output Formats

[← Back to README](../README.md)

Both classifiers can write results to CSV files. The structure classifier additionally supports JSON output.

---

## Sequence Classifier CSV

Generated with `nbframe classify-sequence -f input.fasta -o results.csv`.

| Column | Description |
|--------|-------------|
| `name` | Sequence identifier (from FASTA header) |
| `sequence` | Input sequence |
| `nbframe_score` | Probability of kinked conformation — P(kinked) |
| `raw_score` | Model logit score |
| `aligned_sequence` | AHo-aligned sequence (if alignment was performed) |
| `label` | Classification label: kinked / extended / uncertain |

---

## Structure Classifier CSV

Generated with `nbframe classify-structure -d pdb_folder/ --output-csv results.csv`.

| Column | Description |
|--------|-------------|
| `pdb_path` | Path to the input PDB file |
| `pdb_name` | PDB filename |
| `chain_id` | Chain identifier |
| `label` | Classification label: kinked / extended / uncertain |
| `prob_kinked` | Probability of kinked conformation |
| `prob_extended` | Probability of extended conformation |
| `feature_*` | Structural feature columns (unless `--summary-only` is used) |

---

## Structure Classifier JSON

Generated with `nbframe classify-structure -p nanobody.pdb --output-json result.json`.

```json
{
  "pdb_path": "nanobody.pdb",
  "chain_id_used": "A",
  "label": "kinked",
  "confidence": 0.95,
  "prob_kinked": 0.95,
  "prob_extended": 0.05,
  "probabilities": {"kinked": 0.95, "extended": 0.05},
  "features": {
    "alpha_N": 1.23,
    "tau_N": 0.45,
    "alpha_C": -2.10,
    "tau_C": 1.85,
    "contact_density": 0.42,
    "fr2_rsa_key": 0.31
  },
  "model_info": {
    "model_file": "structure_classifier_pipeline_2026-01-19.joblib",
    "date_trained": "2026-01-19",
    "thresholds": {"kinked": 0.55, "extended": 0.25}
  },
  "warnings": []
}
```
