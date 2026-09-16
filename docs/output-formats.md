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
| `status` | `classified` or `insufficient_quality` |
| `error` | Explanation when a prediction is withheld |
| `model_id` | Selected Biopython model index (currently the first model, 0) |
| `warnings` | JSON list of quality messages |
| `quality` | JSON coordinate-quality report, retained with `--summary-only` |
| `feature_*` | Structural feature columns (unless `--summary-only` is used) |

Withheld predictions have empty label/probability cells. This is distinct from
an `uncertain` classification, which has a numeric probability. The separate
feature-table API also includes a JSON `quality` column; invalid measurements
are empty rather than zero.

---

## Structure Classifier JSON

Generated with `nbframe classify-structure -p nanobody.pdb --output-json result.json`.
The following is an illustrative excerpt; full results also contain the detailed
quality regions, model feature list, and model performance metadata.

```json
{
  "pdb_path": "nanobody.pdb",
  "chain_id_used": "A",
  "model_id_used": 0,
  "status": "classified",
  "error": null,
  "label": "kinked",
  "confidence": 0.95,
  "prob_kinked": 0.95,
  "prob_extended": 0.05,
  "probabilities": {"kinked": 0.95, "extended": 0.05},
  "features": {
    "alpha_N": 169.2,
    "tau_N": 107.8,
    "alpha_C": 46.8,
    "tau_C": 104.0,
    "cos_alpha_N": -0.982,
    "cos_alpha_C": 0.685,
    "contact_density": 0.42,
    "contact_nres": 3.4,
    "fr2_rsa_key": 0.31
  },
  "model_info": {
    "model_file": "structure_classifier_pipeline_2026-06-13.joblib",
    "date_trained": "2026-06-13",
    "thresholds": {"kinked": 0.55, "extended": 0.25}
  },
  "warnings": [],
  "quality": {
    "version": 1,
    "status": "passed",
    "chain_id": "A",
    "model_id": 0,
    "sequence_coverage": "not_assessed",
    "issues": [],
    "invalid_features": []
  }
}
```

For insufficient coordinates, `status` is `insufficient_quality`, `label`,
`confidence`, and both probabilities are `null`, and `quality.issues` explains
why. Valid individual measurements remain available, while affected features
are `null`. The CLI writes requested reports before exiting with code 1 when
no prediction could be produced. See the [coordinate-quality policy](structure-classifier.md#coordinate-completeness-and-withheld-predictions).
