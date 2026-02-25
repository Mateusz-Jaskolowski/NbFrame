# NbFrame

**Predict nanobody CDR3 conformation as kinked or extended from sequence or structure.**

NbFrame is a Python package that predicts whether a nanobody (VHH) has a kinked or extended CDR3 loop conformation. It provides both sequence-based and structure-based classifiers with high accuracy.

## Features

- **Sequence-based classification** using a logistic regression model trained on 20 hallmark features (ROC-AUC: 0.94)
- **Structure-based classification** from PDB/mmCIF files using geometric and contact features (ROC-AUC: 0.99)
- **Three-class output**: kinked, extended, or uncertain (with transparent probability thresholds)
- **Batch processing** for high-throughput analysis (millions of pre-aligned sequences in minutes; alignment adds overhead)
- **CLI and Python API** for flexible integration

## Installation

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

NbFrame requires HMMER and ANARCI for sequence alignment. The Conda environment handles these automatically.

## Quick Start

Classify a sequence:

```bash
nbframe classify-sequence -s EVQLVESGGGLVQAGGSLRLSCAASGRIEDINAMGWFRQAPGKEREFVALISHIGSNRVYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGGQVFATRVKSYAYWGKGTPVTVSS
```

```
Prediction:  KINKED
P(kinked):   0.8647
Thresholds:  kinked >0.70, extended <0.40
```

Classify from sequences in a FASTA file (with output to csv file):

```bash
nbframe classify-sequence -f sequences.fasta -o results.csv
```

Classify a PDB structure (auto-detects VHH chains):

```bash
nbframe classify-structure -p nanobody.pdb
```

Classify a directory of PDBs:

```bash
nbframe classify-structure -d pdb_folder/ --output-csv results.csv
```

## Documentation

| I want to... | Guide |
|--------------|-------|
| Classify sequences (full CLI & Python reference) | [Sequence Classifier](docs/sequence-classifier.md) |
| Classify structures (full CLI & Python reference) | [Structure Classifier](docs/structure-classifier.md) |
| Understand output CSV / JSON files | [Output Formats](docs/output-formats.md) |
| Use NbFrame from Python | [Sequence API](docs/sequence-classifier.md#python-api) / [Structure API](docs/structure-classifier.md#python-api) |
| Tune batch processing for large datasets | [Batch Tuning](docs/sequence-classifier.md#batch-size-recommendations) |

## Citation

If you use NbFrame in your research, please cite:

> Ali, M., Jaskolowski, M., Greenig, M., Nguyen, T. H., Crnogaj, M., Smorodina, E., Ramon, A., Zhao, H., Fernández-Quintero, M. L., Ghedin, E., Greiff, V., & Sormanni, P. (2026). Disulphide and sequence-encoded conformational priors guide nanobody structure prediction. *bioRxiv*, 2026.02.13.705647. https://doi.org/10.64898/2026.02.13.705647

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

Developed with high affinity for caffeine and a low dissociation constant for long coding sessions in [Sormanni Lab](https://www-sormanni.ch.cam.ac.uk/) at the University of Cambridge, Department of Chemistry.

No actual llamas were harmed (or even consulted) during the making of this classifier.
