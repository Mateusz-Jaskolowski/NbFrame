# NbFrame Tests

This directory contains the comprehensive test suite for the NbFrame package.

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures and shared setup
├── test_sequence_classifier.py    # Sequence classifier tests
├── test_structure_classifier.py   # Structure classifier tests
├── test_structure_features.py     # Structure feature computation tests
├── test_cli.py                    # CLI integration tests
├── data/
│   ├── sequences/
│   │   ├── single_kinked.fasta           # 1 kinked sequence
│   │   ├── single_extended.fasta         # 1 extended sequence
│   │   ├── batch_mixed.fasta             # 6 sequences (3K + 3E)
│   │   ├── batch_mixed_aligned.fasta     # 6 sequences, pre-aligned
│   │   ├── large_batch.fasta             # 100 sequences for performance
│   │   └── invalid_sequences.fasta       # Edge cases
│   └── pdbs/
│       ├── kinked/                       # Kinked conformation PDBs
│       │   ├── 1f2x.pdb, 1mvf.pdb, 8im0.pdb  # Core test set
│       │   └── 9bsv.pdb, 9co6.pdb, ...   # High-confidence validation
│       ├── extended/                     # Extended conformation PDBs
│       │   ├── 4cdg.pdb, 5m2w.pdb, 5h8d.pdb  # Core test set
│       │   └── 9bdo.pdb, 9bsu.pdb, ...   # High-confidence validation
│       ├── edge_cases/                   # Edge case PDBs
│       │   ├── missing_VHH_chains_5GRY.pdb  # No VHH chains
│       │   └── 9ato.pdb, 9bt8.pdb, ...   # Classifier disagreement
│       └── mmcif/                        # mmCIF format test files
│           ├── 9bsv.cif                  # Kinked (mmCIF)
│           └── 9bdo.cif                  # Extended (mmCIF)
├── TestSequences.md               # Documentation of curated test sequences
└── README.md                      # This file
```

## Running Tests

### Using pytest (Recommended)

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_sequence_classifier.py

# Run specific test
pytest tests/test_sequence_classifier.py::TestSequenceClassifier::test_single_prediction

# Run with coverage
pytest tests/ --cov=nbframe --cov-report=html
```

## Test Categories

### Sequence Classifier Tests (`test_sequence_classifier.py`)
- Single sequence prediction
- Batch prediction
- FASTA file processing
- DataFrame integration
- Alignment functions
- Blueprint scoring
- Error handling

### Structure Classifier Tests (`test_structure_classifier.py`)
- Model loading
- Feature-based prediction
- PDB file classification
- Multi-chain handling
- Edge cases (no VHH chains)
- mmCIF format support

### Structure Features Tests (`test_structure_features.py`)
- Module imports
- PDB/mmCIF loading
- Feature computation
- Angle calculations
- Contact density

### CLI Tests (`test_cli.py`)
- `nbframe classify-sequence` command
- `nbframe classify-structure` command
- FASTA input
- CSV output
- Custom thresholds

## Test Data

### Core Test Sequences (from TestSequences.md)

**Kinked:**
- `1f2x_L`: Chain L from PDB 1f2x
- `1mvf_B`: Chain B from PDB 1mvf
- `8im0_B`: Chain B from PDB 8im0

**Extended:**
- `4cdg_D`: Chain D from PDB 4cdg
- `5m2w_B`: Chain B from PDB 5m2w
- `5h8d_A`: Chain A from PDB 5h8d

### Validation Set (High Confidence)

**Kinked:** 9bsv, 9co6, 9g5k, 9jrs, 9kgk
**Extended:** 9bdo, 9bsu, 9lds, 9gv4, 9h39

### Edge Cases (Classifier Disagreement)

9ato, 9bt8, 9dvq, 9e7k, 9j7y

## Adding New Tests

1. Add test methods to the appropriate test class
2. Use fixtures from `conftest.py` for common setup
3. Add any new test data files to `data/` directory
4. Update this README if adding new test categories

## Dependencies

Tests require:
- pytest (recommended)
- pytest-cov (for coverage)
- pandas
- numpy
- typer (for CLI tests)

All dependencies are included in the nbframe conda environment.
