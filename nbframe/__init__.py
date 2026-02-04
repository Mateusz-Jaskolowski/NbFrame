# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

# NbFrame: Nanobody CDR3 conformation classifier
# Classifies nanobodies as kinked or extended based on sequence and/or structure.

# Expose sequence-based prediction functions
from .sequence_predictor import (
    predict_kink_probability,
    predict_kink_probabilities,
    predict_dataframe,
    predict_from_fasta,
    read_fasta,
    results_to_dataframe,
    save_results_to_csv,
    # Classification functions with labels
    classify_sequence,
    classify_sequences,
)

# Also expose alignment helpers
from .sequence_align import batch_align_sequences

# Expose sequence classifier config
from .sequence_config import (
    DEFAULT_SEQ_KINKED_THRESHOLD,
    DEFAULT_SEQ_EXTENDED_THRESHOLD,
)

# Expose structure-based helpers (PDB → AHo → features / VHH detection)
from .structure_numbering import (
    compute_features_from_pdb,
    compute_features_for_pdbs,
    compute_features_for_pdb_directory,
    identify_nanobody_chains_from_pdb,
    identify_unique_nanobody_chains_from_pdb,
)

# Expose high-level structure classifier API
from .structure_classifier import (
    classify_structure,
    classify_all_nanobodies_in_pdb,
)
from .structure_config import (
    DEFAULT_STRUCT_KINKED_THRESHOLD as DEFAULT_KINKED_THRESHOLD,
    DEFAULT_STRUCT_EXTENDED_THRESHOLD as DEFAULT_EXTENDED_THRESHOLD,
)

__version__ = "1.0.0"

__all__ = [
    # Sequence prediction
    "predict_kink_probability",
    "predict_kink_probabilities",
    "predict_dataframe",
    "predict_from_fasta",
    "read_fasta",
    "results_to_dataframe",
    "save_results_to_csv",
    "classify_sequence",
    "classify_sequences",
    # Sequence alignment
    "batch_align_sequences",
    # Sequence config / thresholds
    "DEFAULT_SEQ_KINKED_THRESHOLD",
    "DEFAULT_SEQ_EXTENDED_THRESHOLD",
    # Structure numbering / features
    "compute_features_from_pdb",
    "compute_features_for_pdbs",
    "compute_features_for_pdb_directory",
    "identify_nanobody_chains_from_pdb",
    "identify_unique_nanobody_chains_from_pdb",
    # Structure classification
    "classify_structure",
    "classify_all_nanobodies_in_pdb",
    "DEFAULT_KINKED_THRESHOLD",
    "DEFAULT_EXTENDED_THRESHOLD",
    # Package metadata
    "__version__",
]
