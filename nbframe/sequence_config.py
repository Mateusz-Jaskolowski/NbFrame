# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

"""
Configuration and constants for sequence-based classifier.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

#
# Default thresholds for sequence classifier labels
#
# These thresholds determine how the classifier assigns labels:
# - "kinked": P(kinked) > DEFAULT_SEQ_KINKED_THRESHOLD
# - "extended": P(kinked) < DEFAULT_SEQ_EXTENDED_THRESHOLD
# - "uncertain": otherwise (classifier is uncertain)
#

# Default thresholds (optimized for accuracy with good coverage)
# 78% coverage, 94.9% accuracy on confident predictions
DEFAULT_SEQ_KINKED_THRESHOLD: float = 0.70
DEFAULT_SEQ_EXTENDED_THRESHOLD: float = 0.40

#
# Uncertainty zone explanation (for CLI output)
#
UNCERTAINTY_EXPLANATION: str = (
    "Falls within uncertainty zone ({extended:.2f}–{kinked:.2f}). "
    "Consider structural analysis for higher confidence."
)

#
# Model file paths (relative to package data directory)
#
# LR-based model (2026-01-19, Soft Labels + Top 20 Combined Correlation)
# Trained on 829 structures with soft labels, using top 20 hallmarks
# Test: ROC-AUC 0.939, AP 0.956, Accuracy 86%, Pearson r 0.812, Agreement 86%
LR_MODEL_PKG_PATH: str = "data/sequence_classifier_lr_2026-01-19.joblib"
LR_METADATA_PKG_PATH: str = "data/sequence_classifier_metadata_2026-01-19.json"

#
# AHo alignment constants
#
# Expected length of AHo-aligned sequence
AHO_ALIGNED_LENGTH: int = 149

# CDR1 gap-fixing positions (0-indexed into the AHo-aligned sequence list)
# Used by sequence_align.py to consolidate/move gaps in the CDR-H1 region
CDR1_START_IDX: int = 23       # Start of CDR1 region in aligned list
CDR1_END_IDX: int = 42         # End of CDR1 region (exclusive) in aligned list
CDR1_GAP_TARGET_IDX: int = 34  # Expected gap position (AHo 35)

def assign_label(
    probability: float,
    kinked_threshold: float = DEFAULT_SEQ_KINKED_THRESHOLD,
    extended_threshold: float = DEFAULT_SEQ_EXTENDED_THRESHOLD,
) -> Tuple[str, Optional[float]]:
    """
    Assign a classification label based on probability thresholds.

    Parameters
    ----------
    probability : float
        P(kinked) probability from the classifier.
    kinked_threshold : float
        Probability above which to classify as "kinked".
    extended_threshold : float
        Probability below which to classify as "extended".

    Returns
    -------
    tuple of (str, float or None)
        (label, confidence) where label is "kinked", "extended", or
        "uncertain", and confidence is the relevant probability (or
        None if uncertain).
    """
    if probability > kinked_threshold:
        return "kinked", probability
    elif probability < extended_threshold:
        return "extended", 1.0 - probability
    else:
        return "uncertain", None


__all__ = [
    "DEFAULT_SEQ_KINKED_THRESHOLD",
    "DEFAULT_SEQ_EXTENDED_THRESHOLD",
    "UNCERTAINTY_EXPLANATION",
    "LR_MODEL_PKG_PATH",
    "LR_METADATA_PKG_PATH",
    "AHO_ALIGNED_LENGTH",
    "CDR1_START_IDX",
    "CDR1_END_IDX",
    "CDR1_GAP_TARGET_IDX",
    "assign_label",
]
