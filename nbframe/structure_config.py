# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

"""
Configuration and constants for structure-based feature calculation.
"""

from __future__ import annotations

from typing import Dict, List

#
# AHo-based region definitions
#

# FR2 region used for CDR3–FR2 contact counting (AHo numbering)
FR2_CONTACT_AHOS: List[int] = list(range(44, 56))  # 44–55 inclusive

# FR2 key RSA feature is computed over these positions together (AHo numbering).
FR2_KEY_RSA_AHOS: List[int] = [44, 54]

# CDR2 region (AHo numbering)
# Used for CDR2 length calculation (descriptive statistics only).
# NOTE: CDR2 length was tested as a classifier feature but showed no improvement.
# See structure_features.py for details.
CDR2_AHOS: List[int] = list(range(57, 70))  # 57–69 inclusive

# CDR3 region (AHo numbering)
# Full CDR3 span used for feature calculation.
CDR3_AHOS: List[int] = list(range(108, 139))  # 108–138 inclusive

# CDR3 stem residues to exclude from contact counting (AHo numbering)
CDR3_STEM_AHOS: List[int] = [108, 109, 136, 137, 138]

# CDR3 N- and C-terminal breakpoints (AHo numbering)
CDR3_N_BREAK_AHO: int = 108  # first CDR3 residue
CDR3_C_BREAK_AHO: int = 137  # equivalent to Chothia 100x

#
# Contact calculation parameters
#

# Distance threshold for defining an atom–atom contact (Angstroms)
CONTACT_RADIUS: float = 4.5

#
# Framework RMSD quality control
#

# Framework AHo positions for RMSD-based quality control
# Used to superimpose structures to a reference and filter outliers
# Positions: 1-24, 42-57, 69-106, 140-149 (inclusive)
FR_ALIGNMENT_AHOS: List[int] = (
    list(range(1, 25))
    + list(range(42, 58))
    + list(range(69, 107))
    + list(range(140, 150))
)

# Default RMSD threshold for quality control (Angstroms)
DEFAULT_RMSD_THRESHOLD: float = 2.0

#
# Default thresholds for structure classifier labels
#
# These thresholds determine how the classifier assigns labels:
# - "kinked": P(kinked) > DEFAULT_STRUCT_KINKED_THRESHOLD
# - "extended": P(kinked) < DEFAULT_STRUCT_EXTENDED_THRESHOLD
# - "uncertain": otherwise (classifier is uncertain)
#
# 90% coverage, 98.9% accuracy on confident predictions
DEFAULT_STRUCT_KINKED_THRESHOLD: float = 0.55
DEFAULT_STRUCT_EXTENDED_THRESHOLD: float = 0.25

# Uncertainty zone explanation (for CLI output)
STRUCT_UNCERTAINTY_EXPLANATION: str = (
    "Falls within uncertainty zone ({extended:.2f}–{kinked:.2f}). "
    "Structural features are ambiguous for this conformation."
)

# Minimum fraction of framework positions required for valid RMSD calculation
MIN_FRAMEWORK_COVERAGE: float = 0.8

# Path to reference VHH structure (relative to nbframe package data directory)
REFERENCE_VHH_PDB: str = "data/reference_VHH_PDB-2p45-chainB.pdb"

# Path to structure classifier metadata (relative to nbframe package data directory)
STRUCT_METADATA_PKG_PATH: str = "data/structure_classifier_metadata_2026-01-19.json"

#
# SASA / RSASA configuration
#

# Standard maximum SASA values (Å²) for amino acids, based on
# Tien et al., 2013 (Ala-X-Ala context, PMCID: PMC3836772).
# Keys are three-letter residue codes as used by Biopython (residue.resname).
MAX_SASA_VALUES: Dict[str, float] = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 204.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

__all__ = [
    "FR2_CONTACT_AHOS",
    "FR2_KEY_RSA_AHOS",
    "CDR2_AHOS",
    "CDR3_AHOS",
    "CDR3_STEM_AHOS",
    "CDR3_N_BREAK_AHO",
    "CDR3_C_BREAK_AHO",
    "CONTACT_RADIUS",
    "MAX_SASA_VALUES",
    "FR_ALIGNMENT_AHOS",
    "DEFAULT_RMSD_THRESHOLD",
    "MIN_FRAMEWORK_COVERAGE",
    "REFERENCE_VHH_PDB",
    "DEFAULT_STRUCT_KINKED_THRESHOLD",
    "DEFAULT_STRUCT_EXTENDED_THRESHOLD",
    "STRUCT_UNCERTAINTY_EXPLANATION",
    "STRUCT_METADATA_PKG_PATH",
]


