# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

"""
Helpers for computing structure-based features from AHo-numbered nanobody PDBs.

This module assumes that:
    - Input PDB files are already AHo-numbered.
    - The relevant nanobody chain is known (typically 'H').

It provides low-level parsing helpers plus higher-level feature calculators
for angles, contacts, and FR2 RSA/RSASA that will be used by the downstream
structure classifier.

Supports both PDB and mmCIF file formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict

from Bio.PDB import PDBParser, MMCIFParser, ShrakeRupley, vectors
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from .structure_config import (
    CDR2_AHOS,
    CDR3_AHOS,
    CDR3_C_BREAK_AHO,
    CDR3_N_BREAK_AHO,
    CDR3_STEM_AHOS,
    CONTACT_RADIUS,
    FR2_CONTACT_AHOS,
    FR2_KEY_RSA_AHOS,
    FR_ALIGNMENT_AHOS,
    MAX_SASA_VALUES,
    MIN_FRAMEWORK_COVERAGE,
)


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

# File extensions that indicate mmCIF format
MMCIF_EXTENSIONS = {".cif", ".mmcif"}


def _is_mmcif_file(file_path: str) -> bool:
    """Check if a file is in mmCIF format based on extension."""
    suffix = Path(file_path).suffix.lower()
    return suffix in MMCIF_EXTENSIONS


def _get_structure_parser(file_path: str):
    """
    Return the appropriate BioPython parser for the given file.

    Returns PDBParser for .pdb/.ent files, MMCIFParser for .cif/.mmcif files.
    """
    if _is_mmcif_file(file_path):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def load_structure(pdb_path: str, seqid_for_log: str = "") -> Structure:
    """
    Load an AHo-numbered structure file into a Biopython Structure object.

    Supports both PDB (.pdb, .ent) and mmCIF (.cif, .mmcif) formats.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB or mmCIF file.
    seqid_for_log : str, optional
        Identifier used in warning/error messages.
    """
    parser = _get_structure_parser(pdb_path)
    try:
        structure = parser.get_structure(seqid_for_log or "structure", pdb_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Structure file not found at {pdb_path!r}") from None
    return structure


def get_chain(structure: Structure, chain_id: str) -> Chain:
    """
    Retrieve a specific chain from the first model of the structure.

    Raises a KeyError with a clear message if the chain is missing.
    """
    model = structure[0]
    if chain_id not in model:
        available = ", ".join(ch.id for ch in model.get_chains())
        raise KeyError(
            f"Chain {chain_id!r} not found in structure. "
            f"Available chains: {available or 'none'}"
        )
    return model[chain_id]


def build_residues_by_aho(chain: Chain) -> Dict[int, Residue]:
    """
    Build a mapping from AHo residue number to Residue object for a chain.

    Assumes the PDB is already AHo-numbered, so residue.id[1] is the AHo position.
    Only standard residues (id[0] == ' ') are included.
    """
    residues_by_aho: Dict[int, Residue] = {}
    for residue in chain:
        if residue.id[0] == " ":
            aho_num = residue.id[1]
            residues_by_aho[aho_num] = residue
    return residues_by_aho


# ---------------------------------------------------------------------------
# Angle helpers (N-terminal and C-terminal CDR3 angles)
# ---------------------------------------------------------------------------


def compute_c_terminal_angles(chain: Chain) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute alpha and tau angles at the C-terminal end of CDR3.

    Mirrors the logic of `calculate_angles_c_terminal` in MAST-CDR-design:
        - Finds the residue with AHo number == CDR3_C_BREAK_AHO (137).
        - Uses chain-order neighbours: residues at indices k-1, k, k+1, k+2.
        - Uses CA atoms of those four residues.
        - alpha_C: dihedral(136, 137, 138, 139)
        - tau_C:   angle(136, 137, 138)
    """
    # Build ordered residue list for the chain
    list_res: List[Residue] = list(chain.get_residues())

    # Find index of the breakpoint residue by AHo number
    breakpoint_aho = CDR3_C_BREAK_AHO
    k: Optional[int] = None
    for idx, res in enumerate(list_res):
        res_id = res.get_id()
        aho_num = res_id[1]
        if aho_num == breakpoint_aho:
            k = idx
            break

    if k is None:
        return None, None

    # Ensure neighbours exist
    if not (0 <= k - 1 < len(list_res) and 0 <= k + 2 < len(list_res)):
        return None, None

    try:
        v_prev = list_res[k - 1]["CA"].get_vector()
        v_break = list_res[k]["CA"].get_vector()
        v_next = list_res[k + 1]["CA"].get_vector()
        v_next2 = list_res[k + 2]["CA"].get_vector()
    except KeyError:
        # Missing CA atom in one of the residues
        return None, None

    alpha = float(
        vectors.calc_dihedral(v_prev, v_break, v_next, v_next2) * 180.0 / 3.141592653589793
    )
    tau = float(vectors.calc_angle(v_prev, v_break, v_next) * 180.0 / 3.141592653589793)

    return alpha, tau


class StructureFeatureDict(TypedDict, total=False):
    """Features returned by compute_structure_features().

    The 6 core features used by the Structure Classifier:
    - alpha_N, tau_N: N-terminal CDR3 angles
    - alpha_C, tau_C: C-terminal CDR3 angles
    - contact_density: CDR3-FR2 contacts normalized by CDR3 length
    - fr2_rsa_key: Relative solvent accessibility at FR2 key positions

    Note: cdr2_length and cdr3_length were tested but NOT included in the
    classifier (see CDR Length Functions section for details).
    """
    alpha_N: Optional[float]
    tau_N: Optional[float]
    alpha_C: Optional[float]
    tau_C: Optional[float]
    contact_density: float
    fr2_rsa_key: Optional[float]
    framework_rmsd: Optional[float]
    # CDR lengths - available via compute_cdr2_length/compute_cdr3_length
    # but NOT included in classifier features (tested, did not improve model)


def compute_n_terminal_angles(chain: Chain) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute alpha and tau angles at the N-terminal end of CDR3.

    Mirrors the logic of `calculate_angles_n_terminal` in MAST-CDR-design:
        - Uses AHo 108 as the N-terminal breakpoint (first CDR3 residue).
        - Finds the residue with that AHo number and uses chain-order
          neighbours at indices k-1, k, k+1, k+2.
        - Uses CA atoms of those four residues.
        - alpha_N: dihedral(107, 108, 109, 110)
        - tau_N:   angle(107, 108, 109)
    """
    list_res: List[Residue] = list(chain.get_residues())

    breakpoint_aho = CDR3_N_BREAK_AHO
    k: Optional[int] = None
    for idx, res in enumerate(list_res):
        res_id = res.get_id()
        aho_num = res_id[1]
        if aho_num == breakpoint_aho:
            k = idx
            break

    if k is None:
        return None, None

    if not (0 <= k - 1 < len(list_res) and 0 <= k + 2 < len(list_res)):
        return None, None

    try:
        v_prev = list_res[k - 1]["CA"].get_vector()
        v_break = list_res[k]["CA"].get_vector()
        v_next = list_res[k + 1]["CA"].get_vector()
        v_next2 = list_res[k + 2]["CA"].get_vector()
    except KeyError:
        return None, None

    alpha = float(
        vectors.calc_dihedral(v_prev, v_break, v_next, v_next2) * 180.0 / 3.141592653589793
    )
    tau = float(vectors.calc_angle(v_prev, v_break, v_next) * 180.0 / 3.141592653589793)

    return alpha, tau


# ---------------------------------------------------------------------------
# CDR3–FR2 contact helpers
# ---------------------------------------------------------------------------


def compute_cdr3_fr2_contacts(
    residues_by_aho: Dict[int, Residue],
    cdr3_ahos: Iterable[int],
) -> float:
    """
    Compute CDR3–FR2 contact density feature.

    Mirrors the logic of `calculate_contacts_from_pdb` in MAST-CDR-design,
    but operates directly on a residues_by_aho mapping.

    The contact density is calculated as:
        num_contacts / total_cdr3_length

    Where:
        - num_contacts: Number of CDR3-FR2 contacts, counted from CDR3 positions
          EXCLUDING stems (108, 109, 136, 137, 138) since stems are structurally
          constrained and don't contribute to the kinked/extended distinction.
        - total_cdr3_length: FULL CDR3 length INCLUDING stems (positions 108-138
          that are present in the structure).

    Parameters
    ----------
    residues_by_aho : Dict[int, Residue]
        Mapping from AHo position to Residue object for the nanobody chain.
    cdr3_ahos : Iterable[int]
        AHo positions belonging to CDR3 (including stems).

    Returns
    -------
    float
        Contact density (num_contacts / total_cdr3_length).
    """
    # Define regions
    fr2_positions = FR2_CONTACT_AHOS
    # For contact counting, exclude stems (they're structurally constrained)
    cdr3_contact_positions = [aho for aho in cdr3_ahos if aho not in CDR3_STEM_AHOS]
    # For length calculation, include ALL CDR3 positions
    cdr3_all_positions = list(cdr3_ahos)

    num_contacts = 0
    contacts_found = set()  # type: ignore[var-annotated]

    for cdr3_aho in cdr3_contact_positions:
        cdr3_res = residues_by_aho.get(cdr3_aho)
        if cdr3_res is None:
            continue

        for fr_aho in fr2_positions:
            fr_res = residues_by_aho.get(fr_aho)
            if fr_res is None:
                continue

            contact_pair = (cdr3_aho, fr_aho)
            if contact_pair in contacts_found:
                continue

            is_contact = False
            for cdr3_atom in cdr3_res:
                if cdr3_atom.name.startswith("H"):
                    continue
                for fr_atom in fr_res:
                    if fr_atom.name.startswith("H"):
                        continue
                    if cdr3_atom - fr_atom <= CONTACT_RADIUS:
                        is_contact = True
                        contacts_found.add(contact_pair)
                        num_contacts += 1
                        break
                if is_contact:
                    break

    # CDR3 length INCLUDING stems, restricted to residues present in the structure
    total_cdr3_len = len(
        [aho for aho in cdr3_all_positions if aho in residues_by_aho]
    )
    contact_density = (
        float(num_contacts) / float(total_cdr3_len)
        if total_cdr3_len > 0
        else 0.0
    )

    return contact_density


# ---------------------------------------------------------------------------
# FR2 RSA / RSASA helpers
# ---------------------------------------------------------------------------


def _compute_region_rsa(
    structure: Structure,
    chain_id: str,
    residue_ids_to_check: Iterable[int],
) -> Optional[float]:
    """
    Compute summed RSA for a specific set of residue numbers in a chain.

    RSA is defined as:
        (sum actual SASA) / (sum max SASA) over the selected residues.

    Returns None if no residues could be processed or if max SASA is zero.
    """
    model = structure[0]
    if chain_id not in model:
        return None

    chain = model[chain_id]
    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    total_actual_sasa = 0.0
    total_max_sasa = 0.0
    found_residues = 0

    for res_id in residue_ids_to_check:
        key = (" ", res_id, " ")
        if key not in chain:
            continue
        residue = chain[key]
        max_sasa = MAX_SASA_VALUES.get(residue.resname)
        if max_sasa is None:
            continue
        sasa = getattr(residue, "sasa", None)
        if sasa is None:
            continue
        total_actual_sasa += float(sasa)
        total_max_sasa += float(max_sasa)
        found_residues += 1

    if found_residues == 0 or total_max_sasa <= 0.0:
        return None

    return float(total_actual_sasa / total_max_sasa)


def compute_fr2_rsa(
    structure: Structure,
    chain_id: str,
) -> Optional[float]:
    """
    Compute RSA-based feature for FR2 key positions.

    Returns
    -------
    float or None
        RSA over AHo positions 44 and 54 together (fr2_rsa_key).
    """
    return _compute_region_rsa(structure, chain_id, FR2_KEY_RSA_AHOS)


# ---------------------------------------------------------------------------
# CDR length helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CDR Length Functions (EXPERIMENTAL - NOT USED IN CLASSIFIER)
# ---------------------------------------------------------------------------
# NOTE (January 2026): These CDR length functions were implemented and tested
# as potential features for the Structure Classifier. Extensive evaluation showed:
#
# - CDR2 length: Statistically different between K/E (p=0.0001) but narrow
#   distribution (95% are 9-10). Adding it to the model showed no improvement.
#
# - CDR3 length: Correlated with Kinked class (r=0.43) but is a CONSEQUENCE
#   of kinked conformation, not a cause. Including it in GMM clustering caused
#   misclassifications (high contact_density structures with short CDR3 were
#   incorrectly clustered as Extended). In supervised training, CDR3 length
#   had a surprising NEGATIVE coefficient (-0.028), suggesting it doesn't help.
#
# DECISION: The Structure Classifier uses only the original 6 features:
#   alpha_N, tau_N, alpha_C, tau_C, contact_density, fr2_rsa_key
#
# These functions are retained for descriptive statistics and potential future use.
# ---------------------------------------------------------------------------


def compute_cdr2_length(residues_by_aho: Dict[int, Residue]) -> int:
    """
    Compute the number of resolved CDR2 residues.

    NOTE: This feature was tested but NOT included in the final Structure
    Classifier model (see note above). Available for descriptive statistics.

    Parameters
    ----------
    residues_by_aho : dict
        Mapping from AHo position to Residue object.

    Returns
    -------
    int
        Number of residues present in the structure for CDR2 (AHo 57-69).
    """
    return sum(1 for aho in CDR2_AHOS if aho in residues_by_aho)


def compute_cdr3_length(residues_by_aho: Dict[int, Residue]) -> int:
    """
    Compute the number of resolved CDR3 residues (including stems).

    NOTE: This feature was tested but NOT included in the final Structure
    Classifier model - it caused misclassifications in clustering and had
    a negative coefficient in supervised training. See note above.

    Counts all CDR3 positions (AHo 108-138) that are present in the structure.

    Parameters
    ----------
    residues_by_aho : dict
        Mapping from AHo position to Residue object.

    Returns
    -------
    int
        Number of CDR3 residues present in the structure (AHo 108-138).
    """
    return sum(1 for aho in CDR3_AHOS if aho in residues_by_aho)


# ---------------------------------------------------------------------------
# Framework RMSD calculation
# ---------------------------------------------------------------------------


def calculate_framework_rmsd(
    target_structure: Structure,
    target_chain_id: str,
    reference_structure: Structure,
    reference_chain_id: str,
    framework_ahos: Optional[List[int]] = None,
    min_coverage: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate framework RMSD between target and reference AHo-numbered structures.

    Uses CA atoms at specified framework AHo positions. Only positions present
    in both structures are used for alignment.

    Parameters
    ----------
    target_structure
        Biopython Structure object for the target (AHo-numbered).
    target_chain_id
        Chain identifier in target structure.
    reference_structure
        Biopython Structure object for the reference (AHo-numbered).
    reference_chain_id
        Chain identifier in reference structure.
    framework_ahos
        List of AHo positions to use for alignment. Defaults to FR_ALIGNMENT_AHOS.
    min_coverage
        Minimum fraction of framework positions required for valid RMSD.
        If fewer positions match, returns None. Defaults to MIN_FRAMEWORK_COVERAGE.

    Returns
    -------
    float or None
        RMSD in Angstroms, or None if insufficient matching positions.
    """
    from Bio.PDB import Superimposer

    if framework_ahos is None:
        framework_ahos = FR_ALIGNMENT_AHOS
    if min_coverage is None:
        min_coverage = MIN_FRAMEWORK_COVERAGE

    # Get chains
    try:
        target_chain = get_chain(target_structure, target_chain_id)
        reference_chain = get_chain(reference_structure, reference_chain_id)
    except KeyError:
        return None

    # Build residue mappings
    target_residues = build_residues_by_aho(target_chain)
    reference_residues = build_residues_by_aho(reference_chain)

    # Collect CA atoms at matching framework positions
    target_atoms = []
    reference_atoms = []
    matched_positions = 0

    for aho_pos in framework_ahos:
        target_res = target_residues.get(aho_pos)
        ref_res = reference_residues.get(aho_pos)

        if target_res is None or ref_res is None:
            continue

        # Check for CA atoms
        if "CA" not in target_res or "CA" not in ref_res:
            continue

        target_atoms.append(target_res["CA"])
        reference_atoms.append(ref_res["CA"])
        matched_positions += 1

    # Check coverage
    total_framework_positions = len(framework_ahos)
    coverage = matched_positions / total_framework_positions if total_framework_positions > 0 else 0.0

    if coverage < min_coverage:
        return None

    if len(target_atoms) < 3:
        # Need at least 3 atoms for superposition
        return None

    # Perform superposition and get RMSD
    super_imposer = Superimposer()
    super_imposer.set_atoms(reference_atoms, target_atoms)

    return float(super_imposer.rms)


# ---------------------------------------------------------------------------
# High-level feature entry point
# ---------------------------------------------------------------------------


def compute_structure_features(
    pdb_path: str,
    chain_id: str = "H",
    seqid_for_log: Optional[str] = None,
) -> StructureFeatureDict:
    """
    High-level helper: compute all structure-based features for a single PDB.

    Parameters
    ----------
    pdb_path : str
        Path to an AHo-numbered PDB file.
    chain_id : str, default 'H'
        Chain identifier containing the nanobody.
    seqid_for_log : str, optional
        Identifier used when loading the structure (for logging/debugging).

    Returns
    -------
    StructureFeatureDict
        Dictionary with 6 features:
        - alpha_N, tau_N: N-terminal CDR3 angles
        - alpha_C, tau_C: C-terminal CDR3 angles
        - contact_density: CDR3-FR2 contact density
        - fr2_rsa_key: RSA at FR2 key positions (44, 54)
        - cdr2_length: Number of resolved CDR2 residues
        - cdr3_length: Number of resolved CDR3 residues (excluding stems)
    """
    structure = load_structure(pdb_path, seqid_for_log=seqid_for_log or "")
    chain = get_chain(structure, chain_id)
    residues_by_aho = build_residues_by_aho(chain)

    # Angles
    alpha_C, tau_C = compute_c_terminal_angles(chain)
    alpha_N, tau_N = compute_n_terminal_angles(chain)

    # Contacts
    contact_density = compute_cdr3_fr2_contacts(residues_by_aho, CDR3_AHOS)

    # FR2 RSA
    fr2_rsa_key = compute_fr2_rsa(structure, chain_id)

    # Note: CDR lengths are NOT included - use compute_cdr2_length/compute_cdr3_length
    # directly if needed for descriptive statistics

    features: StructureFeatureDict = {
        "alpha_N": alpha_N,
        "tau_N": tau_N,
        "alpha_C": alpha_C,
        "tau_C": tau_C,
        "contact_density": contact_density,
        "fr2_rsa_key": fr2_rsa_key,
    }

    return features


__all__ = [
    # Core structure loading
    "load_structure",
    "get_chain",
    "build_residues_by_aho",
    # Feature computation (6 features used in classifier)
    "compute_c_terminal_angles",
    "compute_n_terminal_angles",
    "compute_cdr3_fr2_contacts",
    "compute_fr2_rsa",
    # CDR lengths (for descriptive statistics only, NOT in classifier)
    "compute_cdr2_length",
    "compute_cdr3_length",
    # Higher-level functions
    "calculate_framework_rmsd",
    "compute_structure_features",
]


