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

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict, Union

from Bio.PDB import PDBParser, MMCIFParser, ShrakeRupley, vectors
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from .structure_quality import assess_structure_quality, is_usable_atom

from .structure_config import (
    CDR2_AHOS,
    CDR3_AHOS,
    CDR3_C_BREAK_AHO,
    CDR3_N_BREAK_AHO,
    CDR3_STEM_AHOS,
    CONTACT_RADIUS,
    CONTACT_SWITCH_WIDTH,
    USE_SOFT_CONTACTS,
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


AhoPosition = Union[int, str]


def _base_aho(position: AhoPosition) -> int:
    """Base position for region membership, retaining insertion keys elsewhere."""
    if isinstance(position, int):
        return position
    return int(position.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


def build_residues_by_aho(chain: Chain) -> Dict[AhoPosition, Residue]:
    """Map complete AHo identities to residues: 123, '123A', '123B', etc.

    Uninserted positions retain integer keys for compatibility. No insertion
    may replace its base residue or another insertion.
    """
    residues_by_aho = {}
    for residue in chain:
        if residue.id[0] == " ":
            _, number, insertion = residue.id
            insertion = insertion.strip()
            key = f"{number}{insertion}" if insertion else number
            residues_by_aho[key] = residue
    return residues_by_aho


def _heavy_atoms(residue):
    """Select by element, falling back to PDB names only when it is unknown."""
    def is_heavy(atom):
        element = (atom.element or "").strip().upper()
        if element not in ("", "X"):
            return element not in ("H", "D")
        name = atom.name.strip().upper().lstrip("0123456789")
        return not name.startswith(("H", "D"))
    return [atom for atom in residue if is_heavy(atom)]


def _compute_terminal_angles(chain, breakpoint):
    residues = [residue for residue in chain if residue.id[0] == " "]
    k = next((i for i, residue in enumerate(residues)
              if residue.id == (" ", breakpoint, " ")), None)
    if k is None or k < 1 or k + 2 >= len(residues):
        return None, None
    segment = residues[k - 1:k + 3]
    if any("CA" not in residue or not is_usable_atom(residue["CA"]) for residue in segment):
        return None, None
    # Consecutive resolved residues need not be covalently adjacent. Confirm
    # peptide connectivity without imposing consecutive AHo numbers (numbering
    # gaps are legitimate). Missing peptide atoms cannot establish continuity.
    for left, right in zip(segment, segment[1:]):
        if ("C" not in left or "N" not in right or
                not is_usable_atom(left["C"]) or not is_usable_atom(right["N"])):
            return None, None
        distance = float(left["C"] - right["N"])
        if not math.isfinite(distance) or not 0.8 <= distance <= 2.0:
            return None, None
        ca_distance = float(left["CA"] - right["CA"])
        if not math.isfinite(ca_distance) or not 2.5 <= ca_distance <= 4.5:
            return None, None
    points = [residue["CA"].get_vector() for residue in segment]
    alpha = math.degrees(float(vectors.calc_dihedral(*points)))
    tau = math.degrees(float(vectors.calc_angle(*points[:3])))
    if not math.isfinite(alpha) or not math.isfinite(tau):
        return None, None
    return alpha, tau


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
    return _compute_terminal_angles(chain, CDR3_C_BREAK_AHO)


class StructureFeatureDict(TypedDict, total=False):
    """Features returned by compute_structure_features().

    The classifier uses cos_alpha_N, tau_N, cos_alpha_C, tau_C, contact_nres,
    and fr2_rsa_key. Raw dihedrals and legacy contact_density are also returned
    for interpretation. The quality entry describes coordinate checks and
    invalidated measurements. High-level numbering APIs add framework_rmsd.

    Note: cdr2_length and cdr3_length were tested but NOT included in the
    classifier (see CDR Length Functions section for details).
    """
    alpha_N: Optional[float]
    tau_N: Optional[float]
    alpha_C: Optional[float]
    tau_C: Optional[float]
    cos_alpha_N: Optional[float]
    cos_alpha_C: Optional[float]
    contact_density: Optional[float]
    contact_nres: Optional[float]
    fr2_rsa_key: Optional[float]
    framework_rmsd: Optional[float]
    quality: dict
    # CDR lengths - available via compute_cdr2_length/compute_cdr3_length
    # but NOT included in classifier features (tested, did not improve model)


STRUCTURE_FEATURE_COLUMNS = tuple(StructureFeatureDict.__annotations__)


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
    return _compute_terminal_angles(chain, CDR3_N_BREAK_AHO)


# ---------------------------------------------------------------------------
# CDR3–FR2 contact helpers
# ---------------------------------------------------------------------------


def cdr3_fr2_pair_min_distances(
    residues_by_aho: Dict[AhoPosition, Residue],
    cdr3_ahos: Iterable[int],
) -> Tuple[List[float], int]:
    """
    Compute the minimum heavy-atom distance for every CDR3(non-stem)–FR2 residue pair.

    This is the geometric core shared by both the binary and soft contact
    density features. Returning raw per-pair minimum distances allows a soft
    switching function to be applied (or re-applied with different parameters)
    without re-parsing the structure.

    Parameters
    ----------
    residues_by_aho : Dict[AhoPosition, Residue]
        Mapping from AHo position to Residue object for the nanobody chain.
    cdr3_ahos : Iterable[int]
        AHo positions belonging to CDR3 (including stems).

    Returns
    -------
    (min_distances, cdr3_len_nonstem)
        ``min_distances`` is a list with one entry per CDR3(non-stem)–FR2
        residue pair where both residues are present (each entry is the minimum
        heavy-atom distance in Angstroms). ``cdr3_len_nonstem`` is the CDR3
        length EXCLUDING stems, restricted to residues present in the structure
        (the density denominator). This matches the definition the structure
        classifier was trained on.
    """
    cdr3_positions = set(cdr3_ahos) - set(CDR3_STEM_AHOS)
    cdr3 = [residue for pos, residue in residues_by_aho.items() if _base_aho(pos) in cdr3_positions]
    fr2 = [residue for pos, residue in residues_by_aho.items() if _base_aho(pos) in FR2_CONTACT_AHOS]
    min_distances = []
    for residue in cdr3:
        atoms = _heavy_atoms(residue)
        if not atoms:
            continue
        for partner in fr2:
            partner_atoms = _heavy_atoms(partner)
            if partner_atoms:
                min_distances.append(float(min(a - b for a in atoms for b in partner_atoms)))
    return min_distances, len(cdr3)


def cdr3_fr2_residue_min_distances(
    residues_by_aho: Dict[AhoPosition, Residue],
    cdr3_ahos: Iterable[int],
) -> Dict[AhoPosition, float]:
    """
    For each non-stem CDR3 residue, the minimum heavy-atom distance to FR2.

    Unlike :func:`cdr3_fr2_pair_min_distances` (which returns one entry per
    CDR3-FR2 residue *pair*), this collapses to one entry per CDR3 residue: the
    closest approach of that residue to *any* FR2 contact residue. This is the
    geometric core of the ``contact_nres`` feature (number of CDR3 residues that
    contact FR2), which - unlike the length-normalised ``contact_density`` - is
    not diluted by loop length and so faithfully captures the expert definition
    "any part of CDR3 contacting FR2 implies kinked".

    Returns
    -------
    Dict[int, float]
        Mapping ``{cdr3_aho: min_heavy_atom_distance_to_FR2}`` for non-stem CDR3
        residues that are present and have at least one FR2 partner present.
    """
    fr2_atoms = [atom for pos, residue in residues_by_aho.items()
                 if _base_aho(pos) in FR2_CONTACT_AHOS for atom in _heavy_atoms(residue)]
    if not fr2_atoms:
        return {}
    cdr3_positions = set(cdr3_ahos) - set(CDR3_STEM_AHOS)
    res_min = {}
    for pos, residue in residues_by_aho.items():
        if _base_aho(pos) not in cdr3_positions:
            continue
        atoms = _heavy_atoms(residue)
        if atoms:
            res_min[pos] = float(min(a - b for a in atoms for b in fr2_atoms))
    return res_min


def compute_cdr3_fr2_contact_nres(
    residues_by_aho: Dict[AhoPosition, Residue],
    cdr3_ahos: Iterable[int],
    *,
    soft: bool = True,
    switch_midpoint: float = CONTACT_RADIUS,
    switch_width: float = CONTACT_SWITCH_WIDTH,
) -> float:
    """
    Number of CDR3 residues in contact with FR2 (the ``contact_nres`` feature).

    Each non-stem CDR3 residue contributes based on its single closest approach
    to FR2:
        - ``soft=True`` (default): a fractional contribution given by the same
          logistic switch used for soft contacts, so the count is continuous and
          robust to small coordinate changes (MD/ensemble jitter).
        - ``soft=False``: an integer count of residues whose closest FR2 approach
          is within ``switch_midpoint``.

    This is NOT normalised by loop length. A localised contact (e.g. a single
    side chain reaching FR2 in an otherwise extended loop) therefore registers
    fully, which matches the expert kinked/extended definition far better than
    the length-averaged ``contact_density`` (see v0.3.0 model notes).

    Returns
    -------
    float
        Soft (or integer) count of contacting CDR3 residues.
    """
    res_min = cdr3_fr2_residue_min_distances(residues_by_aho, cdr3_ahos)
    if not res_min:
        return 0.0
    if soft:
        return float(
            sum(
                _logistic_contact_weight(d, switch_midpoint, switch_width)
                for d in res_min.values()
            )
        )
    return float(sum(1 for d in res_min.values() if d <= switch_midpoint))


def _logistic_contact_weight(distance: float, midpoint: float, width: float) -> float:
    """Smooth switching function mapping a distance to a contact weight in (0, 1).

    Returns ~1 when atoms are much closer than ``midpoint`` and ~0 when much
    farther, transitioning smoothly through 0.5 at ``distance == midpoint``.
    ``width`` controls the sharpness of the transition (smaller = sharper).
    """
    z = (distance - midpoint) / width
    if z > 50.0:
        return 0.0
    if z < -50.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def compute_cdr3_fr2_contacts(
    residues_by_aho: Dict[AhoPosition, Residue],
    cdr3_ahos: Iterable[int],
    *,
    soft: bool = False,
    switch_midpoint: float = CONTACT_RADIUS,
    switch_width: float = CONTACT_SWITCH_WIDTH,
) -> float:
    """
    Compute CDR3–FR2 contact density feature.

    Mirrors the logic of `calculate_contacts_from_pdb` in MAST-CDR-design,
    but operates directly on a residues_by_aho mapping.

    The contact density is calculated as:
        num_contacts / cdr3_length_nonstem

    Where:
        - num_contacts: contacts counted over CDR3 positions EXCLUDING stems
          (108, 109, 136, 137, 138), which are structurally constrained and
          don't contribute to the kinked/extended distinction.
        - cdr3_length_nonstem: CDR3 length EXCLUDING stems, restricted to
          residues present in the structure. This matches the definition the
          classifier was trained on.

    Two contact definitions are supported:
        - ``soft=False`` (default): a residue pair counts as 1 contact when its
          minimum heavy-atom distance is <= ``switch_midpoint``. This reproduces
          the original hard-cutoff behaviour exactly.
        - ``soft=True``: each residue pair contributes a fractional contact given
          by a logistic switching function of its minimum heavy-atom distance.
          This removes the discontinuity at the cutoff and makes the feature
          robust to small coordinate changes (e.g. MD/ensemble jitter).

    Parameters
    ----------
    residues_by_aho : Dict[AhoPosition, Residue]
        Mapping from AHo position to Residue object for the nanobody chain.
    cdr3_ahos : Iterable[int]
        AHo positions belonging to CDR3 (including stems).
    soft : bool, default False
        If True, use the logistic soft-contact definition.
    switch_midpoint : float, default CONTACT_RADIUS
        Distance (Angstroms) at which a pair counts as half a contact (soft) or
        the hard cutoff (binary).
    switch_width : float, default CONTACT_SWITCH_WIDTH
        Width (Angstroms) of the logistic transition (only used when soft=True).

    Returns
    -------
    float
        Contact density (sum of contacts / cdr3_length_nonstem).
    """
    min_distances, total_cdr3_len = cdr3_fr2_pair_min_distances(
        residues_by_aho, cdr3_ahos
    )

    if total_cdr3_len <= 0:
        return 0.0

    if soft:
        num_contacts = sum(
            _logistic_contact_weight(d, switch_midpoint, switch_width)
            for d in min_distances
        )
    else:
        num_contacts = float(sum(1 for d in min_distances if d <= switch_midpoint))

    return float(num_contacts) / float(total_cdr3_len)


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
    # The trained RSA feature uses heavy atoms. Work on a copy so added
    # hydrogens cannot change the feature or mutate the caller's structure.
    structure = structure.copy()
    for residue in structure.get_residues():
        keep = {atom.id for atom in _heavy_atoms(residue)}
        for atom in list(residue):
            if atom.id not in keep:
                residue.detach_child(atom.id)
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
        RSA over the FR2 key position(s) in ``FR2_KEY_RSA_AHOS``
        (AHo 44 as of v0.3.0; was 44 + 54 in v0.2.0) -> ``fr2_rsa_key``.
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
# CDR lengths remain excluded. The v0.3.0 classifier uses six features:
#   cos_alpha_N, tau_N, cos_alpha_C, tau_C, contact_nres, fr2_rsa_key
#
# These functions are retained for descriptive statistics and potential future use.
# ---------------------------------------------------------------------------


def compute_cdr2_length(residues_by_aho: Dict[AhoPosition, Residue]) -> int:
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
    return sum(1 for pos in residues_by_aho if _base_aho(pos) in CDR2_AHOS)


def compute_cdr3_length(residues_by_aho: Dict[AhoPosition, Residue]) -> int:
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
    return sum(1 for pos in residues_by_aho if _base_aho(pos) in CDR3_AHOS)


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
    High-level helper: compute structure features from an AHo-numbered file.

    Parameters
    ----------
    pdb_path : str
        Path to an AHo-numbered PDB or mmCIF file.
    chain_id : str, default 'H'
        Chain identifier containing the nanobody.
    seqid_for_log : str, optional
        Identifier used when loading the structure (for logging/debugging).

    Returns
    -------
    StructureFeatureDict
        The six classifier inputs plus raw alpha_N/alpha_C dihedrals and
        legacy contact_density. fr2_rsa_key measures RSA at AHo 44. CDR lengths
        are available through separate helpers; framework_rmsd is added by
        the higher-level numbering APIs. The quality entry contains coordinate
        checks; affected measurements are None when their inputs are incomplete.
    """
    structure = load_structure(pdb_path, seqid_for_log=seqid_for_log or "")
    chain = get_chain(structure, chain_id)
    residues_by_aho = build_residues_by_aho(chain)

    # Angles
    alpha_C, tau_C = compute_c_terminal_angles(chain)
    alpha_N, tau_N = compute_n_terminal_angles(chain)

    quality = assess_structure_quality(chain, {
        "alpha_N": alpha_N, "tau_N": tau_N, "alpha_C": alpha_C, "tau_C": tau_C,
    })
    invalid = set(quality["invalid_features"])

    # Contacts (soft logistic switch by default in v0.2.0+; see structure_config)
    contact_density = None if "contact_density" in invalid else compute_cdr3_fr2_contacts(
        residues_by_aho,
        CDR3_AHOS,
        soft=USE_SOFT_CONTACTS,
        switch_midpoint=CONTACT_RADIUS,
        switch_width=CONTACT_SWITCH_WIDTH,
    )

    # Number of CDR3 residues contacting FR2 (v0.3.0 classifier feature). Unlike
    # contact_density this is NOT normalised by loop length, so a localised
    # contact in a long loop is not diluted - matching the expert definition.
    contact_nres = None if "contact_nres" in invalid else compute_cdr3_fr2_contact_nres(
        residues_by_aho,
        CDR3_AHOS,
        soft=USE_SOFT_CONTACTS,
        switch_midpoint=CONTACT_RADIUS,
        switch_width=CONTACT_SWITCH_WIDTH,
    )

    # FR2 RSA (v0.3.0: key position 44 only; see structure_config.FR2_KEY_RSA_AHOS)
    fr2_rsa_key = None if "fr2_rsa_key" in invalid else compute_fr2_rsa(structure, chain_id)

    # Circular (cosine) encoding of the CDR3 dihedrals (v0.2.0+). alpha_N/alpha_C
    # are dihedral angles in (-180, 180]; feeding raw degrees to the linear model
    # is unstable when the angle sits near the +/-180 branch cut (a ~2 deg change
    # flips +179 <-> -179, i.e. a 358 deg jump). cos() is wraparound-safe and
    # captures the physically meaningful trans/gauche character. tau_N/tau_C are
    # bounded bond angles (no wrap) and are kept linear. The deployed classifier
    # is trained on these cosine features (see metadata feature_cols).
    cos_alpha_N = math.cos(math.radians(alpha_N)) if alpha_N is not None else None
    cos_alpha_C = math.cos(math.radians(alpha_C)) if alpha_C is not None else None

    # Note: CDR lengths are NOT included - use compute_cdr2_length/compute_cdr3_length
    # directly if needed for descriptive statistics

    features: StructureFeatureDict = {
        "alpha_N": alpha_N,
        "tau_N": tau_N,
        "alpha_C": alpha_C,
        "tau_C": tau_C,
        "cos_alpha_N": cos_alpha_N,
        "cos_alpha_C": cos_alpha_C,
        "contact_density": contact_density,
        "contact_nres": contact_nres,
        "fr2_rsa_key": fr2_rsa_key,
        "quality": quality,
    }

    for name in invalid:
        if name in features:
            features[name] = None
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
    "compute_cdr3_fr2_contact_nres",
    "cdr3_fr2_pair_min_distances",
    "cdr3_fr2_residue_min_distances",
    "compute_fr2_rsa",
    # CDR lengths (for descriptive statistics only, NOT in classifier)
    "compute_cdr2_length",
    "compute_cdr3_length",
    # Higher-level functions
    "calculate_framework_rmsd",
    "compute_structure_features",
]

