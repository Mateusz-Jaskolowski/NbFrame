# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

"""
Helpers for going from raw nanobody PDBs to AHo-numbered structures.

This module provides:

  - PDB parsing and per-chain sequence extraction.
  - Nanobody (VHH-like) chain identification using ANARCI (AHo scheme).
  - Construction of PDB→AHo residue mappings.
  - A minimal renumbering helper that writes an AHo-numbered PDB.
  - High-level helpers that feed AHo-numbered PDBs into
    :mod:`nbframe.structure_features` to compute structural metrics.

The public entry points are:

  - :func:`identify_nanobody_chain_from_pdb`
  - :func:`compute_features_from_pdb`
  - :func:`compute_features_for_pdbs`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import sys
import tempfile

import anarci
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.Data.IUPACData import protein_letters_3to1

from importlib import resources

from .sequence_align import AnarciChainResult, run_anarci_for_chains
from .structure_config import (
    DEFAULT_RMSD_THRESHOLD,
    REFERENCE_VHH_PDB,
)
from .structure_features import (
    StructureFeatureDict,
    _get_structure_parser,
    _is_mmcif_file,
    calculate_framework_rmsd,
    compute_structure_features,
    load_structure,
)

if TYPE_CHECKING:  # pragma: no cover - optional dependency for tabular helpers
    import pandas as pd


# ---------------------------------------------------------------------------
# PDB parsing helpers
# ---------------------------------------------------------------------------


@dataclass
class ChainInfo:
    """
    Minimal representation of a single chain in a PDB model.

    sequence
        Amino-acid sequence derived from standard residues only.
    residue_ids
        One entry per sequence position, giving the original PDB residue id
        tuple (hetfield, resseq, icode).
    """

    sequence: str
    residue_ids: List[Tuple[str, int, str]]


@dataclass
class ChainClassification:
    """
    Simple ANARCI-based classification of a chain in a PDB.

    chain_type
        Chain type as reported by ANARCI ('H', 'L', 'K', etc.), or None if
        ANARCI failed to classify the sequence.
    kind
        High-level interpretation:
            - 'VHH'  : heavy-only, VHH-like nanobody candidate
            - 'VH'   : heavy chain that is part of a VH/VL or Fab
            - 'VL'   : light chain
            - 'OTHER': anything else / unclassified
    is_vhh
        Convenience flag: True iff kind == 'VHH'.
    length
        Sequence length (number of residues) used for heuristics.
    """

    chain_type: Optional[str]
    kind: str
    is_vhh: bool
    length: int


def _three_to_one(resname: str) -> str:
    """
    Minimal three-letter to one-letter AA conversion, compatible with older
    Biopython versions where Bio.PDB.Polypeptide.three_to_one is unavailable.

    Bio.Data.IUPACData.protein_letters_3to1 typically uses title-case or
    mixed-case keys (e.g. 'Glu', 'ALA'). Here we try a few reasonable
    normalisations of the incoming residue name (often upper-case like 'GLU').
    """
    raw = (resname or "").strip()
    candidates = {raw, raw.capitalize(), raw.title(), raw.upper(), raw.lower()}
    for key in candidates:
        if key in protein_letters_3to1:
            return protein_letters_3to1[key]
    raise KeyError(resname)


def parse_pdb_chains(pdb_path: str) -> Dict[str, ChainInfo]:
    """
    Parse a PDB or mmCIF file and extract per-chain sequences and residue-id mappings.

    Supports both PDB (.pdb, .ent) and mmCIF (.cif, .mmcif) formats.

    Only standard amino-acid residues (``is_aa(..., standard=True)`` and
    ``hetfield == ' '``) are included. Non-standard residues, waters, and
    ligands are ignored.
    """
    parser = _get_structure_parser(pdb_path)
    structure = parser.get_structure("nbframe_structure", pdb_path)

    # Use first model only – nanobody structures are almost always single-model.
    model: Model = structure[0]

    chains: Dict[str, ChainInfo] = {}
    for chain in model:
        seq_chars: List[str] = []
        residue_ids: List[Tuple[str, int, str]] = []

        for residue in chain:
            if not isinstance(residue, Residue):
                continue

            hetfield, resseq, icode = residue.id
            if hetfield != " ":
                continue

            # Treat any residue with a recognised 3-letter AA code as protein.
            try:
                aa = _three_to_one(residue.resname)
            except KeyError:
                # Unknown residue type – skip it entirely
                continue

            seq_chars.append(aa)
            # Normalise insertion code to a string (Biopython can use ' ' or '')
            icode_str = icode if isinstance(icode, str) else str(icode or " ")
            residue_ids.append((hetfield, resseq, icode_str))

        if seq_chars:
            chains[chain.id] = ChainInfo("".join(seq_chars), residue_ids)

    return chains


# ---------------------------------------------------------------------------
# Nanobody chain identification
# ---------------------------------------------------------------------------


def _anarci_chain_type(sequence: str) -> Optional[str]:
    """
    Run ANARCI on a single sequence and return its chain type ('H', 'L', 'K', ...).

    Returns None if ANARCI fails to number the sequence.
    """
    if not sequence:
        return None
    try:
        numbering_tuple = anarci.number(sequence, scheme="aho")
    except Exception:
        return None
    if numbering_tuple is None:
        return None
    numbering_list, chain_type = numbering_tuple
    if not numbering_list:
        return None
    return chain_type


def _classify_chains_with_anarci(
    chains: Dict[str, ChainInfo]
) -> Dict[str, ChainClassification]:
    """
    Classify each chain as VHH / VH / VL / OTHER using ANARCI.

    Heuristic:
      - First, get ANARCI chain_type for each sequence.
      - If at least one light chain ('L' or 'K') is present, treat all heavy
        chains as part of VH/VL or Fab ('VH'), not VHH.
      - If no light chains are present, treat heavy chains ('H') with length
        between 90 and 150 residues as VHH-like ('VHH').
    """
    types: Dict[str, Optional[str]] = {}
    for cid, info in chains.items():
        types[cid] = _anarci_chain_type(info.sequence)

    has_light = any(
        t in ("L", "K") for t in types.values() if t is not None
    )

    annotations: Dict[str, ChainClassification] = {}
    for cid, info in chains.items():
        chain_type = types[cid]
        length = len(info.sequence)

        kind = "OTHER"
        is_vhh = False

        if chain_type == "H":
            if not has_light and 90 <= length <= 150:
                kind = "VHH"
                is_vhh = True
            else:
                kind = "VH"
        elif chain_type in ("L", "K"):
            kind = "VL"

        annotations[cid] = ChainClassification(
            chain_type=chain_type,
            kind=kind,
            is_vhh=is_vhh,
            length=length,
        )

    return annotations


def identify_nanobody_chains(
    chains: Dict[str, ChainInfo]
) -> List[str]:
    """
    Return all chain IDs that look like standalone VHH-like nanobodies.

    A chain is considered VHH-like if:
      - ANARCI classifies it as heavy ('H'),
      - There are no light chains ('L' or 'K') in the structure, and
      - Its length is between 90 and 150 residues.
    """
    annotations = _classify_chains_with_anarci(chains)
    return [cid for cid, ann in annotations.items() if ann.is_vhh]


def identify_nanobody_chains_from_pdb(pdb_path: str) -> List[str]:
    """
    Convenience wrapper: PDB path → list of VHH-like chain IDs.
    """
    chains = parse_pdb_chains(pdb_path)
    return identify_nanobody_chains(chains)


def identify_unique_nanobody_chains(
    chains: Dict[str, ChainInfo]
) -> List[str]:
    """
    Return one representative VHH-like chain ID per unique nanobody sequence.

    Behavior
    --------
    1. Start from :func:`identify_nanobody_chains` to get all VHH-like chains.
    2. Group chains by exact amino-acid sequence.
    3. For each group, pick the lexicographically smallest chain ID as the
       representative (all chains in a group share the same sequence, so
       sequence length is identical).
    """
    vhh_ids = identify_nanobody_chains(chains)
    if not vhh_ids:
        return []

    # Group by sequence
    groups: Dict[str, List[str]] = {}
    for cid in vhh_ids:
        seq = chains[cid].sequence
        groups.setdefault(seq, []).append(cid)

    selected: List[str] = []
    for seq, cids in groups.items():
        selected.append(min(cids))

    return selected


def identify_unique_nanobody_chains_from_pdb(pdb_path: str) -> List[str]:
    """
    Convenience wrapper: PDB path → representative VHH chain IDs per unique
    nanobody sequence.
    """
    chains = parse_pdb_chains(pdb_path)
    return identify_unique_nanobody_chains(chains)


def identify_nanobody_chain(
    chains: Dict[str, ChainInfo], chain_hint: Optional[str] = None
) -> str:
    """
    Identify the most likely nanobody (VHH-like) chain among parsed chains.

    Heuristics:
      1. If ``chain_hint`` is provided, validate and require it to number as
         a heavy chain with ANARCI (AHo).
      2. Otherwise:
         - First, restrict to chains with length between 90 and 150 residues.
         - Run ANARCI on these candidates and prefer heavy chains with the
           longest effective domain.
         - If no candidate is found, fall back to running ANARCI on *all*
           protein chains.
      3. If no heavy-like chain can be identified, raise a ValueError.
    """
    if not chains:
        raise ValueError("No protein chains found in PDB – cannot identify nanobody.")

    # Explicit hint: only validate existence and basic length – assume caller
    # knows that the hinted chain is a suitable nanobody candidate.
    if chain_hint is not None:
        if chain_hint not in chains:
            raise ValueError(
                f"Chain hint {chain_hint!r} not found in PDB chains "
                f"({', '.join(sorted(chains))})."
            )
        if len(chains[chain_hint].sequence) < 80:
            raise ValueError(
                f"Chain {chain_hint!r} sequence too short "
                f"({len(chains[chain_hint].sequence)} residues) to be a VHH."
            )

        return chain_hint

    # No hint: identify all VHH-like heavy chains and pick the longest one.
    vhh_ids = identify_nanobody_chains(chains)
    if not vhh_ids:
        raise ValueError(
            "Could not identify a VHH-like heavy chain in the PDB: "
            "structure appears to contain VH/VL or other chains but no "
            "standalone VHH."
        )

    return max(vhh_ids, key=lambda cid: len(chains[cid].sequence))


def identify_nanobody_chain_from_pdb(
    pdb_path: str, chain_hint: Optional[str] = None
) -> Optional[str]:
    """
    Convenience wrapper around :func:`identify_nanobody_chain` that starts from
    a PDB path.

    If no VHH-like chain can be identified, this returns None instead of
    raising, and emits a warning to stderr. This makes it safer to use in
    batch workflows where some PDBs are known not to contain nanobodies
    (e.g. Fab or VH/VL structures).
    """
    chains = parse_pdb_chains(pdb_path)
    try:
        return identify_nanobody_chain(chains, chain_hint=chain_hint)
    except ValueError as exc:
        sys.stderr.write(
            f"[nbframe] WARNING: {exc} for PDB {pdb_path!r}; returning None.\n"
        )
        return None


# ---------------------------------------------------------------------------
# PDB → AHo mapping helpers
# ---------------------------------------------------------------------------


def _build_pdb_to_aho_mapping(
    chain_info: ChainInfo, anarci_result: AnarciChainResult
) -> Dict[str, Union[int, str]]:
    """
    Construct a mapping from PDB residue identifiers (e.g. \"42\", \"100A\")
    to AHo positions for a single chain.

    This assumes:
      - ``chain_info.sequence`` is the ungapped sequence used as ANARCI input.
      - ``anarci_result.positions`` is the AHo-aligned sequence including gaps.
    """
    sequence = chain_info.sequence
    residue_ids = chain_info.residue_ids

    positions = anarci_result.get("positions") or []
    if not positions:
        return {}

    # Build the ungapped domain sequence from the ANARCI-aligned positions.
    aligned_aas = [pos.get("aa", "") for pos in positions]
    domain_seq = "".join(aa for aa in aligned_aas if aa not in ("", "-"))
    if not domain_seq:
        return {}

    # Find where this domain sits within the full chain sequence.
    start_idx = sequence.find(domain_seq)
    if start_idx == -1:
        # Fallback: if direct substring search fails, we cannot safely map
        # sequence indices back to PDB residues.
        raise ValueError(
            "Failed to align ANARCI-numbered domain to PDB chain sequence; "
            "cannot build PDB→AHo mapping."
        )

    pdb_to_aho: Dict[str, Union[int, str]] = {}

    seq_idx = start_idx
    for pos in positions:
        aa = pos.get("aa", "")
        aho_label = pos.get("aho_label")

        # Gaps in the alignment do not correspond to actual residues.
        if aa in ("", "-"):
            continue

        if seq_idx >= len(sequence):
            break

        # Sanity check – ensure AA from ANARCI matches PDB-derived sequence.
        if sequence[seq_idx] != aa:
            # Mismatch – this should be rare; bail out rather than creating
            # a misleading mapping.
            raise ValueError(
                f"Sequence mismatch between PDB ({sequence[seq_idx]!r}) and "
                f"ANARCI ({aa!r}) at sequence index {seq_idx}."
            )

        hetfield, resseq, icode = residue_ids[seq_idx]
        if hetfield != " ":
            seq_idx += 1
            continue

        icode_clean = (icode or "").strip()
        if icode_clean:
            pdb_key = f"{resseq}{icode_clean}"
        else:
            pdb_key = str(resseq)

        if aho_label is not None:
            pdb_to_aho[pdb_key] = aho_label

        seq_idx += 1

    return pdb_to_aho


# ---------------------------------------------------------------------------
# Renumbering helper
# ---------------------------------------------------------------------------


def _parse_aho_label(
    aho_label: Union[int, str]
) -> Tuple[int, str]:
    """
    Split an AHo label into (resseq, insertion code).

    Examples
    --------
    35        -> (35, ' ')
    \"100A\"  -> (100, 'A')
    """
    if isinstance(aho_label, int):
        return aho_label, " "

    s = str(aho_label)
    digits = "".join(ch for ch in s if ch.isdigit())
    letters = "".join(ch for ch in s if ch.isalpha())
    if not digits:
        raise ValueError(f"Cannot parse AHo label {aho_label!r} into an integer + insertion code.")
    return int(digits), (letters or " ")


def renumber_structure_to_aho(
    structure: Structure,
    chain_id: str,
    pdb_to_aho: Dict[str, Union[int, str]],
    *,
    temp_dir: Optional[Path] = None,
    basename: Optional[str] = None,
    ) -> Tuple[Structure, Path]:
    """
    Create a new structure containing a single AHo-numbered H-chain.

    Parameters
    ----------
    structure
        Original Biopython Structure loaded from the raw PDB.
    chain_id
        Chain identifier corresponding to the nanobody.
    pdb_to_aho
        Mapping from PDB residue identifiers (e.g. \"42\", \"100A\") to
        AHo labels (ints or strings).
    temp_dir
        Directory in which to write the renumbered PDB file. If None, a
        temporary directory will be created.
    basename
        Basename for the output PDB file. If None, derived from structure.id.

    Returns
    -------
    (new_structure, pdb_path)
        The AHo-numbered Structure and the path to the written PDB file.
    """
    model: Model = structure[0]
    if chain_id not in model:
        raise KeyError(
            f"Chain {chain_id!r} not found in structure; available chains: "
            f"{', '.join(ch.id for ch in model.get_chains()) or 'none'}"
        )

    original_chain = model[chain_id]

    new_model = Model(0)
    # Preserve the original chain identifier so that AHo-numbered PDBs keep
    # the same chain ID as the input structure (useful for downstream tools
    # and for user-facing AHo PDB exports).
    new_chain = Chain(chain_id)

    for residue in original_chain:
        if not isinstance(residue, Residue):
            continue
        hetfield, resseq, icode = residue.id
        if hetfield != " ":
            continue

        icode_clean = (icode or "").strip()
        if icode_clean:
            pdb_key = f"{resseq}{icode_clean}"
        else:
            pdb_key = str(resseq)

        if pdb_key not in pdb_to_aho:
            continue

        aho_label = pdb_to_aho[pdb_key]
        new_resseq, new_icode = _parse_aho_label(aho_label)
        new_res_id = (hetfield, new_resseq, new_icode)

        new_residue = Residue(new_res_id, residue.get_resname(), residue.get_segid())
        for atom in residue:
            new_residue.add(atom.copy())
        new_chain.add(new_residue)

    new_model.add(new_chain)
    new_structure = Structure(structure.id + "_aho")
    new_structure.add(new_model)

    if temp_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="nbframe_aho_")
        temp_dir_path = Path(tmp.name)
    else:
        temp_dir_path = Path(temp_dir)
        temp_dir_path.mkdir(parents=True, exist_ok=True)

    if basename is None:
        basename = f"{structure.id}_aho.pdb"

    pdb_path = temp_dir_path / basename
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(str(pdb_path))

    return new_structure, pdb_path


# ---------------------------------------------------------------------------
# Batch context and high-level feature APIs
# ---------------------------------------------------------------------------


class RenumberingBatchContext:
    """
    Context manager that owns a temporary directory for AHo-numbered PDBs.

    When used with ``save_pdb=False`` (the default in the public APIs), the
    temporary directory and all files within it are cleaned up automatically
    when the context exits.
    """

    def __init__(self, keep_files: bool = False, base_dir: Optional[Path] = None):
        self.keep_files = keep_files
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self.temp_dir: Optional[Path] = None

    def __enter__(self) -> "RenumberingBatchContext":
        if self.keep_files:
            self.temp_dir = self.base_dir or Path.cwd()
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="nbframe_aho_batch_")
            self.temp_dir = Path(self._tmpdir.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmpdir is not None:
            # TemporaryDirectory handles cleanup automatically on cleanup.
            self._tmpdir.cleanup()
            self._tmpdir = None
        self.temp_dir = None


# Cache for the reference structure (loaded once)
_REFERENCE_STRUCTURE: Optional[Structure] = None
_REFERENCE_CHAIN_ID: Optional[str] = None


def _load_reference_structure(
    reference_pdb_path: Optional[str] = None,
) -> Tuple[Structure, str]:
    """
    Load the reference VHH structure for RMSD calculations.

    Uses a cached copy if already loaded. If no path is provided, loads the
    bundled reference from package data.

    Returns
    -------
    (structure, chain_id)
        The Biopython Structure and the chain identifier (always 'H' for bundled).
    """
    global _REFERENCE_STRUCTURE, _REFERENCE_CHAIN_ID

    if reference_pdb_path is not None:
        # User-provided reference - load fresh (don't cache)
        structure = load_structure(reference_pdb_path, seqid_for_log="reference")
        # Auto-detect chain ID from reference (use first protein chain)
        model = structure[0]
        chain_id = None
        for chain in model.get_chains():
            if any(res.id[0] == " " for res in chain):
                chain_id = chain.id
                break
        if chain_id is None:
            raise ValueError(
                f"Could not identify protein chain in reference PDB {reference_pdb_path!r}."
            )
        return structure, chain_id

    # Use bundled reference
    if _REFERENCE_STRUCTURE is not None:
        return _REFERENCE_STRUCTURE, _REFERENCE_CHAIN_ID  # type: ignore

    try:
        ref_path = resources.files("nbframe").joinpath(REFERENCE_VHH_PDB)
        with resources.as_file(ref_path) as pdb_file:
            _REFERENCE_STRUCTURE = load_structure(
                str(pdb_file), seqid_for_log="reference_VHH"
            )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Bundled reference VHH structure not found at {REFERENCE_VHH_PDB}; "
            "ensure the PDB file is included in package data."
        ) from exc

    # Bundled reference uses chain 'H'
    _REFERENCE_CHAIN_ID = "H"
    return _REFERENCE_STRUCTURE, _REFERENCE_CHAIN_ID


def compute_features_from_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
    *,
    save_pdb: bool = False,
    filter_by_rmsd: bool = True,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    reference_pdb_path: Optional[str] = None,
) -> Optional[StructureFeatureDict]:
    """
    Convenience helper: PDB path → nanobody chain identification → AHo
    renumbering → structural feature vector.

    Parameters
    ----------
    pdb_path
        Path to the raw (non-AHo-numbered) nanobody PDB.
    chain_id
        Optional chain identifier to force; if None, heuristics are used.
    save_pdb
        If True, keep the intermediate AHo-numbered PDB on disk (in the
        current working directory). If False (default), the PDB is written
        to a temporary directory that is cleaned up automatically.
    filter_by_rmsd
        If True (default), filter out structures with framework RMSD above
        ``rmsd_threshold``.
    rmsd_threshold
        Maximum allowed framework RMSD in Angstroms. Structures with higher
        RMSD are filtered out. Default: 2.0 Å.
    reference_pdb_path
        Optional path to a custom reference PDB for RMSD calculation. If not
        provided, the bundled reference structure is used.

    Returns
    -------
    StructureFeatureDict or None
        Feature dictionary if the structure passes RMSD filtering (or filtering
        is disabled), otherwise None.
    """
    results = compute_features_for_pdbs(
        [pdb_path],
        chain_ids=[chain_id] if chain_id is not None else None,
        save_pdb=save_pdb,
        batch_size=1,
        filter_by_rmsd=filter_by_rmsd,
        rmsd_threshold=rmsd_threshold,
        reference_pdb_path=reference_pdb_path,
    )
    return results[0] if results else None


def compute_features_for_pdbs(
    pdb_paths: List[str],
    chain_ids: Optional[List[Optional[str]]] = None,
    *,
    save_pdb: bool = False,
    batch_size: int = 32,
    aho_output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    filter_by_rmsd: bool = True,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    reference_pdb_path: Optional[str] = None,
) -> List[StructureFeatureDict]:
    """
    Batch version of :func:`compute_features_from_pdb`.

    Parameters
    ----------
    pdb_paths
        List of raw PDB paths to process.
    chain_ids
        Optional list of chain hints (same length as ``pdb_paths``). Use
        ``None`` to trigger auto-detection for a given structure.
    save_pdb
        If True, keep the AHo-numbered PDBs on disk. If False (default),
        they are written to a temporary directory that is cleaned up after
        each batch.
    aho_output_dir
        Optional directory in which to write AHo-numbered PDB files when
        ``save_pdb`` is True. If not provided, files are written in the
        current working directory.
    batch_size
        Maximum number of structures to process per batch.
    verbose
        If True, print simple progress updates to stderr while processing.
    filter_by_rmsd
        If True (default), filter out structures with framework RMSD above
        ``rmsd_threshold``. Filtered structures are excluded from the result.
    rmsd_threshold
        Maximum allowed framework RMSD in Angstroms. Structures with higher
        RMSD are filtered out. Default: 2.0 Å.
    reference_pdb_path
        Optional path to a custom reference PDB for RMSD calculation. If not
        provided, the bundled reference structure is used.

    Returns
    -------
    list
        List of feature dictionaries for structures that pass RMSD filtering.
        Note: the returned list may be shorter than ``pdb_paths`` if structures
        are filtered out.
    """
    if chain_ids is not None and len(chain_ids) != len(pdb_paths):
        raise ValueError(
            "If provided, chain_ids must have the same length as pdb_paths."
        )

    # Load reference structure once if RMSD filtering is enabled
    ref_structure: Optional[Structure] = None
    ref_chain_id: Optional[str] = None
    if filter_by_rmsd:
        ref_structure, ref_chain_id = _load_reference_structure(reference_pdb_path)

    all_features: List[StructureFeatureDict] = []
    total = len(pdb_paths)
    processed = 0
    rmsd_filtered = 0

    for start in range(0, total, batch_size):
        batch_paths = pdb_paths[start : start + batch_size]
        if chain_ids is not None:
            batch_chain_hints = chain_ids[start : start + batch_size]
        else:
            batch_chain_hints = [None] * len(batch_paths)

        with RenumberingBatchContext(
            keep_files=save_pdb,
            base_dir=Path(aho_output_dir) if aho_output_dir is not None else None,
        ) as ctx:
            assert ctx.temp_dir is not None
            temp_dir = ctx.temp_dir

            for pdb_path, chain_hint in zip(batch_paths, batch_chain_hints):
                chains = parse_pdb_chains(pdb_path)
                selected_chain_id = (
                    chain_hint if chain_hint is not None else identify_nanobody_chain(chains)
                )
                if selected_chain_id not in chains:
                    raise ValueError(
                        f"Selected chain {selected_chain_id!r} not found in PDB {pdb_path!r}."
                    )

                chain_info = chains[selected_chain_id]
                anarci_results = run_anarci_for_chains(
                    {selected_chain_id: chain_info.sequence}
                )
                if selected_chain_id not in anarci_results:
                    raise ValueError(
                        f"ANARCI failed to number selected chain {selected_chain_id!r} "
                        f"in PDB {pdb_path!r}."
                    )

                anarci_result = anarci_results[selected_chain_id]
                pdb_to_aho = _build_pdb_to_aho_mapping(chain_info, anarci_result)

                parser = _get_structure_parser(pdb_path)
                structure = parser.get_structure(
                    Path(pdb_path).stem, pdb_path
                )
                aho_structure, aho_pdb_path = renumber_structure_to_aho(
                    structure,
                    selected_chain_id,
                    pdb_to_aho,
                    temp_dir=temp_dir,
                    basename=f"{Path(pdb_path).stem}_chain{selected_chain_id}.pdb",
                )

                # Calculate framework RMSD if filtering is enabled
                framework_rmsd: Optional[float] = None
                if filter_by_rmsd and ref_structure is not None:
                    framework_rmsd = calculate_framework_rmsd(
                        aho_structure,
                        selected_chain_id,
                        ref_structure,
                        ref_chain_id,  # type: ignore
                    )

                    # Filter based on RMSD threshold
                    if framework_rmsd is None or framework_rmsd > rmsd_threshold:
                        rmsd_filtered += 1
                        processed += 1
                        if verbose:
                            rmsd_str = f"{framework_rmsd:.2f}" if framework_rmsd else "N/A"
                            print(
                                f"[nbframe] Filtered {Path(pdb_path).name}: "
                                f"framework RMSD {rmsd_str} Å > {rmsd_threshold} Å threshold",
                                file=sys.stderr,
                            )
                        continue

                feats = compute_structure_features(
                    str(aho_pdb_path),
                    chain_id=selected_chain_id,
                    seqid_for_log=Path(pdb_path).name,
                )

                # Add framework RMSD to features
                feats["framework_rmsd"] = framework_rmsd

                all_features.append(feats)
                processed += 1
                if verbose and (processed % 50 == 0 or processed == total):
                    print(
                        f"[nbframe] Processed {processed}/{total} PDBs "
                        f"(filtered: {rmsd_filtered})...",
                        file=sys.stderr,
                    )

    if verbose and rmsd_filtered > 0:
        print(
            f"[nbframe] Total RMSD-filtered structures: {rmsd_filtered}/{total}",
            file=sys.stderr,
        )

    return all_features


def compute_features_for_pdb_directory(
    pdb_dir: Union[str, Path],
    pattern: str = "*.pdb",
    *,
    save_pdb: bool = False,
    save_pdb_dir: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    as_dataframe: bool = False,
    batch_size: int = 32,
    filter_by_rmsd: bool = True,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    reference_pdb_path: Optional[str] = None,
) -> Union[Dict[str, StructureFeatureDict], "pd.DataFrame"]:
    """
    Convenience helper: process all PDBs in a directory into structure features.

    Parameters
    ----------
    pdb_dir
        Directory containing raw PDB files to process.
    pattern
        Glob pattern for selecting PDB files within the directory. Defaults
        to ``\"*.pdb\"``.
    save_pdb
        If True, keep the intermediate AHo-numbered PDBs on disk.
    save_pdb_dir
        Directory in which to write AHo-numbered PDB files when
        ``save_pdb`` is True. Required if ``save_pdb`` is True.
    output_csv
        Optional path to a CSV file to write the resulting feature table to.
        If provided, a pandas DataFrame is returned.
    recursive
        If True, search for PDB files recursively under ``pdb_dir`` using
        :meth:`Path.rglob`. If False (default), only the top-level directory
        is scanned.
    as_dataframe
        If True, return a pandas DataFrame instead of a dictionary. Requires
        pandas to be installed. If ``output_csv`` is provided, this flag is
        implied.
    batch_size
        Maximum number of structures to process per batch.
    filter_by_rmsd
        If True (default), filter out structures with framework RMSD above
        ``rmsd_threshold``.
    rmsd_threshold
        Maximum allowed framework RMSD in Angstroms. Default: 2.0 Å.
    reference_pdb_path
        Optional path to a custom reference PDB for RMSD calculation.

    Returns
    -------
    dict or pandas.DataFrame
        - If ``as_dataframe`` is False and ``output_csv`` is None, returns a
          dict mapping ``Structure_ID`` to :class:`StructureFeatureDict`.
        - Otherwise, returns a DataFrame with columns:
          ``Structure_ID``, ``pdb_path``, and the feature columns.
    """
    pdb_dir_path = Path(pdb_dir)
    if not pdb_dir_path.is_dir():
        raise ValueError(f"{pdb_dir!r} is not a directory.")

    if save_pdb and save_pdb_dir is None:
        raise ValueError(
            "save_pdb_dir must be provided when save_pdb is True so that "
            "AHo-numbered PDBs have a well-defined output location."
        )

    # Gather PDB files according to the requested pattern / recursion mode.
    if recursive:
        file_iter = pdb_dir_path.rglob(pattern)
    else:
        file_iter = pdb_dir_path.glob(pattern)

    pdb_paths_path: List[Path] = sorted(
        [p for p in file_iter if p.is_file()],
        key=lambda p: str(p),
    )
    total_files = len(pdb_paths_path)
    if total_files == 0:
        if as_dataframe or output_csv is not None:
            try:
                import pandas as pd  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "pandas is required to return an empty DataFrame when no "
                    "PDB files are found."
                ) from exc
            empty_columns = [
                "Structure_ID",
                "pdb_path",
                "alpha_N",
                "tau_N",
                "alpha_C",
                "tau_C",
                "contact_density",
                "fr2_rsa_key",
                "framework_rmsd",
            ]
            df = pd.DataFrame(columns=empty_columns)
            if output_csv is not None:
                df.to_csv(output_csv, index=False)
            return df

        # No files and no tabular output requested – return an empty dict.
        return {}

    # Initial progress message.
    print(
        f"[nbframe] Found {total_files} PDB files in {pdb_dir_path} "
        f"(recursive={recursive}, pattern={pattern!r}).",
        file=sys.stderr,
    )

    # Process each PDB individually so that failures can be skipped while
    # still producing partial results.
    successes: List[Tuple[str, str, StructureFeatureDict]] = []
    failures: List[Tuple[str, str]] = []
    rmsd_filtered_count = 0

    for idx, pdb_path in enumerate(pdb_paths_path, start=1):
        sid = pdb_path.stem
        path_str = str(pdb_path)

        try:
            feats_list = compute_features_for_pdbs(
                [path_str],
                chain_ids=None,
                save_pdb=save_pdb,
                batch_size=1,
                aho_output_dir=save_pdb_dir,
                verbose=False,
                filter_by_rmsd=filter_by_rmsd,
                rmsd_threshold=rmsd_threshold,
                reference_pdb_path=reference_pdb_path,
            )
            if feats_list:
                feats = feats_list[0]
                successes.append((sid, path_str, feats))
            else:
                # Structure was filtered by RMSD
                rmsd_filtered_count += 1
        except Exception as exc:
            failures.append((path_str, str(exc)))
            print(
                f"[nbframe] WARNING: failed to process {path_str!r}: {exc}",
                file=sys.stderr,
            )

        if idx % 50 == 0 or idx == total_files:
            print(
                f"[nbframe] Processed {idx}/{total_files} PDBs "
                f"(successes={len(successes)}, rmsd_filtered={rmsd_filtered_count}, "
                f"failures={len(failures)}).",
                file=sys.stderr,
            )

    # Final summary
    print(
        f"[nbframe] Finished processing {total_files} PDB files. "
        f"Successfully processed {len(successes)}, "
        f"RMSD-filtered {rmsd_filtered_count}, "
        f"failed on {len(failures)}.",
        file=sys.stderr,
    )
    if failures:
        print(
            "[nbframe] Failed PDB examples: "
            + ", ".join(path for path, _ in failures[:5])
            + (" ..." if len(failures) > 5 else ""),
            file=sys.stderr,
        )

    # If no tabular representation is requested, return a simple mapping
    # from Structure_ID to feature dicts for successful structures only.
    if not as_dataframe and output_csv is None:
        return {sid: feats for sid, _, feats in successes}

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pandas is required when as_dataframe=True or output_csv is set."
        ) from exc

    rows: List[Dict[str, object]] = []
    for sid, path_str, feats in successes:
        row: Dict[str, object] = {
            "Structure_ID": sid,
            "pdb_path": path_str,
        }
        # Merge the structure-based features into the row.
        row.update(feats)
        rows.append(row)

    columns = [
        "Structure_ID",
        "pdb_path",
        "alpha_N",
        "tau_N",
        "alpha_C",
        "tau_C",
        "contact_density",
        "fr2_rsa_key",
        "framework_rmsd",
    ]

    df = pd.DataFrame(rows)
    # Ensure all expected columns exist, then enforce their order.
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df


__all__ = [
    "ChainInfo",
    "parse_pdb_chains",
    "identify_nanobody_chain",
    "identify_nanobody_chain_from_pdb",
    "renumber_structure_to_aho",
    "compute_features_from_pdb",
    "compute_features_for_pdbs",
    "compute_features_for_pdb_directory",
]


