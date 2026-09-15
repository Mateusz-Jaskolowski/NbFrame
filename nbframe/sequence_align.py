# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

from __future__ import annotations

import sys
from typing import Dict, List, Optional, TypedDict, Union

import anarci
import numpy as np

from .sequence_config import (
    AHO_ALIGNED_LENGTH,
    CDR1_END_IDX,
    CDR1_GAP_TARGET_IDX,
    CDR1_START_IDX,
)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class AnarciPosition(TypedDict):
    """
    Single-position ANARCI numbering entry.

    aho_label
        AHo position label (e.g. 1, 35, or \"100A\"). Can be None for gap-only
        entries where no residue is present in the original sequence.
    aa
        Amino-acid character for this aligned position (\"-\" for gaps).
    """

    aho_label: Union[int, str, None]
    aa: str


class AnarciChainResult(TypedDict, total=False):
    """
    Lightweight representation of ANARCI output for a single chain.

    This structure is intentionally minimal so it can be used both by the
    sequence-only alignment helpers and the PDB renumbering pipeline.
    """

    chain_type: str
    positions: List[AnarciPosition]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def _validate_sequence(sequence: str) -> Optional[str]:
    """
    Validate that *sequence* is a non-empty string of standard amino acids.

    Parameters
    ----------
    sequence : str
        The sequence to validate.

    Returns
    -------
    str or None
        An error message string if validation fails, or None if the
        sequence is valid.
    """
    if not isinstance(sequence, str) or not sequence:
        return "Input sequence must be a non-empty string."
    if len(sequence) >= 10000:
        return "Input sequence must contain fewer than 10000 residues (ANARCI limit)."
    invalid_chars = set(sequence.upper()) - _VALID_AMINO_ACIDS
    if invalid_chars:
        return f"Input sequence contains invalid characters: {', '.join(sorted(invalid_chars))}"
    return None


def validate_aligned_sequence(sequence, hallmark_positions=()) -> Optional[str]:
    """Validate the canonical 149-column input used by the sequence model.

    At least 80 residues and 80% of the model's distinct hallmark positions
    must be resolved. This permits ordinary AHo gaps (including position 85)
    while rejecting sparse inputs that would otherwise score as all zeros.
    X is allowed as an unknown residue, but does not count as resolved.
    It does not verify domain identity when the caller skips ANARCI.
    """
    if not isinstance(sequence, str) or len(sequence) != AHO_ALIGNED_LENGTH:
        return f"AHo-aligned input must be a string of exactly {AHO_ALIGNED_LENGTH} columns."
    sequence = sequence.upper()
    invalid = set(sequence) - (_VALID_AMINO_ACIDS | {"-", "X"})
    if invalid:
        return f"AHo-aligned input contains invalid characters: {', '.join(sorted(invalid))}"
    if sum(aa in _VALID_AMINO_ACIDS for aa in sequence) < 80:
        return "AHo-aligned input must contain at least 80 resolved amino acids."
    positions = set(hallmark_positions)
    if positions and sum(sequence[p - 1] in _VALID_AMINO_ACIDS for p in positions) / len(positions) < 0.8:
        return "Fewer than 80% of the model's hallmark positions are resolved."
    return None


def _canonical_aho_string(numbering, fix_cdr1_gaps=False):
    """Project labelled numbering onto base AHo positions 1–149.

    Insertions remain distinct in the numbering, but occupy no model column:
    36A must never displace the residue at 37. The original input sequence is
    retained in prediction results; this projection is not a lossless sequence.
    """
    positions = {}
    for raw_pos, aa in numbering:
        label = _decode_aho_position(raw_pos)
        if label is None or label in positions:
            raise ValueError(f"Invalid or duplicate ANARCI AHo position: {raw_pos!r}")
        positions[label] = aa
    aligned = [positions.get(p, "-") for p in range(1, AHO_ALIGNED_LENGTH + 1)]
    if fix_cdr1_gaps:
        _fix_cdr1_gaps(aligned)
    return "".join(aligned)


def _validate_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _contiguous_regions(condition):
    """
    Finds contiguous True regions in a boolean array.

    Parameters
    ----------
    condition : numpy.ndarray
        A 1D boolean numpy array.

    Returns
    -------
    list[tuple[int, int]]
        A list of tuples, where each tuple is (start_index, end_index_exclusive).
    """
    d = np.diff(condition.astype(int))
    idx, = d.nonzero()
    idx += 1
    if condition[0]:
        idx = np.r_[0, idx]
    if condition[-1]:
        idx = np.r_[idx, condition.size]
    idx.shape = (-1, 2)
    return [tuple(row) for row in idx]


def _fix_cdr1_gaps(aligned_list: list, verbose: bool = False, seq_label: str = "") -> None:
    """
    Consolidate or move gaps within the AHo CDR-H1 region in-place.

    Uses ``CDR1_START_IDX``, ``CDR1_END_IDX``, and ``CDR1_GAP_TARGET_IDX``
    from :mod:`nbframe.sequence_config`.

    Parameters
    ----------
    aligned_list : list
        Mutable list of single-character strings representing the AHo-aligned
        sequence. Modified **in-place**.
    verbose : bool, optional
        If True, print messages when gap fixing actions are taken.
    seq_label : str, optional
        Short identifier for log messages (e.g. first 10 characters of the
        original sequence).
    """
    if len(aligned_list) < CDR1_END_IDX:
        if verbose:
            sys.stderr.write(
                f"Warning: Sequence too short ({len(aligned_list)} residues) "
                "to perform CDR-H1 gap fixing.\n"
            )
        return

    cdr1_region_array = np.array(aligned_list[CDR1_START_IDX:CDR1_END_IDX])
    is_gap = cdr1_region_array == "-"
    gap_regions = _contiguous_regions(is_gap)

    # Case 1: Single gap region -- move to expected position if misplaced
    if len(gap_regions) == 1:
        start, end = gap_regions[0]
        num_gaps = end - start
        # The CDR1 correction must never move framework/model columns.
        if CDR1_GAP_TARGET_IDX + num_gaps > CDR1_END_IDX:
            return
        current_start = CDR1_START_IDX + start
        if current_start != CDR1_GAP_TARGET_IDX and aligned_list[CDR1_GAP_TARGET_IDX] != "-":
            if verbose:
                print(f"Note: Moving single CDR-H1 gap for seq starting {seq_label}...")
            del aligned_list[current_start:current_start + num_gaps]
            for _ in range(num_gaps):
                aligned_list.insert(CDR1_GAP_TARGET_IDX, "-")

    # Case 2: Multiple gap regions -- merge iteratively
    elif len(gap_regions) > 1:
        if verbose:
            print(
                f"Note: Merging multiple ({len(gap_regions)}) CDR-H1 gap "
                f"regions for seq starting {seq_label}..."
            )
        while len(gap_regions) > 1:
            first_start, first_end = gap_regions[0]
            second_start, _ = gap_regions[1]
            num_gaps_to_move = first_end - first_start
            gaps_to_insert = ["-"] * num_gaps_to_move

            abs_insert_point = CDR1_START_IDX + second_start
            abs_delete_start = CDR1_START_IDX + first_start
            abs_delete_end = CDR1_START_IDX + first_end

            # Insert THEN delete
            aligned_list[abs_insert_point:abs_insert_point] = gaps_to_insert
            if abs_insert_point <= abs_delete_start:
                adjusted_delete_start = abs_delete_start + num_gaps_to_move
                adjusted_delete_end = abs_delete_end + num_gaps_to_move
            else:
                adjusted_delete_start = abs_delete_start
                adjusted_delete_end = abs_delete_end
            del aligned_list[adjusted_delete_start:adjusted_delete_end]

            # Recalculate gap regions on the modified list
            cdr1_region_array = np.array(aligned_list[CDR1_START_IDX:CDR1_END_IDX])
            is_gap = cdr1_region_array == "-"
            gap_regions = _contiguous_regions(is_gap)


# ---------------------------------------------------------------------------
# Single-sequence alignment
# ---------------------------------------------------------------------------


def get_aho_aligned_vhh_string(sequence: str, fix_cdr1_gaps: bool = True, verbose: bool = False) -> str | None:
    """
    Takes a VHH sequence string and returns the AHo-aligned sequence string.

    Optionally fixes gaps in the CDR-H1 region based on common ANARCI
    post-processing heuristics. The length of the returned sequence should
    be 149 base-position columns; labelled insertions do not occupy model columns.

    Parameters
    ----------
    sequence : str
        The antibody sequence string (must contain only valid AA letters,
        no gaps).
    fix_cdr1_gaps : bool, optional
        If True, attempts to consolidate or move gaps within the AHo CDR-H1
        region. Defaults to True.
    verbose : bool, optional
        If True, print messages to stdout when gap fixing actions are taken.
        Defaults to False.

    Returns
    -------
    str or None
        The AHo-aligned sequence string (including gaps '-') or None if
        ANARCI numbering fails or the input sequence is invalid.
    """
    error = _validate_sequence(sequence)
    if error is not None:
        sys.stderr.write(f"ERROR: {error}\n")
        return None

    try:
        numbering_tuple = anarci.number(sequence, scheme="aho")

        if numbering_tuple is None:
            sys.stderr.write(f"ANARCI numbering failed for sequence: {sequence[:20]}...\n")
            return None

        numbering_list, chain_type = numbering_tuple

        if chain_type != 'H':
            sys.stderr.write(f"Sequence identified as chain type '{chain_type}', not 'H'. Required for VHH processing.\n")
            return None

        return _canonical_aho_string(numbering_list, fix_cdr1_gaps)

    except Exception as e:
        sys.stderr.write(f"An error occurred during ANARCI processing or gap fixing: {e}\n")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# AHo position decoding and chain-level numbering
# ---------------------------------------------------------------------------


def _decode_aho_position(raw_pos: object) -> Union[int, str, None]:
    """
    Best-effort decoder for the position object returned by ANARCI.

    Depending on the ANARCI version and scheme, ``raw_pos`` can take several
    shapes, e.g.:

        - (num, icode)
        - (chain_type, num)
        - (chain_type, num, icode)
        - num

    Tuple layout determines the base number and insertion code. A single-letter
    insertion (including H, K, or L) is appended to the number; unsupported
    layouts return None.
    """
    if raw_pos is None:
        return None

    if isinstance(raw_pos, int):
        return raw_pos

    if isinstance(raw_pos, tuple):
        # ANARCI uses (number, insertion); older wrappers may prepend a chain
        # type. The insertion is determined by its position, not its letter.
        if len(raw_pos) == 2 and isinstance(raw_pos[0], int):
            number, insertion = raw_pos
        elif len(raw_pos) in (2, 3) and isinstance(raw_pos[0], str) and isinstance(raw_pos[1], int):
            number = raw_pos[1]
            insertion = raw_pos[2] if len(raw_pos) == 3 else " "
        else:
            return None
        if not isinstance(insertion, str):
            return None
        insertion = insertion.strip()
        if insertion and (len(insertion) != 1 or not insertion.isalpha()):
            return None
        return f"{number}{insertion}" if insertion else number

    return None


def number_chain_to_aho(sequence: str) -> Optional[AnarciChainResult]:
    """
    Run ANARCI (scheme='aho') on a *single* sequence and return heavy-chain
    numbering in a lightweight, implementation-agnostic format.

    This is a thin wrapper around :func:`anarci.number` that:

      - Ensures the input is non-empty and AA-only.
      - Requires the detected chain type to be 'H' (heavy).
      - Converts the ANARCI position objects into simple AHo labels
        (e.g. ``35`` or ``\"100A\"``) alongside aligned amino acids.

    Returns ``None`` if numbering fails or the chain is not heavy.
    """
    error = _validate_sequence(sequence)
    if error is not None:
        sys.stderr.write(f"ERROR: {error}\n")
        return None

    try:
        numbering_tuple = anarci.number(sequence, scheme="aho")
    except Exception as e:  # pragma: no cover - defensive against ANARCI internals
        sys.stderr.write(f"ANARCI.number failed with an unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return None

    if numbering_tuple is None:
        sys.stderr.write(f"ANARCI numbering failed for sequence: {sequence[:20]}...\n")
        return None

    numbering_list, chain_type = numbering_tuple
    if chain_type != "H":
        return None

    positions: List[AnarciPosition] = []
    for raw_pos, aa in numbering_list:
        aho_label = _decode_aho_position(raw_pos)
        positions.append({"aho_label": aho_label, "aa": aa})

    return {"chain_type": chain_type, "positions": positions}


def run_anarci_for_chains(chain_seqs: Dict[str, str]) -> Dict[str, AnarciChainResult]:
    """
    Convenience wrapper: run heavy-chain ANARCI (AHo scheme) on multiple
    sequences provided as a ``{chain_id: sequence}`` mapping.

    Only successfully numbered heavy chains are returned.
    """
    results: Dict[str, AnarciChainResult] = {}
    for chain_id, seq in chain_seqs.items():
        res = number_chain_to_aho(seq)
        if res is not None:
            results[chain_id] = res
    return results


# ---------------------------------------------------------------------------
# Batch alignment
# ---------------------------------------------------------------------------


def batch_align_sequences(sequences, fix_cdr1_gaps=False, verbose=False, chunk_size=None, ncpu=None):
    """Align sequences in bounded ANARCI chunks, preserving input order.

    Invalid records return None without preventing valid records from being
    aligned. Infrastructure failures propagate to the caller. Results contain
    149 base AHo columns; insertion residues do not shift those columns.
    ``chunk_size`` defaults to 100 and ``ncpu`` caps parallel workers.
    """
    import multiprocessing

    chunk_size = 100 if chunk_size is None else chunk_size
    _validate_positive_int(chunk_size, "chunk_size")
    if ncpu is not None:
        _validate_positive_int(ncpu, "ncpu")
    workers = ncpu if ncpu is not None else max(1, multiprocessing.cpu_count() - 1)
    result = [None] * len(sequences)
    for start in range(0, len(sequences), chunk_size):
        valid = [(i, sequences[i]) for i in range(start, min(start + chunk_size, len(sequences)))
                 if _validate_sequence(sequences[i]) is None]
        if not valid:
            continue
        _, numbered, details, _ = anarci.run_anarci(
            [(f"seq_{i}", seq.upper()) for i, seq in valid],
            scheme="aho", ncpu=min(workers, len(valid)), output=False,
            allow={"H"}, assign_germline=False,
        )
        if len(numbered) != len(valid) or len(details) != len(valid):
            raise RuntimeError("ANARCI returned a different number of records than requested.")
        for (i, _), domains, domain_details in zip(valid, numbered, details):
            if not domains:
                continue
            for domain, detail in zip(domains, domain_details):
                if detail["chain_type"] == "H":
                    result[i] = _canonical_aho_string(domain[0], fix_cdr1_gaps)
                    break
        if verbose:
            print(f"Aligned {min(start + chunk_size, len(sequences))}/{len(sequences)} sequences", file=sys.stderr)
    return result
