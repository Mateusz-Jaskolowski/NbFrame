# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

from __future__ import annotations

import sys
from typing import Dict, List, Optional, TypedDict, Union

import anarci
import numpy as np

from .sequence_config import (
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
    invalid_chars = set(sequence.upper()) - _VALID_AMINO_ACIDS
    if invalid_chars:
        return f"Input sequence contains invalid characters: {', '.join(sorted(invalid_chars))}"
    return None


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
    always match the length of the direct ANARCI alignment.

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

        aligned_list = [aa for (_, aa) in numbering_list]

        if fix_cdr1_gaps:
            _fix_cdr1_gaps(aligned_list, verbose=verbose, seq_label=sequence[:10])

        return ''.join(aligned_list)

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

    We extract the first integer we see as the base AHo number, and, if there
    is a single-letter alphabetic insertion code (e.g. 'A'), we append it.
    """
    if raw_pos is None:
        return None

    if isinstance(raw_pos, int):
        return raw_pos

    if isinstance(raw_pos, tuple):
        ints = [x for x in raw_pos if isinstance(x, int)]
        if not ints:
            return None
        aho_num = ints[0]

        insertion: Optional[str] = None
        for x in raw_pos:
            if isinstance(x, str) and len(x) == 1 and x.isalpha() and x not in {"H", "L", "K"}:
                insertion = x

        if insertion:
            return f"{aho_num}{insertion}"
        return aho_num

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


def batch_align_sequences(sequences, fix_cdr1_gaps=False, verbose=False, chunk_size=None):
    """
    Align multiple sequences using ANARCI in batches for better performance.

    Parameters
    ----------
    sequences : list
        List of sequences to align.
    fix_cdr1_gaps : bool, optional
        If True, attempts to fix gaps in the CDR-H1 region.
        Defaults to False.
    verbose : bool, optional
        If True, print progress messages.
        Defaults to False.
    chunk_size : int, optional
        Not used directly as run_anarci handles batching internally.
        Included for API compatibility.

    Returns
    -------
    list
        List of aligned sequences in the same order as input.
        None values are returned for sequences that couldn't be aligned.
    """
    import time
    import multiprocessing

    if verbose:
        print(f"Starting batch alignment of {len(sequences)} sequences...", file=sys.stderr)
        start_time = time.time()

    result = [None] * len(sequences)
    total_sequences = len(sequences)

    # Prepare sequence tuples for ANARCI batch processing
    sequence_tuples = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

    # Determine optimal number of CPUs to use (leave one for system)
    num_cpus = max(1, multiprocessing.cpu_count() - 1)

    if verbose:
        print(f"Running ANARCI alignment using {num_cpus} CPU cores...", file=sys.stderr)

    # Use run_anarci for batch processing
    _, numbered, alignment_details, _ = anarci.run_anarci(
        sequence_tuples,
        scheme="aho",
        ncpu=num_cpus,
        output=False,
        allow={"H"},
        assign_germline=False
    )

    # Process the results with progress reporting
    if verbose:
        print("Processing alignment results...", file=sys.stderr)
        processing_start_time = time.time()
        last_update_time = processing_start_time

    successful_count = 0

    for i, (numbering, details) in enumerate(zip(numbered, alignment_details)):
        # Update progress periodically
        if verbose and (i % 1000 == 0 or i == total_sequences - 1):
            current_time = time.time()
            elapsed = current_time - processing_start_time

            if i > 0:
                seqs_per_sec = (i + 1) / elapsed
                remaining_seqs = total_sequences - (i + 1)
                eta_seconds = remaining_seqs / seqs_per_sec if seqs_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                if current_time - last_update_time >= 1.0:
                    print(
                        f"Processing alignment {i+1}/{total_sequences} sequences "
                        f"({successful_count} successful) - "
                        f"ETA: {eta_minutes:.1f} minutes remaining",
                        file=sys.stderr, end="\r",
                    )
                    last_update_time = current_time
            else:
                print(f"Processing alignment {i+1}/{total_sequences} sequences", file=sys.stderr, end="\r")

        # Skip sequences that didn't align
        if numbering is None or not numbering:
            continue

        # Check for domains identified in this sequence
        for domain_idx, domain_data in enumerate(numbering):
            domain_numbering = domain_data[0]

            chain_type = alignment_details[i][domain_idx]["chain_type"]
            if chain_type != 'H':
                continue

            aligned_list = [aa for (_, aa) in domain_numbering]

            if fix_cdr1_gaps:
                _fix_cdr1_gaps(aligned_list)

            aligned_sequence = ''.join(aligned_list)
            result[i] = aligned_sequence
            successful_count += 1

            # We only care about the first domain found for each sequence
            break

    if verbose:
        total_time = time.time() - start_time
        minutes = total_time / 60
        print(
            f"\nCompleted batch alignment in {minutes:.2f} minutes. "
            f"Successfully aligned {successful_count}/{total_sequences} sequences.",
            file=sys.stderr,
        )

    return result
