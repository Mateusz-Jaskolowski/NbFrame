# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

# nbframe/sequence_predictor.py
"""
Sequence-based CDR3 conformation classifier.

This module provides functions to predict whether a nanobody has a kinked or
extended CDR3 conformation based on its amino acid sequence.
"""
import warnings
import joblib
import numpy as np
import pandas as pd
import os
from importlib import resources
import time
from rich.console import Console

# Configure console logs without file:line or timestamp for cleaner verbose output
console = Console(stderr=True, log_time=True, log_path=False)

# --- Relative imports for modules within the nbframe package ---
from . import sequence_align as align
from .sequence_config import (
    DEFAULT_SEQ_KINKED_THRESHOLD,
    DEFAULT_SEQ_EXTENDED_THRESHOLD,
    LR_MODEL_PKG_PATH,
    LR_METADATA_PKG_PATH,
    assign_label,
)

# --- Global Variables (Lazy Loaded) ---
lr_model_bundle = None

# --- FASTA handling functions ---
def read_fasta(fasta_file):
    """
    Read sequences from a FASTA file.

    Parameters
    ----------
    fasta_file : str
        Path to the FASTA file.

    Returns
    -------
    list of tuple
        A list of ``(sequence_name, sequence)`` tuples.
    """
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    sequences = []
    current_name = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save the previous sequence if there was one
                if current_name is not None:
                    sequences.append((current_name, ''.join(current_seq)))

                # Start a new sequence
                current_name = line[1:].strip()  # Remove the '>' character
                current_seq = []
            else:
                # Add to the current sequence
                current_seq.append(line)

    # Add the last sequence if there is one
    if current_name is not None:
        sequences.append((current_name, ''.join(current_seq)))

    return sequences

def results_to_dataframe(results_list, names=None, do_alignment=True):
    """
    Convert prediction results to a DataFrame.

    Parameters
    ----------
    results_list : list of dict
        List of prediction result dictionaries.
    names : list of str, optional
        Sequence names (must match length of *results_list*).
    do_alignment : bool, default=True
        Whether alignment was performed (controls column inclusion).

    Returns
    -------
    pandas.DataFrame
        DataFrame with prediction results.
    """
    # Validate inputs
    if names is not None and len(names) != len(results_list):
        raise ValueError("Length of names must match length of results_list")

    # Create a default sequence name list if none provided
    if names is None:
        names = [f"Sequence_{i+1}" for i in range(len(results_list))]

    # Create basic dataframe columns
    data = {
        'name': names,
        'sequence': [r['input_sequence'] for r in results_list],
        'probability': [r['probability'] for r in results_list],
        'raw_score': [r['raw_score'] for r in results_list]
    }

    # Add aligned_sequence column if alignment was performed
    if do_alignment:
        data['aligned_sequence'] = [r['aligned_sequence'] for r in results_list]

    # Add error column if any errors occurred
    if any(r['error'] is not None for r in results_list):
        data['error'] = [r['error'] for r in results_list]

    # Create the dataframe
    df = pd.DataFrame(data)

    # Rename probability column to 'nbframe_score'
    df = df.rename(columns={'probability': 'nbframe_score'})

    return df

def save_results_to_csv(results_list, output_file, names=None, do_alignment=True):
    """
    Save prediction results to a CSV file.

    Parameters
    ----------
    results_list : list of dict
        List of prediction result dictionaries.
    output_file : str
        Path to save the CSV file.
    names : list of str, optional
        Sequence names.
    do_alignment : bool, default=True
        Whether alignment was performed.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    df = results_to_dataframe(results_list, names, do_alignment)
    df.to_csv(output_file, index=False)
    return output_file

# --- Loading Functions using importlib.resources ---

def _load_lr_model(verbose=False):
    """Load the LR-based sequence classifier."""
    global lr_model_bundle
    if lr_model_bundle is None:
        try:
            with resources.files('nbframe').joinpath(LR_MODEL_PKG_PATH).open('rb') as f:
                # Suppress sklearn version mismatch warnings - the model is robust to minor version differences
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                    lr_model_bundle = joblib.load(f)
            if verbose:
                console.log("Loaded LR sequence classifier (2026-01-19, Top 20 hallmarks).")
        except FileNotFoundError:
            console.print(f"[red]Error[/]: LR model not found at {LR_MODEL_PKG_PATH}")
            raise
        except Exception as e:
            console.print(f"[red]Error loading LR model[/]: {e}")
            raise
    return lr_model_bundle


def _predict_with_lr_model(aligned_sequence, model_bundle, verbose=False):
    """
    Predict kinking probability using the LR model.

    Parameters
    ----------
    aligned_sequence : str
        AHo-aligned sequence (149 characters)
    model_bundle : dict
        Loaded model bundle with 'scaler', 'classifier', 'hallmark_features'
    verbose : bool
        Whether to print debug info

    Returns
    -------
    tuple
        (probability, raw_score) where raw_score is the logit (log-odds)
    """
    scaler = model_bundle['scaler']
    classifier = model_bundle['classifier']
    hallmark_features = model_bundle['hallmark_features']

    # Extract features
    features = []
    for hf in hallmark_features:
        pos = hf['position'] - 1  # Convert to 0-indexed
        aa = hf['amino_acid']
        log2fc = hf['log2fc']

        # Case-insensitive matching (sequences may contain lowercase)
        if pos < len(aligned_sequence) and aligned_sequence[pos].upper() == aa.upper():
            features.append(log2fc)
        else:
            features.append(0.0)

    # Scale and predict
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    probability = classifier.predict_proba(X_scaled)[0, 1]

    # Get raw score (logit) for compatibility
    # LogisticRegression: log-odds = X @ coef_ + intercept_
    raw_score = float((X_scaled @ classifier.coef_.T + classifier.intercept_).item())

    if verbose:
        console.log(f"LR features extracted: {np.sum(np.array(features) != 0)} non-zero")
        console.log(f"LR raw score (logit): {raw_score:.4f}")
        console.log(f"LR probability: {probability:.4f}")

    return probability, raw_score


def _predict_batch_with_lr_model(aligned_sequences, model_bundle, verbose=False):
    """
    Batch predict kinking probabilities using the LR model.

    Parameters
    ----------
    aligned_sequences : list[str]
        List of AHo-aligned sequences
    model_bundle : dict
        Loaded model bundle
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        (probabilities, raw_scores) as numpy arrays
    """
    scaler = model_bundle['scaler']
    classifier = model_bundle['classifier']
    hallmark_features = model_bundle['hallmark_features']

    n_seqs = len(aligned_sequences)
    n_features = len(hallmark_features)

    # Feature extraction: for each sequence, check whether the amino acid at
    # each hallmark position matches the expected residue.  With only ~20
    # features the inner loop is negligible and this avoids materialising a
    # full (n_seqs × seq_len) character matrix that would consume tens of GB
    # for large datasets.
    X = np.zeros((n_seqs, n_features), dtype=np.float32)

    for seq_idx, seq in enumerate(aligned_sequences):
        for feat_idx, hf in enumerate(hallmark_features):
            pos = hf['position'] - 1  # Convert to 0-indexed
            aa = hf['amino_acid']
            log2fc = hf['log2fc']

            if pos < len(seq) and seq[pos].upper() == aa.upper():
                X[seq_idx, feat_idx] = log2fc

    # Scale and predict
    X_scaled = scaler.transform(X)
    probabilities = classifier.predict_proba(X_scaled)[:, 1]

    # Raw scores (logits)
    raw_scores = (X_scaled @ classifier.coef_.T + classifier.intercept_).flatten()

    if verbose:
        console.log(f"LR batch prediction complete for {n_seqs} sequences.")

    return probabilities, raw_scores


# --- Core Prediction Function ---
def predict_kink_probability(
    sequence: str,
    fix_cdr1_gaps: bool = True,
    verbose: bool = False,
    do_alignment: bool = True,
    _lr_model=None
):
    """
    Predicts kinking probability for a single VHH sequence.

    Parameters
    ----------
    sequence : str
        Input VHH sequence string.
    fix_cdr1_gaps : bool, default=True
        Whether to fix gaps in CDR1 during alignment.
    verbose : bool, default=False
        Whether to print verbose output.
    do_alignment : bool, default=True
        If True, perform AHo alignment. If False, assume input sequence is already aligned.
    _lr_model : internal
        Used internally to avoid reloading model in parallel mode.

    Returns
    -------
    dict
        Dictionary with prediction results:
        {
            'input_sequence': original input sequence,
            'aligned_sequence': AHo-aligned sequence (if do_alignment=True),
            'raw_score': raw score (logit),
            'probability': probability of kinked conformation,
            'error': error message (if an error occurred)
        }
    """
    # Initialize return structure
    result = {
        'input_sequence': sequence,
        'aligned_sequence': None,
        'raw_score': None,
        'probability': None,
        'error': None
    }

    try:
        # Step 1: Align sequence to AHo numbering (if requested)
        aligned_sequence = sequence
        if do_alignment:
            if verbose:
                console.log("Aligning sequence to AHo numbering scheme...")
            aligned_sequence = align.get_aho_aligned_vhh_string(sequence, fix_cdr1_gaps, verbose)
            if aligned_sequence is None:
                result['error'] = "Sequence alignment failed"
                return result
            if verbose:
                console.log("Alignment complete.")
            result['aligned_sequence'] = aligned_sequence
        else:
            # If skipping alignment, assume the input is already aligned
            if verbose:
                console.log("Skipping alignment (assuming input is already AHo-aligned).")
            result['aligned_sequence'] = sequence
            aligned_sequence = sequence

        # Step 2 & 3: Calculate score and probability using LR model
        lr_model = _lr_model if _lr_model is not None else _load_lr_model(verbose)

        if verbose:
            console.log("Using LR sequence classifier (2026-01-19, Top 20 hallmarks)...")

        probability, raw_score = _predict_with_lr_model(aligned_sequence, lr_model, verbose)
        result['raw_score'] = raw_score
        result['probability'] = probability

    except Exception as e:
        # Return gracefully with error info instead of crashing
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        if verbose:
            console.print(f"[red]Error during prediction[/]: {error_detail}")
        result['error'] = str(e)

    return result


def classify_sequence(
    sequence: str,
    *,
    kinked_threshold: float = DEFAULT_SEQ_KINKED_THRESHOLD,
    extended_threshold: float = DEFAULT_SEQ_EXTENDED_THRESHOLD,
    fix_cdr1_gaps: bool = True,
    do_alignment: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Classify a VHH sequence as kinked, extended, or uncertain.

    This is a convenience wrapper around :func:`predict_kink_probability` that
    adds classification labels based on configurable thresholds.

    Parameters
    ----------
    sequence : str
        Input VHH sequence string.
    kinked_threshold : float, default=0.70
        Probability threshold above which to classify as "kinked".
    extended_threshold : float, default=0.40
        Probability threshold below which to classify as "extended".
    fix_cdr1_gaps : bool, default=True
        Whether to fix gaps in CDR1 during alignment.
    do_alignment : bool, default=True
        If True, perform AHo alignment. If False, assume input is already aligned.
    verbose : bool, default=False
        Whether to print verbose output.

    Returns
    -------
    dict
        Dictionary with classification results:
        {
            'label': "kinked" | "extended" | "uncertain",
            'probability': float (P(kinked)),
            'prob_kinked': float,
            'prob_extended': float,
            'confidence': float or None (None if uncertain),
            'input_sequence': str,
            'aligned_sequence': str or None,
            'raw_score': float,
            'thresholds': {'kinked': float, 'extended': float},
            'error': str or None
        }

    Examples
    --------
    >>> result = classify_sequence("EVQLVESGGGLVQAGG...")
    >>> print(result['label'])
    'kinked'
    >>> print(result['probability'])
    0.85
    """
    # Get prediction
    pred_result = predict_kink_probability(
        sequence=sequence,
        fix_cdr1_gaps=fix_cdr1_gaps,
        do_alignment=do_alignment,
        verbose=verbose,
    )

    # Build classification result
    prob = pred_result['probability']
    label, confidence = (None, None)
    if prob is not None:
        label, confidence = assign_label(prob, kinked_threshold, extended_threshold)

    return {
        'input_sequence': pred_result['input_sequence'],
        'aligned_sequence': pred_result['aligned_sequence'],
        'raw_score': pred_result['raw_score'],
        'probability': prob,
        'prob_kinked': prob,
        'prob_extended': 1.0 - prob if prob is not None else None,
        'label': label,
        'confidence': confidence,
        'thresholds': {
            'kinked': kinked_threshold,
            'extended': extended_threshold,
        },
        'error': pred_result['error'],
    }


def classify_sequences(
    sequences: list[str],
    *,
    kinked_threshold: float = DEFAULT_SEQ_KINKED_THRESHOLD,
    extended_threshold: float = DEFAULT_SEQ_EXTENDED_THRESHOLD,
    fix_cdr1_gaps: bool = True,
    do_alignment: bool = True,
    batch_size: int = 100,
    verbose: bool = False,
) -> list[dict]:
    """
    Classify multiple VHH sequences as kinked, extended, or uncertain.

    This is the batch version of :func:`classify_sequence`.

    Parameters
    ----------
    sequences : list[str]
        List of input VHH sequence strings.
    kinked_threshold : float, default=0.70
        Probability threshold above which to classify as "kinked".
    extended_threshold : float, default=0.40
        Probability threshold below which to classify as "extended".
    fix_cdr1_gaps : bool, default=True
        Whether to fix gaps in CDR1 during alignment.
    do_alignment : bool, default=True
        If True, perform AHo alignment. If False, assume inputs are already aligned.
    batch_size : int, default=100
        Number of sequences to process in each batch.
    verbose : bool, default=False
        Whether to print verbose output.

    Returns
    -------
    list[dict]
        List of classification result dictionaries (same schema as :func:`classify_sequence`).
    """
    # Get batch predictions
    pred_results = predict_kink_probabilities(
        sequences=sequences,
        fix_cdr1_gaps=fix_cdr1_gaps,
        do_alignment=do_alignment,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Build classification results
    results = []
    for pred_result in pred_results:
        prob = pred_result['probability']
        label, confidence = (None, None)
        if prob is not None:
            label, confidence = assign_label(prob, kinked_threshold, extended_threshold)

        results.append({
            'input_sequence': pred_result['input_sequence'],
            'aligned_sequence': pred_result['aligned_sequence'],
            'raw_score': pred_result['raw_score'],
            'probability': prob,
            'prob_kinked': prob,
            'prob_extended': 1.0 - prob if prob is not None else None,
            'label': label,
            'confidence': confidence,
            'thresholds': {
                'kinked': kinked_threshold,
                'extended': extended_threshold,
            },
            'error': pred_result['error'],
        })

    return results


# High‑throughput batch prediction
def predict_kink_probabilities(
    sequences: list[str],
    fix_cdr1_gaps: bool = True,
    verbose: bool = False,
    do_alignment: bool = True,
    batch_size: int = 100,
):
    """
    Predict kinking probabilities for a list of sequences using efficient batch processing.

    This is the primary high‑throughput API for scoring many sequences. It uses
    vectorised scoring in chunks instead of per‑sequence processing and is
    designed to scale to millions of sequences.

    Parameters
    ----------
    sequences : list[str]
        A list of input VHH sequence strings.
    fix_cdr1_gaps : bool, default=True
        Whether to fix CDR1 gaps during alignment.
    verbose : bool, default=False
        If True, print progress messages.
    do_alignment : bool, default=True
        If True, perform AHo alignment on each input sequence.
        If False, assumes the input sequences are already AHo-aligned.
    batch_size : int, default=100
        Number of sequences to process in each batch.

    Returns
    -------
    list
        A list of dictionaries with prediction results for each sequence.
    """
    n_total = len(sequences)

    if verbose:
        start_time = time.time()
        # Load model first so we can show metadata
        lr_model = _load_lr_model(verbose=False)
        model_name = lr_model.get("model_name", "LR 2026-01-19")
        n_features = len(lr_model.get("hallmark_features", []))
        alignment_status = "yes" if do_alignment else "skipped"
        console.log(
            f"NbFrame Sequence Classifier\n"
            f"           Model: {model_name} ({n_features} hallmark features)\n"
            f"           Input: {n_total:,} sequences (alignment: {alignment_status})"
        )
    else:
        lr_model = _load_lr_model(verbose=False)

    results = []
    aligned_sequences = sequences

    # Step 1: Batch align all sequences if needed
    if do_alignment:
        if verbose:
            align_start = time.time()
            console.log(f"Aligning {n_total:,} sequences...")
        aligned_sequences = align.batch_align_sequences(
            sequences,
            fix_cdr1_gaps=fix_cdr1_gaps,
            verbose=verbose,
            chunk_size=batch_size
        )
        if verbose:
            align_time = time.time() - align_start
            console.log(f"Alignment complete ({align_time:.1f}s)")

    # Create a list of valid aligned sequences for scoring
    valid_seqs = []
    valid_indices = []

    for i, (seq, aligned_seq) in enumerate(zip(sequences, aligned_sequences)):
        result = {
            'input_sequence': seq,
            'aligned_sequence': aligned_seq,
            'raw_score': None,
            'probability': None,
            'error': None
        }

        if aligned_seq is None:
            result['error'] = "Alignment failed"
            results.append(result)
            continue

        valid_seqs.append(aligned_seq)
        valid_indices.append(i)
        results.append(result)

    # Step 2 & 3: Batch calculate scores and probabilities
    if valid_seqs:
        if verbose:
            feat_start = time.time()
            console.log(f"Extracting features & scoring {len(valid_seqs):,} valid sequences...")

        probabilities, scores = _predict_batch_with_lr_model(valid_seqs, lr_model, verbose=False)

        if verbose:
            score_time = time.time() - feat_start
            console.log(f"Scoring complete ({score_time:.1f}s)")

        for i, idx in enumerate(valid_indices):
            results[idx]['raw_score'] = float(scores[i]) if isinstance(scores, np.ndarray) else scores[i]
            results[idx]['probability'] = float(probabilities[i]) if isinstance(probabilities, np.ndarray) else probabilities[i]

    if verbose:
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r['probability'] is not None)
        failed = n_total - successful
        speed = successful / total_time if total_time > 0 else 0
        console.log(
            f"Results: {successful:,} successful / {failed:,} failed\n"
            f"           Speed:  {speed:,.0f} seq/s\n"
            f"           Total:  {total_time / 60:.2f} minutes"
        )

    return results

def predict_from_fasta(
    fasta_file: str,
    fix_cdr1_gaps: bool = True,
    verbose: bool = False,
    do_alignment: bool = True,
    batch_size: int = 100,
    output_csv: str | None = None
):
    """
    Predict kinking probability for all sequences in a FASTA file using batch processing.

    Parameters
    ----------
    fasta_file : str
        Path to FASTA file containing sequences.
    fix_cdr1_gaps : bool, default=True
        Whether to fix gaps in CDR1 during alignment.
    verbose : bool, default=False
        If True, print progress messages.
    do_alignment : bool, default=True
        If True, perform AHo alignment on each input sequence.
        If False, assumes the input sequences are already AHo-aligned.
    batch_size : int, default=100
        Number of sequences to process in each batch.
    output_csv : str, optional
        Path to save results as CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with prediction results.
    """
    if verbose:
        console.log(f"Reading sequences from FASTA file: {fasta_file}")

    # Read sequences from FASTA file
    fasta_sequences = read_fasta(fasta_file)

    if not fasta_sequences:
        raise ValueError(f"No valid sequences found in FASTA file {fasta_file}")

    sequence_names = [name for name, _ in fasta_sequences]
    sequences = [seq for _, seq in fasta_sequences]

    if verbose:
        console.log(f"Found {len(sequences)} sequences.")
        console.log(f"Processing sequences in batches of {batch_size}...")

    # Process sequences in batches
    results = predict_kink_probabilities(
        sequences,
        fix_cdr1_gaps=fix_cdr1_gaps,
        verbose=verbose,
        do_alignment=do_alignment,
        batch_size=batch_size
    )

    # Convert to DataFrame
    df = results_to_dataframe(results, sequence_names, do_alignment)

    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        if verbose:
            console.log(f"Results saved to {output_csv}")

    return df


def predict_dataframe(
    df: pd.DataFrame,
    sequence_column: str,
    *,
    inplace: bool = False,
    probability_column: str = "nbframe_score",
    raw_score_column: str = "raw_score",
    aligned_sequence_column: str | None = "aligned_sequence",
    error_column: str | None = "error",
    **predict_kwargs
) -> pd.DataFrame:
    """
    Add NbFrame predictions to an existing DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing sequences.
    sequence_column : str
        Name of the column that holds the sequences to score.
    inplace : bool, default=False
        If True, modify the original DataFrame in place. Otherwise, return a copy.
    probability_column : str, default="nbframe_score"
        Column name to store calibrated probabilities.
    raw_score_column : str, default="raw_score"
        Column name to store raw scores (logits).
    aligned_sequence_column : str or None, default="aligned_sequence"
        Column name to store aligned sequences. Set to None to skip adding.
    error_column : str or None, default="error"
        Column name to store error messages. Set to None to skip adding.
    **predict_kwargs :
        Additional keyword arguments forwarded to
        :func:`predict_kink_probabilities`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with prediction columns appended (or the original DataFrame
        when ``inplace=True``).
    """
    if sequence_column not in df.columns:
        raise KeyError(f"Sequence column '{sequence_column}' not found in DataFrame.")

    if df[sequence_column].isnull().any():
        raise ValueError(f"Sequence column '{sequence_column}' contains missing values.")

    verbose = predict_kwargs.get("verbose", False)

    sequences = df[sequence_column].tolist()

    if verbose:
        columns_to_add = [probability_column, raw_score_column]
        if aligned_sequence_column is not None:
            columns_to_add.append(aligned_sequence_column)
        if error_column is not None:
            columns_to_add.append(error_column)
        console.log(
            f"predict_dataframe: scoring {len(sequences):,} sequences "
            f"from column '{sequence_column}'"
        )

    results = predict_kink_probabilities(
        sequences=sequences,
        **predict_kwargs
    )

    target_df = df if inplace else df.copy()

    target_df[probability_column] = [result['probability'] for result in results]
    target_df[raw_score_column] = [result['raw_score'] for result in results]

    if aligned_sequence_column is not None:
        target_df[aligned_sequence_column] = [
            result['aligned_sequence'] for result in results
        ]

    if error_column is not None:
        target_df[error_column] = [result['error'] for result in results]

    if verbose:
        columns_str = ", ".join(columns_to_add)
        console.log(f"Added columns: {columns_str}")

    return target_df

