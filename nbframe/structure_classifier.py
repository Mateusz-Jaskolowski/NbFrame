# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

"""
Runtime structure-based classifier for nanobody CDR3 conformation.

This module loads the trained logistic regression (with preprocessing pipeline)
and associated metadata from :mod:`nbframe.data` and exposes helpers to:

  - Turn a structure feature dictionary into an ordered feature vector.
  - Predict kinked vs extended probabilities from those features.
  - Run the full PDB → AHo → feature → prediction pipeline via
    :func:`classify_structure`.

The classifier outputs three labels based on confidence thresholds:
  - "kinked": P(kinked) > 0.55
  - "extended": P(kinked) < 0.25
  - "uncertain": 0.25 ≤ P(kinked) ≤ 0.55
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import json
import math
import warnings

import joblib
import numpy as np
from importlib import resources

from .structure_config import (
    DEFAULT_STRUCT_KINKED_THRESHOLD as DEFAULT_KINKED_THRESHOLD,
    DEFAULT_STRUCT_EXTENDED_THRESHOLD as DEFAULT_EXTENDED_THRESHOLD,
    STRUCT_METADATA_PKG_PATH,
)
from .structure_numbering import (
    compute_features_for_pdbs,
    identify_nanobody_chains_from_pdb,
    identify_unique_nanobody_chains_from_pdb,
)


@dataclass
class StructureModelMetadata:
    model_file: str
    date_trained: str
    feature_cols: List[str]
    label_mapping: Dict[str, int]
    train_csv: str
    kinked_threshold: float = DEFAULT_KINKED_THRESHOLD
    extended_threshold: float = DEFAULT_EXTENDED_THRESHOLD
    performance: Dict[str, float] = field(default_factory=dict)


_STRUCTURE_METADATA: Optional[StructureModelMetadata] = None
_STRUCTURE_CLF: Optional[Any] = None


def _load_metadata() -> StructureModelMetadata:
    """
    Load and cache the structure classifier metadata bundled with the package.
    """
    global _STRUCTURE_METADATA
    if _STRUCTURE_METADATA is not None:
        return _STRUCTURE_METADATA

    try:
        with resources.files("nbframe").joinpath(STRUCT_METADATA_PKG_PATH).open(
            "rt", encoding="utf-8"
        ) as f:
            raw = json.load(f)
    except FileNotFoundError as exc:  # pragma: no cover - packaging error
        raise RuntimeError(
            f"Structure classifier metadata not found at {STRUCT_METADATA_PKG_PATH}; "
            "ensure the JSON file is included in package data."
        ) from exc

    try:
        # Extract confidence thresholds (with defaults for backward compatibility)
        confidence = raw.get("confidence_thresholds", {})
        kinked_threshold = confidence.get("kinked_threshold", DEFAULT_KINKED_THRESHOLD)
        extended_threshold = confidence.get("extended_threshold", DEFAULT_EXTENDED_THRESHOLD)

        meta = StructureModelMetadata(
            model_file=raw["model_file"],
            date_trained=raw["date_trained"],
            feature_cols=list(raw["feature_cols"]),
            label_mapping={k: int(v) for k, v in raw["label_mapping"].items()},
            train_csv=raw["train_csv"],
            kinked_threshold=kinked_threshold,
            extended_threshold=extended_threshold,
            performance=raw.get("performance", {}),
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Structure classifier metadata JSON is missing required key {exc!r}."
        ) from exc

    _STRUCTURE_METADATA = meta
    return meta


def load_structure_classifier(verbose: bool = False) -> Tuple[Any, StructureModelMetadata]:
    """
    Lazy-load the trained structure classifier and its metadata.

    Returns
    -------
    (clf, metadata)
        ``clf`` is the scikit-learn Pipeline (StandardScaler + LogisticRegression).
        ``metadata`` is a :class:`StructureModelMetadata` instance.
    """
    global _STRUCTURE_CLF

    meta = _load_metadata()
    if _STRUCTURE_CLF is not None:
        return _STRUCTURE_CLF, meta

    model_pkg_path = f"data/{meta.model_file}"
    try:
        with resources.files("nbframe").joinpath(model_pkg_path).open("rb") as f:
            # Suppress sklearn version mismatch warnings - the model is robust to minor version differences
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                clf = joblib.load(f)
    except FileNotFoundError as exc:  # pragma: no cover - packaging error
        raise RuntimeError(
            f"Structure classifier model file {meta.model_file!r} not found "
            f"in nbframe package data (expected at {model_pkg_path})."
        ) from exc

    _STRUCTURE_CLF = clf

    if verbose:
        # Avoid importing rich just for this small message; keep it simple.
        print(
            f"[nbframe] Loaded structure classifier model {meta.model_file} "
            f"(trained {meta.date_trained})."
        )

    return clf, meta


def prepare_feature_vector(
    features: Mapping[str, Optional[float]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a structure feature mapping into an ordered feature vector.

    Parameters
    ----------
    features
        Mapping from feature name to value. Expected keys must match the
        ``feature_cols`` entry in the metadata JSON.

    Returns
    -------
    (X, missing)
        X is a numpy array of shape (1, n_features) suitable for scikit-learn.
        ``missing`` is a list of feature names that were missing or invalid;
        if non-empty, callers should treat this as an error.
    """
    meta = _load_metadata()
    vals: List[float] = []
    missing: List[str] = []

    for name in meta.feature_cols:
        val = features.get(name)
        if val is None:
            missing.append(name)
            vals.append(float("nan"))
            continue

        try:
            f_val = float(val)
        except (TypeError, ValueError):
            missing.append(name)
            vals.append(float("nan"))
            continue

        if math.isnan(f_val):
            missing.append(name)
        vals.append(f_val)

    X = np.asarray(vals, dtype=float).reshape(1, -1)
    return X, missing


def predict_structure_from_features(
    features: Mapping[str, Optional[float]],
    *,
    use_confidence_thresholds: bool = True,
    kinked_threshold: Optional[float] = None,
    extended_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Predict kinked vs extended probabilities from a structure feature dict.

    Parameters
    ----------
    features
        Mapping with at least the keys listed in the metadata ``feature_cols``.
    use_confidence_thresholds
        If True (default), apply confidence thresholds to determine label:
        - P(kinked) > kinked_threshold → "kinked"
        - P(kinked) < extended_threshold → "extended"
        - otherwise → "uncertain"
        If False, return argmax label (binary kinked/extended).
    kinked_threshold
        Probability threshold above which to classify as "kinked".
        If None, uses DEFAULT_KINKED_THRESHOLD (0.55).
    extended_threshold
        Probability threshold below which to classify as "extended".
        If None, uses DEFAULT_EXTENDED_THRESHOLD (0.25).

    Returns
    -------
    dict
        {
          "label": "kinked" | "extended" | "uncertain",
          "confidence": float or None (None if uncertain),
          "prob_kinked": float,
          "prob_extended": float,
          "probabilities": {"kinked": float, "extended": float},
          "features": { ... original feature mapping ... },
          "model_info": {
              "model_file": str,
              "date_trained": str,
              "train_csv": str,
              "thresholds": {"kinked": float, "extended": float},
          },
        }

    Raises
    ------
    ValueError
        If one or more required features are missing or invalid.
    RuntimeError
        If the classifier or metadata cannot be loaded.
    """
    clf, meta = load_structure_classifier()
    X, missing = prepare_feature_vector(features)

    if missing:
        raise ValueError(
            "Cannot run structure classifier because the following features "
            f"are missing or invalid: {', '.join(sorted(missing))}."
        )

    # scikit-learn encodes class labels as integers; we map them back using
    # the label_mapping provided in metadata.
    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("Loaded structure classifier does not support predict_proba().")

    proba = clf.predict_proba(X)[0]

    # Build mapping from numeric class code -> probability index.
    try:
        class_codes = [int(c) for c in clf.classes_]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Unexpected classifier classes_ format; expected numeric labels."
        ) from exc

    idx_by_code = {code: idx for idx, code in enumerate(class_codes)}

    # Invert metadata label_mapping (text -> code) to (code -> text).
    name_by_code = {code: name for name, code in meta.label_mapping.items()}

    def _prob_for(label: str) -> float:
        if label not in meta.label_mapping:
            raise RuntimeError(
                f"Label {label!r} not present in metadata label_mapping; "
                "retrain or update metadata."
            )
        code = meta.label_mapping[label]
        if code not in idx_by_code:
            raise RuntimeError(
                f"Classifier classes_ {class_codes!r} do not contain code {code!r} "
                f"for label {label!r}."
            )
        return float(proba[idx_by_code[code]])

    prob_kinked = _prob_for("kinked")
    prob_extended = _prob_for("extended")

    # Use provided thresholds or fall back to defaults
    effective_kinked_threshold = kinked_threshold if kinked_threshold is not None else DEFAULT_KINKED_THRESHOLD
    effective_extended_threshold = extended_threshold if extended_threshold is not None else DEFAULT_EXTENDED_THRESHOLD

    # Determine label based on confidence thresholds or argmax
    if use_confidence_thresholds:
        if prob_kinked > effective_kinked_threshold:
            label = "kinked"
            confidence = prob_kinked
        elif prob_kinked < effective_extended_threshold:
            label = "extended"
            confidence = prob_extended
        else:
            label = "uncertain"
            confidence = None
    else:
        # Fallback to argmax (binary classification)
        best_idx = int(np.argmax(proba))
        best_code = class_codes[best_idx]
        label = name_by_code.get(best_code, str(best_code))
        confidence = prob_kinked if label == "kinked" else prob_extended

    model_info = {
        "model_file": meta.model_file,
        "date_trained": meta.date_trained,
        "train_csv": meta.train_csv,
        "feature_cols": list(meta.feature_cols),
        "thresholds": {
            "kinked": effective_kinked_threshold,
            "extended": effective_extended_threshold,
        },
        "performance": meta.performance,
    }

    return {
        "label": label,
        "confidence": confidence,
        "prob_kinked": prob_kinked,
        "prob_extended": prob_extended,
        "probabilities": {
            "kinked": prob_kinked,
            "extended": prob_extended,
        },
        "features": dict(features),
        "model_info": model_info,
    }


def classify_structure(
    pdb_path: str,
    chain_id: Optional[str] = None,
    *,
    save_pdb: bool = False,
    aho_output_dir: Optional[str] = None,
    filter_by_rmsd: bool = True,
    rmsd_threshold: float = 2.0,
    use_confidence_thresholds: bool = True,
    kinked_threshold: Optional[float] = None,
    extended_threshold: Optional[float] = None,
    strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run the full PDB → AHo → features → classification pipeline.

    Parameters
    ----------
    pdb_path
        Path to the raw (non-AHo-numbered) nanobody PDB file.
    chain_id
        Optional chain identifier; if None, heuristics are used to identify
        the most likely VHH-like nanobody chain.
    save_pdb
        If True, keep the intermediate AHo-numbered PDB on disk (in
        ``aho_output_dir`` if provided, otherwise the current working dir).
    aho_output_dir
        Optional directory in which to write AHo-numbered PDB files when
        ``save_pdb`` is True.
    filter_by_rmsd
        If True (default), filter out structures with framework RMSD above
        ``rmsd_threshold``.
    rmsd_threshold
        Maximum allowed framework RMSD in Angstroms. Default: 2.0 Å.
    use_confidence_thresholds
        If True (default), apply confidence thresholds:
        - P(kinked) > kinked_threshold → "kinked"
        - P(kinked) < extended_threshold → "extended"
        - otherwise → "uncertain"
        If False, return binary argmax label.
    kinked_threshold
        Probability threshold above which to classify as "kinked".
        If None, uses DEFAULT_KINKED_THRESHOLD (0.55).
    extended_threshold
        Probability threshold below which to classify as "extended".
        If None, uses DEFAULT_EXTENDED_THRESHOLD (0.25).
    strict
        If True (default), raise ValueError when no VHH chain is found.
        If False, return None with a warning instead.

    Returns
    -------
    dict or None
        Result dictionary combining structural features and classifier output,
        or None if:
        - The structure was filtered out by RMSD threshold
        - No VHH chain was found (when strict=False)

        Keys include:
        - "label": "kinked" | "extended" | "uncertain"
        - "confidence": float or None (None if uncertain)
        - "prob_kinked": float
        - "prob_extended": float
    """
    # First compute structural features from the raw PDB. We go via the batch
    # helper so that ``aho_output_dir`` can be honoured when saving AHo PDBs.
    try:
        features_list = compute_features_for_pdbs(
            [pdb_path],
            chain_ids=[chain_id] if chain_id is not None else None,
            save_pdb=save_pdb,
            batch_size=1,
            aho_output_dir=aho_output_dir,
            verbose=False,
            filter_by_rmsd=filter_by_rmsd,
            rmsd_threshold=rmsd_threshold,
        )
    except ValueError as e:
        # Handle "no VHH chain found" errors
        if not strict:
            import warnings
            warnings.warn(f"Could not classify {pdb_path}: {e}")
            return None
        raise

    if not features_list:
        # Structure was filtered out by RMSD
        return None

    feature_dict = features_list[0]

    # Currently compute_features_for_pdbs does not expose the actual chain_id
    # chosen when auto-detecting; if this becomes important we can extend its
    # API. For now, we simply echo the hint (or None).
    pred = predict_structure_from_features(
        feature_dict,
        use_confidence_thresholds=use_confidence_thresholds,
        kinked_threshold=kinked_threshold,
        extended_threshold=extended_threshold,
    )

    result: Dict[str, Any] = {
        "pdb_path": pdb_path,
        "chain_id_used": chain_id,
        "features": pred["features"],
        "label": pred["label"],
        "confidence": pred["confidence"],
        "probabilities": pred["probabilities"],
        "prob_kinked": pred["probabilities"]["kinked"],
        "prob_extended": pred["probabilities"]["extended"],
        "model_info": pred["model_info"],
        "warnings": [],
    }

    return result


def classify_all_nanobodies_in_pdb(
    pdb_path: str,
    *,
    chain_ids: Optional[List[str]] = None,
    unique_sequences: bool = True,
    save_pdb: bool = False,
    aho_output_dir: Optional[str] = None,
    filter_by_rmsd: bool = True,
    rmsd_threshold: float = 2.0,
    use_confidence_thresholds: bool = True,
    kinked_threshold: Optional[float] = None,
    extended_threshold: Optional[float] = None,
    strict: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Classify all VHH-like nanobody chains in a PDB.

    Parameters
    ----------
    pdb_path
        Path to the raw (non-AHo-numbered) PDB file.
    chain_ids
        Optional explicit list of chain IDs to classify. When provided, this
        list is used directly (``unique_sequences`` is ignored).
    unique_sequences
        If True (default), collapse chains with identical amino-acid sequences
        and return one representative chain ID per unique sequence. If False,
        return one result per VHH-like chain.
    save_pdb
        If True, keep the intermediate AHo-numbered PDB(s) on disk (in
        ``aho_output_dir`` if provided, otherwise the current working dir).
    aho_output_dir
        Optional directory in which to write AHo-numbered PDB files when
        ``save_pdb`` is True.
    filter_by_rmsd
        If True (default), filter out structures with framework RMSD above
        ``rmsd_threshold``.
    rmsd_threshold
        Maximum allowed framework RMSD in Angstroms. Default: 2.0 Å.
    use_confidence_thresholds
        If True (default), apply confidence thresholds for labeling.
        If False, return binary argmax label.
    kinked_threshold
        Probability threshold above which to classify as "kinked".
        If None, uses DEFAULT_KINKED_THRESHOLD (0.55).
    extended_threshold
        Probability threshold below which to classify as "extended".
        If None, uses DEFAULT_EXTENDED_THRESHOLD (0.25).
    strict
        If True (default), raise ValueError when no VHH chains are found.
        If False, return empty dict with a warning instead.

    Returns
    -------
    dict
        Mapping from chain_id -> classification result dict (same schema as
        :func:`classify_structure`), for each detected VHH-like nanobody chain
        that passes RMSD filtering. Returns empty dict if no VHH chains found
        and strict=False.
    """
    if chain_ids is None:
        if unique_sequences:
            chain_ids = identify_unique_nanobody_chains_from_pdb(pdb_path)
        else:
            chain_ids = identify_nanobody_chains_from_pdb(pdb_path)

    if not chain_ids:
        if not strict:
            import warnings
            warnings.warn(f"No VHH-like nanobody chains found in PDB {pdb_path!r}.")
            return {}
        raise ValueError(
            f"No VHH-like nanobody chains found in PDB {pdb_path!r}."
        )

    # Process each chain individually to handle RMSD filtering properly
    results: Dict[str, Dict[str, Any]] = {}
    for cid in chain_ids:
        features_list = compute_features_for_pdbs(
            [pdb_path],
            chain_ids=[cid],
            save_pdb=save_pdb,
            batch_size=1,
            aho_output_dir=aho_output_dir,
            verbose=False,
            filter_by_rmsd=filter_by_rmsd,
            rmsd_threshold=rmsd_threshold,
        )

        if not features_list:
            # Chain was filtered out by RMSD
            continue

        feats = features_list[0]
        pred = predict_structure_from_features(
            feats,
            use_confidence_thresholds=use_confidence_thresholds,
            kinked_threshold=kinked_threshold,
            extended_threshold=extended_threshold,
        )
        results[cid] = {
            "pdb_path": pdb_path,
            "chain_id_used": cid,
            "features": pred["features"],
            "label": pred["label"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "prob_kinked": pred["probabilities"]["kinked"],
            "prob_extended": pred["probabilities"]["extended"],
            "model_info": pred["model_info"],
            "warnings": [],
        }

    return results


__all__ = [
    "StructureModelMetadata",
    "load_structure_classifier",
    "prepare_feature_vector",
    "predict_structure_from_features",
    "classify_structure",
    "classify_all_nanobodies_in_pdb",
]


