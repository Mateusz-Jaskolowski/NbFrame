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
from numbers import Real

import joblib
import numpy as np
from importlib import resources

from .validation import validate_thresholds, validate_rmsd_threshold

from .structure_config import (
    DEFAULT_STRUCT_KINKED_THRESHOLD as DEFAULT_KINKED_THRESHOLD,
    DEFAULT_STRUCT_EXTENDED_THRESHOLD as DEFAULT_EXTENDED_THRESHOLD,
    STRUCT_METADATA_PKG_PATH,
)
from .structure_numbering import (
    compute_features_for_pdbs,
    identify_nanobody_chains_from_pdb,
    parse_pdb_chains,
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


def resolve_structure_thresholds(kinked=None, extended=None):
    """Use bundled model metadata for unspecified thresholds; validate overrides."""
    meta = _load_metadata()
    return validate_thresholds(meta.kinked_threshold if kinked is None else kinked,
                               meta.extended_threshold if extended is None else extended)


def _failure_result(pdb_path, chain_id, status, error, *, features=None, quality=None,
                    kinked_threshold=None, extended_threshold=None):
    meta = _load_metadata()
    kinked, extended = resolve_structure_thresholds(kinked_threshold, extended_threshold)
    return dict(pdb_path=str(pdb_path), chain_id_used=chain_id,
                model_id_used=(quality or {}).get("model_id"), status=status,
                error=str(error), warnings=[str(error)], label=None, confidence=None,
                prob_kinked=None, prob_extended=None,
                probabilities={"kinked": None, "extended": None}, features=features or {},
                quality=quality or {"status": "not_assessed"},
                model_info=dict(model_file=meta.model_file, date_trained=meta.date_trained,
                                feature_cols=meta.feature_cols, train_csv=meta.train_csv,
                                performance=meta.performance,
                                thresholds={"kinked": kinked, "extended": extended}))


def _result_rank(result):
    # Select using applicability and coordinate quality, never model confidence.
    status_order = {"classified": 0, "insufficient_quality": 1, "filtered": 2, "error": 3}
    rmsd = result.get("features", {}).get("framework_rmsd")
    return (status_order.get(result["status"], 4),
            rmsd if isinstance(rmsd, Real) and math.isfinite(rmsd) else math.inf,
            result.get("chain_id_used") or "")


def prepare_feature_vector(
    features: Mapping[str, Any],
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

        if not math.isfinite(f_val):
            missing.append(name)
        vals.append(f_val)

    X = np.asarray(vals, dtype=float).reshape(1, -1)
    return X, missing


def predict_structure_from_features(
    features: Mapping[str, Any],
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
        An optional quality report (dict or JSON string) accompanies coordinate
        measurements. An insufficient-quality report withholds inference.
    use_confidence_thresholds
        If True (default), apply confidence thresholds to determine label:
        - P(kinked) > kinked_threshold → "kinked"
        - P(kinked) < extended_threshold → "extended"
        - otherwise → "uncertain"
        If False, return argmax label (binary kinked/extended).
    kinked_threshold
        Probability threshold above which to classify as "kinked".
        If None, uses the bundled model metadata (currently 0.55).
    extended_threshold
        Probability threshold below which to classify as "extended".
        If None, uses the bundled model metadata (currently 0.25).

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

    A withheld result has status="insufficient_quality", null probabilities,
    and a quality report with affected features. Successful results have
    status="classified". Without a coordinate report, quality is "not_assessed".

    Raises
    ------
    ValueError
        If required features are invalid without an insufficient-quality report.
    RuntimeError
        If the classifier or metadata cannot be loaded.
    """
    clf, meta = load_structure_classifier()
    effective_kinked_threshold, effective_extended_threshold = resolve_structure_thresholds(kinked_threshold, extended_threshold)
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
    quality = features.get("quality")
    if isinstance(quality, str):
        quality = json.loads(quality)
    if quality is not None and (not isinstance(quality, dict) or
            quality.get("status") not in ("passed", "insufficient_quality", "not_assessed")):
        raise ValueError("Invalid structure quality report.")
    quality = quality if quality is not None else {"status": "not_assessed", "issues": []}
    measurements = {
        key: None if isinstance(value, Real) and not math.isfinite(value) else value
        for key, value in features.items() if key != "quality"
    }
    messages = [issue["message"] for issue in quality.get("issues", [])]
    if quality["status"] == "insufficient_quality":
        return dict(status="insufficient_quality", error="; ".join(messages),
                    label=None, confidence=None, prob_kinked=None, prob_extended=None,
                    probabilities={"kinked": None, "extended": None},
                    features=measurements, model_info=model_info, quality=quality, warnings=messages)
    X, missing = prepare_feature_vector(measurements)
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

    return {
        "label": label,
        "confidence": confidence,
        "prob_kinked": prob_kinked,
        "prob_extended": prob_extended,
        "probabilities": {
            "kinked": prob_kinked,
            "extended": prob_extended,
        },
        "features": measurements,
        "model_info": model_info,
        "status": "classified",
        "error": None,
        "quality": quality,
        "warnings": messages,
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
) -> Dict[str, Any]:
    """
    Run the full PDB → AHo → features → classification pipeline.

    Parameters
    ----------
    pdb_path
        Path to the raw (non-AHo-numbered) nanobody PDB file.
    chain_id
        Optional chain identifier. If None, evaluate all eligible copies and select
        by usable prediction, then framework RMSD, then chain ID. The result
        includes a selection report; use the multi-chain API to retain all copies.
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
        If None, uses the bundled model metadata (currently 0.55).
    extended_threshold
        Probability threshold below which to classify as "extended".
        If None, uses the bundled model metadata (currently 0.25).
    strict
        If True (default), raise ValueError when no VHH chain is found.
        If False, retain input/processing failures as status="error" records.
        Invalid configuration always raises ValueError.

    Returns
    -------
    dict
        Result dictionary combining features, quality, and classifier output.
        Coordinate-quality failures have status="insufficient_quality"; framework
        filter failures have status="filtered". Both retain reasons without a
        label/probability. Processing failures raise unless strict=False.

        Keys include:
        - "label": "kinked" | "extended" | "uncertain"
        - "confidence": float or None (None if uncertain)
        - "prob_kinked": float
        - "prob_extended": float
    """
    kinked_threshold, extended_threshold = resolve_structure_thresholds(kinked_threshold, extended_threshold)
    validate_rmsd_threshold(rmsd_threshold)
    if chain_id is None:
        candidates = classify_all_nanobodies_in_pdb(
            pdb_path, unique_sequences=False, save_pdb=save_pdb, aho_output_dir=aho_output_dir,
            filter_by_rmsd=filter_by_rmsd, rmsd_threshold=rmsd_threshold,
            use_confidence_thresholds=use_confidence_thresholds,
            kinked_threshold=kinked_threshold, extended_threshold=extended_threshold, strict=strict)
        selected = dict(min(candidates.values(), key=_result_rank))
        selected["selection"] = {"candidate_chain_ids": [r["chain_id_used"] for r in candidates.values()],
                                 "rule": "usable_prediction_then_framework_rmsd_then_chain_id"}
        return selected
    try:
        features_list = compute_features_for_pdbs(
            [pdb_path], chain_ids=[chain_id], save_pdb=save_pdb, batch_size=1,
            aho_output_dir=aho_output_dir, verbose=False, filter_by_rmsd=filter_by_rmsd,
            rmsd_threshold=rmsd_threshold, _retain_filtered=True)
        feature_dict = features_list[0]
        if feature_dict.get("processing_status") == "filtered":
            return _failure_result(pdb_path, chain_id, "filtered", feature_dict["processing_error"],
                features={"framework_rmsd": feature_dict["framework_rmsd"]}, quality=feature_dict["quality"],
                kinked_threshold=kinked_threshold, extended_threshold=extended_threshold)
        pred = predict_structure_from_features(feature_dict,
            use_confidence_thresholds=use_confidence_thresholds,
            kinked_threshold=kinked_threshold, extended_threshold=extended_threshold)
    except Exception as exc:
        if strict:
            raise
        return _failure_result(pdb_path, chain_id, "error", exc,
                               kinked_threshold=kinked_threshold, extended_threshold=extended_threshold)
    return dict(pred, pdb_path=str(pdb_path),
                chain_id_used=pred["quality"].get("chain_id", chain_id),
                model_id_used=pred["quality"].get("model_id"))


def classify_all_nanobodies_in_pdb(
    pdb_path: str,
    *,
    chain_ids: Optional[List[str]] = None,
    unique_sequences: bool = False,
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
        If False (default), retain every VHH-like chain. If True, classify every
        copy before grouping identical observed sequences. Select a representative
        by usable prediction, then framework RMSD, then chain ID; preserve all
        copy results, probability range, and label disagreement in sequence_group.
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
        If None, uses the bundled model metadata (currently 0.55).
    extended_threshold
        Probability threshold below which to classify as "extended".
        If None, uses the bundled model metadata (currently 0.25).
    strict
        If True (default), raise ValueError when no VHH chains are found.
        If False, return input-level failures under the reserved "__input__" key
        with chain_id_used=None. Individual chain failures are always retained
        as error records so other chains can finish. Invalid settings always raise.

    Returns
    -------
    dict
        Mapping from chain_id -> classification result dict (same schema as
        :func:`classify_structure`), for each detected VHH-like nanobody chain
        including filtered, insufficient-quality, and failed chains. With
        strict=False, files with no eligible chains return an input-level error.
    """
    kinked_threshold, extended_threshold = resolve_structure_thresholds(kinked_threshold, extended_threshold)
    validate_rmsd_threshold(rmsd_threshold)
    auto_detect = chain_ids is None
    try:
        if auto_detect:
            chain_ids = identify_nanobody_chains_from_pdb(pdb_path)
        if not chain_ids:
            raise ValueError(f"No VHH-like nanobody chains found in PDB {pdb_path!r}.")
    except Exception as exc:
        if strict:
            raise
        return {"__input__": _failure_result(pdb_path, None, "error", exc,
            kinked_threshold=kinked_threshold, extended_threshold=extended_threshold)}

    results = {}
    for cid in dict.fromkeys(chain_ids):
        results[cid] = classify_structure(pdb_path, chain_id=cid, strict=False,
            save_pdb=save_pdb, aho_output_dir=aho_output_dir, filter_by_rmsd=filter_by_rmsd,
            rmsd_threshold=rmsd_threshold, use_confidence_thresholds=use_confidence_thresholds,
            kinked_threshold=kinked_threshold, extended_threshold=extended_threshold)
    if unique_sequences and auto_detect:
        chains = parse_pdb_chains(pdb_path)
        groups = {}
        for cid in results:
            groups.setdefault(chains[cid].sequence, []).append(results[cid])
        grouped = {}
        for copies in groups.values():
            representative = dict(min(copies, key=_result_rank))
            valid = [r for r in copies if r["status"] == "classified"]
            probabilities = [r["prob_kinked"] for r in valid]
            representative["sequence_group"] = {
                "chain_ids": [r["chain_id_used"] for r in copies],
                "representative_chain_id": representative["chain_id_used"],
                "label_disagreement": len({r["label"] for r in valid}) > 1,
                "prob_kinked_range": [min(probabilities), max(probabilities)] if probabilities else None,
                "results": copies,
            }
            grouped[representative["chain_id_used"]] = representative
        return grouped
    return results


__all__ = [
    "StructureModelMetadata",
    "load_structure_classifier",
    "resolve_structure_thresholds",
    "prepare_feature_vector",
    "predict_structure_from_features",
    "classify_structure",
    "classify_all_nanobodies_in_pdb",
]

