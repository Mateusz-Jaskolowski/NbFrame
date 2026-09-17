"""Shared validation for public classification settings."""
import math
from numbers import Real


def validate_thresholds(kinked, extended):
    """Require finite probability bounds with a nonempty uncertainty interval."""
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           for v in (kinked, extended)) or not 0 <= extended < kinked <= 1:
        raise ValueError('Thresholds must be finite numbers satisfying 0 <= extended < kinked <= 1.')
    return float(kinked), float(extended)


def validate_rmsd_threshold(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('rmsd_threshold must be a finite non-negative number.')
