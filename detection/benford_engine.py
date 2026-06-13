"""Benford's Law digit-distribution analysis for transaction amounts.

Implements the chi-square statistic, per-digit z-scores, and the Mean
Absolute Deviation (MAD) test described in the LedgerLens methodology,
applied to the leading digit of trade amounts.
"""

import math
from collections import Counter
from typing import Iterable

# Expected frequency of each leading digit (1-9) under Benford's Law.
BENFORD_EXPECTED = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

# MAD thresholds for conformity, per Nigrini's classification.
MAD_NONCONFORMITY_THRESHOLD = 0.015


def leading_digit(amount: float) -> int:
    """Return the first significant digit (1-9) of a positive amount."""
    if amount <= 0:
        raise ValueError("amount must be positive")
    while amount < 1:
        amount *= 10
    while amount >= 10:
        amount /= 10
    return int(amount)


def digit_distribution(amounts: Iterable[float]) -> dict[int, float]:
    """Observed frequency of each leading digit (1-9) across `amounts`."""
    digits = [leading_digit(a) for a in amounts if a > 0]
    if not digits:
        return {d: 0.0 for d in range(1, 10)}
    counts = Counter(digits)
    total = len(digits)
    return {d: counts.get(d, 0) / total for d in range(1, 10)}


def chi_square_statistic(amounts: Iterable[float]) -> float:
    """Chi-square goodness-of-fit statistic against the Benford distribution."""
    amounts = [a for a in amounts if a > 0]
    n = len(amounts)
    if n == 0:
        return 0.0
    observed_counts = Counter(leading_digit(a) for a in amounts)
    chi_sq = 0.0
    for d in range(1, 10):
        expected_count = BENFORD_EXPECTED[d] * n
        observed_count = observed_counts.get(d, 0)
        chi_sq += (observed_count - expected_count) ** 2 / expected_count
    return chi_sq


def digit_z_scores(amounts: Iterable[float]) -> dict[int, float]:
    """Z-score of each digit's observed frequency vs. the Benford expectation."""
    amounts = [a for a in amounts if a > 0]
    n = len(amounts)
    if n == 0:
        return {d: 0.0 for d in range(1, 10)}
    observed = digit_distribution(amounts)
    z_scores = {}
    for d in range(1, 10):
        p = BENFORD_EXPECTED[d]
        std_err = math.sqrt(p * (1 - p) / n)
        z_scores[d] = 0.0 if std_err == 0 else (observed[d] - p) / std_err
    return z_scores


def mad_score(amounts: Iterable[float]) -> float:
    """Mean Absolute Deviation between observed and expected digit frequencies."""
    amounts = list(amounts)
    if not amounts:
        return 0.0
    observed = digit_distribution(amounts)
    deviations = [abs(observed[d] - BENFORD_EXPECTED[d]) for d in range(1, 10)]
    return sum(deviations) / len(deviations)


def is_non_conforming(amounts: Iterable[float]) -> bool:
    """True if the MAD score exceeds the non-conformity threshold."""
    return mad_score(amounts) > MAD_NONCONFORMITY_THRESHOLD


def benford_report(amounts: Iterable[float]) -> dict:
    """Full Benford analysis for a window of transaction amounts."""
    amounts = list(amounts)
    return {
        "sample_size": len([a for a in amounts if a > 0]),
        "chi_square": chi_square_statistic(amounts),
        "mad": mad_score(amounts),
        "z_scores": digit_z_scores(amounts),
        "non_conforming": is_non_conforming(amounts),
    }
