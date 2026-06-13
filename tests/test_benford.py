import math

from detection.benford_engine import (
    BENFORD_EXPECTED,
    benford_report,
    chi_square_statistic,
    digit_distribution,
    is_non_conforming,
    leading_digit,
    mad_score,
)


def test_leading_digit():
    assert leading_digit(123.45) == 1
    assert leading_digit(0.0456) == 4
    assert leading_digit(9.99) == 9
    assert leading_digit(500) == 5


def test_benford_expected_sums_to_one():
    assert math.isclose(sum(BENFORD_EXPECTED.values()), 1.0, rel_tol=1e-9)


def test_digit_distribution_sums_to_one():
    amounts = [123, 234, 345, 11, 19, 91, 555, 678, 789, 999]
    dist = digit_distribution(amounts)
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-9)


def test_conforming_sample_has_low_mad():
    # Synthetic sample roughly matching the Benford distribution.
    amounts = (
        [100] * 30
        + [200] * 18
        + [300] * 12
        + [400] * 10
        + [500] * 8
        + [600] * 7
        + [700] * 6
        + [800] * 5
        + [900] * 4
    )
    report = benford_report(amounts)
    assert report["sample_size"] == len(amounts)
    assert report["mad"] < 0.015
    assert not report["non_conforming"]


def test_non_conforming_sample_flagged():
    # Wash-trading-style sample: fixed lot size dominated by digit 5.
    amounts = [500] * 100
    assert is_non_conforming(amounts)
    chi_sq = chi_square_statistic(amounts)
    assert chi_sq > 0


def test_empty_input_handled():
    report = benford_report([])
    assert report["sample_size"] == 0
    assert report["mad"] == 0.0
    assert report["non_conforming"] is False
