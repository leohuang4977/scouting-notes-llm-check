"""Agreement statistics checked against values worked out by hand.

Each table lists counts as rows = coder A's category, columns = coder B's.
The expected values are written as the arithmetic, so they can be checked
with a pencil.
"""

import math

import numpy as np
import pytest

from src.metrics import cohen_kappa, gwet_ac1, pabak, percent_agreement


def arrays_from_table(table: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a table of counts into the two coders' code arrays."""
    a, b = [], []
    for i, row in enumerate(table):
        for j, count in enumerate(row):
            a += [i] * count
            b += [j] * count
    return np.array(a), np.array(b)


def test_two_by_two_textbook_table():
    # 50 notes: both yes 20, A yes B no 5, A no B yes 10, both no 15.
    a, b = arrays_from_table([[20, 5], [10, 15]])
    po = 35 / 50                          # .70
    pe_kappa = 0.5 * 0.6 + 0.5 * 0.4      # A says yes 25/50, B says yes 30/50 -> .50
    assert percent_agreement(a, b) == pytest.approx(po)
    assert cohen_kappa(a, b, 2) == pytest.approx((po - pe_kappa) / (1 - pe_kappa))  # .40
    pi_yes = (0.5 + 0.6) / 2              # .55
    pe_ac1 = 2 * pi_yes * (1 - pi_yes)    # .495
    assert gwet_ac1(a, b, 2) == pytest.approx((po - pe_ac1) / (1 - pe_ac1))  # .40594
    assert gwet_ac1(a, b, 2) == pytest.approx(0.405941, abs=1e-6)
    assert pabak(a, b, 2) == pytest.approx(2 * po - 1)  # .40


def test_kappa_paradox_table_where_ac1_stays_high():
    # 125 notes, one category very common: kappa near zero, AC1 high.
    a, b = arrays_from_table([[118, 5], [2, 0]])
    po = 118 / 125                                    # .944
    p_a, p_b = 123 / 125, 120 / 125                   # .984, .96
    pe_kappa = p_a * p_b + (1 - p_a) * (1 - p_b)      # .94528
    assert cohen_kappa(a, b, 2) == pytest.approx((po - pe_kappa) / (1 - pe_kappa))
    assert cohen_kappa(a, b, 2) == pytest.approx(-0.023392, abs=1e-6)
    pi = (p_a + p_b) / 2                              # .972
    pe_ac1 = 2 * pi * (1 - pi)                        # .054432
    assert gwet_ac1(a, b, 2) == pytest.approx(0.940776, abs=1e-6)
    assert gwet_ac1(a, b, 2) == pytest.approx((po - pe_ac1) / (1 - pe_ac1))


def test_three_by_three_table():
    # 30 notes. Row totals 12, 12, 6. Column totals 11, 12, 7. Diagonal 22.
    a, b = arrays_from_table([[10, 2, 0], [1, 8, 3], [0, 2, 4]])
    po = 22 / 30
    pe_kappa = (12 * 11 + 12 * 12 + 6 * 7) / 30**2    # .353333
    assert cohen_kappa(a, b, 3) == pytest.approx((po - pe_kappa) / (1 - pe_kappa))
    assert cohen_kappa(a, b, 3) == pytest.approx(0.587629, abs=1e-6)
    pi = np.array([(12 + 11) / 60, (12 + 12) / 60, (6 + 7) / 60])
    pe_ac1 = float(np.sum(pi * (1 - pi)) / 2)         # 1163/3600 = .323056
    assert gwet_ac1(a, b, 3) == pytest.approx((po - pe_ac1) / (1 - pe_ac1))
    assert gwet_ac1(a, b, 3) == pytest.approx(0.606073, abs=1e-6)
    assert pabak(a, b, 3) == pytest.approx((3 * po - 1) / 2)  # .60


def test_an_unused_category_changes_ac1_and_pabak_but_not_kappa():
    # Same 3 x 3 data, but the codebook has a fourth category nobody used.
    a, b = arrays_from_table([[10, 2, 0], [1, 8, 3], [0, 2, 4]])
    assert cohen_kappa(a, b, 4) == pytest.approx(cohen_kappa(a, b, 3))
    assert pabak(a, b, 4) == pytest.approx((4 * 22 / 30 - 1) / 3)  # .644444
    assert gwet_ac1(a, b, 4) != pytest.approx(gwet_ac1(a, b, 3))


def test_perfect_and_undefined_cases():
    a = np.array([0, 1, 2, 1, 0])
    assert cohen_kappa(a, a, 3) == pytest.approx(1.0)
    assert gwet_ac1(a, a, 3) == pytest.approx(1.0)
    same = np.zeros(10, dtype=int)
    assert math.isnan(cohen_kappa(same, same, 2))  # chance agreement is 1: kappa undefined


def test_kappa_matches_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    for q in (2, 3, 4):
        a = rng.integers(0, q, size=300)
        b = np.where(rng.random(300) < 0.7, a, rng.integers(0, q, size=300))
        assert cohen_kappa(a, b, q) == pytest.approx(sklearn_metrics.cohen_kappa_score(a, b))
