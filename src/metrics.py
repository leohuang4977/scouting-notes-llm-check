"""Agreement statistics for two coders on one field.

All functions take two integer arrays of category indices (0 to q - 1)
and the number of categories q in the codebook for that field. Using the
codebook's q, not the number of categories that happen to appear, keeps
AC1 and PABAK comparable across fields, coders and bootstrap resamples.

Why three chance-corrected statistics and not one:
  Cohen's kappa corrects for the agreement two coders would reach by
  chance given how often each uses each category. When one category is
  very common (most notes have no health flag), kappa can be low even
  when the coders almost always agree. This is the known kappa paradox.
  Gwet's AC1 and PABAK are less sensitive to that, so a field is trusted
  only when kappa and AC1 agree that it is good.
"""

from __future__ import annotations

import numpy as np


def confusion(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """q x q table of counts: rows are coder a's categories, columns coder b's."""
    return np.bincount(a * q + b, minlength=q * q).reshape(q, q)


def percent_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Share of notes where the two coders gave the same code."""
    return float(np.mean(a == b))


def cohen_kappa(a: np.ndarray, b: np.ndarray, q: int) -> float:
    """Cohen's kappa: (po - pe) / (1 - pe), with pe from each coder's own margins.

    Returns nan when pe is 1 (both coders used one and the same category
    for every note), where kappa is undefined.
    """
    table = confusion(a, b, q) / len(a)
    po = np.trace(table)
    pe = float(table.sum(axis=1) @ table.sum(axis=0))
    return float("nan") if np.isclose(pe, 1.0) else float((po - pe) / (1 - pe))


def gwet_ac1(a: np.ndarray, b: np.ndarray, q: int) -> float:
    """Gwet's AC1: (po - pe) / (1 - pe), with pe = sum(pi_k * (1 - pi_k)) / (q - 1).

    pi_k is the average of the two coders' shares for category k.
    """
    table = confusion(a, b, q) / len(a)
    po = np.trace(table)
    pi = (table.sum(axis=1) + table.sum(axis=0)) / 2
    pe = float(np.sum(pi * (1 - pi)) / (q - 1))
    return float((po - pe) / (1 - pe))


def pabak(a: np.ndarray, b: np.ndarray, q: int) -> float:
    """Prevalence- and bias-adjusted kappa: (q * po - 1) / (q - 1)."""
    po = percent_agreement(a, b)
    return float((q * po - 1) / (q - 1))
