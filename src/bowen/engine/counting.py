"""An exact integer total that refuses a float.

Purpose: Python 3.12 changed the builtin ``sum()`` over floats to compensated summation, so a float total could differ
         in its last bits between 3.11 and 3.12. Every float total in ``src/bowen`` is a ``math.fsum``, exact on every
         version; every integer total is a ``count``, which fails on a float rather than summing it the old way.
Spec:    docs/implementation_plan_version_2.md#S1.1
Tests:   tests/bowen/test_float_sums.py::test_s1_1b_count_refuses_a_float
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable

import numpy


def count(values: Iterable[int]) -> int:
    """Purpose: the total of integers (or booleans, numpy's included, as counted conditions), refusing any other number.
    Spec:    docs/implementation_plan_version_2.md#S1.1
    Tests:   tests/bowen/test_float_sums.py::test_s1_1b_count_refuses_a_float
    """
    total = 0
    for value in values:
        if isinstance(value, numpy.bool_):  # a counted condition over an array: not a numbers.Integral (review)
            value = bool(value)
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"count() totals integers; got {type(value).__name__} {value!r}: use math.fsum")
        total += value
    return total
