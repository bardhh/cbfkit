"""Tests for closed-form risk-aware CBF margins (ACC 2026 Eqs. 8, 13)."""
import math

import pytest

from cbfkit.certificates.conditions.barrier_conditions.risk_aware_margins import (
    ct_margin,
    dt_robust_margin,
    dt_tight_margin,
)

ETA, T, RHO = 0.2, 5.0, 0.3


def test_ct_margin_known_value():
    assert ct_margin(ETA, T, RHO) == pytest.approx(0.46353, abs=1e-4)


def test_dt_robust_margin_known_value():
    assert dt_robust_margin(ETA, T, RHO) == pytest.approx(0.69397, abs=1e-4)


def test_dt_tight_equals_ct():
    assert dt_tight_margin(ETA, T, RHO) == pytest.approx(ct_margin(ETA, T, RHO), abs=1e-12)


def test_dt_robust_at_least_ct_across_rho():
    for i in range(1, 100):
        rho = i / 100.0
        assert dt_robust_margin(ETA, T, rho) >= ct_margin(ETA, T, rho) - 1e-9


def test_margins_increase_with_eta_and_T():
    assert dt_robust_margin(0.4, T, RHO) > dt_robust_margin(0.2, T, RHO)
    assert ct_margin(ETA, 10.0, RHO) > ct_margin(ETA, 5.0, RHO)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_rho_raises(bad):
    with pytest.raises((AssertionError, ValueError)):
        ct_margin(ETA, T, bad)


def test_invalid_eta_and_T_raise():
    with pytest.raises((AssertionError, ValueError)):
        ct_margin(0.0, T, RHO)
    with pytest.raises((AssertionError, ValueError)):
        ct_margin(ETA, 0.0, RHO)
