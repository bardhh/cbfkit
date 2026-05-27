"""Tests for the discrete-time RA-CBF barrier condition (h - Delta_rho)/T."""
import pytest

from cbfkit.certificates.conditions.barrier_conditions import stochastic_barrier
from cbfkit.certificates.conditions.barrier_conditions.discrete_time_risk_aware_barrier import (
    right_hand_side,
)
from cbfkit.certificates.conditions.barrier_conditions.risk_aware_margins import (
    ct_margin,
    dt_robust_margin,
)

ETA, T, RHO = 0.2, 5.0, 0.3


def test_returns_callable_with_correct_value():
    cond = right_hand_side(RHO, ETA, T, margin="dt_robust")
    delta = dt_robust_margin(ETA, T, RHO)
    assert cond(1.0) == pytest.approx((1.0 - delta) / T, abs=1e-9)
    assert cond(0.0) == pytest.approx((0.0 - delta) / T, abs=1e-9)


def test_equivalent_to_stochastic_barrier_parameterization():
    delta = ct_margin(ETA, T, RHO)
    cond = right_hand_side(RHO, ETA, T, margin="ct")
    equiv = stochastic_barrier.right_hand_side(alpha=1.0 / T, beta=delta / T)
    for h in (-0.5, 0.0, 0.3, 1.0):
        assert cond(h) == pytest.approx(equiv(h), abs=1e-9)


def test_ct_less_conservative_than_dt():
    h = 0.5
    assert right_hand_side(RHO, ETA, T, margin="ct")(h) > right_hand_side(
        RHO, ETA, T, margin="dt_robust"
    )(h)


def test_invalid_margin_name_raises():
    with pytest.raises(ValueError):
        right_hand_side(RHO, ETA, T, margin="bogus")
