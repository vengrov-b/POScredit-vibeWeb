import numpy as np
import pandas as pd
import pytest

from profit_model import (calculate_expected_profit, compute_profit_vector,
                          portfolio_profit, opportunity_cost)


def test_zero_loss_profit_equals_rate_income():
    """Если p_d = p_f = 0, прибыль = процентный доход."""
    L, r, T = 100_000, 0.15, 12
    expected = L * r * (T / 12)
    assert calculate_expected_profit(L, 0.0, 0.0, r, T, 0.2, 5000) == pytest.approx(expected)


def test_full_fraud_no_profit():
    """При p_f = 1 вся сумма теряется (мошенничество)."""
    p = calculate_expected_profit(100_000, 0.0, 1.0, 0.15, 12, 0.2, 5000)
    assert p < 0


def test_full_default_returns_gamma_only():
    """При p_d = 1, p_f = 0 — возвращаем только γ от суммы."""
    L, gamma = 100_000, 0.2
    p = calculate_expected_profit(L, 1.0, 0.0, 0.15, 12, gamma, 0)
    assert p == pytest.approx(-L * (1 - gamma), rel=1e-6)


def test_vectorized_matches_scalar():
    df = pd.DataFrame({
        "id": [1, 2],
        "requested_amount": [50_000, 200_000],
        "p_default": [0.1, 0.3],
        "p_fraud": [0.01, 0.05],
    })
    vec = compute_profit_vector(df)
    for i, row in df.iterrows():
        scalar = calculate_expected_profit(
            row["requested_amount"], row["p_default"], row["p_fraud"],
            0.15, 12, 0.2, 5000)
        assert vec[i] == pytest.approx(scalar, rel=1e-9)


def test_portfolio_profit_aggregation():
    profits = np.array([100.0, -50.0, 30.0])
    decisions = np.array([1, 0, 1])
    assert portfolio_profit(profits, decisions) == pytest.approx(130.0)


def test_opportunity_cost_non_negative():
    profits = np.array([100.0, -50.0, 30.0])
    assert opportunity_cost(profits, realized_profit=50.0) >= 0