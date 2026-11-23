import pandas as pd
import pytest

from optimizer import solve_lp_relaxation, solve_milp, PULP_AVAILABLE
from profit_model import compute_profit_vector


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "requested_amount": [50_000, 80_000, 150_000, 300_000, 60_000],
        "p_default": [0.05, 0.2, 0.1, 0.4, 0.03],
        "p_fraud": [0.01, 0.01, 0.02, 0.1, 0.005],
    })


def test_lp_returns_feasible_solution(sample_df):
    res = solve_lp_relaxation(sample_df, 300_000, 0.2, 100_000, 0)
    assert res["status"] == "optimal"
    assert res["profit_bound"] >= 0
    # все x в [0, 1]
    assert ((res["x"] >= -1e-9) & (res["x"] <= 1 + 1e-9)).all()


def test_lp_bound_ge_zero(sample_df):
    res = solve_lp_relaxation(sample_df, 100_000, 0.1, 100_000, 20)
    assert res["profit_bound"] >= 0


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp не установлен")
def test_milp_profit_le_lp_bound(sample_df):
    lp = solve_lp_relaxation(sample_df, 300_000, 0.2, 100_000, 0)
    milp = solve_milp(sample_df, 300_000, 0.2, 100_000, 0)
    assert milp["profit"] <= lp["profit_bound"] + 1e-6


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp не установлен")
def test_milp_respects_C_max(sample_df):
    C_max = 200_000
    milp = solve_milp(sample_df, C_max, 0.5, 100_000, 0)
    assert milp["issued"] <= C_max + 1e-6


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp не установлен")
def test_milp_better_than_greedy_subset(sample_df):
    """MILP должен быть не хуже любой допустимой подвыборки."""
    milp = solve_milp(sample_df, 300_000, 0.2, 100_000, 0)
    profits = compute_profit_vector(sample_df)
    assert milp["profit"] >= profits.max() - 1e-6  # хотя бы одну лучшую заявку