import pandas as pd
import pytest

from strategies import process_applications, STRATEGY_NAMES
from config import DEFAULT_CONFIG


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "requested_amount": [50_000, 80_000, 150_000, 300_000],
        "p_default": [0.05, 0.2, 0.1, 0.4],
        "p_fraud": [0.01, 0.01, 0.02, 0.1],
    })


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_strategy_returns_expected_shape(sample_df, strategy):
    res_df, issued, profit, avg, t = process_applications(
        sample_df, strategy, C_max=500_000, theta_high=0.15,
        L_high=100_000, reserve_pct=20)
    assert list(res_df.columns) == [
        "id", "amount", "p_default", "expected_profit", "decision", "reason"
    ]
    assert len(res_df) == len(sample_df)
    assert issued >= 0
    assert t >= 0


def test_greedy_takes_all_profitable(sample_df):
    res_df, _, _, _, _ = process_applications(
        sample_df, "Жадный (без ограничений)", 1_000_000, 0.15, 100_000, 0)
    approved = res_df[res_df["decision"] == "Одобрено"]
    # все прибыльные (expected_profit > 0) должны быть одобрены
    profitable = res_df[res_df["expected_profit"] > 0]
    assert set(approved["id"]) == set(profitable["id"])


def test_scoring_limit_respects_C_max(sample_df):
    C_max = 130_000
    res_df, issued, _, _, _ = process_applications(
        sample_df, "Скоринг + Лимит (без доли риска)",
        C_max, 0.15, 100_000, 0)
    assert issued <= C_max


def test_proposed_respects_high_risk_share(sample_df):
    theta = 0.15
    res_df, _, _, _, _ = process_applications(
        sample_df, "Предложенный (с ограничениями)",
        C_max=1_000_000, theta_high=theta, L_high=100_000, reserve_pct=0)
    approved = res_df[res_df["decision"] == "Одобрено"]
    hr = approved[approved["amount"] >= 100_000]
    if len(approved) > 0:
        assert len(hr) / len(approved) <= theta + 1e-9


def test_unknown_strategy_raises(sample_df):
    with pytest.raises(ValueError):
        process_applications(sample_df, "Неизвестная", 500_000, 0.15, 100_000, 20)