"""Экономическая модель ожидаемой прибыли по одному кредиту."""

import numpy as np
import pandas as pd
from config import EconomicsConfig, DEFAULT_CONFIG


def calculate_expected_profit(L, p_d, p_f, r, T_months, gamma, C_op):
    """
    Ожидаемая прибыль по одному кредиту.

    Формула:
        profit = (1 - p_f) * [r_income * (1 - p_d) - p_d * L * (1 - γ)]
                 - p_f * (L + C_op),
    где r_income = L * r * T/12 — процентный доход без дефолта.
    """
    r_income = L * r * (T_months / 12)
    loss_default = p_d * L * (1 - gamma)
    loss_fraud = p_f * (L + C_op)
    return (1 - p_f) * (r_income * (1 - p_d) - loss_default) - loss_fraud


def compute_profit_vector(df: pd.DataFrame,
                          econ: EconomicsConfig = DEFAULT_CONFIG.economics) -> np.ndarray:
    """Векторизованный расчёт ожидаемой прибыли по всем заявкам."""
    L = df["requested_amount"].to_numpy(dtype=float)
    pd_ = df["p_default"].to_numpy(dtype=float)
    pf_ = df["p_fraud"].to_numpy(dtype=float)
    r_income = L * econ.r * (econ.T_months / 12)
    loss_default = pd_ * L * (1 - econ.gamma)
    loss_fraud = pf_ * (L + econ.C_op)
    return (1 - pf_) * (r_income * (1 - pd_) - loss_default) - loss_fraud


def portfolio_profit(profits: np.ndarray, decisions: np.ndarray) -> float:
    """Суммарная прибыль портфеля по булевому вектору решений."""
    return float((profits * decisions).sum())


def opportunity_cost(profits: np.ndarray, realized_profit: float) -> float:
    """Упущенная выгода = прибыль идеального портфеля − фактическая."""
    ideal = float(np.clip(profits, 0, None).sum())
    return max(ideal - realized_profit, 0.0)