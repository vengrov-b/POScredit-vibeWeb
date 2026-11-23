"""
Оптимизационная модель портфеля POS-заявок.

Реализованы:
- LP-релаксация (scipy.optimize.linprog, HiGHS) — быстрая оценка upper bound;
- MILP (pulp + CBC) — точное целочисленное решение.

Сравнение с эвристиками показывает, насколько эвристики близки к оптимуму.
"""
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:  # pragma: no cover
    PULP_AVAILABLE = False

from config import EconomicsConfig, DEFAULT_CONFIG
from profit_model import compute_profit_vector


def _prepare(df: pd.DataFrame, econ: EconomicsConfig):
    profits = compute_profit_vector(df, econ)
    amounts = df["requested_amount"].to_numpy(dtype=float)
    return profits, amounts


def _mask_only_profitable(profits: np.ndarray) -> np.ndarray:
    return (profits > 0).astype(float)


def solve_lp_relaxation(df: pd.DataFrame, C_max: float, theta_high: float,
                        L_high: float, reserve_pct: float,
                        econ: EconomicsConfig = DEFAULT_CONFIG.economics) -> Dict:
    """
    LP-релаксация: x_i ∈ [0, 1]. Даёт верхнюю границу оптимума MILP.
    Возвращает dict с полем 'x' (доли), 'profit_bound', 'status'.
    """
    profits, amounts = _prepare(df, econ)
    n = len(df)
    high_mask = (amounts >= L_high).astype(float)
    small_mask = 1.0 - high_mask

    # max p·x  ->  min -p·x
    c = -profits

    A_ub, b_ub = [], []
    # (1) дневной лимит
    A_ub.append(amounts); b_ub.append(C_max)
    # (2) доля высокорисковых: sum_H x - theta * sum x <= 0
    A_ub.append(high_mask - theta_high * np.ones(n)); b_ub.append(0.0)
    # (3) резерв для мелких
    A_ub.append(small_mask * amounts); b_ub.append((1 - reserve_pct / 100) * C_max)

    bounds = [(0.0, float(m)) for m in _mask_only_profitable(profits)]

    t0 = time.perf_counter()
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method="highs")
    dt_ms = (time.perf_counter() - t0) * 1000

    if not res.success:
        return {"status": res.message, "x": np.zeros(n), "profit_bound": 0.0,
                "exec_ms": dt_ms}

    return {
        "status": "optimal",
        "x": res.x,
        "profit_bound": float(-res.fun),
        "exec_ms": dt_ms,
    }


def solve_milp(df: pd.DataFrame, C_max: float, theta_high: float,
               L_high: float, reserve_pct: float,
               econ: EconomicsConfig = DEFAULT_CONFIG.economics,
               time_limit_s: int = 10) -> Dict:
    """
    Точная MILP через PuLP+CBC. Возвращает бинарный вектор решений.
    """
    if not PULP_AVAILABLE:
        raise RuntimeError("pulp не установлен. Установите: pip install pulp")

    profits, amounts = _prepare(df, econ)
    n = len(df)
    high_mask = (amounts >= L_high).astype(float)
    small_mask = 1.0 - high_mask

    prob = pulp.LpProblem("pos_portfolio", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # цель
    prob += pulp.lpSum(profits[i] * x[i] for i in range(n))

    # (1) дневной лимит
    prob += pulp.lpSum(amounts[i] * x[i] for i in range(n)) <= C_max
    # (2) доля высокорисковых
    prob += (pulp.lpSum(high_mask[i] * x[i] for i in range(n))
             - theta_high * pulp.lpSum(x[i] for i in range(n)) <= 0)
    # (3) резерв для мелких
    prob += (pulp.lpSum(small_mask[i] * amounts[i] * x[i] for i in range(n))
             <= (1 - reserve_pct / 100) * C_max)
    # (4) только прибыльные
    for i in range(n):
        if profits[i] <= 0:
            prob += x[i] == 0

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
    dt_ms = (time.perf_counter() - t0) * 1000

    xs = np.array([pulp.value(v) or 0.0 for v in x], dtype=float)
    xs = (xs > 0.5).astype(float)
    profit = float((profits * xs).sum())
    issued = float((amounts * xs).sum())

    return {
        "status": pulp.LpStatus[prob.status],
        "x": xs,
        "profit": profit,
        "issued": issued,
        "exec_ms": dt_ms,
    }


def milp_to_dataframe(df: pd.DataFrame, x: np.ndarray,
                      econ: EconomicsConfig = DEFAULT_CONFIG.economics) -> pd.DataFrame:
    """Конвертирует бинарный вектор решений в журнал в том же формате, что и стратегии."""
    profits = compute_profit_vector(df, econ)
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        approved = x[i] > 0.5
        reason = "" if approved else ("Убыток" if profits[i] <= 0 else "Не выбрано оптимизатором")
        rows.append({
            "id": row["id"],
            "amount": row["requested_amount"],
            "p_default": round(row["p_default"], 3),
            "expected_profit": round(float(profits[i]), 2),
            "decision": "Одобрено" if approved else "Отказано",
            "reason": reason,
        })
    return pd.DataFrame(rows)


def optimal_upper_bound(df: pd.DataFrame, C_max: float, theta_high: float,
                        L_high: float, reserve_pct: float,
                        econ: EconomicsConfig = DEFAULT_CONFIG.economics) -> float:
    """Верхняя граница оптимума из LP-релаксации."""
    return solve_lp_relaxation(df, C_max, theta_high, L_high, reserve_pct, econ)["profit_bound"]