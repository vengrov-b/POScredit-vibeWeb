"""
Эвристические стратегии принятия решений.
Все стратегии возвращают DataFrame с полями decision/reason,
а также агрегаты (issued, profit, avg_loan, exec_time_ms).
"""

import time
import numpy as np
import pandas as pd
from config import (
    EconomicsConfig, ScoringConfig, PortfolioConfig, DEFAULT_CONFIG,
)
from profit_model import calculate_expected_profit


STRATEGY_NAMES = [
    "Предложенный (с ограничениями)",
    "Скоринг + Лимит (без доли риска)",
    "Скоринговый порог",
    "Жадный (без ограничений)",
]


def process_applications(
    df: pd.DataFrame,
    strategy: str,
    C_max: float,
    theta_high: float,
    L_high: float,
    reserve_pct: float,
    econ: EconomicsConfig = DEFAULT_CONFIG.economics,
    scoring: ScoringConfig = DEFAULT_CONFIG.scoring,
):
    decisions = []
    total_issued = 0.0
    total_profit = 0.0
    approved_count = 0
    high_risk_count = 0
    limit_for_small = C_max * (1 - reserve_pct / 100)
    start = time.perf_counter()

    for _, row in df.iterrows():
        L = row["requested_amount"]
        p_d = row["p_default"]
        p_f = row["p_fraud"]
        ep = calculate_expected_profit(L, p_d, p_f, econ.r, econ.T_months,
                                       econ.gamma, econ.C_op)
        decision = "Отказано"
        reason = ""

        if strategy == "Жадный (без ограничений)":
            if ep > 0:
                decision = "Одобрено"
            else:
                reason = "Убыток"

        elif strategy == "Скоринговый порог":
            if p_d < scoring.PD_THRESHOLD and p_f < scoring.PF_THRESHOLD and ep > 0:
                decision = "Одобрено"
            else:
                reason = "Не прошел скоринг"

        elif strategy == "Скоринг + Лимит (без доли риска)":
            if not (p_d < scoring.PD_THRESHOLD and p_f < scoring.PF_THRESHOLD and ep > 0):
                reason = "Не прошел скоринг"
            elif total_issued + L > C_max:
                reason = "Исчерпан дневной лимит"
            else:
                decision = "Одобрено"

        elif strategy == "Предложенный (с ограничениями)":
            is_high_risk = L >= L_high
            if ep <= 0:
                reason = "Убыток"
            elif not is_high_risk:
                if total_issued + L > limit_for_small:
                    reason = "Исчерпан лимит (резерв)"
                else:
                    decision = "Одобрено"
            else:
                if total_issued + L > C_max:
                    reason = "Исчерпан полный лимит"
                else:
                    new_hr = high_risk_count + 1
                    new_tot = approved_count + 1
                    if new_hr / new_tot > theta_high:
                        reason = f"Доля риска > {theta_high * 100:.0f}%"
                    else:
                        decision = "Одобрено"
                        high_risk_count += 1
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")

        if decision == "Одобрено":
            total_issued += L
            total_profit += ep
            approved_count += 1

        decisions.append({
            "id": row["id"], "amount": L,
            "p_default": round(p_d, 3), "expected_profit": round(ep, 2),
            "decision": decision, "reason": reason,
        })

    exec_time = (time.perf_counter() - start) / max(len(df), 1) * 1000
    avg_loan = total_issued / approved_count if approved_count else 0.0
    return pd.DataFrame(decisions), total_issued, total_profit, avg_loan, exec_time