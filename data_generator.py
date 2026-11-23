"""
Генератор синтетического потока POS-заявок.

Обоснование выбора бета-распределений:
- p_default ~ Beta(2, 8): среднее 0.2, мода 0.125 — имитирует портфель
  с преобладанием «средних» заёмщиков и редкими очень плохими.
  В POS-кредитовании исторически наблюдается перекос в сторону
  низких вероятностей дефолта.
- p_fraud ~ Beta(1, 20): среднее ≈0.048, сильно «прижато» к нулю —
  отражает низкую базовую частоту мошенничества в POS-каналах
  (обычно 0.5–3 %), но с «тяжёлым хвостом» для единичных выбросов.
- Сумма заявки зависит от времени суток (GENERATION_MODE="second_half"):
  к вечеру растёт доля крупных заявок — типичный паттерн для
  товаров длительного пользования.
"""

import numpy as np
import pandas as pd
from config import DataConfig, DEFAULT_CONFIG


def generate_applications(seed: int, cfg: DataConfig = DEFAULT_CONFIG.data) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = cfg.TOTAL_APPLICATIONS
    amounts = []
    for i in range(n):
        time_of_day = i / n
        if cfg.GENERATION_MODE == "second_half":
            prob_large = 0.05 + (time_of_day ** 2) * 0.45
        else:
            prob_large = 0.15
        if rng.random() < prob_large:
            amount = int(rng.integers(cfg.LARGE_LOAN_MIN, cfg.LARGE_LOAN_MAX))
        else:
            amount = int(rng.integers(cfg.SMALL_LOAN_MIN, cfg.SMALL_LOAN_MAX))
        amounts.append(amount)

    amounts = [(a // 1000) * 1000 for a in amounts]
    df = pd.DataFrame({
        "id": range(1, len(amounts) + 1),
        "requested_amount": amounts,
        "p_default": rng.beta(cfg.PD_ALPHA, cfg.PD_BETA, len(amounts)),
        "p_fraud": rng.beta(cfg.PF_ALPHA, cfg.PF_BETA, len(amounts)),
    })
    return df