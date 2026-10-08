import numpy as np
from schemas import LoanApplication

class CreditScoringModel:
    """
    Упрощенный скоринговый движок.
    В продакшене здесь должен быть загружен обученный XGBoost/RandomForest.
    """

    def __init__(self):
        # Веса признаков (имитация обученной модели)
        # В реальности — коэффициенты из SHAP-анализа
        self.weights = {
            'credit_score': 0.35,
            'income_ratio': 0.20,
            'dti': -0.25,
            'employment_stability': 0.10,
            'existing_loans': -0.05,
            'criminal_record': -0.05,
        }
        self.base_score = 0.5

    def _compute_dti(self, app: LoanApplication) -> float:
        """Debt-to-Income Ratio — ключевой показатель долговой нагрузки."""
        monthly_income = app.income / 12
        # Упрощенный расчет платежа по кредиту
        monthly_payment = app.loan_amount / app.loan_term_months
        # Предполагаемая нагрузка по существующим кредитам
        existing_payment = app.existing_loans * 200  # условно
        total_monthly_debt = monthly_payment + existing_payment
        if monthly_income <= 0:
            return 1.0
        return total_monthly_debt / monthly_income

    def _employment_stability(self, app: LoanApplication) -> float:
        """Оценка стабильности занятости (0–1)."""
        status_scores = {
            'employed': 1.0,
            'self-employed': 0.7,
            'retired': 0.6,
            'student': 0.3,
            'unemployed': 0.0,
        }
        base = status_scores.get(app.employment_status, 0.5)
        # Бонус за длительный стаж
        tenure_bonus = min(app.employment_years / 10, 0.3)
        return min(base + tenure_bonus, 1.0)

    def _income_ratio(self, app: LoanApplication) -> float:
        """Отношение дохода к запрашиваемой сумме (нормализованное)."""
        if app.loan_amount <= 0:
            return 0.0
        ratio = app.income / app.loan_amount
        # Нормализация: ratio > 2 считается отличным
        return min(ratio / 2, 1.0)

    def predict(self, app: LoanApplication) -> dict:
        """
        Возвращает вероятность одобрения и объяснение ключевых факторов.
        """
        dti = self._compute_dti(app)
        emp_stab = self._employment_stability(app)
        inc_ratio = self._income_ratio(app)

        # --- Логистическая функция (имитация) ---
        # Нормализация credit_score (300–850) в 0–1
        cs_norm = (app.credit_score - 300) / 550
        score = (
            self.weights['credit_score'] * cs_norm
            + self.weights['income_ratio'] * inc_ratio
            + self.weights['dti'] * min(dti, 1.0)  # DTI > 1 штрафуется
            + self.weights['employment_stability'] * emp_stab
            + self.weights['existing_loans'] * min(app.existing_loans / 5, 1.0)
            + self.weights['criminal_record'] * (1.0 if app.has_criminal_record else 0.0)
        )
        score = score + self.base_score
        prob = 1 / (1 + np.exp(-score))

        # --- Решение ---
        approved = prob >= 0.5 and dti < 0.45  # DTI выше 0.45 — авто-отказ

        # --- Объяснение (SHAP-подобное) ---
        contributions = {
            'Кредитный рейтинг': round(self.weights['credit_score'] * cs_norm, 3),
            'Отношение дохода к сумме': round(self.weights['income_ratio'] * inc_ratio, 3),
            'Долговая нагрузка (DTI)': round(-self.weights['dti'] * min(dti, 1.0), 3),
            'Стабильность занятости': round(self.weights['employment_stability'] * emp_stab, 3),
            'Текущие кредиты': round(-self.weights['existing_loans'] * min(app.existing_loans / 5, 1.0), 3),
        }

        return {
            'approved': approved,
            'approval_probability': round(float(prob), 4),
            'dti': round(dti, 4),
            'decision': 'Одобрено' if approved else 'Отказано',
            'explanation': contributions,
            'recommended_limit': self._recommend_limit(app, prob) if approved else 0,
        }

    def _recommend_limit(self, app: LoanApplication, prob: float) -> float:
        """Рекомендуемая сумма кредита на основе вероятности одобрения."""
        base_limit = app.income * 0.3  # 30% годового дохода
        multiplier = min(prob * 1.5, 1.0)
        return round(base_limit * multiplier, 2)