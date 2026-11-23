"""Проверки корректности входных параметров и портфельных ограничений."""

from dataclasses import dataclass


class ValidationError(ValueError):
    """Исключение для некорректных входных данных."""


def validate_portfolio_params(C_max, theta_high, L_high, reserve_pct):
    """Проверяет допустимость портфельных ограничений."""
    errors = []
    if C_max <= 0:
        errors.append("Дневной лимит C_max должен быть положительным.")
    if not (0.0 <= theta_high <= 1.0):
        errors.append("Доля высокорисковых θ должна быть в диапазоне [0, 1].")
    if L_high <= 0:
        errors.append("Порог крупной суммы L_high должен быть положительным.")
    if L_high > C_max:
        errors.append("Порог L_high не может превышать дневной лимит C_max.")
    if not (0 <= reserve_pct < 100):
        errors.append("Резерв должен быть в диапазоне [0, 100).")
    if errors:
        raise ValidationError(" ".join(errors))
    return True


def validate_economics_params(r, T_months, gamma, C_op):
    """Проверяет параметры экономики кредита."""
    errors = []
    if r < 0:
        errors.append("Ставка r не может быть отрицательной.")
    if T_months <= 0 or T_months > 60:
        errors.append("Срок T_months должен быть в диапазоне (0, 60].")
    if not (0.0 <= gamma <= 1.0):
        errors.append("Возврат при дефолте γ должен быть в [0, 1].")
    if C_op < 0:
        errors.append("Операционные расходы не могут быть отрицательными.")
    if errors:
        raise ValidationError(" ".join(errors))
    return True


def validate_dataframe(df):
    """Проверяет корректность датафрейма заявок."""
    required = {"id", "requested_amount", "p_default", "p_fraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(f"Отсутствуют столбцы: {missing}")
    if (df["requested_amount"] <= 0).any():
        raise ValidationError("Все суммы кредитов должны быть > 0.")
    if not df["p_default"].between(0, 1).all():
        raise ValidationError("p_default должен быть в [0, 1].")
    if not df["p_fraud"].between(0, 1).all():
        raise ValidationError("p_fraud должен быть в [0, 1].")
    return True


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""