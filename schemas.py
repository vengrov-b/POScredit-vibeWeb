from pydantic import BaseModel, Field, field_validator
from typing import Optional

class LoanApplication(BaseModel):
    # --- Финансовые признаки (обязательные) ---
    age: int = Field(..., ge=18, le=100, description="Возраст заявителя")
    income: float = Field(..., ge=0, description="Годовой доход")
    employment_years: float = Field(..., ge=0, le=50, description="Стаж работы (лет)")
    credit_score: int = Field(..., ge=300, le=850, description="Кредитный рейтинг (FICO)")
    loan_amount: float = Field(..., gt=0, description="Запрашиваемая сумма")
    loan_term_months: int = Field(..., ge=1, le=360, description="Срок кредита (мес.)")
    existing_loans: int = Field(..., ge=0, le=20, description="Количество текущих кредитов")
    num_dependents: int = Field(..., ge=0, le=10, description="Количество иждивенцев")

    # --- Категориальные признаки ---
    employment_status: str = Field(..., description="Статус занятости")
    marital_status: str = Field(..., description="Семейное положение")
    has_criminal_record: bool = Field(..., description="Судимости")

    @field_validator('employment_status')
    @classmethod
    def validate_employment(cls, v):
        allowed = {'employed', 'self-employed', 'unemployed', 'retired', 'student'}
        if v.lower() not in allowed:
            raise ValueError(f'employment_status must be one of {allowed}')
        return v.lower()

    @field_validator('marital_status')
    @classmethod
    def validate_marital(cls, v):
        allowed = {'single', 'married', 'divorced', 'widowed'}
        if v.lower() not in allowed:
            raise ValueError(f'marital_status must be one of {allowed}')
        return v.lower()