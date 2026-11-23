"""
Централизованная конфигурация прототипа DSS.
Все числа и пороги собраны здесь.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class DataConfig:
    """Параметры генератора синтетических заявок."""
    TOTAL_APPLICATIONS: int = 100
    GENERATION_MODE: str = "second_half"   # "second_half" | "uniform"
    SMALL_LOAN_MIN: int = 30_000
    SMALL_LOAN_MAX: int = 120_000
    LARGE_LOAN_MIN: int = 150_000
    LARGE_LOAN_MAX: int = 400_000
    # Бета-распределения: (alpha, beta)
    PD_ALPHA: float = 2.0
    PD_BETA: float = 8.0
    PF_ALPHA: float = 1.0
    PF_BETA: float = 20.0


@dataclass
class ScoringConfig:
    """Пороги скоринговой стратегии."""
    PD_THRESHOLD: float = 0.15
    PF_THRESHOLD: float = 0.05


@dataclass
class PortfolioConfig:
    """Портфельные ограничения (по умолчанию)."""
    C_max: int = 500_000
    theta_high: float = 0.15
    L_high: int = 100_000
    reserve_pct: int = 20


@dataclass
class EconomicsConfig:
    """Экономика кредита."""
    r: float = 0.15
    T_months: int = 12
    gamma: float = 0.20
    C_op: int = 5_000


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = AppConfig()