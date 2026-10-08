from collections import defaultdict
from datetime import datetime

class StatsStore:
    """Хранилище агрегированной статистики (in-memory)."""

    def __init__(self):
        self.total_applications = 0
        self.total_approved = 0
        self.by_status = defaultdict(int)
        self.by_marital = defaultdict(int)
        self.approval_by_income_bucket = defaultdict(lambda: {'total': 0, 'approved': 0})
        self.recent_decisions = []  # последние N решений

    def record(self, application, result):
        self.total_applications += 1
        if result['approved']:
            self.total_approved += 1

        self.by_status[application.employment_status] += 1
        self.by_marital[application.marital_status] += 1

        # Доходные корзины
        income = application.income
        if income < 30000:
            bucket = '<30k'
        elif income < 60000:
            bucket = '30k-60k'
        elif income < 100000:
            bucket = '60k-100k'
        else:
            bucket = '>100k'

        self.approval_by_income_bucket[bucket]['total'] += 1
        if result['approved']:
            self.approval_by_income_bucket[bucket]['approved'] += 1

        self.recent_decisions.append({
            'timestamp': datetime.now().isoformat(),
            'approved': result['approved'],
            'probability': result['approval_probability'],
            'dti': result['dti'],
        })
        # Держим только последние 50
        self.recent_decisions = self.recent_decisions[-50:]

    def get_stats(self):
        approval_rate = (
            self.total_approved / self.total_applications
            if self.total_applications > 0 else 0
        )
        income_stats = {}
        for bucket, data in self.approval_by_income_bucket.items():
            income_stats[bucket] = {
                'total': data['total'],
                'approved': data['approved'],
                'approval_rate': (
                    data['approved'] / data['total'] if data['total'] > 0 else 0
                ),
            }
        return {
            'total_applications': self.total_applications,
            'total_approved': self.total_approved,
            'overall_approval_rate': round(approval_rate, 4),
            'by_employment_status': dict(self.by_status),
            'by_marital_status': dict(self.by_marital),
            'by_income_bucket': income_stats,
            'recent_decisions': self.recent_decisions[-10:],
        }