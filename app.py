import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# =====================================================================
# ⚙️ НАСТРОЙКИ ГЕНЕРАТОРА ЗАЯВОК (Меняйте здесь!)
# =====================================================================
CONFIG = {
    "TOTAL_APPLICATIONS": 100,
    "GENERATION_MODE": "second_half",
    "SMALL_LOAN_MIN": 30000,
    "SMALL_LOAN_MAX": 120000,
    "LARGE_LOAN_MIN": 150000,
    "LARGE_LOAN_MAX": 400000
}
# =====================================================================

st.set_page_config(page_title="POS-Credit DSS", layout="wide")
st.markdown("""
<style>
    /* Уменьшаем высоту верхней панели */
    header[data-testid="stHeader"] {
        height: 40px; /* Было ~60px по умолчанию */
    }
    /* Убираем лишние отступы у основного контейнера */
    .block-container {
        padding-top: 2rem; 
    }
</style>
""", unsafe_allow_html=True)
st.title("Прототип системы поддержки решений в POS-кредитовании")
st.markdown("*Оптимизационная модель с портфельными ограничениями. Демонстрационный прототип.*")

if 'seed' not in st.session_state:
    st.session_state.seed = 11639
if 'current_strategy' not in st.session_state:
    st.session_state.current_strategy = "Предложенный (с ограничениями)"


def set_strategy(strategy_name):
    st.session_state.current_strategy = strategy_name


def format_money(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} млн. ₽"
    elif value >= 1_000:
        return f"{value / 1_000:.0f} тыс. ₽"
    else:
        return f"{value:.0f} ₽"


# --- ГЕНЕРАЦИЯ ДАННЫХ ---
def generate_applications(seed):
    np.random.seed(seed)
    amounts = []
    n = CONFIG["TOTAL_APPLICATIONS"]
    for i in range(n):
        time_of_day = i / n
        if CONFIG["GENERATION_MODE"] == "second_half":
            prob_large = 0.05 + (time_of_day ** 2) * 0.45
        else:
            prob_large = 0.15
        if np.random.rand() < prob_large:
            amount = np.random.randint(CONFIG["LARGE_LOAN_MIN"], CONFIG["LARGE_LOAN_MAX"])
        else:
            amount = np.random.randint(CONFIG["SMALL_LOAN_MIN"], CONFIG["SMALL_LOAN_MAX"])
        amounts.append(amount)
    amounts = [(amt // 1000) * 1000 for amt in amounts]
    df = pd.DataFrame({
        'id': range(1, len(amounts) + 1),
        'requested_amount': amounts,
        'p_default': np.random.beta(2, 8, len(amounts)),
        'p_fraud': np.random.beta(1, 20, len(amounts)),
    })
    return df


df = generate_applications(st.session_state.seed)

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.title("Настройки")
tab1, tab2 = st.sidebar.tabs(["Параметры", "Данные"])

with tab1:
    st.markdown("**Лимиты портфеля**")
    C_max = st.slider("Дневной лимит (₽)", min_value=100000, max_value=2000000, value=500000, step=100000,
                      help="Максимальная сумма, которую банк может выдать за день.")
    theta_high = st.slider("Доля высокорисковых", 0.0, 1.0, 0.15, 0.05,
                           help="Максимально допустимая доля крупных кредитов в портфеле.")
    L_high = st.slider("Порог крупной суммы (₽)", min_value=50000, max_value=500000, value=100000, step=10000,
                       help="Сумма, начиная с которой кредит считается крупным.")

    st.markdown("**Резервирование**")
    reserve_pct = st.slider("Резерв для крупных (%)", min_value=0, max_value=50, value=20, step=5,
                            help="Процент дневного лимита, который запрещено тратить на мелкие заявки.")

    with st.expander("Экономика кредита"):
        r = st.number_input("Ставка (r)", value=0.15, step=0.01, help="Годовая процентная ставка.")
        T_months = st.number_input("Срок (мес.)", value=12, step=1, help="Срок кредитования.")
        gamma = st.slider("Возврат при дефолте (γ)", 0.0, 1.0, 0.20, 0.05, help="Доля возврата при дефолте.")
        C_op = st.number_input("Опер. расходы (₽)", value=5000, step=500, help="Стоимость оформления.")

with tab2:
    st.markdown("**Генерация потока заявок**")
    seed_input = st.number_input("Сид (Seed):", min_value=0, max_value=999999, value=st.session_state.seed, step=1,
                                 help="Введите число для воспроизведения потока.")
    if seed_input != st.session_state.seed:
        st.session_state.seed = seed_input
        st.rerun()

    if st.button("Новый случайный сид", use_container_width=True):
        st.session_state.seed = np.random.randint(0, 999999)
        st.rerun()


# --- ЛОГИКА ---
def calculate_expected_profit(L, p_d, p_f, r, T_months, gamma, C_op):
    r_income = L * r * (T_months / 12)
    loss_default = p_d * L * (1 - gamma)
    loss_fraud = p_f * (L + C_op)
    profit = (1 - p_f) * (r_income * (1 - p_d) - loss_default) - loss_fraud
    return profit


def process_applications(df, strategy, C_max, theta_high, L_high, reserve_pct):
    decisions = []
    total_issued = 0
    total_profit = 0
    approved_count = 0
    high_risk_count = 0
    limit_for_small = C_max * (1 - reserve_pct / 100)
    start_time = time.time()

    for _, row in df.iterrows():
        L = row['requested_amount']
        p_d = row['p_default']
        p_f = row['p_fraud']
        expected_profit = calculate_expected_profit(L, p_d, p_f, r, T_months, gamma, C_op)
        decision = "Отказано"
        reason = ""

        if strategy == "Жадный (без ограничений)":
            if expected_profit > 0:
                decision = "Одобрено"
            else:
                reason = "Убыток"
        elif strategy == "Скоринговый порог":
            if p_d < 0.15 and p_f < 0.05 and expected_profit > 0:
                decision = "Одобрено"
            else:
                reason = "Не прошел скоринг"
        elif strategy == "Скоринг + Лимит (без доли риска)":
            if not (p_d < 0.15 and p_f < 0.05 and expected_profit > 0):
                reason = "Не прошел скоринг"
            elif total_issued + L > C_max:
                reason = "Исчерпан дневной лимит"
            else:
                decision = "Одобрено"
        elif strategy == "Предложенный (с ограничениями)":
            is_high_risk = L >= L_high
            if expected_profit <= 0:
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
                        reason = f"Доля риска > {theta_high * 100}%"
                    else:
                        decision = "Одобрено"
                        high_risk_count += 1

        if decision == "Одобрено":
            total_issued += L
            total_profit += expected_profit
            approved_count += 1

        decisions.append({
            'id': row['id'], 'amount': L, 'p_default': round(p_d, 3),
            'expected_profit': round(expected_profit, 2), 'decision': decision, 'reason': reason
        })

    exec_time = (time.time() - start_time) / len(df) * 1000
    avg_loan = total_issued / approved_count if approved_count > 0 else 0
    return pd.DataFrame(decisions), total_issued, total_profit, avg_loan, exec_time


# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
st.subheader("Выберите стратегию принятия решений")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.button("Предложенный", on_click=set_strategy, args=("Предложенный (с ограничениями)",),
              type="primary" if st.session_state.current_strategy == "Предложенный (с ограничениями)" else "secondary",
              use_container_width=True)
with col_b2:
    st.button("Скоринг + Лимит", on_click=set_strategy, args=("Скоринг + Лимит (без доли риска)",),
              type="primary" if st.session_state.current_strategy == "Скоринг + Лимит (без доли риска)" else "secondary",
              use_container_width=True)
with col_b3:
    st.button("Скоринговый порог", on_click=set_strategy, args=("Скоринговый порог",),
              type="primary" if st.session_state.current_strategy == "Скоринговый порог" else "secondary",
              use_container_width=True)
with col_b4:
    st.button("Жадный алгоритм", on_click=set_strategy, args=("Жадный (без ограничений)",),
              type="primary" if st.session_state.current_strategy == "Жадный (без ограничений)" else "secondary",
              use_container_width=True)

st.markdown("---")

# Запускаем выбранную стратегию
strategy = st.session_state.current_strategy
results_df, total_issued, total_profit, avg_loan, exec_time = process_applications(df, strategy, C_max, theta_high,
                                                                                   L_high, reserve_pct)

# === ГРАФИКИ (с крестиками для отказов) ===
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(f"**Поток заявок. Текущий сид: `{st.session_state.seed}`**")
    fig1, ax1 = plt.subplots(figsize=(10, 3.5), dpi=100)

    # Разделяем на одобренные и отклоненные для разных маркеров
    approved_mask = results_df['decision'] == 'Одобрено'
    rejected_mask = results_df['decision'] == 'Отказано'

    # Одобренные - кружочки
    colors_appr = ['#ff4b4b' if amt >= L_high else '#1f77b4' for amt in df[approved_mask]['requested_amount']]
    ax1.scatter(df[approved_mask]['id'], df[approved_mask]['requested_amount'] / 1000,
                c=colors_appr, s=20, alpha=0.8, marker='o', label='Одобрено')

    # Отклоненные - крестики (серые, чтобы не отвлекать)
    ax1.scatter(df[rejected_mask]['id'], df[rejected_mask]['requested_amount'] / 1000,
                c='gray', s=20, alpha=0.5, marker='x', label='Отказано')

    ax1.axhline(y=L_high / 1000, color='black', linestyle='--', linewidth=1.5, label=f'Порог L_high')
    ax1.set_xlabel("Номер заявки (время)")
    ax1.set_ylabel("Сумма (тыс. ₽)")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    st.pyplot(fig1, use_container_width=True)

with col_right:
    st.markdown("**Средние суммы заявок**")
    avg_small = df[df['requested_amount'] < L_high]['requested_amount'].mean()
    avg_large = df[df['requested_amount'] >= L_high]['requested_amount'].mean()
    if pd.isna(avg_large): avg_large = 0

    labels = ['Обычные', 'Крупные']
    amounts = [avg_small / 1000, avg_large / 1000]
    colors = ['#1f77b4', '#ff4b4b']

    fig2, ax2 = plt.subplots(figsize=(5, 3.5), dpi=100)
    y_pos = np.arange(len(amounts))
    bars = ax2.barh(y_pos, amounts, color=colors, height=0.4)
    ax2.axvline(x=L_high / 1000, color='black', linestyle='--', linewidth=1.5,
                label=f'Порог ({L_high / 1000:.0f} тыс.)')

    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 5, bar.get_y() + bar.get_height() / 2, f'{width:.0f} тыс.', ha='left', va='center',
                 fontweight='bold', fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Тыс. ₽")
    ax2.legend(loc='lower right', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    st.pyplot(fig2, use_container_width=True)

# === РЕЗУЛЬТАТЫ (с упущенной выгодой) ===
st.subheader("Результаты выбранной стратегии")

# Считаем упущенную выгоду (сколько можно было бы заработать без ограничений, одобрив всё, что в плюсе)
ideal_profit = sum(
    [calculate_expected_profit(row['requested_amount'], row['p_default'], row['p_fraud'], r, T_months, gamma, C_op)
     for _, row in df.iterrows() if
     calculate_expected_profit(row['requested_amount'], row['p_default'], row['p_fraud'], r, T_months, gamma,
                               C_op) > 0])
opportunity_cost = ideal_profit - total_profit

metrics_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px; margin-bottom: 20px;">
        <div style="flex: 1; min-width: 130px;">
            <div style="font-size: 1rem; color: #888; margin-bottom: 5px;">Ожидаемая прибыль</div>
            <div style="font-size: 2.0rem; font-weight: 600; line-height: 1.2;">{total_profit:,.0f} ₽</div>
        </div>
        <div style="flex: 1.5; min-width: 180px;">
            <div style="font-size: 1rem; color: #888; margin-bottom: 5px;">Использовано лимита</div>
            <div style="font-size: 2.0rem; font-weight: 600; line-height: 1.2;">{format_money(total_issued)} <span style="font-size: 1.3rem; color: #888;">/ {format_money(C_max)}</span></div>
        </div>
        <div style="flex: 1; min-width: 120px;">
            <div style="font-size: 1rem; color: #888; margin-bottom: 5px;">Одобрено заявок</div>
            <div style="font-size: 2.0rem; font-weight: 600; line-height: 1.2;">{len(results_df[results_df['decision'] == 'Одобрено'])} / {len(df)}</div>
        </div>
        <div style="flex: 1; min-width: 130px;">
            <div style="font-size: 1rem; color: #888; margin-bottom: 5px;">Упущенная выгода</div>
            <div style="font-size: 2.0rem; font-weight: 600; line-height: 1.2; color: #d9534f;">-{opportunity_cost:,.0f} ₽</div>
        </div>
    </div>
    """
st.markdown(metrics_html, unsafe_allow_html=True)
st.progress(min(total_issued / C_max, 1.0))

# === СВОДНОЕ СРАВНЕНИЕ ВСЕХ СТРАТЕГИЙ  ===
st.subheader("Сводное сравнение всех стратегий (Количество одобренных заявок)")

strategies_list = [
    "Предложенный (с ограничениями)",
    "Скоринг + Лимит (без доли риска)",
    "Скоринговый порог",
    "Жадный (без ограничений)"
]

# Собираем данные
summary_data = []
for strat in strategies_list:
    r_df, r_issued, r_profit, r_avg, r_time = process_applications(df, strat, C_max, theta_high, L_high, reserve_pct)
    summary_data.append({
        "name": strat,
        "profit": r_profit,
        "issued": r_issued,
        "approved": len(r_df[r_df['decision'] == 'Одобрено'])
    })

# Сортируем по количеству заявок
summary_data = sorted(summary_data, key=lambda x: x["approved"])

# Строим график
fig_sum, ax_sum = plt.subplots(figsize=(10, 2.5), dpi=100)

names = [s["name"] for s in summary_data]
approved_counts = [s["approved"] for s in summary_data]

# Цвета: выбранная - красный, остальные - серые
bar_colors = ['#ff4b4b' if n == strategy else '#a0a0a0' for n in names]

y_pos = np.arange(len(names))
bars = ax_sum.barh(y_pos, approved_counts, color=bar_colors, height=0.6)

# Добавляем обводку для выбранной стратегии
for bar, s in zip(bars, summary_data):
    if s["name"] == strategy:
        bar.set_edgecolor('black')
        bar.set_linewidth(2.0)

# Подписываем значения на столбиках
for bar, s in zip(bars, summary_data):
    width = bar.get_width()
    is_selected = s["name"] == strategy
    font_w = 'bold' if is_selected else 'normal'

    # Убрали звездочку
    label_text = f'{int(width)} заявок (прибыль: {s["profit"]:,.0f} ₽)'

    ax_sum.text(width + (max(approved_counts) * 0.02), bar.get_y() + bar.get_height() / 2,
                label_text,
                ha='left', va='center', fontweight=font_w, fontsize=9, color='black')

ax_sum.set_yticks(y_pos)
# Убрали звездочку из подписей по оси Y
ax_sum.set_yticklabels(names, fontsize=10, color='black')
for label, name in zip(ax_sum.get_yticklabels(), names):
    if name == strategy:
        label.set_fontweight('bold')

ax_sum.set_xlabel("Количество одобренных заявок (шт.)", fontsize=10, color='black')
ax_sum.spines['top'].set_visible(False)
ax_sum.spines['right'].set_visible(False)
ax_sum.tick_params(axis='x', colors='black')

# Ограничиваем правую границу
ax_sum.set_xlim(0, max(approved_counts) * 1.4)

st.pyplot(fig_sum, use_container_width=True)


# === ЖУРНАЛ ===
st.subheader(f"Журнал решений")


def highlight_decision(val):
    if val == 'Одобрено':
        return 'background-color: #85e085; color: black; font-weight: bold;'
    elif val == 'Отказано':
        return 'background-color: #ff9999; color: black; font-weight: bold;'
    return ''


styled_df = results_df.style.applymap(highlight_decision, subset=['decision'])
st.dataframe(styled_df, use_container_width=True)

csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button("Скачать решения (CSV)", csv, f"decisions_seed_{st.session_state.seed}.csv", "text/csv")