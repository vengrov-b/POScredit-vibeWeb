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
    "NUM_WAVES": 4,
    "WAVE_LARGE_PROB": 0.4,
    "SMALL_LOAN_MIN": 30000,
    "SMALL_LOAN_MAX": 50000,
    "LARGE_LOAN_MIN": 100000,
    "LARGE_LOAN_MAX": 400000
}
# =====================================================================

st.set_page_config(page_title="POS-Credit DSS", layout="wide")
st.title("Прототип системы поддержки решений в POS-кредитовании")
st.markdown("*Оптимизационная модель с портфельными ограничениями. Демонстрационный прототип.*")

if 'seed' not in st.session_state:
    st.session_state.seed = 42
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
    wave_size = n // CONFIG["NUM_WAVES"]

    for _ in range(CONFIG["NUM_WAVES"]):
        if np.random.rand() < CONFIG["WAVE_LARGE_PROB"]:
            block_amounts = np.random.randint(CONFIG["LARGE_LOAN_MIN"], CONFIG["LARGE_LOAN_MAX"], wave_size)
        else:
            block_amounts = np.random.randint(CONFIG["SMALL_LOAN_MIN"], CONFIG["SMALL_LOAN_MAX"], wave_size)
        block_amounts = (block_amounts // 1000) * 1000
        amounts.extend(block_amounts)

    amounts = amounts[:n]
    df = pd.DataFrame({
        'id': range(1, len(amounts) + 1),
        'requested_amount': amounts,
        'p_default': np.random.beta(2, 8, len(amounts)),
        'p_fraud': np.random.beta(1, 20, len(amounts)),
    })
    return df


df = generate_applications(st.session_state.seed)

# --- БОКОВАЯ ПАНЕЛЬ (ВОЗВРАТ К ВКЛАДКАМ) ---
st.sidebar.title("Настройки")
# Возвращаем аккуратные вкладки
tab1, tab2 = st.sidebar.tabs(["⚙️ Параметры", "🎲 Данные"])

with tab1:
    st.markdown("**Лимиты портфеля**")
    C_max = st.slider("Дневной лимит (₽)", min_value=100000, max_value=5000000, value=1000000, step=100000)
    theta_high = st.slider("Доля высокорисковых", 0.0, 1.0, 0.20, 0.05)
    L_high = st.slider("Порог крупной суммы (₽)", min_value=50000, max_value=500000, value=150000, step=10000)

    st.markdown("**Резервирование**")
    reserve_pct = st.slider("Резерв для крупных (%)", min_value=0, max_value=50, value=20, step=5)

    with st.expander("📈 Экономика кредита"):
        r = st.number_input("Ставка (r)", value=0.15, step=0.01)
        T_months = st.number_input("Срок (мес.)", value=12, step=1)
        gamma = st.slider("Возврат при дефолте (γ)", 0.0, 1.0, 0.20, 0.05)
        C_op = st.number_input("Опер. расходы (₽)", value=3000, step=500)

with tab2:
    st.markdown("**Генерация потока заявок**")
    seed_input = st.number_input("Сид (Seed):", min_value=0, max_value=999999, value=st.session_state.seed, step=1)
    if seed_input != st.session_state.seed:
        st.session_state.seed = seed_input
        st.rerun()

    if st.button("🎲 Новый случайный сид", use_container_width=True):
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

# === ГРАФИКИ (Компоновка 2/3 на 1/3, как вы просили ранее) ===
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(f"**Поток заявок во времени (волны). Текущий сид: `{st.session_state.seed}`**")
    fig1, ax1 = plt.subplots(figsize=(10, 3.5), dpi=100)

    # Используем аккуратные точки размера 15-20, как было в удачном дизайне
    colors = ['#ff4b4b' if amt >= L_high else '#1f77b4' for amt in df['requested_amount']]
    ax1.scatter(df['id'], df['requested_amount'] / 1000, c=colors, s=20, alpha=0.7)

    ax1.axhline(y=L_high / 1000, color='black', linestyle='--', linewidth=1.5, label=f'Порог L_high')
    ax1.set_xlabel("Номер заявки (время)")
    ax1.set_ylabel("Сумма (тыс. ₽)")
    ax1.legend(loc='upper right', fontsize=9)
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

# === РЕЗУЛЬТАТЫ ===
st.subheader("Результаты модели")
strategy = st.session_state.current_strategy
results_df, total_issued, total_profit, avg_loan, exec_time = process_applications(df, strategy, C_max, theta_high,
                                                                                   L_high, reserve_pct)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Ожидаемая прибыль", f"{total_profit:,.0f} ₽")
col2.metric("Использовано лимита", f"{format_money(total_issued)} / {format_money(C_max)}")
col3.metric("Одобрено заявок", f"{(results_df['decision'] == 'Одобрено').sum()} / {len(df)}")
col4.metric("Средняя сумма кредита", f"{avg_loan:,.0f} ₽")
col5.metric("Время на заявку", f"{exec_time:.3f} мс")

st.progress(min(total_issued / C_max, 1.0))

st.subheader(f"Журнал решений")
st.dataframe(
    results_df.style.applymap(lambda x: 'background-color: #d4edda' if x == 'Одобрено' else 'background-color: #f8d7da',
                              subset=['decision']), use_container_width=True)

csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Скачать решения (CSV)", csv, f"decisions_seed_{st.session_state.seed}.csv", "text/csv")