import streamlit as st
import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from validators import (validate_portfolio_params, validate_economics_params,
                        validate_dataframe, ValidationError)
from data_generator import generate_applications
from strategies import process_applications, STRATEGY_NAMES
from optimizer import (solve_lp_relaxation, solve_milp, milp_to_dataframe,
                       optimal_upper_bound, PULP_AVAILABLE)
from visualization import (plot_application_flow, plot_avg_amounts,
                           plot_strategies_comparison, plot_efficient_frontier)

st.set_page_config(page_title="POS-Credit DSS", layout="wide")
st.markdown("""
<style>
    header[data-testid="stHeader"] { height: 40px; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("Прототип системы поддержки решений в POS-кредитовании")
st.markdown("*Оптимизационная модель с портфельными ограничениями. Демонстрационный прототип.*")

# --- Сессия ---
if "seed" not in st.session_state:
    st.session_state.seed = 11639
if "current_strategy" not in st.session_state:
    st.session_state.current_strategy = "MILP (оптимум)"
if "use_milp" not in st.session_state:
    st.session_state.use_milp = True

def set_strategy(name):
    st.session_state.current_strategy = name


def format_money(v):
    if v >= 1_000_000: return f"{v/1_000_000:.1f} млн. ₽"
    if v >= 1_000:     return f"{v/1_000:.0f} тыс. ₽"
    return f"{v:.0f} ₽"


# --- Боковая панель ---
st.sidebar.title("Настройки")
tab1, tab2, tab3 = st.sidebar.tabs(["Портфель", "Экономика", "Данные"])

with tab1:
    C_max = st.slider("Дневной лимит (₽)", 100_000, 2_000_000, 500_000, 100_000)
    theta_high = st.slider("Доля высокорисковых", 0.0, 1.0, 0.15, 0.05)
    L_high = st.slider("Порог крупной суммы (₽)", 50_000, 500_000, 100_000, 10_000)
    reserve_pct = st.slider("Резерв для крупных (%)", 0, 50, 20, 5)

with tab2:
    r = st.number_input("Ставка (r)", value=0.15, step=0.01)
    T_months = st.number_input("Срок (мес.)", value=12, step=1)
    gamma = st.slider("Возврат при дефолте (γ)", 0.0, 1.0, 0.20, 0.05)
    C_op = st.number_input("Опер. расходы (₽)", value=5000, step=500)
    st.markdown("**Скоринг**")
    pd_thr = st.number_input("Порог p_default", value=0.15, step=0.01)
    pf_thr = st.number_input("Порог p_fraud", value=0.05, step=0.01)

with tab3:
    seed_input = st.number_input("Сид", 0, 999_999, st.session_state.seed, 1)
    if seed_input != st.session_state.seed:
        st.session_state.seed = seed_input
        st.rerun()
    if st.button("Новый случайный сид", use_container_width=True):
        st.session_state.seed = int(np.random.randint(0, 999_999))
        st.rerun()
    st.checkbox("Использовать MILP (pulp) для оптимизации", key="use_milp",
                disabled=not PULP_AVAILABLE,
                help="Если pulp недоступен, используется LP-релаксация.")


# --- Валидация ---
try:
    validate_portfolio_params(C_max, theta_high, L_high, reserve_pct)
    validate_economics_params(r, T_months, gamma, C_op)
except ValidationError as e:
    st.error(f"Ошибка параметров: {e}")
    st.stop()

# --- Данные ---
econ = DEFAULT_CONFIG.economics
econ.r, econ.T_months, econ.gamma, econ.C_op = r, T_months, gamma, C_op
scoring = DEFAULT_CONFIG.scoring
scoring.PD_THRESHOLD, scoring.PF_THRESHOLD = pd_thr, pf_thr

df = generate_applications(st.session_state.seed)
try:
    validate_dataframe(df)
except ValidationError as e:
    st.error(f"Ошибка данных: {e}")
    st.stop()

# --- Выбор стратегии ---
st.subheader("Выберите стратегию")
cols = st.columns(5)

btn_labels = ["MILP (оптимум)", "Предложенный", "Скоринг + Лимит", "Скоринг", "Жадный"]
btn_targets = ["MILP (оптимум)"] + STRATEGY_NAMES

for c, label, target in zip(cols, btn_labels, btn_targets):
    with c:
        st.button(
            label, on_click=set_strategy, args=(target,),
            type="primary" if st.session_state.current_strategy == target else "secondary",
            use_container_width=True,
        )
st.markdown("---")

# --- Выполнение выбранной стратегии ---
strategy = st.session_state.current_strategy

milp_result = None
if strategy == "MILP (оптимум)":
    if not PULP_AVAILABLE:
        st.warning("pulp не установлен, использую LP-релаксацию.")
        lp = solve_lp_relaxation(df, C_max, theta_high, L_high, reserve_pct, econ)
        x = (lp["x"] > 0.5).astype(float)
        results_df = milp_to_dataframe(df, x, econ)
        total_issued = float((df["requested_amount"].to_numpy() * x).sum())
        total_profit = float(results_df["expected_profit"].clip(lower=0).sum())
        exec_time = lp["exec_ms"]
    else:
        milp_result = solve_milp(df, C_max, theta_high, L_high, reserve_pct, econ)
        results_df = milp_to_dataframe(df, milp_result["x"], econ)
        total_issued = milp_result["issued"]
        total_profit = milp_result["profit"]
        exec_time = milp_result["exec_ms"]
    avg_loan = total_issued / max((results_df["decision"] == "Одобрено").sum(), 1)
else:
    results_df, total_issued, total_profit, avg_loan, exec_time = process_applications(
        df, strategy, C_max, theta_high, L_high, reserve_pct, econ, scoring)

# --- Графики потока ---
col_l, col_r = st.columns([2, 1])
with col_l:
    st.markdown(f"**Поток заявок. Сид: `{st.session_state.seed}`**")
    st.plotly_chart(plot_application_flow(df, results_df, L_high), use_container_width=True)
with col_r:
    st.markdown("**Средние суммы заявок**")
    st.plotly_chart(plot_avg_amounts(df, L_high), use_container_width=True)

# --- Метрики ---
from profit_model import compute_profit_vector, opportunity_cost as oc
profits_vec = compute_profit_vector(df, econ)
opp = oc(profits_vec, total_profit)

metrics_html = f"""
<div style="display:flex;justify-content:space-between;gap:15px;margin-bottom:20px;flex-wrap:wrap;">
  <div style="flex:1;min-width:130px;"><div style="color:#888;">Ожидаемая прибыль</div>
  <div style="font-size:2rem;font-weight:600;">{total_profit:,.0f} ₽</div></div>
  <div style="flex:1.5;min-width:180px;"><div style="color:#888;">Использовано лимита</div>
  <div style="font-size:2rem;font-weight:600;">{format_money(total_issued)} <span style="font-size:1.3rem;color:#888;">/ {format_money(C_max)}</span></div></div>
  <div style="flex:1;min-width:120px;"><div style="color:#888;">Одобрено заявок</div>
  <div style="font-size:2rem;font-weight:600;">{(results_df['decision']=='Одобрено').sum()} / {len(df)}</div></div>
  <div style="flex:1;min-width:130px;"><div style="color:#888;">Упущенная выгода</div>
  <div style="font-size:2rem;font-weight:600;color:#d9534f;">-{opp:,.0f} ₽</div></div>
</div>"""
st.markdown(metrics_html, unsafe_allow_html=True)
st.progress(min(total_issued / C_max, 1.0))

# --- Сравнение стратегий ---
st.subheader("Сравнение стратегий")
summary = []
for strat in STRATEGY_NAMES:
    r_df, r_issued, r_profit, _, r_time = process_applications(
        df, strat, C_max, theta_high, L_high, reserve_pct, econ, scoring)
    summary.append({
        "name": strat, "profit": r_profit, "issued": r_issued,
        "approved": int((r_df["decision"] == "Одобрено").sum()),
    })

# MILP для сравнения
milp_profit = None
if PULP_AVAILABLE:
    try:
        m = solve_milp(df, C_max, theta_high, L_high, reserve_pct, econ, time_limit_s=5)
        milp_profit = m["profit"]
        summary.append({
            "name": "MILP (оптимум)", "profit": m["profit"], "issued": m["issued"],
            "approved": int((m["x"] > 0.5).sum()),
        })
    except Exception as e:
        st.warning(f"MILP недоступен: {e}")

st.plotly_chart(plot_strategies_comparison(summary, strategy, milp_profit),
                use_container_width=True)

# --- Граница эффективности ---
st.subheader("Эффективность стратегий: риск vs прибыль")
frontier_points = []
for s in summary:
    r_df, _, _, _, _ = process_applications(
        df, s["name"] if s["name"] != "MILP (оптимум)" else "Жадный (без ограничений)",
        C_max, theta_high, L_high, reserve_pct, econ, scoring)
    approved_df = r_df[r_df["decision"] == "Одобрено"]
    hr = approved_df[approved_df["amount"] >= L_high]
    risk_share = len(hr) / max(len(approved_df), 1)
    frontier_points.append({
        "name": s["name"], "risk_share": risk_share, "profit": s["profit"],
    })
st.plotly_chart(plot_efficient_frontier(frontier_points), use_container_width=True)

# --- LP/MILP диагностика ---
with st.expander("Диагностика оптимизатора"):
    lp = solve_lp_relaxation(df, C_max, theta_high, L_high, reserve_pct, econ)
    st.write(f"**LP upper bound:** {lp['profit_bound']:,.0f} ₽ "
             f"(время: {lp['exec_ms']:.1f} мс, статус: `{lp['status']}`)")
    if milp_profit is not None:
        gap = (lp["profit_bound"] - milp_profit) / max(abs(milp_profit), 1e-9) * 100
        st.write(f"**MILP:** {milp_profit:,.0f} ₽ · **MIP-gap:** {gap:.2f} %")

# --- Журнал ---
st.subheader("Журнал решений")
def hl(v):
    if v == "Одобрено": return "background-color:#85e085;color:black;font-weight:bold;"
    if v == "Отказано": return "background-color:#ff9999;color:black;font-weight:bold;"
    return ""

st.dataframe(results_df.style.applymap(hl, subset=["decision"]),
             use_container_width=True, height=350)
csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button("Скачать решения (CSV)", csv,
                   f"decisions_seed_{st.session_state.seed}.csv", "text/csv")
