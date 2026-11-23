"""Интерактивные графики Plotly для DSS."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_application_flow(df: pd.DataFrame, results_df: pd.DataFrame,
                          L_high: float) -> go.Figure:
    approved_mask = results_df["decision"] == "Одобрено"
    rejected_mask = ~approved_mask

    fig = go.Figure()
    appr = df[approved_mask]
    rej = df[rejected_mask]
    colors = ["#ff4b4b" if a >= L_high else "#1f77b4" for a in appr["requested_amount"]]

    fig.add_trace(go.Scatter(
        x=appr["id"], y=appr["requested_amount"] / 1000,
        mode="markers", marker=dict(color=colors, size=7, opacity=0.85),
        name="Одобрено",
        customdata=np.stack([appr["p_default"], appr["p_fraud"]], axis=-1),
        hovertemplate="Заявка %{x}<br>Сумма: %{y:.0f} тыс. ₽<br>"
                      "p_def: %{customdata[0]:.3f}<br>p_fraud: %{customdata[1]:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rej["id"], y=rej["requested_amount"] / 1000,
        mode="markers", marker=dict(color="gray", size=5, symbol="x", opacity=0.5),
        name="Отказано",
        hovertemplate="Заявка %{x}<br>Сумма: %{y:.0f} тыс. ₽<extra></extra>",
    ))
    fig.add_hline(y=L_high / 1000, line_dash="dash", line_color="black",
                  annotation_text=f"Порог L_high = {L_high/1000:.0f} тыс. ₽")
    fig.update_layout(
        xaxis_title="Номер заявки (время)",
        yaxis_title="Сумма (тыс. ₽)",
        height=380, margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_avg_amounts(df: pd.DataFrame, L_high: float) -> go.Figure:
    small = df[df["requested_amount"] < L_high]["requested_amount"]
    large = df[df["requested_amount"] >= L_high]["requested_amount"]
    avg_small = small.mean() if len(small) else 0.0
    avg_large = large.mean() if len(large) else 0.0
    fig = go.Figure(go.Bar(
        x=[avg_small / 1000, avg_large / 1000],
        y=["Обычные", "Крупные"],
        orientation="h",
        marker_color=["#1f77b4", "#ff4b4b"],
        text=[f"{avg_small/1000:.0f} тыс.", f"{avg_large/1000:.0f} тыс."],
        textposition="outside",
    ))
    fig.add_vline(x=L_high / 1000, line_dash="dash", line_color="black",
                  annotation_text=f"Порог {L_high/1000:.0f} тыс.")
    fig.update_layout(
        xaxis_title="Тыс. ₽", height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


def plot_strategies_comparison(summary: list, selected: str,
                               milp_profit: float | None = None) -> go.Figure:
    summary = sorted(summary, key=lambda x: x["approved"])
    names = [s["name"] for s in summary]
    counts = [s["approved"] for s in summary]
    colors = ["#ff4b4b" if n == selected else "#a0a0a0" for n in names]

    fig = go.Figure(go.Bar(
        x=counts, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color="black", width=1.5)),
        text=[f'{s["approved"]} заявок · прибыль {s["profit"]:,.0f} ₽' for s in summary],
        textposition="outside",
        customdata=[[s["profit"], s["issued"]] for s in summary],
        hovertemplate="Стратегия: %{y}<br>Одобрено: %{x}<br>"
                      "Прибыль: %{customdata[0]:,.0f} ₽<br>"
                      "Выдано: %{customdata[1]:,.0f} ₽<extra></extra>",
    ))
    if milp_profit is not None:
        fig.add_vline(x=max(counts) * 1.05, line_dash="dot", line_color="green",
                      annotation_text=f"MILP прибыль: {milp_profit:,.0f} ₽")
    fig.update_layout(
        xaxis_title="Количество одобренных заявок",
        height=320, margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(range=[0, max(counts) * 1.5]),
    )
    return fig


def plot_efficient_frontier(points: list) -> go.Figure:
    """
    Точки: [{'name': str, 'risk_share': float, 'profit': float}]
    Строит кривую «доля высокорисковых vs прибыль».
    """
    fig = go.Figure()
    names = [p["name"] for p in points]
    risks = [p["risk_share"] * 100 for p in points]
    profits = [p["profit"] for p in points]

    fig.add_trace(go.Scatter(
        x=risks, y=profits, mode="markers+text",
        text=names, textposition="top center",
        marker=dict(size=12, color=profits, colorscale="Viridis", showscale=True),
        hovertemplate="%{text}<br>Доля риска: %{x:.1f}%<br>Прибыль: %{y:,.0f} ₽<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Доля высокорисковых кредитов (%)",
        yaxis_title="Ожидаемая прибыль портфеля (₽)",
        height=360, margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig