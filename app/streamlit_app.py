"""Streamlit dashboard: English primary care prescribing trends. Run: streamlit run app/streamlit_app.py"""
import json, pathlib
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

P = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="NHS Prescribing Trends", layout="wide")

@st.cache_data
def load():
    return (pd.read_csv(P / "national_monthly.csv", parse_dates=["month"]), pd.read_csv(P / "substance_monthly.csv", parse_dates=["month"]),
            pd.read_csv(P / "sicbl_variation_fy2526.csv"), pd.read_csv(P / "forecast.csv", parse_dates=["month"]),
            json.loads((P / "key_findings.json").read_text()))
nat, sub, var, fc, F = load()

st.title("English primary care prescribing: antidepressants, antibiotics, diabetes and lipid drugs")
st.caption(f"NHSBSA English Prescribing Dataset, {F['first_month']} to {F['latest_month']} (Open Government Licence). "
           "Built by Anthony Adusei-Poku (github.com/anthony07-max).")
area = st.sidebar.selectbox("Therapeutic area", list(F["area_summary"]))
s = F["area_summary"][area]
c = st.columns(4)
c[0].metric(f"Items, {F['latest_12m']}", f"{s['items_latest12']/1e6:.1f}m", f"{s['items_growth_pct']:+.1f}% vs 2021")
c[1].metric("Actual cost", f"£{s['cost_latest12']/1e6:.0f}m", f"{s['cost_growth_pct']:+.1f}% vs 2021", delta_color="inverse")
c[2].metric("Cost per item", f"£{s['cost_per_item_latest12']:.2f}", f"£{s['cost_per_item_latest12']-s['cost_per_item_2021']:+.2f} vs 2021", delta_color="inverse")
lv = F["london_vs_rest_items_per_1000"][area]
c[3].metric("London items per 1,000 (FY25/26)", f"{lv['London']:,}", f"{lv['London_vs_rest_pct']:+.1f}% vs rest of England", delta_color="off")

t1, t2, t3, t4 = st.tabs(["Trends", "Drugs driving change", "Variation & outliers", "Forecast"])
with t1:
    metric = st.radio("Measure", ["n_items", "actual_cost", "cost_per_item"], horizontal=True)
    d = nat.assign(cost_per_item=nat.actual_cost / nat.n_items)
    st.plotly_chart(px.line(d, x="month", y=metric, color="area", title=f"{metric} by month"), width="stretch")
with t2:
    d = sub[sub.area == area]
    top = d.groupby("substance").n_items.sum().nlargest(st.slider("Top substances", 3, 15, 8)).index
    by = st.radio("Show", ["n_items", "actual_cost"], horizontal=True, key="sub")
    st.plotly_chart(px.area(d[d.substance.isin(top)], x="month", y=by, color="substance", title=f"{area}: top substances"), width="stretch")
with t3:
    v = var[var.area == area].copy()
    r0 = v.national_rate_per_1000.iloc[0] / 1000; phi = v.phi.iloc[0]
    n = np.linspace(v.population.min() * .8, v.population.max() * 1.1, 200)
    fig = go.Figure()
    for zc, dash in [(1.96, "dash"), (3.09, "dot")]:
        for sgn in (1, -1):
            fig.add_trace(go.Scatter(x=n, y=(r0 + sgn * zc * np.sqrt(phi * r0 / n)) * 1000, mode="lines", showlegend=False,
                                     line=dict(color="#003087", dash=dash, width=1)))
    fig.add_trace(go.Scatter(x=v.population, y=v.items_per_1000, mode="markers", text=v.sicbl_name, name="Sub-ICB location",
                             marker=dict(color=np.where(v.london, "#DA291C", "#A0AEC0"), size=9)))
    fig.add_hline(y=r0 * 1000, line_color="#005EB8")
    fig.update_layout(title=f"{area}: funnel plot, FY 2025/26 (London in red; 95% and 99.8% overdispersed limits)",
                      xaxis_title="Resident population", yaxis_title="Items per 1,000 residents", height=550)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(v[["sicbl_name", "icb_name", "population", "pct_65_plus", "items_per_1000", "cost_per_1000", "cost_per_item",
                    "z_overdispersed", "z_age_adjusted", "outlier"]].sort_values("z_overdispersed", ascending=False).round(2),
                 width="stretch", hide_index=True)
with t4:
    h = nat[nat.area == area]; f = fc[fc.area == area]
    fig = go.Figure([go.Scatter(x=h.month, y=h.n_items, name="Actual"),
                     go.Scatter(x=f.month, y=f.forecast, name="Forecast", line=dict(dash="dash")),
                     go.Scatter(x=list(f.month) + list(f.month[::-1]), y=list(f.upper) + list(f.lower[::-1]), fill="toself",
                                opacity=.2, line=dict(width=0), name="Approx. 95% band")])
    st.plotly_chart(fig, width="stretch")
    m = F["forecast"][area]
    st.write(f"Holt-Winters (damped additive trend, multiplicative seasonality). 12-month backtest MAPE {m['backtest_mape_12m']}% "
             f"vs {m['seasonal_naive_mape_12m']}% for a seasonal naive benchmark. Next 12 months: {m['next_12m_items_m']}m items.")
