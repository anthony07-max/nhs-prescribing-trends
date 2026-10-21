"""SQL analysis, variation/outlier detection, forecasts and charts. Run after clean.py.
Writes charts/*.png, data/processed/*.csv summary tables and data/processed/key_findings.json."""
import json, sqlite3, pathlib, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, CH = ROOT / "data" / "processed", ROOT / "charts"
CH.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "font.size": 10})
NHS_BLUE, NHS_DARK, RED, GREY, GREEN = "#005EB8", "#003087", "#DA291C", "#A0AEC0", "#009639"
AREAS = ["Antidepressants", "Antibacterials", "Diabetes drugs", "Lipid-regulating drugs"]
COL = dict(zip(AREAS, [NHS_BLUE, RED, "#ED8B00", GREEN]))

nat = pd.read_csv(PROC / "national_monthly.csv", parse_dates=["month"])
sic = pd.read_csv(PROC / "sicbl_monthly.csv", parse_dates=["month"])
sub = pd.read_csv(PROC / "substance_monthly.csv", parse_dates=["month"])
pop = pd.read_csv(PROC / "sicbl_population.csv")
con = sqlite3.connect(":memory:")
for name, df in [("nat", nat), ("sic", sic), ("sub", sub)]:
    df.assign(month=df.month.dt.strftime("%Y-%m")).to_sql(name, con)
pop.to_sql("pop", con)
q = lambda sql: pd.read_sql(sql, con)
last = nat.month.max()
L12 = (last - pd.DateOffset(months=11)).strftime("%Y-%m")
F = {"first_month": nat.month.min().strftime("%B %Y"), "latest_month": last.strftime("%B %Y"),
     "latest_12m": f"{(last - pd.DateOffset(months=11)):%b %Y} to {last:%b %Y}"}

# 1. National trends: calendar 2021 vs latest 12 months (SQL) -----------------------------------
tr = q(f"""SELECT area,
      SUM(CASE WHEN month BETWEEN '2021-01' AND '2021-12' THEN n_items END) AS items_2021,
      SUM(CASE WHEN month >= '{L12}' THEN n_items END) AS items_latest12,
      SUM(CASE WHEN month BETWEEN '2021-01' AND '2021-12' THEN actual_cost END) AS cost_2021,
      SUM(CASE WHEN month >= '{L12}' THEN actual_cost END) AS cost_latest12
    FROM nat GROUP BY area""")
tr["items_growth_pct"] = ((tr.items_latest12 / tr.items_2021 - 1) * 100).round(1)
tr["cost_growth_pct"] = ((tr.cost_latest12 / tr.cost_2021 - 1) * 100).round(1)
tr["cost_per_item_2021"] = (tr.cost_2021 / tr.items_2021).round(2)
tr["cost_per_item_latest12"] = (tr.cost_latest12 / tr.items_latest12).round(2)
tr.to_csv(PROC / "area_summary.csv", index=False)
F["area_summary"] = tr.set_index("area").round(1).to_dict("index")

fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for a, ax in zip(AREAS, axs.flat):
    d = nat[nat.area == a].set_index("month")
    ax.plot(d.index, d.n_items / 1e6, color=COL[a], alpha=.35, lw=1, label="Monthly")
    ax.plot(d.index, d.n_items.rolling(12).mean() / 1e6, color=COL[a], lw=2.4, label="12-month average")
    g = F["area_summary"][a]["items_growth_pct"]
    ax.set_title(f"{a} ({g:+.1f}% items, 2021 to latest 12m)"); ax.set_ylim(0, None); ax.set_ylabel("Items (millions)")
axs[0, 0].legend(loc="lower left")
fig.suptitle(f"Items prescribed in English primary care, {F['first_month']} to {F['latest_month']}")
fig.tight_layout(); fig.savefig(CH / "01_items_trend.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for a in AREAS:
    d = nat[nat.area == a].set_index("month")
    axs[0].plot(d.index, d.actual_cost.rolling(3).mean() / 1e6, color=COL[a], lw=2, label=a)
    axs[1].plot(d.index, (d.actual_cost / d.n_items).rolling(3).mean(), color=COL[a], lw=2, label=a)
axs[0].set(title="Actual cost per month (£m, 3-month average)"); axs[0].legend(fontsize=8)
axs[1].set(title="Cost per item (£, 3-month average)"); axs[1].yaxis.set_major_formatter(mt.StrMethodFormatter("£{x:.0f}"))
fig.tight_layout(); fig.savefig(CH / "02_cost_and_cost_per_item.png"); plt.close(fig)

# 2. Substance drivers ---------------------------------------------------------------------------
sd = q(f"""SELECT area, substance,
      SUM(CASE WHEN month BETWEEN '2021-01' AND '2021-12' THEN n_items ELSE 0 END) AS items_2021,
      SUM(CASE WHEN month >= '{L12}' THEN n_items ELSE 0 END) AS items_latest12,
      SUM(CASE WHEN month BETWEEN '2021-01' AND '2021-12' THEN actual_cost ELSE 0 END) AS cost_2021,
      SUM(CASE WHEN month >= '{L12}' THEN actual_cost ELSE 0 END) AS cost_latest12
    FROM sub GROUP BY area, substance""")
sd["cost_change"] = sd.cost_latest12 - sd.cost_2021
sd["items_change"] = sd.items_latest12 - sd.items_2021
sd.to_csv(PROC / "substance_change.csv", index=False)
top_cost = sd.nlargest(10, "cost_change")
F["top_cost_increase_substances"] = top_cost[["substance", "area", "cost_2021", "cost_latest12"]].round(0).to_dict("records")
F["top_substance_by_items_latest12"] = {a: g.nlargest(1, "items_latest12")[["substance", "items_latest12"]].iloc[0].tolist()
                                        for a, g in sd.groupby("area")}
dia = sd[sd.area == "Diabetes drugs"]
F["diabetes_cost_change_total"] = round(float(dia.cost_change.sum()))
F["glp1_tirzepatide_semaglutide_share_of_diabetes_cost_increase_pct"] = round(
    float(dia[dia.substance.isin(["Tirzepatide", "Semaglutide", "Dulaglutide", "Liraglutide", "Exenatide", "Lixisenatide"])]
          .cost_change.sum() / dia.cost_change.sum() * 100), 1)
fig, ax = plt.subplots(figsize=(10, 5.5))
d = top_cost.iloc[::-1]
ax.barh(d.substance + " (" + d.area.str.split().str[0] + ")", d.cost_change / 1e6, color=[COL[a] for a in d.area])
ax.set(title=f"Biggest increases in annual cost: 2021 vs {F['latest_12m']} (£m)", xlabel="Change in annual actual cost, £m")
fig.tight_layout(); fig.savefig(CH / "03_cost_drivers.png"); plt.close(fig)

# 3. Variation across sub-ICB locations, financial year 2025/26 ------------------------------------
# FY 2025/26 (Apr 2025 - Mar 2026) is the latest full year on the 2024 SICBL geography that matches
# the ONS population file (several ICBs merged in April 2026).
FY = ("2025-04", "2026-03")
v = q(f"""SELECT s.sicbl_ods, p.sicbl_name, p.icb_name, p.region, p.london, p.population, p.pct_65_plus, s.area,
                 SUM(s.n_items) AS n_items, SUM(s.actual_cost) AS actual_cost
          FROM sic s JOIN pop p USING (sicbl_ods)
          WHERE s.month BETWEEN '{FY[0]}' AND '{FY[1]}'
          GROUP BY s.sicbl_ods, s.area""")
v["london"] = v.london.astype(bool)
v["items_per_1000"] = v.n_items / v.population * 1000
v["cost_per_1000"] = v.actual_cost / v.population * 1000
v["cost_per_item"] = v.actual_cost / v.n_items
F["fy"] = "2025/26"

def funnel(d):
    """Funnel plot statistics for a rate per 1,000 with overdispersion (Spiegelhalter 2005):
    z = (r - r0) / sqrt(r0 / n) on the Poisson scale, phi = mean of 10% winsorised z^2."""
    r0 = d.n_items.sum() / d.population.sum()
    r = d.n_items / d.population
    z = (r - r0) / np.sqrt(r0 / d.population)
    zw = z.clip(z.quantile(.1), z.quantile(.9))
    phi = float((zw ** 2).mean())
    return r0, z / np.sqrt(phi), phi

out = []
for a, d in v.groupby("area"):
    r0, zadj, phi = funnel(d)
    d = d.assign(national_rate_per_1000=r0 * 1000, z_overdispersed=zadj.round(2), phi=round(phi, 1))
    # Age-adjusted z: residual from a linear fit of rate on % aged 65+, standardised
    b = np.polyfit(d.pct_65_plus, d.items_per_1000, 1)
    res = d.items_per_1000 - np.polyval(b, d.pct_65_plus)
    d["expected_per_1000_age"] = np.polyval(b, d.pct_65_plus)
    d["z_age_adjusted"] = ((res - res.mean()) / res.std()).round(2)
    d["corr_with_pct65"] = round(float(np.corrcoef(d.pct_65_plus, d.items_per_1000)[0, 1]), 2)
    out.append(d)
v = pd.concat(out)
v["outlier"] = np.select([v.z_overdispersed > 3.09, v.z_overdispersed < -3.09], ["High (>99.8%)", "Low (<99.8%)"], "Within limits")
v.round(3).to_csv(PROC / "sicbl_variation_fy2526.csv", index=False)
F["corr_rate_vs_pct65"] = v.groupby("area").corr_with_pct65.first().to_dict()
F["n_sicbl"] = int(v.sicbl_ods.nunique())
F["outliers_count"] = v[v.outlier != "Within limits"].groupby(["area", "outlier"]).size().unstack(fill_value=0).to_dict("index")
F["variation_ratio_max_min"] = v.groupby("area").items_per_1000.agg(lambda s: round(s.max() / s.min(), 1)).to_dict()
F["variation_ratio_p95_p5"] = v.groupby("area").items_per_1000.agg(lambda s: round(s.quantile(.95) / s.quantile(.05), 2)).to_dict()
F["outliers"] = v[v.outlier != "Within limits"][["area", "sicbl_name", "items_per_1000", "z_overdispersed", "z_age_adjusted"]].round(1).to_dict("records")
F["highest_rate_sicbl"] = {a: [g.nlargest(1, "items_per_1000").sicbl_name.iloc[0], round(g.items_per_1000.max())]
                           for a, g in v.groupby("area")}
F["lowest_rate_sicbl"] = {a: [g.nsmallest(1, "items_per_1000").sicbl_name.iloc[0], round(g.items_per_1000.min())]
                          for a, g in v.groupby("area")}

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
for a, ax in zip(["Antidepressants", "Antibacterials"], axs):
    d = v[v.area == a]; r0 = d.national_rate_per_1000.iloc[0] / 1000; phi = d.phi.iloc[0]
    n = np.linspace(d.population.min() * .8, d.population.max() * 1.1, 300)
    for zc, ls in [(1.96, "--"), (3.09, ":")]:
        for sgn in (1, -1):
            ax.plot(n / 1e6, (r0 + sgn * zc * np.sqrt(phi * r0 / n)) * 1000, color=NHS_DARK, ls=ls, lw=1)
    ax.axhline(r0 * 1000, color=NHS_DARK, lw=1)
    ax.scatter(d.population / 1e6, d.items_per_1000, c=np.where(d.london, RED, GREY), s=28, zorder=3)
    for _, r in d[d.london].iterrows():
        short = "".join(w[0] for w in r.sicbl_name.replace("NHS ", "").split(" ICB")[0].split())  # e.g. NEL
        ax.annotate(short, (r.population / 1e6, r.items_per_1000), fontsize=8, color=RED, xytext=(0, 6),
                    textcoords="offset points", ha="center")
    ax.set_ylim(d.items_per_1000.min() * .6, d.items_per_1000.max() * 1.2)
    ax.set(title=f"{a}: items per 1,000 residents, FY 2025/26", xlabel="Resident population (millions)",
           ylabel="Items per 1,000 residents")
axs[0].plot([], [], "--", color=NHS_DARK, label="95% limits (overdispersed)"); axs[0].plot([], [], ":", color=NHS_DARK, label="99.8% limits")
axs[0].scatter([], [], color=RED, label="London (NCL, NEL, NWL, SEL, SWL)"); axs[0].legend(fontsize=8)
fig.suptitle("Funnel plots across 106 sub-ICB locations in England")
fig.tight_layout(); fig.savefig(CH / "04_funnel_plots.png"); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(15, 4))
for a, ax in zip(AREAS, axs):
    d = v[v.area == a]
    ax.scatter(d.pct_65_plus, d.items_per_1000, c=np.where(d.london, RED, GREY), s=16)
    xs = np.linspace(d.pct_65_plus.min(), d.pct_65_plus.max(), 10)
    b = np.polyfit(d.pct_65_plus, d.items_per_1000, 1); ax.plot(xs, np.polyval(b, xs), color=NHS_DARK, lw=1)
    ax.set(title=f"{a}\nr = {d.corr_with_pct65.iloc[0]}", xlabel="% residents aged 65+")
axs[0].set_ylabel("Items per 1,000 residents")
fig.suptitle("Prescribing rate vs population age, by sub-ICB location (London in red), FY 2025/26")
fig.tight_layout(); fig.savefig(CH / "05_rate_vs_age.png"); plt.close(fig)

# 4. London focus --------------------------------------------------------------------------------
lon = v[v.london].copy()
lon["vs_england_pct"] = ((lon.items_per_1000 / lon.national_rate_per_1000 - 1) * 100).round(1)
lon["vs_age_expected_pct"] = ((lon.items_per_1000 / lon.expected_per_1000_age - 1) * 100).round(1)
lon.to_csv(PROC / "london_fy2526.csv", index=False)
F["london"] = {a: g.set_index(g.sicbl_name.str.replace("NHS ", "").str.split(" ICB").str[0])[
    ["items_per_1000", "vs_england_pct", "vs_age_expected_pct", "z_age_adjusted"]].round(1).to_dict("index")
    for a, g in lon.groupby("area")}
lr = v.groupby(["area", "london"]).agg(items=("n_items", "sum"), pop=("population", "sum")).reset_index()
lr["rate"] = lr["items"] / lr["pop"] * 1000
lr = lr.pivot(index="area", columns="london", values="rate")
F["london_vs_rest_items_per_1000"] = {a: {"London": round(r[True]), "Rest of England": round(r[False]),
                                         "London_vs_rest_pct": round((r[True] / r[False] - 1) * 100, 1)} for a, r in lr.iterrows()}
F["london_pct65"] = round(float((pop[pop.london].population * pop[pop.london].pct_65_plus).sum() / pop[pop.london].population.sum()), 1)
F["rest_pct65"] = round(float((pop[~pop.london].population * pop[~pop.london].pct_65_plus).sum() / pop[~pop.london].population.sum()), 1)
fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
for a, ax in zip(AREAS, axs):
    d = lon[lon.area == a].sort_values("sicbl_name")
    nm = d.sicbl_name.str.replace("NHS ", "").str.split(" ICB").str[0]
    ax.barh(nm, d.items_per_1000, color=COL[a], label="Actual")
    ax.scatter(d.expected_per_1000_age, nm, color=NHS_DARK, zorder=3, marker="|", s=250, label="Expected for age mix")
    ax.axvline(d.national_rate_per_1000.iloc[0], color=GREY, ls="--", lw=1, label="England")
    ax.set_title(a)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
fig.suptitle("London sub-ICB locations: items per 1,000 residents, FY 2025/26")
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(CH / "06_london_focus.png"); plt.close(fig)

# 5. Forecasts -----------------------------------------------------------------------------------
fc_all, F["forecast"] = [], {}
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
H = 12
for a, ax in zip(AREAS, axs.flat):
    y = nat[nat.area == a].set_index("month").n_items.asfreq("MS")
    fit = lambda s: ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12).fit()
    bt = fit(y[:-H]).forecast(H)
    mape = float((abs(bt.values - y[-H:].values) / y[-H:].values).mean() * 100)
    naive = float((abs(y[-H - 12:-12].values - y[-H:].values) / y[-H:].values).mean() * 100)  # seasonal naive
    m = fit(y); f = m.forecast(H)
    sd = float((y - m.fittedvalues).std())
    band = 1.96 * sd * np.sqrt(np.arange(1, H + 1)) ** .5
    fc_all.append(pd.DataFrame({"area": a, "month": f.index, "forecast": f.round(), "lower": (f - band).round(), "upper": (f + band).round()}))
    F["forecast"][a] = {"backtest_mape_12m": round(mape, 1), "seasonal_naive_mape_12m": round(naive, 1),
                        "next_12m_items_m": round(float(f.sum()) / 1e6, 1), "last_12m_items_m": round(float(y[-12:].sum()) / 1e6, 1)}
    ax.plot(y.index, y / 1e6, color=COL[a], lw=1.6, label="Actual")
    ax.plot(bt.index, bt / 1e6, color=GREY, lw=1.2, ls="--", label=f"Backtest (MAPE {mape:.1f}%)")
    ax.plot(f.index, f / 1e6, color=NHS_DARK, lw=2, ls="--", label="Forecast")
    ax.fill_between(f.index, (f - band) / 1e6, (f + band) / 1e6, color=NHS_DARK, alpha=.12)
    ax.set(title=a, ylabel="Items (millions)"); ax.legend(fontsize=7, loc="lower left")
fig.suptitle("Monthly items: Holt-Winters forecasts (damped trend, multiplicative seasonality)")
fig.tight_layout(); fig.savefig(CH / "07_forecasts.png"); plt.close(fig)
pd.concat(fc_all).to_csv(PROC / "forecast.csv", index=False)

(PROC / "key_findings.json").write_text(json.dumps(F, indent=2, default=str))
print(json.dumps(F, indent=2, default=str))
