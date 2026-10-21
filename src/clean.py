"""Combine the monthly prescribing extracts and the ONS population file into tidy tables in data/processed/.

Outputs
- national_monthly.csv    : England items, cost, NIC by therapeutic area and month
- sicbl_monthly.csv       : items and cost by sub-ICB location (SICBL), area and month
- substance_monthly.csv   : England items and cost by chemical substance, area and month
- sicbl_population.csv    : ONS mid-2024 resident population and % aged 65+ for 2024 SICBLs
"""
import glob, pathlib, re
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
LABELS = {"antidepressants": "Antidepressants", "antibacterials": "Antibacterials",
          "diabetes": "Diabetes drugs", "lipid_regulating": "Lipid-regulating drugs"}

def stack(kind):
    out = []
    for f in sorted(glob.glob(str(RAW / f"epd_*_{kind}.csv"))):
        d = pd.read_csv(f).drop(columns="YEAR_MONTH")  # YEAR_MONTH format varies by month; use the file's month
        d["month"] = pd.to_datetime(re.search(r"epd_(\d{6})", f).group(1), format="%Y%m")
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["area"] = d["area"].map(LABELS)
    return d

# Population: ONS mid-2024 estimates for 2024 sub-ICB locations ------------------------------
pop = pd.read_excel(RAW / "ons_sicbl_population.xlsx", sheet_name="Mid-2024 ICB 2024", header=3)
age_cols = [c for c in pop.columns if re.fullmatch(r"[FM]\d+", str(c))]
over65 = [c for c in age_cols if int(c[1:]) >= 65]
pop = pd.DataFrame({
    "sicbl_ods": pop["SICBL 2024 Name"].str.extract(r"- (\w+)$")[0],
    "sicbl_code": pop["SICBL 2024 Code"], "sicbl_name": pop["SICBL 2024 Name"],
    "icb_name": pop["ICB 2024 Name"].str.replace(" Integrated Care Board", ""), "region": pop["NHSER 2024 Name"],
    "population": pop["Total"], "pct_65_plus": (pop[over65].sum(axis=1) / pop["Total"] * 100).round(2)})
pop["london"] = pop.region.eq("London")
pop.to_csv(PROC / "sicbl_population.csv", index=False)

# Prescribing by sub-ICB ----------------------------------------------------------------------
s = stack("sicbl")
code5, code3 = s.PCO_CODE.str[:5], s.PCO_CODE.str[:3]
s["sicbl_ods"] = code5.where(code5.isin(pop.sicbl_ods), code3.where(code3.isin(pop.sicbl_ods)))
s = s.rename(columns={"items": "n_items"})
nat = s.groupby(["month", "area"], as_index=False)[["n_items", "actual_cost", "nic"]].sum()
nat.to_csv(PROC / "national_monthly.csv", index=False)
sic = (s.dropna(subset=["sicbl_ods"]).groupby(["month", "sicbl_ods", "area"], as_index=False)[["n_items", "actual_cost"]].sum())
sic.to_csv(PROC / "sicbl_monthly.csv", index=False)

sub = stack("substance").rename(columns={"items": "n_items"})
sub.to_csv(PROC / "substance_monthly.csv", index=False)

unm = s[s.sicbl_ods.isna()].groupby("month").n_items.sum() / s.groupby("month").n_items.sum() * 100
print(f"{nat.month.min():%b %Y} to {nat.month.max():%b %Y}: {nat.n_items.sum()/1e6:.1f}m items, "
      f"£{nat.actual_cost.sum()/1e6:.0f}m actual cost; {len(pop)} SICBLs")
print("Items not attributable to a 2024 SICBL (out-of-hours, trusts, merged orgs), % by month: "
      f"min {unm.min():.2f}, max {unm.max():.2f}")
