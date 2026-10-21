"""Download monthly English prescribing aggregates and population denominators into data/raw/.

Prescribing: NHSBSA Open Data Portal, English Prescribing Dataset (EPD) with SNOMED code,
https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code
(Open Government Licence v3.0). The EPD has one row per practice x presentation per month (~18m rows),
so instead of downloading whole files this script asks the portal's CKAN `datastore_search_sql`
endpoint to aggregate each month server-side, for four BNF areas, by sub-ICB location and by chemical substance.

Population: ONS mid-year population estimates for 2024 sub-ICB locations (mid-2022 to mid-2024),
https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/clinicalcommissioninggroupmidyearpopulationestimates
(Open Government Licence v3.0).
"""
import pathlib, time, requests, pandas as pd

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
API = "https://opendata.nhsbsa.net/api/3/action/"
START = "202101"
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio project; NHSBSA open data)"}
ONS_POP = ("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/"
           "datasets/clinicalcommissioninggroupmidyearpopulationestimates/"
           "mid2022revisednov2025tomid2024integratedcareboards2024geography/sapeicb20222024.xlsx")

# BNF sections used to define each therapeutic area
AREAS = {"antidepressants": ["0403"],           # BNF 4.3 Antidepressant drugs
         "antibacterials": ["0501"],            # BNF 5.1 Antibacterial drugs
         "diabetes": ["060101", "060102"],      # BNF 6.1.1 Insulins, 6.1.2 Antidiabetic drugs
         "lipid_regulating": ["0212"]}          # BNF 2.12 Lipid-regulating drugs (mainly statins)

def columns(resource):
    """Column names changed over time (STP->ICB in July 2022, BNF_CODE->BNF_PRESENTATION_CODE in 2025)."""
    r = requests.get(API + "datastore_search", params={"resource_id": resource, "limit": 1}, headers=HEADERS, timeout=120)
    f = {x["id"] for x in r.json()["result"]["fields"]}
    pick = lambda *opts: next(o for o in opts if o in f)
    return dict(bnf=pick("BNF_PRESENTATION_CODE", "BNF_CODE"), region=pick("ICB_CODE", "STP_CODE"),
                region_name=pick("ICB_NAME", "STP_NAME"),
                substance=pick("CHEMICAL_SUBSTANCE_BNF_DESCR", "BNF_CHEMICAL_SUBSTANCE")
                if "BNF_CHEMICAL_SUBSTANCE_CODE" not in f else "BNF_CHEMICAL_SUBSTANCE")

def sql(query, resource):
    for attempt in range(5):
        try:
            r = requests.get(API + "datastore_search_sql", params={"resource_id": resource, "sql": query},
                             headers=HEADERS, timeout=300)
            d = r.json()
            if not d.get("success"):
                raise RuntimeError(d.get("error"))
            return pd.DataFrame(d["result"]["result"]["records"])
        except Exception as e:  # transient API errors: back off and retry
            print("  retry", attempt + 1, e); time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Query failed for {resource}")

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    pkg = requests.get(API + "package_show", params={"id": "english-prescribing-dataset-epd-with-snomed-code"},
                       headers=HEADERS, timeout=60).json()["result"]
    months = sorted(r["name"].split("_")[-1] for r in pkg["resources"] if r["name"].startswith("EPD_SNOMED_"))
    months = [m for m in months if m >= START]
    print(f"{len(months)} months available: {months[0]} to {months[-1]}")
    for m in months:
        res = f"EPD_SNOMED_{m}"
        f1, f2 = RAW / f"epd_{m}_sicbl.csv", RAW / f"epd_{m}_substance.csv"
        if f1.exists() and f2.exists():
            continue
        c = columns(res)
        CASE = "CASE " + " ".join(f"WHEN {c['bnf']} LIKE '{p}%' THEN '{a}'" for a, ps in AREAS.items() for p in ps) + " END"
        WHERE = " OR ".join(f"{c['bnf']} LIKE '{p}%'" for ps in AREAS.values() for p in ps)
        if not f1.exists():
            sql(f"SELECT YEAR_MONTH, {c['region']} AS ICB_CODE, {c['region_name']} AS ICB_NAME, PCO_CODE, PCO_NAME, {CASE} AS area, SUM(SAFE_CAST(ITEMS AS FLOAT64)) AS items, "
                f"SUM(SAFE_CAST(ACTUAL_COST AS FLOAT64)) AS actual_cost, SUM(SAFE_CAST(NIC AS FLOAT64)) AS nic FROM `{res}` WHERE {WHERE} GROUP BY 1,2,3,4,5,6", res).to_csv(f1, index=False)
        if not f2.exists():
            sql(f"SELECT YEAR_MONTH, {CASE} AS area, {c['substance']} AS substance, SUM(SAFE_CAST(ITEMS AS FLOAT64)) AS items, "
                f"SUM(SAFE_CAST(ACTUAL_COST AS FLOAT64)) AS actual_cost, SUM(SAFE_CAST(NIC AS FLOAT64)) AS nic FROM `{res}` WHERE {WHERE} GROUP BY 1,2,3", res).to_csv(f2, index=False)
        print("  ", m, "ok")
    r = requests.get(ONS_POP, headers=HEADERS, timeout=120); r.raise_for_status()
    (RAW / "ons_sicbl_population.xlsx").write_bytes(r.content)
    print("Done")

if __name__ == "__main__":
    main()
