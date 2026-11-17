"""Projection engine for the African Field Epidemiology Workforce Model.

The engine contains conditional mentor-capacity, workload-attrition,
unabsorbed-pool emigration, and post-creation feedbacks. Their parameters are
analyst-selected scenarios, not empirically estimated causal effects. The
model distinguishes serving staff from unabsorbed public-sector seekers and
transfers existing people between tiers when internal progression occurs.
Structural checks are in ``fetp_validate.py``; they do not establish empirical
fit or predictive validity. This module has no Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

DATA_FILE = "fetp_countries.csv"

TIERS = ("advanced", "intermediate", "frontline")
TIER_LABEL = {"advanced": "Advanced", "intermediate": "Intermediate", "frontline": "Frontline"}

BASE_YEAR = 2026
HORIZON_DEFAULT = 25
DT_DEFAULT = 1.0 / 12.0
EPS = 1e-9

AGE_MIN, AGE_MAX = 22, 70
AGES = np.arange(AGE_MIN, AGE_MAX + 1)

ENTRY_AGE_MEAN_SD = {"advanced": (34.0, 6.0), "intermediate": (37.0, 7.0), "frontline": (35.0, 8.0)}


def _mortality(ages: np.ndarray) -> np.ndarray:
    """Gompertz-type all-cause hazard, per year. Calibrated loosely to adult
    mortality in sub-Saharan Africa; a scenario parameter, not a life table."""
    return 0.0015 * np.exp(0.055 * (ages - 25.0))


AGE_MORTALITY = _mortality(AGES)

DEFAULT_SKILL_WEIGHT = {"advanced": 1.00, "intermediate": 0.60, "frontline": 0.25}
DEFAULT_PROTECTED_TIME = {"advanced": 0.65, "intermediate": 0.35, "frontline": 0.15}
DEFAULT_ACTIVE_SHARE = {"advanced": 0.80, "intermediate": 0.80, "frontline": 0.75}
DEFAULT_TIER_MONTHS = {"advanced": 24.0, "intermediate": 9.0, "frontline": 3.0}
DEFAULT_COMPLETION = {"advanced": 0.85, "intermediate": 0.90, "frontline": 0.95}
DEFAULT_ATTRITION = {"advanced": 0.030, "intermediate": 0.030, "frontline": 0.035}
DEFAULT_EMIGRATION = {"advanced": 0.030, "intermediate": 0.020, "frontline": 0.010}

COST_PER_FTE_USD = {"KEN": 18000.0}
COST_NOTES = {
    "KEN": "Machakos County PSB epidemiologist post (Job Group Q, 2026): "
           "KES 105,570-139,790/mo basic, plus 15% employer PSSS pension and a "
           "bracketed allowance load. Range USD 15,500-21,600; 18,000 is the midpoint. "
           "Cross-checked against CEMA (KES 47.8bn / 41,170 workers = USD 8,980 all-cadre) "
           "and Council of Governors (KES 7.8bn / 7,476 = USD 8,070 all-cadre), both of "
           "which should and do sit below a professional epidemiologist post.",
}
TRAINING_COST_USD = {
    "frontline": (5000.0, 8000.0,
                  "Andre et al., EID 2017, 24 countries, 2015-16 cost base. "
                  "Published as a range with 'data not shown'. Needs inflating to 2026."),
    "intermediate": (6200.0, 6200.0,
                     "Wilson et al., Public Health Reports 2021. Tanzania 6-month "
                     "intermediate, USD 73,499 per course. Single-country analogue."),
    "advanced": (None, None,
                 "NO PUBLISHED FIGURE EXISTS for advanced-tier FETP cost per trainee, "
                 "for any country, anywhere. Treat as an explicit unknown."),
}

# iso3 -> (note, countries pooled, documented pooled total, documented host count).
# The default correction is applied only when the country table reports the
# documented pooled total; otherwise the table value is used as supplied and
# the app shows why. An `AdvancedHostShare` column in the country table
# overrides this rule for any country.
POOLED_ADVANCED = {
    "BFA": ("BFELTP pools 10 West African countries (2010-2022). The published 133 "
            "is NOT Burkinabe output. Per-nationality split is unpublished.", 10, 133.0, None),
    "CMR": ("CAFETP pools Cameroon, Chad, CAR, Gabon, Equatorial Guinea and Congo. "
            "Cohort 10 (Nov 2025) explicitly included Congolese, Gabonese and "
            "Central African nationals. Split is unpublished.", 6, None, None),
    "GHA": ("56 of 192 GFELTP advanced graduates (29%) are non-Ghanaian. "
            "Ghanaian-only advanced output is 139.", 1, 192.0, 139.0),
}

TIER_SHORT = {"advanced": "adv", "intermediate": "int", "frontline": "front"}


# ===========================================================================
# Data loading
# ===========================================================================

REQUIRED_COLS = {"country", "iso3", "region", "hosts_fetp", "pop_2026",
                 "grads_adv", "grads_int", "grads_front", "admin2_count"}

SIMPLE_COLS = {"Country", "Frontline", "Intermediate", "Advanced", "Population"}

ISO3 = {
    "Angola": "AGO", "Benin": "BEN", "Botswana": "BWA", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cameroon": "CMR", "Central African Republic": "CAF", "Chad": "TCD", "Ivory Coast": "CIV",
    "Djibouti": "DJI", "Ethiopia": "ETH", "Gambia": "GMB", "Ghana": "GHA", "Guinea": "GIN",
    "Kenya": "KEN", "Lesotho": "LSO", "Liberia": "LBR", "Madagascar": "MDG", "Malawi": "MWI",
    "Mali": "MLI", "Mauritania": "MRT", "Mozambique": "MOZ", "Namibia": "NAM", "Nigeria": "NGA",
    "Rwanda": "RWA", "Senegal": "SEN", "Sierra Leone": "SLE", "Somalia": "SOM",
    "South Africa": "ZAF", "South Sudan": "SSD", "Tanzania": "TZA", "Togo": "TGO",
    "Uganda": "UGA", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}
REGION = {
    "AGO": "Southern", "BWA": "Southern", "LSO": "Southern", "MWI": "Southern", "MOZ": "Southern",
    "NAM": "Southern", "ZAF": "Southern", "ZMB": "Southern", "ZWE": "Southern", "MDG": "Southern",
    "BEN": "West", "BFA": "West", "CIV": "West", "GMB": "West", "GHA": "West", "GIN": "West",
    "LBR": "West", "MLI": "West", "MRT": "West", "NGA": "West", "SEN": "West", "SLE": "West",
    "TGO": "West",
    "CMR": "Central", "CAF": "Central", "TCD": "Central",
    "BDI": "East", "DJI": "East", "ETH": "East", "KEN": "East", "RWA": "East", "SOM": "East",
    "SSD": "East", "TZA": "East", "UGA": "East",
}

REPLACEMENT_TFR = 2.1
WPP_ANCHORS = (2026, 2030, 2040, 2050, 2060, 2070)


def validate_countries(df: pd.DataFrame, source: str = "countries table") -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    errs = []
    if df["country"].duplicated().any():
        errs.append(f"duplicate countries: {sorted(df.loc[df.country.duplicated(), 'country'])}")
    for c in ("grads_adv", "grads_int", "grads_front"):
        v = pd.to_numeric(df[c], errors="coerce")
        if (v.notna() & ~np.isfinite(v.fillna(0))).any() or (v < 0).any():
            errs.append(f"{c} must be finite and non-negative")
        if (df[c].notna() & v.isna()).any():
            errs.append(f"{c} has non-numeric values")
    pop_cols = [c for c in df.columns if str(c).startswith("pop_2")]
    for c in pop_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if (v <= 0).any():
            errs.append(f"{c} must be positive")
    if "established" in df.columns:
        est = pd.to_numeric(df["established"], errors="coerce")
        if ((est > BASE_YEAR) | (est < 1950)).any():
            errs.append(f"established years must lie between 1950 and {BASE_YEAR}")
    if errs:
        raise ValueError(f"{source}: " + "; ".join(errs))
    return df.sort_values(["region", "country"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Is the rate column a fertility rate or a population growth rate?
# ---------------------------------------------------------------------------

def classify_rate_column(values: pd.Series) -> Tuple[str, str]:
    """Heuristically classify a numeric column as TFR or annual growth percent.

    This distribution-based rule is only a prompt to inspect the source
    metadata; the values themselves cannot confirm a column's meaning.
    Returns a verdict ('growth_pct', 'tfr' or 'ambiguous') and a reason.
    """
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return "ambiguous", "no numeric values"
    below_two = float((v < 2.0).mean())
    med = float(v.median())
    if below_two >= 0.15 or med < 2.6:
        return ("growth_pct",
                f"{below_two:.0%} of values are below 2.0 and the median is {med:.2f}; "
                "this distribution looks more like annual growth percent than TFR. "
                "Verify against the original source metadata.")
    if med > 3.4 and below_two < 0.05:
        return "tfr", (f"median {med:.2f} with few values below 2.0; this looks "
                        "more like TFR than growth percent. Verify source metadata.")
    return "ambiguous", (f"median {med:.2f}, {below_two:.0%} below 2.0; "
                         "the distribution does not resolve the column meaning")


def project_population_tfr(pop0: float, base_year: int, tfr0: float, years: np.ndarray,
                           growth_floor: float = 0.008, growth_slope: float = 0.0065,
                           gap_closure: float = 0.022, min_growth: float = -0.01) -> np.ndarray:
    """Population path from a single anchor and a total fertility rate.

    g = growth_floor + growth_slope x (TFR_t - 2.1), floored at min_growth, with
    the gap above replacement closing by gap_closure per year. Ignores mortality
    and age structure. A scenario rule, not a UN WPP substitute.
    """
    years = np.asarray(years)
    hi = int(max(base_year, years.max()))
    lo = int(min(base_year, years.min()))
    path = {base_year: pop0}
    for y in range(base_year + 1, hi + 1):
        gap = tfr0 - REPLACEMENT_TFR
        tfr = REPLACEMENT_TFR + gap * np.exp(-gap_closure * (y - 1 - base_year)) if gap > 0 else tfr0
        g = max(min_growth, growth_floor + growth_slope * (tfr - REPLACEMENT_TFR))
        path[y] = path[y - 1] * (1.0 + g)
    for y in range(base_year - 1, lo - 1, -1):
        path[y] = path[y + 1] / (1.0 + growth_floor)
    return np.array([path[int(y)] for y in years])


def project_population_growth(pop0: float, base_year: int, g0: float, years: np.ndarray,
                              decay: float = 0.015) -> np.ndarray:
    """Population path from an observed annual growth rate g0 (decimal).

    g0 is net: births - deaths + net migration, so mortality and migration are
    captured by construction. The rate declines geometrically by `decay` per
    year. UN WPP levels, where loaded, supersede this entirely.
    """
    years = np.asarray(years)
    hi = int(max(base_year, years.max()))
    lo = int(min(base_year, years.min()))
    path = {base_year: pop0}
    for y in range(base_year + 1, hi + 1):
        path[y] = path[y - 1] * (1.0 + g0 * np.exp(-decay * (y - 1 - base_year)))
    for y in range(base_year - 1, lo - 1, -1):
        path[y] = path[y + 1] / (1.0 + g0)
    return np.array([path[int(y)] for y in years])


def parse_growth(df: pd.DataFrame) -> Dict[str, float]:
    """iso3 -> annual growth rate (decimal) from a user-supplied table."""
    rate_col = next((c for c in ("growth_rate_pct", "growth_rate", "PopGrowthRate",
                                 "pop_growth_pct") if c in df.columns), None)
    key_col = next((c for c in ("iso3", "ISO3_code", "country", "Country") if c in df.columns), None)
    if rate_col is None or key_col is None:
        raise ValueError("Growth file needs an iso3/country column and growth_rate_pct "
                         "(percent per year).")
    out = {}
    for _, r in df.iterrows():
        iso = r[key_col] if key_col in ("iso3", "ISO3_code") else ISO3.get(str(r[key_col]).strip())
        if iso and pd.notna(r[rate_col]):
            out[str(iso)] = float(r[rate_col]) / 100.0
    return out


def adapt_simple_countries(raw: pd.DataFrame, pop_base_year: int = 2023,
                           fert_multiplier: float = 1.0,
                           growth: Optional[Dict[str, float]] = None,
                           growth_decay: float = 0.015,
                           rate_meaning: str = "auto", **growth_kw) -> pd.DataFrame:
    """Map the simple verified file onto the model schema.

    rate_meaning: 'auto' (classify the column), 'growth_pct', or 'tfr'.
    """
    df = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")],
                  errors="ignore").copy()
    unmapped = sorted(set(df["Country"]) - set(ISO3))
    if unmapped:
        raise ValueError(f"No ISO3 code defined for: {unmapped}")

    rate_col = next((c for c in ("pop_growth_pct", "growth_rate_pct", "fertility_rate")
                     if c in df.columns), None)
    verdict, reason = ("none", "no rate column present")
    if rate_col is not None:
        verdict, reason = classify_rate_column(df[rate_col])
        if rate_meaning != "auto":
            verdict, reason = rate_meaning, f"forced by caller (auto-detect said: {reason})"
        elif rate_col in ("pop_growth_pct", "growth_rate_pct"):
            # An explicitly named growth column is taken at its word; the
            # distribution heuristic is reported only as a cross-check.
            verdict, reason = "growth_pct", (f"column '{rate_col}' is named as annual "
                                             f"growth percent (distribution check: {reason})")

    out = pd.DataFrame({
        "country": df["Country"], "iso3": df["Country"].map(ISO3),
        "grads_adv": df["Advanced"].astype(float),
        "grads_int": df["Intermediate"].astype(float),
        "grads_front": df["Frontline"].astype(float),
        "pop_base": df["Population"].astype(float),
        "established": df.get("Established"),
    })
    # Optional per-tier programme start years (e.g. EstablishedFrontline) and an
    # optional share of advanced graduates who are host-country nationals.
    for t in TIERS:
        col = f"Established{TIER_LABEL[t]}"
        out[f"established_{TIER_SHORT[t]}"] = (pd.to_numeric(df[col], errors="coerce")
                                               if col in df.columns else np.nan)
    out["adv_host_share"] = (pd.to_numeric(df["AdvancedHostShare"], errors="coerce")
                             if "AdvancedHostShare" in df.columns else np.nan)
    out["rate_value"] = pd.to_numeric(df[rate_col], errors="coerce") if rate_col else np.nan
    out["rate_meaning"] = verdict
    out["rate_detect_reason"] = reason
    out["region"] = out["iso3"].map(REGION).fillna("Other")
    out["hosts_fetp"] = 1
    out["admin2_count"] = np.nan
    out["admin2_unit"] = ""
    for t, col in (("advanced", "grads_adv"), ("intermediate", "grads_int"),
                   ("frontline", "grads_front")):
        out[f"tier_{t}"] = (out[col] > 0).astype(int)
    # The simple country table holds actual graduate counts by country, so a
    # zero is an observed zero (the tier produced no graduates). Tiers listed
    # in an optional `UnconfirmedZero` column (e.g. "Advanced;Intermediate")
    # are flagged instead, for tables where a zero may mean "not published".
    unconfirmed = {}
    if "UnconfirmedZero" in df.columns:
        for i, v in df["UnconfirmedZero"].items():
            unconfirmed[i] = {x.strip().lower() for x in str(v).replace(",", ";").split(";")
                              if x.strip() and str(v) != "nan"}
    for t, col, nm in (("advanced", "grads_adv", "advanced"),
                       ("intermediate", "grads_int", "intermediate"),
                       ("frontline", "grads_front", "frontline")):
        st_vals = []
        for i in out.index:
            v = out.loc[i, col]
            if pd.isna(v):
                st_vals.append("missing")
            elif v > 0:
                st_vals.append("reported")
            elif nm in unconfirmed.get(i, set()):
                st_vals.append("zero_unconfirmed")
            else:
                st_vals.append("reported_zero")
        out[f"status_{t}"] = st_vals
    out["source"] = df["Source"] if "Source" in df.columns else np.nan
    out["source_date"] = df["SourceDate"] if "SourceDate" in df.columns else np.nan
    out["counts_by_country"] = True
    out["grads_basis"] = ("actual cumulative graduate counts by country, as supplied; "
                          "no source dates in file")
    out["notes"] = (f"Population anchor {pop_base_year}; 2026-2070 projected. "
                    "No admin-2, retention or regional-hub data in this file.")

    ys = np.array(WPP_ANCHORS)
    growth = growth or {}
    out["pop_source"] = ""
    for i, r in out.iterrows():
        if r.iso3 in growth:
            pops = project_population_growth(r.pop_base, pop_base_year, growth[r.iso3],
                                             ys, growth_decay)
            out.loc[i, "pop_source"] = (f"uploaded growth rate {growth[r.iso3]*100:.2f}%/yr, "
                                        f"declining {growth_decay*100:g}%/yr")
        elif verdict == "growth_pct" and pd.notna(r.rate_value):
            pops = project_population_growth(r.pop_base, pop_base_year,
                                             float(r.rate_value) / 100.0, ys, growth_decay)
            out.loc[i, "pop_source"] = (f"INTERIM: column read as growth rate "
                                        f"{float(r.rate_value):.2f}%/yr, declining "
                                        f"{growth_decay*100:g}%/yr. Load UN WPP for publication.")
        elif verdict == "tfr" and pd.notna(r.rate_value):
            pops = project_population_tfr(r.pop_base, pop_base_year,
                                          float(r.rate_value) * fert_multiplier, ys, **growth_kw)
            out.loc[i, "pop_source"] = "LAST RESORT: fertility-rate approximation"
        else:
            pops = np.full(len(ys), r.pop_base, dtype=float)
            out.loc[i, "pop_source"] = "NO GROWTH DATA: population held constant"
        for y, v in zip(ys, pops):
            out.loc[i, f"pop_{y}"] = v
    return out.sort_values(["region", "country"]).reset_index(drop=True)


def parse_wpp(wpp: pd.DataFrame) -> pd.DataFrame:
    """Reduce a UN WPP 2024 download to one row per ISO3 with pop_<year> columns."""
    cols = set(wpp.columns)
    iso_col = next((c for c in ("ISO3_code", "ISO3 Alpha-code", "iso3") if c in cols), None)
    pop_col = next((c for c in ("TPopulation1July", "PopTotal", "TPopulation1Jan") if c in cols), None)
    if iso_col is None or "Time" not in cols or pop_col is None:
        raise ValueError("Unrecognised WPP file: need ISO3_code, Time and "
                         "TPopulation1July (or PopTotal) columns.")
    w = wpp
    if "Variant" in cols:
        w = w[w["Variant"].astype(str).str.lower() == "medium"]
    w = w[w["Time"].isin(WPP_ANCHORS)]
    piv = w.pivot_table(index=iso_col, columns="Time", values=pop_col, aggfunc="first") * 1000.0
    piv.columns = [f"pop_{int(c)}" for c in piv.columns]
    piv.index.name = "iso3"
    return piv.reset_index()


def apply_wpp(countries: pd.DataFrame, wpp: pd.DataFrame) -> pd.DataFrame:
    tbl = parse_wpp(wpp).set_index("iso3")
    out = countries.copy()
    used = []
    for i, r in out.iterrows():
        iso = r["iso3"]
        if iso in tbl.index and pd.notna(tbl.loc[iso].get("pop_2026")):
            for c in tbl.columns:
                out.loc[i, c] = tbl.loc[iso, c]
            out.loc[i, "pop_source"] = "UN WPP 2024 medium variant"
            used.append(iso)
    out["notes"] = (f"Population: UN WPP 2024 medium variant for {len(used)} of "
                    f"{len(out)} countries. ") + out["notes"].astype(str)
    return out


def prepare_countries(raw: pd.DataFrame, source: str = "countries table",
                      pop_base_year: int = 2023, fert_multiplier: float = 1.0,
                      wpp: Optional[pd.DataFrame] = None,
                      growth: Optional[Dict[str, float]] = None,
                      growth_decay: float = 0.015,
                      rate_meaning: str = "auto", **growth_kw) -> pd.DataFrame:
    if REQUIRED_COLS.issubset(raw.columns):
        out = validate_countries(raw, source)
    elif SIMPLE_COLS.issubset(raw.columns):
        out = validate_countries(
            adapt_simple_countries(raw, pop_base_year, fert_multiplier, growth,
                                   growth_decay, rate_meaning, **growth_kw), source)
    else:
        raise ValueError(f"{source}: unrecognised columns {list(raw.columns)}")
    if wpp is not None:
        out = apply_wpp(out, wpp)
    return out


def load_countries(path: str | Path = DATA_FILE, **kw) -> pd.DataFrame:
    return prepare_countries(pd.read_csv(path), str(path), **kw)


def data_completeness(row: pd.Series) -> Dict[str, bool]:
    return {
        "population": pd.notna(row.get("pop_2026")),
        "advanced_supply": pd.notna(row.get("grads_adv")),
        "intermediate_supply": pd.notna(row.get("grads_int")),
        "frontline_supply": pd.notna(row.get("grads_front")),
        "admin2": pd.notna(row.get("admin2_count")),
        "cost": row.get("iso3") in COST_PER_FTE_USD,
        "retention": pd.notna(row.get("retention")),
        "provenance": pd.notna(row.get("source")) and pd.notna(row.get("source_date")),
    }


_TIER_COLS = {"advanced": ("tier_advanced", "grads_adv"),
              "intermediate": ("tier_intermediate", "grads_int"),
              "frontline": ("tier_frontline", "grads_front")}


def tier_status(row: pd.Series, tier: str) -> str:
    tier_col, grad_col = _TIER_COLS[tier]
    has_tier = bool(row.get(tier_col, 0))
    has_count = pd.notna(row.get(grad_col))
    flag = row.get(f"status_{tier}")
    if has_count and flag == "zero_unconfirmed":
        return "zero reported, unconfirmed (observed zero or unpublished?)"
    if has_count and flag == "reported_zero":
        return "reported zero (confirmed)"
    if has_count:
        return "reported"
    if row.get("adv_source") in ("abroad", "regional hub") and tier == "advanced":
        return "unknown (trained via regional hub, split unpublished)"
    if has_tier:
        return "unknown (tier operates, count not published)"
    return "no tier"


def supply_data_quality(row: pd.Series, count_frontline: bool = False) -> Dict[str, object]:
    tiers = ["advanced", "intermediate"] + (["frontline"] if count_frontline else [])
    st = {t: tier_status(row, t) for t in tiers}
    unknown = [t for t, s in st.items() if s.startswith("unknown")]
    zero_unconf = [t for t, s in st.items() if s.startswith("zero reported, unconfirmed")]
    return {"status": st, "unknown_tiers": unknown,
            "zero_unconfirmed_tiers": zero_unconf,
            "interpretable": len(unknown) == 0,
            "strictly_interpretable": len(unknown) == 0 and len(zero_unconf) == 0,
            "basis": row.get("grads_basis", "")}


def population_series(row: pd.Series, years: np.ndarray) -> Optional[np.ndarray]:
    """Log-linear interpolation through population anchor years."""
    anchors = [(y, row.get(f"pop_{y}")) for y in WPP_ANCHORS]
    pts = [(y, float(v)) for y, v in anchors if pd.notna(v)]
    if not pts:
        return None
    if len(pts) == 1:
        return np.full_like(np.asarray(years, dtype=float), pts[0][1], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.log(np.array([p[1] for p in pts], dtype=float))
    out = np.interp(years, xs, ys)
    if np.max(years) > xs.max():
        tail = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        m = years > xs.max()
        out[m] = ys[-1] + tail * (years[m] - xs[-1])
    return np.exp(out)


# ===========================================================================
# Parameters
# ===========================================================================

@dataclass
class Params:
    # --- initial stock ----------------------------------------------------
    survival_to_2026: Dict[str, float] = field(default_factory=lambda: {
        "advanced": 0.70, "intermediate": 0.75, "frontline": 0.70})
    public_sector_share: Dict[str, float] = field(default_factory=lambda: {
        "advanced": 0.80, "intermediate": 0.80, "frontline": 0.85})
    mean_years_since_graduation: Dict[str, float] = field(default_factory=lambda: {
        "advanced": 8.0, "intermediate": 5.0, "frontline": 5.0})
    # Share of FUTURE completers who seek domestic public-sector posts. The
    # starting stock is cumulative x survival x public_sector_share; without
    # this, projected inflows bypassed the public-sector adjustment applied to
    # the starting stock and later estimates were biased upward.
    public_entry_share: Dict[str, float] = field(default_factory=lambda: {
        "advanced": 0.80, "intermediate": 0.80, "frontline": 0.85})

    # --- tier overlap in the CUMULATIVE counts (new in v2) ----------------
    # Share of this tier's graduates who had already completed the tier below
    # and are therefore double-counted when the published totals are summed.
    overlap_adv_from_int: float = 0.30
    overlap_int_from_front: float = 0.40
    overlap_adv_from_front: float = 0.15
    correct_initial_overlap: bool = True

    # --- pipeline ---------------------------------------------------------
    annual_intake: Dict[str, float] = field(default_factory=lambda: {
        "advanced": 0.0, "intermediate": 0.0, "frontline": 0.0})
    completion: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COMPLETION))
    tier_months: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIER_MONTHS))
    progression_from_below: Dict[str, float] = field(default_factory=lambda: {
        "advanced": 0.30, "intermediate": 0.40})

    # --- outflows ---------------------------------------------------------
    attrition: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ATTRITION))
    emigration: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EMIGRATION))
    retirement_age: float = 60.0

    # --- capability conversion -------------------------------------------
    active_share: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTIVE_SHARE))
    protected_time: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PROTECTED_TIME))
    skill_weight: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SKILL_WEIGHT))

    # --- establishment ----------------------------------------------------
    model_posts: bool = True
    initial_posts_ratio: float = 1.0
    post_creation_per_year: float = 0.0

    # --- demand -----------------------------------------------------------
    people_per_fe: float = 200_000.0
    frontline_counts_toward_target: bool = False
    fe_per_admin2: float = 1.0
    frontline_per_admin2: float = 2.0
    productive_hours_per_fte: float = 1500.0
    hours_per_reporting_unit_week: float = 0.20
    reporting_units_per_admin2: float = 40.0
    annual_outbreaks_per_million: float = 0.5
    response_team_epi: float = 4.0
    deployment_days: float = 10.0
    field_hours_per_day: float = 8.0
    national_readiness_fte: float = 10.0
    analytics_fte_per_admin2: float = 0.10

    attribute_pooled_to_host: bool = False

    # =====================================================================
    # FEEDBACK STRUCTURE (new in v2). Every sensitivity at 0 and every
    # switch False reproduces the v1 open-loop model exactly.
    # =====================================================================

    # R1 mentor capacity trap -------------------------------------------
    mentor_feedback: bool = False
    mentor_share: float = 0.35          # share of advanced stock available to mentor
    residents_per_mentor: float = 2.0   # advanced residents per mentor at a time
    residents_per_mentor_int: float = 6.0   # intermediate trainees per mentor
    mentor_tier_eligible: str = "advanced"  # who can mentor intermediate
    # Mentor-equivalents from outside the domestic serving advanced stock
    # (regional hubs, visiting/partner faculty, universities). None means
    # "calibrate": set at the base year to whatever is needed to sustain the
    # baseline intake, so R1 limits CHANGES from the observed baseline rather
    # than declaring it infeasible (e.g. intermediate programmes in countries
    # with no domestic advanced graduates). A number fixes it explicitly.
    external_mentors: Optional[float] = None

    # R2 burnout spiral ---------------------------------------------------
    burnout_feedback: bool = False
    burnout_sensitivity: float = 0.25   # d(attrition)/d(excess load ratio)
    burnout_max_multiplier: float = 3.0

    # R3 unabsorbed-graduate emigration ----------------------------------
    # Acts ONLY on the identifiable pool of trained graduates who sought a
    # public post and did not get one (the `pool` stock), not on post holders.
    emigration_feedback: bool = False
    emigration_sensitivity: float = 1.2  # d(emigration)/d(unabsorbed share)
    emigration_max_multiplier: float = 4.0

    # B1 establishment response ------------------------------------------
    post_feedback: bool = False
    political_ambition: float = 0.60     # share of the JEE target that is funded eventually
    post_adjust_years: float = 6.0       # time to close the posts gap
    post_perception_delay: float = 3.0   # years before the gap is perceived
    max_post_growth_per_year: float = 40.0
    max_post_loss_per_year: float = 20.0

    def feedback_on(self) -> bool:
        return bool(self.mentor_feedback or self.burnout_feedback
                    or self.emigration_feedback or self.post_feedback)


def validate_params(p: Params) -> None:
    """Reject values outside their meaningful domain before a run."""
    errs = []

    def finite(name, v):
        if v is None or not np.isfinite(float(v)):
            errs.append(f"{name} must be finite")
            return False
        return True

    for t in TIERS:
        for nm, d, lo, hi in (("annual_intake", p.annual_intake, 0.0, None),
                              ("attrition", p.attrition, 0.0, None),
                              ("emigration", p.emigration, 0.0, None),
                              ("completion", p.completion, 0.0, 1.0),
                              ("survival_to_2026", p.survival_to_2026, 0.0, 1.0),
                              ("public_sector_share", p.public_sector_share, 0.0, 1.0),
                              ("public_entry_share", p.public_entry_share, 0.0, 1.0),
                              ("active_share", p.active_share, 0.0, 1.0),
                              ("protected_time", p.protected_time, 0.0, 1.0),
                              ("skill_weight", p.skill_weight, 0.0, None),
                              ("mean_years_since_graduation", p.mean_years_since_graduation, 0.0, None)):
            if t not in d:
                errs.append(f"{nm} has no value for tier {t}")
                continue
            v = d[t]
            if finite(f"{nm}[{t}]", v) and (v < lo or (hi is not None and v > hi)):
                errs.append(f"{nm}[{t}]={v} outside [{lo}, {hi if hi is not None else 'inf'}]")
        if finite(f"tier_months[{t}]", p.tier_months.get(t)) and p.tier_months[t] <= 0:
            errs.append(f"tier_months[{t}] must be positive")
    for nm in ("overlap_adv_from_int", "overlap_int_from_front", "overlap_adv_from_front",
               "mentor_share"):
        v = getattr(p, nm)
        if finite(nm, v) and not 0.0 <= v <= 1.0:
            errs.append(f"{nm}={v} outside [0, 1]")
    for tier in ("advanced", "intermediate"):
        v = p.progression_from_below.get(tier, 0.0)
        if finite(f"progression_from_below[{tier}]", v) and not 0.0 <= v <= 1.0:
            errs.append(f"progression_from_below[{tier}]={v} outside [0, 1]")
    for nm in ("people_per_fe", "productive_hours_per_fte", "residents_per_mentor",
               "residents_per_mentor_int", "post_adjust_years", "post_perception_delay"):
        v = getattr(p, nm)
        if finite(nm, v) and v <= 0:
            errs.append(f"{nm} must be positive")
    for nm in ("initial_posts_ratio", "burnout_sensitivity", "emigration_sensitivity",
               "political_ambition", "max_post_growth_per_year", "max_post_loss_per_year",
               "annual_outbreaks_per_million", "national_readiness_fte"):
        v = getattr(p, nm)
        if finite(nm, v) and v < 0:
            errs.append(f"{nm} must be non-negative")
    for nm in ("burnout_max_multiplier", "emigration_max_multiplier"):
        v = getattr(p, nm)
        if finite(nm, v) and v < 1:
            errs.append(f"{nm} must be at least 1")
    if finite("retirement_age", p.retirement_age) and p.retirement_age <= AGE_MIN:
        errs.append(f"retirement_age must exceed {AGE_MIN}")
    if p.external_mentors is not None and finite("external_mentors", p.external_mentors) \
            and p.external_mentors < 0:
        errs.append("external_mentors must be non-negative or None")
    if p.mentor_tier_eligible not in ("advanced", "advanced+intermediate"):
        errs.append("mentor_tier_eligible must be 'advanced' or 'advanced+intermediate'")
    if errs:
        raise ValueError("Invalid parameters: " + "; ".join(errs))


def validate_cohort_history(ch: pd.DataFrame) -> pd.DataFrame:
    """Check a cohort-history table (year, tier, intake) and return a clean copy."""
    missing = {"year", "tier", "intake"} - set(ch.columns)
    if missing:
        raise ValueError(f"cohort history is missing columns: {sorted(missing)}")
    out = ch.copy()
    out["tier"] = out["tier"].astype(str).str.strip().str.lower()
    bad_tier = sorted(set(out["tier"]) - set(TIERS))
    if bad_tier:
        raise ValueError(f"cohort history has unknown tiers: {bad_tier}")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["intake"] = pd.to_numeric(out["intake"], errors="coerce")
    if out[["year", "intake"]].isna().any().any():
        raise ValueError("cohort history has non-numeric year or intake values")
    if (out["intake"] < 0).any():
        raise ValueError("cohort history has negative intake")
    if (out["year"] != out["year"].round()).any():
        raise ValueError("cohort history years must be whole years")
    if (out["year"] > BASE_YEAR).any():
        raise ValueError(f"cohort history has years after the base year {BASE_YEAR}")
    return out


def tier_established(row: pd.Series, tier: str) -> Optional[float]:
    """Programme start year for a tier: the per-tier column if supplied,
    otherwise the programme-wide `established` year."""
    v = row.get(f"established_{TIER_SHORT[tier]}")
    if v is not None and pd.notna(v):
        return float(v)
    v = row.get("established")
    return None if v is None or pd.isna(v) else float(v)


def default_params_for(row: pd.Series) -> Params:
    p = Params()
    if pd.notna(row.get("retention")):
        r = float(row["retention"])
        for t in TIERS:
            p.public_sector_share[t] = r
            p.public_entry_share[t] = r
    for t in TIERS:
        est = tier_established(row, t)
        if est is not None:
            half_age = max(1.0, (BASE_YEAR - est) / 2.0)
            p.mean_years_since_graduation[t] = min(p.mean_years_since_graduation[t], half_age)
    return p


def implied_historical_intake(row: pd.Series, p: Optional[Params] = None) -> Dict[str, float]:
    """Average annual INTAKE implied by cumulative graduates and programme age.

    intake = cumulative_graduates / years_running / completion_rate

    This is only an approximate initialization from cumulative counts, not an
    observed annual intake series.
    """
    p = p or Params()
    # Use the SAME hub-attributed counts as the starting stock. Before this,
    # the starting stock honoured the pooled-graduate adjustment but projected
    # intake silently used the raw pooled total.
    cum = effective_cumulative_graduates(row, p)
    out = {}
    for t in TIERS:
        est = tier_established(row, t)   # per-tier start year when supplied
        if est is None or cum[t] is None:
            out[t] = 0.0
            continue
        yrs = max(1.0, BASE_YEAR - est)
        out[t] = float(cum[t]) / yrs / max(p.completion[t], EPS)
    return out


# ===========================================================================
# Supply: initial stock
# ===========================================================================

def _entry_age_weights(tier: str) -> np.ndarray:
    mean, sd = ENTRY_AGE_MEAN_SD[tier]
    w = np.exp(-0.5 * ((AGES - mean) / sd) ** 2)
    return w / w.sum()


ENTRY_WEIGHTS = {t: _entry_age_weights(t) for t in TIERS}


def effective_cumulative_graduates(row: pd.Series, p: Params) -> Dict[str, Optional[float]]:
    """Cumulative graduates after regional-hub attribution."""
    raw = {"advanced": row.get("grads_adv"), "intermediate": row.get("grads_int"),
           "frontline": row.get("grads_front")}
    out: Dict[str, Optional[float]] = {t: (None if pd.isna(raw[t]) else float(raw[t]))
                                       for t in TIERS}
    if out["advanced"] is not None and not p.attribute_pooled_to_host:
        applied, factor, _ = hub_correction(row)
        if applied:
            out["advanced"] = out["advanced"] * factor
    return out


def hub_correction(row: pd.Series) -> Tuple[bool, float, str]:
    """Return (applied, factor, explanation) for regional-hub attribution.

    A supplied `adv_host_share` is used directly. Otherwise the documented
    correction is applied only when the table reports the pooled total that
    the correction was derived from; a different value may already be a
    host-country figure, and dividing it again would double-correct.
    """
    v = row.get("grads_adv")
    share = row.get("adv_host_share")
    if share is not None and pd.notna(share):
        return True, float(share), f"Host-country share {float(share):.2f} from the country table."
    iso = row.get("iso3")
    if iso not in POOLED_ADVANCED or v is None or pd.isna(v):
        return False, 1.0, ""
    cbc = row.get("counts_by_country")
    if cbc is not None and not (isinstance(cbc, float) and np.isnan(cbc)) and bool(cbc):
        return False, 1.0, ("The country table reports graduates by country, so no "
                            "regional-hub correction is applied.")
    _, n_pooled, total, host = POOLED_ADVANCED[iso]
    if total is None:
        return False, 1.0, (f"No documented pooled total for this hub, so the table value "
                            f"({float(v):.0f}) is used as supplied. Add an AdvancedHostShare "
                            "column to apportion it.")
    if abs(float(v) - total) > 0.5:
        return False, 1.0, (f"The documented correction refers to a pooled total of {total:.0f}; "
                            f"the table reports {float(v):.0f}, so it is used as supplied. Add an "
                            "AdvancedHostShare column if this figure includes other nationals.")
    factor = (host / total) if host is not None else 1.0 / n_pooled
    return True, factor, (f"Table value matches the documented pooled total ({total:.0f}); "
                          f"host-country attribution factor {factor:.3f} applied.")


def disjoint_graduates(cum: Dict[str, Optional[float]], p: Params) -> Dict[str, Optional[float]]:
    """Apply the analyst-selected tier-overlap adjustment to cumulative counts.

    Without person-level progression records, these adjustments cannot
    establish the number of distinct people. Setting ``correct_initial_overlap``
    to false preserves the supplied tier totals without adjustment.
    """
    if not p.correct_initial_overlap:
        return dict(cum)
    A = cum["advanced"] or 0.0
    I = cum["intermediate"] or 0.0
    F = cum["frontline"] or 0.0
    int_only = max(0.0, I - p.overlap_adv_from_int * A)
    front_only = max(0.0, F - p.overlap_int_from_front * I - p.overlap_adv_from_front * A)
    return {
        "advanced": None if cum["advanced"] is None else A,
        "intermediate": None if cum["intermediate"] is None else int_only,
        "frontline": None if cum["frontline"] is None else front_only,
    }


def overlap_report(cum: Dict[str, Optional[float]], p: Params) -> List[Dict[str, object]]:
    """Compare the overlap assumption with the recorded counts.

    For each lower tier, the assumed number of people who also completed a
    higher tier cannot exceed the number recorded in the lower tier. When it
    does, the adjustment is capped (all lower-tier graduates are treated as
    having progressed) and the row is flagged: the assumption is incompatible
    with the data, not merely uncertain.
    """
    if not p.correct_initial_overlap:
        return []
    A = cum["advanced"] or 0.0
    I = cum["intermediate"] or 0.0
    F = cum["frontline"] or 0.0
    out = []
    for tier, recorded, assumed, basis in (
            ("intermediate", I, p.overlap_adv_from_int * A,
             f"{p.overlap_adv_from_int:.0%} of {A:.0f} advanced"),
            ("frontline", F, p.overlap_int_from_front * I + p.overlap_adv_from_front * A,
             f"{p.overlap_int_from_front:.0%} of {I:.0f} intermediate + "
             f"{p.overlap_adv_from_front:.0%} of {A:.0f} advanced")):
        if cum[tier] is None:
            continue
        capped = assumed > recorded + 1e-9
        out.append({
            "tier": tier, "recorded": recorded, "assumed_overlap": assumed,
            "applied_overlap": min(assumed, recorded), "basis": basis,
            "status": ("incompatible: assumption exceeds recorded count; capped"
                       if capped else "consistent"),
            "max_consistent_share": (recorded / max(A, EPS) if tier == "intermediate" else np.nan),
        })
    return out


def initialize_stock(row: pd.Series, p: Params,
                     cohort_history: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
    stock = {t: np.zeros(len(AGES)) for t in TIERS}

    if cohort_history is not None and len(cohort_history):
        cohort_history = validate_cohort_history(cohort_history)
        completers = {t: 0.0 for t in TIERS}
        for t in TIERS:
            sub = cohort_history[cohort_history["tier"].str.lower() == t]
            for _, r in sub.iterrows():
                yr, n = int(r["year"]), float(r["intake"])
                if n <= 0:
                    continue
                grads = n * p.completion[t] * p.public_sector_share[t]
                elapsed = BASE_YEAR - yr - p.tier_months[t] / 12.0
                if elapsed < 0:
                    continue
                completers[t] += n * p.completion[t]
                loss = p.attrition[t] + p.emigration[t]
                survivors = grads * np.exp(-loss * elapsed)
                aged = np.zeros(len(AGES))
                shift = int(round(elapsed))
                w = ENTRY_WEIGHTS[t]          # sums to 1
                if shift < len(AGES):
                    aged[shift:] = w[: len(AGES) - shift]
                    aged[-1] += w[len(AGES) - shift:].sum()   # oldest bin absorbing
                else:
                    aged[-1] = 1.0
                # Mortality is accumulated along each member's age path, from
                # entry age to current age, and is NOT renormalized away: the
                # headcount falls with deaths and with retirement. (Previously
                # the profile was rescaled back to `survivors`, so mortality
                # and partial retirement changed only the age shape.)
                cum_m = np.concatenate([[0.0], np.cumsum(AGE_MORTALITY)])
                idx = np.arange(len(AGES))
                path_m = cum_m[idx] - cum_m[np.clip(idx - shift, 0, None)]
                if shift > 0:
                    path_m = path_m * (elapsed / shift)
                alive = aged * np.exp(-path_m)
                alive[AGES >= p.retirement_age] = 0.0
                stock[t] += survivors * alive
        # Reconcile people who completed more than one tier. Cohort records
        # count a person once per tier, so the same overlap assumption as the
        # default path is applied to the cohort completer totals, and each
        # lower tier is scaled to its distinct-person share.
        if p.correct_initial_overlap:
            distinct = disjoint_graduates(completers, p)
            for t in TIERS:
                if completers[t] > EPS:
                    stock[t] *= float(distinct[t]) / completers[t]
        return stock

    # Default path. `survival_to_2026` is defined as the share of cumulative
    # graduates still active in the base year, so it already includes those
    # who retired or died. The age profile is therefore rescaled to that
    # active count rather than reduced a second time for retirement.

    cum = disjoint_graduates(effective_cumulative_graduates(row, p), p)
    for t in TIERS:
        if cum[t] is None:
            continue
        active = cum[t] * p.survival_to_2026[t] * p.public_sector_share[t]
        shift = int(round(p.mean_years_since_graduation[t]))
        w = ENTRY_WEIGHTS[t]
        aged = np.zeros(len(AGES))
        if shift < len(AGES):
            aged[shift:] = w[: len(AGES) - shift]
        aged[AGES >= p.retirement_age] = 0.0
        if aged.sum() > 0:
            stock[t] = active * aged / aged.sum()
    return stock


# ===========================================================================
# Supply: stepping, with exits tracked separately
# ===========================================================================

def step_stock(stock: Dict[str, np.ndarray], p: Params, dt: float,
               attrition: Optional[Dict[str, float]] = None,
               emigration: Optional[Dict[str, float]] = None,
               age_step: bool = True
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Advance one step. Returns (new_stock, exits) where exits is
    {tier: {'attrition':..,'emigration':..,'mortality':..,'retirement':..}}.

    Exit channels are reported separately so scenarios can vary assumptions
    without conflating attrition, emigration, mortality, and retirement.
    """
    attrition = attrition or p.attrition
    emigration = emigration or p.emigration
    new, exits = {}, {}
    for t in TIERS:
        s = stock[t]
        a, e, m = attrition[t], emigration[t], AGE_MORTALITY
        total_haz = a + e + m
        # Exact survival for a hazard that is constant within the step. The
        # earlier first-order form (1 - h dt) carried an O(dt) error.
        keep = np.exp(-total_haz * dt)
        lost = s * (1.0 - keep)
        denom = np.maximum(total_haz, EPS)
        ex_att = float((lost * (a / denom)).sum())
        ex_emi = float((lost * (e / denom)).sum())
        ex_mor = float((lost * (m / denom)).sum())
        survivors = s * keep

        # Ageing happens once a year, on the anniversary step (`age_step`).
        # The earlier scheme moved a fraction dt of each bin up every step;
        # that is an upwind advection with numerical diffusion of ~0.5 yr^2/yr
        # independent of dt (age spread of about +-5 years by 2051), which a
        # dt-halving test cannot reveal. Annual shifting is exact.
        if age_step:
            aged = np.zeros_like(survivors)
            aged[1:] = survivors[:-1]
            # The oldest bin is absorbing: without this, people ageing out of
            # AGE_MAX are silently destroyed (found by the extreme-condition
            # test, 33% of stock over 25 years).
            aged[-1] += survivors[-1]
        else:
            aged = survivors.copy()

        retired_mask = AGES >= p.retirement_age
        ex_ret = float(aged[retired_mask].sum())
        aged[retired_mask] = 0.0

        new[t] = aged
        exits[t] = {"attrition": ex_att, "emigration": ex_emi,
                    "mortality": ex_mor, "retirement": ex_ret}
    return new, exits


def add_graduates(stock: Dict[str, np.ndarray], grads: Dict[str, float],
                  p: Params) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Add new external entrants to the serving stock.

    Internal progressions are handled in ``enter_graduates`` at the instant
    of graduation, where both the serving and unabsorbed stocks are available.
    Keeping this helper limited to external entrants prevents a graduate who
    is waiting for a post from being counted in both the pool and the tier
    below.
    """
    out = {t: stock[t].copy() for t in TIERS}
    for t in TIERS:
        g = grads.get(t, 0.0)
        if np.ndim(g) == 0:
            # a headcount: place at the tier's entry-age profile
            if g <= 0:
                continue
            out[t] = out[t] + g * ENTRY_WEIGHTS[t]
        else:
            # an age profile (e.g. hired from the age-structured pool): keep ages
            out[t] = out[t] + np.asarray(g, dtype=float)
    return out, {t: 0.0 for t in TIERS}


def compliance_headcount(stock: Dict[str, np.ndarray], p: Params) -> float:
    total = stock["advanced"].sum() + stock["intermediate"].sum()
    if p.frontline_counts_toward_target:
        total += stock["frontline"].sum()
    return float(total)


def effective_fte(stock: Dict[str, np.ndarray], p: Params) -> Tuple[float, Dict[str, float]]:
    by = {t: float(stock[t].sum() * p.active_share[t] *
                   min(1.0, p.protected_time[t]) * p.skill_weight[t]) for t in TIERS}
    return sum(by.values()), by


# ===========================================================================
# Demand
# ===========================================================================

def demand_measures(population: float, admin2: Optional[float], p: Params) -> Dict[str, float]:
    jee_target_heads = population / p.people_per_fe
    if admin2 is not None and np.isfinite(admin2):
        coverage_floor_heads = admin2 * p.fe_per_admin2
        reporting_units = admin2 * p.reporting_units_per_admin2
        analytics = admin2 * p.analytics_fte_per_admin2
    else:
        coverage_floor_heads = np.nan
        reporting_units = population / 5000.0
        analytics = 0.0
    routine_h = reporting_units * 52.0 * p.hours_per_reporting_unit_week
    outbreaks = population / 1e6 * p.annual_outbreaks_per_million
    outbreak_h = outbreaks * p.response_team_epi * p.deployment_days * p.field_hours_per_day
    workload_fte = ((routine_h + outbreak_h) / p.productive_hours_per_fte
                    + p.national_readiness_fte + analytics)
    return {
        "jee_target_heads": jee_target_heads,
        "coverage_floor_heads": coverage_floor_heads,
        "workload_fte": workload_fte,
        "routine_fte": routine_h / p.productive_hours_per_fte,
        "outbreak_fte": outbreak_h / p.productive_hours_per_fte,
        "readiness_fte": p.national_readiness_fte,
        "analytics_fte": analytics,
        "annual_outbreaks": outbreaks,
    }


# ===========================================================================
# Feedback functions - each isolated and unit-testable
# ===========================================================================

def f_burnout(base: Dict[str, float], load_ratio: float, p: Params) -> Dict[str, float]:
    """R2. Attrition rises when effective capacity falls short of workload."""
    if not p.burnout_feedback:
        return dict(base)
    excess = max(0.0, load_ratio - 1.0)
    mult = min(p.burnout_max_multiplier, 1.0 + p.burnout_sensitivity * excess)
    return {t: base[t] * mult for t in TIERS}


def f_emigration(base: Dict[str, float], unabsorbed_share: float, p: Params) -> Dict[str, float]:
    """R3. Emigration hazard of the UNABSORBED POOL rises with the share of
    trained graduates who hold no funded post. Post holders keep the base
    hazard. The elasticity is an analyst-selected scenario assumption."""
    if not p.emigration_feedback:
        return dict(base)
    mult = min(p.emigration_max_multiplier,
               1.0 + p.emigration_sensitivity * max(0.0, unabsorbed_share))
    return {t: base[t] * mult for t in TIERS}


def mentor_supply(stock: Dict[str, np.ndarray], p: Params,
                  external_mentors: float = 0.0) -> Tuple[float, float]:
    """(shared mentors, intermediate-only mentors).

    Shared mentors are domestic serving advanced staff x `mentor_share` plus
    `external_mentors` (regional, visiting, university or partner mentors,
    in mentor-equivalents). They can supervise either programme. Serving
    intermediate staff supervise intermediate trainees only, and only when
    `mentor_tier_eligible` is not "advanced".
    """
    shared = float(stock["advanced"].sum()) * p.mentor_share + max(0.0, external_mentors)
    int_only = (0.0 if p.mentor_tier_eligible == "advanced"
                else float(stock["intermediate"].sum()) * p.mentor_share)
    return shared, int_only


def f_training_capacity(stock: Dict[str, np.ndarray], p: Params,
                        desired: Optional[Dict[str, float]] = None,
                        continuing: Optional[Dict[str, float]] = None,
                        dt: Optional[float] = None,
                        external_mentors: float = 0.0) -> Dict[str, float]:
    """R1. Maximum annual INTAKE (admission rate) per tier given mentors.

    Supervision is a concurrent load: one mentor supervises
    `residents_per_mentor` advanced residents or `residents_per_mentor_int`
    intermediate trainees at a time. Shared mentors serve both programmes.

    With `continuing` (people already enrolled who remain in training after
    this step) and `dt`, enrolled trainees keep their supervision and only the
    remaining capacity is offered to new admissions in this step. Without
    them, a steady-state approximation is used (admissions x duration), which
    gives the same answer once intake has been constant for one programme
    duration. When admissions need more mentors than remain, both programmes
    are rationed in proportion to their need. Trainees already enrolled are
    never removed.
    """
    if not p.mentor_feedback:
        return {t: np.inf for t in TIERS}
    desired = desired if desired is not None else p.annual_intake
    shared, int_only = mentor_supply(stock, p, external_mentors)
    rpm_a = max(p.residents_per_mentor, EPS)
    rpm_i = max(p.residents_per_mentor_int, EPS)
    if continuing is None:
        per_a = max(p.tier_months["advanced"] / 12.0, EPS) / rpm_a
        per_i = max(p.tier_months["intermediate"] / 12.0, EPS) / rpm_i
        load_a = load_i = 0.0
    else:
        if dt is None:
            raise ValueError("dt is required when continuing enrolment is given")
        per_a, per_i = dt / rpm_a, dt / rpm_i
        load_a = max(0.0, continuing.get("advanced", 0.0)) / rpm_a
        load_i = max(0.0, continuing.get("intermediate", 0.0)) / rpm_i
    # Intermediate-only mentors cover intermediate load first.
    int_only_free = max(0.0, int_only - load_i)
    load_i_shared = max(0.0, load_i - int_only)
    free = max(0.0, shared - load_a - load_i_shared)
    new_a = max(0.0, float(desired.get("advanced", 0.0))) * per_a
    new_i = max(0.0, float(desired.get("intermediate", 0.0))) * per_i
    new_i_shared = max(0.0, new_i - int_only_free)
    if new_a + new_i_shared <= free:
        cap_a = (free - new_i_shared) / per_a
        cap_i = (int_only_free + free - new_a) / per_i
    else:
        scale = free / (new_a + new_i_shared)
        cap_a = new_a * scale / per_a
        cap_i = (min(new_i, int_only_free) + new_i_shared * scale) / per_i
    return {"advanced": max(0.0, cap_a), "intermediate": max(0.0, cap_i),
            "frontline": np.inf}


def calibrate_external_mentors(stock: Dict[str, np.ndarray], p: Params,
                               baseline_intake: Dict[str, float]) -> float:
    """Mentor-equivalents from outside the domestic serving advanced stock
    needed to sustain the baseline (historical or active) intake at the base
    year. Used when `external_mentors` is None, so R1 constrains changes from
    the baseline rather than declaring the observed baseline infeasible."""
    if not p.mentor_feedback:
        return 0.0
    shared, int_only = mentor_supply(stock, p, 0.0)
    need_a = max(0.0, baseline_intake.get("advanced", 0.0)) * (
        p.tier_months["advanced"] / 12.0) / max(p.residents_per_mentor, EPS)
    need_i = max(0.0, baseline_intake.get("intermediate", 0.0)) * (
        p.tier_months["intermediate"] / 12.0) / max(p.residents_per_mentor_int, EPS)
    return max(0.0, need_a + max(0.0, need_i - int_only) - shared)


def post_demand_gap(jee_target: float, deployed: float, posts: float) -> float:
    """Staffing shortfall that vacant posts will NOT already fill.

    service gap = target - post holders. Funded-but-vacant posts (posts -
    deployed) are subtracted because they can be filled without new money.
    """
    service_gap = max(0.0, jee_target - deployed)
    vacancies = max(0.0, posts - deployed)
    return max(0.0, service_gap - vacancies)


def f_post_creation(posts: float, perceived_gap: float, jee_target: float,
                    p: Params, dt: float) -> float:
    """B1. Funded posts grow in response to the PERCEIVED staffing gap
    (target minus current post holders, net of vacancies), closing it over
    `post_adjust_years`, and never beyond a fiscal ceiling of
    `political_ambition` x target. Posts stop growing when the gap closes.
    Incumbents are never laid off, so posts falling below incumbents only
    blocks refilling of vacancies."""
    if not p.post_feedback:
        return p.post_creation_per_year
    ceiling = max(0.0, jee_target * p.political_ambition)
    rate = min(perceived_gap, ceiling - posts) / max(p.post_adjust_years, EPS)
    return float(np.clip(rate, -p.max_post_loss_per_year, p.max_post_growth_per_year))


# ===========================================================================
# Unabsorbed pool: trained graduates who sought a public post and have none.
# The pool is age-structured like the serving stock: its members age, die,
# retire, and keep their ages when they are hired or promoted.
# ===========================================================================

def counted_tiers(p: Params) -> Tuple[str, ...]:
    return ("advanced", "intermediate") + (("frontline",) if p.frontline_counts_toward_target else ())


def empty_pool() -> Dict[str, np.ndarray]:
    return {t: np.zeros(len(AGES)) for t in TIERS}


def pool_total(pool: Dict[str, np.ndarray], tiers=TIERS) -> float:
    return float(sum(float(np.sum(pool[t])) for t in tiers))


def split_initial_by_posts(stock: Dict[str, np.ndarray], p: Params, posts0: float
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split the initial trained stock into post holders (serving) and the
    unabsorbed pool, pro rata across counted tiers and ages."""
    pool = empty_pool()
    serving = {t: stock[t].copy() for t in TIERS}
    counted = counted_tiers(p)
    total = sum(float(stock[t].sum()) for t in counted)
    if total <= EPS or posts0 >= total:
        return serving, pool
    frac = max(0.0, posts0) / total
    for t in counted:
        serving[t] = stock[t] * frac
        pool[t] = stock[t] * (1.0 - frac)
    return serving, pool


def step_pool(pool: Dict[str, np.ndarray], p: Params, dt: float,
              emigration: Dict[str, float], age_step: bool = False
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Advance the pool one step with the same exact competing-hazard survival,
    annual ageing and retirement rule as the serving stock. The pool uses the
    base attrition hazard and the (possibly R3-modified) pool emigration."""
    new, exits = step_stock(pool, p, dt, attrition=p.attrition,
                            emigration=emigration, age_step=age_step)
    ex = {ch: float(sum(exits[t][ch] for t in TIERS))
          for ch in ("attrition", "emigration", "mortality", "retirement")}
    return new, ex


def enter_graduates(stock: Dict[str, np.ndarray], pool: Dict[str, np.ndarray],
                    grads: Dict[str, float], free_posts: float, p: Params
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Route completions without double-counting internal progressors.

    A completion drawn from the tier below transfers that person from the
    lower tier to the higher tier immediately, keeping their age and whether
    they were serving or waiting in the unabsorbed pool. A person promoted
    from a tier without post limits (by default frontline) into a post-limited
    tier joins the higher tier's pool and competes for posts. Remaining
    external completions are subject to ``public_entry_share``; public-sector
    seekers join the pool at the tier's entry-age profile, then available
    posts absorb pool members pro rata across ages. Tiers excluded from the
    post benchmark enter serving stock directly.

    If the requested progression share exceeds the available lower-tier
    stock, the shortfall is treated as external recruitment.

    ``tracked_inflow`` is the net inflow of new public-sector people (the
    quantity required for mass balance); ``promoted`` is a within-system
    transfer and is reported separately from gross completions.
    """
    counted = counted_tiers(p)
    out = {t: stock[t].copy() for t in TIERS}
    pool2 = {t: np.asarray(pool[t], dtype=float).copy() for t in TIERS}

    external = {t: max(0.0, float(grads.get(t, 0.0))) for t in TIERS}
    promoted_total = promoted_serving = promoted_pool = 0.0
    for tier, below in (("advanced", "intermediate"),
                        ("intermediate", "frontline")):
        requested = external[tier] * max(0.0, p.progression_from_below.get(tier, 0.0))
        available_serving = float(out[below].sum())
        available_pool = float(pool2[below].sum())
        available = available_serving + available_pool
        moved = min(requested, available)
        if moved <= EPS or available <= EPS:
            continue
        share = moved / available
        moved_serving = out[below] * share
        moved_pool = pool2[below] * share
        out[below] = out[below] - moved_serving
        pool2[below] = pool2[below] - moved_pool
        if p.model_posts and tier in counted and below not in counted:
            pool2[tier] = pool2[tier] + moved_serving
        else:
            out[tier] = out[tier] + moved_serving
        pool2[tier] = pool2[tier] + moved_pool
        external[tier] -= moved
        promoted_total += moved
        promoted_serving += float(moved_serving.sum())
        promoted_pool += float(moved_pool.sum())

    seekers = {t: external[t] * p.public_entry_share[t] for t in TIERS}
    nonpublic = float(sum(external[t] - seekers[t] for t in TIERS))
    entering = {t: np.zeros(len(AGES)) for t in TIERS}
    if p.model_posts:
        for t in TIERS:
            if t in counted:
                pool2[t] = pool2[t] + seekers[t] * ENTRY_WEIGHTS[t]
            else:
                entering[t] = seekers[t] * ENTRY_WEIGHTS[t]
        candidates = pool_total(pool2, counted)
        share = (1.0 if candidates <= EPS else
                 min(1.0, max(0.0, free_posts) / candidates))
        for t in counted:
            entering[t] = pool2[t] * share
            pool2[t] = pool2[t] - entering[t]
    else:
        entering = {t: seekers[t] * ENTRY_WEIGHTS[t] for t in TIERS}
    out, _ = add_graduates(out, entering, p)
    info = {
        "tracked_inflow": float(sum(seekers.values())),
        "nonpublic": nonpublic,
        "absorbed": float(sum(float(entering[t].sum()) for t in counted)),
        "total_completers": float(sum(max(0.0, float(grads.get(t, 0.0))) for t in TIERS)),
        "promoted": promoted_total,
        "promoted_serving": promoted_serving,
        "promoted_pool": promoted_pool,
    }
    return out, pool2, info


# ===========================================================================
# Simulation
# ===========================================================================

FLOW_COLUMNS = ("exit_attrition", "exit_emigration", "exit_mortality", "exit_retirement",
                "graduates_total", "graduate_completions_total", "grads_nonpublic",
                "absorbed_counted", "promoted_total", "promoted_serving", "promoted_pool")


def run(row: pd.Series, p: Params, horizon: int = HORIZON_DEFAULT,
        cohort_history: Optional[pd.DataFrame] = None,
        dt: float = DT_DEFAULT,
        initial_pipeline_intake: Optional[Dict[str, float]] = None,
        initial_stock: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Integrate the model.

    STOCKS (all in people, counted tiers per `counted_tiers`)
      serving   post holders, age-structured, by tier      -> `deployed_heads`
      pool      trained, sought a public post, none held,
                age-structured, by tier                    -> `unabsorbed_heads`
      trained   serving + pool                             -> `compliance_heads`
    Graduates who do not seek public posts leave the tracked system on
    completion (`grads_nonpublic`), consistent with the starting stock which
    already excludes them via `public_sector_share`.
    Capability (effective FTE) counts only the serving stock.

    Common initial conditions for scenario comparisons:
      `initial_stock` replaces the reconstructed starting workforce, so a
      scenario's prospective assumptions (e.g. lower attrition) cannot change
      the historical starting point; `initial_pipeline_intake` fixes the
      trainees already enrolled and the baseline used to calibrate external
      mentors. Without them, both default to the active parameters.

    Flow columns (exits, graduates, promotions) on row k record what happened
    during the interval ending at that row's time; row 0 holds zeros.
    """
    validate_params(p)
    if not np.isfinite(dt) or dt <= 0 or dt > 1:
        raise ValueError("dt must be finite, positive, and no greater than one year")
    n = int(round(horizon / dt)) + 1
    t_years = np.arange(n) * dt
    cal = BASE_YEAR + t_years
    pop = population_series(row, cal)
    if pop is None:
        raise ValueError(f"No population data for {row['country']}; cannot project demand.")
    admin2 = float(row["admin2_count"]) if pd.notna(row.get("admin2_count")) else None

    if initial_stock is not None:
        stock = {t: np.asarray(initial_stock[t], dtype=float).copy() for t in TIERS}
    else:
        stock = initialize_stock(row, p, cohort_history)
    qlen = {t: max(1, int(round(p.tier_months[t] / 12.0 / dt))) for t in TIERS}
    pipeline_start = dict(initial_pipeline_intake or p.annual_intake)
    queues = {t: [max(0.0, pipeline_start[t]) * dt] * qlen[t] for t in TIERS}

    posts = compliance_headcount(stock, p) * p.initial_posts_ratio
    if p.model_posts:
        stock, pool = split_initial_by_posts(stock, p, posts)
    else:
        pool = empty_pool()
    external_mentors = (p.external_mentors if p.external_mentors is not None
                        else calibrate_external_mentors(stock, p, pipeline_start))
    d0 = demand_measures(pop[0], admin2, p)
    perceived_gap = post_demand_gap(d0["jee_target_heads"], compliance_headcount(stock, p), posts)

    rows: List[dict] = []
    pending_flows = {c: 0.0 for c in FLOW_COLUMNS}
    for i in range(n):
        served = compliance_headcount(stock, p)
        unabs = pool_total(pool, counted_tiers(p))
        heads = served + unabs
        fte, by_tier = effective_fte(stock, p)
        d = demand_measures(pop[i], admin2, p)

        load_ratio = d["workload_fte"] / max(fte, EPS)
        deployed = served
        unabsorbed_share = unabs / max(heads, EPS)

        att = f_burnout(p.attrition, load_ratio, p)
        emi_pool = f_emigration(p.emigration, unabsorbed_share, p)
        # Trainees who remain enrolled after this step's completions.
        continuing = {t: float(sum(queues[t][1:])) for t in TIERS}
        cap = f_training_capacity(stock, p, p.annual_intake, continuing, dt, external_mentors)
        intake_eff = {t: float(min(p.annual_intake[t], cap[t])) for t in TIERS}
        intake_rationed = {t: float(max(0.0, p.annual_intake[t] - intake_eff[t])) for t in TIERS}

        if p.model_posts and unabs > 0.5:
            binding = "established posts"
        elif p.mentor_feedback and sum(intake_rationed.values()) > 0.01:
            binding = "mentor capacity"
        elif heads < d["jee_target_heads"]:
            binding = "trained supply"
        else:
            binding = "none (target met)"

        rec = {
            "time_years": t_years[i], "calendar_year": cal[i], "population": pop[i],
            "heads_advanced": stock["advanced"].sum() + pool["advanced"].sum(),
            "heads_intermediate": stock["intermediate"].sum() + pool["intermediate"].sum(),
            "heads_frontline": stock["frontline"].sum() + pool["frontline"].sum(),
            "serving_advanced": stock["advanced"].sum(),
            "serving_intermediate": stock["intermediate"].sum(),
            "serving_frontline": stock["frontline"].sum(),
            "pool_advanced": pool["advanced"].sum(),
            "pool_intermediate": pool["intermediate"].sum(),
            "pool_frontline": pool["frontline"].sum(),
            "compliance_heads": heads, "deployed_heads": deployed,
            "unabsorbed_heads": unabs, "unabsorbed_share": unabsorbed_share,
            "established_posts": posts, "effective_fte": fte,
            "fte_advanced": by_tier["advanced"], "fte_intermediate": by_tier["intermediate"],
            "fte_frontline": by_tier["frontline"],
            "jee_target_heads": d["jee_target_heads"],
            "coverage_floor_heads": d["coverage_floor_heads"],
            "workload_fte": d["workload_fte"], "routine_fte": d["routine_fte"],
            "outbreak_fte": d["outbreak_fte"], "readiness_fte": d["readiness_fte"],
            "analytics_fte": d["analytics_fte"],
            "jee_gap_heads": d["jee_target_heads"] - heads,
            "jee_ratio": heads / max(d["jee_target_heads"], EPS),
            "deployed_ratio": deployed / max(d["jee_target_heads"], EPS),
            "density_per_200k": heads / max(pop[i] / 200_000.0, EPS),
            "workload_gap_fte": d["workload_fte"] - fte,
            "load_ratio": load_ratio,
            "binding_constraint": binding,
            "attrition_rate_adv": att["advanced"],
            "emigration_rate_adv": emi_pool["advanced"],
            "emigration_rate_adv_base": p.emigration["advanced"],
            "train_cap_advanced": cap["advanced"],
            "train_cap_intermediate": cap["intermediate"],
            "intake_eff_advanced": intake_eff["advanced"],
            "intake_eff_intermediate": intake_eff["intermediate"],
            "intake_rationed_advanced": intake_rationed["advanced"],
            "intake_rationed_intermediate": intake_rationed["intermediate"],
            "enrolled_advanced": float(sum(queues["advanced"])),
            "enrolled_intermediate": float(sum(queues["intermediate"])),
            "external_mentors": external_mentors,
            "perceived_gap": perceived_gap,
        }
        # Flows for the interval ending at this row (row 0: none yet).
        rec.update(pending_flows)
        rows.append(rec)
        if i == n - 1:
            break

        grads = {}
        for t in TIERS:
            grads[t] = queues[t].pop(0) * p.completion[t]
            queues[t].append(intake_eff[t] * dt)

        age_now = (np.floor(t_years[i + 1] + EPS) >
                   np.floor(t_years[i] + EPS))
        stock, exits = step_stock(stock, p, dt, att, p.emigration, age_step=age_now)
        pool, pexits = step_pool(pool, p, dt, emi_pool, age_step=age_now)
        free = (max(0.0, posts - compliance_headcount(stock, p)) if p.model_posts else np.inf)
        stock, pool, info = enter_graduates(stock, pool, grads, free, p)

        flows = {f"exit_{ch}": sum(exits[t][ch] for t in TIERS) + pexits[ch]
                 for ch in ("attrition", "emigration", "mortality", "retirement")}
        flows.update({
            "graduates_total": info["tracked_inflow"],
            "graduate_completions_total": info["total_completers"],
            "grads_nonpublic": info["nonpublic"],
            "absorbed_counted": info["absorbed"],
            "promoted_total": info["promoted"],
            "promoted_serving": info["promoted_serving"],
            "promoted_pool": info["promoted_pool"],
        })
        pending_flows = flows

        # establishment (B1): perceived staffing gap is a first-order delay
        gap = post_demand_gap(d["jee_target_heads"], deployed, posts)
        perceived_gap += (gap - perceived_gap) / max(p.post_perception_delay, EPS) * dt
        posts = max(0.0, posts + f_post_creation(posts, perceived_gap, d["jee_target_heads"], p, dt) * dt)

    df = pd.DataFrame(rows)
    for ch in ("attrition", "emigration", "mortality", "retirement"):
        df[f"cum_exit_{ch}"] = df[f"exit_{ch}"].cumsum()
    df["cum_graduates"] = df["graduates_total"].cumsum()
    df["cum_graduate_completions"] = df["graduate_completions_total"].cumsum()
    return df


# ===========================================================================
# Cost and cross-country
# ===========================================================================

def cost_block(row: pd.Series, df: pd.DataFrame, p: Params) -> Optional[Dict[str, object]]:
    """ILLUSTRATIVE wage bill only (post holders x one sourced unit cost).
    Not a budget-impact or cost-effectiveness analysis: training costs are
    notes, not scenario inputs, and there is no discounting or comparator."""
    iso = row.get("iso3")
    if iso not in COST_PER_FTE_USD:
        return None
    unit = COST_PER_FTE_USD[iso]
    final = df.iloc[-1]
    return {
        "unit_cost_usd": unit,
        "basis": COST_NOTES[iso],
        "annual_wage_bill_now": df.iloc[0]["deployed_heads"] * unit,
        "annual_wage_bill_final": final["deployed_heads"] * unit,
        "annual_wage_bill_at_target": final["jee_target_heads"] * unit,
        "training_cost_notes": {k: v[2] for k, v in TRAINING_COST_USD.items()},
    }


def cross_country_snapshot(countries: pd.DataFrame, count_frontline: bool,
                           attribute_pooled: bool, people_per_fe: float,
                           correct_overlap: bool = True) -> pd.DataFrame:
    """Assumption-based current density for FETP hosts.

    Uses the shared cumulative-count, overlap, and age-profile initialization.
    It does not split the estimated stock by funded-post occupancy and does not
    receive all of the selected country-page parameters.
    """
    recs = []
    for _, r2 in countries[countries.hosts_fetp == 1].iterrows():
        if pd.isna(r2.get("pop_2026")):
            continue
        p2 = default_params_for(r2)
        p2.frontline_counts_toward_target = count_frontline
        p2.attribute_pooled_to_host = attribute_pooled
        p2.correct_initial_overlap = correct_overlap
        cum = effective_cumulative_graduates(r2, p2)
        if all(v is None for v in cum.values()):
            continue
        q = supply_data_quality(r2, count_frontline)
        stock = initialize_stock(r2, p2)
        heads = compliance_headcount(stock, p2)
        raw_heads = sum((v or 0.0) * p2.survival_to_2026[t] * p2.public_sector_share[t]
                        for t, v in cum.items()
                        if v is not None and (t != "frontline" or count_frontline))
        target = float(r2["pop_2026"]) / people_per_fe
        recs.append({"country": r2["country"], "region": r2["region"],
                     "countable_heads": heads,
                     "heads_before_overlap_correction": raw_heads,
                     "target": target,
                     # ratio to the selected benchmark (1.0 = benchmark met)
                     "benchmark_ratio": heads / max(target, EPS),
                     "density_per_200k": heads / max(float(r2["pop_2026"]) / 200_000.0, EPS),
                     "gap": target - heads,
                     "interpretable": q["strictly_interpretable"],
                     "unknown_tiers": ", ".join(q["unknown_tiers"]) or "-",
                     "zero_unconfirmed_tiers": ", ".join(q["zero_unconfirmed_tiers"]) or "-"})
    cols = ["country", "region", "countable_heads", "heads_before_overlap_correction",
            "target", "benchmark_ratio", "density_per_200k", "gap", "interpretable",
            "unknown_tiers", "zero_unconfirmed_tiers"]
    snap = pd.DataFrame(recs, columns=cols)
    return snap.sort_values("benchmark_ratio", ascending=False).reset_index(drop=True)


def params_to_dict(p: Params) -> dict:
    return asdict(p)
