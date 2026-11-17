"""Value-of-information analysis for the African FETP workforce model.

Decision: which of four strategies to adopt for one country,

    status_quo      active intake and settings
    train_more      new annual intake doubled (enrolled trainees unchanged)
    fund_posts      funded posts committed up to the benchmark (B1 on,
                    ceiling 100% of the benchmark, 3-year adjustment,
                    up to 80 new posts a year)
    train_and_fund  both

valued by net monetary benefit over the projection horizon,

    NMB_s(lambda) = lambda x E_s - C_s
    E_s = discounted deployed epidemiologist-years (counted tiers, in posts)
    C_s = discounted training costs (admissions x cost per trainee, by tier)
        + discounted salary costs (deployed epidemiologists x cost per post-year)

where lambda is the decision maker's willingness to pay per deployed
epidemiologist-year, inclusive of the salary it requires.

Uncertainty: the model's PSA_RANGES (except the intake multiplier, which the
strategies define) plus unit-cost ranges, sampled by Latin hypercube. Each draw
reconstructs the base-year workforce once and runs all four strategies from
the same initial conditions.

  EVPI(lambda)   = E[max_s NMB_s] - max_s E[NMB_s]
  EVPPI(phi)     = E_phi[max_s E(NMB_s | phi)] - max_s E[NMB_s]

EVPPI is estimated by regression on the probabilistic sample (Strong, Oakley &
Brennan 2014): incremental NMB versus status quo is regressed on cubic
regression splines of the parameter (additive across parameters for groups,
with fewer knots per parameter to limit overfitting), and the fitted values
replace the conditional expectations. The additive form can understate group
EVPPI when parameters interact; overfitting can overstate it when EVPI is
small. Tables therefore report a placebo noise floor, bootstrap intervals,
and cap estimates at EVPI.

All input ranges are analyst-selected uniform intervals, not elicited or
fitted distributions. EVPI and EVPPI are therefore conditional on those
ranges: they rank where better information would most change the decision
under the stated assumptions; they are not calibrated monetary values.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import fetp_model as M
import fetp_validate as V
from fetp_model import Params, TIERS

STRATEGIES = ("status_quo", "train_more", "fund_posts", "train_and_fund")
STRATEGY_LABEL = {
    "status_quo": "Status quo",
    "train_more": "Train more (2x intake)",
    "fund_posts": "Fund posts to benchmark",
    "train_and_fund": "Train more + fund posts",
}

# Unit costs, USD. Training costs are charged per admission.
#  frontline: Andre et al., EID 2017 (24 countries, 2015-16 USD 5,000-8,000);
#             upper bound widened to ~11,000 to allow for inflation to 2026.
#  intermediate: Wilson et al., Public Health Reports 2021 (Tanzania, ~6,200);
#             widened to 4,000-12,000 because it is a single-country figure.
#  advanced: no published per-trainee figure; wide analyst range for a
#             two-year residency with stipends, supervision and field work.
COST_RANGES: Dict[str, Tuple[float, float, str]] = {
    "cost_frontline": (5_000.0, 11_000.0, "USD/trainee"),
    "cost_intermediate": (4_000.0, 12_000.0, "USD/trainee"),
    "cost_advanced": (20_000.0, 120_000.0, "USD/trainee"),
}
# Salary cost per filled post-year. Kenya: Machakos County Job Group Q range
# (fetp_model.COST_NOTES). Elsewhere: wide analyst range.
POST_COST_RANGE_DEFAULT = (6_000.0, 30_000.0)
POST_COST_RANGE_BY_ISO = {"KEN": (15_500.0, 21_600.0)}

PARAMETER_GROUPS: Dict[str, Tuple[str, ...]] = {
    "Funded-post response (B1)": ("political_ambition", "post_adjust_years"),
    "Retention and exits": ("attrition_advanced", "emigration_advanced", "retirement_age"),
    "Starting workforce and public entry": ("survival_advanced", "survival_intermediate",
                                            "public_share_advanced", "overlap_adv_from_int"),
    "Training pipeline and mentors": ("completion_advanced", "mentor_share",
                                      "residents_per_mentor"),
    "Feedback strengths (R2, R3)": ("burnout_sensitivity", "emigration_sensitivity"),
    "Demand and capability": ("protected_time_advanced", "skill_weight_intermediate",
                              "annual_outbreaks_per_million"),
    "Training costs": ("cost_frontline", "cost_intermediate", "cost_advanced"),
    "Salary cost per post": ("post_cost",),
}


def voi_ranges(row: pd.Series) -> Dict[str, Tuple[float, float, str]]:
    """Sampled inputs: model ranges (without the intake multiplier) + costs."""
    out = {nm: v for nm, v in V.PSA_RANGES.items() if nm != "intake_multiplier"}
    out.update(COST_RANGES)
    lo, hi = POST_COST_RANGE_BY_ISO.get(row.get("iso3"), POST_COST_RANGE_DEFAULT)
    out["post_cost"] = (lo, hi, "USD/post-year")
    return out


def _lhs(n: int, ranges: Dict[str, Tuple[float, float, str]], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cut = np.arange(n + 1) / n
    cols = {}
    for nm, (lo, hi, _) in ranges.items():
        u = rng.permutation(rng.uniform(cut[:n], cut[1:]))
        cols[nm] = lo + u * (hi - lo)
    return pd.DataFrame(cols)


def make_strategies(q: Params) -> Dict[str, Params]:
    """The four strategies, built from one parameter draw."""
    out = {"status_quo": copy.deepcopy(q)}
    t = copy.deepcopy(q)
    t.annual_intake = {k: v * 2.0 for k, v in q.annual_intake.items()}
    out["train_more"] = t

    def fund(x: Params) -> Params:
        f = copy.deepcopy(x)
        f.post_feedback = True
        f.political_ambition = 1.0
        f.post_adjust_years = 3.0
        f.max_post_growth_per_year = 80.0
        return f

    out["fund_posts"] = fund(q)
    out["train_and_fund"] = fund(t)
    return out


def discounted_totals(df: pd.DataFrame, dt: float, discount: float,
                      costs: Dict[str, float]) -> Dict[str, float]:
    """Discounted effect and costs over the horizon (interval at row i uses
    the state at its start, discounted to its midpoint)."""
    d = df.iloc[:-1]
    t_mid = d.time_years.to_numpy() + dt / 2.0
    w = dt * (1.0 + discount) ** (-t_mid)
    deployed = d.deployed_heads.to_numpy()
    effect = float((deployed * w).sum())
    train = float(((d.intake_eff_advanced.to_numpy() * costs["cost_advanced"]
                    + d.intake_eff_intermediate.to_numpy() * costs["cost_intermediate"]
                    + d.intake_eff_frontline.to_numpy() * costs["cost_frontline"]) * w).sum())
    wage = float((deployed * costs["post_cost"] * w).sum())
    return {"effect": effect, "cost_training": train, "cost_salary": wage,
            "cost_total": train + wage}


def simulate_draws(row: pd.Series, p: Params, n: int = 500, horizon: int = 25,
                   discount: float = 0.03,
                   cohort_history: Optional[pd.DataFrame] = None,
                   dt: float = M.DT_DEFAULT, seed: int = 20261117,
                   progress=None) -> pd.DataFrame:
    """Run every strategy for each Latin-hypercube draw.

    Returns one row per draw: sampled inputs, then for each strategy s the
    columns E_<s>, Ctrain_<s>, Csal_<s>, C_<s>. Failed draws are dropped and
    counted in the `failed_draws` attribute.
    """
    if n < 10:
        raise ValueError("n must be at least 10")
    if not 0.0 <= discount < 0.2:
        raise ValueError("discount must be in [0, 0.2)")
    ranges = voi_ranges(row)
    X = _lhs(n, ranges, seed)
    base_intake = {t: max(0.0, float(p.annual_intake.get(t, 0.0))) for t in TIERS}
    recs, failed = [], 0
    for i in range(n):
        s = X.iloc[i].to_dict()
        q = V._apply_sample(p, s, base_intake)
        costs = {k: s[k] for k in ("cost_frontline", "cost_intermediate",
                                   "cost_advanced", "post_cost")}
        try:
            stock0 = M.initialize_stock(row, q, cohort_history)
            rec = dict(s)
            for nm, qs in make_strategies(q).items():
                df = M.run(row, qs, horizon=horizon, dt=dt,
                           initial_pipeline_intake=base_intake, initial_stock=stock0)
                tot = discounted_totals(df, dt, discount, costs)
                rec[f"E_{nm}"] = tot["effect"]
                rec[f"Ctrain_{nm}"] = tot["cost_training"]
                rec[f"Csal_{nm}"] = tot["cost_salary"]
                rec[f"C_{nm}"] = tot["cost_total"]
            recs.append(rec)
        except Exception:
            failed += 1
        if progress is not None:
            progress((i + 1) / n)
    out = pd.DataFrame(recs)
    out.attrs["failed_draws"] = failed
    out.attrs["sampled_draws"] = n
    out.attrs["ranges"] = ranges
    out.attrs["discount"] = discount
    out.attrs["horizon"] = horizon
    return out


# ---------------------------------------------------------------------------
# Decision analysis on the draws
# ---------------------------------------------------------------------------

def nmb_matrix(draws: pd.DataFrame, lam: float,
               strategies: Sequence[str] = STRATEGIES) -> np.ndarray:
    return np.column_stack([lam * draws[f"E_{s}"].to_numpy() - draws[f"C_{s}"].to_numpy()
                            for s in strategies])


def evpi(nmb: np.ndarray) -> float:
    return float(np.mean(nmb.max(axis=1)) - nmb.mean(axis=0).max())


def summary_table(draws: pd.DataFrame, lam: float) -> pd.DataFrame:
    nmb = nmb_matrix(draws, lam)
    best = nmb.argmax(axis=1)
    rows = []
    for j, s in enumerate(STRATEGIES):
        rows.append({
            "strategy": STRATEGY_LABEL[s],
            "deployed_epi_years": draws[f"E_{s}"].mean(),
            "training_cost_usd": draws[f"Ctrain_{s}"].mean(),
            "salary_cost_usd": draws[f"Csal_{s}"].mean(),
            "total_cost_usd": draws[f"C_{s}"].mean(),
            "nmb_usd": nmb[:, j].mean(),
            "prob_best": float((best == j).mean()),
        })
    t = pd.DataFrame(rows)
    t["expected_best"] = t.nmb_usd == t.nmb_usd.max()
    return t


def frontier(draws: pd.DataFrame) -> pd.DataFrame:
    """Mean costs and effects with dominance and ICERs (cost per additional
    deployed epidemiologist-year) along the efficiency frontier."""
    t = pd.DataFrame({"strategy": [STRATEGY_LABEL[s] for s in STRATEGIES],
                      "effect": [draws[f"E_{s}"].mean() for s in STRATEGIES],
                      "cost": [draws[f"C_{s}"].mean() for s in STRATEGIES]})
    t = t.sort_values(["effect", "cost"]).reset_index(drop=True)
    t["status"], t["icer"] = "", np.nan
    keep = list(range(len(t)))
    # strict dominance: more costly and no more effective than another
    for i in range(len(t)):
        for j in range(len(t)):
            if i != j and t.effect[j] >= t.effect[i] and t.cost[j] <= t.cost[i] and \
                    (t.effect[j] > t.effect[i] or t.cost[j] < t.cost[i]):
                t.loc[i, "status"] = "dominated"
    keep = [i for i in keep if t.status[i] == ""]
    # extended dominance
    changed = True
    while changed and len(keep) > 2:
        changed = False
        icers = [(t.cost[keep[k]] - t.cost[keep[k - 1]]) /
                 max(t.effect[keep[k]] - t.effect[keep[k - 1]], 1e-12)
                 for k in range(1, len(keep))]
        for k in range(1, len(icers)):
            if icers[k] < icers[k - 1]:
                t.loc[keep[k], "status"] = "extendedly dominated"
                keep.pop(k)
                changed = True
                break
    for k in range(1, len(keep)):
        a, b = keep[k - 1], keep[k]
        t.loc[b, "icer"] = (t.cost[b] - t.cost[a]) / max(t.effect[b] - t.effect[a], 1e-12)
    t.loc[t.status == "", "status"] = "frontier"
    return t


def ceac_and_evpi(draws: pd.DataFrame, lambdas: Iterable[float]) -> pd.DataFrame:
    rows = []
    for lam in lambdas:
        nmb = nmb_matrix(draws, lam)
        best = nmb.argmax(axis=1)
        rec = {"wtp": float(lam), "evpi_usd": evpi(nmb),
               "expected_best": STRATEGY_LABEL[STRATEGIES[int(nmb.mean(axis=0).argmax())]]}
        for j, s in enumerate(STRATEGIES):
            rec[f"prob_best_{s}"] = float((best == j).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EVPPI by spline regression (Strong, Oakley & Brennan 2014)
# ---------------------------------------------------------------------------

def _spline_basis(x: np.ndarray, n_knots: int = 4) -> np.ndarray:
    """Cubic regression spline basis (truncated power, interior knots at
    quantiles), without intercept, on a standardized scale."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd <= 0:
        return np.zeros((len(x), 0))
    z = (x - x.mean()) / sd
    cols = [z, z ** 2, z ** 3]
    for k in np.quantile(z, np.linspace(0, 1, n_knots + 2)[1:-1]):
        cols.append(np.clip(z - k, 0.0, None) ** 3)
    return np.column_stack(cols)


def _fitted(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ beta


def _knots_for(n_params: int, n_knots: Optional[int]) -> int:
    """Fewer knots per parameter in groups, to limit overfitting."""
    if n_knots is not None:
        return n_knots
    return 4 if n_params == 1 else 2


def _evppi_from(nmb: np.ndarray, X: np.ndarray) -> float:
    inc = nmb[:, 1:] - nmb[:, [0]]                 # vs status quo
    if X.shape[1] == 0:
        return 0.0
    g = np.column_stack([np.zeros(len(inc))] + [_fitted(inc[:, j], X) for j in range(inc.shape[1])])
    return max(0.0, float(np.mean(g.max(axis=1)) - max(0.0, float(g.mean(axis=0).max()))))


def _basis(draws: pd.DataFrame, params: Sequence[str], n_knots: Optional[int]) -> np.ndarray:
    k = _knots_for(len(params), n_knots)
    return np.column_stack([_spline_basis(draws[nm].to_numpy(), k) for nm in params])


def evppi(draws: pd.DataFrame, lam: float, params: Sequence[str],
          n_knots: Optional[int] = None) -> float:
    """Regression-based EVPPI for a parameter or an (additive) group.
    Not capped; see `evppi_table` for capping and uncertainty."""
    return _evppi_from(nmb_matrix(draws, lam), _basis(draws, params, n_knots))


def evppi_bootstrap(draws: pd.DataFrame, lam: float, params: Sequence[str],
                    n_boot: int = 200, seed: int = 11,
                    n_knots: Optional[int] = None) -> Tuple[float, float]:
    """Percentile interval (2.5%, 97.5%) for EVPPI from resampling the draws.
    Reflects Monte Carlo and regression uncertainty, not input uncertainty."""
    nmb = nmb_matrix(draws, lam)
    X = _basis(draws, params, n_knots)
    rng = np.random.default_rng(seed)
    n = len(draws)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(_evppi_from(nmb[i], X[i]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evppi_table(draws: pd.DataFrame, lam: float,
                groups: Optional[Dict[str, Sequence[str]]] = None,
                n_boot: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(group EVPPI, single-parameter EVPPI) at willingness to pay `lam`.

    Each table includes a placebo row (an input of pure random noise, or a
    group of three) as the estimator's noise floor, a bootstrap interval, and
    caps estimates at EVPI (flagged), since EVPPI cannot exceed EVPI. When
    EVPI is small relative to the incremental NMB, regression noise dominates
    and the estimates should not be interpreted.
    """
    groups = dict(groups or PARAMETER_GROUPS)
    ranges = draws.attrs.get("ranges", {})
    d = draws.copy()
    rng = np.random.default_rng(2026)
    for k in range(3):
        d[f"_placebo{k}"] = rng.uniform(size=len(d))
    total = evpi(nmb_matrix(d, lam))

    def row_for(label_key, label, ps, extra):
        raw = evppi(d, lam, ps)
        lo, hi = evppi_bootstrap(d, lam, ps, n_boot=n_boot) if n_boot else (np.nan, np.nan)
        return {label_key: label, **extra,
                "evppi_usd": min(raw, total), "ci_low": min(lo, total), "ci_high": min(hi, total),
                "capped_at_evpi": raw > total + 1e-9}

    g_rows = [row_for("group", nm, [p for p in ps if p in d.columns], {"parameters": ", ".join(ps)})
              for nm, ps in groups.items()]
    g_rows.append(row_for("group", "Placebo (3 random inputs)",
                          ["_placebo0", "_placebo1", "_placebo2"], {"parameters": "noise floor"}))
    s_rows = [row_for("parameter", nm, [nm], {"units": ranges[nm][2],
                                              "range": f"{ranges[nm][0]:g}–{ranges[nm][1]:g}"})
              for nm in ranges if nm in d.columns]
    s_rows.append(row_for("parameter", "Placebo (random input)", ["_placebo0"],
                          {"units": "", "range": "noise floor"}))
    g, s = pd.DataFrame(g_rows), pd.DataFrame(s_rows)
    for t in (g, s):
        t["share_of_evpi"] = t.evppi_usd / total if total > 0 else np.nan
    g = g.sort_values("evppi_usd", ascending=False).reset_index(drop=True)
    s = s.sort_values("evppi_usd", ascending=False).reset_index(drop=True)
    g.attrs["evpi_usd"] = s.attrs["evpi_usd"] = total
    return g, s


# ---------------------------------------------------------------------------
# Method checks on synthetic data (no model runs)
# ---------------------------------------------------------------------------

def method_check(n: int = 4000, seed: int = 7) -> Tuple[bool, str]:
    """Two-option decision with incremental NMB = 10 (x - 0.5), x ~ U(0,1),
    and an irrelevant input z. Analytic EVPI = EVPPI(x) = 1.25; EVPPI(z) = 0."""
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(size=n), rng.uniform(size=n)
    d = pd.DataFrame({"x": x, "z": z})
    # Encode as four strategies: only train_more differs from status quo.
    d["E_status_quo"], d["C_status_quo"] = 0.0, 0.0
    d["E_train_more"], d["C_train_more"] = 10.0 * (x - 0.5), 0.0
    for s in ("fund_posts", "train_and_fund"):
        d[f"E_{s}"], d[f"C_{s}"] = -1e6, 0.0       # never chosen
    ev = evpi(nmb_matrix(d, 1.0))
    ex = evppi(d, 1.0, ["x"])
    ez = evppi(d, 1.0, ["z"])
    ok = abs(ev - 1.25) < 0.06 and abs(ex - ev) < 0.02 and ez < 0.05
    return ok, f"EVPI {ev:.3f} (analytic 1.25); EVPPI(x) {ex:.3f}; EVPPI(irrelevant z) {ez:.3f}"
