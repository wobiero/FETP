"""
Validation and uncertainty analysis for the African FETP workforce model.

The structural checks cover selected accounting identities, limiting cases,
analytic behavior, and unit relationships. They do not establish empirical
fit or predictive validity, and they need to be run in the target environment.

Sensitivity analysis uses Latin hypercube sampling with partial rank correlation
coefficients (Blower & Dowlatabadi 1994; Marino et al. 2008). IMPORTANT: every
input range below is an analyst-chosen uniform interval, not an elicited or
fitted distribution. The output is therefore a CONDITIONAL SENSITIVITY
ANALYSIS: it shows how results move across those ranges, and its percentiles
and "share of runs meeting the target" are NOT calibrated probabilities or
confidence intervals.

Every check returns a (passed, detail) pair for review in the app or from a
local script.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import fetp_model as M
from fetp_model import Params, TIERS


# ===========================================================================
# Structural tests.  Every test takes (row, p) where p is the ACTIVE parameter
# set (the sidebar values in the app), so results describe the scenario the
# user is looking at, not a default rebuilt from the country row.
# ===========================================================================

def _clone(p: Params) -> Params:
    return copy.deepcopy(p)


def prepare_params(row: pd.Series, p: Optional[Params] = None) -> Params:
    """Active params if given, else defaults with data-implied intake."""
    if p is not None:
        return _clone(p)
    q = M.default_params_for(row)
    q.annual_intake = M.implied_historical_intake(row, q)
    return q


def _all_loops(p: Params, on: bool) -> Params:
    p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = on
    return p


def _isolated(row: pd.Series, p: Params) -> Tuple[pd.Series, Params]:
    """A synthetic single-tier, no-ageing-loss setting for analytic tests."""
    r = row.copy()
    for c in ("grads_adv", "grads_int", "grads_front"):
        r[c] = 0.0
    q = _clone(p)
    _all_loops(q, False)
    q.model_posts = False
    q.progression_from_below = {"advanced": 0.0, "intermediate": 0.0}
    q.retirement_age = 999.0
    q.attribute_pooled_to_host = True
    q.annual_intake = {t: 0.0 for t in TIERS}
    q.attrition = {"advanced": 0.05, "intermediate": 0.0, "frontline": 0.0}
    q.emigration = {"advanced": 0.03, "intermediate": 0.0, "frontline": 0.0}
    return r, q


class _NoMortality:
    def __enter__(self):
        self.saved = M.AGE_MORTALITY.copy()
        M.AGE_MORTALITY = np.zeros_like(self.saved)
    def __exit__(self, *a):
        M.AGE_MORTALITY = self.saved
        return False


def test_conservation(row: pd.Series, p: Params, horizon: int = 25,
                      cohort_history: Optional[pd.DataFrame] = None,
                      dt: float = M.DT_DEFAULT) -> Tuple[bool, str]:
    """People are neither created nor destroyed outside the named flows.

    trained(T) - trained(0) == tracked_inflow - (attrition + emigration +
                                                 mortality + retirement)

    `trained` is serving + unabsorbed pool. Progression remains enabled; it
    must transfer people between tiers without changing the all-tier total."""
    p2 = _clone(p)
    df = M.run(row, p2, horizon=horizon, cohort_history=cohort_history, dt=dt)
    first, last = df.iloc[0], df.iloc[-1]
    cols = ["heads_advanced", "heads_intermediate", "heads_frontline"]
    start, end = float(first[cols].sum()), float(last[cols].sum())
    inflow = float(df["graduates_total"].sum())
    outflow = float(sum(df[f"exit_{c}"].sum()
                        for c in ("attrition", "emigration", "mortality", "retirement")))
    resid = (end - start) - (inflow - outflow)
    scale = max(start, inflow, 1.0)
    ok = abs(resid) / scale < 1e-9
    return ok, (f"start={start:.2f} end={end:.2f} in={inflow:.2f} out={outflow:.2f} "
                f"residual={resid:.2e} ({abs(resid)/scale:.1e} of scale, tol 1e-9)")


def test_inflow_accounting(row: pd.Series, p: Params, horizon: int = 25,
                           cohort_history: Optional[pd.DataFrame] = None,
                           dt: float = M.DT_DEFAULT) -> Tuple[bool, str]:
    """Gross completions split into internal transfers, public entrants, and
    people who do not enter the tracked public-sector workforce."""
    df = M.run(row, _clone(p), horizon=horizon,
               cohort_history=cohort_history, dt=dt)
    gross = float(df.graduate_completions_total.sum())
    accounted = float(df.graduates_total.sum() + df.grads_nonpublic.sum()
                      + df.promoted_total.sum())
    resid = gross - accounted
    ok = abs(resid) <= 1e-8 * max(gross, 1.0)
    return ok, (f"gross completions={gross:.2f}; external public inflow="
                f"{df.graduates_total.sum():.2f}; non-public={df.grads_nonpublic.sum():.2f}; "
                f"internal promotions={df.promoted_total.sum():.2f}; residual={resid:.2e}")


def test_analytic_decay(row: pd.Series, p: Params, T: int = 10) -> Tuple[bool, str]:
    """One cohort, no inflow, constant hazards: the stock must equal N0 exp(-hT)
    to machine precision (survival is integrated exactly within each step)."""
    r, q = _isolated(row, p)
    r["grads_adv"] = 1000.0
    dt = 1.0 / 48.0
    with _NoMortality():
        df = M.run(r, q, horizon=T, dt=dt)
    n0 = float(df.iloc[0].compliance_heads)
    h = q.attrition["advanced"] + q.emigration["advanced"]
    cont = n0 * np.exp(-h * T)
    got = float(df.iloc[-1].compliance_heads)
    ok = abs(got - cont) / cont < 1e-9 and n0 > 0
    return ok, f"N0={n0:.3f} got={got:.6f} N0*exp(-hT)={cont:.6f}"


def test_pool_analytic_decay(row: pd.Series, p: Params, T: int = 10) -> Tuple[bool, str]:
    """The unabsorbed pool follows the same exact constant-hazard survival."""
    q = _clone(p)
    q.attrition = {"advanced": 0.05, "intermediate": 0.0, "frontline": 0.0}
    q.emigration = {"advanced": 0.03, "intermediate": 0.0, "frontline": 0.0}
    q.emigration_feedback = False
    q.retirement_age = 999.0
    dt = 1.0 / 48.0
    pool = M.empty_pool()
    pool["advanced"][10] = 1000.0
    exits = {"advanced": 0.03, "intermediate": 0.0, "frontline": 0.0}
    with _NoMortality():
        steps = int(round(T / dt))
        for k in range(steps):
            age_now = np.floor((k + 1) * dt + M.EPS) > np.floor(k * dt + M.EPS)
            pool, _ = M.step_pool(pool, q, dt, exits, age_step=age_now)
    h = q.attrition["advanced"] + exits["advanced"]
    expected = 1000.0 * np.exp(-h * T)
    got = float(pool["advanced"].sum())
    aged_ok = abs(float(pool["advanced"][10 + T]) - got) < 1e-9
    ok = abs(got - expected) / expected < 1e-10 and aged_ok
    return ok, (f"pool 1000 -> {got:.6f}; exact 1000*exp(-hT)={expected:.6f}; "
                f"cohort aged {T} years: {aged_ok}")


def test_pool_ages_and_retires(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Pool members age and retire like post holders, and keep their age when
    hired: 100 graduates aged 58 waiting for a post retire at 60, and a hired
    pool member enters the serving stock at their own age."""
    q = _clone(p)
    q.retirement_age = 60.0
    q.emigration_feedback = False
    pool = M.empty_pool()
    i58 = int(np.where(M.AGES == 58)[0][0])
    pool["advanced"][i58] = 100.0
    dt = M.DT_DEFAULT
    retired = exited = 0.0
    for k in range(int(round(3 / dt))):
        age_now = np.floor((k + 1) * dt + M.EPS) > np.floor(k * dt + M.EPS)
        pool, ex = M.step_pool(pool, q, dt, q.emigration, age_step=age_now)
        retired += ex["retirement"]
        exited += sum(ex.values())
    left = float(pool["advanced"].sum())
    # Hiring keeps ages
    q.model_posts = True
    q.progression_from_below = {"advanced": 0.0, "intermediate": 0.0}
    stock = {t: np.zeros(len(M.AGES)) for t in TIERS}
    pool2 = M.empty_pool()
    pool2["advanced"][i58] = 10.0
    out, newp, _ = M.enter_graduates(stock, pool2, {t: 0.0 for t in TIERS}, 10.0, q)
    hired_age = float((out["advanced"] * M.AGES).sum() / max(out["advanced"].sum(), 1e-12))
    ok = (left < 1e-9 and abs(exited - 100.0) < 1e-9 and retired > 80.0
          and abs(hired_age - 58.0) < 1e-9)
    return ok, (f"left in pool after 3 years {left:.2e}; retired {retired:.1f}, other exits "
                f"{exited - retired:.1f}; "
                f"mean age of hired pool members {hired_age:.1f}")


def test_progression_in_pool(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """A lower-tier pool member promoted to advanced remains one person."""
    q = _clone(p)
    q.model_posts = True
    q.progression_from_below = {"advanced": 0.50, "intermediate": 0.0}
    q.public_entry_share = {t: 1.0 for t in TIERS}
    stock = {t: np.zeros(len(M.AGES)) for t in TIERS}
    pool = M.empty_pool()
    pool["intermediate"] = 100.0 * M.ENTRY_WEIGHTS["intermediate"]
    grads = {"advanced": 10.0, "intermediate": 0.0, "frontline": 0.0}
    out, new_pool, info = M.enter_graduates(stock, pool, grads, free_posts=0.0, p=q)
    before = sum(float(stock[t].sum()) + float(pool[t].sum()) for t in TIERS)
    after = sum(float(out[t].sum()) + float(new_pool[t].sum()) for t in TIERS)
    pi, pa = float(new_pool["intermediate"].sum()), float(new_pool["advanced"].sum())
    ok = (abs(after - (before + info["tracked_inflow"])) < 1e-9
          and abs(pi - 95.0) < 1e-9 and abs(pa - 10.0) < 1e-9
          and abs(info["promoted"] - 5.0) < 1e-9)
    return ok, (f"total heads {before:.1f} -> {after:.1f}; intermediate pool="
                f"{pi:.1f}; advanced pool={pa:.1f}; "
                f"promoted={info['promoted']:.1f}; new inflow={info['tracked_inflow']:.1f}")


def test_analytic_equilibrium(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Constant intake I, hazard h, unlimited posts: stock -> I c s / h."""
    r, q = _isolated(row, p)
    q.annual_intake = {"advanced": 20.0, "intermediate": 0.0, "frontline": 0.0}
    with _NoMortality():
        df = M.run(r, q, horizon=220)
    h = q.attrition["advanced"] + q.emigration["advanced"]
    # per-step inflow I c s dt entering a stock with survival exp(-h dt):
    # equilibrium = I c s dt / (1 - exp(-h dt)) ~ I c s / h (exact as dt->0)
    dt = M.DT_DEFAULT
    expected = (20.0 * q.completion["advanced"] * q.public_entry_share["advanced"]
                * dt / (1.0 - np.exp(-h * dt)))
    got = float(df.iloc[-1].compliance_heads)
    ok = abs(got - expected) / expected < 1e-6
    cont = 20.0 * q.completion["advanced"] * q.public_entry_share["advanced"] / h
    return ok, (f"equilibrium {got:.4f} vs discrete-exact {expected:.4f} "
                f"(continuous I*c*s/h = {cont:.4f})")


def test_ageing_exact_and_dt_free(row: pd.Series, p: Params, T: int = 25) -> Tuple[bool, str]:
    """No hazards, no inflow: the people who retire by year T are exactly those
    whose age reaches the retirement age within T annual shifts, and the answer
    must not depend on dt (no numerical diffusion in the age chain)."""
    q = _clone(p)
    _all_loops(q, False)
    q.attrition = {t: 0.0 for t in TIERS}
    q.emigration = {t: 0.0 for t in TIERS}
    q.annual_intake = {t: 0.0 for t in TIERS}
    q.model_posts = False
    q.progression_from_below = {"advanced": 0.0, "intermediate": 0.0}
    with _NoMortality():
        st0 = M.initialize_stock(row, q)
        expected = float(sum(st0[t][M.AGES >= q.retirement_age - T].sum() for t in TIERS))
        got = {dt: float(M.run(row, q, horizon=T, dt=dt).exit_retirement.sum())
               for dt in (1 / 12, 1 / 48)}
    ok = all(abs(g - expected) < 1e-8 * max(expected, 1.0) for g in got.values())
    return ok, (f"retirees by year {T}: expected {expected:.4f}; dt=1/12 {got[1/12]:.4f}; "
                f"dt=1/48 {got[1/48]:.4f}")


def test_extreme_no_inflow_no_outflow(row: pd.Series, p: Params, horizon: int = 25) -> Tuple[bool, str]:
    """With every outflow and inflow switched off the stock must be constant."""
    q = _clone(p)
    _all_loops(q, False)
    q.attrition = {t: 0.0 for t in TIERS}
    q.emigration = {t: 0.0 for t in TIERS}
    q.annual_intake = {t: 0.0 for t in TIERS}
    q.retirement_age = 999.0
    with _NoMortality():
        df = M.run(row, q, horizon=horizon)
        a, b = df.iloc[0].compliance_heads, df.iloc[-1].compliance_heads
    ok = abs(a - b) < max(1e-6, 1e-6 * max(a, 1.0))
    return ok, f"heads {a:.6f} -> {b:.6f} (drift {b-a:+.2e})"


def test_extreme_total_attrition(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """100%/yr attrition empties the stock and never drives it negative."""
    q = _clone(p)
    q.attrition = {t: 1.0 for t in TIERS}
    q.annual_intake = {t: 0.0 for t in TIERS}
    df = M.run(row, q, horizon=10)
    end = df.iloc[-1].compliance_heads
    cols = ["heads_advanced", "heads_intermediate", "heads_frontline"]
    neg = bool((df[cols] < -1e-9).any().any())
    ok = end < 0.01 * max(df.iloc[0].compliance_heads, 1e-9) and not neg
    return ok, f"heads {df.iloc[0].compliance_heads:.1f} -> {end:.4f}; negative stock: {neg}"


def test_extreme_zero_population(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Near-zero population implies near-zero JEE target and routine workload."""
    r = row.copy()
    for y in M.WPP_ANCHORS:
        r[f"pop_{y}"] = 1.0
    df = M.run(r, _clone(p), horizon=5)
    tgt, rout = df.iloc[0].jee_target_heads, df.iloc[0].routine_fte
    ok = tgt < 1e-4 and rout < 1e-3
    return ok, f"target={tgt:.3e} routine_fte={rout:.3e} (readiness FTE stays by design)"


def test_extreme_infinite_posts(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """If posts never bind there is no unabsorbed pool and deployed == trained."""
    q = _clone(p)
    q.model_posts = True
    q.initial_posts_ratio = 1000.0
    q.post_feedback = False
    q.post_creation_per_year = 0.0
    df = M.run(row, q, horizon=25)
    d = float((df.deployed_heads - df.compliance_heads).abs().max())
    u = float(df.unabsorbed_heads.max())
    return (d < 1e-6 and u < 1e-6), f"max |deployed - trained| = {d:.2e}; max pool = {u:.2e}"


def test_pool_identity(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """R3 acts on the identifiable pool only: with the loop on and a pool
    present, the pool hazard exceeds the base while post-holder emigration is
    untouched (checked via the loop-on/off difference in exits)."""
    q = _clone(p)
    _all_loops(q, False)
    st0 = M.initialize_stock(row, q)
    if sum(float(st0[t].sum()) for t in M.counted_tiers(q)) <= 0:
        return True, ("not applicable: no advanced/intermediate starting stock, so no "
                      "pool can exist at t0")
    q.model_posts = True
    q.initial_posts_ratio = 0.5            # force a pool at t0
    q.post_creation_per_year = 0.0
    q.emigration_feedback = True
    q.emigration_sensitivity = 2.0
    q.emigration_max_multiplier = 4.0
    on = M.run(row, q, horizon=3)
    q.emigration_feedback = False
    off = M.run(row, q, horizon=3)
    # Compare after one simulated year (t0 is identical by construction).
    i1 = int(round(1.0 / M.DT_DEFAULT))
    pool0 = float(on.iloc[0].unabsorbed_heads)
    rate_on, rate_off = float(on.iloc[0].emigration_rate_adv), float(off.iloc[0].emigration_rate_adv)
    emi_on = float(on.exit_emigration.iloc[1:i1 + 1].sum())
    emi_off = float(off.exit_emigration.iloc[1:i1 + 1].sum())
    pool_on, pool_off = float(on.iloc[i1].unabsorbed_heads), float(off.iloc[i1].unabsorbed_heads)
    serv_diff = abs(float(on.iloc[i1].deployed_heads) - float(off.iloc[i1].deployed_heads))
    # While the pool exceeds the vacancies, absorption is limited by posts in
    # both runs, so post holders must be unaffected by R3.
    serv_same = serv_diff < 1e-6 * max(float(off.iloc[i1].deployed_heads), 1.0)
    ok = (pool0 > 0 and rate_on > rate_off and emi_on > emi_off
          and pool_on < pool_off and serv_same)
    return ok, (f"pool0={pool0:.1f}; pool emigration hazard {rate_off:.3f} -> {rate_on:.3f}; "
                f"year-1 emigration {emi_off:.2f} -> {emi_on:.2f}; pool at year 1 "
                f"{pool_off:.1f} -> {pool_on:.1f}; post holders differ by {serv_diff:.1e}")


def test_cohort_history_losses(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Cohort-history initialization: with no mortality and no retirement the
    stock equals completers x public share x exp(-(attrition+emigration) t);
    mortality and retirement then each REDUCE the headcount."""
    q = _clone(p)
    q.attrition = {t: 0.03 for t in TIERS}
    q.emigration = {t: 0.03 for t in TIERS}
    yr = M.BASE_YEAR - 26
    ch = pd.DataFrame({"year": [yr], "tier": ["advanced"], "intake": [100.0]})
    elapsed = M.BASE_YEAR - yr - q.tier_months["advanced"] / 12.0
    expected = (100.0 * q.completion["advanced"] * q.public_sector_share["advanced"]
                * np.exp(-0.06 * elapsed))
    q.retirement_age = 999.0
    with _NoMortality():
        bare = float(M.initialize_stock(row, q, ch)["advanced"].sum())
    with_mort = float(M.initialize_stock(row, q, ch)["advanced"].sum())
    q.retirement_age = 60.0
    with_ret = float(M.initialize_stock(row, q, ch)["advanced"].sum())
    ok = abs(bare - expected) < 1e-6 * expected and with_mort < bare and with_ret < with_mort
    return ok, (f"no mortality/retirement {bare:.2f} (expected {expected:.2f}); "
                f"+mortality {with_mort:.2f}; +retirement at 60 {with_ret:.2f}")


def test_posts_not_exceeded(row: pd.Series, p: Params, horizon: int = 25) -> Tuple[bool, str]:
    """With fixed posts and no B1, deployed counted staff never exceed funded
    posts, including when frontline staff are promoted into intermediate."""
    q = _all_loops(_clone(p), False)
    q.model_posts = True
    q.initial_posts_ratio = 0.8
    q.post_creation_per_year = 0.0
    q.frontline_counts_toward_target = False
    q.progression_from_below = {"advanced": 0.3, "intermediate": 0.6}
    q.annual_intake = {"advanced": max(p.annual_intake["advanced"], 10.0),
                       "intermediate": max(p.annual_intake["intermediate"], 20.0),
                       "frontline": max(p.annual_intake["frontline"], 50.0)}
    df = M.run(row, q, horizon=horizon)
    excess = float((df.deployed_heads - df.established_posts).max())
    ok = excess <= 1e-6 * max(float(df.established_posts.iloc[0]), 1.0)
    return ok, f"max(deployed - posts) = {excess:.2e} with posts fixed at {df.established_posts.iloc[0]:.1f}"


def test_mentor_capacity_shared(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """R1: advanced mentors are shared, so mentor-years used by the capped
    advanced and intermediate intakes never exceed the mentors available,
    and capacity does not bind when demand is below it."""
    q = _clone(p)
    q.mentor_feedback = True
    q.mentor_tier_eligible = "advanced"
    stock = {t: np.zeros(len(M.AGES)) for t in TIERS}
    stock["advanced"][10] = 40.0
    mentors = 40.0 * q.mentor_share
    per_a = q.tier_months["advanced"] / 12.0 / q.residents_per_mentor
    per_i = q.tier_months["intermediate"] / 12.0 / q.residents_per_mentor_int
    big = {"advanced": 1e4, "intermediate": 1e4, "frontline": 0.0}
    cap = M.f_training_capacity(stock, q, big)
    used = cap["advanced"] * per_a + cap["intermediate"] * per_i
    small = {"advanced": 1.0, "intermediate": 1.0, "frontline": 0.0}
    cap_s = M.f_training_capacity(stock, q, small)
    ok = (abs(used - mentors) < 1e-9 * max(mentors, 1.0)
          and cap_s["advanced"] >= 1.0 and cap_s["intermediate"] >= 1.0)
    return ok, (f"mentors {mentors:.2f}; mentor-years used at saturation {used:.2f}; "
                f"caps under low demand adv {cap_s['advanced']:.1f}, int {cap_s['intermediate']:.1f}")


def test_scenario_pipeline(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """All scenarios inherit the enrolled trainees: completions in the first
    frontline duration are identical across scenarios."""
    import fetp_scenarios as SC
    runs = SC.run_scenarios(row, horizon=3, feedback=False, base_params=p)
    k = max(1, int(round(p.tier_months["frontline"] / 12.0 / M.DT_DEFAULT)))
    ref = runs["status_quo"].graduate_completions_total.iloc[1:k + 1].to_numpy()
    worst = max(float(np.abs(df.graduate_completions_total.iloc[1:k + 1].to_numpy() - ref).max())
                for df in runs.values())
    return worst < 1e-9, (f"max difference from status_quo in first {k} steps' "
                          f"completions = {worst:.1e}")


def _synthetic_history() -> pd.DataFrame:
    yrs = list(range(M.BASE_YEAR - 20, M.BASE_YEAR - 1))
    return pd.DataFrame({"year": yrs * 2,
                         "tier": ["advanced"] * len(yrs) + ["intermediate"] * len(yrs),
                         "intake": [12.0] * len(yrs) + [10.0] * len(yrs)})


def test_scenario_common_start(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """With a cohort history, every scenario starts from the same workforce:
    prospective assumptions (e.g. halved attrition) must not be applied to
    the historical reconstruction."""
    import fetp_scenarios as SC
    runs = SC.run_scenarios(row, horizon=1, feedback=True, base_params=p,
                            cohort_history=_synthetic_history())
    t0 = {nm: float(df.compliance_heads.iloc[0]) for nm, df in runs.items()}
    spread = max(t0.values()) - min(t0.values())
    return spread < 1e-9, "t0 trained heads: " + ", ".join(f"{k} {v:.2f}" for k, v in t0.items())


def test_r1_baseline_feasible(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """With external mentors calibrated (the default), R1 does not ration the
    baseline intake at the base year, even with no domestic advanced staff."""
    q = _clone(p)
    q.mentor_feedback = True
    q.external_mentors = None
    r = row.copy()
    r["grads_adv"] = 0.0
    q.annual_intake = {"advanced": 5.0, "intermediate": 20.0, "frontline": 40.0}
    df = M.run(r, q, horizon=2)
    rationed = float(df.intake_rationed_advanced.iloc[0] + df.intake_rationed_intermediate.iloc[0])
    q.external_mentors = 0.0
    df0 = M.run(r, q, horizon=2)
    rationed0 = float(df0.intake_rationed_intermediate.iloc[0])
    ok = rationed < 1e-9 and rationed0 > 0
    return ok, (f"no domestic advanced staff: rationed at t0 with calibrated external mentors "
                f"{rationed:.2e} (ext={df.external_mentors.iloc[0]:.2f}); with none {rationed0:.1f}")


def test_r1_uses_enrolment(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """R1 counts trainees already enrolled: when continuing trainees occupy all
    mentors, no new admissions are possible; with no one enrolled the
    enrolment form allows admissions."""
    q = _clone(p)
    q.mentor_feedback = True
    q.mentor_tier_eligible = "advanced"
    stock = {t: np.zeros(len(M.AGES)) for t in TIERS}
    stock["advanced"][10] = 20.0
    mentors = 20.0 * q.mentor_share
    full = {"advanced": mentors * q.residents_per_mentor, "intermediate": 0.0}
    want = {"advanced": 10.0, "intermediate": 10.0, "frontline": 0.0}
    cap_full = M.f_training_capacity(stock, q, want, full, 1 / 12)
    cap_empty = M.f_training_capacity(stock, q, want, {"advanced": 0.0, "intermediate": 0.0}, 1 / 12)
    ok = (cap_full["advanced"] < 1e-9 and cap_full["intermediate"] < 1e-9
          and cap_empty["advanced"] >= 10.0 and cap_empty["intermediate"] >= 10.0)
    return ok, (f"mentors fully occupied: caps adv {cap_full['advanced']:.2f}, int "
                f"{cap_full['intermediate']:.2f}; none enrolled: adv {cap_empty['advanced']:.0f}, "
                f"int {cap_empty['intermediate']:.0f}")


def test_flow_timing(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Flows on row k belong to the interval ending at row k: the base-year
    row has no flows, and the stock change between rows equals that row's
    net inflow (open loop, all tiers)."""
    df = M.run(row, _clone(p), horizon=2)
    flow_cols = [c for c in M.FLOW_COLUMNS]
    first_zero = float(df.iloc[0][flow_cols].abs().sum()) == 0.0
    cols = ["heads_advanced", "heads_intermediate", "heads_frontline"]
    tot = df[cols].sum(axis=1)
    net = df.graduates_total - sum(df[f"exit_{c}"] for c in
                                   ("attrition", "emigration", "mortality", "retirement"))
    resid = float((tot.diff().iloc[1:] - net.iloc[1:]).abs().max())
    ok = first_zero and resid < 1e-8 * max(float(tot.max()), 1.0)
    return ok, f"base-year row flows zero: {first_zero}; max per-row balance residual {resid:.1e}"


def test_input_validation(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Invalid inputs are rejected rather than simulated."""
    cases = []
    q = _clone(p); q.annual_intake = dict(q.annual_intake, advanced=-1.0); cases.append(("negative intake", q, None))
    q = _clone(p); q.completion = dict(q.completion, advanced=1.5); cases.append(("completion > 1", q, None))
    q = _clone(p); q.attrition = dict(q.attrition, frontline=float("nan")); cases.append(("NaN attrition", q, None))
    bad_ch = pd.DataFrame({"year": [2010], "tier": ["expert"], "intake": [5]})
    cases.append(("unknown cohort tier", _clone(p), bad_ch))
    rejected = []
    for name, q, ch in cases:
        try:
            M.run(row, q, horizon=1, cohort_history=ch)
        except ValueError:
            rejected.append(name)
    ok = len(rejected) == len(cases)
    return ok, f"rejected {len(rejected)} of {len(cases)}: {', '.join(rejected) or 'none'}"


def test_overlap_report(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """The overlap check flags assumptions that exceed recorded counts."""
    q = _clone(p)
    q.correct_initial_overlap = True
    q.overlap_adv_from_int = 0.30
    rep = M.overlap_report({"advanced": 300.0, "intermediate": 20.0, "frontline": 500.0}, q)
    rep_ok = M.overlap_report({"advanced": 30.0, "intermediate": 200.0, "frontline": 500.0}, q)
    flagged = rep[0]["status"].startswith("incompatible") and abs(rep[0]["applied_overlap"] - 20.0) < 1e-9
    clean = rep_ok[0]["status"] == "consistent"
    return flagged and clean, (f"300 advanced vs 20 intermediate: {rep[0]['status']}; "
                               f"30 vs 200: {rep_ok[0]['status']}")


def test_integration_error(row: pd.Series, p: Params, horizon: int = 25,
                           tol: float = 0.01) -> Tuple[bool, str]:
    """Halving the time step must not change the trajectory materially."""
    q = _all_loops(_clone(p), True)
    out = {}
    for dt in (1 / 12, 1 / 24, 1 / 48):
        out[dt] = M.run(row, q, horizon=horizon, dt=dt).set_index("calendar_year").compliance_heads
    ref = out[1 / 48]
    worst, detail = 0.0, []
    for dt in (1 / 12, 1 / 24):
        s = out[dt]
        common = s.index.intersection(ref.index)
        rel = float(((s[common] - ref[common]).abs() / ref[common].clip(lower=1e-9)).max())
        if dt == 1 / 12:
            worst = rel
        detail.append(f"dt=1/{int(1/dt)}: max rel diff vs dt=1/48 = {rel:.4%}")
    return worst < tol, "; ".join(detail)


def test_switches_equal_zero_sensitivity(row: pd.Series, p: Params,
                                         horizon: int = 25) -> Tuple[bool, str]:
    """INTERNAL CONSISTENCY, not a v1 regression: with loops switched off the
    run must equal the run with loops on but sensitivities at zero and the
    mentor cap unbounded. (A true v1 comparison needs the v1 code and data,
    which are not in this bundle; see regression_report.py.)"""
    base = _clone(p)
    off = _all_loops(_clone(base), False)
    zero = _clone(base)
    zero.mentor_feedback = True
    zero.burnout_feedback = zero.emigration_feedback = True
    zero.post_feedback = False
    zero.burnout_sensitivity = 0.0
    zero.emigration_sensitivity = 0.0
    zero.external_mentors = 1e12        # mentor capacity effectively unbounded
    a = M.run(row, off, horizon=horizon).compliance_heads.values
    b = M.run(row, zero, horizon=horizon).compliance_heads.values
    rel = float(np.max(np.abs(a - b) / np.clip(np.abs(a), 1e-9, None)))
    return rel < 1e-9, f"max relative difference = {rel:.3e}"


def test_monotone_in_intake(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """More training never reduces TRAINED stock (open loop). Deployed stock is
    reported separately because posts bind."""
    q = _all_loops(_clone(p), False)
    base = dict(p.annual_intake)
    vals = []
    for mult in (0.5, 1.0, 2.0, 4.0):
        q.annual_intake = {t: base[t] * mult for t in TIERS}
        vals.append(M.run(row, q, horizon=25).iloc[-1].compliance_heads)
    ok = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
    return ok, "trained heads at intake x(0.5,1,2,4) = " + ", ".join(f"{v:.1f}" for v in vals)


def test_demand_scaling(row: pd.Series, p: Params) -> Tuple[bool, str]:
    """Numerical homogeneity checks on the REAL demand code: each output must
    scale exactly as its declared units imply."""
    pop, a2 = 1_000_000.0, 100.0
    b = M.demand_measures(pop, a2, p)
    checks = []
    q = _clone(p); q.people_per_fe *= 2
    checks.append(("target ~ 1/people_per_fe",
                   abs(M.demand_measures(pop, a2, q)["jee_target_heads"] / b["jee_target_heads"] - 0.5)))
    q = _clone(p); q.productive_hours_per_fte *= 2
    checks.append(("routine_fte ~ 1/hours_per_fte",
                   abs(M.demand_measures(pop, a2, q)["routine_fte"] / b["routine_fte"] - 0.5)))
    q = _clone(p); q.hours_per_reporting_unit_week *= 2
    checks.append(("routine_fte ~ hours_per_unit_week",
                   abs(M.demand_measures(pop, a2, q)["routine_fte"] / b["routine_fte"] - 2.0)))
    checks.append(("target ~ population",
                   abs(M.demand_measures(2 * pop, a2, p)["jee_target_heads"] / b["jee_target_heads"] - 2.0)))
    checks.append(("outbreak_fte ~ population",
                   abs(M.demand_measures(2 * pop, a2, p)["outbreak_fte"] / b["outbreak_fte"] - 2.0)))
    q = _clone(p); q.deployment_days *= 3
    checks.append(("outbreak_fte ~ deployment_days",
                   abs(M.demand_measures(pop, a2, q)["outbreak_fte"] / b["outbreak_fte"] - 3.0)))
    bad = [n for n, e in checks if e > 1e-9]
    return len(bad) == 0, f"{len(checks)} scaling relations; failing: {bad or 'none'}"


STRUCTURAL_TESTS: List[Tuple[str, Callable]] = [
    ("Mass conservation incl. pool and active progression (tol 1e-9)", test_conservation),
    ("Promotion while unabsorbed does not double-count heads", test_progression_in_pool),
    ("Inflow accounting: completions = promotions + public + non-public", test_inflow_accounting),
    ("Analytic: single cohort decay = N0 exp(-hT)", test_analytic_decay),
    ("Analytic: unabsorbed pool decay = N0 exp(-hT)", test_pool_analytic_decay),
    ("Analytic: constant intake equilibrium = discrete-exact I c s dt/(1-e^-h dt)", test_analytic_equilibrium),
    ("Analytic: retirement timing exact and independent of dt", test_ageing_exact_and_dt_free),
    ("Extreme condition: no inflow, no outflow -> constant stock", test_extreme_no_inflow_no_outflow),
    ("Extreme condition: 100%/yr attrition -> empties, stays non-negative", test_extreme_total_attrition),
    ("Extreme condition: population -> 0 -> demand -> 0", test_extreme_zero_population),
    ("Extreme condition: unlimited posts -> no pool, deployed == trained", test_extreme_infinite_posts),
    ("R3 acts on the unabsorbed pool only", test_pool_identity),
    ("Step-size convergence: dt = 1/12 and 1/24 vs 1/48", test_integration_error),
    ("Internal consistency: loops off == sensitivities zero (not a v1 regression)",
     test_switches_equal_zero_sensitivity),
    ("Monotonicity: trained stock non-decreasing in intake", test_monotone_in_intake),
    ("Demand block: numerical unit-scaling relations", test_demand_scaling),
    ("Cohort-history start: mortality and retirement reduce headcount", test_cohort_history_losses),
    ("Fixed posts: deployed never exceeds funded posts (incl. promotions)", test_posts_not_exceeded),
    ("R1: shared mentors are not double-counted", test_mentor_capacity_shared),
    ("Scenarios start from the same enrolled trainee pipeline", test_scenario_pipeline),
    ("Scenarios start from the same workforce (cohort history)", test_scenario_common_start),
    ("Unabsorbed pool ages, retires, and keeps ages when hired", test_pool_ages_and_retires),
    ("R1: calibrated external mentors sustain the baseline intake", test_r1_baseline_feasible),
    ("R1: supervision demand counts enrolled trainees", test_r1_uses_enrolment),
    ("Flow timing: flows recorded at the end of their interval", test_flow_timing),
    ("Input validation rejects invalid parameters and cohorts", test_input_validation),
    ("Overlap check flags assumptions above recorded counts", test_overlap_report),
]


# ---------------------------------------------------------------------------
# Dimensional analysis: a small unit algebra, not a label check.
# ---------------------------------------------------------------------------

class Dim:
    """Exponent vector over base dimensions. Addition requires equal dims."""
    def __init__(self, **e):
        self.e = {k: v for k, v in e.items() if v}
    def __mul__(self, o):
        r = dict(self.e)
        for k, v in o.e.items():
            r[k] = r.get(k, 0) + v
        return Dim(**r)
    def __truediv__(self, o):
        return self * Dim(**{k: -v for k, v in o.e.items()})
    def __add__(self, o):
        if self.e != o.e:
            raise TypeError(f"cannot add {self} and {o}")
        return Dim(**self.e)
    __sub__ = __add__
    def __eq__(self, o):
        return self.e == o.e
    def __repr__(self):
        return "*".join(f"{k}^{v}" for k, v in sorted(self.e.items())) or "1"


def _u(**e): return Dim(**e)


def test_dimensional_consistency() -> Tuple[bool, str]:
    """Verify the algebra of the demand and feedback blocks with a real unit
    system. Also proves the checker can fail by adding a rate to a stock."""
    person, head, hour, yr = _u(person=1), _u(head=1), _u(hour=1), _u(yr=1)
    week, unit, event, day, epi = _u(week=1), _u(unit=1), _u(event=1), _u(day=1), _u(epi=1)
    fte, mperson = _u(fte=1), _u(mperson=1)
    one = _u()
    pop = person
    ppfe = person / head
    admin2, unit_per_a2 = _u(admin2=1), unit / _u(admin2=1)
    hr_per_unit_week = hour / unit / week
    weeks_per_yr = week / yr
    prod_hours = hour / yr / fte
    out_per_m = event / yr / mperson
    mp_per_p = mperson / person
    team, dep_days, fh_per_day = epi / event, day, hour / day
    checks = [
        ("jee_target_heads = pop / people_per_fe", pop / ppfe, head),
        ("routine_hours = admin2 x unit/admin2 x week/yr x hour/unit/week",
         admin2 * unit_per_a2 * weeks_per_yr * hr_per_unit_week, hour / yr),
        ("routine_fte = routine_hours / productive_hours",
         (hour / yr) / prod_hours, fte),
        ("outbreak_hours = (pop x Mperson/person) x event/yr/Mperson x epi/event x day/event x hour/day",
         pop * mp_per_p * out_per_m * team * dep_days * fh_per_day, epi * hour / yr),
        ("training_intake_cap = (mentors x resident/mentor) / yr",
         _u(mentor=1) * (_u(resident=1) / _u(mentor=1)) / yr, _u(resident=1) / yr),
        ("post_creation = (head - head) / yr", (head - head) / yr, head / yr),
        ("burnout_multiplier = 1 + sens x excess", one + (one * one), one),
    ]
    bad = [n for n, got, want in checks if not (got == want)]
    try:
        _ = head + head / yr
        checker_can_fail = False
    except TypeError:
        checker_can_fail = True
    ok = (len(bad) == 0) and checker_can_fail
    return ok, (f"{len(checks)} expressions checked by unit algebra; mismatches: {bad or 'none'}; "
                f"rejects stock + rate: {checker_can_fail}. Note: outbreak_hours carries an "
                "epidemiologist factor (epi x hour), reconciled to FTE by the productive-hours "
                "denominator, as in the code.")


def run_structural_tests(row: pd.Series, p: Optional[Params] = None,
                         horizon: int = 25,
                         cohort_history: Optional[pd.DataFrame] = None,
                         dt: float = M.DT_DEFAULT) -> pd.DataFrame:
    """Run structural checks using the active parameters and starting cohort.

    The conservation and inflow checks also use the app's horizon, integration
    step, and optional cohort history. Analytic tests deliberately construct
    their own controlled inputs and step sizes.
    """
    p = prepare_params(row, p)
    recs = []
    for name, fn in STRUCTURAL_TESTS:
        try:
            if fn in (test_conservation, test_inflow_accounting):
                ok, detail = fn(row, p, horizon, cohort_history, dt)
            else:
                ok, detail = fn(row, p)
        except Exception as exc:                      # a crash is a failure
            ok, detail = False, f"EXCEPTION: {type(exc).__name__}: {exc}"
        recs.append({"test": name, "passed": bool(ok), "detail": detail})
    try:
        ok, detail = test_dimensional_consistency()
    except Exception as exc:
        ok, detail = False, f"EXCEPTION: {type(exc).__name__}: {exc}"
    recs.append({"test": "Dimensional consistency (unit algebra)", "passed": bool(ok),
                 "detail": detail})
    return pd.DataFrame(recs)


# ===========================================================================
# Uncertainty: Latin hypercube + PRCC
# ===========================================================================

# name -> (low, high, kind). 'kind' is only used for reporting.
# Every range is an ANALYST-CHOSEN UNIFORM INTERVAL with no empirical
# distribution behind it (see PSA_BASIS). Treat the output as conditional
# sensitivity analysis.
PSA_RANGES: Dict[str, Tuple[float, float, str]] = {
    "survival_advanced":      (0.50, 0.85, "share"),
    "survival_intermediate":  (0.55, 0.90, "share"),
    "public_share_advanced":  (0.55, 0.95, "share"),
    "attrition_advanced":     (0.015, 0.070, "rate/yr"),
    "emigration_advanced":    (0.010, 0.080, "rate/yr"),
    "retirement_age":         (55.0, 65.0, "years"),
    "overlap_adv_from_int":   (0.10, 0.60, "share"),
    "completion_advanced":    (0.65, 0.95, "share"),
    "protected_time_advanced": (0.40, 0.85, "share"),
    "skill_weight_intermediate": (0.40, 0.80, "weight"),
    # people_per_fe is deliberately NOT sampled: it defines the benchmark being
    # tested, so it stays at the active (sidebar) value. Benchmark alternatives
    # belong in separate scenarios, not in the parameter-uncertainty draws.
    "annual_outbreaks_per_million": (0.2, 2.0, "event/yr/M"),
    "mentor_share":           (0.15, 0.60, "share"),
    "residents_per_mentor":   (1.0, 4.0, "resident/mentor"),
    "burnout_sensitivity":    (0.0, 0.80, "1/excess load"),
    "emigration_sensitivity": (0.0, 3.0, "1/unabsorbed share"),
    "political_ambition":     (0.20, 1.00, "share"),
    "post_adjust_years":      (2.0, 15.0, "years"),
    "intake_multiplier":      (0.5, 2.0, "multiplier"),
}


PSA_BASIS = {nm: "analyst range; no empirical distribution" for nm in PSA_RANGES}


def _apply_sample(p: Params, s: Dict[str, float], base_intake: Dict[str, float]) -> Params:
    q = copy.deepcopy(p)

    q.survival_to_2026["advanced"] = s["survival_advanced"]
    q.survival_to_2026["intermediate"] = s["survival_intermediate"]
    q.public_sector_share["advanced"] = s["public_share_advanced"]
    q.public_entry_share["advanced"] = s["public_share_advanced"]
    q.attrition["advanced"] = s["attrition_advanced"]
    q.emigration["advanced"] = s["emigration_advanced"]
    q.retirement_age = s["retirement_age"]
    q.overlap_adv_from_int = s["overlap_adv_from_int"]
    q.completion["advanced"] = s["completion_advanced"]
    q.protected_time["advanced"] = s["protected_time_advanced"]
    q.skill_weight["intermediate"] = s["skill_weight_intermediate"]
    q.annual_outbreaks_per_million = s["annual_outbreaks_per_million"]
    q.mentor_share = s["mentor_share"]
    q.residents_per_mentor = s["residents_per_mentor"]
    q.burnout_sensitivity = s["burnout_sensitivity"]
    q.emigration_sensitivity = s["emigration_sensitivity"]
    q.political_ambition = s["political_ambition"]
    q.post_adjust_years = s["post_adjust_years"]
    intake_multiplier = float(s.get("intake_multiplier", 1.0))
    q.annual_intake = {t: base_intake[t] * intake_multiplier for t in TIERS}
    return q


def latin_hypercube(n: int, names: List[str], seed: int = 20261113) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = len(names)
    cut = np.arange(n + 1) / n
    out = np.empty((n, k))
    for j in range(k):
        u = rng.uniform(cut[:n], cut[1:])
        out[:, j] = rng.permutation(u)
    cols = {}
    for j, nm in enumerate(names):
        lo, hi, _ = PSA_RANGES[nm]
        cols[nm] = lo + out[:, j] * (hi - lo)
    return pd.DataFrame(cols)


def psa(row: pd.Series, p: Optional[Params] = None, n: int = 500, horizon: int = 25,
        seed: int = 20261113, feedback: Optional[bool] = None,
        cohort_history: Optional[pd.DataFrame] = None,
        dt: float = M.DT_DEFAULT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Latin hypercube CONDITIONAL SENSITIVITY ANALYSIS over PSA_RANGES.

    Returns (samples_with_outputs, trajectory_percentiles). The percentiles are
    over uniform analyst-chosen ranges; they are not credible intervals.
    feedback=None keeps the active loop switches; True/False forces all four.
    """
    if p is None:
        p = prepare_params(row, None)
    p = copy.deepcopy(p)
    if feedback is not None:
        p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = feedback
    # intake multiplier is applied to the ACTIVE intake, not a rebuilt default
    base_intake = dict(p.annual_intake)

    names = list(PSA_RANGES)
    X = latin_hypercube(n, names, seed)
    traj = []
    outs = []
    for i in range(n):
        q = _apply_sample(p, X.iloc[i].to_dict(), base_intake)
        try:
            df = M.run(row, q, horizon=horizon,
                       cohort_history=cohort_history, dt=dt)
        except Exception:
            outs.append({k: np.nan for k in
                         ("heads_final", "deployed_final", "gap_final", "ratio_final",
                          "years_to_target", "cum_emigration")})
            traj.append(np.full(int(round(horizon / dt)) + 1, np.nan))
            continue
        f = df.iloc[-1]
        met = df.index[df.jee_ratio >= 1.0]
        outs.append({
            "heads_final": f.compliance_heads,
            "deployed_final": f.deployed_heads,
            "gap_final": f.jee_gap_heads,
            "ratio_final": f.jee_ratio,
            "years_to_target": (df.loc[met[0], "time_years"] if len(met) else np.nan),
            "cum_emigration": f.cum_exit_emigration,
        })
        traj.append(df.compliance_heads.values)
    S = pd.concat([X, pd.DataFrame(outs)], axis=1)
    T = np.vstack([t for t in traj])
    years = M.BASE_YEAR + np.arange(T.shape[1]) * dt
    Q = pd.DataFrame({
        "calendar_year": years,
        "p2_5": np.nanpercentile(T, 2.5, axis=0),
        "p25": np.nanpercentile(T, 25, axis=0),
        "median": np.nanpercentile(T, 50, axis=0),
        "p75": np.nanpercentile(T, 75, axis=0),
        "p97_5": np.nanpercentile(T, 97.5, axis=0),
    })
    return S, Q


def required_intake_by_year(
    row: pd.Series,
    p: Optional[Params] = None,
    n: int = 50,
    horizon: int = 25,
    target_ratio: float = 1.0,
    outcome: str = "deployed_ratio",
    seed: int = 20261113,
    cohort_history: Optional[pd.DataFrame] = None,
    dt: float = M.DT_DEFAULT,
    max_multiplier: float = 32.0,
    refine_points: int = 3,
    refine_rounds: int = 2,
) -> pd.DataFrame:
    """Estimate annual intake run-rates needed by each target year, verified
    by simulation.

    For each Latin-hypercube parameter draw, simulate a grid of uniform
    multipliers on the active tier mix (0, 0.5, 1, 2, ... max). For each
    target year, the first simulated multiplier that reaches the target
    brackets the crossing; each bracket is then refined with `refine_points`
    additional simulations, repeated `refine_rounds` times (each round
    narrows the bracket by a factor of refine_points + 1). The reported intake for a draw is the smallest
    SIMULATED multiplier that reaches the target, so every reported value has
    been run and verified (it is conservative by at most the final bracket
    width, reported as resolution). Draws whose simulations fail are counted
    separately from draws that cannot reach the target within the search.

    Initial conditions are common within a draw: the base-year workforce is
    reconstructed once with that draw's assumptions, and trainees already
    enrolled stay at the active intake while the candidate intake applies
    prospectively. Results are conditional on the analyst-selected ranges and
    the active tier mix; they are not calibrated probabilities or an
    optimized year-varying schedule.
    """
    if outcome not in ("deployed_ratio", "jee_ratio"):
        raise ValueError("outcome must be deployed_ratio or jee_ratio")
    if not 0.0 < target_ratio <= 2.0:
        raise ValueError("target_ratio must be greater than 0 and no greater than 2")
    if n < 2:
        raise ValueError("n must be at least 2")
    if max_multiplier <= 0:
        raise ValueError("max_multiplier must be positive")
    if not np.isfinite(dt) or dt <= 0 or dt > 1:
        raise ValueError("dt must be finite, positive, and no greater than one year")
    if not np.isclose(1.0 / dt, round(1.0 / dt), atol=1e-8):
        raise ValueError("annual target table requires a step size that divides one year")

    if p is None:
        p = prepare_params(row, None)
    p = copy.deepcopy(p)
    base_intake = {t: max(0.0, float(p.annual_intake.get(t, 0.0))) for t in TIERS}
    base_total = float(sum(base_intake.values()))
    if base_total <= 0.0:
        raise ValueError("Set a positive annual intake mix before estimating required intake")

    years = np.arange(M.BASE_YEAR + 1, M.BASE_YEAR + int(horizon) + 1, dtype=int)
    idx = [int(round((year - M.BASE_YEAR) / dt)) for year in years]
    names = [nm for nm in PSA_RANGES if nm != "intake_multiplier"]
    X = latin_hypercube(n, names, seed)
    grid = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    multipliers = sorted({m for m in grid if m <= max_multiplier} | {float(max_multiplier)})
    ny = len(years)

    req = np.full((n, ny), np.nan)        # verified multiplier
    res = np.full((n, ny), np.nan)        # final bracket width (multiplier units)
    nonmono = np.zeros((n, ny), dtype=bool)
    failed = np.zeros(n, dtype=bool)

    for d in range(n):
        q = _apply_sample(p, X.iloc[d].to_dict(), base_intake)
        try:
            stock0 = M.initialize_stock(row, q, cohort_history)
        except Exception:
            failed[d] = True
            continue
        sims: Dict[float, np.ndarray] = {}

        def simulate(m: float) -> None:
            if m in sims:
                return
            cand = copy.deepcopy(q)
            cand.annual_intake = {t: base_intake[t] * m for t in TIERS}
            df = M.run(row, cand, horizon=int(horizon), dt=dt,
                       initial_pipeline_intake=base_intake, initial_stock=stock0)
            sims[m] = df.iloc[idx][outcome].to_numpy(dtype=float)

        def brackets():
            ms = sorted(sims)
            vals = np.vstack([sims[m] for m in ms])
            out = []
            for y in range(ny):
                hit = np.flatnonzero(vals[:, y] >= target_ratio)
                if len(hit) == 0:
                    out.append(None)
                elif hit[0] == 0:
                    out.append((ms[0], ms[0]))
                else:
                    out.append((ms[hit[0] - 1], ms[hit[0]]))
            return out, ms, vals

        try:
            for m in multipliers:
                simulate(m)
            for _ in range(max(0, refine_rounds)):
                br, _, _ = brackets()
                for lo_hi in {b for b in br if b is not None and b[1] > b[0]}:
                    lo, hi = lo_hi
                    for k in range(1, refine_points + 1):
                        simulate(lo + (hi - lo) * k / (refine_points + 1))
            br, ms, vals = brackets()
        except Exception:
            failed[d] = True
            continue
        for y in range(ny):
            nonmono[d, y] = bool(np.any(np.diff(vals[:, y]) < -1e-6))
            if br[y] is not None:
                req[d, y] = br[y][1]
                res[d, y] = br[y][1] - br[y][0]

    ok_draws = ~failed
    n_ok = int(ok_draws.sum())

    def quantile(values: np.ndarray, qq: float) -> float:
        return float(np.quantile(values, qq)) if len(values) else np.nan

    records = []
    for y, year in enumerate(years):
        feasible = ok_draws & np.isfinite(req[:, y])
        fm = req[feasible, y]
        total_values = fm * base_total
        tier_values = {t: fm * base_intake[t] for t in TIERS}
        records.append({
            "target_year": int(year),
            "target_ratio": float(target_ratio),
            "outcome": outcome,
            "total_intake_p05": quantile(total_values, 0.05),
            "total_intake_p50": quantile(total_values, 0.50),
            "total_intake_p95": quantile(total_values, 0.95),
            "advanced_intake_p50": quantile(tier_values["advanced"], 0.50),
            "intermediate_intake_p50": quantile(tier_values["intermediate"], 0.50),
            "frontline_intake_p50": quantile(tier_values["frontline"], 0.50),
            "resolution_p50": quantile(res[feasible, y] * base_total, 0.50),
            "share_sampled_sets_feasible": (float(feasible.sum()) / n_ok if n_ok else np.nan),
            "share_sampled_sets_nonmonotone": (float(nonmono[ok_draws, y].mean()) if n_ok else np.nan),
            "feasible_draws": int(feasible.sum()),
            "failed_draws": int(failed.sum()),
            "sampled_draws": int(n),
            "max_multiplier_searched": float(max_multiplier),
        })
    return pd.DataFrame(records)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    from math import lgamma, log, exp
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, dof: float) -> float:
    """Two-sided p-value of Student's t with `dof` degrees of freedom."""
    if not np.isfinite(t) or dof <= 0:
        return np.nan
    return float(_betainc(dof / 2.0, 0.5, dof / (dof + t * t)))


def prcc(samples: pd.DataFrame, outcome: str = "ratio_final") -> pd.DataFrame:
    """Partial rank correlation coefficients with two-sided p-values.

    PRCC is the right statistic here because the model is monotone but
    strongly non-linear in most inputs, and the inputs are sampled
    independently (Marino et al. 2008). p-values use Student's t with
    n - k - 1 degrees of freedom; inputs that do not vary are dropped and a
    constant outcome returns NaN with a note.
    """
    names = [c for c in PSA_RANGES if c in samples.columns]
    d = samples[names + [outcome]].dropna()
    n = len(d)
    varying = [nm for nm in names if d[nm].nunique() > 1]
    if n < 3 or d[outcome].nunique() <= 1 or not varying:
        # A constant outcome (or no varying input) has no rank correlation.
        why = ("outcome is constant across runs" if d[outcome].nunique() <= 1
               else "too few runs or no varying inputs")
        return pd.DataFrame([{"parameter": nm, "prcc": np.nan, "p_value": np.nan,
                              "units": PSA_RANGES[nm][2], "range_basis": PSA_BASIS[nm],
                              "note": why} for nm in names])
    names = varying
    R = d.rank().to_numpy(dtype=float)
    R = (R - R.mean(axis=0)) / R.std(axis=0, ddof=1)
    C = np.corrcoef(R, rowvar=False)
    try:
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(C)
    k = len(names)
    recs = []
    for j, nm in enumerate(names):
        r = -P[j, k] / np.sqrt(P[j, j] * P[k, k])
        dof = n - k - 1
        if dof > 0 and abs(r) < 1:
            tstat = r * np.sqrt(dof / (1 - r ** 2))
            pval = student_t_two_sided_p(tstat, dof)   # exact Student's t
        else:
            pval = np.nan
        recs.append({"parameter": nm, "prcc": r, "p_value": pval,
                     "units": PSA_RANGES[nm][2], "range_basis": PSA_BASIS[nm], "note": ""})
    out = pd.DataFrame(recs)
    out["abs"] = out.prcc.abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def tornado(row: pd.Series, p: Optional[Params] = None, horizon: int = 25,
            outcome: str = "ratio_final") -> pd.DataFrame:
    """One-at-a-time low/high swings around the range midpoints, for the
    tornado plot. Loop switches and unsampled settings are the active ones."""
    if p is None:
        p = prepare_params(row, None)
    p = copy.deepcopy(p)
    base_intake = dict(p.annual_intake)
    mid = {nm: (lo + hi) / 2 for nm, (lo, hi, _) in PSA_RANGES.items()}

    def outcome_of(sample):
        q = _apply_sample(p, sample, base_intake)
        f = M.run(row, q, horizon=horizon).iloc[-1]
        return {"ratio_final": f.jee_ratio, "heads_final": f.compliance_heads,
                "deployed_final": f.deployed_heads}[outcome]

    base = outcome_of(mid)
    recs = []
    for nm, (lo, hi, unit) in PSA_RANGES.items():
        s_lo, s_hi = dict(mid), dict(mid)
        s_lo[nm], s_hi[nm] = lo, hi
        v_lo, v_hi = outcome_of(s_lo), outcome_of(s_hi)
        recs.append({"parameter": nm, "units": unit, "low_value": lo, "high_value": hi,
                     "outcome_low": v_lo, "outcome_high": v_hi,
                     "swing": abs(v_hi - v_lo)})
    return (pd.DataFrame(recs).assign(baseline=base)
            .sort_values("swing", ascending=False).reset_index(drop=True))


def sensitivity_summary(S: pd.DataFrame) -> Dict[str, float]:
    """Labelled summary of a sensitivity run. Names say what these are."""
    good = S.dropna(subset=["ratio_final"])
    return {
        "n_runs": int(len(good)),
        "median_ratio_final": float(good.ratio_final.median()),
        "sampled_p2_5": float(good.ratio_final.quantile(.025)),
        "sampled_p97_5": float(good.ratio_final.quantile(.975)),
        "share_of_runs_meeting_target": float((good.ratio_final >= 1).mean()),
    }
