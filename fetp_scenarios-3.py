"""Named conditional policy scenarios for the African FETP model.

Scenarios inherit the active application parameters and change only the
assumptions named in their descriptions. They are exploratory comparisons,
not causal estimates or probability-weighted forecasts.

Every scenario starts from the same initial conditions: one reconstruction of
the base-year workforce under the active (historical) assumptions, and the
trainee pipeline at the active intake. A change in intake therefore affects
completions only after the relevant programme duration has elapsed, and a
change in exit rates applies only from the base year onward.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import fetp_model as M
from fetp_model import Params, TIERS


SCENARIO_DOC = {
    "status_quo": "Active intake and model settings continue.",
    "donor_cliff": "To represent an immediate donor-cliff scenario, new annual intake is set to 30% of the active value from the base year; trainees already enrolled still complete. Financing itself is not modeled.",
    "train_harder": "New annual intake doubles from the base year; trainees already enrolled are unchanged, so extra graduates appear only after each programme's duration.",
    "absorb_first": "Intake is unchanged and B1 (post response) is switched on: the funded-post ceiling is 100% of the benchmark, adjustment time is 3 years, and annual post growth is capped at 80. It has no open-loop counterpart, so no feedback effect is reported for it.",
    "domestic_transition": "Intake is unchanged; the funded-post ceiling is 90% of the benchmark and advanced-tier attrition and emigration are halved.",
}

# Scenarios whose definition requires a feedback loop, so an open-loop run
# cannot represent them.
REQUIRES_FEEDBACK = {"absorb_first"}


def _base(row: pd.Series, base_params: Optional[Params] = None,
          feedback: bool = True) -> Params:
    """Base parameters for a scenario run.

    feedback=True keeps the active loop switches from `base_params` (or turns
    all four on when no active parameters are given); feedback=False turns all
    four off for the open-loop comparison.
    """
    if base_params is None:
        p = M.default_params_for(row)
        p.annual_intake = M.implied_historical_intake(row, p)
        if feedback:
            p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = True
    else:
        p = copy.deepcopy(base_params)
    if not feedback:
        p.mentor_feedback = p.burnout_feedback = p.emigration_feedback = p.post_feedback = False
    return p


def make_scenarios(row: pd.Series, feedback: bool = True,
                   base_params: Optional[Params] = None) -> Dict[str, Params]:
    base = _base(row, base_params, feedback)
    out = {"status_quo": copy.deepcopy(base)}

    q = copy.deepcopy(base)
    q.annual_intake = {t: v * 0.30 for t, v in base.annual_intake.items()}
    out["donor_cliff"] = q

    q = copy.deepcopy(base)
    q.annual_intake = {t: v * 2.0 for t, v in base.annual_intake.items()}
    out["train_harder"] = q

    q = copy.deepcopy(base)
    if feedback:
        q.post_feedback = True          # the scenario is defined with B1 active
    q.political_ambition = 1.00
    q.post_adjust_years = 3.0
    q.max_post_growth_per_year = 80.0
    out["absorb_first"] = q

    q = copy.deepcopy(base)
    q.political_ambition = 0.90
    q.attrition["advanced"] *= 0.5
    q.emigration["advanced"] *= 0.5
    out["domestic_transition"] = q
    return out


def run_scenarios(row: pd.Series, horizon: int = 25, feedback: bool = True,
                  base_params: Optional[Params] = None,
                  cohort_history: Optional[pd.DataFrame] = None,
                  dt: float = M.DT_DEFAULT) -> Dict[str, pd.DataFrame]:
    scen = make_scenarios(row, feedback, base_params)
    base = scen["status_quo"]
    # Common initial conditions: the starting workforce is reconstructed once
    # with the active (historical) assumptions, and every scenario inherits
    # it together with the trainees already enrolled at the active intake.
    # Prospective changes such as halved attrition therefore cannot alter the
    # historical starting point.
    stock0 = M.initialize_stock(row, base, cohort_history)
    pipeline = dict(base.annual_intake)
    return {
        name: M.run(row, p, horizon=horizon, dt=dt,
                    initial_pipeline_intake=pipeline, initial_stock=stock0)
        for name, p in scen.items()
    }


def _final_record(name: str, fb: bool, df: pd.DataFrame) -> dict:
    f = df.iloc[-1]
    return {
        "scenario": name, "feedback": fb,
        "trained_heads": f.compliance_heads,
        "deployed_heads": f.deployed_heads,
        "unabsorbed": f.unabsorbed_heads,
        "posts": f.established_posts,
        "target": f.jee_target_heads,
        "ratio_deployed": f.deployed_heads / max(f.jee_target_heads, 1e-9),
        "cum_emigration": f.cum_exit_emigration,
        "cum_retirement": f.cum_exit_retirement,
        "binding": f.binding_constraint,
    }


def scenario_results(row: pd.Series, horizon: int = 25,
                     base_params: Optional[Params] = None,
                     cohort_history: Optional[pd.DataFrame] = None,
                     dt: float = M.DT_DEFAULT
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[bool, Dict[str, pd.DataFrame]]]:
    """Run every scenario once open-loop and once with the active loops.

    Returns (long table, open-loop vs active-loop comparison, trajectories),
    with trajectories keyed by feedback flag then scenario name.
    """
    curves = {fb: run_scenarios(row, horizon, fb, base_params, cohort_history, dt)
              for fb in (False, True)}
    long = pd.DataFrame([_final_record(name, fb, df)
                         for fb in (False, True) for name, df in curves[fb].items()])
    wide = long.pivot(index="scenario", columns="feedback", values="deployed_heads")
    wide.columns = ["deployed_open_loop", "deployed_active_loops"]
    wide["feedback_effect_pct"] = 100 * (
        wide.deployed_active_loops / wide.deployed_open_loop.clip(lower=1e-9) - 1
    )
    wide["note"] = ""
    for nm in REQUIRES_FEEDBACK & set(wide.index):
        wide.loc[nm, "feedback_effect_pct"] = np.nan
        wide.loc[nm, "note"] = ("defined by B1; the open-loop value equals status_quo, "
                                "so no feedback effect is reported")
    return long, wide.reset_index(), curves


def scenario_table(row: pd.Series, horizon: int = 25,
                   base_params: Optional[Params] = None,
                   cohort_history: Optional[pd.DataFrame] = None,
                   dt: float = M.DT_DEFAULT
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed outcomes and the open-loop/active-loop comparison."""
    long, wide, _ = scenario_results(row, horizon, base_params, cohort_history, dt)
    return long, wide


def decompose_decline(row: pd.Series, horizon: int = 25,
                      feedback: bool = True,
                      base_params: Optional[Params] = None,
                      cohort_history: Optional[pd.DataFrame] = None,
                      dt: float = M.DT_DEFAULT) -> pd.DataFrame:
    p = _base(row, base_params, feedback)
    df = M.run(row, p, horizon=horizon, cohort_history=cohort_history, dt=dt)
    f = df.iloc[-1]
    total = sum(f[f"cum_exit_{c}"] for c in
                ("attrition", "emigration", "mortality", "retirement"))
    return pd.DataFrame([
        {"channel": c, "people": float(f[f"cum_exit_{c}"]),
         "share_pct": 100 * float(f[f"cum_exit_{c}"]) / max(total, 1e-9),
         "addressable": c in ("attrition", "emigration")}
        for c in ("retirement", "emigration", "attrition", "mortality")
    ])


def policy_resistance_curve(row: pd.Series, horizon: int = 25,
                            multipliers=(0.5, 1, 2, 3, 4, 6, 8),
                            feedback: bool = True,
                            base_params: Optional[Params] = None,
                            cohort_history: Optional[pd.DataFrame] = None,
                            dt: float = M.DT_DEFAULT) -> pd.DataFrame:
    """Final-year outputs across intake multipliers, conditional on inputs.

    The trainee pipeline at the base year is held at the active intake; the
    multiplier applies to new enrolment only.
    """
    base = _base(row, base_params, feedback)
    initial_intake = dict(base.annual_intake)
    stock0 = M.initialize_stock(row, base, cohort_history)
    recs = []
    for mult in multipliers:
        p = copy.deepcopy(base)
        p.annual_intake = {t: initial_intake[t] * mult for t in TIERS}
        f = M.run(row, p, horizon=horizon, dt=dt,
                  initial_pipeline_intake=initial_intake, initial_stock=stock0).iloc[-1]
        recs.append({
            "intake_multiplier": mult,
            "trained_heads": f.compliance_heads,
            "deployed_heads": f.deployed_heads,
            "unabsorbed": f.unabsorbed_heads,
            "cum_emigration": f.cum_exit_emigration,
            "cost_of_training_relative": mult,
        })
    return pd.DataFrame(recs)
