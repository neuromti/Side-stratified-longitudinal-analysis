"""
Supplementary analyses in the matched cohort: study withdrawal, dopaminergic
dose, dose-adjusted motor models, acute levodopa response, and DAT decline
conditioned on baseline binding. Shared rules (visits, censoring, model
fitting) come from 02_analyse. group: 0 left-, 1 right-sided onset.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

A = import_module("02_analyse")

# Matching covariates other than sex, z-scored in the Cox models.
COX_COVARIATES = ["AGE_AT_VISIT", "disease_duration", "NP1RTOT", "NP2PTOT",
                  "UPDRS3_nat", "MCATOT"]
WITHDRAWAL = {"Withdrew", "Baseline Withdraw", "Lost to follow-up"}
DEATH = {"Withdraw Deceased"}


# ─────────────────────────────────────────────────────────────────────────────
# Study withdrawal
# ─────────────────────────────────────────────────────────────────────────────

def withdrawal(matched: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """
    Cox proportional-hazards model of time to withdrawal on group, adjusted for
    the six z-scored baseline covariates. Time = months from enrolment to the
    last status date. Two event definitions: voluntary withdrawal (withdrawal
    or loss to follow-up), and withdrawal or death.

    status: PATNO, ENROLL_STATUS, months_in_study.
    """
    d = matched.merge(status, on="PATNO").dropna(subset=COX_COVARIATES + ["months_in_study"])
    d = d[d["months_in_study"] > 0].copy()
    for c in COX_COVARIATES:
        d[c] = (d[c] - d[c].mean()) / d[c].std(ddof=1)

    rows = []
    for label, events in [("withdrawal", WITHDRAWAL), ("withdrawal or death", WITHDRAWAL | DEATH)]:
        d["event"] = d["ENROLL_STATUS"].isin(events).astype(int)
        cph = CoxPHFitter().fit(d[["group", "months_in_study", "event"] + COX_COVARIATES],
                                duration_col="months_in_study", event_col="event")
        s = cph.summary.loc["group"]
        rows.append({"event": label, "hr": s["exp(coef)"],
                     "hr_ci": (s["exp(coef) lower 95%"], s["exp(coef) upper 95%"]),
                     "p": s["p"], "n": len(d), "events": int(d["event"].sum())})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Dopaminergic dose
# ─────────────────────────────────────────────────────────────────────────────
#
# Levodopa-equivalent daily dose (LEDD) per visit = sum over all medications
# active in the calendar month of the visit, using PPMI's conversion factors.
# COMT inhibitors and istradefylline are computed from the concurrent levodopa
# dose (PPMI records them as an unresolved "LD x factor"); safinamide,
# zonisamide, trihexyphenidyl and benztropine take the protocol's fixed values
# (150, 100, 100, 0 mg). A visit that cannot be fully resolved is missing.

def dose_trajectory(ledd: pd.DataFrame, escalation_month: dict) -> dict:
    """LEDD ~ group * months + (1 + months | PATNO). ledd: PATNO, EVENT_ID, group, value."""
    d = A.prepare(ledd, escalation_month)
    r, re = A.fit_best(d)
    return A.summarise(r, d, re)


def dose_adjusted(outcomes: dict[str, pd.DataFrame], ledd: pd.DataFrame,
                  escalation_month: dict) -> pd.DataFrame:
    """
    Motor models before and after adding concurrent LEDD, centred at its grand
    mean over all visits, as a covariate:

        value ~ group * months              + (1 + months | PATNO)
        value ~ group * months + LEDD_c     + (1 + months | PATNO)

    The adjusted model uses visits with a recorded dose.
    """
    dose = ledd[["PATNO", "EVENT_ID", "value"]].rename(columns={"value": "LEDD"})
    grand_mean = dose["LEDD"].mean()
    rows = []
    for name, long in outcomes.items():
        d = A.prepare(long, escalation_month)
        r0, _ = A.fit_best(d)
        dl = d.merge(dose, on=["PATNO", "EVENT_ID"], how="left").dropna(subset=["LEDD"])
        dl["LEDD_c"] = dl["LEDD"] - grand_mean
        r1, _ = A.fit_best(dl, "value ~ group * months + LEDD_c")
        b0, b1 = r0.params["group:months"], r1.params["group:months"]
        rows.append({"outcome": name,
                     "unadjusted_per_year": 12 * b0, "p_unadjusted": r0.pvalues["group:months"],
                     "adjusted_per_year": 12 * b1, "p_adjusted": r1.pvalues["group:months"],
                     "change_pct": 100 * (b1 - b0) / abs(b0)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Acute levodopa response
# ─────────────────────────────────────────────────────────────────────────────

def levodopa_response(u3: pd.DataFrame, group: pd.DataFrame,
                      escalation_month: dict) -> pd.DataFrame:
    """
    Response = OFF minus ON MDS-UPDRS Part III at the same visit, for the total
    score and the four subscores. OFF: untreated examinations without a state
    flag or examinations flagged OFF; ON: flagged ON; several examinations in
    one state at one visit are averaged before pairing.

        response ~ group + (1 | PATNO)

    A pooled group difference in response (no time term); every paired visit
    is a repeated measure. Same censoring and per-visit MIN_N rule as the other
    models. q = BH across the five outcomes.

    u3: MDS-UPDRS Part III examinations with subscores already added
        (02_analyse.subscores); group: PATNO, group.
    """
    scores = ["NP3TOT"] + list(A.SUBSCORE_ITEMS)
    off = u3[(u3["PDSTATE"].isna() & (u3["PDTRTMNT"] == 0)) | (u3["PDSTATE"] == "OFF")]
    on = u3[u3["PDSTATE"] == "ON"]
    off = off.groupby(["PATNO", "EVENT_ID"], as_index=False)[scores].mean()
    on = on.groupby(["PATNO", "EVENT_ID"], as_index=False)[scores].mean()
    paired = off.merge(on, on=["PATNO", "EVENT_ID"], suffixes=("_off", "_on")).merge(group, on="PATNO")

    rows = []
    for s in scores:
        long = paired[["PATNO", "EVENT_ID", "group"]].assign(value=paired[f"{s}_off"] - paired[f"{s}_on"])
        d = A.prepare(long, escalation_month)
        r, _ = A.fit_best(d, "value ~ group", slope=False)
        ci = r.conf_int().loc["group"]
        rows.append({"outcome": "Total" if s == "NP3TOT" else s,
                     "n_patients": d["PATNO"].nunique(), "n_obs": len(d),
                     "diff": r.params["group"], "diff_ci": (ci[0], ci[1]),
                     "p": r.pvalues["group"]})
    res = pd.DataFrame(rows).set_index("outcome")
    res["q"] = A.bh(res["p"])
    return res


# ─────────────────────────────────────────────────────────────────────────────
# DAT decline conditioned on baseline binding
# ─────────────────────────────────────────────────────────────────────────────

def dat_conditional(regions: dict[str, pd.DataFrame], age_at_scan: pd.DataFrame,
                    escalation_month: dict) -> pd.DataFrame:
    """
    Only post-baseline scans (years 1, 2, 4) are outcomes; baseline binding and
    age at the baseline scan enter as centred covariates, so baseline is never
    both an outcome and a covariate:

        SBR ~ group * months + sbr_bl_c + sbr_bl_c:months + age_bl_c + (1 | PATNO)

    L-BFGS and Powell both fitted, higher REML log-likelihood kept. Fitted to
    the onset-oriented regions (02_analyse.orient); q = BH across the four.

    age_at_scan: PATNO, age at the baseline DAT scan (age at visit where missing).
    """
    rows = []
    for name, long in regions.items():
        d = long.copy()
        d["months"] = d["EVENT_ID"].map(A.DAT_VISIT_MONTHS)
        d = A.censor(d.dropna(subset=["value", "months"]), escalation_month)
        bl = d[d["months"] == 0].groupby("PATNO")["value"].mean().rename("sbr_bl")
        d = d.merge(bl, on="PATNO").merge(age_at_scan, on="PATNO").dropna(subset=["sbr_bl", "age_bl"])
        d = A.restrict_visits(d[d["months"] > 0])
        d["sbr_bl_c"] = d["sbr_bl"] - d["sbr_bl"].mean()
        d["age_bl_c"] = d["age_bl"] - d["age_bl"].mean()

        model = smf.mixedlm("value ~ group * months + sbr_bl_c + sbr_bl_c:months + age_bl_c",
                            d, groups=d["PATNO"])
        fits = []
        for method, maxiter in [("lbfgs", 500), ("powell", 2000)]:
            try:
                fits.append(model.fit(reml=True, method=method, maxiter=maxiter,
                                      warn_convergence=False))
            except Exception:
                continue
        r = max(fits, key=lambda f: f.llf)
        ci = r.conf_int().loc["group:months"]
        rows.append({"region": name, "n_patients": d["PATNO"].nunique(), "n_obs": len(d),
                     "diff_per_year": 12 * r.params["group:months"],
                     "diff_per_year_ci": (12 * ci[0], 12 * ci[1]),
                     "p": r.pvalues["group:months"]})
    res = pd.DataFrame(rows).set_index("region")
    res["q"] = A.bh(res["p"])
    return res
