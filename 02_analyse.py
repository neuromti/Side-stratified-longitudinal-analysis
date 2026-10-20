"""
Longitudinal outcome models in the matched cohort: clinical scores,
MDS-UPDRS Part III subscores, and DAT-SPECT striatal binding ratio.

Input: one long table per outcome with columns PATNO, EVENT_ID, group, value,
restricted to matched participants, and each participant's escalation month
(first device-aided therapy, see `censor`). group: 0 left-, 1 right-sided onset.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Annual visits only; interim visits have about half the sample.
VISIT_MONTHS = {"BL": 0, "V04": 12, "V06": 24, "V08": 36, "V10": 48, "V12": 60,
                "V13": 72, "V14": 84, "V15": 96, "V16": 108, "V17": 120,
                "V18": 132, "V19": 144}
DAT_VISIT_MONTHS = {"BL": 0, "V04": 12, "V06": 24, "V10": 48}

# A visit enters a model only if both groups have at least this many values.
MIN_N = 20

# Benjamini-Hochberg families for the group x time term.
FAMILIES = {
    "motor":     ["UPDRS II", "UPDRS III unmedicated", "UPDRS III ON", "UPDRS IV"],
    "non-motor": ["UPDRS I", "MoCA", "GDS", "ESS"],
}

# DAT: the grouping is a body side, so anatomical left/right regions differ by
# construction. The group comparison uses regions re-oriented to each
# participant's onset side (one family); anatomical regions are corrected as
# their own family but only verify the side assignment; bilateral averages
# are side-neutral and uncorrected.
DAT_ORIENTED = ["caudate_contra", "caudate_ipsi", "putamen_contra", "putamen_ipsi"]
DAT_ANATOMICAL = ["Caudate L", "Caudate R", "Putamen L", "Putamen R"]
DAT_BILATERAL = ["Caudate bilateral", "Putamen bilateral"]


# ─────────────────────────────────────────────────────────────────────────────
# Outcome definitions
# ─────────────────────────────────────────────────────────────────────────────
#
# UPDRS III unmedicated: examinations with no medication-state flag in untreated
# participants, plus practically defined OFF examinations; averaged per visit.
# UPDRS III ON: examinations flagged ON.
# GDS-15 total requires >= 14 of 15 items; ESS total requires >= 7 of 8.

SUBSCORE_ITEMS = {
    "Bradykinesia": ["NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL",
                     "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL"],
    "Rigidity":     ["NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL"],
    "Tremor":       ["NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU",
                     "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON"],
    "Axial":        ["NP3SPCH", "NP3FACXP", "NP3RISNG", "NP3POSTR", "NP3FRZGT",
                     "NP3GAIT", "NP3PSTBL"],
}


def subscores(u3: pd.DataFrame) -> pd.DataFrame:
    """Item sums per examination. Item values outside 0-4 (PPMI's 101 'unable to
    rate') are set missing, and a subscore with any missing item is missing."""
    u3 = u3.copy()
    for items in SUBSCORE_ITEMS.values():
        for c in items:
            u3[c] = pd.to_numeric(u3[c], errors="coerce").where(lambda s: s.between(0, 4))
    for name, items in SUBSCORE_ITEMS.items():
        u3[name] = u3[items].sum(axis=1, skipna=False)
    return u3


def orient(scans: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Re-label each participant's striatal regions relative to their onset side.
    Contralateral = the striatum opposite the side of onset (left onset -> right
    striatum; right onset -> left striatum); ipsilateral = the other one.

    scans: PATNO, EVENT_ID, group and the curated binding ratios
           MIA_{CAUDATE,PUTAMEN}_{L,R}. Returns one long table per region.
    """
    right_onset = scans["group"] == 1
    out = {}
    for region in ["CAUDATE", "PUTAMEN"]:
        left, right = scans[f"MIA_{region}_L"], scans[f"MIA_{region}_R"]
        for name, values in [("contra", np.where(right_onset, left, right)),
                             ("ipsi", np.where(right_onset, right, left))]:
            out[f"{region.lower()}_{name}"] = scans[["PATNO", "EVENT_ID", "group"]].assign(value=values)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared rules
# ─────────────────────────────────────────────────────────────────────────────

def censor(df: pd.DataFrame, escalation_month: dict) -> pd.DataFrame:
    """Drop observations from the first device-aided therapy onward: deep brain
    stimulation (first annual visit flagged DBS) or pump/infusion therapy
    (start date moved forward to the next annual visit)."""
    cens = df["PATNO"].map(escalation_month)
    return df[cens.isna() | (df["months"] < cens)].copy()


def restrict_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Keep visits where both groups have >= MIN_N observations."""
    counts = df.groupby(["months", "group"]).size().unstack(fill_value=0)
    ok = counts[(counts.get(0, 0) >= MIN_N) & (counts.get(1, 0) >= MIN_N)].index
    return df[df["months"].isin(ok)]


def prepare(long: pd.DataFrame, escalation_month: dict, visits=VISIT_MONTHS) -> pd.DataFrame:
    d = long.copy()
    d["months"] = d["EVENT_ID"].map(visits)
    d = d.dropna(subset=["value", "months"])
    return restrict_visits(censor(d, escalation_month))


def bh(p) -> np.ndarray:
    return multipletests(np.asarray(p, dtype=float), method="fdr_bh")[1]


# ─────────────────────────────────────────────────────────────────────────────
# Mixed models
# ─────────────────────────────────────────────────────────────────────────────
#
#   value ~ group + months + group:months + (1 + months | PATNO), REML
#
# group:months is the inferential term (difference in rate of change, right
# minus left onset, per month); group is the modelled difference at baseline.

def fit(d: pd.DataFrame, formula: str = "value ~ group * months"):
    """Clinical scores: random intercept and slope (L-BFGS); random intercept
    only if the slope model fails or does not converge."""
    try:
        r = smf.mixedlm(formula, d, groups=d["PATNO"], re_formula="~months").fit(
            reml=True, method="lbfgs", warn_convergence=False)
        if not r.converged:
            raise RuntimeError
        return r, "intercept + slope"
    except Exception:
        r = smf.mixedlm(formula, d, groups=d["PATNO"]).fit(reml=True, warn_convergence=False)
        return r, "intercept only"


def fit_best(d: pd.DataFrame, formula: str = "value ~ group * months", slope: bool = True):
    """Subscores, dose and levodopa response: L-BFGS and Powell both fitted and
    the solution with the higher REML log-likelihood kept, among those with
    finite likelihood and standard errors (the variance surface is flat, so a
    'converged' flag alone is not trusted). Falls back to random intercept only."""
    for re_formula in (["~months"] if slope else []) + [None]:
        model = smf.mixedlm(formula, d, groups=d["PATNO"], re_formula=re_formula)
        fits = []
        for method, maxiter in [("lbfgs", 500), ("powell", 2000)]:
            try:
                r = model.fit(reml=True, method=method, maxiter=maxiter, warn_convergence=False)
            except Exception:
                continue
            if np.isfinite(r.llf) and np.all(np.isfinite(r.bse)):
                fits.append(r)
        if fits:
            return max(fits, key=lambda r: r.llf), ("intercept + slope" if re_formula else "intercept only")
    raise RuntimeError("no model could be fitted")


def summarise(r, d: pd.DataFrame, re: str) -> dict:
    ci = r.conf_int()
    return {"n_patients": d["PATNO"].nunique(), "n_obs": len(d), "random_effects": re,
            "left_per_year": 12 * r.params["months"],
            "right_per_year": 12 * (r.params["months"] + r.params["group:months"]),
            "baseline_diff": r.params["group"],
            "baseline_diff_ci": tuple(ci.loc["group"]),
            "baseline_diff_p": r.pvalues["group"],
            "diff_per_year": 12 * r.params["group:months"],
            "diff_per_year_ci": tuple(12 * ci.loc["group:months"]),
            "p": r.pvalues["group:months"]}


# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────

def clinical(outcomes: dict[str, pd.DataFrame], escalation_month: dict) -> pd.DataFrame:
    """Every clinical score; q = BH within the motor and within the non-motor family."""
    rows = []
    for name, long in outcomes.items():
        d = prepare(long, escalation_month)
        r, re = fit(d)
        rows.append({"outcome": name, **summarise(r, d, re)})
    res = pd.DataFrame(rows).set_index("outcome")
    res["q"] = np.nan
    for members in FAMILIES.values():
        res.loc[members, "q"] = bh(res.loc[members, "p"])
    return res


def updrs3_subscores(by_state: dict[str, dict[str, pd.DataFrame]],
                     escalation_month: dict) -> pd.DataFrame:
    """Four subscores per medication state (unmedicated, ON); q = BH across the
    four subscores within each state."""
    out = []
    for state, tables in by_state.items():
        rows = []
        for name, long in tables.items():
            d = prepare(long, escalation_month)
            r, re = fit_best(d)
            rows.append({"state": state, "subscore": name, **summarise(r, d, re)})
        block = pd.DataFrame(rows)
        block["q"] = bh(block["p"])
        out.append(block)
    return pd.concat(out, ignore_index=True)


def dat(regions: dict[str, pd.DataFrame], escalation_month: dict,
        family: list[str]) -> pd.DataFrame:
    """
    Striatal binding ratio at baseline and years 1, 2 and 4. Random intercepts
    only: with four time points the random-slope model can converge to a
    degenerate solution that inflates the interaction interval.

    Both terms are reported: group (baseline difference, also as % of the
    observed left-onset baseline mean) and group x time. Each term is
    BH-corrected across the regions in `family` (DAT_ORIENTED for the group
    comparison, DAT_ANATOMICAL for the side-assignment check); other regions
    (bilateral averages) are reported uncorrected.
    """
    rows = []
    for name, long in regions.items():
        d = prepare(long, escalation_month, visits=DAT_VISIT_MONTHS)
        r = smf.mixedlm("value ~ group * months", d, groups=d["PATNO"]).fit(
            reml=True, warn_convergence=False)
        s = summarise(r, d, "intercept only")
        ref = d.loc[(d["months"] == 0) & (d["group"] == 0), "value"].mean()
        s["baseline_diff_pct"] = 100 * s["baseline_diff"] / ref
        rows.append({"region": name, **s})
    res = pd.DataFrame(rows).set_index("region")
    res["q_baseline"] = np.nan
    res["q"] = np.nan
    fam = [r for r in family if r in res.index]
    res.loc[fam, "q_baseline"] = bh(res.loc[fam, "baseline_diff_p"])
    res.loc[fam, "q"] = bh(res.loc[fam, "p"])
    return res
