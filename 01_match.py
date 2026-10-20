"""
Grouping and propensity score matching.

Groups are defined by the side predominantly affected at onset (PPMI
`DOMSIDE`, PD Diagnosis History form, recorded at screening): left- versus
right-sided onset. Participants with Parkinson's disease are matched 1:1 on
seven baseline covariates; balance is assessed by standardised mean difference.

Input: one row per participant with the baseline covariates below and the
onset side (`domside`). Baseline definitions:

  AGE_AT_VISIT      age at the baseline visit
  disease_duration  years from first symptom to the baseline visit
  NP1RTOT           MDS-UPDRS Part I total
  NP2PTOT           MDS-UPDRS Part II total
  UPDRS3_nat        MDS-UPDRS Part III total in the untreated state
                    (no medication-state flag and not on treatment)
  MCATOT            MoCA total, screening visit preferred, baseline otherwise
  sex_true          sex (1 = male); a confounder of the grouping
  domside           1 left, 2 right, 3 symmetric, 4 unknown (fixed at onset;
                    earliest record)
"""

import numpy as np
import pandas as pd
from psmpy import PsmPy

LEFT, RIGHT = 1, 2      # DOMSIDE codes; 3 (symmetric) and 4 (unknown) are excluded

MATCH_COVARIATES = ["AGE_AT_VISIT", "disease_duration", "NP1RTOT", "NP2PTOT",
                    "UPDRS3_nat", "MCATOT", "sex_true"]

CALIPER = 0.2          # in SD of the logit propensity score
BALANCE_LIMIT = 0.1    # |SMD| below this counts as balanced


# ─────────────────────────────────────────────────────────────────────────────
# Grouping
# ─────────────────────────────────────────────────────────────────────────────

def assign_group(baseline: pd.DataFrame) -> pd.DataFrame:
    """group = 1 for right-sided onset, 0 for left-sided onset (the reference).
    Symmetric, unknown and missing onset side are excluded."""
    df = baseline[baseline["domside"].isin([LEFT, RIGHT])].copy()
    df["group"] = (df["domside"] == RIGHT).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def match(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Logistic propensity model on the seven covariates (complete cases), then
    1:1 nearest-neighbour matching on the logit propensity score, without
    replacement, within a caliper of 0.2 SD. Unmatched participants are dropped.

    Returns (eligible complete cases, matched pairs, matched participants).
    """
    eligible = df[["PATNO", "group"] + MATCH_COVARIATES].dropna().reset_index(drop=True)
    eligible["PATNO"] = eligible["PATNO"].astype(int)

    psm = PsmPy(eligible, treatment="group", indx="PATNO", exclude=[])
    psm.logistic_ps(balance=False)          # natural class distribution
    psm.kdtree_matched(matcher="propensity_logit", replacement=False,
                       caliper=CALIPER, drop_unmatched=True)

    pairs = psm.matched_ids.dropna().astype(int)
    matched_ids = set(pairs.iloc[:, 0]) | set(pairs.iloc[:, 1])
    matched = eligible[eligible["PATNO"].isin(matched_ids)].copy()
    return eligible, pairs, matched


# ─────────────────────────────────────────────────────────────────────────────
# Balance
# ─────────────────────────────────────────────────────────────────────────────

def smd(x0: pd.Series, x1: pd.Series) -> float:
    """Standardised mean difference, right minus left onset, over the pooled SD."""
    pooled = np.sqrt((x0.std(ddof=1) ** 2 + x1.std(ddof=1) ** 2) / 2)
    return 0.0 if pooled == 0 else (x1.mean() - x0.mean()) / pooled


def balance(eligible: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """SMD for every covariate before and after matching. Balance is judged by
    SMD alone: significance tests depend on sample size and ignore the pairing."""
    rows = []
    for cov in MATCH_COVARIATES:
        row = {"covariate": cov}
        for when, d in [("before", eligible), ("after", matched)]:
            x0 = d.loc[d["group"] == 0, cov].dropna()
            x1 = d.loc[d["group"] == 1, cov].dropna()
            row[f"mean_left_{when}"] = x0.mean()
            row[f"mean_right_{when}"] = x1.mean()
            row[f"smd_{when}"] = smd(x0, x1)
        row["balanced"] = abs(row["smd_after"]) < BALANCE_LIMIT
        rows.append(row)
    return pd.DataFrame(rows)


def run(baseline: pd.DataFrame):
    df = assign_group(baseline)
    eligible, pairs, matched = match(df)
    return pairs, matched, balance(eligible, matched)
