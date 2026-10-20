# Side of motor symptom onset and progression in Parkinson's disease

Analysis code for a propensity-matched analysis of the Parkinson's Progression
Markers Initiative (PPMI), comparing participants with Parkinson's disease whose
motor symptoms began on the left with those whose symptoms began on the right,
over up to 12 years.

Onset side is PPMI's `DOMSIDE` ("side predominantly affected at onset", PD
Diagnosis History form, recorded at screening). Participants with symmetric or
unknown onset are excluded.

| Grouping | Eligible | Matched pairs |
|---|---:|---:|
| Left- versus right-sided onset | 439 / 561 | 431 |

## Contents

| File | Purpose |
|---|---|
| `01_match.py` | Grouping, 1:1 propensity score matching, covariate balance |
| `02_analyse.py` | Mixed models for clinical scores, MDS-UPDRS Part III subscores and DAT-SPECT (regions re-oriented to onset side); false discovery rate control |
| `03_supplementary.py` | Study withdrawal, dopaminergic dose, dose-adjusted motor models, acute levodopa response, DAT decline conditioned on baseline |

Python 3.12.7 with pandas 2.2.2, NumPy 1.26.4, SciPy 1.13.1, statsmodels 0.14.2,
psmpy 0.3.16, scikit-learn 1.7.0, lifelines 0.30.0.

## Data

The scripts document the analysis and cannot be executed without the source
data, which are not redistributable. PPMI data are available to qualified
researchers at [ppmi-info.org](https://www.ppmi-info.org/access-data-specimens/download-data)
(RRID:SCR_006431), subject to PPMI data-use agreements.

Cohort assembly from the raw PPMI files is performed upstream and is not
included here. The functions take participant-level and visit-level tables as
arguments; their expected columns are given in each docstring.

## Method

**Matching.** Logistic propensity model on seven baseline covariates (age,
disease duration, MDS-UPDRS Parts I and II, untreated MDS-UPDRS Part III, MoCA,
sex); 1:1 nearest-neighbour matching on the logit propensity score without
replacement, caliper 0.2 SD. Balance criterion |standardised mean difference|
< 0.1.

**Outcome models.** Linear mixed-effects models,
`value ~ group + months + group:months + (1 + months | participant)`, fitted by
REML on annual visits, with left-sided onset as the reference. The group × time
term is the inferential term; the group term is the modelled difference at
baseline. A visit enters a model only where both groups contribute at least 20
observations, and observations from the start of device-aided therapy (deep
brain stimulation, pump or infusion) onward are censored. DAT and acute
levodopa response models use random intercepts only.

**Imaging.** Because the grouping is a body side, anatomical left and right
striatal regions differ between groups by construction. Each participant's
regions are therefore re-labelled relative to their onset side — contralateral
(opposite the side of onset) and ipsilateral — and the group comparison is made
on these regions. The anatomical regions are fitted as well, to verify the side
assignment.

**Multiplicity.** Benjamini–Hochberg within pre-defined families: a motor family
(MDS-UPDRS Parts II, III unmedicated, III ON, IV) and a non-motor family
(MDS-UPDRS Part I, MoCA, GDS-15, ESS); the four subscores within each medication
state; the five acute-response outcomes; the four onset-oriented DAT regions,
separately for the baseline and group × time terms and for the
baseline-conditioned model. Bilateral DAT averages are reported without
correction.

**Supplementary analyses.** Cox model of time to withdrawal; mixed model of
levodopa-equivalent daily dose; motor models adjusted for concurrent dose;
same-visit OFF minus ON response in MDS-UPDRS Part III; DAT decline modelled on
post-baseline scans with baseline binding as a covariate.
