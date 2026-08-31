# -*- coding: utf-8 -*-
"""
RQ2 — Model (3) Monthly first-differenced regression
Created : 2026-06-04
 
@author: Eunice
 
SPECIFICATION
    Δln(EI_t) = β₀ + β₁ · ΔPRD_exposure_{t-k}
                + β₂·ΔTemp + β₃·ΔRH + β₄·ΔRainfall + β₅·ΔPressure
                + Month FE + Year FE + ε_t
 
SCOPE
    Pollutants : NOx, PM10, SO2                              (3)
    Periods    : 2001–2023, 2001–2013, 2014–2023            (3)
    Lags       : k ∈ {0, 1, 2, 3, 6} months                  (5)
    Total      : 45 regressions
 
ECONOMETRICS
    Standard errors : Newey-West HAC (maxlags = 3)
    Reference cats  : Month = January (1); Year = first year of each period
 
INTERPRETATION
    β₁ < 0  →  HK abates more when PRD exposure rises (H3 supported)
    β₁ = 0  →  no systematic abatement response
    β₁ > 0  →  HK co-moves with PRD (no abatement / amplification)
 
PRE-REGISTERED CAVEAT — SO2
    SO2 in 2014–2023 is expected to yield β₁ ≈ 0 because the post-2014
    transboundary SO2 mechanism collapsed (HK marine fuel sulfur regs,
    2015 & 2019). That null documents the structural break and should be
    reported as a substantive finding, not a methodological failure.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1 — LOAD PREPARED DATASET
# =============================================================================
df = pd.read_csv('RQ2_monthly_regression_data_0608.csv')
print(f"Loaded RQ2 dataset: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Year range: {df['Year'].min()} – {df['Year'].max()}\n")
 
# =============================================================================
# STEP 2 — REGRESSION GRID
# =============================================================================
POLLUTANTS = ['NOx', 'PM10', 'SO2']
PERIODS    = {
    '2001-2023': (2001, 2023),
    '2001-2013': (2001, 2013),
    '2014-2023': (2014, 2023),
}
LAGS = [0, 1, 2, 3, 6]
 
N_total = len(POLLUTANTS) * len(PERIODS) * len(LAGS)
print(f"Regression grid: {len(POLLUTANTS)} pollutants "
      f"× {len(PERIODS)} periods × {len(LAGS)} lags = {N_total} regressions\n")
 
# =============================================================================
# STEP 3 — HELPER: significance stars
# =============================================================================
def stars(p):
    if pd.isna(p):  return ''
    if p < 0.01:    return '***'
    if p < 0.05:    return '**'
    if p < 0.10:    return '*'
    return ''

# =============================================================================
# STEP 4 — RUN ALL REGRESSIONS
# =============================================================================
results = []
 
for pol in POLLUTANTS:
    print(f"\n──── Pollutant: {pol} ────")
    for period_name, (y_start, y_end) in PERIODS.items():
        for k in LAGS:
            dv = f'dln_EI_{pol}'
            iv = f'dPRD_exposure_{pol}_lag{k}'
 
            # Sub-sample + drop NaN
            sub_cols = [dv, iv,
                        'dTemp_mean', 'dRH_mean',
                        'dRainfall_mean', 'dPressure_mean',
                        'Year', 'Month']
            sub = df[(df['Year'] >= y_start) & (df['Year'] <= y_end)]
            sub = sub.dropna(subset=sub_cols).copy()
 
            if sub.shape[0] < 30:
                print(f"  {period_name} | lag{k}: N={sub.shape[0]} — skipped")
                continue
 
            # Formula: Month + Year FE via C(); Jan + first year as reference
            formula = (f'{dv} ~ {iv} + dTemp_mean + dRH_mean + '
                       f'dRainfall_mean + dPressure_mean + '
                       f'C(Month) + C(Year)')
 
            try:
                m = smf.ols(formula, data=sub).fit(
                    cov_type='HAC', cov_kwds={'maxlags': 3}
                )
 
                # Diagnostics
                dw   = durbin_watson(m.resid)
                lb6  = acorr_ljungbox(m.resid, lags=[6], return_df=True)
                lb_p = lb6['lb_pvalue'].iloc[0]
 
                # Control coefficients (for reference; not the main interest)
                ctrl = {f'beta_{c}': m.params.get(c, np.nan)
                        for c in ['dTemp_mean', 'dRH_mean',
                                  'dRainfall_mean', 'dPressure_mean']}
 
                row = {
                    'Pollutant':   pol,
                    'Period':      period_name,
                    'Lag':         k,
                    'beta1':       m.params[iv],
                    'SE':          m.bse[iv],
                    't_stat':      m.tvalues[iv],
                    'p_value':     m.pvalues[iv],
                    'sig':         stars(m.pvalues[iv]),
                    'N':           int(m.nobs),
                    'R2':          m.rsquared,
                    'adj_R2':      m.rsquared_adj,
                    'DW':          dw,
                    'LB_p_lag6':   lb_p,
                    **ctrl,
                }
                results.append(row)
 
                print(f"  {period_name} | lag{k}: "
                      f"β₁ = {m.params[iv]:+.4f}{stars(m.pvalues[iv]):<3} "
                      f"(SE = {m.bse[iv]:.4f}, N = {int(m.nobs)}, "
                      f"R²adj = {m.rsquared_adj:.3f})")
 
            except Exception as e:
                print(f"  ✗ {period_name} | lag{k}: {e}")
 
results_df = pd.DataFrame(results)
 

# =============================================================================
# STEP 5 — PIVOT TABLES FOR COMPACT REVIEW
# =============================================================================
results_df['beta_str'] = results_df.apply(
    lambda r: f"{r['beta1']:+.4f}{r['sig']}", axis=1
)
results_df['se_str']   = results_df['SE'].map(lambda x: f"({x:.4f})")
 
# β₁ pivot: rows = Lag, columns = Pollutant × Period
pivot_beta = results_df.pivot_table(
    index='Lag',
    columns=['Pollutant', 'Period'],
    values='beta_str',
    aggfunc='first'
)
 
# SE pivot (same layout)
pivot_se = results_df.pivot_table(
    index='Lag',
    columns=['Pollutant', 'Period'],
    values='se_str',
    aggfunc='first'
)
 
print("\n" + "="*80)
print("β₁ on ΔPRD_exposure_{t-k}, all specifications")
print("="*80)
print(pivot_beta.to_string())
 
print("\n" + "="*80)
print("Standard errors (Newey-West, maxlags=3)")
print("="*80)
print(pivot_se.to_string())
 

# =============================================================================
# STEP 6 — SAVE
# =============================================================================
out_xlsx = 'RQ2_Model3_results_0608.xlsx'
 
 
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as wr:
    # All 45 regressions, full detail
    results_df.drop(columns=['beta_str', 'se_str']).to_excel(
        wr, sheet_name='All_regressions', index=False
    )
    # Compact β₁ pivot
    pivot_beta.to_excel(wr, sheet_name='Beta1_pivot')
    pivot_se.to_excel  (wr, sheet_name='SE_pivot')
    # Diagnostics-only
    diag_cols = ['Pollutant', 'Period', 'Lag',
                 'N', 'R2', 'adj_R2', 'DW', 'LB_p_lag6']
    results_df[diag_cols].to_excel(wr, sheet_name='Diagnostics', index=False)
 

print(f"✓ Saved → {out_xlsx}")

# =============================================================================
# STEP 7 — QUICK SUBSTANTIVE SUMMARY
# =============================================================================
print("\n" + "="*80)
print("Summary: significant β₁ (p < 0.10)")
print("="*80)
sig = results_df[results_df['p_value'] < 0.10][
    ['Pollutant', 'Period', 'Lag', 'beta1', 'SE', 'p_value', 'sig', 'N']
].sort_values(['Pollutant', 'Period', 'Lag'])
 
if sig.empty:
    print("  (no specifications with p < 0.10)")
else:
    print(sig.to_string(index=False))
 
print("\nDone.")

# =============================================================================
# STEP 8 — RESIDUAL NORMALITY CHECK (headline spec: PM10, lag 1, full period)
# =============================================================================
from scipy import stats
import matplotlib.pyplot as plt

dv, iv = 'dln_EI_PM10', 'dPRD_exposure_PM10_lag1'
sub = df.dropna(subset=[dv, iv, 'dTemp_mean', 'dRH_mean',
                        'dRainfall_mean', 'dPressure_mean']).copy()

m = smf.ols(f'{dv} ~ {iv} + dTemp_mean + dRH_mean + '
            f'dRainfall_mean + dPressure_mean + C(Month) + C(Year)',
            data=sub).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

resid = m.resid
w, p = stats.shapiro(resid)
print(f"Shapiro-Wilk on residuals: W = {w:.4f}, p = {p:.4f}")
print(f"Skew = {stats.skew(resid):.2f}, excess kurtosis = {stats.kurtosis(resid):.2f}")

fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(resid, dist='norm', plot=ax)
ax.set_title('Q-Q plot of residuals, Model 3 (PM10, lag 1)')
plt.tight_layout()
plt.savefig('QQ_resid_M3_PM10_lag1.png', dpi=300)
plt.show()


# ==========
# Check extreme months
# ====
extreme = m.resid.abs().sort_values(ascending=False).head(6)
print(sub.loc[extreme.index, ['Year', 'Month']].assign(resid=extreme))