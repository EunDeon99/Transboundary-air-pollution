# -*- coding: utf-8 -*-
"""
@author: Eunice Ma


Flow:
1. Loads the prepared dataset 
2. Runs stratified OLS regressions (Favorable vs Unfavorable wind days)
   for 3 periods: 2001-2023, 2001-2013, 2014-2023
   for 3 pollutants: NOx, PM10, SO2
3. Reports coefficients, standard errors, significance
4. Reports diagnostic tests (R², Durbin-Watson)
5. Saves results 

# Requires: RQ1_daily_regression_data.csv (output from Step 1)
# Output:   RQ1_regression_results.xlsx

# Model 2- Daily regression (RQ1)
 ln(HK_Conc)_t = β₀ + β₁·ln(PRD_lag{t-k})
                + β₂·Temp_t + β₃·RH_t + β₄·Rainfall_t + β₅·Pressure_t
                + C(DayOfWeek) + C(Month) + C(Year) + ε_t

# Run separately for Favorable and Unfavorable wind days

## Variables definitions:
ln_PRD_NOx_lag1: Effect of a 1% increase in PRD emissions to HK air quality after 1 day
FE: dd,mm,yyyy 
Met controls: Temp_t, RH_t, Rainfall_t, Pressure_t
FavorableBinary wind flag (1=favorable, 0=unfavorable)
Clustered SE at month-year level

# Time period            
- Daily Regression to be run in 3 times:
Total 2001-2023 
2 Time split: 
    2001-2013, 2014-2023 (to capture China national pollution control policy)

"""
"""
###
==================Setup packages 

"""
#################################
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import jarque_bera

os.chdir(r'C:/Users/Jackal/Desktop/Master/Leipzig/Thesis/TAP/Data/RQ(1)_daily')
print("Working directory:", os.getcwd())

#######################################

# Load Data
###################################
df = pd.read_csv('RQ1_daily_regression_data.csv', parse_dates=['DATE'])
print(f"\n[1] Dataset loaded: {len(df)} rows")
print(f"    Date range: {df['DATE'].min().date()} to {df['DATE'].max().date()}")
print(f"    Favorable days:   {df['Favorable'].sum()}")
print(f"    Unfavorable days: {(df['Favorable']==0).sum()}")
#####
#######################################
#=== CREATE LAG VARIABLES {t=0,1,2,3}
######################################
print("\n[2] Creating additional lag variables...")

# CREATE LAGS BEFORE subsetting by period

pollutants = ['NOx', 'PM10', 'SO2']

for pol in pollutants:
    prd_col = f'ln_PRD_{pol}'
    df[f'ln_PRD_{pol}_lag0'] = df[prd_col]
    for lag in [1, 2, 3]:
        df[f'ln_PRD_{pol}_lag{lag}'] = df[prd_col].shift(lag)

print("   Created lags: 0, 1, 2, 3 (on full dataset)")
#######################################
#=== Center controls of Met Controls 
######################################
print("\n[3] Centering meteorological controls...")
df['Temp_c'] = df['Temp_t'] - df['Temp_t'].mean()
df['RH_c'] = df['RH_t'] - df['RH_t'].mean()
df['Rainfall_c'] = df['Rainfall_t'] - df['Rainfall_t'].mean()
df['Pressure_c'] = df['Pressure_t'] - df['Pressure_t'].mean()

print("    Centered: Temp, RH, Rainfall, Pressure")

#######################################
#=== 4. Define settings  
######################################
# Pollutants
pollutants_dict = {
    'NOx':  ('ln_NOx',  'NOx'),
    'PM10': ('ln_PM10', 'PM10'),
    'SO2':  ('ln_SO2',  'SO2'),
}
 
periods = {
    '2001-2023': (2001, 2023),
    '2001-2013': (2001, 2013),
    '2014-2023': (2014, 2023),
}
 
strata = {
    'Favorable':   1,
    'Unfavorable': 0,
    'Baseline':    None,  # None means include all days
}
 
lag_specs = {
    'Lag0': {'lags': [0], 'label': 'Contemporary (no lag)'},
    'Lag1': {'lags': [1], 'label': 'One-day lag'},
    'Lag2': {'lags': [2], 'label': 'Two-day lag'},
    'Lag3': {'lags': [3], 'label': 'Three-day lag'},
}

#######################################
#=== 4.1 Show sample size Table 
######################################
print("\n[4.1] Sample size summary by lag specification and stratum")
print("="*100)
 
sample_sizes = []
for p_label, (y0, y1) in periods.items():
    for s_label, flag in strata.items():
        #subset = df[(df['Favorable'] == flag) & (df['Year'] >= y0) & (df['Year'] <= y1)].copy()
        if flag is None:
            subset = df[(df['Year'] >= y0) & (df['Year'] <= y1)].copy()
        else:
            subset = df[(df['Favorable'] == flag) & (df['Year'] >= y0) & (df['Year'] <= y1)].copy()
        row = {'Period': p_label, 'Stratum': s_label}
        
        for lag_spec, lag_config in lag_specs.items():
            lag_day = lag_config['lags'][0]
            prd_var = f'ln_PRD_NOx_lag{lag_day}'
            n = subset[prd_var].notna().sum()
            row[lag_spec] = int(n)
        
        sample_sizes.append(row)
 
sample_df = pd.DataFrame(sample_sizes)
print(f"\n{'Period':<12} {'Stratum':<14} {'Lag0':<8} {'Lag1':<8} {'Lag2':<8} {'Lag3':<8}")
print("-"*100)
 
for _, row in sample_df.iterrows():
    print(f"{row['Period']:<12} {row['Stratum']:<14} {row['Lag0']:<8} {row['Lag1']:<8} "
          f"{row['Lag2']:<8} {row['Lag3']:<8} ")


print("="*100)

#######################################
#=== 5. Regression function
######################################
def run_regression(data, dep_var, prd_var):
    formula = (f"{dep_var} ~ {prd_var} + "
               f"Temp_c + RH_c + Rainfall_c + Pressure_c + "
               f"C(DayOfWeek) + C(Month) + C(Year)")
    
    try:
        model = smf.ols(formula, data=data).fit(
            cov_type='cluster',
            cov_kwds={'groups': data['month_year']}
        )
        return model
    except:
        return None
    
#######################################
#=== VIF
######################################
def compute_vif(data, prd_var):
    """Compute FE-corrected VIF by partialling out fixed effects"""
    key_vars = [prd_var, 'Temp_c', 'RH_c', 'Rainfall_c', 'Pressure_c']
    fe_vars = ['Year', 'Month', 'DayOfWeek']
    
    resid_df = pd.DataFrame()
    for v in key_vars:
        fe_formula = f"{v} ~ " + " + ".join([f"C({f})" for f in fe_vars])
        try:
            resid_df[v] = smf.ols(fe_formula, data=data).fit().resid
        except:
            return None
    
    resid_df = resid_df.dropna()
    try:
        vif_dict = {}
        for i, v in enumerate(key_vars):
            vif_dict[v] = variance_inflation_factor(resid_df.values, i)
        return vif_dict
    except:
        return None


###########################################
#=== 6. Loop & run regression x 108 times 
###########################################

print("\n[6] Running regressions...")
print(f"    3 periods x 3 strata x 3 pollutants x 4 lag specs = 108 regressions")
print(f"\n    {'Pollutant':<8} {'Period':<12} {'Stratum':<14} {'Lag':<10} {'N':>5} "
      f"{'Beta1':>9} {'SE':>9} {'p-val':>8} {'Sig':>5}")
print(f"    {'-'*90}")

results = {}
summary_list = []
diagnostic_list = []

# THEN in the regression loop, subset by period:
for pol, (dep, pol_short) in pollutants_dict.items():
    for p_label, (y0, y1) in periods.items():
        for s_label, flag in strata.items():
            # Subset HERE after lags are created
            if flag is None:
                subset = df[(df['Year'] >= y0) & (df['Year'] <= y1)].copy()
            else:
                subset = df[(df['Favorable'] == flag) & (df['Year'] >= y0) & (df['Year'] <= y1)].copy()
            
            # NOW the lag loop - INSIDE s_label loop, OUTSIDE if/else
            for lag_spec, lag_config in lag_specs.items():
                lag_day = lag_config['lags'][0]
                prd_var = f'ln_PRD_{pol}_lag{lag_day}'
                subset_clean = subset.dropna(subset=[prd_var]).copy()
                
                if len(subset_clean) < 100:
                    continue
                
                model = run_regression(subset_clean, dep, prd_var)
                if model is None:
                    continue
                
                coef = model.params[prd_var]
                se = model.bse[prd_var]
                pval = model.pvalues[prd_var]
                sig = ('***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else 'ns')
                r2 = model.rsquared
                n = int(model.nobs)
                dw = durbin_watson(model.resid)
                
                # STEP 8: COMPUTE DIAGNOSTICS
                # 1. VIF (FE-corrected)
                vif_dict = compute_vif(subset_clean, prd_var)
                vif_prd = vif_dict[prd_var] if vif_dict else np.nan
                
                # 2. Breusch-Pagan test (heteroscedasticity)
                try:
                    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
                    hetero = 'Yes' if bp_pval < 0.05 else 'No'
                except:
                    bp_stat = bp_pval = np.nan
                    hetero = 'N/A'
                
                # 3. Jarque-Bera test (normality)
                try:
                    jb_stat, jb_pval = jarque_bera(model.resid)
                    normal = 'No' if jb_pval < 0.05 else 'Yes'
                except:
                    jb_stat = jb_pval = np.nan
                    normal = 'N/A'
                
                # 4. DW interpretation
                if dw < 1.5:
                    dw_interp = 'Positive autocorr'
                elif dw > 2.5:
                    dw_interp = 'Negative autocorr'
                else:
                    dw_interp = 'No autocorr'
                
                results[(pol, p_label, s_label, lag_spec)] = model
                
                sig_flag = ' SIGNIFICANT' if sig != 'ns' else ''
                print(f"    {pol:<8} {p_label:<12} {s_label:<14} {lag_spec:<10} {n:>5} "
                      f"{coef:>9.4f}  {se:>8.4f}  {pval:>8.4f} {sig:>5}{sig_flag}")
                
                # Store results
                summary_list.append({
                    'Pollutant': pol,
                    'Period': p_label,
                    'Stratum': s_label,
                    'Lag': lag_spec,
                    'Lag_days': lag_day,
                    'N_obs': n,
                    'Coef': round(coef, 4),
                    'StdErr': round(se, 4),
                    'p_value': round(pval, 4),
                    'Sig': sig,
                    'R_squared': round(r2, 4),
                    'Adj_R2': round(model.rsquared_adj, 4),
                })
                
                # Store diagnostics
                diagnostic_list.append({
                    'Pollutant': pol,
                    'Period': p_label,
                    'Stratum': s_label,
                    'Lag': lag_spec,
                    'N': n,
                    'DW_stat': round(dw, 4),
                    'DW_interp': dw_interp,
                    'VIF_PRD': round(vif_prd, 2) if not np.isnan(vif_prd) else 'N/A',
                    'BP_stat': round(bp_stat, 4) if not np.isnan(bp_stat) else 'N/A',
                    'BP_p': round(bp_pval, 4) if not np.isnan(bp_pval) else 'N/A',
                    'Heterosked': hetero,
                    'JB_stat': round(jb_stat, 4) if not np.isnan(jb_stat) else 'N/A',
                    'JB_p': round(jb_pval, 4) if not np.isnan(jb_pval) else 'N/A',
                    'Normality': normal,
                })

print(f"\n*** = p<0.01, ** = p<0.05, * = p<0.10, ns = not significant")

##############################################################
# STEP 7: DETAILED SIGNIFICANT RESULTS
##############################################################
print("\n\n" + "="*100)
print("Results with significance (p < 0.10)")
print("="*100)
 
met_labels = {
    'Temp_c': 'Temperature',
    'RH_c': 'Rel_Humidity',
    'Rainfall_c': 'Rainfall',
    'Pressure_c': 'Pressure',
}
 
sig_count = 0
for (pol, p_label, s_label, lag_spec), model in results.items():
    lag_day = lag_specs[lag_spec]['lags'][0]
    prd_var = f'ln_PRD_{pol}_lag{lag_day}'
    pval_main = model.pvalues[prd_var]
    sig_main = 'YES' if pval_main < 0.10 else 'NO'
    
    if sig_main == 'NO':
        continue
    
    sig_count += 1
    dw = durbin_watson(model.resid)
    
    print(f"\n{sig_count}. {pol} | {p_label} | {s_label} | {lag_spec}")
    print(f"   N={int(model.nobs)}, R2={model.rsquared:.4f}, DW={dw:.3f}")
    print(f"   ln_PRD_{pol}_lag{lag_day}: coef={model.params[prd_var]:>10.4f}, p={pval_main:.4f}")
    
    for v, lbl in met_labels.items():
        print(f"   {lbl}: coef={model.params[v]:>10.4f}, p={model.pvalues[v]:.4f}")
 
if sig_count == 0:
    print("\n   No significant results at p<0.10")
    
    
####################################################################
# STEP 8: CREATE SUMMARY TABLE WITH COMBINED COEF+SIG
######################################################################

summary_df = pd.DataFrame(summary_list)
diagnostic_df = pd.DataFrame(diagnostic_list)

# Create combined coefficient + significance column
summary_df['Coef_Sig'] = summary_df['Coef'].astype(str) + summary_df['Sig'].astype(str)

# Select columns for display
display_cols = ['Pollutant', 'Period', 'Stratum', 'Lag', 'N_obs', 'Coef_Sig', 'StdErr', 'p_value', 'R_squared']
summary_display = summary_df[display_cols].copy()
summary_display.columns = ['Pollutant', 'Period', 'Stratum', 'Lag', 'N', 'Coef + Sig', 'SE', 'p-value', 'R²']

print("\n[RESULTS] Summary Table with Combined Coefficient + Significance")
print("="*180)
print(summary_display.to_string(index=False))

####################################################################
# STEP 8B: CREATE PIVOT TABLES
######################################################################

# Create pivot table with custom period order
period_order = ['2001-2023', '2001-2013', '2014-2023']
pivot_df = summary_df.copy()
pivot_df['Period'] = pd.Categorical(pivot_df['Period'], categories=period_order, ordered=True)
pivot_df['Coef_Sig'] = pivot_df['Coef'].round(4).astype(str) + pivot_df['Sig'].astype(str)

# Pivot for coefficients by lag
pivot_table = pivot_df.pivot_table(
    index=['Pollutant', 'Period', 'Stratum'],
    columns='Lag',
    values='Coef_Sig',
    aggfunc='first'
)
pivot_table = pivot_table.sort_index(level=['Pollutant', 'Period'], sort_remaining=False)

# Pivot for p-values by lag
pivot_pval = pivot_df.pivot_table(
    index=['Pollutant', 'Period', 'Stratum'],
    columns='Lag',
    values='p_value',
    aggfunc='first'
)
pivot_pval = pivot_pval.sort_index(level=['Pollutant', 'Period'], sort_remaining=False)

# Pivot for diagnostics
diagnostic_df['Period'] = pd.Categorical(diagnostic_df['Period'], 
                                         categories=period_order, 
                                         ordered=True)

dw_pivot = diagnostic_df.pivot_table(
    index=['Pollutant', 'Period', 'Stratum'],
    columns='Lag',
    values='DW_stat',
    aggfunc='first'
)
dw_pivot = dw_pivot.sort_index(level=['Pollutant', 'Period'], sort_remaining=False)

bp_pval_pivot = diagnostic_df.pivot_table(
    index=['Pollutant', 'Period', 'Stratum'],
    columns='Lag',
    values='BP_p',
    aggfunc='first'
)
bp_pval_pivot = bp_pval_pivot.sort_index(level=['Pollutant', 'Period'], sort_remaining=False)

jb_pval_pivot = diagnostic_df.pivot_table(
    index=['Pollutant', 'Period', 'Stratum'],
    columns='Lag',
    values='JB_p',
    aggfunc='first'
)
jb_pval_pivot = jb_pval_pivot.sort_index(level=['Pollutant', 'Period'], sort_remaining=False)

####################################################################
# STEP 8C: SAVE ALL RESULTS TO EXCEL
######################################################################

print("\n\n[8] Saving results to Excel...")
output_file = 'RQ1_Model2_Results_daily_0621.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet 1: All results (long format)
    summary_display.to_excel(writer, sheet_name='All Results', index=False)
    
    # Sheet 2: Significant only
    summary_df[summary_df['Sig'] != 'ns'].to_excel(writer, sheet_name='Significant Only', index=False)
    
    # Sheet 3-4: Pivot tables (results)
    pivot_table.to_excel(writer, sheet_name='Pivot_Coef_by_Lag')
    pivot_pval.to_excel(writer, sheet_name='Pivot_Pval_by_Lag')
    
    # Sheet 5: Full diagnostics (long format)
    diagnostic_df.to_excel(writer, sheet_name='Diagnostics_Full', index=False)
    
    # Sheet 6-8: Pivot tables (diagnostics)
    dw_pivot.to_excel(writer, sheet_name='Diagnostics_DW')
    bp_pval_pivot.to_excel(writer, sheet_name='Diagnostics_BP_pval')
    jb_pval_pivot.to_excel(writer, sheet_name='Diagnostics_JB_pval')
    
    # Sheet 9: Diagnostic guide
    diag_guide = pd.DataFrame({
        'Test': ['Durbin-Watson (DW)', 'Breusch-Pagan (BP)', 'Jarque-Bera (JB)', 'VIF'],
        'What it tests': [
            'Autocorrelation in residuals',
            'Heteroskedasticity (unequal variance)',
            'Normality of residuals',
            'Multicollinearity among predictors'
        ],
        'Good value': ['~2', 'p>0.05', 'p>0.05', '<5'],
        'Interpretation': [
            'DW~2: no AC | <1.5: positive | >2.5: negative',
            'p>0.05: homoskedastic (constant variance)',
            'p>0.05: normally distributed',
            'VIF<5: low multicollinearity'
        ],
        'How SE corrects it': [
            'Clustered SE handles autocorr',
            'Clustered SE handles heterosked',
            'Large N (8000+) makes this less critical',
            'Fixed effects partially control this'
        ]
    })
    diag_guide.to_excel(writer, sheet_name='Diagnostic_Guide', index=False)

print(f"✓ Saved: {output_file}")
print(f"  Sheets (9 total):")
print(f"    1. All Results - Long format (all regressions)")
print(f"    2. Significant Only - Long format (p<0.10 only)")
print(f"    3. Pivot_Coef_by_Lag - β by Lag (like your image)")
print(f"    4. Pivot_Pval_by_Lag - p-values by Lag")
print(f"    5. Diagnostics_Full - Full diagnostic results")
print(f"    6. Diagnostics_DW - Durbin-Watson pivot")
print(f"    7. Diagnostics_BP_pval - Breusch-Pagan pivot")
print(f"    8. Diagnostics_JB_pval - Jarque-Bera pivot")
print(f"    9. Diagnostic_Guide - Reference table")

print("\n" + "="*100)
print("RQ1 COMPLETE")
print("="*100)
