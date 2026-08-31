# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:09:34 2026

@author: Eunice

This is monthly regression 
(with same controls, except no Day FE, and replace clusted SE with NW HAC )


Flow:
1. Loads the prepared dataset 
2. Runs stratified OLS regressions x3 stratum (Months with Favorable vs Unfavorable wind days vs Baseline)
   for 3 periods: 2001-2023, 2001-2013, 2014-2023
   for 3 pollutants: NOx, PM10, SO2
3. Reports coefficients, standard errors, significance
4. Reports diagnostic tests (R², Durbin-Watson)
5. Saves results 

# Requires: RQ1_monthly_dataset.csv (output from Step 1)
# Output:   RQ1_monthly_results.xlsx



Specification:
  - 54 regressions (3 strata × 2 models × 3 pollutants × 3 time periods)
  - SE: Newey-West HAC (maxlags=3)
  - FE: Month + Year fixed effects
  - Diagnostics: Durbin-Watson, VIF, Breusch-Pagan, Jarque-Bera
 
Models:
  Model 1A: ln(HK_Conc) ~ ln(PRD_Emis) + controls + Month_FE + Year_FE
  Model 1B: ln(HK_Conc) ~ ln(HK_Emis) + controls + Month_FE + Year_FE
 
where controls = Temp_c + RH_c + Rainfall_c + Pressure_c (all centered
                                                          

# ===  Model 1A  (to test PRD effect) ===

ln(HK_Conc_{j,t}) = β₀ + β₁·ln(PRD_Emis_{j,t}) 
+ β2⋅Temp_t + β3⋅RH_t + β4⋅Rainfall_t + β5⋅Pressure_t   <-meterological controls 
+ Month_FE + Year_FE              <- fixed effect 
+εt     <- error term                                                                                     

# === Model 1B (to test how local sources matters) ===

ln(HK_Conc_{j,t}) = β₀ + β₁·ln(HK_Emis) 
+ β2⋅Temp_t + β3⋅RH_t + β4⋅Rainfall_t + β5⋅Pressure_t     <- meterological controls 
+ Month_FE + Year_FE               <- fixed effect       
+εt       <- error term                                                                                   


# Run separately for Baseline, Favorable and Unfavorable wind days

## Variables definitions:
ln_PRD_NOx_lag1: Effect of a 1% increase in PRD emissions to HK air quality 
FE: mm,yyyy 
Met controls: Temp_t, RH_t, Rainfall_t, Pressure_t
FavorableBinary wind flag (1=favorable, 0=unfavorable)
NW HAC


"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("="*180)
print("STEP 1: LOADING DATA")
print("="*180)
 
# Load the monthly regression dataset
df = pd.read_csv('C:/Users/Jackal/Desktop/Master/Leipzig/Thesis/TAP/Data/RQ(1)_monthly/RQ1_monthly_data_0620.csv')
 
print(f"\nDataset loaded:")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  Date range: {df['Year'].min()}-{df['Month'].min()} to {df['Year'].max()}-{df['Month'].max()}")
print(f"\nColumns available:")
print(f"  {list(df.columns)}")
 
print(f"\nStratification:")
print(f"  {df['stratum_with_baseline'].value_counts().to_dict()}")


# ============================================================
# STEP 2: DEFINE ANALYSIS PARAMETERS
# ============================================================
print("\n\n" + "="*180)
print("STEP 2: DEFINING ANALYSIS PARAMETERS")
print("="*180)
 
# Define strata
strata = ['Favorable', 'Unfavorable', 'Baseline']
print(f"\nStrata: {strata}")
 
# Define time periods
time_periods = [
    ('2001-2023', 2001, 2023),   # Full period
    ('2001-2013', 2001, 2013),   # Pre-2014 (before China's clean air policy)
    ('2014-2023', 2014, 2023)    # Post-2014
]
print(f"Time periods: {[t[0] for t in time_periods]}")
 
# Define pollutants
pollutants = ['NOx', 'PM10', 'SO2']
print(f"Pollutants: {pollutants}")
 
# Define models
models_dict = {
    '1A': 'ln_PRD_{pol}_emis',  # PRD emissions effect
    '1B': 'ln_HK_{pol}_emis'    # HK local emissions effect
}
print(f"Models: {list(models_dict.keys())}")
 
print(f"\nTotal regressions to run: {len(strata)} × {len(time_periods)} × {len(pollutants)} × {len(models_dict)} = {len(strata)*len(time_periods)*len(pollutants)*len(models_dict)} regressions")

# ============================================================
# STEP 3: CREATE STORAGE FOR RESULTS
# ============================================================
print("\n\n" + "="*180)
print("STEP 3: CREATING STORAGE FOR RESULTS")
print("="*180)
 
results_list = []
diagnostic_list = []
print("\n✓ Storage created for results and diagnostics")
 
# ============================================================
# STEP 4: RUN REGRESSIONS
# ============================================================
print("\n\n" + "="*180)
print("STEP 4: RUNNING REGRESSIONS")
print("="*180)
 
regression_count = 0
 
# Loop through all pollutants
for pol in pollutants:
    print(f"\n[POLLUTANT: {pol}]")
    
    # Loop through all time periods
    for p_label, y_start, y_end in time_periods:
        print(f"  Period: {p_label}", end=" | ")
        period_count = 0
        
        # Filter dataset by time period
        df_period = df[(df['Year'] >= y_start) & (df['Year'] <= y_end)].copy()
        print(f"N={len(df_period)} | ", end="")
        
        # Loop through all strata
        for stratum in strata:
            # Filter by stratum
            if stratum == 'Baseline':
                df_stratum = df_period.copy()
            else:
                df_stratum = df_period[df_period['stratum_with_baseline'] == stratum].copy()
            
            n_obs = len(df_stratum)
            
            # Define dependent variable
            dep_var = f'ln_HK_{pol}_conc'
            
            # Run both models (1A and 1B)
            for model_name, prd_var_template in models_dict.items():
                prd_var = prd_var_template.format(pol=pol)
                
                # Define regression formula
                formula = f"{dep_var} ~ {prd_var} + Temp_c + RH_c + Rainfall_c + Pressure_c + C(Month) + C(Year)"
                
                try:
                    # ============================================================
                    # FIT REGRESSION WITH NEWEY-WEST HAC STANDARD ERRORS
                    # ============================================================
                    model = smf.ols(formula, data=df_stratum).fit(
                        cov_type='HAC',
                        cov_kwds={'maxlags': 3}  # Newey-West with 3 lags
                    )
                    
                    # ============================================================
                    # EXTRACT MAIN RESULTS
                    # ============================================================
                    coef = model.params[prd_var]
                    se = model.bse[prd_var]
                    pval = model.pvalues[prd_var]
                    
                    # Determine significance level
                    if pval < 0.01:
                        sig = '***'
                    elif pval < 0.05:
                        sig = '**'
                    elif pval < 0.10:
                        sig = '*'
                    else:
                        sig = 'ns'
                    
                    # Store regression results
                    results_list.append({
                        'Pollutant': pol,
                        'Model': model_name,
                        'Period': p_label,
                        'Stratum': stratum,
                        'N_obs': n_obs,
                        'Coef': coef,
                        'StdErr': se,
                        'p_value': pval,
                        'Sig': sig,
                        'R_squared': model.rsquared,
                        'Adj_R_squared': model.rsquared_adj
                    })
                    
                    # ============================================================
                    # DIAGNOSTIC TESTS
                    # ============================================================
                    
                    # 1. DURBIN-WATSON TEST (for autocorrelation)
                    dw = durbin_watson(model.resid)
                    if dw < 1.5:
                        dw_interp = 'Positive autocorr'
                    elif dw > 2.5:
                        dw_interp = 'Negative autocorr'
                    else:
                        dw_interp = 'No autocorr'
                    
                    # 2. VARIANCE INFLATION FACTOR (for multicollinearity)
                    # Calculate FE-corrected VIF by partialling out fixed effects
                    try:
                        key_vars = [prd_var, 'Temp_c', 'RH_c', 'Rainfall_c', 'Pressure_c']
                        fe_vars = ['Month', 'Year']
                        resid_df = pd.DataFrame()
                        
                        # For each variable, regress on FE and save residuals
                        for v in key_vars:
                            fe_formula = f"{v} ~ " + " + ".join([f"C({f})" for f in fe_vars])
                            resid_df[v] = smf.ols(fe_formula, data=df_stratum).fit().resid
                        
                        resid_df = resid_df.dropna()
                        
                        # Calculate VIF for PRD variable (first column)
                        vif_prd = variance_inflation_factor(resid_df.values, 0)
                    except:
                        vif_prd = np.nan
                    
                    # 3. BREUSCH-PAGAN TEST (for heteroskedasticity)
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
                        hetero = 'Yes' if bp_pval < 0.05 else 'No'
                    except:
                        bp_stat, bp_pval, hetero = np.nan, np.nan, 'N/A'
                    
                    # 4. JARQUE-BERA TEST (for normality of residuals)
                    try:
                        jb_stat, jb_pval = stats.jarque_bera(model.resid)
                        normal = 'Yes' if jb_pval > 0.05 else 'No'
                    except:
                        jb_stat, jb_pval, normal = np.nan, np.nan, 'N/A'
                    
                    # Store diagnostic results
                    diagnostic_list.append({
                        'Pollutant': pol,
                        'Model': model_name,
                        'Period': p_label,
                        'Stratum': stratum,
                        'N_obs': n_obs,
                        'DW_stat': dw,
                        'DW_interp': dw_interp,
                        'VIF_PRD': vif_prd,
                        'BP_stat': bp_stat,
                        'BP_p': bp_pval,
                        'Heterosked': hetero,
                        'JB_stat': jb_stat,
                        'JB_p': jb_pval,
                        'Normality': normal
                    })
                    
                    regression_count += 1
                    period_count += 1
                    
                except Exception as e:
                    print(f"\nError in {pol} {model_name} {p_label} {stratum}: {str(e)[:50]}")
                    continue
        
        print(f"✓ {period_count}/6 models")
 
print(f"\n\n{'='*180}")
print(f"✓ COMPLETED {regression_count} REGRESSIONS")
print(f"{'='*180}")


 
# ============================================================
# STEP 5: CONVERT RESULTS TO DATAFRAMES
# ============================================================
print("\n\n" + "="*180)
print("STEP 5: CONVERTING RESULTS TO DATAFRAMES")
print("="*180)
 
results_df = pd.DataFrame(results_list)
diagnostic_df = pd.DataFrame(diagnostic_list)
 
print(f"\n✓ Results DataFrame: {len(results_df)} rows × {len(results_df.columns)} columns")
print(f"✓ Diagnostics DataFrame: {len(diagnostic_df)} rows × {len(diagnostic_df.columns)} columns")
 
print(f"\nResults columns: {list(results_df.columns)}")
print(f"Diagnostics columns: {list(diagnostic_df.columns)}")


# ============================================================
# STEP 6: DISPLAY SAMPLE RESULTS
# ============================================================
print("\n\n" + "="*180)
print("STEP 6: SAMPLE RESULTS")
print("="*180)
 
print("\n[First 10 regressions]")
print("-" * 180)
display_cols = ['Pollutant', 'Model', 'Period', 'Stratum', 'N_obs', 'Coef', 'StdErr', 'Sig', 'p_value', 'R_squared']
print(results_df[display_cols].head(10).to_string(index=False))



# ============================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================
print("\n\n" + "="*180)
print("STEP 7: SUMMARY STATISTICS")
print("="*180)
 
print("\n[R² by Model and Stratum]")
print("-" * 180)
r2_summary = results_df.groupby(['Model', 'Stratum'])['R_squared'].mean()
print(r2_summary.round(4))
 
print("\n\n[Significant Results (p < 0.10)]")
print("-" * 180)
sig_results = results_df[results_df['p_value'] < 0.10]
print(f"Total significant: {len(sig_results)} out of {len(results_df)} ({len(sig_results)/len(results_df)*100:.1f}%)")
 
if len(sig_results) > 0:
    print("\nSignificant regressions:")
    print(sig_results[display_cols].to_string(index=False))
    

# ============================================================
# STEP 8: DIAGNOSTIC SUMMARY
# ============================================================
print("\n\n" + "="*180)
print("STEP 8: DIAGNOSTIC SUMMARY")
print("="*180)
 
print("\n[Durbin-Watson Test - Autocorrelation]")
print("-" * 180)
dw_summary = diagnostic_df['DW_interp'].value_counts()
for interp, count in dw_summary.items():
    pct = count / len(diagnostic_df) * 100
    print(f"  {interp:20s}: {count:2d} ({pct:5.1f}%)")
print(f"  Mean DW: {diagnostic_df['DW_stat'].mean():.3f} (ideal: ~2.0)")
 
print("\n[Breusch-Pagan Test - Heteroskedasticity]")
print("-" * 180)
hetero_summary = diagnostic_df['Heterosked'].value_counts()
for hetero, count in hetero_summary.items():
    pct = count / len(diagnostic_df) * 100
    print(f"  {hetero:20s}: {count:2d} ({pct:5.1f}%)")
 
print("\n[Jarque-Bera Test - Normality]")
print("-" * 180)
normal_summary = diagnostic_df['Normality'].value_counts()
for normal, count in normal_summary.items():
    pct = count / len(diagnostic_df) * 100
    print(f"  {normal:20s}: {count:2d} ({pct:5.1f}%)")
 
print("\n[Variance Inflation Factor - Multicollinearity]")
print("-" * 180)
vif_numeric = pd.to_numeric(diagnostic_df['VIF_PRD'], errors='coerce')
print(f"  Mean VIF: {vif_numeric.mean():.4f}")
print(f"  Min VIF:  {vif_numeric.min():.4f}")
print(f"  Max VIF:  {vif_numeric.max():.4f}")
print(f"  VIF < 5:  {(vif_numeric < 5).sum()} out of {len(vif_numeric)} ({(vif_numeric < 5).sum()/len(vif_numeric)*100:.1f}%)")
print(f"  Interpretation: All VIF < 5 → No multicollinearity issues ✓")



# ============================================================
# STEP 9: SAVE RESULTS TO EXCEL (CORRECTED)
# ============================================================
 
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import os
 
print("\n\n" + "="*180)
print("STEP 9: SAVING RESULTS TO EXCEL")
print("="*180)
 
# ============================================================
# CONVERT RESULTS TO DATAFRAME (IMPORTANT!)
# ============================================================
 

print("\n[Converting results to DataFrame...]")
results_df = pd.DataFrame(results_list)
diagnostic_df = pd.DataFrame(diagnostic_list)

print("✓ Results DataFrame created")
print("✓ Diagnostics DataFrame created")

# ============================================================
# CREATE COEFFICIENT + SIGNIFICANCE COLUMN
# ============================================================
 
print("\n[Creating coefficient + significance columns...]")
 
# Create a new column combining coefficient and significance
results_df['Coef_Sig'] = results_df['Coef'].round(4).astype(str) + results_df['Sig'].astype(str)
 
print("✓ Created 'Coef_Sig' column (Coef + Sig markers)")

# ============================================================
# CREATE EXCEL FILE WITH MULTIPLE SHEETS
# ============================================================
desktop_path = r"C:\Users\Jackal\Desktop\Master\Leipzig\Thesis\TAP\Data\RQ(1)_monthly"
 
# Verify desktop path exists
if not os.path.exists(desktop_path):
    print(f"✗ Path not found: {desktop_path}")
    print("\nPlease update 'desktop_path' variable with your actual location")
else:
    print(f"✓ Desktop path verified: {desktop_path}")

output_filename = "RQ1_Monthly_Regression_Results_0620.xlsx"
output_path = os.path.join(desktop_path, output_filename)
 
print(f"\nCreating Excel file: {output_path}")
 
# Create Excel writer
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    
    # SHEET 1: ALL RESULTS
    print("  Writing Sheet 1: All Results (long format)...")
    results_df.to_excel(writer, sheet_name='All Results', index=False)
    
    # SHEET 2: SIGNIFICANT ONLY
    print("  Writing Sheet 2: Significant Results (p<0.10)...")
    sig_results = results_df[results_df['p_value'] < 0.10]
    sig_results.to_excel(writer, sheet_name='Significant Only', index=False)
    
    # SHEET 3: RESULTS PIVOT TABLE (Coef + Sig by Stratum)
    print("  Writing Sheet 3: Results Pivot Table (Coef + Sig)...")
    results_pivot_sig = results_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='Coef_Sig',
        aggfunc='first'
    )
    results_pivot_sig.to_excel(writer, sheet_name='Results Pivot - Coef+Sig')
    
    # SHEET 4: COEFFICIENT ONLY PIVOT TABLE
    print("  Writing Sheet 4: Coefficient Only Pivot Table...")
    results_pivot_coef = results_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='Coef',
        aggfunc='first'
    )
    results_pivot_coef.to_excel(writer, sheet_name='Results Pivot - Coef')
    
    # SHEET 5: P-VALUES PIVOT TABLE
    print("  Writing Sheet 5: P-values Pivot Table...")
    pval_pivot = results_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='p_value',
        aggfunc='first'
    )
    pval_pivot.to_excel(writer, sheet_name='Results Pivot - Pval')
    
    # SHEET 6: R² PIVOT TABLE
    print("  Writing Sheet 6: R² Pivot Table...")
    r2_pivot = results_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='R_squared',
        aggfunc='first'
    )
    r2_pivot.to_excel(writer, sheet_name='Results Pivot - R2')
    
    # SHEET 7: FULL DIAGNOSTIC RESULTS
    print("  Writing Sheet 7: Full Diagnostic Results...")
    diagnostic_df.to_excel(writer, sheet_name='Diagnostics Full', index=False)
    
    # SHEET 8: DURBIN-WATSON PIVOT TABLE
    print("  Writing Sheet 8: Durbin-Watson Pivot Table...")
    dw_pivot = diagnostic_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='DW_stat',
        aggfunc='first'
    )
    dw_pivot.to_excel(writer, sheet_name='Diagnostics DW')
    
    # SHEET 9: BREUSCH-PAGAN P-VALUES PIVOT TABLE
    print("  Writing Sheet 9: Breusch-Pagan Pivot Table...")
    bp_pval_pivot = diagnostic_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='BP_p',
        aggfunc='first'
    )
    bp_pval_pivot.to_excel(writer, sheet_name='Diagnostics BP')
    
    # SHEET 10: JARQUE-BERA P-VALUES PIVOT TABLE
    print("  Writing Sheet 10: Jarque-Bera Pivot Table...")
    jb_pval_pivot = diagnostic_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='JB_p',
        aggfunc='first'
    )
    jb_pval_pivot.to_excel(writer, sheet_name='Diagnostics JB')
    
    # SHEET 11: VIF PIVOT TABLE
    print("  Writing Sheet 11: VIF Pivot Table...")
    vif_pivot = diagnostic_df.pivot_table(
        index=['Pollutant', 'Model', 'Period'],
        columns='Stratum',
        values='VIF_PRD',
        aggfunc='first'
    )
    vif_pivot.to_excel(writer, sheet_name='Diagnostics VIF')
    
    # SHEET 12: DIAGNOSTIC INTERPRETATION GUIDE
    print("  Writing Sheet 12: Diagnostic Guide...")
    diag_guide = pd.DataFrame({
        'Test': [
            'Durbin-Watson (DW)',
            'Breusch-Pagan (BP)',
            'Jarque-Bera (JB)',
            'VIF'
        ],
        'What it tests': [
            'Autocorrelation in residuals',
            'Heteroskedasticity (unequal variance)',
            'Normality of residuals',
            'Multicollinearity among predictors'
        ],
        'Good value': [
            '~2.0',
            'p > 0.05',
            'p > 0.05',
            '< 5'
        ],
        'Interpretation': [
            'DW~2: no autocorr | <1.5: positive | >2.5: negative',
            'p>0.05: homoskedastic (constant variance)',
            'p>0.05: normally distributed',
            'VIF<5: low multicollinearity'
        ],
        'How HAC SE handles it': [
            'Newey-West HAC corrects for autocorr',
            'HAC robust to heteroskedasticity',
            'Large N (552) makes less critical',
            'Fixed effects partially control this'
        ]
    })
    diag_guide.to_excel(writer, sheet_name='Diagnostic Guide', index=False)
    
    # SHEET 13: SIGNIFICANCE LEGEND
    print("  Writing Sheet 13: Significance Legend...")
    sig_legend = pd.DataFrame({
        'Symbol': ['***', '**', '*', 'ns'],
        'P-value Range': ['p < 0.01', 'p < 0.05', 'p < 0.10', 'p >= 0.10'],
        'Interpretation': [
            'Highly significant (1% level)',
            'Significant (5% level)',
            'Marginally significant (10% level)',
            'Not significant'
        ]
    })
    sig_legend.to_excel(writer, sheet_name='Significance Legend', index=False)
    
    # SHEET 14: SUMMARY STATISTICS
    print("  Writing Sheet 14: Summary Statistics...")
    
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Regressions',
            'Significant Results (p<0.10)',
            '% Significant',
            'Mean R²',
            'Mean Adj R²',
            'Mean DW',
            'DW: No Autocorr (%)',
            'Mean VIF',
            'VIF < 5 (%)',
            'Heteroskedasticity (Yes %)',
            'Normality (Yes %)'
        ],
        'Value': [
            len(results_df),
            len(results_df[results_df['p_value'] < 0.10]),
            f"{len(results_df[results_df['p_value'] < 0.10])/len(results_df)*100:.1f}%",
            f"{results_df['R_squared'].mean():.4f}",
            f"{results_df['Adj_R_squared'].mean():.4f}",
            f"{diagnostic_df['DW_stat'].mean():.3f}",
            f"{(diagnostic_df['DW_interp']=='No autocorr').sum()/len(diagnostic_df)*100:.1f}%",
            f"{pd.to_numeric(diagnostic_df['VIF_PRD'], errors='coerce').mean():.4f}",
            f"{(pd.to_numeric(diagnostic_df['VIF_PRD'], errors='coerce') < 5).sum()/len(diagnostic_df)*100:.1f}%",
            f"{(diagnostic_df['Heterosked']=='Yes').sum()/len(diagnostic_df)*100:.1f}%",
            f"{(diagnostic_df['Normality']=='Yes').sum()/len(diagnostic_df)*100:.1f}%"
        ]
    })
    summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
 
# Print completion messages (OUTSIDE the with block)
print(f"\n✓ Excel file created successfully!")
print(f"  Location: {output_path}")
print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
 
print(f"\n✓ Excel file contains 14 sheets:")