# app.py
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from lifelines import (KaplanMeierFitter, WeibullFitter,
                        LogNormalFitter, LogLogisticFitter, ExponentialFitter)
from lifelines.exceptions import ConvergenceError, StatisticalWarning
from lifelines.utils import qth_survival_times
import io

import os
import glob

# ================== My Modules ================== 
from zz_utilities.Util_Preprocess import * # import Preprocessing functions
from zz_utilities.Util_Plot import * # import Plotting functions
from zz_utilities.Util_Lifetime import * # import Lifetime Plotting functions



# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Reliability Analysis", layout="wide")
st.title("🔧 Insert Lifetime Reliability Analysis")

df = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    # st.session_state.df = None
    # df = None

# ----------------------------
# Reset button (always visible, top right)
# ----------------------------
# if st.button("🔄 Reset Data"):
#     st.session_state.data_loaded = False
#     # st.session_state.df = None
#     # df = None
#     st.rerun()  # reload the app cleanly

if not st.session_state.data_loaded:
    # Dataset selection
    st.sidebar.header("⚙️ Data Options")
    data_choice = st.sidebar.radio("Select Data Source:",("Use Sample Data", "Upload My Data"))

    df = pd.DataFrame() # placeholder
    if data_choice == "Use Sample Data":
        if st.sidebar.button("📥 Load Sample Data"):
            # Load bundled CSV: loop through each CSV file and append its contents to the main dataframe
            df  = pd.DataFrame()
            path = os.getcwd() #get current path
            sub_folder ='data'
            csv_files = glob.glob(os.path.join(path,sub_folder, "*.csv"))
            # loop over the list of csv files
            for csv_file in csv_files:
                dfi = pd.read_csv(csv_file, sep=';')
                df = pd.concat([df, dfi],ignore_index=True) # concatenate
            df.drop_duplicates(inplace=True)
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded sample dataset with dimensions: records {df.shape[0]} "
                        f" and columns {df.shape[1]}")
    else:
        uploaded_files = st.sidebar.file_uploader("Upload your dataset (CSV only)", type=["csv"], accept_multiple_files = True)
        if st.sidebar.button("📥 Load Uploaded Data"):
            if uploaded_files:
                dfs = [pd.read_csv(file,sep=";") for file in uploaded_files]
                # If multiple files → concatenate
                df = pd.concat([df, dfs], ignore_index=True)
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(uploaded_files)} file(s), {df.shape[0]} rows total and {df.shape[1]} columns ")
            else:
                st.warning("⚠️ Please upload at least one CSV file.")#prevent running analysis without data
    # if df is not None:
        # st.write("### Preview of Data")
        # st.dataframe(df.head())


if df is not None:
# if st.session_state.data_loaded and st.session_state.df is not None:
# if st.session_state.data_loaded and df is not None:

    # set initial parameters
    st.sidebar.header("⚙️ Data Settings")
    threshold_pct = st.sidebar.number_input("Threshold for insert change detection - below", value=0.4, step=0.05)
    threshold_rec = st.sidebar.number_input("Minimum number of records for each triplet (utensil, tid, machine)", value=1000, step=200)
    threshold_delta = st.sidebar.number_input("Threshold to detect anomalies with instantaneuos increment", value=1.5, step=0.05)
    gno_filter = st.sidebar.text_input("Filter for specific insert group (e.g. 1450, default empty)", np.nan)
    tid_filter = st.sidebar.text_input("Filter for specific tid unique label (default empty)", np.nan)
    machine_filter = st.sidebar.text_input("Filter for a specific machine (e.g MU305, default empty)", np.nan)
    filters = {'mtdb_gno': gno_filter,'mtdb_tid': tid_filter,'mmdb_name': machine_filter}
    
    df = input_prep(df,filters)

    # Apply function to detect outliers computing delta increase
    df = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name'], group_keys=False).apply(process_delta)
    # remove outliers (use 0 and delta 0)
    df = df.drop(df[(df['mtdb_use']==0)&(df['delta_increase']==0)].index)
    # remove outliers (delta use higher that threshold_delta*lif)
    df = df.drop(df[(df['delta_increase']>threshold_delta*df['mtdb_lif'])].index)

    # Remove delta increase column because also process_group create it new
    df.drop(columns="delta_increase",inplace=True)
    # Compute cumulated use and other measurements
    df = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name'], group_keys=False).apply(process_group, threshold_pct)
    st.write("Calculation completed: cumulated use available:", df.head())

    min_ratio = st.number_input("Minimum ratio cumulated use / nominal life", value=1.0, step=0.1)
    max_ratio = st.number_input("Maximum ratio cumulated use / nominal life", value=8000, step=5)
    min_reset = st.number_input("Minimum number of reset", value=1, step=1)
    # create group etc
    df, df_detailed, df_ratio = process_grouping(df,threshold_rec,min_ratio,max_ratio,min_reset)

    st.success(f"✅ Dataframes created:\n aggregated:, {df.shape[0]} rows and {df.shape[1]} columns \ndetailed {df_detailed.shape[0]} rows and {df_detailed.shape[1]} columns ")

    # --- Ranking of Groups ---
    st.subheader("📑 Ranking of Utensil by Max Ratio CumUse/LIF")
    anomalies = df_ratio.head(20)
    st.write("Top 20", anomalies.head())
    # ----------------------------------------------------
    P_TARGET   = 0.80           # target survival for recommended lif (e.g., 90% => B10 life)
    # ---------------------------------------------------
    st.subheader("🔍 Deep-Dive into Utensil")
    
    for i in range(anomalies.shape[0]):
        # ------------------- GET TRIPLETS -------------------
        gno_code = anomalies.mtdb_gno_.iloc[i]
        tid_code = anomalies.mtdb_tid_.iloc[i]
        mach_code = anomalies.mmdb_name_.iloc[i]
        if st.checkbox(f"Show plots for {gno_code}, {tid_code}, {mach_code}"):
            # ----------------------------------------------------
            # (1) Build lifecycle table (handles failures & censored as requested)
            lifecycles = df.copy()
            lifecycles = lifecycles[(lifecycles['mtdb_gno']==gno_code)&(lifecycles['mtdb_tid']==tid_code)&(lifecycles['mmdb_name']==mach_code)]
            lifecycles["ratio_vs_lif"]=lifecycles['cum_use']/lifecycles['mtdb_lif']
            lifecycles = lifecycles[lifecycles['cum_use']>0] #remove 0 and negative values if any
            # lifecycles = lifecycles[(lifecycles['ratio_vs_lif']>0.05) & (lifecycles['ratio_vs_lif']<5)] #remove 0 and negative values if any
            lifecycles_full = df_detailed[(df_detailed['mtdb_gno']==gno_code)&(df_detailed['mtdb_tid']==tid_code)&(df_detailed['mmdb_name']==mach_code)]

            # (2) Recommendations & fits using KM and Weibull
            rec, kmf, wf = recommend_lif(lifecycles, p_target=P_TARGET)
            current_spec = rec["current_lif_median"]
            km_reco = rec["KM_recommended_lif"]

            # (3) Various analysis with LogLogistics, LogNormal, Exponential, Weibull distributions
            durations=lifecycles["cum_use"]
            events_observed=lifecycles["event_state"]
            lif = lifecycles["mtdb_lif"].median()
            n = len(durations)

            # (4) Plot
            is_sub = True
            if is_sub:
                fig = plt.figure(figsize=(15, 9))
                gs = gridspec.GridSpec(2, 3, figure=fig)
                ax1 = fig.add_subplot(gs[0, :]) # wide subplot spanning 2 columns
                ax2 = fig.add_subplot(gs[1, 0])
                ax3 = fig.add_subplot(gs[1, 1])
                ax4 = fig.add_subplot(gs[1, 2])   
                plot_cumVSuse_triplet(lifecycles_full,ax=ax1) # detailed cum use for one triplet
                if n>1:
                    # working only when 

                    # Subplot with Kaplan Meier Analysis
                    kmf, best_model, wf, metrics = analyze_group(durations, events_observed, lif)
                    plot_km(kmf, current_lif=current_spec, p_target=P_TARGET, km_reco=km_reco, ax=ax2)
                    ax2.legend(fontsize=8)
                    ax2.set_title("Kaplan–Meier Reliability-Survival")

        
                    # Subplot Failure probability with weibull and best model
                    # ---- Empirical probabilities (median ranks) ----
                    sorted_times = np.sort(durations)
                    empirical_probs = np.arange(1, n+1) / (n+1)
                    ax3.scatter(sorted_times, empirical_probs, color="black", zorder=3,alpha = 0.7,
                                label="Empirical data")
                    # ---- Fitted Weibull CDF ----
                    t_grid = wf.timeline#.values
                    cdf_fit = 1 - wf.survival_function_.values.flatten()
                    # Confidence intervals Weibull
                    ci_surv = wf.confidence_interval_survival_function_
                    cdf_lower = 1 - ci_surv.iloc[:, 1].values   # upper survival → lower failure
                    cdf_upper = 1 - ci_surv.iloc[:, 0].values   # lower survival → upper failure
                    # ax3.plot(t_grid, cdf_fit, color="blue", linewidth=2,
                    #         label=f"Weibull fit (β={wf.rho_:.2f}, η={wf.lambda_:.0f})")
                    ax3.plot(t_grid, cdf_fit, color="blue", linewidth=2,
                            label=f"Weibull fit")
                    ax3.fill_between(t_grid, cdf_lower, cdf_upper, color="blue", alpha=0.1,
                                    label="95% CI Weibull")
                    # Current lif
                    ax3.axvline(lif, color="tab:red", ls="--", label=f"Current lif ≈ {lif:.0f}")
                    # ---- Extract info best model CDF ----
                    if isinstance(best_model, WeibullFitter):
                        # print(f"  Shape (rho):  {best_model.rho_:.4f}")
                        # print(f"  Scale (lambda): {best_model.lambda_:.4f}")
                        label = f"Weibul fit: shape: {best_model.rho_:.2f}, scale:{best_model.lambda_:.2f}"
                        label = f"Best fit - Weibul"
                    elif isinstance(best_model, ExponentialFitter):
                        # print(f"  Hazard rate (lambda): {best_model.lambda_:.6f}")
                        # print(f"  Scale (1/lambda): {1/best_model.lambda_:.4f}")
                        label = f"Best fit - Exponential"
                    elif isinstance(best_model, LogLogisticFitter):
                        # print(f"  Shape (beta):  {best_model.beta_:.4f}")
                        # print(f"  Scale (alpha): {best_model.alpha_:.4f}")
                        label = f"Best fit - LogLogistic"
                    elif isinstance(best_model, LogNormalFitter):
                        # print(f"  Location (mu):   {best_model.mu_:.4f}")
                        # print(f"  Scale (sigma):   {best_model.sigma_:.4f}")
                        label = f"Best fit - LogNormal"
                    else:
                        print('here nothing')
                    # Extract timeline
                    t_grid = best_model.timeline#.values
                    # Failure probability curve
                    cdf_fit = 1 - best_model.survival_function_.values.flatten()
                    # Confidence intervals (transform survival CI into CDF CI)
                    ci_surv = best_model.confidence_interval_survival_function_
                    cdf_lower = 1 - ci_surv.iloc[:, 1].values  # upper survival → lower failure
                    cdf_upper = 1 - ci_surv.iloc[:, 0].values  # lower survival → upper failure
                    ax3.plot(t_grid, cdf_fit, linewidth=2, label=label, color='g', linestyle=':')
                    ax3.fill_between(t_grid, cdf_lower, cdf_upper, color="g", alpha=0.1,
                                    label="95% CI Best Model")
                    ax3.axvline(best_model.percentile(P_TARGET), color="tab:green", ls="-.", label=f"Recommended lif={P_TARGET:.0%} ≈ {best_model.percentile(P_TARGET):.0f}")
                    ax3.set_xlabel("Time")
                    ax3.set_ylabel("Failure probability")
                    ax3.set_ylim(0,1)
                    ax3.grid(True, alpha=0.4)
                    ax3.legend(fontsize=8)
                    ax3.set_title("Failure probability Analysis")
                    fig.suptitle(f"Analysis of {gno_code} | {tid_code} | {mach_code}")
                else:
                    ax2.text(0.5, 0.5, f'Warning: chart not created!\n No insert change detected!', horizontalalignment='center', verticalalignment='center')
                    ax3.text(0.5, 0.5, f'Warning: chart not created!\n No insert change detected!', horizontalalignment='center', verticalalignment='center')
                
                # Subplot with histogram of durations
                plot_histograms(lifecycles, ax=ax4)
                ax4.set_title("Histogram of observed durations")
                ax4.set_xlabel("Cumulated use")
                plt.xticks(rotation=90)
                plt.tight_layout()
                st.pyplot(fig)
                # st.pyplot(fig)
                # figs_to_export = []
                # figs_to_export.append(("survival_comparison.png", save_plot(fig)))
                # Export plots
                # for fname, buf in figs_to_export:
                    # st.download_button(f"⬇️ Download {fname}", data=buf, file_name=fname, mime="image/png")

