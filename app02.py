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

# ---------------------------
# Session-state init
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None            # original loaded dataset (sample or uploaded)
if "filtered_df1" not in st.session_state:
    st.session_state.filtered_df1 = None  # after applying first filters
if "filtered_df2" not in st.session_state:
    st.session_state.filtered_df2 = None  # after applying second filters
if "step" not in st.session_state:
    st.session_state.step = 0             # 0=choose/load, 1=first filters, 2=second filters, 3=charts

# --- FIX: auto-promote step once df is loaded ---
if st.session_state.df is not None and st.session_state.step == 0:
    st.session_state.step = 1

# ---------------------------
# Top controls: Reset buttons
# ---------------------------
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🔄 Reset Filters (keep dataset)"):
        st.session_state.filtered_df1 = None
        st.session_state.filtered_df2 = None
        st.session_state.step = 1 if st.session_state.df is not None else 0
        # no explicit rerun needed; button causes rerun automatically

with c2:
    if st.button("🗑️ Full Reset (start over)"):
        st.session_state.df = None
        st.session_state.filtered_df1 = None
        st.session_state.filtered_df2 = None
        st.session_state.step = 0

# =======================================================
# STEP 1: Load Dataset
# =======================================================
# if st.session_state.df is None:
if st.session_state.step == 0:
    st.sidebar.header("⚙️ Step 1: Data Options")
    data_choice = st.sidebar.radio("Select Data Source:", ("Sample Data", "Custom Data"))

    if data_choice == "Sample Data":
        if st.sidebar.button("📥 Load Sample Data"):
            # Progress
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Loading in progress..")
            df  = pd.DataFrame()

            # # Local version
            # path = os.getcwd() #get current path
            # sub_folder ='data'
            # csv_files = glob.glob(os.path.join(path,sub_folder, "*.csv"))
            
            # Streamlit
            csv_files = glob.glob('data/*.CSV')

            # loop over the list of csv files
            for csv_file in csv_files:
                dfi = pd.read_csv(csv_file, sep=';')
                df = pd.concat([df, dfi],ignore_index=True) # concatenate
            df.drop_duplicates(inplace=True)
            st.session_state.df = df
            st.session_state.filtered_df1 = None
            st.session_state.filtered_df2 = None
            st.session_state.step = 1
            status_placeholder.empty()
            st.success(f"✅ Sample dataset loaded with {st.session_state.df.shape[0]} rows"
                        f" and columns {st.session_state.df.shape[1]}")
            st.write("Data set preview:", df.head())
            if st.button("➡️ Go to next step"):
                st.rerun()
            

    elif data_choice == "Custom Data":
        uploaded_files = st.sidebar.file_uploader("Upload your dataset (CSV only)", type=["csv"], accept_multiple_files = True)
        if st.sidebar.button("📥 Load Uploaded Data"):
            if uploaded_files:
                # Progress
                status_placeholder = st.empty()
                status_placeholder.info("⏳ Loading in progress..")
                df  = pd.DataFrame()
                dfs = [pd.read_csv(file,sep=";") for file in uploaded_files]
                # If multiple files → concatenate
                # df = pd.concat([df, dfs], ignore_index=True)
                df = pd.concat(dfs, ignore_index=True)
                st.session_state.df = df
                st.session_state.filtered_df1 = None
                st.session_state.filtered_df2 = None
                st.session_state.step = 1
                status_placeholder.empty()
                st.success(f"✅ Loaded {len(uploaded_files)} file(s), {df.shape[0]} rows total and {df.shape[1]} columns ")
                st.write("Data set preview:", df.head())
                if st.button("➡️ Go to next step"):
                    st.rerun()
            else:
                st.warning("⚠️ Please upload at least one CSV file.")#prevent running analysis without data

# =======================================================
# STEP 2: First Filtering
# =======================================================
# elif st.session_state.filtered_df1 is None:
elif st.session_state.step == 1:
    st.sidebar.header("🔍 Step 2: First Filters")
    # Guard: ensure dataset exists
    if st.session_state.df is None:
        st.error("No dataset loaded. Go back to Step 1.")
        st.session_state.step = 0
    else:
        threshold_pct = st.sidebar.number_input("Threshold to detect insert change if below", value=0.4, step=0.05)
        threshold_delta = st.sidebar.number_input("Threshold to detect df_final with instantaneuos increment", value=1.5, step=0.05)
        gno_filter = st.sidebar.text_input("Filter by specific insert group (leave blank for all)")
        
        st.success(f"{type(gno_filter)} -> {gno_filter.strip()} and {gno_filter}")
        gno_filter = int(gno_filter)
        tid_filter = st.sidebar.text_input("Filter by specific tid unique label (leave blank for all)")
        tid_filter = int(tid_filter)
        machine_filter = st.sidebar.text_input("Filter by a specific machine (e.g MU305, leave blank for all)")

        if st.sidebar.button("Apply First Filters"):
            # Progress
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Processing first filters...")

            # Copy dataset 
            df = st.session_state.df.copy()

            # apply filters related to gno, tid, machine
            if gno_filter.strip():
                df = df[df["mtdb_gno"] == gno_filter]
            if tid_filter.strip():
                df = df[df["mtdb_tid"] == tid_filter]
            if machine_filter.strip():
                df = df[df["mmdb_name"] == machine_filter]
            if df.shape[0]==0:
                st.warning("⚠️ Wrong filter utensil, tid, machine! Please reset filters.")

            # Pre-processing function
            df = input_prep(df)

            # Outliers detection using a function to compute delta increase in each step
            df = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name'], group_keys=False).apply(process_delta)
            df = df.drop(df[(df['mtdb_use']==0)&(df['delta_increase']==0)].index) # remove outliers (use 0 and delta 0)
            # while loop to remove outliers (delta use higher that threshold_delta*lif) to avoid creation of new outliers
            for count_loop in range(5):
                print('outliers: ',len(df[(df['delta_increase']>threshold_delta*df['mtdb_lif'])].index))
                df = df.drop(df[(df['delta_increase']>threshold_delta*df['mtdb_lif'])].index)
                df.drop(columns="delta_increase",inplace=True)
                df = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name'], group_keys=False).apply(process_delta)
                count_loop =+ 1

            # Cumulated use computation and other attributes
            df.drop(columns="delta_increase",inplace=True) # Remove delta increase column because also process_group create it new
            df = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name'], group_keys=False).apply(process_group, threshold_pct)
            st.session_state.filtered_df1 = df
            st.session_state.filtered_df2 = None
            st.session_state.step = 2
            
            # clear the message and print completed step
            status_placeholder.empty()
            st.success("✅ First filters applied and completed calculation of cumulated use")
            st.write("Data set available as following preview:", df.head())
            if st.button("➡️ Go to next step"):
                st.rerun()


# =======================================================
# STEP 3: Second Filtering
# =======================================================
# if st.session_state.filtered_df2 is None:
elif st.session_state.step == 2:
    st.sidebar.header("🎯 Step 3: Second Filters")
    if st.session_state.filtered_df1 is None:
        st.error("First filters not applied. Go back to Step 2.")
        st.session_state.step = 1
    else:
        min_ratio = st.sidebar.number_input("Minimum ratio cumulated use / nominal life", value=1.0, step=0.1, min_value=0.2, max_value= 5.0)
        max_ratio = st.sidebar.number_input("Maximum ratio cumulated use / nominal life", value=100, step=5, min_value=10, max_value= 500)
        min_reset = st.sidebar.number_input("Minimum number of reset", value=1, step=1, min_value=1, max_value= 100)
        threshold_rec = st.sidebar.number_input("Minimum number of records for each triplet (utensil, tid, machine)", value=1000, step=200, min_value=1, max_value= 10000)

        if st.sidebar.button("Apply Second Filters"):
            # Progress
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Processing second filters...")
            # create group etc
            df = st.session_state.filtered_df1.copy()
            df, df_detailed, df_ratio = process_grouping(df,threshold_rec,min_ratio,max_ratio,min_reset)
            st.session_state.filtered_df2 = df
            st.session_state.filtered_df2_detailed = df_detailed
            st.session_state.filtered_df2_ratio = df_ratio
            st.session_state.step = 3
            # clear the message and print completed step
            status_placeholder.empty()
            st.success(f"✅ Second filters applied and Dataframes created: \n "
                    f"--> data set with aggregated : {df.shape[0]} rows and {df.shape[1]} columns \n"
                    f"--> data set with ratio cum-use vs use: {df_ratio.shape[0]} rows and {df_ratio.shape[1]} columns \n"
                    f"--> detailed data set: {df_detailed.shape[0]} rows and {df_detailed.shape[1]} columns")
            st.write("Aggregated data set available (preview):", df.head())
            # Download final dataset
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("💾 Download aggregated dataset (CSV)", csv_buf.getvalue(), file_name="aggregated_dataset.csv")
            if st.button("➡️ Go to next step"):
                st.rerun()
    
# =======================================================
# STEP 4: Charts (always based on filtered_df2)
# =======================================================
# if st.session_state.filtered_df2 is not None:
elif st.session_state.step == 3:
    if st.session_state.filtered_df2 is None:
        st.error("No filtered data available — please go back to Step 3.")
        st.session_state.step = 2
    else:
        df = st.session_state.filtered_df2
        df_detailed = st.session_state.filtered_df2_detailed
        df_ratio = st.session_state.filtered_df2_ratio
        # --- Ranking of Groups ---
        st.subheader("📑 Ranking of Utensil by Max Ratio CumUse/LIF")
        n_triplets = st.number_input("Define number of triplets to be analyzed", value=20, step=5, min_value=10, max_value=50)
        df_final = df_ratio.head(n_triplets)
        st.write(f"Top {n_triplets} triplets - dataset preview", df_final.head())
        # Download final dataset
        csv_buf = io.StringIO()
        df_ratio.to_csv(csv_buf, index=False)
        st.download_button("💾 Download final summary dataset (CSV)", csv_buf.getvalue(), file_name="summary_dataset.csv")
        # ----------------------------------------------------
        P_TARGET = st.number_input("Define percentile for recommed lifetime (e.g., 80% => B80 life)", value=0.8, step=0.1, min_value=0.1, max_value=0.9)
        # P_TARGET   = 0.80           # target survival for recommended lif (e.g., 90% => B10 life)
        # ---------------------------------------------------
        # Use session_state to cache plot images so toggling doesn't trigger recompute heavy work
        if "plots_cache" not in st.session_state:
            st.session_state.plots_cache = {}
        st.subheader("🔍📊 Step 4: Deep-Dive with Charts into Utensil lifetime")        
        for i in range(df_final.shape[0]):
            # ------------------- GET TRIPLETS -------------------
            gno_code = df_final.mtdb_gno_.iloc[i]
            tid_code = df_final.mtdb_tid_.iloc[i]
            mach_code = df_final.mmdb_name_.iloc[i]
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

                    # save image buffer for download
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    name_image = f"utensil_overview_{gno_code}_{tid_code}_{mach_code}.png"
                    st.session_state.plots_cache[name_image] = buf.getvalue()
                    st.download_button("💾 Download image", st.session_state.plots_cache[name_image], file_name=name_image, mime="image/png")
                    # st.pyplot(fig)
                    # figs_to_export = []
                    # figs_to_export.append(("survival_comparison.png", save_plot(fig)))
                    # Export plots
                    # for fname, buf in figs_to_export:
                        # st.download_button(f"⬇️ Download {fname}", data=buf, file_name=fname, mime="image/png")

