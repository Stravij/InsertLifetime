# app.py
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from lifelines import (KaplanMeierFitter, WeibullFitter,
                        LogNormalFitter, LogLogisticFitter, ExponentialFitter)
from lifelines.exceptions import ConvergenceError, StatisticalWarning
from lifelines.utils import qth_survival_times
import io


# ---------------------------
# 0. My Utility functions
# ---------------------------
def input_prep(data):
    df = data.copy()
    # remove not relevant columns
    df.drop(columns=['Extra_ToolData.location_subid','mtdb_pno','Extra_ToolData.location_id',
                    'mtdb_rot', 'mtdb_tedg','mtdb_ang', 'mtdb_twid', 'mtdb_mtang', 'mtdb_dpw',
                    'mtdb_clif', 'mtdb_cuse','mtdb_mamox', 'mtdb_mamoz'],inplace=True)
    # rename columns
    df = df.rename(columns={'Extra_ToolData.Mazak_MachineData.mmdb_name': 'mmdb_name'})
    # fix time date in the correct format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
    # modify columns with the correct unit
    df[['mtdb_len']]=df[['mtdb_len']]/1000000
    df[['mtdb_dia']]=df[['mtdb_dia']]/1000000
    # filtrare righe in cui è presente il limite vita utensile
    df = df[df['mtdb_lif']>0]
    # rimuovere utensili tornio e con 6 cifre
    df = df[~df['mtdb_gno'].isin([0, 36,111111, 111112])]
    # rimuovere tid = 0 
    df = df[~df['mtdb_tid'].isin([0])]
    # drop additional columns not relevant
    df.drop(columns=['mtdb_dia','mtdb_len','mtdb_hpw','mtdb_tpr','mtdb_lenb','mtdb_nomDia','mtdb_tno','id','mtdb_id'],inplace=True)
    # Sorting dataset
    df = df.sort_values(['mtdb_gno','mtdb_tid','mmdb_name','timestamp'])    
    # Stampare dimensioni dataset
    print('Total n. of records:', df.shape[0], ' | and columns:', df.shape[1])
    return df

# function to manipulate data
def process_group(group):
    # set list of values
    cum_use = []
    reset_counter = []
    drop_counter = []
    drop_increase = []
    event_state = []

    # set counters
    running_total = 0
    reset_count = 1 # start from 1
    drop_count = 0
    drop_increase_total = 0
    prev_use = None

    # read max use allowed
    lif = group['mtdb_lif'].iloc[0]
    # print('gno', group['mtdb_gno'].iloc[0],'| and lif:',lif)
    
    # iter through each group of triplets
    for use in group['mtdb_use']:
        if prev_use is None:
            running_total = use
        else:
            # a decrease happened
            if use < prev_use:  
                if use < threshold_pct * lif:
                    # in case of change of the insert
                    reset_count += 1 # increase
                    running_total = use  # restart from current use
                    drop_count = 0 # reset drop count
                    drop_increase_total = 0 # reset increase after a drop
                    # print('current use',use, '| current lif', lif)
                else:
                    # Drop detected
                    drop_count += 1 # increase drop count
                    drop_increase_total = 0 # reset increase after a drop 
                    running_total = running_total  # keep summing values
            # an increase happened
            else:
                running_total += (use - prev_use)
                drop_increase_total += (use - prev_use)
        # if group.iloc[-1]:
        # if group.loc[group.index[-1]]:
        #     event_ = 0
        # else:
        #     event_ = 1
        # appen computed values
        cum_use.append(running_total)
        reset_counter.append(reset_count)
        drop_counter.append(drop_count)
        drop_increase.append(drop_increase_total)
        # event_state.append(event_)
        prev_use = use
    # return new columns with previous lists
    group['cum_use'] = cum_use
    group['reset_counter'] = reset_counter
    group['drop_counter'] = drop_counter
    group['drop_increase'] = drop_increase
    
    # add event_state: default 1 (event)
    group["event_state"] = 1
    # mark last row as censored (0)
    group.loc[group.index[-1], "event_state"] = 0 
    return group


def process_grouping(df):
    # 1st level of aggragation
    df_gr = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name','reset_counter']).agg({'mtdb_lif':'max', 
                                                                        'cum_use':'max',
                                                                        'timestamp':'count',
                                                                        'drop_counter':'max',
                                                                        'drop_increase':'max',
                                                                        'event_state':'min'}).reset_index()
    # 2nd level of aggragataton adding metrix (min, max, mean, std, count)
    df_gr2 = df_gr.groupby(['mtdb_gno','mtdb_tid','mmdb_name']).agg({'mtdb_lif':'max',
                                                                    'timestamp':'sum',
                                                                'reset_counter':'max',
                                                                'cum_use':['min','max','mean','std'],
                                                                'drop_counter':['min','max','mean'],
                                                                'drop_increase':['min','max','mean'],}).copy().reset_index()
    print('Dimensions',df.shape[0])
    # 3rd level of filter
    df3 = df_gr2[df_gr2[('timestamp',  'sum')]>threshold_rec].sort_values(by=('timestamp','sum'),ascending=False).copy()
    # Flatten multiindex columns
    if isinstance(df3.columns, pd.MultiIndex):
        df3.columns = ['_'.join(map(str, col)).strip() for col in df3.columns.values]
    # Round all numeric results to 2 decimals
    float_cols = df3.select_dtypes(include="float")
    df3[float_cols.columns] = float_cols.round(2)

    # Filter
    # Create a set of triplets from df3
    main_triplets = set(df3[['mtdb_gno_','mtdb_tid_','mmdb_name_']].itertuples(index=False, name=None))

    # Keep only rows that match most important triplets
    df_detailed = df[df[['mtdb_gno','mtdb_tid','mmdb_name']].apply(tuple, axis=1).isin(main_triplets)]
    print(' --> shape:',df_detailed.shape)

    # Keep only rows that match most important triplets
    df_gr = df_gr[df_gr[['mtdb_gno','mtdb_tid','mmdb_name']].apply(tuple, axis=1).isin(main_triplets)]
    print(' --> shape:',df_gr.shape)

    # # rename columns
    # df = df.rename(columns={'mtdb_gno_':'mtdb_gno','mtdb_tid_':'mtdb_tid','mmdb_name_': 'mmdb_name'})


    return df_gr, df_detailed


# ---------------------------
# 1. Utility functions
# ---------------------------
def prepare_data(df):
    """Ensure data contains cum_use and event_state columns."""
    if "event_state" not in df.columns:
        # Example: mark last row of each triplet as censored
        df["event_state"] = 1
        for _, g in df.groupby(["mtdb_gno", "mtdb_tid", "mmdb_name"]):
            df.loc[g.index[-1], "event_state"] = 0
    return df


def compute_bm(fitter, percentiles=[0.1, 0.5, 0.9]):
    res = {}
    for p in percentiles:
        try:
            res[f"BM{int(p*100)}"] = fitter.percentile(p)
        except Exception:
            res[f"BM{int(p*100)}"] = np.nan
    return res


def fit_models(durations, events):
    """Fit multiple models and select best by AIC."""
    models = {
        "Weibull": WeibullFitter(),
        "LogNormal": LogNormalFitter(),
        "LogLogistic": LogLogisticFitter(),
        "Exponential": ExponentialFitter(),
    }
    results = {}
    for name, model in models.items():
        try:
            model.fit(durations, events, label=name)
            results[name] = {
                "model": model,
                "aic": model.AIC_,
                **compute_bm(model)
            }
        except Exception:
            pass
    # Pick best by lowest AIC
    best_name = min(results, key=lambda x: results[x]["aic"])
    return results, best_name


def analyze_group(durations, events, lif):
    # Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(durations, events, label="KM")

    km_median = kmf.median_survival_time_
    km_p90 = kmf.percentile(0.9)

    # Fit parametric models
    results, best_name = fit_models(durations, events)
    best_model = results[best_name]["model"]

    # Always keep Weibull metrics
    wf = results.get("Weibull", None)

    metrics = {
        "KM_median": km_median,
        "KM_p90": km_p90,
        "Best_model": best_name,
        "Best_AIC": results[best_name]["aic"],
        "Best_BM10": results[best_name].get("BM10"),
        "Best_BM50": results[best_name].get("BM50"),
        "Best_BM90": results[best_name].get("BM90"),
        "Weibull_BM10": wf.get("BM10") if wf else np.nan,
        "Weibull_BM50": wf.get("BM50") if wf else np.nan,
        "Weibull_BM90": wf.get("BM90") if wf else np.nan,
        "LIF": lif,
        "BM90/LIF": (results[best_name].get("BM90") or np.nan) / lif if lif else np.nan
    }
    print(metrics)
    return kmf, best_model, wf["model"] if wf else None, metrics


# new set of functions
# ---------------------------
# 1. Utility functions
# ---------------------------
def prepare_data(df):
    """Ensure data contains cum_use and event_state columns."""
    if "event_state" not in df.columns:
        df["event_state"] = 1
        for _, g in df.groupby(["mtdb_gno", "mtdb_tid", "mmdb_name"]):
            df.loc[g.index[-1], "event_state"] = 0
    return df


def compute_bm(model, percentiles=[0.1, 0.5, 0.9]):
    """Compute BM10, BM50, BM90 from fitted survival model."""
    results = {}
    for p in percentiles:
        try:
            results[f"BM{int(p*100)}"] = model.percentile(p)
        except Exception:
            results[f"BM{int(p*100)}"] = np.nan
    return results


def fit_models(durations, events):
    """Fit multiple models and return results, best model, and warnings."""
    models = {
        "Weibull": WeibullFitter(),
        "LogNormal": LogNormalFitter(),
        "LogLogistic": LogLogisticFitter(),
        "Exponential": ExponentialFitter(),
    }

    results = {}
    model_warnings = {}

    for name, model in models.items():
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always", StatisticalWarning)

                model.fit(durations, events, label=name)

                bad_variance = (
                    hasattr(model, "variance_matrix_")
                    and (np.any(pd.isna(model.variance_matrix_)) 
                         or np.any(np.diag(model.variance_matrix_) < 0))
                )

                if wlist or bad_variance:
                    warn_msg = "; ".join([str(w.message) for w in wlist])
                    if bad_variance:
                        warn_msg += " | Bad variance matrix"
                    model_warnings[name] = warn_msg
                else:
                    model_warnings[name] = "OK"

                results[name] = {
                    "model": model,
                    "aic": model.AIC_,
                    **compute_bm(model),
                }

        except (ConvergenceError, ValueError, RuntimeError) as e:
            model_warnings[name] = f"Error: {str(e)}"

    if not results:
        return None, None, model_warnings

    best_name = min(results, key=lambda x: results[x]["aic"])
    return results, best_name, model_warnings


def analyze_group(durations, events, lif):
    kmf = KaplanMeierFitter()
    kmf.fit(durations, events, label="KM")

    km_median = kmf.median_survival_time_
    km_p90 = kmf.percentile(0.9)

    # results, best_model, fit_warn = fit_models(durations, events)
    # returns: results dict, best_name (string), fit_warn dict
    results, best_name, fit_warn = fit_models(durations, events)

    metrics = {
        "KM_median": km_median,
        "KM_p90": km_p90,
        "LIF": lif,
        "BM90/LIF": np.nan,
        "Best_model": None,
        "Best_AIC": np.nan,
        "Best_BM10": np.nan,
        "Best_BM50": np.nan,
        "Best_BM90": np.nan,
        "Weibull_BM10": np.nan,
        "Weibull_BM50": np.nan,
        "Weibull_BM90": np.nan
    }

    best_model = None
    wf = None
    if results:
        best_model = results[best_name]["model"]
        metrics.update({
            "Best_model": best_name,
            "Best_AIC": results[best_name]["aic"],
            "Best_BM10": results[best_name].get("BM10"),
            "Best_BM50": results[best_name].get("BM50"),
            "Best_BM90": results[best_name].get("BM90"),
            "BM90/LIF": (results[best_name].get("BM90") or np.nan) / lif if lif else np.nan
        })
        if "Weibull" in results:
            wf = results["Weibull"]["model"]
            metrics.update({
                "Weibull_BM10": results["Weibull"].get("BM10"),
                "Weibull_BM50": results["Weibull"].get("BM50"),
                "Weibull_BM90": results["Weibull"].get("BM90"),
            })

    return kmf, best_model, wf, metrics

def compute_km(durations, events):
    """Kaplan–Meier with median, BM90 and confidence intervals."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, events, label="KM")

    ci_df = kmf.confidence_interval_survival_function_

    try:
        median = kmf.median_survival_time_
    except Exception:
        median = np.nan

    try:
        bm90 = qth_survival_times(0.9, kmf.survival_function_).values[0]
    except Exception:
        bm90 = np.nan

    return kmf, median, bm90, ci_df


def save_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


# ---------------------------
# 2. Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Survival Analysis", layout="wide")
st.title("🔧 Product Lifetime Survival Analysis")
threshold_pct = st.number_input("Threshold for insert change - below", value=0.4, step=0.05)
threshold_rec = st.number_input("Minimum threshold of records for each group", value=1000, step=200)

uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"], accept_multiple_files = True)
if uploaded:
    df = pd.DataFrame()
    for i_upload in uploaded:
        dfi = pd.read_csv(i_upload,sep=";")
        df = pd.concat([df, dfi],ignore_index=True)
    # df = pd.read_csv(uploaded,sep=";")

    # Preprocess data
    df = input_prep(df)
    st.write("Simplified data sample:", df.head())

    # Compute cumulated use and other measurements
    df = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name'], group_keys=False).apply(process_group)

    # Create a copy for detailed analysis
    # dfd = df.copy()
    
    # create group etc
    df, df_detailed = process_grouping(df)
    print('Grouped data set',df.shape)
    print('Detailed data set',df_detailed.shape)

    # Ensure cum_use + event_state exist
    # df = prepare_data(df)

    # lif = st.number_input("Defined Lifetime (LIF)", value=10000, step=100)

    # --- Ranking of Groups ---
    # st.subheader("📑 Ranking of Groups by BM90/LIF")
    results = []
    results_list = []
    j = 0
    for key, g in df.groupby(["mtdb_gno", "mtdb_tid", "mmdb_name"]):
        durations = g["cum_use"].values
        events = g["event_state"].values
        lif = g["mtdb_lif"].median()
        print(key)
        print(g)
        j += 1
        if j >= 50:
            break

        if len(durations) < 3:
            continue

        # Kaplan–Meier
        kmf, km_median, km_bm90, km_ci = compute_km(durations, events)

        # Parametric models
        model_results, best_model, model_warnings = fit_models(durations, events)

        if not model_results:
            continue
        # Build results row
        row = {
            "Group": str(key), #key
            "Lif":lif,
            "KM_median": km_median,
            "KM_BM90": km_bm90,
            "Weibull_BM10": model_results.get("Weibull", {}).get("BM10", np.nan),
            "Weibull_BM50": model_results.get("Weibull", {}).get("BM50", np.nan),
            "Weibull_BM90": model_results.get("Weibull", {}).get("BM90", np.nan),
            "BestModel": best_model,
            "Best_BM10": model_results[best_model]["BM10"],
            "Best_BM50": model_results[best_model]["BM50"],
            "Best_BM90": model_results[best_model]["BM90"],
            "FitWarnings": "; ".join([f"{m}: {w}" for m, w in model_warnings.items()])
        }

        results_list.append(row)

        # try:
        #     kmf, best_model, wf, metrics = analyze_group(durations, events, lif)
        #     row = {"Group": group_key,"BestModel": best_model,
        #            "FitWarnings": fit_warn,# plus BM10/BM50/BM90 from best + KM + Weibull
        #            }
        #     row.update(metrics)
        #     results.append(row)
        # except Exception as e:
        #     st.warning(f"Error with group {key}: {e}")
        
        # --- Plots for key groups ---
        if st.checkbox(f"Show plots for {key}", key=str(key)):
            fig, ax = plt.subplots(figsize=(8, 5))
            kmf.plot_survival_function(ax=ax, ci_show=True)
            if best_model:
                model_results[best_model]["model"].plot_survival_function(ax=ax)
            if "Weibull" in model_results:
                model_results["Weibull"]["model"].plot_survival_function(ax=ax)
            ax.set_title(f"Survival Analysis for {key}")
            ax.set_xlabel("cum_use")
            ax.set_ylabel("Survival probability")
            st.pyplot(fig)

            # Weibull Probability Plot
            if "Weibull" in model_results:
                wf = model_results["Weibull"]["model"]
                sorted_durations = np.sort(durations)
                ranks = np.arange(1, len(sorted_durations)+1) / (len(sorted_durations)+1)
                fig, ax = plt.subplots()
                ax.plot(np.log(sorted_durations), np.log(-np.log(1-ranks)), "o", label="Empirical")
                ax.plot(
                    np.log(sorted_durations),
                    np.log(-np.log(1-wf.survival_function_at_times(sorted_durations)))),
                ax.set_title("Weibull Probability Plot")
                ax.set_xlabel("ln(Time)")
                ax.set_ylabel("ln(-ln(Survival))")
                st.pyplot(fig)

    # Final table
    if results_list:
        results_df = pd.DataFrame(results_list)

        # Sort by BM90/LIF ratio
        results_df["BM90/LIF"] = results_df["Best_BM90"] / results_df["Lif"] 
        results_df = results_df.sort_values("BM90/LIF", ascending=False)

        st.subheader("📑 Ranking of Groups by BM90/LIF")
        st.dataframe(results_df)
    
    # if results:
    #     res_df = pd.DataFrame(results)
    #     res_sorted = res_df.sort_values("BM90/LIF", ascending=False)
    #     st.dataframe(res_sorted)

        # Download CSV
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Results as CSV", data=csv_buf.getvalue(), file_name="survival_results.csv")

        # --- Select Group for Deep-Dive ---
        st.subheader("🔍 Deep-Dive into One Group")
        group_selected = st.selectbox("Pick a group", results_df["Group"].astype(str).unique())
        group_data = df[df.apply(lambda r: str((r["mtdb_gno"], r["mtdb_tid"], r["mmdb_name"])) == group_selected, axis=1)]
        durations = group_data["cum_use"].values
        events = group_data["event_state"].values
        kmf, best_model, wf, metrics = analyze_group(durations, events, lif)

        st.write("Metrics:", metrics)

        figs_to_export = []

        # KM vs Best vs Weibull
        fig, ax = plt.subplots(figsize=(8, 5))
        kmf.plot_survival_function(ax=ax, ci_show=True)
        if best_model: best_model.plot_survival_function(ax=ax, ci_show=True)
        if wf: wf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_title(f"Survival Comparison for {group_selected}")
        ax.set_ylabel("Survival Probability")
        ax.set_xlabel("Cumulated Use")
        st.pyplot(fig)
        figs_to_export.append(("survival_comparison.png", save_plot(fig)))

        # Weibull Probability Plot
        if wf:
            fig, ax = plt.subplots(figsize=(8, 5))
            wf.plot(ax=ax)
            ax.set_title("Weibull Probability Plot")
            st.pyplot(fig)
            figs_to_export.append(("weibull_probability.png", save_plot(fig)))

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(durations, bins=20, alpha=0.7)
        ax.set_title("Histogram of Cumulated Use")
        ax.set_xlabel("Cumulated Use")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        figs_to_export.append(("histogram.png", save_plot(fig)))

        # Export plots
        for fname, buf in figs_to_export:
            st.download_button(f"⬇️ Download {fname}", data=buf, file_name=fname, mime="image/png")
