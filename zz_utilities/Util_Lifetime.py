# ================== Lifetime Analysis  ==================
from matplotlib.ticker import MaxNLocator
from lifelines import (KaplanMeierFitter, WeibullFitter,
                        LogNormalFitter, LogLogisticFitter, ExponentialFitter)
from lifelines.exceptions import ConvergenceError, StatisticalWarning
from lifelines.utils import qth_survival_times
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# ----------------------------------------------------
P_TARGET   = 0.90           # target survival for recommended lif (e.g., 90% => B10 life)
LIF_COL    = "mtdb_lif"     # defined/spec lifetime (same within a triplet)
# ----------------------------------------------------

def coalesce_lif(series: pd.Series) -> float:
    """Pick a representative 'lif' for a group (mode, then median)."""
    s = series.dropna()
    if s.empty:
        return np.nan
    modes = s.mode()
    return (modes.iloc[0] if not modes.empty else s.median())

# ------------------------- Survival & Recommendations -------------------------

def km_fit(lifecycles: pd.DataFrame):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=lifecycles["cum_use"], event_observed=lifecycles["event_state"])
    return kmf

def km_quantile_at_survival(kmf: KaplanMeierFitter, s_target: float) -> float | None:
    """
    Find smallest t where S(t) <= s_target on the stepwise KM curve.
    Returns None if survival never drops to s_target (i.e., too few failures).
    """
    sf = kmf.survival_function_  # index: time, col 'KM_estimate'
    sf = sf.rename(columns={sf.columns[0]: "S"})
    hit = sf[sf["S"] <= s_target]
    return float(hit.index[0]) if not hit.empty else None

def weibull_fit(lifecycles: pd.DataFrame) -> WeibullFitter | None:
    if lifecycles["event_state"].sum() == 0:
        return None  # no failures -> Weibull cannot be identified
    wf = WeibullFitter()
    wf.fit(lifecycles["cum_use"], event_observed=lifecycles["event_state"])
    return wf

def weibull_time_at_survival(wf: WeibullFitter, s_target: float) -> float:
    """
    For Weibull with shape rho_ and scale lambda_:
      S(t) = exp(-(t/lambda)^rho)  ->  t = lambda * (-ln S)^(1/rho)
    """
    lam = float(wf.lambda_)  # scale
    rho = float(wf.rho_)     # shape
    return lam * ((-np.log(s_target)) ** (1.0 / rho))

def recommend_lif(lifecycles: pd.DataFrame, p_target=P_TARGET):
    """
    Suggest updated lif to achieve survival ~= p_target at lif.
    Returns dict with both KM-based and Weibull-based (if available).
    """
    kmf = km_fit(lifecycles)
    rec = {"target_survival": p_target}

    # KM-based recommendation
    km_t = km_quantile_at_survival(kmf, p_target)
    rec["KM_recommended_lif"] = km_t  # may be None if survival never drops to p_target

    # Weibull-based recommendation
    wf = weibull_fit(lifecycles)
    if wf is not None:
        rec["Weibull_shape_beta"] = float(wf.rho_)
        rec["Weibull_scale_eta"]  = float(wf.lambda_)
        rec["Weibull_recommended_lif"] = weibull_time_at_survival(wf, p_target)
    else:
        rec["Weibull_shape_beta"] = None
        rec["Weibull_scale_eta"]  = None
        rec["Weibull_recommended_lif"] = None

    # Diagnostics vs current spec (if spec varies, compare to its median)
    lif_spec = lifecycles["mtdb_lif"].dropna()
    rec["current_lif_median"] = float(lif_spec.median()) if not lif_spec.empty else None

    return rec, kmf, (wf if 'wf' in locals() else None)

def compare_distributions(durations, events):
    # NOT USED
    """Fit several parametric survival models and compare AIC."""
    
    models = {
        "Weibull": WeibullFitter(),
        "Exponential": ExponentialFitter(),
        "LogNormal": LogNormalFitter(),
        "LogLogistic": LogLogisticFitter(),
    }
    
    results = []
    for name, model in models.items():
        try:
            model.fit(durations, event_observed=events)
            aic = model.AIC_
            results.append((name, aic, model))
        except Exception as e:
            results.append((name, np.inf, None))

    # Sort by AIC (lower is better)
    results = sorted(results, key=lambda x: x[1])
    print("Best fitting distributions (by AIC):")
    for name, aic, _ in results:
        print(f"{name}: AIC={aic:.2f}")
    return results


# ----------------------------------- Plots -----------------------------------

def plot_km(kmf: KaplanMeierFitter, current_lif: float | None = None,
            p_target: float = P_TARGET, km_reco: float | None = None,
            title="Kaplan–Meier Survival (duration = cum_use at cycle end)", ax=None):
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        # ax = plt.gca()
        fig, ax = plt.subplots(figsize=(8,6))
    kmf.plot(ci_show=True,ax=ax)
    if current_lif:
        ax.axvline(current_lif, color="tab:red", ls="--", label=f"Current lif ≈ {current_lif:.0f}")
    if km_reco:
        ax.axvline(km_reco, color="tab:green", ls="--", label=f"KM lif@S={p_target:.0%} ≈ {km_reco:.0f}")
    ax.set_xlabel("Cumulated use (same unit as cum_use)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if standalone:
        plt.show()
        plt.tight_layout()
    # print('Plot KM done')

def plot_histograms(lifecycles: pd.DataFrame, ax=None):
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        # ax = plt.gca()
        fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(lifecycles["cum_use"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Observed durations (cum_use at cycle end)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of observed durations (failures + censored)")
    ax.grid(True, alpha=0.3)
    if standalone:
        plt.tight_layout()
        plt.show()
    # # print('Plot HIST done')

def plot_histogramsLifRatio(lifecycles: pd.DataFrame, ax=None):
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        fig, ax = plt.subplots(figsize=(8,6))
    if lifecycles[LIF_COL].notna().any():
        ratios = lifecycles["ratio_vs_lif"].dropna()
        if not ratios.empty:
            # fig, ax = plt.subplots(figsize=(8,6))
            ax.hist(ratios, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(1.0, color="tab:red", ls="--", label="Spec lif (ratio=1)")
            ax.set_xlabel("Duration / Spec lif")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of lifetime ratios")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if standalone:
                plt.show()
        else:
            ax.text(0.5, 0.5, f'Warning: no ratio available', horizontalalignment='center', verticalalignment='center')
    else:
        ax.text(0.5, 0.5, f'Warning: no ratio available', horizontalalignment='center', verticalalignment='center')
    if standalone:
        plt.tight_layout()
        plt.show()

def plot_weibull_probability(lifecycles: pd.DataFrame, ax=None):
    """
    Quick Weibull probability plot (approximate, includes censoring by using KM estimate):
    Plot ln(-ln(S)) vs ln(t). A straight line indicates Weibull behavior.
    """
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        # ax = plt.gca()
        fig, ax = plt.subplots(figsize=(8,6))
    kmf = km_fit(lifecycles)
    sf = kmf.survival_function_.rename(columns={kmf.survival_function_.columns[0]:"S"}).reset_index()
    sf = sf[(sf["S"] > 0) & (sf["S"] < 1)]  # avoid infinities
    if sf.empty:
        return
    x = np.log(sf["timeline"].astype(float).values + 1e-9)
    y = np.log(-np.log(sf["S"].values))
    # fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(x, y, s=14)
    ax.set_xlabel("ln(duration)")
    ax.set_ylabel("ln(-ln(S))")
    ax.set_title("Weibull probability plot (KM-based)")
    ax.grid(True, alpha=0.3)
    if standalone:
        plt.show()
        plt.tight_layout()
    # print('Plot weibull done')

def plot_box_by(lifecycles: pd.DataFrame, by_col: str, ax=None):
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        fig, ax = plt.subplots(figsize=(8,6))
    if by_col not in lifecycles.columns:
        return
    lifecycles.boxplot(column="cum_use", by=by_col, grid=False, ax=ax)
    ax.set_title(f"Observed durations by {by_col}")
    plt.suptitle("")
    ax.set_ylabel("Duration")
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))
    ax.grid(True, axis="y", alpha=0.3)
    if standalone:
        plt.show()
        plt.tight_layout()
    # print('Plot BOX BY done')

def plot_survival_comparison(lifecycles: pd.DataFrame, weib_n_bootstrap=200, ax = None):
    """
    Plot Kaplan-Meier survival vs Weibull fitted distribution.
    lif = threshold line (optional).
    """
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        fig, ax = plt.subplots(figsize=(8,6))

    durations=lifecycles["cum_use"]
    event_observed=lifecycles["event_state"]
    lif=float(lifecycles["mtdb_lif"].median())

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed, label="Kaplan-Meier")
    
    wf = WeibullFitter()
    wf.fit(durations, event_observed, label="Weibull")
    
    # Create time grid
    t = np.linspace(0, durations.max()*1.1, 200)
    
    # Plot Kaplan Meier fit
    kmf.plot_survival_function(ci_show=True, color="blue",ax=ax)

    # Weibull survival function
    weibull_surv = np.exp(-(t / wf.lambda_)**wf.rho_)
    print(f'Weibull shape:{wf.rho_} | and Weibull scale {wf.lambda_}')
    ax.plot(t, weibull_surv, "r--", label="Weibull CDF (fitted)")

    # Bootstrap Weibull for CI
    weibull_samples = []
    rng = np.random.default_rng(42)
    for _ in range(weib_n_bootstrap):
        sample_idx = rng.choice(len(durations), size=len(durations), replace=True)
        d_sample = durations.iloc[sample_idx].reset_index(drop=True)
        e_sample = event_observed.iloc[sample_idx].reset_index(drop=True)

        try:
            wf_sample = WeibullFitter()
            wf_sample.fit(d_sample, event_observed=e_sample)
            weibull_samples.append(np.exp(-(t / wf_sample.lambda_)**wf_sample.rho_))
        except:
            continue

    if weibull_samples:
        weibull_samples = np.vstack(weibull_samples)
        lower = np.percentile(weibull_samples, 2.5, axis=0)
        upper = np.percentile(weibull_samples, 97.5, axis=0)
        ax.fill_between(t, lower, upper, color="red", alpha=0.2, label="Weibull CI (95%)")

    # Optional LIF marker
    if lif:
        ax.axvline(lif, color="green", linestyle=":", label=f"LIF={lif}")

    # plt.title(f"Survival Analysis: {group_label}")
    ax.set_title("Survival Analysis:")
    ax.set_xlabel("Cumulative Use (time)")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    ax.grid(True)
    if standalone:
        plt.show()
        plt.tight_layout()
    
    print("\nMedian survival time\n -- KM:", kmf.median_survival_time_, "\n -- Weibull:", wf.median_survival_time_)


def weibull_probability_plot(lifecycles: pd.DataFrame,ax = None):
    """Plot Weibull probability plot (only for observed failures)."""
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        fig, ax = plt.subplots(figsize=(8,6))
    durations=lifecycles["cum_use"]
    event_observed=lifecycles["event_state"]

    # Only keep uncensored data
    failures = durations[event_observed == 1]

    # Fit Weibull
    shape, loc, scale = stats.weibull_min.fit(failures, floc=0)

    # Create probability plot
    (osm, osr), (slope, intercept, r) = stats.probplot(failures, dist=stats.weibull_min(shape, 0, scale))

    ax.plot(osm, osr, 'o', label="Observed data")
    ax.plot(osm, slope*osm + intercept, 'r-', label=f"Weibull fit (shape={shape:.2f}, scale={scale:.2f})")
    # plt.title(f"Weibull probability plot: {group_key}")
    ax.set_title(f"Weibull probability plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Values")
    ax.legend()
    ax.grid(True)
    if standalone:
        plt.show()
        plt.tight_layout()

# ----------------------------------- Others  -----------------------------------

def compute_bm(model, percentiles=[0.1, 0.5, 0.9]):
    """Compute BM10, BM50, BM90 from fitted survival model."""
    results = {}
    for p in percentiles:
        try:
            results[f"BM{int(p*100)}"] = model.percentile(p)
        except Exception:
            results[f"BM{int(p*100)}"] = np.nan
    return results

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