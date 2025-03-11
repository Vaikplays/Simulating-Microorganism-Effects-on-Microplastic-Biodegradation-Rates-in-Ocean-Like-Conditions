"""
dashboard.py - Biodegradation Dashboard using Streamlit

This dashboard performs the following tasks:
1. Loads and cleans biodegradation data from a CSV file.
2. Identifies plastics with sufficient data and lists valid microorganisms.
3. Allows the user to compare 1–5 microorganisms for a selected plastic.
4. For each microorganism:
   - Applies a fallback chain of models (Weibull, Log-transform Polynomial, PCHIP, Isotonic).
   - Plots experimental data and the fitted fallback curve.
   - Optionally overlays a log-fit curve.
   - Computes model evaluation metrics (RMSE, R², MAE).
5. Produces a final overlay graph comparing all fallback/log-fit curves.
6. Determines which microorganism is predicted to be best at the user-specified day.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging
import warnings

from scipy.optimize import curve_fit, OptimizeWarning
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

###############################################################################
# 1) LOGGING
###############################################################################
def setup_logging():
    """Configure logging to show messages with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

###############################################################################
# 2) DATA LOADING & CLEANING
###############################################################################
def load_and_clean(csv_path="experimental_data.csv"):
    """
    Loads data from a CSV file and cleans it.

    Required columns:
      - plastic_type, microorganism, degradation_rate, days, temperature, pH

    Steps:
      - Drop rows missing these columns.
      - Strip non-digit characters from numeric fields.
      - Convert to numeric and drop remaining NaNs.
      - Filter for days >= 0 and degradation in [0, 100].
      - Strip whitespace from text fields.
    """
    df = pd.read_csv(csv_path)
    needed = ["plastic_type", "microorganism", "degradation_rate", "days", "temperature", "pH"]
    df.dropna(subset=needed, inplace=True)

    for col in ["days", "degradation_rate", "temperature", "pH"]:
        df[col] = df[col].astype(str).str.replace(r"[^\d.]+", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["days", "degradation_rate", "temperature", "pH"], inplace=True)
    df = df[(df["days"] >= 0) & (df["degradation_rate"].between(0, 100))]

    df["plastic_type"] = df["plastic_type"].astype(str).str.strip()
    df["microorganism"] = df["microorganism"].astype(str).str.strip()
    return df

###############################################################################
# 3) GATHER VALID PLASTICS & MICROS
###############################################################################
def gather_plastics_with_valid_micros(df, threshold=2):
    """
    Returns a list of plastics that have at least one microorganism
    with >= threshold data points. Sorted by descending average degrade.
    """
    grp = df.groupby("plastic_type").size().reset_index(name="count")
    grp = grp[grp["count"] >= threshold].copy()
    valid_list = []
    for _, row in grp.iterrows():
        plastic = row["plastic_type"]
        sub = df[df["plastic_type"] == plastic]
        micro_grp = sub.groupby("microorganism").size().reset_index(name="m_count")
        micro_valid = micro_grp[micro_grp["m_count"] >= threshold]
        if not micro_valid.empty:
            valid_list.append(plastic)
    if not valid_list:
        return []
    df_avg = (
        df[df["plastic_type"].isin(valid_list)]
        .groupby("plastic_type")["degradation_rate"]
        .mean()
        .reset_index(name="avg_deg")
    )
    df_avg.sort_values("avg_deg", ascending=False, inplace=True)
    return df_avg["plastic_type"].tolist()

def gather_micros_options(df, plastic, threshold=2):
    """
    For a given plastic, return multiple sorting options for microorganisms:
      - Best Biodegradation (Highest avg)
      - Most Data Points
      - Alphabetical
      - Worst Biodegradation (Lowest avg)
    """
    sub = df[df["plastic_type"] == plastic]
    grp = sub.groupby("microorganism").agg(
        count=('microorganism', 'size'),
        avg_deg=('degradation_rate', 'mean')
    ).reset_index()
    grp = grp[grp["count"] >= threshold]
    options = {
        "Best Biodegradation (Highest avg)": grp.sort_values("avg_deg", ascending=False),
        "Most Data Points": grp.sort_values("count", ascending=False),
        "Alphabetical": grp.sort_values("microorganism"),
        "Worst Biodegradation (Lowest avg)": grp.sort_values("avg_deg", ascending=True)
    }
    return options

###############################################################################
# 4) FORCE (0,0) & SATURATE AT 100
###############################################################################
def force_zero_start(subset):
    """
    If earliest day > 0, insert an extra row at day=0, degrade=0.
    Ensures the curve starts from zero degrade at day=0.
    """
    if subset.empty:
        return subset
    earliest_day = subset["days"].min()
    if earliest_day > 0:
        new_row = {
            "plastic_type": subset["plastic_type"].iloc[0],
            "microorganism": subset["microorganism"].iloc[0],
            "degradation_rate": 0.0,
            "days": 0.0,
            "temperature": subset["temperature"].iloc[0],
            "pH": subset["pH"].iloc[0]
        }
        subset = pd.concat([subset, pd.DataFrame([new_row])], ignore_index=True)
    return subset

def force_saturate_100(x_data, y_data, offset=200.0):
    """
    If final degrade < 100, append a new point at (last_day + offset, 100%).
    """
    if y_data[-1] < 100:
        big_day = x_data[-1] + offset
        x_data = np.append(x_data, big_day)
        y_data = np.append(y_data, 100.0)
    return x_data, y_data

###############################################################################
# 5) DIRECT LINE (2 POINTS)
###############################################################################
def fit_two_points(subset):
    """
    If exactly 2 data points => direct line through them.
    Negative slope allowed if data shows it.
    """
    if len(subset) != 2:
        return None, "Not exactly 2 points"
    sub_s = subset.sort_values("days")
    x1, y1 = sub_s.iloc[0]["days"], sub_s.iloc[0]["degradation_rate"]
    x2, y2 = sub_s.iloc[1]["days"], sub_s.iloc[1]["degradation_rate"]
    dx = x2 - x1
    if dx <= 0:
        return None, "Two points have invalid day => can't fit"
    slope = (y2 - y1) / dx
    def predict_line(t):
        t_arr = np.array(t)
        pred = y1 + slope * (t_arr - x1)
        return np.clip(pred, 0, 100)
    return predict_line, None

###############################################################################
# 6) WEIBULL MODEL
###############################################################################
def weibull_model(t, lambd, kappa):
    """Weibull function for biodegradation: y = 100*(1 - e^-((t/lambd)^kappa))."""
    return 100.0 * (1 - np.exp(-(t/lambd)**kappa))

def fit_weibull(subset):
    """
    Fit the Weibull model to data.
    Pre-process => force start at 0, saturate at 100 => curve_fit => handle errors.
    """
    from scipy.optimize import curve_fit
    sub_s = subset.sort_values("days")
    x_data = sub_s["days"].values
    y_data = sub_s["degradation_rate"].values
    if x_data[0] > 0:
        x_data = np.insert(x_data, 0, 0)
        y_data = np.insert(y_data, 0, 0)
    if y_data[-1] < 100:
        x_data, y_data = force_saturate_100(x_data, y_data, offset=100)
    p0 = [max(1.0, np.median(x_data)), 1.0]
    try:
        popt, _ = curve_fit(
            weibull_model, x_data, y_data, p0=p0,
            bounds=([1, 0.1], [max(x_data)*2, 10]),
            maxfev=10000
        )
        def predict_weibull(t):
            return np.clip(weibull_model(t, *popt), 0, 100)
        return predict_weibull, 2, None
    except RuntimeError as e:
        return None, 0, f"Weibull error: {str(e)}"

###############################################################################
# 7) LOG-TRANSFORM POLY
###############################################################################
def fit_logtransform_poly(subset):
    """
    Fit a polynomial in log(1+t). Usually gives a smooth curve for biodegradation.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    sub_s = subset.sort_values("days")
    x_data = sub_s["days"].values
    y_data = sub_s["degradation_rate"].values
    if len(x_data) < 2:
        return None, "Not enough data"
    if x_data[0] > 0:
        x_data = np.insert(x_data, 0, 0)
        y_data = np.insert(y_data, 0, 0)
    if y_data[-1] < 100:
        x_data, y_data = force_saturate_100(x_data, y_data, offset=100)

    X_log = np.log1p(x_data)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_log.reshape(-1, 1))
    linreg = LinearRegression()
    linreg.fit(X_poly, y_data)

    def predict_logpoly(t):
        t_arr = np.clip(np.array(t), 0, None)
        t_log = np.log1p(t_arr)
        t_poly = poly.transform(t_log.reshape(-1, 1))
        pr = linreg.predict(t_poly)
        return np.clip(pr, 0, 100)
    return predict_logpoly, -5, None

###############################################################################
# 8) PCHIP & ISOTONIC
###############################################################################
def fit_pchip(subset, saturate_100=False):
    """
    Fit a Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
    Optionally saturate at 100% if degrade < 100 at the end.
    """
    from scipy.interpolate import PchipInterpolator
    if len(subset) < 2:
        return None, "Not enough data"
    sub_s = subset.sort_values("days")
    x_data = sub_s["days"].values
    y_data = sub_s["degradation_rate"].values
    if saturate_100 and y_data[-1] < 100:
        x_data, y_data = force_saturate_100(x_data, y_data, offset=100)
    pchip = PchipInterpolator(x_data, y_data, extrapolate=True)
    def predict_pchip(t):
        t_arr = np.clip(t, 0, None)
        val = pchip(t_arr)
        return np.clip(val, 0, 100)
    return predict_pchip, None

def fit_isotonic(subset, saturate_100=True):
    """
    Fit an isotonic regression to ensure a non-decreasing curve.
    Optionally saturate at 100% if degrade < 100.
    """
    if len(subset) < 2:
        return None, "Not enough data"
    sub_s = subset.sort_values("days").copy()
    x_data = sub_s["days"].values
    y_data = sub_s["degradation_rate"].values
    if x_data[0] > 0:
        x_data = np.insert(x_data, 0, 0)
        y_data = np.insert(y_data, 0, 0)
    if saturate_100 and y_data[-1] < 100:
        x_data, y_data = force_saturate_100(x_data, y_data, offset=100)
    iso = IsotonicRegression(y_min=0, y_max=100, increasing=True, out_of_bounds="clip")
    iso.fit(x_data, y_data)
    def predict_iso(t):
        t_arr = np.clip(t, 0, None)
        return iso.predict(t_arr)
    return predict_iso, None

###############################################################################
# 9) CHECK CURVE & REPEATED X
###############################################################################
def check_curve_quality(t_range, degrade_range, max_jump=30.0):
    """
    Ensure the curve does not have negative slopes or large jumps > max_jump.
    """
    for i in range(len(degrade_range)-1):
        jump = degrade_range[i+1] - degrade_range[i]
        if jump < 0 or jump > max_jump:
            return False
    return True

def repeated_x_values(subset):
    """
    Check if there are repeated 'days' in the subset.
    """
    if len(subset) < 3:
        return False
    counts = subset.groupby("days").size()
    return any(counts > 1)

###############################################################################
# 10) FALLBACK CHAIN
###############################################################################
def fit_auto_model(subset, user_day):
    """
    Model selection fallback chain:
      1) <2 => error
      2) =2 => direct line
      3) repeated x => isotonic
      4) else => Weibull -> Log-Transform -> PCHIP -> Isotonic
    """
    subset = force_zero_start(subset)
    if len(subset) < 2:
        return None, 0, "none", "Not enough data"
    if len(subset) == 2:
        func, err = fit_two_points(subset)
        if err:
            return None, 0, "none", f"2-pt line error: {err}"
        return func, -4, "two_points", None
    if repeated_x_values(subset):
        iso_func, iso_err = fit_isotonic(subset, True)
        if iso_err:
            return None, 0, "none", f"Isotonic error: {iso_err}"
        return iso_func, -3, "isotonic", None

    # Try Weibull
    w_func, w_param, w_err = fit_weibull(subset)
    if not w_err:
        sub_s = subset.sort_values("days")
        max_day_in_data = sub_s["days"].max()
        max_day = max(max_day_in_data, user_day, 60)
        t_range = np.linspace(0, max_day, 100)
        degrade_range = w_func(t_range)
        if check_curve_quality(t_range, degrade_range, 30.0):
            return w_func, w_param, "weibull", None
        w_err = "Weibull shape is weird"

    # Log-Transform poly
    logpoly_func, param_l, err_l = fit_logtransform_poly(subset)
    if not err_l:
        sub_s = subset.sort_values("days")
        max_day_in_data = sub_s["days"].max()
        max_day = max(max_day_in_data, user_day, 60)
        t_range = np.linspace(0, max_day, 100)
        degrade_range = logpoly_func(t_range)
        if check_curve_quality(t_range, degrade_range, 30.0):
            return logpoly_func, param_l, "log_transform_poly", None
        err_l = "Log-transform poly shape is weird"

    # PCHIP
    pchip_func, pchip_err = fit_pchip(subset, saturate_100=False)
    if not pchip_err:
        sub_s = subset.sort_values("days")
        max_day_in_data = sub_s["days"].max()
        max_day = max(max_day_in_data, user_day, 60)
        degrade_range = pchip_func(np.linspace(0, max_day, 100))
        if check_curve_quality(np.linspace(0, max_day, 100), degrade_range, 30.0):
            return pchip_func, -2, "pchip", None
        else:
            iso_func, iso_err = fit_isotonic(subset, True)
            if iso_err:
                return pchip_func, -2, "pchip", f"Isotonic fails: {iso_err}"
            return iso_func, -3, "isotonic", None
    else:
        iso_func, iso_err = fit_isotonic(subset, True)
        if iso_err:
            return None, 0, "none", f"All methods fail: {w_err}, {err_l}, {pchip_err}, {iso_err}"
        return iso_func, -3, "isotonic", None

###############################################################################
# 11) LOG-FIT
###############################################################################
def log_equation(t, a, b, c):
    """Log eq: y = a + b*ln(1 + c*t)."""
    return a + b * np.log(1 + c * t)

def fit_log_to_fallback(fallback_func, max_day):
    """
    Fit a log eq y = a + b ln(1 + c t) to the fallback curve.
    Returns the log function or an error if fitting fails.
    """
    from scipy.optimize import curve_fit
    t_sample = np.linspace(0, max_day, 100)
    degrade_sample = fallback_func(t_sample)
    a_init = min(degrade_sample)
    b_init = 10.0
    c_init = 0.1
    try:
        popt, _ = curve_fit(
            log_equation, t_sample, degrade_sample,
            p0=[a_init, b_init, c_init],
            bounds=([0, 0, 0], [200, 200, 10]),
            maxfev=10000
        )
        def predict_log(t):
            val = log_equation(np.clip(t, 0, None), *popt)
            return np.clip(val, 0, 100)
        return predict_log, None
    except RuntimeError as e:
        return None, f"Log eq error: {e}"

###############################################################################
# NEW: MODEL EVALUATION METRICS
###############################################################################
def compute_metrics(actual, predicted):
    """
    Compute RMSE, R², and MAE between arrays of actual and predicted values.
      - RMSE: Root Mean Squared Error
      - R²: Coefficient of Determination
      - MAE: Mean Absolute Error
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot != 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0
    return rmse, r2, mae

###############################################################################
# 12) STREAMLIT DASHBOARD
###############################################################################
def run_streamlit():
    st.title("Biodegradation Dashboard - Compare Multiple Microorganisms (with Metrics)")
    st.write(
        "Load a CSV of biodegradation data, select a plastic, and compare multiple microorganisms. "
        "Each microorganism's fallback curve is plotted with optional log-fit, and we compute "
        "evaluation metrics (RMSE, R², MAE) to gauge model performance."
    )

    # 1) Load & Clean
    try:
        df = load_and_clean("experimental_data.csv")
        st.success("CSV loaded & cleaned!")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    # 2) Pick a plastic from those with valid data
    plastics = gather_plastics_with_valid_micros(df, threshold=2)
    if not plastics:
        st.warning("No valid plastics found (≥2 data points).")
        return
    plastic_choice = st.selectbox("Select Plastic", plastics, key="plastic_select")

    # 3) Number of microorganisms to compare
    num_micro = st.number_input("Number of Microorganisms to Compare (1-5)", min_value=1, max_value=5, value=1, step=1, key="num_micro")

    # 4) Build list of microorganisms & parameters
    micro_params = []
    micros_options = gather_micros_options(df, plastic_choice, threshold=2)
    if micros_options["Best Biodegradation (Highest avg)"].empty:
        st.warning(f"No microorganisms with ≥2 data points for {plastic_choice}.")
        return
    all_micros = micros_options["Alphabetical"]["microorganism"].tolist()

    for i in range(int(num_micro)):
        st.markdown(f"### Microorganism {i+1} Settings")
        micro = st.selectbox(f"Select Microorganism {i+1}", all_micros, key=f"micro_{i}")
        use_std = st.selectbox(f"Use standard ocean conditions (25°C, pH=8.0)? (Micro {i+1})", ["Yes", "No"], key=f"std_{i}")
        if use_std == "Yes":
            t_temp = 25.0
            t_pH = 8.0
        else:
            t_temp = st.number_input(f"Target Temperature (°C) (Micro {i+1})", value=25.0, step=0.5, key=f"temp_{i}")
            t_pH = st.number_input(f"Target pH (Micro {i+1})", value=8.0, step=0.1, key=f"pH_{i}")
        micro_params.append({
            "micro": micro,
            "use_std": (use_std == "Yes"),
            "t_temp": t_temp,
            "t_pH": t_pH
        })

    # 5) Days to predict
    user_day = st.number_input("Days to Predict", min_value=1.0, value=30.0, step=1.0, key="user_day")

    # 6) Fit fallback curves for each microorganism
    micro_results = []
    for i, params in enumerate(micro_params):
        sub = df[(df["plastic_type"] == plastic_choice) & (df["microorganism"] == params["micro"])]
        # Filter by T/pH if not using standard conditions
        if not params["use_std"]:
            temp_mask = sub["temperature"].between(params["t_temp"] - 5, params["t_temp"] + 5)
            pH_mask = sub["pH"].between(params["t_pH"] - 1, params["t_pH"] + 1)
            sub_filtered = sub[temp_mask & pH_mask]
            if len(sub_filtered) >= 2:
                sub = sub_filtered
            else:
                st.warning(f"Microorganism {i+1}: Not enough data after T/pH filter => using full subset.")
        if len(sub) < 2:
            st.error(f"Microorganism {i+1}: Not enough data (<2). Skipping.")
            micro_results.append(None)
            continue

        fallback_func, param_count, used_model, err = fit_auto_model(sub, user_day)
        if err:
            st.error(f"Microorganism {i+1} fitting error: {err}")
            micro_results.append(None)
            continue

        # Evaluate fallback curve over a range
        sub_sorted = sub.sort_values("days")
        max_day_val = max(sub_sorted["days"].max(), user_day, 60)
        t_range = np.linspace(0, max_day_val, 100)
        degrade_fallback = fallback_func(t_range)
        degrade_fallback[0] = 0  # Force start at 0 degrade
        fallback_pred = fallback_func([user_day])[0]

        # Store the fallback function so we can compute metrics
        micro_results.append({
            "index": i,
            "micro": params["micro"],
            "used_model": used_model,
            "fallback_func": fallback_func,  # <-- store function for metrics
            "t_range": t_range,
            "degrade_fallback": degrade_fallback,
            "fallback_pred": fallback_pred,
            "subset": sub
        })

    micro_results = [mr for mr in micro_results if mr is not None]
    if not micro_results:
        st.error("No valid microorganisms after filtering. Exiting.")
        return

    # 7) Show each fallback curve, compute & display metrics
    for mr in micro_results:
        idx = mr["index"]
        st.markdown(f"#### Microorganism {idx+1}: {mr['micro']} - Fallback Curve")

        # 7a) Compute metrics on the fallback curve
        sub_s = mr["subset"].sort_values("days")
        x_vals = sub_s["days"].values
        y_actual = sub_s["degradation_rate"].values
        y_predicted = mr["fallback_func"](x_vals)  # Key fix: we stored fallback_func
        rmse, r2, mae = compute_metrics(y_actual, y_predicted)

        # 7b) Plot the fallback curve
        fig_fallback = px.scatter(
            x=sub_s["days"],
            y=sub_s["degradation_rate"],
            labels={"x": "Days", "y": "% Biodegraded"},
            title=f"Fallback Model: {plastic_choice} + {mr['micro']} (model={mr['used_model']})"
        )
        fig_fallback.update_traces(mode="markers", marker=dict(color="blue"), name="Experimental Data")
        fig_fallback.add_scatter(
            x=mr["t_range"],
            y=mr["degrade_fallback"],
            mode="lines",
            line=dict(color="red"),
            name=f"{mr['used_model']} curve"
        )
        fig_fallback.add_scatter(
            x=[user_day],
            y=[mr["fallback_pred"]],
            mode="markers",
            marker=dict(size=10, color="green", symbol="x"),
            name=f"Predict => {mr['fallback_pred']:.2f}%"
        )
        st.plotly_chart(fig_fallback, key=f"fallback_chart_{idx}")

        # 7c) Display fallback metrics
        st.write(
            f"**Fallback Metrics for {mr['micro']}:** "
            f"RMSE = {rmse:.2f}, R² = {r2:.2f}, MAE = {mae:.2f}"
        )

        # 7d) Optional log fit
        do_log = st.checkbox(f"Fit a log equation for Micro {idx+1}?", key=f"log_{idx}")
        if do_log:
            log_func, log_err = fit_log_to_fallback(
                lambda t: np.interp(t, mr["t_range"], mr["degrade_fallback"]), 
                mr["t_range"][-1]
            )
            if log_err:
                st.error(f"Log eq error for Micro {idx+1}: {log_err}")
            else:
                degrade_log = log_func(mr["t_range"])
                degrade_log[0] = 0
                log_pred = log_func([user_day])[0]

                # Compute metrics for the log-fit
                y_log_predicted = log_func(x_vals)
                rmse_log, r2_log, mae_log = compute_metrics(y_actual, y_log_predicted)

                fig_log = px.scatter(
                    x=sub_s["days"],
                    y=sub_s["degradation_rate"],
                    labels={"x": "Days", "y": "% Biodegraded"},
                    title=f"Log Fit to Fallback Curve ({mr['micro']})"
                )
                fig_log.update_traces(mode="markers", marker=dict(color="blue"), name="Experimental Data")
                # Show fallback curve in dashed line for comparison
                fig_log.add_scatter(
                    x=mr["t_range"],
                    y=mr["degrade_fallback"],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name=f"{mr['used_model']} fallback"
                )
                fig_log.add_scatter(
                    x=mr["t_range"],
                    y=degrade_log,
                    mode="lines",
                    line=dict(color="magenta"),
                    name="Log eq approx"
                )
                fig_log.add_scatter(
                    x=[user_day],
                    y=[log_pred],
                    mode="markers",
                    marker=dict(size=10, color="green", symbol="x"),
                    name=f"log => {log_pred:.2f}%"
                )
                st.plotly_chart(fig_log, key=f"log_chart_{idx}")

                st.write(
                    f"**Log-Fit Metrics for {mr['micro']}:** "
                    f"RMSE = {rmse_log:.2f}, R² = {r2_log:.2f}, MAE = {mae_log:.2f}"
                )

    # 8) Final overlay graph: fallback or log
    st.markdown("### Final Overlay Graph")
    use_log_overlay = st.checkbox("Use log-fit curves in the overlay instead of fallback curves?")
    fig_overlay = px.line()
    fig_overlay.update_layout(
        title="Overlay of All Microorganisms",
        xaxis_title="Days",
        yaxis_title="% Biodegraded"
    )
    best_pred_val = -1
    best_micro = None

    for mr in micro_results:
        if use_log_overlay:
            log_func, log_err = fit_log_to_fallback(
                lambda t: np.interp(t, mr["t_range"], mr["degrade_fallback"]), 
                mr["t_range"][-1]
            )
            if log_err:
                st.error(f"Log eq error in overlay for {mr['micro']}: {log_err}")
                curve_y = mr["degrade_fallback"]
                pred_val = mr["fallback_pred"]
            else:
                curve_y = log_func(mr["t_range"])
                pred_val = log_func([user_day])[0]
        else:
            curve_y = mr["degrade_fallback"]
            pred_val = mr["fallback_pred"]
        curve_y[0] = 0
        fig_overlay.add_scatter(
            x=mr["t_range"],
            y=curve_y,
            mode="lines",
            name=f"{mr['micro']}"
        )
        fig_overlay.add_scatter(
            x=[user_day],
            y=[pred_val],
            mode="markers",
            marker=dict(size=10),
            name=f"{mr['micro']} => {pred_val:.2f}%"
        )
        if pred_val > best_pred_val:
            best_pred_val = pred_val
            best_micro = mr["micro"]

    # Set overlay axes
    max_x = max([mr["t_range"][-1] for mr in micro_results] + [user_day]) + 2
    fig_overlay.update_xaxes(range=[0, max_x])
    fig_overlay.update_yaxes(range=[0, 105])
    st.plotly_chart(fig_overlay, key="final_overlay_chart")

    # 9) Summarize which microorganism is best
    if best_micro:
        st.markdown("### Which microorganism is best?")
        st.write(
            f"At day={user_day:.1f}, **{best_micro}** achieves the highest predicted "
            f"degradation of **{best_pred_val:.2f}%** among the compared microorganisms "
            f"for plastic **{plastic_choice}**."
        )
    else:
        st.write("No best microorganism determined.")

def main():
    setup_logging()
    run_streamlit()

if __name__ == "__main__":
    main()