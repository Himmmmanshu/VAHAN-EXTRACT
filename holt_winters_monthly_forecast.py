"""
HOLT-WINTERS MONTHLY AUTOMOBILE SALES FORECAST
=============================================

Forecasts the next 6 months using Holt-Winters triple exponential smoothing.

Input:
  Excel or CSV containing monthly sales data. Supported layouts:
    1) MONTH, YEAR, Sum of Value
    2) Row Label (e.g. Jan_21), Sum of Value
    3) Date, Sales (or similarly named columns)

The script:
  - detects the date and sales columns (or lets you configure them below)
  - parses and sorts monthly observations chronologically
  - aggregates duplicate months
  - checks for missing months and non-positive values
  - decomposes the series into trend/seasonal/residual components
  - evaluates additive and multiplicative Holt-Winters models
  - selects the model with the lower test RMSE
  - refits the selected model on all observations
  - forecasts the next 6 months
  - saves charts and an Excel workbook

Install:
  pip install pandas numpy matplotlib seaborn statsmodels scikit-learn openpyxl

Run:
  python holt_winters_monthly_forecast.py

IMPORTANT:
  Set INPUT_FILE below to your actual workbook path.
  Missing months are NOT silently interpolated. The script stops and lists
  them, because a missing month is not necessarily a zero-sales month.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# =============================================================================
# 1. USER SETTINGS - CHANGE THESE
# =============================================================================

INPUT_FILE = r"C:\Users\YourName\Desktop\Automobile_Sales.xlsx"

# If automatic detection does not identify your columns, enter exact names:
# For the screenshot structure, these are likely:
DATE_COLUMN = None       # e.g. "Row Label" OR leave None for automatic detection
SALES_COLUMN = None      # e.g. "Sum of Value" OR leave None for automatic detection
MONTH_COLUMN = "MONTH"   # used when MONTH and YEAR columns exist
YEAR_COLUMN = "YEAR"

FORECAST_MONTHS = 6
TEST_MONTHS = 12
SEASONAL_PERIODS = 12

# Missing-month policy:
# "stop"      = recommended; script stops and lists missing months
# "zero"      = use only if a missing month genuinely means zero sales
# "interpolate" = use only if interpolation is analytically justified
MISSING_MONTH_POLICY = "stop"

OUTPUT_DIR = "holt_winters_output"


# =============================================================================
# 2. HELPERS
# =============================================================================

def find_column(columns, candidates):
    """Find a column by case-insensitive exact candidate match."""
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def parse_monthly_dates(data):
    """Create a monthly Date column from common Excel layouts."""
    cols = data.columns

    # Explicitly configured date column takes priority.
    if DATE_COLUMN is not None:
        if DATE_COLUMN not in cols:
            raise ValueError(
                f"DATE_COLUMN='{DATE_COLUMN}' was not found. "
                f"Available columns: {list(cols)}"
            )
        raw = data[DATE_COLUMN].astype(str).str.strip()

        # Screenshot format: Jan_21
        dates = pd.to_datetime(raw, format="%b_%y", errors="coerce")

        # Other common date formats
        if dates.isna().all():
            dates = pd.to_datetime(raw, errors="coerce", dayfirst=True)

        return dates

    # Screenshot layout: MONTH and YEAR
    month_col = find_column(cols, [MONTH_COLUMN, "Month", "MONTH"])
    year_col = find_column(cols, [YEAR_COLUMN, "Year", "YEAR"])

    if month_col is not None and year_col is not None:
        month_text = data[month_col].astype(str).str.strip()
        year_text = data[year_col].astype(str).str.strip()
        dates = pd.to_datetime(
            month_text + " " + year_text,
            format="%b %Y",
            errors="coerce"
        )

        # Support full month names too.
        if dates.isna().any():
            fallback = pd.to_datetime(
                month_text + " " + year_text,
                errors="coerce"
            )
            dates = dates.fillna(fallback)

        return dates

    # Row Label format: Jan_21
    row_label_col = find_column(
        cols, ["Row Label", "Row Labels", "Month-Year", "Month Year"]
    )
    if row_label_col is not None:
        raw = data[row_label_col].astype(str).str.strip()
        dates = pd.to_datetime(raw, format="%b_%y", errors="coerce")
        if dates.isna().all():
            dates = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        return dates

    # Generic date-like columns
    generic_date_col = find_column(
        cols, ["Date", "Month", "Period", "年月"]
    )
    if generic_date_col is not None:
        return pd.to_datetime(
            data[generic_date_col], errors="coerce", dayfirst=True
        )

    raise ValueError(
        "Could not identify the date columns. Configure DATE_COLUMN, "
        "or provide MONTH + YEAR, or Row Label (e.g. Jan_21)."
    )


def calculate_metrics(actual, predicted):
    """Return common forecast error metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero])
                       / actual[nonzero])) * 100
        if nonzero.any() else np.nan
    )

    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator != 0
    smape = (
        np.mean(200 * np.abs(actual[valid] - predicted[valid])
                / denominator[valid])
        if valid.any() else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "SMAPE_percent": smape,
    }


def fit_holt_winters(series, seasonal_type):
    """Fit Holt-Winters with additive trend and damped trend."""
    model = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal=seasonal_type,
        seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / filename, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# 3. LOAD FILE
# =============================================================================

INPUT_PATH = Path(INPUT_FILE)
OUTPUT_PATH = Path(OUTPUT_DIR)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}\n"
        "Update INPUT_FILE at the top of this script."
    )

if INPUT_PATH.suffix.lower() in [".xlsx", ".xls"]:
    raw_df = pd.read_excel(INPUT_PATH)
elif INPUT_PATH.suffix.lower() == ".csv":
    raw_df = pd.read_csv(INPUT_PATH)
else:
    raise ValueError("Input file must be .xlsx, .xls, or .csv")

raw_df.columns = raw_df.columns.astype(str).str.strip()

print("\n========== INPUT DATA ==========")
print("Rows and columns:", raw_df.shape)
print("Columns:", raw_df.columns.tolist())
print(raw_df.head())


# =============================================================================
# 4. IDENTIFY SALES COLUMN AND BUILD DATE
# =============================================================================

if SALES_COLUMN is not None:
    if SALES_COLUMN not in raw_df.columns:
        raise ValueError(
            f"SALES_COLUMN='{SALES_COLUMN}' not found. "
            f"Available columns: {raw_df.columns.tolist()}"
        )
    sales_col = SALES_COLUMN
else:
    sales_col = find_column(
        raw_df.columns,
        [
            "Sum of Value", "Sales", "Volume", "Units",
            "Total Sales", "Value", "Sum of Sales"
        ]
    )

if sales_col is None:
    raise ValueError(
        "Could not identify the sales column. Set SALES_COLUMN at the top "
        "of the script to the exact column name (for your screenshot, "
        "likely 'Sum of Value')."
    )

dates = parse_monthly_dates(raw_df)

work = pd.DataFrame({
    "Date": dates,
    "Sales": pd.to_numeric(
        raw_df[sales_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )
})

work = work.dropna(subset=["Date", "Sales"]).copy()

if work.empty:
    raise ValueError("No valid date and sales records were found.")

# Convert every date to the first day of its month.
work["Date"] = work["Date"].dt.to_period("M").dt.to_timestamp()

# Aggregate duplicate monthly records. This is useful if the source has
# multiple rows for the same month; if each row is already a monthly total,
# there should normally be no duplicate months.
monthly = (
    work.groupby("Date", as_index=True)["Sales"]
    .sum()
    .sort_index()
    .astype(float)
)

monthly.index = pd.DatetimeIndex(monthly.index, freq="MS")
monthly.index.name = "Date"


# =============================================================================
# 5. CHECK MONTHLY CONTINUITY
# =============================================================================

expected_index = pd.date_range(
    start=monthly.index.min(),
    end=monthly.index.max(),
    freq="MS"
)

missing_months = expected_index.difference(monthly.index)

print("\n========== DATA QUALITY CHECKS ==========")
print("First month:", monthly.index.min().strftime("%b-%Y"))
print("Last month:", monthly.index.max().strftime("%b-%Y"))
print("Observed monthly records:", len(monthly))
print("Missing months:", len(missing_months))

if len(missing_months) > 0:
    print("Missing month list:")
    for missing_date in missing_months:
        print(" -", missing_date.strftime("%b-%Y"))

    if MISSING_MONTH_POLICY == "stop":
        raise ValueError(
            "Missing months detected. Check the source data. If a missing "
            "month represents zero sales, set MISSING_MONTH_POLICY='zero'. "
            "If interpolation is justified, set it to 'interpolate'."
        )

    monthly = monthly.reindex(expected_index)

    if MISSING_MONTH_POLICY == "zero":
        monthly = monthly.fillna(0)
    elif MISSING_MONTH_POLICY == "interpolate":
        monthly = monthly.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError("Invalid MISSING_MONTH_POLICY setting.")

if len(monthly) < 2 * SEASONAL_PERIODS + TEST_MONTHS:
    raise ValueError(
        "Not enough monthly observations for a 12-month seasonal model "
        "and the selected test period. Provide more history or reduce "
        "TEST_MONTHS carefully."
    )

if monthly.isna().any():
    raise ValueError("Missing sales values remain after preprocessing.")

if (monthly < 0).any():
    raise ValueError("Negative sales values found. Investigate source data.")

print("\nClean monthly data:")
print(monthly.head(15))
print("\nLast observations:")
print(monthly.tail(15))


# =============================================================================
# 6. HISTORICAL SALES PLOT
# =============================================================================

plt.figure(figsize=(15, 6))
plt.plot(monthly.index, monthly.values, linewidth=2, label="Monthly sales")
plt.title("Monthly Automobile Sales - Historical Trend")
plt.xlabel("Month")
plt.ylabel("Sales volume")
plt.legend()
plt.grid(alpha=0.3)
save_plot("01_historical_monthly_sales.png")


# =============================================================================
# 7. ROLLING MEAN AND STANDARD DEVIATION
# =============================================================================

rolling_mean = monthly.rolling(SEASONAL_PERIODS).mean()
rolling_std = monthly.rolling(SEASONAL_PERIODS).std()

plt.figure(figsize=(15, 6))
plt.plot(monthly.index, monthly, label="Actual sales", alpha=0.65)
plt.plot(rolling_mean.index, rolling_mean, label="12-month rolling mean")
plt.plot(rolling_std.index, rolling_std, label="12-month rolling std")
plt.title("Rolling Mean and Standard Deviation")
plt.xlabel("Month")
plt.ylabel("Sales volume")
plt.legend()
plt.grid(alpha=0.3)
save_plot("02_rolling_statistics.png")


# =============================================================================
# 8. ADDITIVE AND MULTIPLICATIVE DECOMPOSITION
# =============================================================================

add_decomp = seasonal_decompose(
    monthly,
    model="additive",
    period=SEASONAL_PERIODS,
    extrapolate_trend="freq"
)

fig = add_decomp.plot()
fig.set_size_inches(14, 9)
fig.suptitle("Additive Seasonal Decomposition", y=1.02)
save_plot("03_additive_decomposition.png")

mul_decomp = None
if (monthly > 0).all():
    mul_decomp = seasonal_decompose(
        monthly,
        model="multiplicative",
        period=SEASONAL_PERIODS,
        extrapolate_trend="freq"
    )

    fig = mul_decomp.plot()
    fig.set_size_inches(14, 9)
    fig.suptitle("Multiplicative Seasonal Decomposition", y=1.02)
    save_plot("04_multiplicative_decomposition.png")
else:
    print(
        "\nMultiplicative decomposition/model unavailable because "
        "the series contains zero or negative values."
    )


# =============================================================================
# 9. COMPARE MONTH-OF-YEAR SEASONAL PATTERNS
# =============================================================================

seasonal_by_month = pd.DataFrame({
    "Additive seasonal effect": add_decomp.seasonal.groupby(
        add_decomp.seasonal.index.month
    ).mean()
})

if mul_decomp is not None:
    seasonal_by_month["Multiplicative seasonal factor"] = (
        mul_decomp.seasonal.groupby(
            mul_decomp.seasonal.index.month
        ).mean()
    )

seasonal_by_month.index = [
    pd.Timestamp(2000, month, 1).strftime("%b")
    for month in seasonal_by_month.index
]
seasonal_by_month.index.name = "Month"

fig, ax1 = plt.subplots(figsize=(13, 6))
seasonal_by_month["Additive seasonal effect"].plot(
    kind="bar", ax=ax1, alpha=0.8
)
ax1.set_title("Average Additive Seasonal Effect by Calendar Month")
ax1.set_xlabel("Calendar month")
ax1.set_ylabel("Additive effect (sales units)")
ax1.grid(axis="y", alpha=0.3)
save_plot("05_additive_seasonality_by_month.png")

if mul_decomp is not None:
    plt.figure(figsize=(13, 6))
    seasonal_by_month["Multiplicative seasonal factor"].plot(
        kind="bar", alpha=0.8
    )
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.title("Average Multiplicative Seasonal Factor by Calendar Month")
    plt.xlabel("Calendar month")
    plt.ylabel("Seasonal factor (1.0 = average)")
    plt.grid(axis="y", alpha=0.3)
    save_plot("06_multiplicative_seasonality_by_month.png")


# =============================================================================
# 10. TRAIN-TEST SPLIT - LAST 12 MONTHS
# =============================================================================

train = monthly.iloc[:-TEST_MONTHS]
test = monthly.iloc[-TEST_MONTHS:]

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training:", train.index.min().strftime("%b-%Y"),
      "to", train.index.max().strftime("%b-%Y"),
      "| observations:", len(train))
print("Testing:", test.index.min().strftime("%b-%Y"),
      "to", test.index.max().strftime("%b-%Y"),
      "| observations:", len(test))


# =============================================================================
# 11. FIT AND EVALUATE ADDITIVE MODEL
# =============================================================================

model_results = {}
test_forecasts = {}
fitted_test_models = {}

add_test_model = fit_holt_winters(train, "add")
add_test_prediction = add_test_model.forecast(TEST_MONTHS)

model_results["Additive"] = calculate_metrics(
    test, add_test_prediction
)
test_forecasts["Additive"] = add_test_prediction
fitted_test_models["Additive"] = add_test_model


# =============================================================================
# 12. FIT AND EVALUATE MULTIPLICATIVE MODEL
# =============================================================================

if (train > 0).all():
    try:
        mul_test_model = fit_holt_winters(train, "mul")
        mul_test_prediction = mul_test_model.forecast(TEST_MONTHS)

        model_results["Multiplicative"] = calculate_metrics(
            test, mul_test_prediction
        )
        test_forecasts["Multiplicative"] = mul_test_prediction
        fitted_test_models["Multiplicative"] = mul_test_model

    except Exception as exc:
        print("\nMultiplicative model fitting failed:", exc)
else:
    print("\nMultiplicative model skipped: training data has zero values.")


# =============================================================================
# 13. MODEL COMPARISON TABLE
# =============================================================================

metrics_df = pd.DataFrame(model_results).T
metrics_df = metrics_df.sort_values("RMSE")

print("\n========== MODEL PERFORMANCE ==========")
print(metrics_df.round(3))

best_model_name = metrics_df["RMSE"].idxmin()
best_seasonal_type = (
    "add" if best_model_name == "Additive" else "mul"
)

print("\nSelected model based on lowest test RMSE:", best_model_name)


# =============================================================================
# 14. ACTUAL VS TEST-PERIOD PREDICTIONS
# =============================================================================

plt.figure(figsize=(15, 7))
plt.plot(train.index, train, label="Training actual", linewidth=1.8)
plt.plot(test.index, test, label="Test actual", linewidth=2.2)

for model_name, prediction in test_forecasts.items():
    plt.plot(
        prediction.index,
        prediction.values,
        linestyle="--",
        marker="o",
        label=f"{model_name} prediction"
    )

plt.title("Test-Period Forecast: Actual vs Holt-Winters Predictions")
plt.xlabel("Month")
plt.ylabel("Sales volume")
plt.legend()
plt.grid(alpha=0.3)
save_plot("07_test_actual_vs_predicted.png")


# =============================================================================
# 15. REFIT BOTH MODELS ON FULL DATA AND FORECAST SIX MONTHS
# =============================================================================

add_full_model = fit_holt_winters(monthly, "add")
add_future = add_full_model.forecast(FORECAST_MONTHS)

mul_full_model = None
mul_future = None

if (monthly > 0).all():
    try:
        mul_full_model = fit_holt_winters(monthly, "mul")
        mul_future = mul_full_model.forecast(FORECAST_MONTHS)
    except Exception as exc:
        print("\nFull-data multiplicative model failed:", exc)

if best_model_name == "Additive":
    final_model = add_full_model
    final_forecast = add_future
else:
    if mul_full_model is None:
        print("Multiplicative refit unavailable; using additive model.")
        best_model_name = "Additive"
        final_model = add_full_model
        final_forecast = add_future
    else:
        final_model = mul_full_model
        final_forecast = mul_future

# Forecast volumes cannot be negative.
final_forecast = final_forecast.clip(lower=0)

forecast_output = pd.DataFrame({
    "Forecast_Date": final_forecast.index,
    "Forecast_Sales": final_forecast.values
})
forecast_output["Forecast_Month"] = (
    forecast_output["Forecast_Date"].dt.strftime("%b-%Y")
)
forecast_output = forecast_output[
    ["Forecast_Date", "Forecast_Month", "Forecast_Sales"]
]

print("\n========== SIX-MONTH FORECAST ==========")
print("Selected model:", best_model_name)
print(forecast_output.to_string(index=False))


# =============================================================================
# 16. FORECAST COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(15, 7))
plt.plot(
    monthly.index, monthly.values,
    label="Historical actual sales", linewidth=2
)
plt.plot(
    add_future.index, add_future.values,
    label="Additive forecast", marker="o", linestyle="--"
)

if mul_future is not None:
    plt.plot(
        mul_future.index, mul_future.values,
        label="Multiplicative forecast", marker="o", linestyle="--"
    )

plt.axvline(
    monthly.index[-1],
    linestyle=":",
    linewidth=1.5,
    label="Forecast origin"
)
plt.title("Six-Month Holt-Winters Forecast: Additive vs Multiplicative")
plt.xlabel("Month")
plt.ylabel("Sales volume")
plt.legend()
plt.grid(alpha=0.3)
save_plot("08_six_month_forecast_comparison.png")


# =============================================================================
# 17. SELECTED MODEL - HISTORICAL + FUTURE FORECAST GRAPH
# =============================================================================

plt.figure(figsize=(15, 7))
plt.plot(
    monthly.index, monthly.values,
    label="Historical actual sales", linewidth=2
)
plt.plot(
    final_forecast.index, final_forecast.values,
    label=f"{best_model_name} six-month forecast",
    marker="o", linewidth=2, linestyle="--"
)
plt.axvline(
    monthly.index[-1],
    linestyle=":",
    linewidth=1.5,
    label="Forecast origin"
)
plt.title(f"Final Holt-Winters {best_model_name} Forecast")
plt.xlabel("Month")
plt.ylabel("Sales volume")
plt.legend()
plt.grid(alpha=0.3)
save_plot("09_final_forecast.png")


# =============================================================================
# 18. RESIDUAL DIAGNOSTICS
# =============================================================================

residuals = pd.Series(
    np.asarray(final_model.resid),
    index=monthly.index[-len(final_model.resid):],
    name="Residual"
)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(residuals.index, residuals.values)
axes[0].axhline(0, linestyle="--", linewidth=1)
axes[0].set_title("Final Model Residuals Over Time")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Residual")
axes[0].grid(alpha=0.3)

sns.histplot(residuals.dropna(), kde=True, ax=axes[1])
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual")

save_plot("10_residual_diagnostics.png")


# =============================================================================
# 19. EXPORT OUTPUTS TO EXCEL
# =============================================================================

historical_output = monthly.rename("Actual_Sales").reset_index()

test_output = pd.DataFrame({
    "Actual_Sales": test,
})
for model_name, prediction in test_forecasts.items():
    test_output[f"{model_name}_Prediction"] = prediction

test_output["Selected_Model"] = best_model_name

forecast_comparison = pd.DataFrame({
    "Additive_Forecast": add_future
})
if mul_future is not None:
    forecast_comparison["Multiplicative_Forecast"] = mul_future
forecast_comparison["Selected_Model"] = best_model_name
forecast_comparison["Selected_Forecast"] = final_forecast

output_excel = OUTPUT_PATH / "Holt_Winters_Forecast_Results.xlsx"

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    historical_output.to_excel(
        writer, sheet_name="Historical Monthly Data", index=False
    )
    metrics_df.to_excel(
        writer, sheet_name="Model Evaluation"
    )
    test_output.to_excel(
        writer, sheet_name="Test Predictions"
    )
    forecast_output.to_excel(
        writer, sheet_name="6 Month Forecast", index=False
    )
    forecast_comparison.to_excel(
        writer, sheet_name="Forecast Comparison"
    )
    seasonal_by_month.to_excel(
        writer, sheet_name="Seasonality"
    )

print("\n========== COMPLETED ==========")
print("Selected model:", best_model_name)
print("Excel output:", output_excel.resolve())
print("Charts folder:", OUTPUT_PATH.resolve())
print("\nForecasting process completed successfully.")
