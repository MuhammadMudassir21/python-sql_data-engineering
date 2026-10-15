import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

df = pd.read_csv("supermarket_sales.csv")
rows_loaded = len(df)

print("Rows loaded:", rows_loaded)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)

required_columns = [
    "Invoice ID", "Branch", "Gender", "Payment",
    "Customer type", "Quantity", "Unit price",
    "Total", "Date", "Time", "Rating"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(
        f"CSV is missing these columns: {missing_columns}"
    )

# ==========================================
# STEP 2: AUDIT DATA
# ==========================================

print("\nMissing values:\n", df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())
print("\nGender values:", df["Gender"].unique())
print("\nPayment values:", df["Payment"].unique())

duplicate_invoice_rows = df.duplicated(
    subset=["Invoice ID"], keep=False
).sum()

print("Rows sharing Invoice ID:", duplicate_invoice_rows)

# ==========================================
# STEP 3: REMOVE EXACT DUPLICATES
# ==========================================

df = df.drop_duplicates().copy()

# ==========================================
# STEP 4: RESOLVE CONFLICTING INVOICE IDS
# Keep the last row because it is the correction.
# ==========================================

conflicts_resolved = int(
    df.duplicated(subset=["Invoice ID"], keep=False).sum()
)

df = df.drop_duplicates(
    subset=["Invoice ID"], keep="last"
).copy()

# ==========================================
# STEP 5: STANDARDISE TEXT
# ==========================================

def clean_text(value):
    if pd.isna(value):
        return np.nan
    return str(value).strip().lower()

for col in ["Gender", "Payment"]:
    df[col] = df[col].apply(clean_text)

gender_map = {
    "male": "Male",
    "female": "Female"
}

payment_map = {
    "cash": "Cash",
    "credit card": "Credit card",
    "creditcard": "Credit card",
    "ewallet": "Ewallet",
    "e-wallet": "Ewallet"
}

df["Gender"] = df["Gender"].map(
    lambda x: gender_map.get(x, x) if pd.notna(x) else np.nan
)

df["Payment"] = df["Payment"].map(
    lambda x: payment_map.get(x, x) if pd.notna(x) else np.nan
)

# ==========================================
# STEP 6: FIX NEGATIVE QUANTITIES
# ==========================================

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

negative_quantities_fixed = int((df["Quantity"] < 0).sum())

df["Quantity"] = df["Quantity"].abs()

# ==========================================
# STEP 7: PARSE MIXED DATES
# ==========================================

def parse_date(value):
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    for fmt in ("%m/%d/%Y", "%d-%m-%Y"):
        result = pd.to_datetime(
            value, format=fmt, errors="coerce"
        )
        if pd.notna(result):
            return result

    return pd.NaT

df["Date"] = df["Date"].apply(parse_date)

# ==========================================
# STEP 8: FIX TOTAL AND TAX ERRORS
# Total = Quantity * Unit price * 1.05
# ==========================================

df["Unit price"] = pd.to_numeric(
    df["Unit price"], errors="coerce"
)

df["Total"] = pd.to_numeric(
    df["Total"], errors="coerce"
)

calculated_total = (
    df["Quantity"] * df["Unit price"] * 1.05
)

difference = (df["Total"] - calculated_total).abs()

fix_total = (
    (difference > 0.50) | df["Total"].isna()
) & calculated_total.notna()

total_tax_errors_fixed = int(fix_total.sum())

df.loc[fix_total, "Total"] = calculated_total[fix_total]

# ==========================================
# STEP 9: HANDLE MISSING VALUES
# ==========================================

df["Customer type"] = df["Customer type"].fillna("Unknown")

df["Rating"] = pd.to_numeric(
    df["Rating"], errors="coerce"
)

median_rating = df["Rating"].median()

if pd.notna(median_rating):
    df["Rating"] = df["Rating"].fillna(median_rating)

rows_before_date_removal = len(df)

df = df.dropna(subset=["Date"]).copy()

rows_removed_for_missing_dates = (
    rows_before_date_removal - len(df)
)

# ==========================================
# STEP 10: CREATE TIME COLUMNS
# ==========================================

df["Year"] = df["Date"].dt.year

df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)

df["Day of week"] = df["Date"].dt.day_name()

df["Hour"] = pd.to_datetime(
    df["Time"].astype(str).str.strip(),
    format="%H:%M",
    errors="coerce"
).dt.hour

# ==========================================
# STEP 11: TRADING PERIODS
# Morning 10-13, Afternoon 13-17, Evening 17-21
# ==========================================

df["Trading period"] = np.select(
    [
        df["Hour"].between(10, 12),
        df["Hour"].between(13, 16),
        df["Hour"].between(17, 20)
    ],
    ["Morning", "Afternoon", "Evening"],
    default="Outside hours"
)

period_hours = {
    "Morning": 3,
    "Afternoon": 4,
    "Evening": 4
}

period_summary = (
    df[df["Trading period"].isin(period_hours)]
    .groupby("Trading period")
    .agg(Transactions=("Invoice ID", "count"))
    .reset_index()
)

period_summary["Hours"] = period_summary["Trading period"].map(
    period_hours
)

period_summary["Transactions per hour"] = (
    period_summary["Transactions"] / period_summary["Hours"]
)

# ==========================================
# STEP 12: QUARTERLY BRANCH REVENUE
# ==========================================

branch_quarter = (
    df.groupby(["Branch", "Quarter"])["Total"]
    .sum()
    .reset_index(name="Revenue")
)

branch_quarter["Target"] = 100000

branch_quarter["Variance"] = (
    branch_quarter["Revenue"] - branch_quarter["Target"]
)

branch_quarter["Percent of target"] = (
    branch_quarter["Revenue"] /
    branch_quarter["Target"] * 100
)

branch_quarter["Hit"] = (
    branch_quarter["Revenue"] >= branch_quarter["Target"]
)

# ==========================================
# STEP 13: BRANCH SUMMARY
# ==========================================

branch_summary = (
    branch_quarter.groupby("Branch")
    .agg(
        **{
            "Average quarterly revenue": ("Revenue", "mean"),
            "Quarters hit": ("Hit", "sum"),
            "Best quarter revenue": ("Revenue", "max"),
            "Worst quarter revenue": ("Revenue", "min"),
            "Average target %": ("Percent of target", "mean")
        }
    )
    .reset_index()
)

# Yearly revenue and growth
yearly = (
    df[df["Year"].isin([2023, 2024])]
    .groupby(["Branch", "Year"])["Total"]
    .sum()
    .unstack()
)

# FIX: Create missing year columns to avoid KeyError: 2024
for year in [2023, 2024]:
    if year not in yearly.columns:
        yearly[year] = np.nan

yearly["YoY growth %"] = np.where(
    yearly[2023].notna()
    & yearly[2024].notna()
    & (yearly[2023] != 0),
    (yearly[2024] / yearly[2023] - 1) * 100,
    np.nan
)

yearly_summary = (
    yearly[["YoY growth %"]]
    .reset_index()
)

branch_summary = branch_summary.merge(
    yearly_summary, on="Branch", how="left"
)

# ==========================================
# STEP 14: BUSIEST DAY OF WEEK
# Average transactions per weekday in the data
# ==========================================

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_transactions = (
    df["Day of week"]
    .value_counts()
    .reindex(weekday_order, fill_value=0)
)

# Count calendar days represented in the data range
calendar_days = pd.date_range(
    df["Date"].min(), df["Date"].max(), freq="D"
)

calendar_weekdays = pd.Series(
    calendar_days.day_name()
).value_counts().reindex(weekday_order, fill_value=0)

weekday_summary = pd.DataFrame({
    "Transactions": weekday_transactions,
    "Days in date range": calendar_weekdays
})

weekday_summary["Average per day"] = (
    weekday_summary["Transactions"] /
    weekday_summary["Days in date range"].replace(0, np.nan)
)

# ==========================================
# STEP 15: HEATMAP
# ==========================================

heatmap_data = (
    df.groupby(["Day of week", "Hour"])["Invoice ID"]
    .count()
    .unstack(fill_value=0)
    .reindex(weekday_order)
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="g",
    cmap="YlOrRd"
)

plt.title("Transactions by Weekday and Hour")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.savefig("busy_times_heatmap.png")
plt.show()

# ==========================================
# STEP 16: QUARTERLY REVENUE CHARTS
# ==========================================

for branch in branch_quarter["Branch"].unique():

    branch_data = branch_quarter[
        branch_quarter["Branch"] == branch
    ].sort_values("Quarter")

    plt.figure(figsize=(10, 5))

    plt.plot(
        branch_data["Quarter"],
        branch_data["Revenue"],
        marker="o",
        label="Revenue"
    )

    plt.axhline(
        y=100000,
        linestyle="--",
        label="Target: $100,000"
    )

    plt.title(f"Quarterly Revenue - Branch {branch}")
    plt.xlabel("Quarter")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"branch_{branch}_quarterly_revenue.png")
    plt.show()

# ==========================================
# STEP 17: EXPORT CLEAN DATA AND REPORTS
# ==========================================

df.to_csv("supermarket_cleaned.csv", index=False)

branch_quarter.to_csv(
    "branch_quarterly_revenue.csv", index=False
)

branch_summary.to_csv(
    "branch_summary.csv", index=False
)

weekday_summary.to_csv(
    "weekday_summary.csv"
)

period_summary.to_csv(
    "trading_period_summary.csv", index=False
)

# ==========================================
# STEP 18: FINAL RESULT
# ==========================================

print("\n" + "=" * 65)
print("SUPERMARKET CASE STUDY - FINAL RESULT")
print("=" * 65)

print(f"Rows loaded: {rows_loaded:,}")

print(
    f"Rows after cleaning: {len(df):,} "
    f"({len(df) / rows_loaded * 100:.1f}% kept)"
)

print(
    f"Trustworthy total revenue: ${df['Total'].sum():,.2f} "
    f"({df['Date'].min():%Y-%m-%d} to "
    f"{df['Date'].max():%Y-%m-%d})"
)

if not branch_summary.empty:
    best = branch_summary.loc[
        branch_summary["Average target %"].idxmax()
    ]

    worst = branch_summary.loc[
        branch_summary["Average target %"].idxmin()
    ]

    print(
        f"Best branch (avg. quarter): {best['Branch']} "
        f"at {best['Average target %']:.1f}% of target, "
        f"hit {int(best['Quarters hit'])} quarters"
    )

    print(
        f"Weakest branch: {worst['Branch']} "
        f"at {worst['Average target %']:.1f}% of target, "
        f"hit {int(worst['Quarters hit'])} quarters"
    )

if weekday_summary["Average per day"].notna().any():
    busiest_day = weekday_summary["Average per day"].idxmax()
    print(f"Busiest day of week: {busiest_day}")

if not period_summary.empty:
    busiest_period = period_summary.loc[
        period_summary["Transactions per hour"].idxmax(),
        "Trading period"
    ]
    print(f"Busiest time of day: {busiest_period} (per trading hour)")

print(f"Total/tax errors fixed: {total_tax_errors_fixed}")
print(f"Negative quantities fixed: {negative_quantities_fixed}")
print(f"Conflicting records resolved: {conflicts_resolved}")
print(f"Rows removed for missing dates: {rows_removed_for_missing_dates}")

print("\nBranch summary:")
print(branch_summary.to_string(index=False))

print("\nAnalysis completed. CSV reports and charts saved.")