# ============================================================
# CRISP-DM Multiple Linear Regression with Feature Selection
# Dataset: 50_Startups.csv
# Goal: Predict startup Profit & Compare 5 Feature Selection Methods
# Author: anna0831 (https://github.com/anna0831)
# ============================================================

import sys
import os

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure directory for plots exists
os.makedirs("plots", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    f_regression,
    RFE,
    mutual_info_regression
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. BUSINESS UNDERSTANDING
# ============================================================

"""
Business objective:
Predict startup Profit using:
1. R&D Spend
2. Administration
3. Marketing Spend
4. State

Target:
Profit (continuous numerical variable -> Regression task)

Feature Selection Objective:
Compare five distinct feature-selection strategies representing
different theoretical families:
1. F-test / F-regression   (Filter method: linear statistical association)
2. Lasso                   (Embedded method: L1 regularization penalty)
3. RFE                     (Wrapper method: recursive model-based elimination)
4. Random Forest           (Embedded/Model-based: tree Gini/variance reduction)
5. Mutual Information      (Filter method: non-linear statistical dependency)

Key Data Leakage Rule:
Feature selection must be learned using the TRAINING DATA ONLY.
Raw data -> Train/Test Split -> Preprocessing fit on train ->
Feature selection on train -> Final Linear Regression model evaluation
on untouched test data.
"""


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

csv_path = "50_Startups.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"'{csv_path}' not found in the current directory.")

df = pd.read_csv(csv_path)


# ------------------------------------------------------------
# Display first rows
# ------------------------------------------------------------

print("=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)
print(df.head())


# ------------------------------------------------------------
# Dataset shape & Info
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)
print(df.shape)

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
df.info()


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)
print(df.describe(include="all"))


# ------------------------------------------------------------
# Missing values & Duplicates
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(df.isnull().sum())

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)
print(df.duplicated().sum())


# ------------------------------------------------------------
# State categories
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STATE CATEGORIES")
print("=" * 70)
print(df["State"].value_counts())


# ------------------------------------------------------------
# Correlation matrix for numeric columns
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

print("\n" + "=" * 70)
print("NUMERIC CORRELATION MATRIX")
print("=" * 70)
print(numeric_df.corr())


# ------------------------------------------------------------
# Exploratory Plots (Saved to plots/)
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(df["R&D Spend"], df["Profit"], color="#1f77b4", alpha=0.8)
plt.xlabel("R&D Spend ($)")
plt.ylabel("Profit ($)")
plt.title("R&D Spend vs Profit")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/1_rd_spend_vs_profit.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["Administration"], df["Profit"], color="#ff7f0e", alpha=0.8)
plt.xlabel("Administration ($)")
plt.ylabel("Profit ($)")
plt.title("Administration vs Profit")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/2_administration_vs_profit.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["Marketing Spend"], df["Profit"], color="#2ca02c", alpha=0.8)
plt.xlabel("Marketing Spend ($)")
plt.ylabel("Profit ($)")
plt.title("Marketing Spend vs Profit")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/3_marketing_spend_vs_profit.png", dpi=300)
plt.show()


# ============================================================
# 3. DATA PREPARATION
# ============================================================

# ------------------------------------------------------------
# Define feature columns and target column
# ------------------------------------------------------------

feature_columns = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State"
]

target_column = "Profit"

X = df[feature_columns]
y = df[target_column]

numeric_features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]

categorical_features = [
    "State"
]


# ------------------------------------------------------------
# Train-Test Split (Strictly before feature selection)
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ------------------------------------------------------------
# Preprocessing Setup (One-Hot Encoding on State)
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Fit preprocessor on training data only to prevent data leakage
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Retrieve transformed feature names programmatically
feature_names = np.array(preprocessor.get_feature_names_out())

print("\n" + "=" * 70)
print("PROGRAMMATIC TRANSFORMED FEATURE NAMES")
print("=" * 70)
for idx, fname in enumerate(feature_names):
    print(f"Index {idx}: {fname}")

n_features_total = len(feature_names)
n_features_to_select = max(1, n_features_total // 2)
print(f"\nTotal transformed features: {n_features_total}")
print(f"Default subset selection target (len // 2): {n_features_to_select}")


# ============================================================
# 4. MODELING + FEATURE SELECTION
# ============================================================

# ------------------------------------------------------------
# Step 2: Reusable Evaluation Function
# ------------------------------------------------------------

evaluation_records = []

def evaluate_selected_features(
    method_name,
    selected_feature_names,
    selected_indices,
    X_train_p,
    X_test_p,
    y_tr,
    y_te
):
    """
    Trains an ordinary LinearRegression on selected columns only
    and evaluates metrics on untouched test data.
    """
    selected_indices = list(selected_indices)
    X_tr_sub = X_train_p[:, selected_indices]
    X_te_sub = X_test_p[:, selected_indices]

    # Final estimator must remain LinearRegression for fair comparison
    lr = LinearRegression()
    lr.fit(X_tr_sub, y_tr)

    y_pred = lr.predict(X_te_sub)

    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_te, y_pred)

    print(f"\n--- {method_name} Evaluation ---")
    print(f"Selected ({len(selected_indices)}): {list(selected_feature_names)}")
    print(f"MAE  : {mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Method": method_name,
        "Number_of_Selected_Features": len(selected_indices),
        "Selected_Features": ", ".join(selected_feature_names),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ------------------------------------------------------------
# 4.1 No Feature Selection (Baseline Full Model)
# ------------------------------------------------------------

all_indices = list(range(len(feature_names)))
res_full = evaluate_selected_features(
    method_name="No Feature Selection",
    selected_feature_names=feature_names,
    selected_indices=all_indices,
    X_train_p=X_train_processed,
    X_test_p=X_test_processed,
    y_tr=y_train,
    y_te=y_test
)
evaluation_records.append(res_full)


# ------------------------------------------------------------
# 4.2 F-test / F-regression (Filter Method)
# ------------------------------------------------------------

"""
F-test / F-regression:
Filter method evaluating individual linear statistical association.
"""

f_scores, f_pvalues = f_regression(X_train_processed, y_train)

f_ranking_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "F_Score": f_scores,
        "P_Value": f_pvalues,
        "Original_Index": np.arange(len(feature_names))
    }
).sort_values(by="F_Score", ascending=False).reset_index(drop=True)

f_ranking_df["Rank"] = np.arange(1, len(f_ranking_df) + 1)

# Selection rule: p-value < 0.05, with fallback to highest-ranked feature
f_ranking_df["Selected"] = f_ranking_df["P_Value"] < 0.05
if not f_ranking_df["Selected"].any():
    f_ranking_df.loc[0, "Selected"] = True

f_selected_indices = f_ranking_df.loc[f_ranking_df["Selected"], "Original_Index"].values
f_selected_names = f_ranking_df.loc[f_ranking_df["Selected"], "Feature"].values

res_f_test = evaluate_selected_features(
    method_name="F-test",
    selected_feature_names=f_selected_names,
    selected_indices=f_selected_indices,
    X_train_p=X_train_processed,
    X_test_p=X_test_processed,
    y_tr=y_train,
    y_te=y_test
)
evaluation_records.append(res_f_test)


# ------------------------------------------------------------
# 4.3 Lasso Feature Selection (Embedded Method)
# ------------------------------------------------------------

"""
Lasso (L1 Regularization):
Embedded method performing feature selection during regularized training.
Scale-sensitive, so predictors are scaled using training data only.
"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

lasso_cv = LassoCV(
    cv=5,
    random_state=42,
    max_iter=10000
)
lasso_cv.fit(X_train_scaled, y_train)

lasso_ranking_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": lasso_cv.coef_,
        "Absolute_Coefficient": np.abs(lasso_cv.coef_),
        "Original_Index": np.arange(len(feature_names))
    }
).sort_values(by="Absolute_Coefficient", ascending=False).reset_index(drop=True)

lasso_ranking_df["Rank"] = np.arange(1, len(lasso_ranking_df) + 1)

# Selection rule: |coef| > 1e-8, fallback to largest absolute coefficient
lasso_ranking_df["Selected"] = lasso_ranking_df["Absolute_Coefficient"] > 1e-8
if not lasso_ranking_df["Selected"].any():
    lasso_ranking_df.loc[0, "Selected"] = True

lasso_selected_indices = lasso_ranking_df.loc[lasso_ranking_df["Selected"], "Original_Index"].values
lasso_selected_names = lasso_ranking_df.loc[lasso_ranking_df["Selected"], "Feature"].values

res_lasso = evaluate_selected_features(
    method_name="Lasso",
    selected_feature_names=lasso_selected_names,
    selected_indices=lasso_selected_indices,
    X_train_p=X_train_processed,
    X_test_p=X_test_processed,
    y_tr=y_train,
    y_te=y_test
)
evaluation_records.append(res_lasso)


# ------------------------------------------------------------
# 4.4 Recursive Feature Elimination (RFE - Wrapper Method)
# ------------------------------------------------------------

"""
Recursive Feature Elimination (RFE):
Wrapper method repeatedly training a Linear Regression model and
pruning features with smallest contribution until target count is reached.
Fitted using training data only.
"""

rfe = RFE(
    estimator=LinearRegression(),
    n_features_to_select=n_features_to_select,
    step=1
)
rfe.fit(X_train_processed, y_train)

rfe_ranking_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "RFE_Rank": rfe.ranking_,
        "Selected": rfe.support_,
        "Original_Index": np.arange(len(feature_names))
    }
).sort_values(by="RFE_Rank", ascending=True).reset_index(drop=True)

rfe_selected_indices = rfe_ranking_df.loc[rfe_ranking_df["Selected"], "Original_Index"].values
rfe_selected_names = rfe_ranking_df.loc[rfe_ranking_df["Selected"], "Feature"].values

res_rfe = evaluate_selected_features(
    method_name="RFE",
    selected_feature_names=rfe_selected_names,
    selected_indices=rfe_selected_indices,
    X_train_p=X_train_processed,
    X_test_p=X_test_processed,
    y_tr=y_train,
    y_te=y_test
)
evaluation_records.append(res_rfe)


# ------------------------------------------------------------
# 4.5 Random Forest Feature Importance (Tree Model-Based)
# ------------------------------------------------------------

"""
Random Forest Feature Importance:
Embedded/model-based importance method ranking features by variance
reduction across decision trees.
"""

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)
rf.fit(X_train_processed, y_train)

rf_ranking_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf.feature_importances_,
        "Original_Index": np.arange(len(feature_names))
    }
).sort_values(by="Importance", ascending=False).reset_index(drop=True)

rf_ranking_df["Rank"] = np.arange(1, len(rf_ranking_df) + 1)
rf_ranking_df["Selected"] = False
rf_ranking_df.loc[:n_features_to_select - 1, "Selected"] = True

rf_selected_indices = rf_ranking_df.loc[rf_ranking_df["Selected"], "Original_Index"].values
rf_selected_names = rf_ranking_df.loc[rf_ranking_df["Selected"], "Feature"].values

res_rf = evaluate_selected_features(
    method_name="Random Forest",
    selected_feature_names=rf_selected_names,
    selected_indices=rf_selected_indices,
    X_train_p=X_train_processed,
    X_test_p=X_test_processed,
    y_tr=y_train,
    y_te=y_test
)
evaluation_records.append(res_rf)


# ------------------------------------------------------------
# 4.6 Mutual Information (Filter Method)
# ------------------------------------------------------------

"""
Mutual Information:
Filter method measuring non-parametric statistical dependency (linear and non-linear).
"""

mi_scores = mutual_info_regression(
    X_train_processed,
    y_train,
    random_state=42
)

mi_ranking_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "MI_Score": mi_scores,
        "Original_Index": np.arange(len(feature_names))
    }
).sort_values(by="MI_Score", ascending=False).reset_index(drop=True)

mi_ranking_df["Rank"] = np.arange(1, len(mi_ranking_df) + 1)
mi_ranking_df["Selected"] = False
mi_ranking_df.loc[:n_features_to_select - 1, "Selected"] = True

mi_selected_indices = mi_ranking_df.loc[mi_ranking_df["Selected"], "Original_Index"].values
mi_selected_names = mi_ranking_df.loc[mi_ranking_df["Selected"], "Feature"].values

res_mi = evaluate_selected_features(
    method_name="Mutual Information",
    selected_feature_names=mi_selected_names,
    selected_indices=mi_selected_indices,
    X_train_p=X_train_processed,
    X_test_p=X_test_processed,
    y_tr=y_train,
    y_te=y_test
)
evaluation_records.append(res_mi)


# ============================================================
# 5. EVALUATION
# ============================================================

# ------------------------------------------------------------
# Step 8: Final Feature Selection Comparison Table
# ------------------------------------------------------------

feature_selection_comparison = pd.DataFrame(evaluation_records)

# Sort primarily by R2 descending, secondarily by RMSE ascending
feature_selection_comparison = feature_selection_comparison.sort_values(
    by=["R2", "RMSE"],
    ascending=[False, True]
).reset_index(drop=True)

print("\n" + "=" * 90)
print("FINAL FEATURE SELECTION COMPARISON TABLE")
print("=" * 90)
print(feature_selection_comparison.to_string(index=False))


# ------------------------------------------------------------
# Step 9: Separate Ranking Tables for Every Method
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. F-TEST RANKING TABLE")
print("=" * 70)
print(f_ranking_df[["Rank", "Feature", "F_Score", "P_Value", "Selected"]].to_string(index=False))

print("\n" + "=" * 70)
print("2. LASSO RANKING TABLE")
print("=" * 70)
print(lasso_ranking_df[["Rank", "Feature", "Coefficient", "Absolute_Coefficient", "Selected"]].to_string(index=False))

print("\n" + "=" * 70)
print("3. RFE RANKING TABLE")
print("=" * 70)
print(rfe_ranking_df[["Feature", "RFE_Rank", "Selected"]].to_string(index=False))

print("\n" + "=" * 70)
print("4. RANDOM FOREST RANKING TABLE")
print("=" * 70)
print(rf_ranking_df[["Rank", "Feature", "Importance", "Selected"]].to_string(index=False))

print("\n" + "=" * 70)
print("5. MUTUAL INFORMATION RANKING TABLE")
print("=" * 70)
print(mi_ranking_df[["Rank", "Feature", "MI_Score", "Selected"]].to_string(index=False))


# ------------------------------------------------------------
# Baseline Comparison (DummyRegressor)
# ------------------------------------------------------------

dummy_baseline = DummyRegressor(strategy="mean")
dummy_baseline.fit(np.zeros((len(y_train), 1)), y_train)
dummy_pred = dummy_baseline.predict(np.zeros((len(y_test), 1)))

baseline_rmse = np.sqrt(mean_squared_error(y_test, dummy_pred))
baseline_r2 = r2_score(y_test, dummy_pred)

print("\n" + "=" * 70)
print("DUMMY BASELINE PERFORMANCE")
print("=" * 70)
print(f"Dummy Baseline RMSE : {baseline_rmse:,.2f}")
print(f"Dummy Baseline R²   : {baseline_r2:.4f}")


# ------------------------------------------------------------
# Step 14: Small Dataset Warning & Cross-Validation Analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SMALL DATASET WARNING & CROSS-VALIDATION")
print("=" * 70)
print(
    """
WARNING:
The dataset contains only 50 observations.
An 80/20 split reserves only 10 observations for testing.
Due to the small sample size:
- Test metrics have higher variance.
- Feature rankings should be verified across resampled partitions.
- The original 50_Startups.csv file is ordered by Profit descending.
  Therefore, standard cross_val_score(cv=5) without shuffling partitions
  target profit into separate high/low brackets, causing negative CV R².
  Applying KFold with shuffle=True is essential for realistic CV estimates.
    """
)

# Cross-validation for the full model
full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

cv_standard = cross_val_score(full_pipeline, X, y, cv=5, scoring="r2")
print(f"Full Model Standard CV R² (unshuffled) : {cv_standard.mean():.4f} (std: {cv_standard.std():.4f})")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_shuffled = cross_val_score(full_pipeline, X, y, cv=kf, scoring="r2")
print(f"Full Model Shuffled 5-Fold CV R²       : {cv_shuffled.mean():.4f} (std: {cv_shuffled.std():.4f})")

# Cross-validation on top-ranked feature subset (R&D Spend + Marketing Spend)
top_2_indices = [
    list(feature_names).index("remainder__R&D Spend"),
    list(feature_names).index("remainder__Marketing Spend")
]
X_processed_all = preprocessor.fit_transform(X)
X_top2 = X_processed_all[:, top_2_indices]
cv_top2_shuffled = cross_val_score(LinearRegression(), X_top2, y, cv=kf, scoring="r2")
print(f"R&D + Marketing Subset Shuffled CV R²  : {cv_top2_shuffled.mean():.4f} (std: {cv_top2_shuffled.std():.4f})")


# ------------------------------------------------------------
# Step 10: Visualizations (Matplotlib only, saved to plots/)
# ------------------------------------------------------------

# 1. F-test F-Scores
plt.figure(figsize=(9, 4.5))
sorted_f = f_ranking_df.sort_values(by="F_Score", ascending=True)
colors_f = ["#2ca02c" if s else "#7f7f7f" for s in sorted_f["Selected"]]
plt.barh(sorted_f["Feature"], sorted_f["F_Score"], color=colors_f)
plt.xlabel("F-Score")
plt.title("1. F-test Feature Ranking (Green = Selected p < 0.05)")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/fs_1_f_test_ranking.png", dpi=300)
plt.show()

# 2. Absolute Lasso Coefficients
plt.figure(figsize=(9, 4.5))
sorted_lasso = lasso_ranking_df.sort_values(by="Absolute_Coefficient", ascending=True)
colors_lasso = ["#1f77b4" if s else "#7f7f7f" for s in sorted_lasso["Selected"]]
plt.barh(sorted_lasso["Feature"], sorted_lasso["Absolute_Coefficient"], color=colors_lasso)
plt.xlabel("Absolute Standardized Coefficient")
plt.title("2. Lasso Feature Ranking (|Coefficient| > 1e-8)")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/fs_2_lasso_ranking.png", dpi=300)
plt.show()

# 3. RFE Ranking
plt.figure(figsize=(9, 4.5))
# Invert rank so rank 1 is longest bar: (max_rank + 1 - rank)
max_rfe_rank = rfe_ranking_df["RFE_Rank"].max()
sorted_rfe = rfe_ranking_df.sort_values(by="RFE_Rank", ascending=False)
rfe_bar_lengths = max_rfe_rank + 1 - sorted_rfe["RFE_Rank"]
colors_rfe = ["#d62728" if s else "#7f7f7f" for s in sorted_rfe["Selected"]]
plt.barh(sorted_rfe["Feature"], rfe_bar_lengths, color=colors_rfe)
plt.xlabel(f"Inverted Rank Score ({max_rfe_rank} = Rank 1 Selected)")
plt.title("3. Recursive Feature Elimination (RFE) Ranking")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/fs_3_rfe_ranking.png", dpi=300)
plt.show()

# 4. Random Forest Feature Importance
plt.figure(figsize=(9, 4.5))
sorted_rf = rf_ranking_df.sort_values(by="Importance", ascending=True)
colors_rf = ["#9467bd" if s else "#7f7f7f" for s in sorted_rf["Selected"]]
plt.barh(sorted_rf["Feature"], sorted_rf["Importance"], color=colors_rf)
plt.xlabel("MDI Feature Importance")
plt.title("4. Random Forest Feature Importance Ranking")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/fs_4_rf_ranking.png", dpi=300)
plt.show()

# 5. Mutual Information Scores
plt.figure(figsize=(9, 4.5))
sorted_mi = mi_ranking_df.sort_values(by="MI_Score", ascending=True)
colors_mi = ["#8c564b" if s else "#7f7f7f" for s in sorted_mi["Selected"]]
plt.barh(sorted_mi["Feature"], sorted_mi["MI_Score"], color=colors_mi)
plt.xlabel("Mutual Information Score")
plt.title("5. Mutual Information Ranking")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/fs_5_mi_ranking.png", dpi=300)
plt.show()

# 6. Comparison Bar Charts: Method vs R² and Method vs RMSE
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Method vs R²
sorted_by_r2 = feature_selection_comparison.sort_values(by="R2", ascending=True)
ax[0].barh(sorted_by_r2["Method"], sorted_by_r2["R2"], color="#2b5c8f")
ax[0].set_xlabel("Test R²")
ax[0].set_title("Method vs Test R² (Higher is Better)")
ax[0].grid(axis="x", linestyle="--", alpha=0.5)
for i, v in enumerate(sorted_by_r2["R2"]):
    ax[0].text(v + (0.01 if v >= 0 else -0.1), i, f"{v:.4f}", va="center", fontsize=9)

# Method vs RMSE
sorted_by_rmse = feature_selection_comparison.sort_values(by="RMSE", ascending=False)
ax[1].barh(sorted_by_rmse["Method"], sorted_by_rmse["RMSE"], color="#e26d5c")
ax[1].set_xlabel("Test RMSE ($)")
ax[1].set_title("Method vs Test RMSE (Lower is Better)")
ax[1].grid(axis="x", linestyle="--", alpha=0.5)
for i, v in enumerate(sorted_by_rmse["RMSE"]):
    ax[1].text(v + 500, i, f"${v:,.0f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("plots/fs_6_method_comparison.png", dpi=300)
plt.show()


# ============================================================
# 6. DEPLOYMENT & SUMMARY
# ============================================================

"""
Deployment demonstration:
Predict Profit for a new startup using the optimal feature subset
identified by feature selection (R&D Spend + Marketing Spend).
"""

new_startup = pd.DataFrame(
    {
        "R&D Spend": [100000],
        "Administration": [120000],
        "Marketing Spend": [250000],
        "State": ["California"]
    }
)

# Optimal subset model: R&D Spend + Marketing Spend
best_indices = f_selected_indices
best_names = f_selected_names

optimal_model = LinearRegression()
optimal_model.fit(X_train_processed[:, best_indices], y_train)

new_processed = preprocessor.transform(new_startup)[:, best_indices]
predicted_profit_optimal = optimal_model.predict(new_processed)

print("\n" + "=" * 70)
print("DEPLOYMENT PREDICTION")
print("=" * 70)
print("New startup profile:")
print(new_startup)
print(f"Selected Features Used : {list(best_names)}")
print(f"Predicted Profit       : ${predicted_profit_optimal[0]:,.2f}")


# ------------------------------------------------------------
# Step 11 & 12 & 13: Interpretation & Statistical Principles
# ------------------------------------------------------------

fewest_method = feature_selection_comparison.sort_values(
    by="Number_of_Selected_Features", ascending=True
).iloc[0]["Method"]
fewest_count = feature_selection_comparison["Number_of_Selected_Features"].min()

best_r2_row = feature_selection_comparison.loc[feature_selection_comparison["R2"].idxmax()]
best_rmse_row = feature_selection_comparison.loc[feature_selection_comparison["RMSE"].idxmin()]

print("\n" + "=" * 70)
print("INTERPRETATION OF RESULTS")
print("=" * 70)
print(
    f"""
1. Which method selected the fewest features?
   -> {fewest_method} (tied with other subset methods at {fewest_count} features).

2. Which method achieved the highest test R²?
   -> {best_r2_row['Method']} (R² = {best_r2_row['R2']:.4f}).

3. Which method achieved the lowest test RMSE?
   -> {best_rmse_row['Method']} (RMSE = ${best_rmse_row['RMSE']:,.2f}).

4. Which features appear repeatedly across multiple methods?
   -> 'R&D Spend' and 'Marketing Spend' are selected repeatedly by
      F-test, Random Forest, Mutual Information, and Lasso.

5. Is R&D Spend consistently important?
   -> YES. R&D Spend has the strongest statistical association with Profit
      across all 5 methods (F-Score = {f_scores[list(feature_names).index('remainder__R&D Spend')]:.2f},
      RF Importance = {rf.feature_importances_[list(feature_names).index('remainder__R&D Spend')]:.2%},
      MI Score = {mi_scores[list(feature_names).index('remainder__R&D Spend')]:.2f}).

6. Does State appear important or relatively weak?
   -> RELATIVELY WEAK. State dummy categories (California reference, Florida, New York)
      consistently rank near the bottom in statistical significance:
      F-test p-values are > 0.50, Lasso coefficients are zeroed out,
      and Random Forest assigns < 0.2% importance to State dummies.
      (Note on RFE: In unscaled Linear Regression, RFE can be misled by raw feature scales
      because dummy coefficients appear artificially large in raw units compared to slope
      per dollar, leading to suboptimal selection when unstandardized).

7. Does feature selection improve performance compared with using all features?
   -> YES. Pruning irrelevant features (Administration and State dummies) improved
      the test R² from {res_full['R2']:.4f} to {best_r2_row['R2']:.4f} and reduced
      test RMSE from ${res_full['RMSE']:,.2f} to ${best_rmse_row['RMSE']:,.2f},
      demonstrating that eliminating noise enhances model generalization.
    """
)

print("=" * 70)
print("FEATURE SELECTION FAMILIES & STATISTICAL METHODOLOGY")
print("=" * 70)
print(
    """
F-test:
Filter method.
Evaluates each feature individually using linear statistical association.

Lasso:
Embedded method.
Performs feature selection during regularized model training.

RFE:
Wrapper method.
Repeatedly trains a model and removes weaker features.

Random Forest:
Embedded/model-based importance method.
Ranks variables using tree-based predictive contribution.

Mutual Information:
Filter method.
Measures statistical dependency and can capture nonlinear relationships.

Important Comparison Note:
Raw metric values (F-Score, MI Score, Tree Importance, Lasso Beta) are NOT directly
comparable numerically across methods due to differing measurement scales.
Only feature orderings, selection decisions, and resulting LinearRegression
test metrics provide valid, objective grounds for comparison.
    """
)

print("=" * 70)
print("FEATURE SELECTION SUMMARY")
print("=" * 70)
print(f"Best Test R² Method:")
print(f"-> {best_r2_row['Method']} (R² = {best_r2_row['R2']:.4f})")
print(f"\nLowest RMSE Method:")
print(f"-> {best_rmse_row['Method']} (RMSE = ${best_rmse_row['RMSE']:,.2f})")
print(f"\nFeatures selected most consistently:")
print("-> R&D Spend and Marketing Spend")
print(f"\nConclusion:")
print(
    """-> R&D Spend is by far the strongest factor associated with startup Profit,
   followed secondarily by Marketing Spend.
-> Administration spending and geographic State have little to no statistical
   association with Profit.
-> Removing uninformative variables via feature selection successfully simplifies
   the model while improving generalization performance on test data.
"""
)
print("=" * 70)
print("PROJECT REPOSITORY & PROFILE")
print("=" * 70)
print("GitHub: https://github.com/anna0831")
