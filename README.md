# 🚀 CRISP-DM Multiple Linear Regression with 5 Feature Selection Methods

[![Author](https://img.shields.io/badge/Author-anna0831-black?style=for-the-badge&logo=github)](https://github.com/anna0831)
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning study following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to predict startup **Profit** using the `50_Startups.csv` dataset. 

This project benchmarks **five distinct feature selection strategies** across filter, embedded, wrapper, and model-based families while strictly preserving data leakage boundaries (evaluating on holdout test data using ordinary `LinearRegression`).

---

## 📌 Author & Profile

- **GitHub:** [@anna0831](https://github.com/anna0831)
- **Repository:** [50_Startups CRISP-DM Feature Selection](https://github.com/anna0831)

---

## 🧠 The 5 Feature Selection Families

```
Raw Data
   └── Train/Test Split (80/20, random_state=42)
        └── Fit Preprocessor on X_train only (OneHotEncoder: State)
             └── Run Feature Selection strictly on X_train_processed
                  ├── 1. F-test (Filter: Univariate linear p-value < 0.05)
                  ├── 2. LassoCV (Embedded: L1 regularization on scaled X)
                  ├── 3. RFE (Wrapper: Recursive feature elimination)
                  ├── 4. Random Forest (Tree-based: MDI impurity importance)
                  └── 5. Mutual Information (Filter: Non-linear statistical dependency)
                       └── Evaluate Selected Subsets on untouched X_test with LinearRegression()
```

1. **F-test / F-regression (Filter):** Measures individual linear statistical association ($F$-statistic) and selects statistically significant features ($p < 0.05$).
2. **Lasso (Embedded):** Enforces $L_1$ penalty shrinkage on standardized predictors; automatically zeroes out redundant parameters.
3. **RFE (Wrapper):** Iteratively trains `LinearRegression` and prunes features with the smallest contribution until the desired subset size is reached.
4. **Random Forest (Model-based):** Measures Mean Decrease in Impurity (MDI) across 500 decision trees to quantify non-parametric variance reduction.
5. **Mutual Information (Filter):** Non-parametric entropy-based metric capturing both linear and non-linear relationships.

---

## 📊 Final Performance Comparison

All models evaluate `LinearRegression()` on the untouched 20% holdout test set ($N=10$):

| Rank | Method | Family | Features Selected | MAE ($) | RMSE ($) | Test $R^2$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **🥇 1** | **F-test** | Filter | `R&D Spend`, `Marketing Spend` | **$6,220.08** | **$8,206.33** | **0.9168** |
| **🥇 1** | **Random Forest** | Tree Model-based | `R&D Spend`, `Marketing Spend` | **$6,220.08** | **$8,206.33** | **0.9168** |
| **🥇 1** | **Mutual Information** | Filter (Non-linear) | `R&D Spend`, `Marketing Spend` | **$6,220.08** | **$8,206.33** | **0.9168** |
| **4** | **No Feature Selection** | Baseline | All 5 Features | $6,961.48 | $9,055.96 | 0.8987 |
| **5** | **Lasso** | Embedded | `R&D`, `Marketing`, `Admin` | $6,903.01 | $9,145.45 | 0.8967 |
| **6** | **RFE (unscaled)** | Wrapper | `State_Florida`, `State_New York` | $29,664.69 | $35,907.82 | -0.5917 |

---

## 🔍 Key Data Science Insights

- **Optimal Subset:** Pruning `Administration` and `State` improves test $R^2$ from **0.8987 to 0.9168** and reduces RMSE by **$849.63** (9.38% error reduction).
- **Core Driver:** **`R&D Spend`** is overwhelmingly the primary driver of profitability across all methods (Tree importance: **92.69%**, $F$-statistic: **676.10**, $p = 8.22 \times 10^{-26}$).
- **Secondary Synergies:** **`Marketing Spend`** provides secondary positive predictive value.
- **Uninformative Variables:** **`Administration`** overhead and **`State`** location show no statistically significant relationship with startup profit.
- **Why Unscaled Linear RFE Prunes Spend:** Ordinary Linear Regression coefficients depend on feature scales. Continuous spend amounts have tiny slopes per dollar compared to binary dummy variables, illustrating the critical importance of feature scaling when applying wrapper methods to linear models.

---

## 🛠️ Project Structure

```text
50_starups/
├── 50_Startups.csv          # Raw dataset (50 startup records)
├── main.py                  # Primary CRISP-DM & 5 Feature Selection pipeline
├── startup_regression.py    # Synchronized pipeline script
├── README.md                # Project documentation & GitHub links
├── .gitignore               # Git ignore configuration
└── plots/                   # Exported high-resolution (300 DPI) visualizations
    ├── 1_rd_spend_vs_profit.png
    ├── 2_administration_vs_profit.png
    ├── 3_marketing_spend_vs_profit.png
    ├── 4_actual_vs_predicted.png
    ├── 5_residual_plot.png
    ├── 6_residual_distribution.png
    ├── fs_1_f_test_ranking.png
    ├── fs_2_lasso_ranking.png
    ├── fs_3_rfe_ranking.png
    ├── fs_4_rf_ranking.png
    ├── fs_5_mi_ranking.png
    └── fs_6_method_comparison.png
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/anna0831/50_startups-crisp-dm.git
cd 50_startups-crisp-dm
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib
```

### 3. Run the pipeline
```bash
python main.py
```

---

## 👤 Author

Developed by **[anna0831](https://github.com/anna0831)**.
Feel free to connect or explore more data science projects on GitHub!
