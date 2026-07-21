# Housing Price Prediction — OLS from Scratch

An Ordinary Least Squares (OLS) linear regression estimator built entirely from scratch using the **Normal Equation** — no `sklearn.LinearRegression` used for the core model. Tested on the real **Ames Housing dataset** (Kaggle's "House Prices: Advanced Regression Techniques"), with manual missing-value handling and multicollinearity detection.

## What This Project Does

- Implements an `OLSEstimator` class from scratch using the Normal Equation: `β = (XᵀX)⁻¹Xᵀy`
- Loads real housing data (1,460 homes, Ames, Iowa) and manually cleans missing values using median imputation
- Detects multicollinearity using **Variance Inflation Factor (VIF)** and iteratively removes unstable/redundant features
- Verifies the from-scratch model's coefficients against `sklearn.LinearRegression` for correctness
- Includes a custom prediction function to estimate the price of a new, user-defined house

## Why Median Imputation (not Mean)?

Housing data often contains outliers (a few very large or very expensive properties). Mean is sensitive to these outliers, while median is robust to them — making it a safer default for filling missing values in this kind of data.

## Why Check for Multicollinearity?

When two features carry overlapping information (e.g., living area and total rooms), a model's coefficients become unstable and hard to interpret. VIF quantifies this overlap; features with VIF > 10 are iteratively removed one at a time until all remaining features are stable.

## Results

| Metric | Value |
|---|---|
| Dataset | Ames Housing (1,460 real homes) |
| Features used | 10 numeric features → reduced after VIF elimination |
| R² Score | 0.60 |
| Verification vs sklearn | Coefficients match |

## Tech Stack

- Python
- NumPy, Pandas
- Statsmodels (VIF calculation)
- Scikit-Learn (verification only)

## Setup

```bash
pip install numpy pandas statsmodels scikit-learn
python day2_housing_ols_REAL.py
```

**Note:** `ames_train.csv` must be in the same folder as the script.

## Bonus: Synthetic Version

`day2_housing_ols_synthetic.py` is a companion script that generates its own synthetic housing dataset (no external file needed) — useful for understanding the pipeline with fully controlled, known ground-truth data before applying it to real-world data.

---

*Part of a self-driven 60-day AI/ML Engineering learning journey, focused on implementing core ML algorithms from first principles.*
