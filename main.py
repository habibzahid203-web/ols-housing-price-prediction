"""
DAY 2 PROJECT (REAL DATASET VERSION) — Housing Valuation Engine
Dataset: Ames Housing (Kaggle "House Prices: Advanced Regression Techniques")
         1460 real houses, 79 real features -- yahan hum kuch numeric
         features select kar rahe hain jo naturally "messy" (missing
         values) aur "collinear" (aapas mein correlated) hain.

Goal: Same as pehle -- OLS scratch se, manual cleaning, VIF se
      multi-collinearity check -- lekin is baar REAL data pe.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

# ============================================================
# STEP 1: Real dataset load karo
# ============================================================
# Ye dataset Kaggle ke mashhoor "House Prices" competition ka hai,
# jo Ames, Iowa (USA) ke 1460 real ghar ki sale details deta hai.

df = pd.read_csv('ames_train.csv')

# 79 columns hain (bohat zyada) -- hum kuch numeric, meaningful
# features select kar rahe hain jo price predict karne mein
# helpful hain:
selected_cols = [
    'GrLivArea',      # ghar ka above-ground living area (sqft)
    'TotalBsmtSF',    # basement ka total area (sqft)
    'GarageArea',     # garage ka area (sqft)
    'GarageCars',     # garage mein kitni cars aa sakti hain
    'OverallQual',    # ghar ki overall quality (1-10 rating)
    'YearBuilt',      # ghar kab bana (saal)
    'LotArea',        # plot/zameen ka total area
    'LotFrontage',    # sadak se ghar ki seedhi doori (isme MISSING values hain!)
    'FullBath',       # kitne full bathrooms hain
    'TotRmsAbvGrd',   # total kamre (GrLivArea se correlated hoga -- collinearity!)
    'SalePrice'       # TARGET: asal bikri ki price
]

df = df[selected_cols].copy()

print("===== REAL DATA (Ames Housing) — pehle 5 rows =====")
print(df.head())
print(f"\nDataset shape: {df.shape[0]} ghar, {df.shape[1]} columns")
print(f"\nTotal missing values (real messy data):\n{df.isnull().sum()}")


# ============================================================
# STEP 2: Manual Data Cleaning (Real missing values handle karna)
# ============================================================
# LotFrontage mein 259 real missing values hain -- ye asal duniya
# ki messiness hai (survey na hona, data entry miss hona, etc.)

for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"'{col}' ke {missing_count} missing values median ({median_val:.1f}) se fill kiye")

print(f"\nMissing values ab: {df.isnull().sum().sum()}")

# Outliers ka bhi chota sa check -- SalePrice mein bohat extreme
# values (bohat mehnge ghar) model ko bigaad sakti hain
print(f"\nSalePrice range: {df['SalePrice'].min()} se {df['SalePrice'].max()}")


# ============================================================
# STEP 3: Multi-collinearity Check (Iterative VIF elimination)
# ============================================================
feature_cols = [c for c in selected_cols if c != 'SalePrice']
X_vif = df[feature_cols].copy()

vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("\n===== VIF SCORES (Real data pe multi-collinearity) =====")
print(vif_data.sort_values('VIF', ascending=False))

current_features = feature_cols.copy()
while True:
    X_temp = df[current_features].copy()
    vifs = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
    max_vif = max(vifs)
    if max_vif <= 10:
        break
    worst_feature = current_features[vifs.index(max_vif)]
    print(f"Dropping '{worst_feature}' (VIF = {max_vif:.2f}) -- sabse unstable")
    current_features.remove(worst_feature)

final_features = current_features
print(f"\nFinal features (in par use honge): {final_features}")

# ===== CONFIRMATION: Final VIF scores of KEPT features =====
# Ye proof hai ke jo features bache hain, un sab ka VIF safely
# 10 se kam hai (multi-collinearity resolve ho chuki hai)
X_final_vif = df[final_features].astype(float).copy()
final_vif_data = pd.DataFrame()
final_vif_data['Feature'] = X_final_vif.columns
final_vif_data['VIF'] = [variance_inflation_factor(X_final_vif.values, i) for i in range(X_final_vif.shape[1])]

print("\n===== FINAL VIF SCORES (kept features -- confirmation) =====")
print(final_vif_data.sort_values('VIF', ascending=False).to_string(index=False))
print(f"All VIF < 10? {(final_vif_data['VIF'] < 10).all()}")


# ============================================================
# STEP 4: Feature Scaling (Manual Standardization)
# ============================================================
def standardize(column):
    return (column - column.mean()) / column.std()

X_scaled = df[final_features].apply(standardize)
y = df['SalePrice'].values


# ============================================================
# STEP 5: OLS Estimator SCRATCH SE (Normal Equation)
# ============================================================
class OLSEstimator:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        XtX = X_with_intercept.T @ X_with_intercept
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_with_intercept.T @ y
        self.coefficients = XtX_inv @ Xty
        return self

    def predict(self, X):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept @ self.coefficients


ols = OLSEstimator()
ols.fit(X_scaled.values, y)

print("\n===== OLS COEFFICIENTS (Real data, scratch se) =====")
print(f"Intercept: {ols.coefficients[0]:.2f}")
for feat, coef in zip(final_features, ols.coefficients[1:]):
    print(f"{feat}: {coef:.2f}")


# ============================================================
# STEP 6: Performance Check (R²)
# ============================================================
predictions = ols.predict(X_scaled.values)
ss_res = np.sum((y - predictions) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\n===== MODEL PERFORMANCE (Real Ames Housing data) =====")
print(f"R² Score: {r_squared:.4f}")
print(f"RMSE: ${np.sqrt(np.mean((y - predictions)**2)):,.2f}")


# ============================================================
# STEP 7: Verification vs sklearn
# ============================================================
sklearn_model = LinearRegression()
sklearn_model.fit(X_scaled.values, y)

print("\n===== VERIFICATION: Scratch OLS vs sklearn =====")
print(f"Scratch intercept : {ols.coefficients[0]:.2f}")
print(f"sklearn intercept  : {sklearn_model.intercept_:.2f}")

for feat, our_coef, sk_coef in zip(final_features, ols.coefficients[1:], sklearn_model.coef_):
    match = np.isclose(our_coef, sk_coef, atol=1)
    print(f"{feat:20s} | Scratch: {our_coef:10.2f} | sklearn: {sk_coef:10.2f} | Match: {match}")


# ============================================================
# STEP 8: Ek Example Prediction karo (real ghar ke liye)
# ============================================================
sample_house = df.iloc[0]
print(f"\n===== SAMPLE PREDICTION =====")
print(f"Actual SalePrice   : ${sample_house['SalePrice']:,.2f}")
print(f"Predicted SalePrice: ${predictions[0]:,.2f}")
