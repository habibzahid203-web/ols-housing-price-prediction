"""
DAY 2 PROJECT — Housing Valuation Engine (OLS from Scratch)
Goal: Raw/messy real estate dataset le kar, use clean karna,
      OLS estimator khud (Normal Equation se) likhna,
      aur VIF se multi-collinearity check karke unstable features drop karna.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)

# ============================================================
# STEP 1: Ek "MESSY" real estate dataset banate hain
# ============================================================
# Real world mein data hamesha "clean" nahi milta — is liye hum
# jaan-boojh kar missing values aur multi-collinear features
# dataset mein daal rahe hain, taake asal problems se deal karna
# seekh sakein (jaisa project mein bola gaya hai).

n = 500  # 500 ghar (rows)

sqft = np.random.normal(1800, 600, n).clip(400, 5000)          # ghar ka size
num_rooms = (sqft / 350) + np.random.normal(0, 0.5, n)          # ROOMS sqft se strongly correlated hai (collinearity)!
age = np.random.randint(0, 50, n)                                # ghar ki umar (saal)
distance_to_city = np.random.exponential(5, n).clip(0.5, 30)    # city center se doori (km)
crime_rate = np.random.exponential(3, n).clip(0, 20)             # crime rate area ki

# Asal price formula (jo hum baad mein optimizer se "seekhna" chahte hain)
price = (
    50000
    + sqft * 120
    + num_rooms * 8000
    - age * 800
    - distance_to_city * 3000
    - crime_rate * 1500
    + np.random.normal(0, 15000, n)   # random noise (real duniya mein hamesha noise hota hai)
)

df = pd.DataFrame({
    'sqft': sqft,
    'num_rooms': num_rooms,
    'age': age,
    'distance_to_city': distance_to_city,
    'crime_rate': crime_rate,
    'price': price
})

# Jaan-boojh kar kuch MISSING VALUES daal rahe hain (real messy data jaisa)
missing_indices = np.random.choice(df.index, size=40, replace=False)
df.loc[missing_indices, 'crime_rate'] = np.nan

missing_indices2 = np.random.choice(df.index, size=25, replace=False)
df.loc[missing_indices2, 'age'] = np.nan

print("===== RAW MESSY DATA (pehle 5 rows) =====")
print(df.head())
print(f"\nTotal missing values:\n{df.isnull().sum()}")


# ============================================================
# STEP 2: Manual Data Cleaning
# ============================================================
# Missing values ko "median" se fill karte hain (mean ke bajaye
# median use karna behtar hai kyunki ye outliers se kam affect hota hai)

for col in ['crime_rate', 'age']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"'{col}' ke missing values median ({median_val:.2f}) se fill kiye")

print(f"\nMissing values ab: {df.isnull().sum().sum()}")


# ============================================================
# STEP 3: Multi-collinearity Check karo (VIF se)
# ============================================================
# VIF (Variance Inflation Factor) batata hai ke ek feature dusre
# features se kitna "predictable" hai. VIF > 10 ka matlab hai
# strong multi-collinearity — is feature ko hataana chahiye.

feature_cols = ['sqft', 'num_rooms', 'age', 'distance_to_city', 'crime_rate']
X_vif = df[feature_cols].copy()

vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("\n===== VIF SCORES (Multi-collinearity check) =====")
print(vif_data)

# IMPORTANT: Jab do features aapas mein collinear hon (dono ka VIF high ho),
# to EK SATH dono ko drop nahi karte -- sirf sabse ZYADA VIF wale feature ko
# hatate hain, VIF dobara calculate karte hain, aur ye process tab tak
# repeat karte hain jab tak sab features ka VIF < 10 na ho jaye.
# (Iterative elimination -- yehi asal/correct tareeqa hai)

current_features = feature_cols.copy()
while True:
    X_temp = df[current_features].copy()
    vifs = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
    max_vif = max(vifs)
    if max_vif <= 10:
        break
    worst_feature = current_features[vifs.index(max_vif)]
    print(f"Dropping '{worst_feature}' (VIF = {max_vif:.2f}) -- sabse unstable feature")
    current_features.remove(worst_feature)

final_features = current_features
print(f"\nFinal features (in par use honge): {final_features}")


# ============================================================
# STEP 4: Feature Scaling (Manual Standardization)
# ============================================================
# Standardization formula: (x - mean) / std_deviation
# Isse sab features ek jaise scale pe aa jaate hain (0 ke around)

def standardize(column):
    return (column - column.mean()) / column.std()

X_scaled = df[final_features].apply(standardize)
y = df['price'].values


# ============================================================
# STEP 5: OLS Estimator SCRATCH SE likhna (Normal Equation)
# ============================================================
# Normal Equation Formula:
#   beta = (X^T X)^(-1) X^T y
#
# Ye formula seedha calculus se derive hota hai (cost function ka
# derivative zero rakh kar solve karne se) — isliye iterative
# gradient descent ki tarah loop nahi chalana parta, seedha
# ek line mein answer mil jata hai!

class OLSEstimator:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Intercept (bias) term add karte hain -- ek column of 1s
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # Normal Equation: beta = (X^T X)^-1 X^T y
        XtX = X_with_intercept.T @ X_with_intercept
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_with_intercept.T @ y

        self.coefficients = XtX_inv @ Xty
        return self

    def predict(self, X):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept @ self.coefficients


# Model train karo
ols = OLSEstimator()
ols.fit(X_scaled.values, y)

print("\n===== OLS COEFFICIENTS (Scratch se nikale gaye) =====")
print(f"Intercept: {ols.coefficients[0]:.2f}")
for feat, coef in zip(final_features, ols.coefficients[1:]):
    print(f"{feat}: {coef:.2f}")


# ============================================================
# STEP 6: Model ki Performance Check karo (R² Score)
# ============================================================
predictions = ols.predict(X_scaled.values)

ss_res = np.sum((y - predictions) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\n===== MODEL PERFORMANCE =====")
print(f"R² Score: {r_squared:.4f}  (1.0 = perfect, 0 = bekar)")
print(f"RMSE: {np.sqrt(np.mean((y - predictions)**2)):.2f}")


# ============================================================
# STEP 7: Verification -- sklearn ke LinearRegression se compare
# ============================================================
from sklearn.linear_model import LinearRegression

sklearn_model = LinearRegression()
sklearn_model.fit(X_scaled.values, y)

print("\n===== VERIFICATION: Scratch OLS vs sklearn =====")
print(f"Scratch intercept : {ols.coefficients[0]:.2f}")
print(f"sklearn intercept  : {sklearn_model.intercept_:.2f}")
print(f"Match? {np.isclose(ols.coefficients[0], sklearn_model.intercept_, atol=1)}")

for feat, our_coef, sk_coef in zip(final_features, ols.coefficients[1:], sklearn_model.coef_):
    match = np.isclose(our_coef, sk_coef, atol=1)
    print(f"{feat:20s} | Scratch: {our_coef:10.2f} | sklearn: {sk_coef:10.2f} | Match: {match}")
