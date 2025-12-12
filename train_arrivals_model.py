# train_arrivals_model.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = r"C:\Krishna\projects\POC"
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load cleaned arrivals
arrivals_path = os.path.join(DATA_DIR, "cleaned_arrivals.csv")
print("Loading:", arrivals_path)
df = pd.read_csv(arrivals_path, parse_dates=["date"])

# 2. Feature engineering
df = df.sort_values("date").reset_index(drop=True)
# keep numeric target
y = df["arrivals"].astype(float)

# Features: month, year, lag_1, lag_3, lag_12
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
# if lag columns missing, create them
for lag in [1,3,12]:
    col = f"lag_{lag}"
    if col not in df.columns:
        df[col] = df["arrivals"].shift(lag)
# drop rows with NA (beginning of series)
df = df.dropna().reset_index(drop=True)
feature_cols = ["month","year","lag_1","lag_3","lag_12"]
X = df[feature_cols].astype(float)

print("Shape X,y:", X.shape, y.loc[X.index].shape)

# 3. Train/test split using time-series aware splitting
tscv = TimeSeriesSplit(n_splits=5)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# Cross-validated score (MAE negative -> convert)
scores = -cross_val_score(model, X, y.loc[X.index], cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
print("CV MAE (5-fold time-split):", scores.mean(), " +/- ", scores.std())

# Fit on full training data
model.fit(X, y.loc[X.index])

# Save model and feature list
model_path = os.path.join(MODELS_DIR, "arrivals_model.joblib")
meta_path = os.path.join(MODELS_DIR, "arrivals_model_meta.joblib")
joblib.dump(model, model_path)
joblib.dump(feature_cols, meta_path)
print("Saved arrivals model to:", model_path)
print("Saved feature list to:", meta_path)

# 4. Quick evaluation on last 12 months (holdout)
if len(X) > 24:
    X_train = X.iloc[:-12]
    y_train = y.loc[X_train.index]
    X_test = X.iloc[-12:]
    y_test = y.loc[X_test.index]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Holdout R2:", r2_score(y_test, preds))
    print("Holdout MAE:", mean_absolute_error(y_test, preds))
else:
    print("Not enough rows for holdout evaluation; model trained on all data.")
