# train_popularity_model.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

BASE = r"C:\Krishna\projects\POC"
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

monthly_path = os.path.join(DATA_DIR, "cleaned_monthly_popularity.csv")
print("Loading:", monthly_path)
df = pd.read_csv(monthly_path)

# Keep useful columns; create month as numeric if not present
if "month" not in df.columns:
    raise ValueError("cleaned_monthly_popularity.csv must contain 'month' column")
df = df.dropna(subset=["popularity_score"])
df["month"] = df["month"].astype(int)

# Features: category (cat), rating (num), reviews (num), month (num)
cat_feats = ["category"]
num_feats = ["rating", "reviews", "month"]
# ensure columns exist
for c in num_feats:
    if c not in df.columns:
        df[c] = 0

X = df[cat_feats + num_feats]
y = df["popularity_score"].astype(float)

# train/test split (random, stratify not necessary)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preproc = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ("num", StandardScaler(), num_feats)
])

model = Pipeline([
    ("pre", preproc),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Popularity model R2:", r2_score(y_test, pred))
print("Popularity model MAE:", mean_absolute_error(y_test, pred))

# Save
model_path = os.path.join(MODELS_DIR, "popularity_model.joblib")
joblib.dump(model, model_path)
print("Saved popularity model to:", model_path)
