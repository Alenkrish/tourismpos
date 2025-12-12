# ----------------------------- IMPORTS -----------------------------
import os
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# optional import
try:
    import joblib
except:
    joblib = None

# ----------------------------- PATH SETUP -----------------------------
# Streamlit Cloud executes from the repo root, so os.getcwd() is correct.
BASE = os.getcwd()
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")

# DATA FILE PATHS
CLEANED_ARRIVALS = os.path.join(DATA_DIR, "cleaned_arrivals.csv")
CLEANED_ATTR = os.path.join(DATA_DIR, "cleaned_attractions.csv")
CLEANED_MONTHLY = os.path.join(DATA_DIR, "cleaned_monthly_popularity.csv")
HIDDEN_GEMS = os.path.join(DATA_DIR, "cleaned_hidden_gems.csv")

# MODEL PATHS
ARRIVALS_MODEL_PATH = os.path.join(MODELS_DIR, "arrivals_model.joblib")
POPULARITY_MODEL_PATH = os.path.join(MODELS_DIR, "popularity_model.joblib")

# ----------------------------- STREAMLIT UI -----------------------------
st.set_page_config(page_title="Sri Lanka Tourism Predictor", layout="wide")
st.title("Sri Lanka Tourism — Month-based Prediction & Suggestions")

# ----------------------------- SAFE LOAD HELPERS -----------------------------
@st.cache_data
def load_csv(path, parse_dates=None):
    """Load CSV safely. Return None if missing."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"Error loading {path}: {e}")
        return None

@st.cache_resource
def load_model(path):
    """Load ML model safely."""
    if joblib is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except:
        return None

# ----------------------------- LOAD DATA -----------------------------
df_arrivals = load_csv(CLEANED_ARRIVALS, parse_dates=["date"])
df_attr = load_csv(CLEANED_ATTR)
df_monthly = load_csv(CLEANED_MONTHLY)
df_hidden = load_csv(HIDDEN_GEMS)

# ----------------------------- SIDEBAR -----------------------------
st.sidebar.header("Preferences")

selected_month = st.sidebar.selectbox("Select month", list(range(1, 13)), index=datetime.now().month - 1)
hidden_pref = st.sidebar.slider("Prefer hidden gems", 0.0, 1.0, 0.3)
num_results = st.sidebar.slider("Number of suggestions", 3, 15, 7)

category_filter = "All"
if df_monthly is not None and "category" in df_monthly.columns:
    categories = ["All"] + sorted(df_monthly["category"].dropna().unique().tolist())
    category_filter = st.sidebar.selectbox("Category", categories)

# ----------------------------- ARRIVALS PREDICTION -----------------------------
st.subheader("Predicted National Tourist Arrivals (Sri Lanka)")

def fallback_mean_prediction(df, month):
    try:
        return int(df[df["month"] == month]["arrivals"].mean())
    except:
        return None

predicted_arrivals = None

if df_arrivals is None:
    st.error(f"Missing arrivals file: cleaned_arrivals.csv")
else:
    df_arrivals["month"] = df_arrivals["date"].dt.month
    df_arrivals["year"] = df_arrivals["date"].dt.year

    model = load_model(ARRIVALS_MODEL_PATH)

    if model is None:
        predicted_arrivals = fallback_mean_prediction(df_arrivals, selected_month)
        if predicted_arrivals:
            st.metric("Predicted arrivals (historical mean)", f"{predicted_arrivals:,}")
    else:
        try:
            df_sorted = df_arrivals.sort_values("date")
            lag_1 = df_sorted["arrivals"].iloc[-1]
            lag_3 = df_sorted["arrivals"].iloc[-3] if len(df_sorted) >= 3 else lag_1
            lag_12 = df_sorted["arrivals"].iloc[-12] if len(df_sorted) >= 12 else lag_1

            latest_date = df_sorted["date"].max()
            next_year = latest_date.year if selected_month > latest_date.month else latest_date.year + 1

            X = pd.DataFrame([{
                "month": selected_month,
                "year": next_year,
                "lag_1": lag_1,
                "lag_3": lag_3,
                "lag_12": lag_12
            }])

            predicted_arrivals = int(model.predict(X)[0])
            st.metric("Predicted arrivals", f"{predicted_arrivals:,}")

        except:
            predicted_arrivals = fallback_mean_prediction(df_arrivals, selected_month)
            if predicted_arrivals:
                st.metric("Predicted arrivals (historical mean)", f"{predicted_arrivals:,}")

# ----------------------------- ATTRACTION SUGGESTIONS -----------------------------
st.subheader(f"Top attraction suggestions for month {selected_month}")

if df_monthly is None:
    st.error("Missing monthly popularity CSV: cleaned_monthly_popularity.csv")
else:
    df_monthly["month"] = pd.to_numeric(df_monthly["month"], errors="coerce")

    df_filtered = df_monthly[df_monthly["month"] == selected_month]

    if category_filter != "All":
        df_filtered = df_filtered[df_filtered["category"] == category_filter]

    if df_filtered.empty:
        st.warning("No attraction data found for this month.")
    else:
        pop_model = load_model(POPULARITY_MODEL_PATH)

        if pop_model:
            try:
                df_filtered["pred_score"] = pop_model.predict(
                    df_filtered[["category", "rating", "reviews", "month"]]
                )
            except:
                df_filtered["pred_score"] = df_filtered.get("popularity_score", 0)
        else:
            df_filtered["pred_score"] = df_filtered.get("popularity_score", 0)

        # Hidden gems logic
        if df_hidden is not None and "attraction_name" in df_hidden.columns:
            df_filtered = df_filtered.merge(
                df_hidden[["attraction_name", "hidden_gem_score"]],
                on="attraction_name",
                how="left"
            )

        # Normalize scores
        def normalize(s):
            return 100 * (s - s.min()) / (s.max() - s.min() + 1e-9)

        df_filtered["pred_norm"] = normalize(df_filtered["pred_score"])
        df_filtered["hidden_norm"] = normalize(df_filtered["hidden_gem_score"].fillna(0))

        df_filtered["final_score"] = (
            (1 - hidden_pref) * df_filtered["pred_norm"]
            + hidden_pref * df_filtered["hidden_norm"]
        )

        df_top = df_filtered.sort_values("final_score", ascending=False).head(num_results)

        st.dataframe(df_top[[
            "attraction_name", "category", "rating", "reviews",
            "pred_score", "hidden_gem_score", "final_score"
        ]].round(2))

# ----------------------------- END -----------------------------
st.caption("This app predicts arrivals and suggests attractions based on month, hidden gem preference, and popularity.")
