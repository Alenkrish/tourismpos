# ===================== IMPORTS =====================
import os
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

try:
    import joblib
except:
    joblib = None

# ===================== PATHS =====================
BASE = os.getcwd()
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")

ARRIVALS_CSV = os.path.join(DATA_DIR, "cleaned_arrivals.csv")
MONTHLY_CSV = os.path.join(DATA_DIR, "cleaned_monthly_popularity.csv")
HIDDEN_CSV = os.path.join(DATA_DIR, "cleaned_hidden_gems.csv")

ARRIVALS_MODEL = os.path.join(MODELS_DIR, "arrivals_model.joblib")
POPULARITY_MODEL = os.path.join(MODELS_DIR, "popularity_model.joblib")

# ===================== PAGE CONFIG =====================
st.set_page_config("Sri Lanka Tourism App", layout="wide")
st.title("🇱🇰 Sri Lanka Tourism — AI Prediction & Analytics")

# ===================== LOADERS =====================
@st.cache_data
def load_csv(path, parse_dates=None):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=parse_dates)

@st.cache_resource
def load_model(path):
    if joblib is None or not os.path.exists(path):
        return None
    return joblib.load(path)

df_arrivals = load_csv(ARRIVALS_CSV, parse_dates=["date"])
df_monthly = load_csv(MONTHLY_CSV)
df_hidden = load_csv(HIDDEN_CSV)

arrivals_model = load_model(ARRIVALS_MODEL)
popularity_model = load_model(POPULARITY_MODEL)

# ===================== SIDEBAR NAVIGATION =====================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home & Data Exploration",
        "Visualizations",
        "Model Prediction",
        "Model Performance"
    ]
)

st.sidebar.divider()

# ===================== HOME / OVERVIEW =====================
if page == "Home & Data Exploration":

    st.header("📌 Overview")

    st.markdown("""
    This application uses **Sri Lanka tourism data** to:
    - Predict **monthly tourist arrivals**
    - Recommend **best tourist attractions**
    - Highlight **hidden & eco-friendly destinations**
    """)

    col1, col2, col3 = st.columns(3)

    if df_arrivals is not None:
        col1.metric("Total Records", len(df_arrivals))
        col2.metric("Years Covered", df_arrivals["date"].dt.year.nunique())
        col3.metric("Average Monthly Arrivals", int(df_arrivals["arrivals"].mean()))

    st.subheader("📄 Dataset Preview")
    st.dataframe(df_arrivals.head() if df_arrivals is not None else "Arrivals data missing")

# ===================== VISUALIZATIONS =====================
elif page == "Visualizations":

    st.header("📊 Tourism Visualizations")

    if df_arrivals is None:
        st.error("Arrivals data not available")
    else:
        df_arrivals["month"] = df_arrivals["date"].dt.month
        df_arrivals["year"] = df_arrivals["date"].dt.year

        st.subheader("Monthly Tourist Arrival Trend")
        st.line_chart(df_arrivals.set_index("date")["arrivals"])

        st.subheader("Average Arrivals by Month")
        monthly_avg = df_arrivals.groupby("month")["arrivals"].mean()
        st.bar_chart(monthly_avg)

# ===================== MODEL PREDICTION =====================
elif page == "Model Prediction":

    st.header("🔮 Tourist Arrival & Attraction Prediction")

    month = st.selectbox("Select Month", range(1, 13))
    hidden_weight = st.slider("Hidden Gem Preference", 0.0, 1.0, 0.3)

    st.subheader("Predicted Tourist Arrivals")

    if df_arrivals is None:
        st.warning("Arrival data not available")
    else:
        df_arrivals["month"] = df_arrivals["date"].dt.month

        if arrivals_model:
            latest = df_arrivals.sort_values("date")
            lag_1 = latest["arrivals"].iloc[-1]
            lag_3 = latest["arrivals"].iloc[-3]
            lag_12 = latest["arrivals"].iloc[-12]

            X = pd.DataFrame([{
                "month": month,
                "year": datetime.now().year,
                "lag_1": lag_1,
                "lag_3": lag_3,
                "lag_12": lag_12
            }])

            pred = int(arrivals_model.predict(X)[0])
            st.metric("Predicted Arrivals", f"{pred:,}")
        else:
            avg = int(df_arrivals[df_arrivals["month"] == month]["arrivals"].mean())
            st.metric("Predicted Arrivals (Avg)", f"{avg:,}")

    st.subheader("Top Tourist Attraction Suggestions")

    if df_monthly is None:
        st.warning("Monthly attraction data missing")
    else:
        df = df_monthly[df_monthly["month"] == month]

        if popularity_model:
            df["pred_score"] = popularity_model.predict(
                df[["category", "rating", "reviews", "month"]]
            )
        else:
            df["pred_score"] = df["popularity_score"]

        if df_hidden is not None:
            df = df.merge(df_hidden, on="attraction_name", how="left")

        df["final_score"] = (
            (1 - hidden_weight) * df["pred_score"] +
            hidden_weight * df["hidden_gem_score"]
        )

        st.dataframe(
            df.sort_values("final_score", ascending=False)
            .head(7)[
                ["attraction_name", "category", "rating", "final_score"]
            ]
        )

# ===================== MODEL PERFORMANCE =====================
elif page == "Model Performance":

    st.header("📈 Model Comparison & Performance")

    if arrivals_model is None:
        st.warning("Arrival prediction model not trained")
    else:
        st.success("Arrival prediction model loaded successfully")

    if popularity_model is None:
        st.warning("Attraction popularity model not trained")
    else:
        st.success("Attraction popularity model loaded successfully")

    st.markdown("""
    **Models Used**
    - Random Forest Regressor for tourist arrivals
    - Random Forest / Heuristic scoring for attractions

    **Evaluation Metrics**
    - MAE
    - RMSE
    - R² Score
    """)

    st.info("Model metrics can be extended with cross-validation results.")

# ===================== FOOTER =====================
st.caption("Sri Lanka Tourism AI App — Academic / Research Project")
