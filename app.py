# app.py
"""
Streamlit app (robust + lazy-loading) for:
- Predicting Sri Lanka monthly tourist arrivals
- Suggesting attractions for a selected month

Paths are fixed for C:\Krishna\projects\POC
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ------------------ USER PATHS ------------------
BASE = r"C:\Krishna\projects\POC"
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")

CLEANED_ARRIVALS = os.path.join(DATA_DIR, "cleaned_arrivals.csv")
CLEANED_ATTR = os.path.join(DATA_DIR, "cleaned_attractions.csv")
CLEANED_MONTHLY = os.path.join(DATA_DIR, "cleaned_monthly_popularity.csv")
HIDDEN_GEMS = os.path.join(DATA_DIR, "cleaned_hidden_gems.csv")

ARRIVALS_MODEL_PATH = os.path.join(MODELS_DIR, "arrivals_model.joblib")
ARRIVALS_META_PATH = os.path.join(MODELS_DIR, "arrivals_model_meta.joblib")
POPULARITY_MODEL_PATH = os.path.join(MODELS_DIR, "popularity_model.joblib")

# ------------------ STREAMLIT PAGE SETUP ------------------
st.set_page_config(page_title="Sri Lanka Tourism — Predictor & Suggestions", layout="wide")
st.title("Sri Lanka Tourism — Month-based Prediction & Suggestions")

# ------------------ SAFE IO HELPERS ------------------
@st.cache_data(ttl=3600)
def safe_read_csv(path, parse_dates=None):
    """Return DataFrame or None. Cache results to avoid reloading repeatedly."""
    try:
        if path is None:
            return None
        if not os.path.exists(path):
            return None
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"Failed to read {os.path.basename(path)}: {e}")
        return None

@st.cache_resource
def load_model_safe(path):
    """Load joblib model if exists, else return None. Cached to avoid reloading."""
    try:
        if path is None or not os.path.exists(path):
            return None
        return joblib.load(path)
    except Exception as e:
        # do not raise — show warning in UI where used
        return None

# ------------------ LAZY LOAD DATA ------------------
df_arrivals = safe_read_csv(CLEANED_ARRIVALS, parse_dates=["date"])
df_attr = safe_read_csv(CLEANED_ATTR)
df_monthly = safe_read_csv(CLEANED_MONTHLY)
df_hidden = safe_read_csv(HIDDEN_GEMS)

# ------------------ SIDEBAR (USER INPUTS) ------------------
st.sidebar.header("Preferences")
selected_month = st.sidebar.selectbox("Select month", list(range(1,13)), index=datetime.now().month-1)
hidden_pref = st.sidebar.slider("Prefer hidden gems (0 = popular, 1 = hidden)", 0.0, 1.0, 0.3, step=0.1)
cat_list = ["All"]
if df_monthly is not None and "category" in df_monthly.columns:
    cat_list = ["All"] + sorted(df_monthly["category"].dropna().unique().tolist())
category_filter = st.sidebar.selectbox("Category", cat_list)
num_results = st.sidebar.slider("Number of suggestions", 3, 15, 7)

st.markdown("---")

# ------------------ PART 1: PREDICT NATIONAL ARRIVALS ------------------
st.subheader("Predicted National Tourist Arrivals (Sri Lanka)")

# Helper to compute fallback prediction
def historical_month_mean(df, month, window_months=60):
    try:
        s = df[df["month"]==month]["arrivals"]
        if s.empty:
            return None
        # use recent window if available
        return int(s.tail(window_months).mean())
    except Exception:
        return None

pred_arrivals = None
arrivals_model = None

# lazy-load arrivals model only when needed
if os.path.exists(ARRIVALS_MODEL_PATH):
    with st.spinner("Loading arrivals model..."):
        arrivals_model = load_model_safe(ARRIVALS_MODEL_PATH)

if df_arrivals is None:
    st.warning(f"Missing arrivals file: {os.path.basename(CLEANED_ARRIVALS)}. Predictions will be unavailable.")
else:
    # ensure date col is datetime
    if "date" in df_arrivals.columns:
        try:
            df_arrivals["date"] = pd.to_datetime(df_arrivals["date"], errors="coerce")
            df_arrivals["month"] = df_arrivals["date"].dt.month
            df_arrivals["year"] = df_arrivals["date"].dt.year
        except Exception:
            pass

    if arrivals_model is not None:
        # build a sensible feature row using the latest known lags
        try:
            arr_series = df_arrivals.sort_values("date")["arrivals"].reset_index(drop=True)
            lag_1 = int(arr_series.iloc[-1])
            lag_3 = int(arr_series.iloc[-3]) if len(arr_series) >= 3 else lag_1
            lag_12 = int(arr_series.iloc[-12]) if len(arr_series) >= 12 else lag_1
            latest_date = pd.to_datetime(df_arrivals["date"]).max()
            next_year = latest_date.year if selected_month > latest_date.month else latest_date.year + 1

            X_new = pd.DataFrame([{
                "month": int(selected_month),
                "year": int(next_year),
                "lag_1": lag_1,
                "lag_3": lag_3,
                "lag_12": lag_12
            }])
            try:
                pred_arrivals = int(arrivals_model.predict(X_new)[0])
                st.metric("Predicted arrivals (model)", f"{pred_arrivals:,}")
            except Exception:
                pred_arrivals = historical_month_mean(df_arrivals, selected_month)
                if pred_arrivals is not None:
                    st.metric("Predicted arrivals (fallback mean)", f"{pred_arrivals:,}")
                else:
                    st.info("Not enough historical data to predict arrivals.")
        except Exception:
            pred_arrivals = historical_month_mean(df_arrivals, selected_month)
            if pred_arrivals is not None:
                st.metric("Predicted arrivals (fallback mean)", f"{pred_arrivals:,}")
    else:
        # model missing -> fallback
        pred_arrivals = historical_month_mean(df_arrivals, selected_month)
        if pred_arrivals is not None:
            st.metric("Predicted arrivals (historical mean)", f"{pred_arrivals:,}")
        else:
            st.info("Not enough historical data to predict arrivals.")

    # chart historical arrivals if available
    try:
        if "date" in df_arrivals.columns and "arrivals" in df_arrivals.columns and not df_arrivals.empty:
            st.line_chart(df_arrivals.set_index("date")["arrivals"])
    except Exception:
        pass

st.markdown("---")

# ------------------ PART 2: SUGGEST ATTRACTIONS FOR THE MONTH ------------------
st.subheader(f"Top attraction suggestions for month {selected_month}")

if df_monthly is None:
    st.error(f"Missing monthly popularity CSV: {os.path.basename(CLEANED_MONTHLY)}")
else:
    # ensure month numeric
    try:
        df_monthly["month"] = pd.to_numeric(df_monthly["month"], errors="coerce").astype("Int64")
    except Exception:
        pass

    dfq = df_monthly[df_monthly["month"]==int(selected_month)].copy()
    if category_filter != "All":
        dfq = dfq[dfq["category"]==category_filter]

    if dfq.empty:
        st.warning("No attractions found for this month / filter.")
    else:
        # lazy-load popularity model if present
        pop_model = None
        if os.path.exists(POPULARITY_MODEL_PATH):
            with st.spinner("Loading popularity model..."):
                pop_model = load_model_safe(POPULARITY_MODEL_PATH)

        # create prediction column
        if pop_model is not None:
            try:
                X = dfq[["category","rating","reviews","month"]]
                preds = pop_model.predict(X)
                dfq["pred_score"] = preds
            except Exception:
                dfq["pred_score"] = dfq.get("popularity_score", 0)
        else:
            dfq["pred_score"] = dfq.get("popularity_score", 0)

        # hidden gem score merge or compute
        if df_hidden is not None and "attraction_name" in df_hidden.columns:
            # merge on attraction name if present
            try:
                dfq = dfq.merge(df_hidden[["attraction_name","hidden_gem_score"]], on="attraction_name", how="left")
            except Exception:
                dfq["hidden_gem_score"] = dfq.get("hidden_gem_score", 0)
        else:
            # compute a simple hidden_gem_score: fewer reviews + good rating
            try:
                rev_rank = dfq["reviews"].rank(ascending=True)
                rating_norm = (dfq["rating"] - dfq["rating"].min()) / (dfq["rating"].max() - dfq["rating"].min() + 1e-9)
                dfq["hidden_gem_score"] = ((rev_rank / rev_rank.max())*60 + rating_norm*40).round(0)
            except Exception:
                dfq["hidden_gem_score"] = 0

        # normalize scores to 0-100 safely
        def norm0to100(s):
            try:
                if s.max() == s.min():
                    return pd.Series(50, index=s.index)
                return 100*(s - s.min())/(s.max() - s.min())
            except Exception:
                return pd.Series(50, index=s.index)

        dfq["pred_norm"] = norm0to100(dfq["pred_score"].fillna(0))
        dfq["hidden_norm"] = norm0to100(dfq["hidden_gem_score"].fillna(0))
        dfq["final_score"] = (1 - hidden_pref)*dfq["pred_norm"] + hidden_pref*dfq["hidden_norm"]

        df_top = dfq.sort_values("final_score", ascending=False).head(num_results).reset_index(drop=True)

        # Left: list, Right: map
        left, right = st.columns([1,1])
        with left:
            st.write(f"Top {len(df_top)} attractions (weighted):")
            for i, r in df_top.iterrows():
                st.markdown(f"**{i+1}. {r['attraction_name']}**  \nCategory: {r.get('category','-')}  — District: {r.get('district','-')}")
                st.write(f"Pred: {r.get('pred_score', 0):.1f} — Hidden: {r.get('hidden_gem_score', 0):.0f} — Final: {r.get('final_score',0):.1f}")
                st.write("---")
        with right:
            # Map if coords available
            map_df = df_top.dropna(subset=["lat","lon"])[["lat","lon","attraction_name","final_score"]]
            if not map_df.empty:
                map_df = map_df.rename(columns={"lat":"latitude","lon":"longitude"})
                try:
                    st.map(map_df)
                except Exception:
                    st.info("Map rendering failed — ensure lat/lon numeric.")
            else:
                st.info("No coordinates available for top results.")

        # suggestions table & download
        display_cols = [c for c in ["attraction_name","district","category","rating","reviews","pred_score","hidden_gem_score","final_score","lat","lon"] if c in df_top.columns]
        st.subheader("Suggestions Table")
        st.dataframe(df_top[display_cols].round(2))

        csv = df_top[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download suggestions CSV", csv, file_name=f"suggestions_month_{selected_month}.csv", mime="text/csv")

st.markdown("---")
st.caption("Note: Models are loaded lazily. If models are missing, app uses historical averages / heuristics. Train the models and place .joblib files in the models folder to enable model predictions.")
