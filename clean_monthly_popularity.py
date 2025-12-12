import os
import pandas as pd
import numpy as np

# ---------- EDITABLE BASE PATH (already set for you) ----------
BASE_PATH = r"C:\Krishna\projects\POC\data"

# ---------- FILE PATHS ----------
monthly_path = os.path.join(BASE_PATH, "sri_lanka_attractions_monthly_popularity.csv")
hidden_gems_path = os.path.join(BASE_PATH, "sri_lanka_hidden_gems.csv")

out_monthly = os.path.join(BASE_PATH, "cleaned_monthly_popularity.csv")
out_hidden = os.path.join(BASE_PATH, "cleaned_hidden_gems.csv")

def clean_monthly():
    print("Loading monthly popularity:", monthly_path)
    df = pd.read_csv(monthly_path)
    print("Initial rows:", len(df))
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Expected columns: attraction_name, month, popularity_score, rating, reviews, lat, lon, category, district
    # If month exists as 'month' or 'month_num' try to detect
    if "month" not in df.columns and "month_num" in df.columns:
        df = df.rename(columns={"month_num":"month"})
    
    # drop rows with no attraction name
    df = df[df["attraction_name"].notna()]
    
    # Ensure month is integer 1-12; if month is text, try to parse
    try:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype('Int64')
    except Exception:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype('Int64')
    df = df[df["month"].between(1,12)]
    
    # Ensure numeric columns
    for col in ["popularity_score","rating","reviews","lat","lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Fill/populate missing popularity_score using a simple heuristic:
    # popularity_score = round( (rating/5)*60 + log1p(reviews)*3 ), if popularity_score missing
    if "popularity_score" not in df.columns:
        df["popularity_score"] = np.nan
    missing_pop = df["popularity_score"].isna()
    if missing_pop.any():
        print("Filling", missing_pop.sum(), "missing popularity_score values using rating+reviews heuristic.")
        # prepare defaults
        df["rating"] = df.get("rating", pd.Series(np.nan, index=df.index)).fillna(4.0)
        df["reviews"] = df.get("reviews", pd.Series(0, index=df.index)).fillna(0)
        df.loc[missing_pop, "popularity_score"] = (
            ((df.loc[missing_pop, "rating"]/5.0)*60) + (np.log1p(df.loc[missing_pop, "reviews"])*3)
        ).round(0)
    
    # Clip/popularity to 0-100 and integer
    df["popularity_score"] = df["popularity_score"].clip(0,100).round(0).astype(int)
    
    # Fill missing lat/lon with NaN (they should be numeric)
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    else:
        df["lat"] = np.nan
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    else:
        df["lon"] = np.nan
    
    # Keep only useful columns (preserve extras if present)
    cols_keep = ["attraction_name","district","category","type","month","rating","reviews","popularity_score","lat","lon"]
    existing = [c for c in cols_keep if c in df.columns]
    df_out = df[existing].copy()
    
    # Remove exact duplicate rows (same attraction+month)
    before = len(df_out)
    df_out = df_out.drop_duplicates(subset=["attraction_name","month"])
    after = len(df_out)
    if before != after:
        print(f"Removed {before-after} duplicate attraction-month rows.")
    
    # Reset index and save
    df_out = df_out.reset_index(drop=True)
    df_out.to_csv(out_monthly, index=False)
    print("Saved cleaned monthly popularity to:", out_monthly)
    print("Final rows:", len(df_out))
    return df_out

def clean_hidden():
    print("\nLoading hidden gems:", hidden_gems_path)
    dfh = pd.read_csv(hidden_gems_path)
    print("Initial rows:", len(dfh))
    
    # Standardize columns
    dfh.columns = dfh.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Ensure numeric columns
    for col in ["uniqueness_score","accessibility","eco_score","cultural_score","hidden_gem_score","lat","lon"]:
        if col in dfh.columns:
            dfh[col] = pd.to_numeric(dfh[col], errors="coerce")
    
    # Fill missing uniqueness/accessibility with reasonable defaults
    if "uniqueness_score" not in dfh.columns or dfh["uniqueness_score"].isna().all():
        dfh["uniqueness_score"] = 5.0
    if "accessibility" not in dfh.columns or dfh["accessibility"].isna().all():
        dfh["accessibility"] = 3
    
    # Compute hidden_gem_score if missing: (uniqueness*7 + (10-accessibility)*3 + eco/cultural*1.5) normalized to 0-100
    if "hidden_gem_score" not in dfh.columns or dfh["hidden_gem_score"].isna().all():
        eco = dfh.get("eco_score", pd.Series(5, index=dfh.index)).fillna(5)
        cult = dfh.get("cultural_score", pd.Series(5, index=dfh.index)).fillna(5)
        raw = dfh["uniqueness_score"]*7 + (10 - dfh["accessibility"])*3 + (eco + cult)*1.5
        # scale raw to 0-100
        raw_scaled = 100 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        dfh["hidden_gem_score"] = raw_scaled.round(0).astype(int)
        print("Computed hidden_gem_score for rows.")
    
    # Ensure lat/lon numeric
    if "lat" in dfh.columns:
        dfh["lat"] = pd.to_numeric(dfh["lat"], errors="coerce")
    else:
        dfh["lat"] = np.nan
    if "lon" in dfh.columns:
        dfh["lon"] = pd.to_numeric(dfh["lon"], errors="coerce")
    else:
        dfh["lon"] = np.nan
    
    # Remove duplicates by attraction_name
    dfh = dfh.drop_duplicates(subset=["attraction_name"])
    
    # Save
    dfh.to_csv(out_hidden, index=False)
    print("Saved cleaned hidden gems to:", out_hidden)
    print("Final rows:", len(dfh))
    return dfh

if __name__ == "__main__":
    try:
        monthly_df = clean_monthly()
    except FileNotFoundError:
        print("ERROR: monthly popularity CSV not found at:", monthly_path)
    except Exception as e:
        print("ERROR while cleaning monthly popularity:", e)
    try:
        hidden_df = clean_hidden()
    except FileNotFoundError:
        print("ERROR: hidden gems CSV not found at:", hidden_gems_path)
    except Exception as e:
        print("ERROR while cleaning hidden gems:", e)
