import pandas as pd
import os

# -------------------------------
# FIXED FILE PATH (YOUR FOLDER)
# -------------------------------
BASE_PATH = r"C:\Krishna\projects\POC\data"

# -----------------------------------------
# CLEAN 1: Monthly Tourist Arrivals Dataset
# -----------------------------------------

arrivals_path = os.path.join(BASE_PATH, "sri_lanka_monthly_arrivals_2010_2024.csv")
print("Loading:", arrivals_path)

df_arrivals = pd.read_csv(arrivals_path)

print("\nBefore Clean (Arrivals):")
print(df_arrivals.head(), "\n")

# Convert date column
df_arrivals["date"] = pd.to_datetime(df_arrivals["date"])

# Extract year & month
df_arrivals["year"] = df_arrivals["date"].dt.year
df_arrivals["month"] = df_arrivals["date"].dt.month

# Sort
df_arrivals = df_arrivals.sort_values("date").reset_index(drop=True)

# Drop missing rows
df_arrivals = df_arrivals.dropna()

print("After Clean (Arrivals):")
print(df_arrivals.head(), "\n")

# Save cleaned file
clean_arrivals_path = os.path.join(BASE_PATH, "cleaned_arrivals.csv")
df_arrivals.to_csv(clean_arrivals_path, index=False)

print("✔ Saved cleaned_arrivals.csv to:", clean_arrivals_path)


# -----------------------------------------
# CLEAN 2: Attractions Dataset
# -----------------------------------------

attr_path = os.path.join(BASE_PATH, "sri_lanka_attractions_master.csv")
print("\nLoading:", attr_path)

df_attr = pd.read_csv(attr_path)

print("\nBefore Clean (Attractions):")
print(df_attr.head(), "\n")

# Remove duplicates
df_attr = df_attr.drop_duplicates(subset=["attraction_name"])

# Fill missing values
df_attr["rating"] = df_attr["rating"].fillna(df_attr["rating"].mean())
df_attr["reviews"] = df_attr["reviews"].fillna(0)

# Convert coordinates
df_attr["lat"] = pd.to_numeric(df_attr["lat"], errors="coerce")
df_attr["lon"] = pd.to_numeric(df_attr["lon"], errors="coerce")

# Drop invalid coordinates
df_attr = df_attr.dropna(subset=["lat", "lon"])

print("After Clean (Attractions):")
print(df_attr.head(), "\n")

# Save cleaned file
clean_attr_path = os.path.join(BASE_PATH, "cleaned_attractions.csv")
df_attr.to_csv(clean_attr_path, index=False)

print("✔ Saved cleaned_attractions.csv to:", clean_attr_path)
