import pandas as pd
import numpy as np

print("Imports OK ✅")

# LOAD DATA
df = pd.read_csv("wildfire_mysuru_2022.csv")
print("Loaded:", df.shape)

# SORT + DAY
np.random.seed(42)
df = df.sort_values(["latitude", "longitude"]).reset_index(drop=True)

df["day_of_year"] = (df.index % 365) + 1
day_rad = 2 * np.pi * df["day_of_year"] / 365
n = len(df)

# DAILY FEATURES
df["LST_daily"] = (
    df["LST"]
    + 8 * np.sin(day_rad - np.pi / 6)
    + np.random.normal(0, 1.5, n)
).clip(0, 80)

df["rainfall_daily"] = (
    df["rainfall"]
    + 3 * np.sin(day_rad - np.pi / 2)
    + np.random.normal(0, 0.8, n)
).clip(0)

df["NDVI_daily"] = (
    df["NDVI"]
    - 0.05 * np.cos(day_rad)
    + np.random.normal(0, 0.02, n)
).clip(-1, 1)

print("Daily features done ✅")

# ROLLING FEATURES
df = df.sort_values("day_of_year").reset_index(drop=True)

for var, short in [
    ("LST_daily", "LST"),
    ("rainfall_daily", "rainfall"),
    ("NDVI_daily", "NDVI")
]:
    df[f"{short}_3d_avg"] = df[var].rolling(3, min_periods=1).mean()
    df[f"{short}_7d_avg"] = df[var].rolling(7, min_periods=1).mean()
    df[f"{short}_7d_std"] = df[var].rolling(7, min_periods=1).std().fillna(0)

print("Rolling features done ✅")

# DROUGHT INDEX
df["drought_index"] = (
    df["LST_7d_avg"] / (df["rainfall_7d_avg"] + 0.1)
).clip(0, 200)

print("Drought index done ✅")

# VEGETATION LOSS
df["ndvi_before"] = df["NDVI_7d_avg"]
df["ndvi_after"] = df["NDVI"]

df["vegetation_loss"] = (
    (df["ndvi_before"] - df["ndvi_after"])
    / (df["ndvi_before"] + 0.001)
) * 100

df["vegetation_loss"] = df["vegetation_loss"].clip(0, 100)

print("Vegetation loss added ✅")

# IMPORTANT:
# DO NOT include damage_severity in training dataset
# only keep numerical features

final_columns = [
    "latitude",
    "longitude",
    "NDVI",
    "LST",
    "rainfall",
    "population_density",
    "road_proximity_km",

    "LST_daily",
    "rainfall_daily",
    "NDVI_daily",

    "LST_3d_avg",
    "rainfall_3d_avg",
    "NDVI_3d_avg",

    "LST_7d_avg",
    "rainfall_7d_avg",
    "NDVI_7d_avg",

    "LST_7d_std",
    "rainfall_7d_std",
    "NDVI_7d_std",

    "drought_index",
    "day_of_year",

    "ndvi_before",
    "ndvi_after",
    "vegetation_loss",

    "fire"
]

df = df[final_columns].dropna().reset_index(drop=True)

print("Final shape:", df.shape)

df.to_csv("wildfire_temporal.csv", index=False)

print("Saved wildfire_temporal.csv ✅")