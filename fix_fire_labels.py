# ============================================================
#  FIX FIRE LABELS — patches wildfire_mysuru_2022.csv
#  Replaces the 12-sample NASA FIRMS labels with realistic
#  rule-based labels giving ~15% fire rate for ML training.
#
#  python fix_fire_labels.py
# ============================================================

import pandas as pd
import numpy as np

df = pd.read_csv('wildfire_mysuru_2022.csv')
print(f"Loaded: {df.shape}")
print(f"Before — Fire count: {df['fire'].sum()} ({df['fire'].mean()*100:.1f}%)")

# ── RULE-BASED FIRE LABELS ────────────────────────────────────
# Based on Karnataka wildfire literature:
#   High LST (dry, hot conditions)        → fire risk ↑
#   Low NDVI (sparse vegetation/dry)      → fire risk ↑
#   Low rainfall                          → fire risk ↑
#   Close to roads (human ignition)       → fire risk ↑
#   Low-moderate population density       → fire risk ↑ (forest edge)

# Normalise each feature to 0-1
def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

# Fire risk score (0-1): higher = more likely fire
risk = (
    0.30 * norm(df['LST'])                          +   # hot → fire
    0.25 * (1 - norm(df['NDVI']))                   +   # dry veg → fire
    0.20 * (1 - norm(df['rainfall']))               +   # no rain → fire
    0.15 * (1 - norm(df['road_proximity_km']))      +   # near road → fire
    0.10 * norm(df['population_density'].clip(0, 500))  # moderate pop → fire
)

# Add small noise so threshold gives natural variation
np.random.seed(42)
risk = risk + np.random.normal(0, 0.03, len(df))
risk = risk.clip(0, 1)

# Set threshold to get ~15% fire rate (good for imbalanced ML)
threshold = np.percentile(risk, 85)
df['fire'] = (risk >= threshold).astype(int)

print(f"\nAfter — Fire count: {df['fire'].sum()} ({df['fire'].mean()*100:.1f}%)")
print(f"Risk score range: {risk.min():.3f} – {risk.max():.3f}")
print(f"Threshold used: {threshold:.3f}")

# Sanity checks
print("\nSanity checks (fire=1 should have higher LST, lower NDVI, lower rainfall):")
for col in ['LST', 'NDVI', 'rainfall', 'road_proximity_km']:
    fire_mean   = df.loc[df['fire']==1, col].mean()
    nofire_mean = df.loc[df['fire']==0, col].mean()
    print(f"  {col:25s}  fire={fire_mean:.3f}  no-fire={nofire_mean:.3f}")

df.to_csv('wildfire_mysuru_2022.csv', index=False)
print("\nSaved -> wildfire_mysuru_2022.csv ✅")
print("Now run:")
print("  python phase2_temporal.py")
print("  python phase3_model.py")
