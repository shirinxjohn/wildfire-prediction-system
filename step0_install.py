# ============================================================
#  STEP 0 — RUN THIS FIRST, ONCE
#  python step0_install.py
# ============================================================

import subprocess, sys

packages = [
    "earthengine-api",
    "google-api-python-client",
    "google-auth",
    "google-auth-oauthlib",
    "geopandas",
    "osmnx",
    "shapely",
    "scikit-learn",
    "xgboost",
    "shap",
    "imbalanced-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "joblib",
]

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("\nAll dependencies installed ✅")
print("Next: earthengine authenticate   (then python phase1_dataset.py)")
