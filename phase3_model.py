# ============================================================
#  PHASE 3 — ML MODELS + SHAP EXPLAINABILITY
#  Models : Random Forest, XGBoost
#  Input  : wildfire_temporal.csv
#  Output : metrics, plots, saved models
#
#  HOW TO RUN:
#  python phase3_model.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("Imports OK ✅")


# ── 1. LOAD ───────────────────────────────────────────────────

df = pd.read_csv("wildfire_temporal.csv")
print(f"Dataset: {df.shape} | Fire%: {df['fire'].mean()*100:.1f}%")


# ── 2. FEATURES ───────────────────────────────────────────────

# REMOVE leakage columns + helper columns
DROP = [
    "latitude",
    "longitude",
    "LST_daily",
    "rainfall_daily",
    "NDVI_daily",
    "ndvi_before",
    "ndvi_after",
    "vegetation_loss"
]

X = df.drop(columns=DROP + ["fire"])
y = df["fire"]

FEATURE_NAMES = X.columns.tolist()
print(f"Features ({len(FEATURE_NAMES)}):")
print(FEATURE_NAMES)


# ── 3. TRAIN TEST SPLIT ──────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape}")
print(f"Test : {X_test.shape}")


# ── 4. HANDLE IMBALANCE USING SMOTE ──────────────────────────

smote = SMOTE(random_state=42)

X_train_bal, y_train_bal = smote.fit_resample(
    X_train,
    y_train
)

print(f"After SMOTE: {X_train_bal.shape}")
print(f"Balanced Fire%: {y_train_bal.mean()*100:.1f}%")


# ── 5. TRAIN RANDOM FOREST ───────────────────────────────────

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_bal, y_train_bal)

print("Random Forest done ✅")


# ── 6. TRAIN XGBOOST ─────────────────────────────────────────

print("\nTraining XGBoost...")

scale_pos = (
    (y_train_bal == 0).sum()
    /
    (y_train_bal == 1).sum()
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(
    X_train_bal,
    y_train_bal,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("XGBoost done ✅")


# ── 7. EVALUATION FUNCTION ───────────────────────────────────

def evaluate(model, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    cv = cross_val_score(
        model,
        X_train_bal,
        y_train_bal,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"AUC-ROC    : {auc:.4f}")
    print(f"AUC-PR     : {ap:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"5-Fold AUC : {cv.mean():.4f} +/- {cv.std():.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Fire", "Fire"]
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": auc,
        "ap": ap,
        "f1": f1,
        "cv_mean": cv.mean(),
        "cv_std": cv.std()
    }


r_rf = evaluate(rf, "Random Forest")
r_xgb = evaluate(xgb, "XGBoost")


# ── 8. SAVE MODELS ───────────────────────────────────────────

joblib.dump(rf, "rf_wildfire_model.joblib")
joblib.dump(xgb, "xgb_wildfire_model.joblib")

print("\nModels saved successfully ✅")


# ── 9. FINAL RESULTS ─────────────────────────────────────────

print("\n" + "=" * 55)
print("FINAL RESULTS")
print("=" * 55)

for r in [r_rf, r_xgb]:
    print(
        f"{r['name']:20s} "
        f"AUC={r['auc']:.4f}  "
        f"F1={r['f1']:.4f}  "
        f"CV={r['cv_mean']:.4f}+/-{r['cv_std']:.4f}"
    )

print("=" * 55)


# ── 10. POST-FIRE VEGETATION ANALYSIS ────────────────────────

print("\n==============================")
print("POST-FIRE VEGETATION ANALYSIS")
print("==============================")

avg_loss = df["vegetation_loss"].mean()
print(f"Average Estimated Vegetation Loss: {avg_loss:.2f}%")

high_damage = (df["vegetation_loss"] > 30).sum()
print(f"High Vegetation Damage Cases: {high_damage}")

print("\n==============================")
print("LOCATION-WISE FIRE & VEGETATION ANALYSIS")
print("==============================")

# 1. Add predictions to dataframe
df["predicted_fire"] = xgb.predict(X)

# 2. Filter high-risk locations
high_risk = df[df["predicted_fire"] == 1].copy()

# Sort by highest vegetation loss
high_risk = high_risk.sort_values(by="vegetation_loss", ascending=False)

# 3. Print top 5 dangerous locations
print("\nTop 5 High-Risk Locations:\n")

top5 = high_risk.head(5)

for i, row in top5.iterrows():
    print(f"Location ({row['latitude']:.4f}, {row['longitude']:.4f}) → "
          f"Vegetation Loss: {row['vegetation_loss']:.2f}%")

# 4. Heatmap visualization
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["vegetation_loss"],
    cmap="hot",
    alpha=0.7
)

plt.colorbar(scatter, label="Vegetation Loss (%)")
plt.title("Wildfire Vegetation Loss Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("vegetation_loss_heatmap.png", dpi=150)
plt.show()

print("\nHeatmap saved as vegetation_loss_heatmap.png ✅")