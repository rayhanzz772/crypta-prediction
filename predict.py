import logging
import warnings
from pathlib import Path

import joblib

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    import numpy as np
    HAS_PANDAS = False

# Suppress warnings from scikit-learn
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "isolation_forest_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "feature_scaler.pkl"

# Load model dan scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

logger.info("Loaded model and scaler successfully")


# =========================
# Recursive Brute Force Score
# =========================
BRUTE_FORCE_TIERS = [
    (3, 1),    # >= 3 attempts: +1
    (5, 2),    # >= 5 attempts: +2 (cumulative: 3)
    (8, 3),    # >= 8 attempts: +3 (cumulative: 6)
    (12, 4),   # >= 12 attempts: +4 (cumulative: 10)
    (20, 5),   # >= 20 attempts: +5 (cumulative: 15)
]


def recursive_brute_force_score(failed_attempts, depth=0):
    if depth >= len(BRUTE_FORCE_TIERS):
        return 0

    threshold, points = BRUTE_FORCE_TIERS[depth]

    if failed_attempts >= threshold:
        return points + recursive_brute_force_score(failed_attempts, depth + 1)

    return 0


# =========================
# Rule Based Score
# =========================
def calculate_rule_score(row):

    score = 0

    # Failed attempts (recursive cumulative scoring)
    score += recursive_brute_force_score(row["failed_attempts"])

    # Login dini hari
    if 0 <= row["login_hour"] <= 4:
        score += 1

    # Device change
    if row["device_change"]:
        score += 1

    # IP change
    if row["ip_change"]:
        score += 1

    # Geo anomaly
    if row["geo_anomaly"]:
        score += 2

    # Access spike
    if row["access_count_10min"] >= 25:
        score += 2
    elif row["access_count_10min"] >= 15:
        score += 1

    # Session duration
    if row["session_duration_min"] >= 300:
        score += 2
    elif row["session_duration_min"] >= 180:
        score += 1

    # Endpoint exploration
    if row["unique_endpoints_visited"] >= 40:
        score += 2
    elif row["unique_endpoints_visited"] >= 20:
        score += 1

    # VPN
    if row["vpn_used"]:
        score += 1

    # Extreme combos
    if row["failed_attempts"] >= 5 and row["access_count_10min"] >= 20:
        score += 2

    if row["vpn_used"] and row["geo_anomaly"]:
        score += 2

    return score


# =========================
# Hybrid Score
# =========================
def hybrid_score(ml_score_norm, rule_score):

    ML_WEIGHT = 0.7
    RULE_WEIGHT = 0.3

    rule_norm = min(rule_score, 20) / 20

    final_score = ((1 - ml_score_norm) * ML_WEIGHT) + (rule_norm * RULE_WEIGHT)

    return final_score


# =========================
# Risk Level
# =========================
def final_risk(score):

    if score >= 0.75: return "HIGH"

    elif score >= 0.5: return "MEDIUM"
    
    else: return "LOW"



# =========================
# Prediction Pipeline
# =========================
def predict_anomaly(features):

    row = features

    if HAS_PANDAS:
        X = pd.DataFrame([{
            "login_hour": row["login_hour"],
            "day_of_week": row["day_of_week"],
            "session_duration_min": row["session_duration_min"],
            "failed_attempts": row["failed_attempts"],
            "device_change": int(row["device_change"]),
            "ip_change": int(row["ip_change"]),
            "geo_anomaly": int(row["geo_anomaly"]),
            "access_count_10min": row["access_count_10min"],
            "unique_endpoints_visited": row["unique_endpoints_visited"],
            "vpn_used": int(row["vpn_used"])
        }])
    else:
        X = np.array([[
            row["login_hour"],
            row["day_of_week"],
            row["session_duration_min"],
            row["failed_attempts"],
            int(row["device_change"]),
            int(row["ip_change"]),
            int(row["geo_anomaly"]),
            row["access_count_10min"],
            row["unique_endpoints_visited"],
            int(row["vpn_used"])
        ]])

    # Scaling
    X_scaled = scaler.transform(X)

    # ML Prediction
    ml_score = model.decision_function(X_scaled)[0]

    # Normalize ML score
    ml_score_norm = (ml_score + 0.5) / 1.0
    ml_score_norm = max(0, min(1, ml_score_norm))

    # Rule score
    rule_score = calculate_rule_score(row)

    # Hybrid score
    final_score = hybrid_score(ml_score_norm, rule_score)

    # Status & Risk level
    status = "ANOMALI" if final_score >= 0.5 else "NORMAL"
    risk_level = final_risk(final_score)

    logger.info(
        "\n"
        "==============================================================\n"
        "                      PREDICTION RESULT                       \n"
        "==============================================================\n"
        "  Status       : %s\n"
        "  Risk Level   : %s\n"
        "  Final Score  : %.4f\n"
        "  Rule Score   : %d / 20\n"
        "  ML Score     : %.4f (Norm: %.4f)\n"
        "==============================================================",
        status,
        risk_level,
        final_score,
        rule_score,
        ml_score,
        ml_score_norm,
    )

    return {
        "status": status,
        "risk_level": risk_level,
        "score": round(float(final_score), 4),
        "rule_score": int(rule_score),
        "ml_score": round(float(ml_score), 4)
    }