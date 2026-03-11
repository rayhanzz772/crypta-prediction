import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model/isolation_forest_model.pkl")
scaler = joblib.load("model/feature_scaler.pkl")


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

    if score < 0.35:
        return "LOW"

    elif score < 0.55:
        return "MEDIUM"

    elif score < 0.75:
        return "HIGH"

    return "CRITICAL"


# =========================
# Prediction Pipeline
# =========================
def predict_anomaly(features):

    row = features

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
    ml_pred = model.predict(X_scaled)[0]

    # Normalize ML score
    ml_score_norm = (ml_score + 0.5) / 1.0
    ml_score_norm = max(0, min(1, ml_score_norm))

    # Rule score
    rule_score = calculate_rule_score(row)

    # Hybrid score
    final_score = hybrid_score(ml_score_norm, rule_score)

    # Risk level
    risk_level = final_risk(final_score)

    return {
        "status": "ANOMALI" if ml_pred == -1 else "NORMAL",
        "risk_level": risk_level,
        "score": float(final_score),
        "rule_score": int(rule_score),
        "ml_score": float(ml_score)
    }