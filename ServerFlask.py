from flask import Flask, request, jsonify
import numpy as np
import joblib
import datetime

# ==========================================
#  LOAD MODEL & SCALER
# ==========================================
print("📦 Loading model...")
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
print("✅ Model and scaler loaded.")

app = Flask(__name__)

# ==========================================
#  HELPER
# ==========================================
def log_event(features, prediction, probability):
    with open("logs.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} | {prediction} | p={probability:.4f} | {features}\n")

# ==========================================
#  API ENDPOINT
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # scale input
        features_scaled = scaler.transform(features)

        # predict
        prob = model.predict_proba(features_scaled)[0, 1]
        prediction = "MALICIOUS" if prob > 0.5 else "BENIGN"

        # log
        log_event(features.tolist(), prediction, prob)

        return jsonify({
            "prediction": prediction,
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ==========================================
#  RUN SERVER
# ==========================================
if __name__ == "__main__":
    print("🚀 Server started on http://127.0.0.1:8080")
    app.run(host="127.0.0.1", port=8080)
