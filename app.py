import logging

from flask import Flask, request, jsonify
from predict import predict_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.get("/")
def root():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    logger.info("Incoming POST /predict request payload: %s", data)

    result = predict_anomaly(data)
    return jsonify(result)

@app.route("/health")
def health_check():
    return jsonify({"status": "ok", "model_loaded": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010)