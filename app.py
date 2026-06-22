import logging

from flask import Flask, request, jsonify
from predict import predict_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.get("/")
def root():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    logger.info("/predict request json=%s", data)

    result = predict_anomaly(data)

    logger.info("/predict response=%s", result)
    return jsonify(result)

@app.route("/health")
def health_check():
    try:
        predict_anomaly({})
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "error", "model_loaded": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010)