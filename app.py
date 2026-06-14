from flask import Flask, request, jsonify
from predict import predict_anomaly

app = Flask(__name__)

@app.get("/")
def root():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    result = predict_anomaly(data)

    print(result)
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