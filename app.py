from flask import Flask, request, jsonify
from predict import predict_anomaly

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    result = predict_anomaly(data)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)