from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)

model = joblib.load("model.pkl")

# Logging (AIOps basic part)
logging.basicConfig(filename="app.log", level=logging.INFO)

@app.route("/")
def home():
    return "ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        prediction = model.predict([np.array(data)])

        logging.info(f"Prediction success: {prediction[0]}")

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)