# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import predict_image

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Tomato Disease API running"})

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img_bytes = request.files["file"].read()
    preds = predict_image(img_bytes)
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)