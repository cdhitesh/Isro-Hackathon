from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load artifacts
model = joblib.load("random_forest_crop.pkl")
columns = joblib.load("columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
# -----------------------------
# Utility: Preprocess Input
# -----------------------------
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df = df.reindex(columns=columns, fill_value=0)
    return df

# -----------------------------
# Prediction Logic
# -----------------------------
def predict_crop(user_input):
    X = preprocess_input(user_input)

    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    crop_name = label_encoder.inverse_transform([pred_class])[0]
    confidence = float(np.max(pred_proba))

    return crop_name, confidence

# -----------------------------
# API Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    crop, confidence = predict_crop(data)

    response = {
        "predicted_crop": crop,
        "confidence": round(confidence * 100, 2)
    }

    # 🔔 Alert logic
    THRESHOLD = 0.70  # 70%

    if confidence >= THRESHOLD:
        send_alert(crop, confidence)
        response["alert"] = "Alert sent (confidence high)"
    else:
        response["alert"] = "No alert (confidence low)"

    return jsonify(response)

# -----------------------------
# Alert Function (SMS / Email)
# -----------------------------
def send_alert(crop, confidence):
    message = f"""
    🌾 Crop Prediction Alert!
    Predicted Crop: {crop}
    Confidence: {confidence*100:.2f}%
    """

    print("ALERT TRIGGERED")
    print(message)

    # Call SMS / Email functions here
    # send_sms(message)
    # send_email(message)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
