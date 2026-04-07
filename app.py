from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained KNN model and preprocessors
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Map land cover type strings to numeric values
    land_cover_mapping = {
        "Forest": 3,
        "Urban": 1,
        "Agriculture": 2
    }

    if data["Land_Cover_Type"] not in land_cover_mapping:
        return jsonify({"error": "Invalid Land_Cover_Type"}), 400

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Rename columns to match training data
    df.columns = ['Slope_Angle (°)', 'Elevation (m)', 'Rainfall (mm/day)', 'Soil_Moisture (%)', 'Land_Cover_Type', 'Distance_To_River (m)', 'Seismic_Activity (Richter)']

    # Convert land cover type to numeric
    df["Land_Cover_Type"] = land_cover_mapping[data["Land_Cover_Type"]]

    # Scale numerical features
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = knn_model.predict(df_scaled)
    probability = knn_model.predict_proba(df_scaled)[:, 1]  # Probability of landslide risk
    result = "Landslide Risk" if prediction[0] == 1 else "No Landslide Risk"
    
    return jsonify({'prediction': result, 'risk_probability': float(probability[0])*100})

if __name__ == '__main__':
    app.run(debug=True)