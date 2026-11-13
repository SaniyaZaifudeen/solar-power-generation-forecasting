import joblib
import pandas as pd

# Load the saved model
model = joblib.load("models/solar_power_model.pkl")

# New input data
new_data = pd.DataFrame({
    'HOUR': [12],
    'MINUTE': [0],
    'AMBIENT_TEMPERATURE': [30],
    'MODULE_TEMPERATURE': [40],
    'IRRADIATION': [0.8]
})

# Make prediction
prediction = model.predict(new_data)

print("Predicted AC Power:", prediction[0])