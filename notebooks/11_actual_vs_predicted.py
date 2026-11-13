import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load processed dataset
df = pd.read_csv("dataset/solar_plant1_processed.csv")

# Select features
features = [
    'HOUR',
    'MINUTE',
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'IRRADIATION'
]

X = df[features]
y = df['AC_POWER']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3
)

plt.xlabel("Actual AC Power")
plt.ylabel("Predicted AC Power")
plt.title("Actual vs Predicted AC Power")

plt.grid(True)
plt.show()
