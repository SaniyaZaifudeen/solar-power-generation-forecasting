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

# Create and train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_

# Create a table
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("Feature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 5))

plt.bar(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=45)
plt.grid(axis='y')

plt.show()
