import pandas as pd

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

# Create X and y
X = df[features]
y = df['AC_POWER']

# Display the selected data
print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)