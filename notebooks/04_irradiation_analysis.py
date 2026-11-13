import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("dataset/solar_plant1_processed.csv")

# Create scatter plot
plt.figure(figsize=(10, 5))

plt.scatter(
    df['IRRADIATION'],
    df['AC_POWER'],
    alpha=0.3
)

plt.xlabel("Irradiation")
plt.ylabel("AC Power")
plt.title("Irradiation vs AC Power")

plt.grid(True)
plt.show()
