import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("dataset/solar_plant1_processed.csv")

# Create scatter plot
plt.figure(figsize=(10, 5))

plt.scatter(
    df['MODULE_TEMPERATURE'],
    df['AC_POWER'],
    alpha=0.3
)

plt.xlabel("Module Temperature")
plt.ylabel("AC Power")
plt.title("Module Temperature vs AC Power")

plt.grid(True)
plt.show()
