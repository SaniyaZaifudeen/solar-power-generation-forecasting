import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("dataset/solar_plant1_processed.csv")

# Calculate average AC power for each hour
hourly_power = df.groupby('HOUR')['AC_POWER'].mean()

# Create graph
plt.figure(figsize=(10, 5))

plt.plot(
    hourly_power.index,
    hourly_power.values,
    marker='o'
)

plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power")
plt.title("Average Solar Power Generation by Hour")

plt.grid(True)
plt.show()
