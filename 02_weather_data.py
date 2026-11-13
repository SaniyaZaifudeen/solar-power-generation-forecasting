import pandas as pd

# Load both datasets
generation = pd.read_csv("dataset/Plant_1_Generation_Data.csv")
weather = pd.read_csv("dataset/Plant_1_Weather_Sensor_Data.csv")

# Convert DATE_TIME to datetime
generation['DATE_TIME'] = pd.to_datetime(
    generation['DATE_TIME'],
    dayfirst=True
)

weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'])

# Merge generation and weather data
df = pd.merge(
    generation,
    weather,
    on=['DATE_TIME', 'PLANT_ID'],
    how='inner'
)

print("Merged Data Shape:")
print(df.shape)

print("\nMerged Columns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary columns
df = df.drop(columns=['PLANT_ID', 'SOURCE_KEY_y'])

# Rename SOURCE_KEY_x
df = df.rename(columns={'SOURCE_KEY_x': 'SOURCE_KEY'})

print("\nFinal Columns:")
print(df.columns)

print("\nFinal Shape:")
print(df.shape)

# Extract time-based features
df['HOUR'] = df['DATE_TIME'].dt.hour
df['MINUTE'] = df['DATE_TIME'].dt.minute
df['DAY'] = df['DATE_TIME'].dt.day
df['MONTH'] = df['DATE_TIME'].dt.month

print("\nNew Features:")
print(df[['DATE_TIME', 'HOUR', 'MINUTE', 'DAY', 'MONTH']].head(10))

# Save processed dataset
df.to_csv("dataset/solar_plant1_processed.csv", index=False)

print("\nProcessed dataset saved successfully!")

# Average AC power by hour
hourly_power = df.groupby('HOUR')['AC_POWER'].mean()

print("\nAverage AC Power by Hour:")
print(hourly_power)