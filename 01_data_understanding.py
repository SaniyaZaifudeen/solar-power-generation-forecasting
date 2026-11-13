import pandas as pd

df = pd.read_csv("dataset/Plant_1_Generation_Data.csv")

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# Convert DATE_TIME to datetime
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)

print("\nUpdated Data Types:")
print(df.dtypes)

print("\nFirst 5 DATE_TIME values:")
print(df['DATE_TIME'].head())