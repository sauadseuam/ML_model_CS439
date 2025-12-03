import pandas as pd
import numpy as np


# Loads in the dataset
df = pd.read_csv("GCHIT_dataset.csv")
df.head()

## Cleaning and Preprocessing ##


# Fixing data types from float to int then creating a date column for better modeling and visualization
for col in ["year", "month", "week"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

df["date"] = pd.to_datetime(df["year"].astype(str) + df["week"].astype(str) + "-1", format="%G%V-%u")


# Drops Duplicate Rows
df = df.drop_duplicates()


# Defining Climate Related Variables
climate_cols = [
    "temperature_celsius",
    "temp_anomaly_celsius",
    "precipitation_mm",
    "heat_wave_days",
    "drought_indicator",
    "flood_indicator",
    "extreme_weather_events",
    "pm25_ugm3",
    "air_quality_index"
]

# Defining Health Related Variables
health_cols = [
    "respiratory_disease_rate",
    "cardio_mortality_rate",
    "vector_disease_risk_score",
    "waterborne_disease_incidents",
    "heat_related_admissions",
    "mental_health_index",
    "food_security_index"
]

# Dropping Rows with any Missing Values in Climate or Health Related Variables
df = df.dropna(subset=climate_cols + health_cols)


# Filling Missing Values for Less Critical Variables w Median or Mode
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

df.head()



