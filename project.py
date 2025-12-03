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



# Standardizing Climate Related Variables
climate_z = (df[climate_cols] - df[climate_cols].mean()) / df[climate_cols].std(ddof=0)
climate_z.columns = [col + "_z" for col in climate_z.columns]

# Standardizing Health Related Variables
health_z = (df[health_cols] - df[health_cols].mean()) / df[health_cols].std(ddof=0)
health_z.columns = [col + "_z" for col in health_z.columns]

# Adding to the main dataframe
df = pd.concat([df, climate_z], axis=1)
df = pd.concat([df, health_z], axis=1)

# Making Climate Shock and Health Impact Values For ML model
df["climate_shock_value"] = climate_z.mean(axis=1)
df["health_harm_value"] = health_z.mean(axis=1)


## ML Model Creation ##
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Socioeconomic Columns
socio_cols = [
    "population_millions",
    "healthcare_access_index",
    "gdp_per_capita_usd"
]

# Create a copy for ML so we dont mess up the original df
ml_df = df.copy()

# We Turn Categories into Numbers since ML models need numerical inputs
ml_df = pd.get_dummies(ml_df, columns=["income_level", "region"], drop_first=True)


# X is our inputs ( climate + the socioeconomic factors + numerical inputs for categories )
X = ml_df[climate_cols + socio_cols + [col for col in ml_df.columns if col.startswith("income_level_") or col.startswith("region_")]]

# Y is our output that we want the model to predict
y = ml_df["health_harm_value"]

# Train-test split (80% train, 20% test) so the model will learn from 80% and then 20% will be kept hidden until evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## MODEL 1 : LINEAR REGRESSION ##

# Creates Empty Linear Regression Model
linreg = LinearRegression()

# Fits the model to the training data
linreg.fit(X_train, y_train)

# Makes predictions on the test data
y_pred_lr = linreg.predict(X_test)

# Calculates 2 different metrics to guage how well the model did
rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5
r2_lr = r2_score(y_test, y_pred_lr)

# Prints out the results for Root Mean Squared Error and R² Score

print("Linear Regression RMSE:", rmse_lr)   # RMSE = average error magnitude
print("Linear Regression R²:", r2_lr)       # R² = proportion of variance explained


## MODEL 2 : RANDOM FOREST REGRESSOR ##

# makes 300 decision trees and averages their outputs
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# fits the model to the training data
rf.fit(X_train, y_train)

# makes predictions on the test data
y_pred_rf = rf.predict(X_test)

# Calculates 2 different metrics to guage how well the model did
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

# Prints out the results for Root Mean Squared Error and R² Score
print("Random Forest RMSE:", rmse_rf) # RMSE = average error magnitude
print("Random Forest R²:", r2_rf)     # R² = proportion of variance explained


# using the random forest model for all data points
ml_df["health_harm_pred"] = rf.predict(X)

# finding resilience gap and resilience score 
# + = more harm than expected AKA less resilient, 
# - = less harm than expected AKA more resilient
ml_df["resilience_gap"] = ml_df["health_harm_pred"] - ml_df["health_harm_value"]

# Converting resilience gap to a 0-100 scale for easier interpretation (higher score = more resilient)
ml_df["resilience_score_raw"] = -ml_df["resilience_gap"]

# Scaling resilience score to 0-100
min_s = ml_df["resilience_score_raw"].min()
max_s = ml_df["resilience_score_raw"].max()
ml_df["CSRI_ML_0_100"] = 100 * (ml_df["resilience_score_raw"] - min_s) / (max_s - min_s)

# Displaying Average CSRI by Country
country_csri = ml_df.groupby("country_name")["CSRI_ML_0_100"].mean().sort_values(ascending=False)

print(country_csri.head(10))   # most resilient
print(country_csri.tail(10))   # least resilient



