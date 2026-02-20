import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset (Kaggle flight price dataset expected)
df = pd.read_excel("Data_Train.xlsx")

# Feature engineering
df.dropna(inplace=True)
df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month

df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute

df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute

# Robust duration parsing
duration = list(df["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i].strip()

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))

df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins

# Features mapping
df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)
df["Total_Stops"] = df["Total_Stops"].astype(int)

# Drop unused columns before encoding
df.drop(["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration", "Route", "Additional_Info"], axis=1, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Ensure correct column order for model training
# This MUST match the order in app.py:predict()
feature_cols = [
    'Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 
    'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins',
    'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways', 
    'Airline_Jet Airways Business', 'Airline_Multiple carriers', 
    'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet', 
    'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
    'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad', 
    'Destination_Kolkata', 'Destination_New Delhi'
]

X = df[feature_cols]
y = df["Price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
pickle.dump(model, open("flight_rf.pkl", "wb"))

print("✅ flight_rf.pkl created successfully")
