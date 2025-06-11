""" Adding Weather to the Data set. """

import requests
import pandas as pd
import os

borough_coords = {
    "Manhattan": {
        "latitude": 40.7831,
        "longitude": -73.9712
    },
    "Brooklyn": {
        "latitude": 40.6782,
        "longitude": -73.9442
    },
    "Queens": {
        "latitude": 40.7282,
        "longitude": -73.7949
    },
    "Bronx": {
        "latitude": 40.8448,
        "longitude": -73.8648
    },
    "Staten Island": {
        "latitude": 40.5795,
        "longitude": -74.1502
    }
}

start_date = '2018-01-01'
end_date = '2024-12-31'

# Ensure output directory exists
output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)

# Loop through each borough and fetch weather data
for boro, coords in borough_coords.items():
    print(f"Fetching weather data for {boro}...")
    
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,weathercode,windspeed_10m",
        "timezone": "America/New_York"
    }
    
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["hourly"])
        
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["time"])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["hour"] = df["timestamp"].dt.hour
        df["borough"] = boro
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"weather_{boro.lower().replace(' ', '_')}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved weather data to {output_path}")
    else:
        print(f"Failed to fetch weather data for {boro}. Status code: {response.status_code}")