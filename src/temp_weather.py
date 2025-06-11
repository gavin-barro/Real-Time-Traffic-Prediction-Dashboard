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

# Map WMO Weather Codes to Descriptions
wmo_weather_codes = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
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


# === Load Traffic Data ===
traffic_path = 'data/processed/traffic_volume_2018_plus_feature_engineered.csv'
traffic_df = pd.read_csv(traffic_path)

# === Load & Concatenate Weather Data ===
weather_dir = 'data/processed'
weather_files = [f for f in os.listdir(weather_dir) if f.startswith("weather_") and f.endswith(".csv")]

weather_dfs = []
for file in weather_files:
    df = pd.read_csv(os.path.join(weather_dir, file))
    weather_dfs.append(df)

weather_df = pd.concat(weather_dfs, ignore_index=True)

# === Standardize columns for merge ===
# Make sure boro and borough columns match case and formatting
traffic_df['boro'] = traffic_df['boro'].str.title().str.strip()
weather_df['borough'] = weather_df['borough'].str.title().str.strip()

# === Merge on boro + datetime keys ===
merge_keys = ['boro', 'year', 'month', 'day', 'hour']

merged_df = pd.merge(
    traffic_df,
    weather_df.rename(columns={'borough': 'boro'})[
        ['boro', 'year', 'month', 'day', 'hour', 'temperature_2m', 'precipitation', 'weathercode', 'windspeed_10m']
    ],
    on=merge_keys,
    how='left'
)
merged_df['weather_description'] = merged_df['weathercode'].map(wmo_weather_codes).fillna("Unknown")

# Output Final Dataset
output_path = 'data/processed/final_training_data.csv'
merged_df.to_csv(output_path, index=False)
print(f"Merged dataset saved as: {output_path}")