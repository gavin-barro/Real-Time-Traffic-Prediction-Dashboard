# Module to load and preprocess data

import pandas as pd

# Load the CSV
df = pd.read_csv('data/processed/traffic_volume_2018_plus.csv')

# Standardize column names (handle both cases)
df.rename(columns=lambda col: col.lower(), inplace=True)
df.rename(columns={'yr': 'year', 'm': 'month', 'd': 'day', 'hh': 'hour', 'mm': 'minute'}, inplace=True)

# Create timestamp
df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

# === TIME FEATURES ===
df['hour'] = df['timestamp'].dt.hour  # Ensure consistent hour
df['dayofweek'] = df['timestamp'].dt.day_name()  # 'Monday', 'Tuesday', etc.
df['is_weekend'] = df['dayofweek'].isin(['Saturday', 'Sunday']).astype(int)
df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

# === HOLIDAY FEATURES ===
# List of holidays (2025)
holidays_2025 = {
    '2025-01-01': {'name': "New Year's Day", 'type': 'Federal'},
    '2025-01-20': {'name': "Martin Luther King Jr. Day", 'type': 'Federal'},
    '2025-02-17': {'name': "Presidents' Day", 'type': 'Federal'},
    '2025-03-31': {'name': 'Eid al-Fitr', 'type': 'Religious'},
    '2025-04-20': {'name': 'Easter Sunday', 'type': 'Religious'},
    '2025-05-26': {'name': 'Memorial Day', 'type': 'Federal'},
    '2025-06-08': {'name': 'Puerto Rican Day Parade', 'type': 'Local'},
    '2025-06-19': {'name': 'Juneteenth', 'type': 'Federal'},
    '2025-07-04': {'name': 'Independence Day', 'type': 'Federal'},
    '2025-09-01': {'name': 'Labor Day', 'type': 'Federal'},
    '2025-10-13': {'name': 'Columbus Day', 'type': 'Federal'},
    '2025-11-02': {'name': 'NYC Marathon', 'type': 'Local'},
    '2025-11-11': {'name': 'Veterans Day', 'type': 'Federal'},
    '2025-11-27': {'name': 'Thanksgiving', 'type': 'Federal'},
    '2025-12-15': {'name': 'Hanukkah', 'type': 'Religious'},
    '2025-12-25': {'name': 'Christmas Day', 'type': 'Federal'}
}

holiday_dates = set(holidays_2025.keys())
holiday_type_map = {k: v['type'] for k, v in holidays_2025.items()}

# Add holiday columns
df['date_str'] = df['timestamp'].dt.date.astype(str)
df['is_holiday'] = df['date_str'].isin(holiday_dates).astype(int)
df['holiday_type'] = df['date_str'].map(holiday_type_map).fillna('None')

# Define multi-day holiday periods (e.g., travel weekends)
holiday_periods = [
    ('2025-05-24', '2025-05-26'),  # Memorial Day weekend
    ('2025-11-26', '2025-11-30'),  # Thanksgiving weekend
    ('2025-12-21', '2026-01-01'),  # Christmas/New Year period
]
df['is_holiday_period'] = 0
for start, end in holiday_periods:
    df.loc[df['date_str'].between(start, end), 'is_holiday_period'] = 1

# Drop temporary column
df.drop(columns=['date_str'], inplace=True)

# Save updated file
output_path = 'data/processed/traffic_volume_with_features.csv'
df.to_csv(output_path, index=False)
print(f"Updated file saved: {output_path}")