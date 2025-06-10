import pandas as pd

# Load the CSV
df = pd.read_csv('data/raw/traffic_volume_2021_plus.csv')

# Rename columns for clarity
df.rename(columns={'yr': 'year', 'm': 'month', 'd': 'day', 'hh': 'hour', 'mm': 'minute'}, inplace=True)

# Create timestamp
df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

# Define 2025 holidays (federal, religious, local)
holidays_2025 = {
    '2025-01-01': {'name': "New Year's Day", 'type': 'Federal'},
    '2025-01-20': {'name': "Martin Luther King Jr. Day", 'type': 'Federal'},
    '2025-01-20': {'name': 'Inauguration Day', 'type': 'Federal'},  # Overlaps with MLK Day
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
    '2025-12-15': {'name': 'Hanukkah', 'type': 'Religious'},  # First day
    '2025-12-25': {'name': 'Christmas Day', 'type': 'Federal'}
}

# Define holiday periods (high-traffic travel windows)
holiday_periods = [
    ('2025-05-24', '2025-05-26'),  # Memorial Day weekend
    ('2025-11-26', '2025-11-30'),  # Thanksgiving weekend
    ('2025-12-21', '2026-01-01')   # Christmas/New Year’s period
]

# Add holiday features
df['date'] = df['timestamp'].dt.date.astype(str)
df['is_holiday'] = df['date'].isin(holidays_2025.keys()).astype(int)
df['is_holiday_period'] = 0
for start, end in holiday_periods:
    df.loc[df['date'].between(start, end), 'is_holiday_period'] = 1
df['holiday_type'] = df['date'].map({k: v['type'] for k, v in holidays_2025.items()}).fillna('None')

# Drop temporary date column
df = df.drop('date', axis=1)

# Save updated CSV
df.to_csv('updated_data_with_holidays.csv', index=False)
print("Updated CSV saved as 'updated_data_with_holidays.csv'")