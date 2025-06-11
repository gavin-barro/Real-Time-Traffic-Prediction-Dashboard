# Feature Creation and Encoding

"""
    Holidays tagged in dataset:
        - Ash Wednesday (Religious)
        - Ascension Day (Religious)
        - Christmas Day (Religious)
        - Easter Sunday (Religious)
        - Eid al-Fitr (Religious)
        - Halloween (Cultural)
        - Hanukkah (Religious)
        - Independence Day (Federal)
        - Juneteenth (Federal)
        - Labor Day (Federal)
        - Martin Luther King Jr. Day (Federal)
        - Memorial Day (Federal)
        - New Year's Day (Federal)
        - NYC Marathon (Local)
        - Pentecost (Religious)
        - Presidents' Day (Federal)
        - Puerto Rican Day Parade (Local)
        - Thanksgiving (Federal)
        - Veterans Day (Federal)
"""

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta, MO, TH
from dateutil.easter import easter
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# Movable holidays
def get_movable_holidays(year: int) -> dict:
    movable = {
        # 3rd Monday in January
        (pd.Timestamp(f'{year}-01-01') + relativedelta(weekday=MO(3))).date(): ('Martin Luther King Jr. Day', 'Federal'),
        # 3rd Monday in February
        (pd.Timestamp(f'{year}-02-01') + relativedelta(weekday=MO(3))).date(): ("Presidents' Day", 'Federal'),
        # Last Monday in May
        (pd.Timestamp(f'{year}-05-31') + relativedelta(weekday=MO(-1))).date(): ('Memorial Day', 'Federal'),
        # 1st Monday in September
        (pd.Timestamp(f'{year}-09-01') + relativedelta(weekday=MO(1))).date(): ('Labor Day', 'Federal'),
        # 2nd Sunday in June
        (pd.Timestamp(f'{year}-06-01') + relativedelta(weekday=MO(1))).date(): ('Puerto Rican Day Parade', 'Local'),
        # 4th Thursday in November
        (pd.Timestamp(f'{year}-11-01') + relativedelta(weekday=TH(4))).date(): ('Thanksgiving', 'Federal'),
        # 1st Monday in November (Note: changed from Sunday to Monday)
        (pd.Timestamp(f'{year}-11-01') + relativedelta(weekday=MO(1))).date(): ('NYC Marathon', 'Local'),
    }

    # Add religious holidays (Easter is already a date)
    movable[easter(year)] = ('Easter Sunday', 'Religious')
    movable[easter(year) - pd.Timedelta(days=47)] = ('Ash Wednesday', 'Religious')
    movable[easter(year) + pd.Timedelta(days=39)] = ('Ascension Day', 'Religious')
    movable[easter(year) + pd.Timedelta(days=49)] = ('Pentecost', 'Religious')

    return movable

# Load the CSV
df = pd.read_csv('data/processed/traffic_volume_2018_plus.csv')

# Normalize column names and create timestamp
df.rename(columns=lambda col: col.lower(), inplace=True)
df.rename(columns={'yr': 'year', 'm': 'month', 'd': 'day', 'hh': 'hour', 'mm': 'minute'}, inplace=True)
df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
df['date'] = df['timestamp'].dt.date

# Time-based features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.day_name()
df['is_weekend'] = df['dayofweek'].isin(['Saturday', 'Sunday']).astype(int)
df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

# Holiday definition
years = df['year'].unique()
holiday_dict = {}

# Fixed holidays (same date every year)
fixed_holidays_mmdd = {
    '01-01': ("New Year's Day", 'Federal'),
    '06-19': ('Juneteenth', 'Federal'),
    '07-04': ('Independence Day', 'Federal'),
    '10-31': ('Halloween', 'Cultural'),
    '11-11': ('Veterans Day', 'Federal'),
    '12-25': ('Christmas Day', 'Religious'),
}

# Combine all holiday dates
for year in years:
    # Add fixed holidays
    for mmdd, (name, htype) in fixed_holidays_mmdd.items():
        date = pd.to_datetime(f"{year}-{mmdd}").date()
        holiday_dict[date] = (name, htype)
    # Add movable holidays
    holiday_dict.update(get_movable_holidays(year))

# Holiday encoding
df['is_holiday'] = df['date'].isin(holiday_dict).astype(int)
df['holiday_name'] = df['date'].map(lambda d: holiday_dict.get(d, ('None', 'None'))[0])
df['holiday_type'] = df['date'].map(lambda d: holiday_dict.get(d, ('None', 'None'))[1])

# Holiday periods (travel traffic windows)
holiday_periods = [
    ('05-24', '05-26'),  # Memorial Day weekend
    ('11-26', '11-30'),  # Thanksgiving weekend
    ('12-21', '01-01')   # Winter holiday season
]

df['is_holiday_period'] = 0
for year in years:
    for start_mmdd, end_mmdd in holiday_periods:
        start_date = pd.to_datetime(f"{year}-{start_mmdd}").date()
        end_year = year if end_mmdd.startswith("0") or int(end_mmdd[:2]) <= 6 else year + 1
        end_date = pd.to_datetime(f"{end_year}-{end_mmdd}").date()
        df.loc[df['date'].between(start_date, end_date), 'is_holiday_period'] = 1

# Drop temporary column
df.drop(columns=['date'], inplace=True)

# Save enriched data
output_path = 'data/processed/traffic_volume_2018_plus_feature_engineered.csv'
df.to_csv(output_path, index=False)
print(f"Updated CSV saved to: {output_path}")




""" Adding Weather to the output. """






