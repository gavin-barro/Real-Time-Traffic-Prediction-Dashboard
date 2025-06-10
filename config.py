# config.py

# Path to raw traffic data
DATA_PATH = "data/raw/traffic_volume_2021_plus.csv"

# Path to trained model
MODEL_PATH = "models/traffic_model.pkl"

# Columns used in model
FEATURE_COLUMNS = ['hour', 'weekday']
TARGET_COLUMN = 'vol'

# App settings
CITY = "New York City"
TIMEZONE = "US/Eastern"