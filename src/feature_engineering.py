# Feature Creation and Encoding

from dotenv import load_dotenv
from sodapy import Socrata
import os
import pandas as pd

load_dotenv()

# Set up the Socrata client
data_url = 'data.cityofnewyork.us'  # NYC Open Data domain
data_set = '7ym2-wayt'  # Dataset ID for Automated Traffic Volume Counts
TRAFFIC_APP_TOKEN = os.getenv("TRAFFIC_APP_TOKEN")

# Initialize the Socrata client (timeout set to 150 seconds for large queries)
client = Socrata(data_url, TRAFFIC_APP_TOKEN, timeout=150)

# Define the output directory and file
output_dir = os.path.join('data', 'processed')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_csv = os.path.join(output_dir, 'traffic_volume_2021_plus.csv')

# Filter for data from 2021 onward
where_clause = "yr >= '2021'"

# Define batch size and file size target
batch_size = 5000  # Number of rows per batch, adjust based on memory
max_file_size_mb = 90  # Target < 100 MB
total_size_mb = 0
offset = 0
keep_going = True
all_data = []  # List to store all batches

try:
    # Loop to fetch data in batches
    while keep_going:
        try:
            # Fetch a batch of data, filtered for 2021 and later
            results = client.get(data_set, limit=batch_size, offset=offset, where=where_clause)
            
            # Stop if no more records are returned
            if not results:
                print("No more data to fetch.")
                break
            
            # Convert batch to Pandas DataFrame and append to list
            df = pd.DataFrame.from_records(results)
            all_data.append(df)
            print(f"Fetched batch of {len(df)} rows, offset: {offset}")
            
            # Increment offset for next batch
            offset += batch_size
            
        except Exception as e:
            print(f"Error in batch fetch: {e}")
            keep_going = False

    # Combine all batches into one DataFrame
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save to a single CSV file
        combined_df.to_csv(output_csv, index=False)
        file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)  # Convert bytes to MB
        total_size_mb = file_size_mb
        print(f"Data saved to {output_csv}, size: {file_size_mb:.2f} MB")
        
        # Check if file size is under GitHub limit
        if total_size_mb >= max_file_size_mb:
            print(f"Warning: File size ({total_size_mb:.2f} MB) is close to or exceeds 90 MB. Consider filtering data further.")
    else:
        print("No data fetched to save.")

except Exception as e:
    print(f"Error connecting to Socrata API: {e}")

finally:
    # Close the Socrata client
    client.close()

print(f"Download complete. Total size: {total_size_mb:.2f} MB")