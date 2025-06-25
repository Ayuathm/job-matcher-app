# migrate_single_csv_to_db.py
import pandas as pd
import re
from db import init_db, insert_jobs_from_df

# Initialize the database
init_db()

# Define the path to your specific CSV file
csv_file = "data/initial_dataset.csv"

try:
    df = pd.read_csv(csv_file)
    if "text" in df.columns:
        df["text"] = df["text"].astype(str)
        df["text"] = df["text"].apply(lambda x: re.sub(r"[\r\n\t]+", " ", x).strip())
        insert_jobs_from_df(df[["text"]])
        print(f"✅ Inserted {len(df)} records from {csv_file}")
    else:
        print("⚠️ 'text' column not found in CSV.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")