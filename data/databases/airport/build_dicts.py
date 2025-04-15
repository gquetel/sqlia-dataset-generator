import pandas as pd
import os

csv_data = "./airports.csv"

df = pd.read_csv(csv_data)
output_dir = "dicts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for column in df.columns:
    file_path = os.path.join(output_dir, f"{column}")
    with open(file_path, 'w') as f:
        for value in df[column]:
            if not pd.isna(value):
                f.write(f"{value}\n")
            
    print(f"Saved {column} data to {file_path}")
