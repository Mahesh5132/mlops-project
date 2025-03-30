import os
import pandas as pd

def preprocess_data(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all CSV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):  # Process only CSV files
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            df = pd.read_csv(input_path)
            df = df.dropna()  # Example preprocessing step
            df.to_csv(output_path, index=False)
            print(f"Processed: {file_name}")

if __name__ == "__main__":
    raw_dir = r"data\raw"
    processed_dir = r"data\processed"
    preprocess_data(raw_dir, processed_dir)
