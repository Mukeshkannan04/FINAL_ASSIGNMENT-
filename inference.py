import torch
import pandas as pd
import numpy as np
import os
import sys
from model import AirDrawModel
from dataset import AirDrawDataset

# --- CONFIGURATION ---
MODEL_PATH = "best_model.pth"
TRAIN_DATA_DIR = "data/processed" # Needed to calibrate the scaler
# ---------------------

def process_single_raw_folder(folder_path):
    """
    Reads a raw folder (Accel + Gyro), merges them, and returns a DataFrame.
    """
    try:
        acc_path = os.path.join(folder_path, 'Accelerometer.csv')
        gyro_path = os.path.join(folder_path, 'Gyroscope.csv')
        
        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            raise FileNotFoundError("Folder must contain Accelerometer.csv and Gyroscope.csv")

        # Load
        acc_df = pd.read_csv(acc_path)
        gyr_df = pd.read_csv(gyro_path)

        # Rename
        acc_df = acc_df.rename(columns={'x': 'ax', 'y': 'ay', 'z': 'az'})
        gyr_df = gyr_df.rename(columns={'x': 'gx', 'y': 'gy', 'z': 'gz'})

        # Merge
        acc_df = acc_df.sort_values('time')
        gyr_df = gyr_df.sort_values('time')
        
        merged = pd.merge_asof(
            acc_df, gyr_df[['time', 'gx', 'gy', 'gz']], 
            on='time', direction='nearest', tolerance=20000000
        )
        merged = merged.dropna()
        
        return merged[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
    except Exception as e:
        print(f"Error processing raw files: {e}")
        return None

def predict_digit(raw_folder_path):
    # 1. Setup Device
    device = torch.device("cpu")
    
    # 2. Load and Fit Scaler (Using training data to be consistent)
    # We need to scale the new data exactly like the training data
    print("Calibrating sensor scaler...")
    dataset = AirDrawDataset(TRAIN_DATA_DIR) 
    scaler = dataset.scaler
    
    # 3. Process the New Input
    print(f"Processing input: {raw_folder_path}")
    features = process_single_raw_folder(raw_folder_path)
    
    if features is None:
        return

    # 4. Normalize and Pad
    features = scaler.transform(features)
    
    max_len = 200
    if len(features) < max_len:
        padding = np.zeros((max_len - len(features), 6))
        features = np.vstack((features, padding))
    else:
        features = features[:max_len, :]
        
    # Convert to Tensor [1, 6, 200]
    input_tensor = torch.tensor(features, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    
    # 5. Load Model
    model = AirDrawModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 6. Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    digit = predicted.item()
    conf_score = confidence.item() * 100
    
    print("\n" + "="*30)
    print(f"🤖 PREDICTION:  DIGIT {digit}")
    print(f"📊 Confidence:  {conf_score:.1f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    # Allow user to pass folder path via command line
    if len(sys.argv) > 1:
        folder_to_test = sys.argv[1]
    else:
        # DEFAULT: Change this to one of your raw folders to test!
        folder_to_test = "raw_data_dump/User2_digit6_trial01_csv..." 
        print("Please provide a folder path. Usage: python inference.py path/to/folder")
        print(f"Using default (if exists): {folder_to_test}")

    if os.path.exists(folder_to_test):
        predict_digit(folder_to_test)
    else:
        print("Folder not found. Please check the path.")