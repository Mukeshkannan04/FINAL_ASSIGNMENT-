import os
import pandas as pd
import re

# ================= CONFIGURATION =================
INPUT_ROOT = 'raw_data_dump'   
OUTPUT_DIR = 'data/processed'  
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all subfolders
subfolders = [f.path for f in os.scandir(INPUT_ROOT) if f.is_dir()]
print(f"Found {len(subfolders)} folders. Starting processing...")

success_count = 0

for folder in subfolders:
    try:
        # 1. Check for required files
        accel_path = os.path.join(folder, 'Accelerometer.csv')
        gyro_path = os.path.join(folder, 'Gyroscope.csv')
        
        if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
            continue

        # 2. Get Label from Folder Name
        folder_name = os.path.basename(folder)
        digit_match = re.search(r'digit(\d+)', folder_name, re.IGNORECASE)
        trial_match = re.search(r'trial(\d+)', folder_name, re.IGNORECASE)
        
        if not digit_match:
            print(f"Skipping {folder_name}: No 'digit' found in name.")
            continue
            
        digit = digit_match.group(1)
        trial = trial_match.group(1) if trial_match else "00"

        # 3. Load Data
        acc_df = pd.read_csv(accel_path)
        gyr_df = pd.read_csv(gyro_path)

        # 4. Rename Columns
        acc_df = acc_df.rename(columns={'x': 'ax', 'y': 'ay', 'z': 'az'})
        gyr_df = gyr_df.rename(columns={'x': 'gx', 'y': 'gy', 'z': 'gz'})

        # 5. Merge Sensors on Time
        acc_df = acc_df.sort_values('time')
        gyr_df = gyr_df.sort_values('time')
        
        merged = pd.merge_asof(
            acc_df, 
            gyr_df[['time', 'gx', 'gy', 'gz']], 
            on='time', 
            direction='nearest',
            tolerance=20000000 
        )
        merged = merged.dropna()

        # 6. Save Clean File
        output_filename = f"user2_digit{digit}_trial{trial}.csv"
        final_df = merged[['ax', 'ay', 'az', 'gx', 'gy', 'gz']]
        
        final_df.to_csv(os.path.join(OUTPUT_DIR, output_filename), index=False)
        success_count += 1

    except Exception as e:
        print(f"Error processing {folder}: {e}")

print(f"\nProcessing Complete! {success_count} clean files saved to 'data/processed'.")