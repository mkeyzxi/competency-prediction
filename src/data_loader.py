import pandas as pd
import os
from src.utils import load_config

def load_and_clean_data(config_path: str = 'configs/data_config.yaml'):
    """
    Load data from raw excel file, handle both sheets, clean column names, 
    and remove metadata rows.
    """
    config = load_config(config_path)
    file_path = config['data']['path']
    sheets = config['data']['sheets']
    
    # Expected columns based on user recommendation to avoid 'Total' duplication
    expected_cols = [
        "No", "NIM", "Nama",
        "Kehadiran_1", "Kehadiran_2", "Kehadiran_3", "Kehadiran_4", "Kehadiran_5", "Kehadiran_6", "Kehadiran_7", "Kehadiran_8", "Kehadiran_9", "Kehadiran_10",
        "Kehadiran_Ket", "Kehadiran_Total",
        "Laporan_1", "Laporan_2", "Laporan_3", "Laporan_4", "Laporan_5", "Laporan_6", "Laporan_7", "Laporan_Total",
        "TP_1", "TP_2", "TP_3", "TP_4", "TP_5", "TP_6", "TP_Total",
        "Respon_1", "Respon_2", "Respon_3", "Respon_4", "Respon_5", "Respon_6", "Respon_Total",
        "Final_Individu", "Final_Kelompok", "Final_Total",
        "NILAI_AKHIR", "PREDIKAT"
    ]
    
    all_data = []
    
    for sheet in sheets:
        try:
            # Skip the first 2 rows of merged headers
            df_raw = pd.read_excel(file_path, sheet_name=sheet, skiprows=2)
            
            # If the number of columns doesn't match expected, we might have an issue
            # We will try to map by index if the count matches exactly, 
            # otherwise we will try to rename duplicates.
            
            # Let's clean the metadata rows at the bottom.
            # Usually, real student rows have a numeric 'No' or valid 'NIM'.
            # We will coerce 'No' or the first column to numeric. If it's NaN, we drop the row.
            
            # Rename columns if lengths match:
            if len(df_raw.columns) == len(expected_cols):
                df_raw.columns = expected_cols
            else:
                print(f"Warning: Columns count in sheet {sheet} ({len(df_raw.columns)}) doesn't match expected ({len(expected_cols)}).")
                # Fallback: rename 'Total.1', 'Total.2' etc. based on some heuristic, but user provided explicit list.
                # If they don't match, we still try our best.
            
            # Drop rows that are just metadata. 
            if "NIM" in df_raw.columns:
                # Drop rows where NIM is NaN
                df_clean = df_raw.dropna(subset=['NIM']).copy()
                # Ensure NIM is string to check for non-student rows
                df_clean['NIM'] = df_clean['NIM'].astype(str)
                # Drop rows where NIM looks like text description (e.g. "Bobot", "Asisten")
                mask_valid = ~df_clean['NIM'].str.contains('Asisten|Bobot|Nilai', case=False, na=False)
                df_clean = df_clean[mask_valid]
                
                # Filter out dropouts (Kehadiran <= 1)
                kehadiran_cols = [f'Kehadiran_{i}' for i in range(1, 11)]
                # Extract valid Kehadiran columns that actually exist in the dataframe
                valid_kehadiran_cols = [col for col in kehadiran_cols if col in df_clean.columns]
                
                if valid_kehadiran_cols:
                    temp_kehadiran = df_clean[valid_kehadiran_cols].apply(pd.to_numeric, errors='coerce')
                    # Count how many times they attended (> 0 or not null)
                    attendance_count = (temp_kehadiran > 0).sum(axis=1)
                    # Keep only students who attended more than 1 time
                    df_clean = df_clean[attendance_count > 1].copy()
            else:
                df_clean = df_raw
            
            # Add Kelas column
            df_clean['Kelas'] = sheet
            all_data.append(df_clean)
            
        except Exception as e:
            print(f"Error loading sheet {sheet}: {e}")
            
    if not all_data:
        raise ValueError("No data could be loaded. Please check the dataset and configuration.")
        
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Save the interim data
    os.makedirs('data/interim', exist_ok=True)
    df_final.to_csv('data/interim/combined_data.csv', index=False)
    
    return df_final

if __name__ == "__main__":
    df = load_and_clean_data()
    print("Data loaded successfully. Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample:\n", df.head())
