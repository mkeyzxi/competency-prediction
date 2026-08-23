import pandas as pd

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Competency_Label based on Final_Individu >= 75.
    Drops Final_Individu and other target-leaking columns.
    """
    df = df.copy()
    
    # Ensure Final_Individu is numeric
    df['Final_Individu'] = pd.to_numeric(df['Final_Individu'], errors='coerce')
    
    # Drop rows where Final_Individu is NaN (we cannot form a label)
    df = df.dropna(subset=['Final_Individu'])
    
    # Create Label
    df['Competency_Label'] = (df['Final_Individu'] >= 75).astype(int)
    df['Competency_Name'] = df['Competency_Label'].map({1: 'Kompeten', 0: 'Belum Kompeten'})
    
    # Drop leaking columns
    cols_to_drop = [
        'Final_Individu', 'Final_Kelompok', 'Final_Total', 
        'NILAI_AKHIR', 'PREDIKAT', 'No', 'Nama' # NIM kept for tracking during analysis, but should be dropped before modeling
    ]
    # We will keep NIM for tracking, but the PRD says drop identifiers from X.
    # The split and modeling pipeline should explicitly exclude NIM.
    
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    return df
