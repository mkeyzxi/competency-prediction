import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run_class_analysis(df: pd.DataFrame):
    """
    Analyzes heterogeneity across classes A, B, C, D (or whatever classes exist).
    Outputs class summary table and distribution plots.
    """
    os.makedirs('results/class_analysis', exist_ok=True)
    
    if 'Kelas' not in df.columns:
        print("No 'Kelas' column found for class analysis.")
        return
        
    summary = df.groupby('Kelas').agg(
        num_students=('NIM', 'count'),
        kompeten_count=('Competency_Label', 'sum'),
        mean_attendance=('Attendance_Rate', 'mean'),
        mean_tp=('TP_Mean', 'mean'),
        mean_respon=('Respons_Mean', 'mean'),
        mean_laporan=('Laporan_Mean', 'mean')
    ).reset_index()
    
    summary['prop_kompeten'] = summary['kompeten_count'] / summary['num_students']
    
    summary.to_csv('results/class_analysis/class_summary.csv', index=False)
    
    # Plot Class Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Kelas', hue='Competency_Name')
    plt.title('Distribusi Kompetensi per Kelas')
    plt.tight_layout()
    plt.savefig('results/class_analysis/class_distribution.png')
    plt.close()
    
    print("Class analysis complete.")
