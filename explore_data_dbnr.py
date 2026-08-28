import pandas as pd

try:
    xl = pd.ExcelFile('data/raw/DBNR.xlsx')
    
    with open('raw_sample.txt', 'w', encoding='utf-8') as f:
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name, header=None)
            f.write(f'--- Sheet {sheet_name} ---\n')
            f.write(df.head(10).to_string())
            f.write('\n\n')
    print("Successfully wrote to raw_sample.txt")
except Exception as e:
    print(f"Error: {e}")
