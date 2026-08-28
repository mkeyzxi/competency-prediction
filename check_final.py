import openpyxl

wb = openpyxl.load_workbook('data/raw/DBNR.xlsx', data_only=True)
ws = wb['FINAL']

rows = []
for row in ws.iter_rows(values_only=True):
    if any(cell is not None for cell in row):
        rows.append(row)

print(f"Total non-empty rows: {len(rows)}")
for i, r in enumerate(rows):
    print(i, r)
