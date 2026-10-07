import pandas as pd

df = pd.read_excel("datasets/SampleSuperstore test file.xlsx")

print("\n===== DATA QUALITY REPORT =====")

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nNumeric Columns:")
print(df.select_dtypes(include="number").columns.tolist())