import mne
import pandas as pd

# 1. Load the data (as you already have)
file_path = "A01T.gdf"
raw = mne.io.read_raw_gdf(file_path, preload=True)

# 2. Convert to a Pandas DataFrame
# 'scalings' ensures the data is in Volts/Units rather than internal machine values
df = raw.to_data_frame()

# 3. Export to a Spreadsheet file
df.to_csv("EEG_Data_Spreadsheet.csv", index=False)
# OR for Excel (requires: pip install openpyxl)
# df.to_excel("EEG_Data_Spreadsheet.xlsx", index=False)

print("Spreadsheet successfully created!")