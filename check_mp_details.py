import pandas as pd

raw_data = pd.read_csv('mp_details_full_info.csv')

print(raw_data["minority_status"].value_counts())