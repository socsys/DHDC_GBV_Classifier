import pandas as pd

original = pd.read_csv("bluesky_posts_currentMPs_Jun_cleaned.csv")
labelled = pd.read_csv("bluesky_posts_currentMPs_Jun_cleaned_predictions.csv")

# Check for missing item_id in original
missing_item_id_original = original['item_id'].isna().sum()
print(f"Number of items with missing item_id in original: {missing_item_id_original}")
# Check for missing data_id in labelled
missing_data_id_labelled = labelled['data_id'].isna().sum()
print(f"Number of items with missing data_id in labelled: {missing_data_id_labelled}")

# check for item_id in original that are not in data_id in labelled
missing_item_id_in_labelled = original[~original['item_id'].isin(labelled['data_id'])]
print(f"Number of items with item_id in original that are not in labelled: {len(missing_item_id_in_labelled)}")


print("Items with item_id in original that are not in labelled:")
print(missing_item_id_in_labelled["cleaned_text"].tolist())