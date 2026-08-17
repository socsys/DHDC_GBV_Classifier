import torch
import pandas as pd
import re
from typing import List, Any
from collections import Counter

def compute_class_weights(labels: List[List[int]], num_classes: int = 6) -> List[float]:
    ''' Computes pos class weights for multi-label classification. Returns a list of weights for each class. '''
    N = len(labels)
    pos_counts = []
    for i in range(num_classes):
        pos_counts.append(sum(label[i] for label in labels))
    if any(pos == 0 for pos in pos_counts):
        print(f"Warning: One or more classes have zero positive samples. Pos counts: {pos_counts}")
    neg_counts = [N - pos for pos in pos_counts]
    class_weights = [x/y if y > 0 else x/1 for x,y in zip(neg_counts, pos_counts)]
    print(f"Calculated class weights: {class_weights}")
    return class_weights

def flatten(xss: List[List[Any]]) -> List[Any]:
    return [x for xs in xss for x in xs]

# ----------------------------------
# Data loading and processing
# ----------------------------------

def clean_text(tweet: str) -> str:
    """Remove social media handles and URLs from text."""
    if not isinstance(tweet, str):
        raise TypeError(f"Expected a string, but got {type(tweet)} with value: {tweet}")
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'https?\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'<MENTION_[1-9]>', '', result)  # for AMI dataset 
    result = re.sub(r'<URL>', '', result)  # for AMI dataset
    #result = re.sub(r'#', ' ', result)
    return result

def create_data_loader(processed_data, tokenizer, batch_size=16, infr=False):
    ''' Create PyTorch dataloader. Includes labels except in inference mode.'''
    encodings = tokenizer(
        processed_data["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    data_id = torch.tensor(processed_data["data_id"].tolist())

    if not infr:
        binary_labels = torch.tensor(processed_data["binary_labels"].tolist())
        category_labels = torch.tensor(processed_data["category_labels"].tolist())
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            binary_labels.to(torch.float),
            category_labels.to(torch.float),
            data_id
        )
    else:
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            data_id
        )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) if not infr else torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) # Do not shuffle at infr to allow for resuming inference from a specific data_id

class CustomDataLoader:
    ''' Custom data loader with function that handles processing raw labeled data files, cleaning text, and converting labels to multi-hot format. '''
    def __init__(self, data_name, data_path, split, label2id_dict, multilingual=False):
        self.data_name = data_name
        self.data_path = data_path
        self.split = split
        self.label2id_dict = label2id_dict
        self.multilingual = multilingual
        if not self.data_name == "EXIST":
            raise ValueError(f"Currently only supports EXIST dataset. Please provide a valid data_name. Got {self.data_name}.")

    def _load_raw_data(self):
        ''' Loads raw data from the specified path and filters based on the split. Currently only implemented for EXIST dataset. '''
        print(self.data_path)
        if self.data_path.endswith(".json"):
            raw_data = pd.read_json(self.data_path, orient='index')
        elif self.data_path.endswith(".csv") or self.data_path.endswith(".tsv"):
            raw_data = pd.read_csv(self.data_path, sep='\t' if self.data_path.endswith(".tsv") else ',')
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}. Please provide a .json, .csv, or .tsv file.")
        if self.split == "train":
            if self.data_name == "EXIST": 
                if self.multilingual:
                    raw_data = raw_data[raw_data["split"].isin(["TRAIN_EN", "TRAIN_ES"])]
                else:
                    raw_data = raw_data[raw_data["split"].isin(["TRAIN_EN"])]
        elif self.split == "dev":
            if self.data_name == "EXIST": 
                raw_data = raw_data[raw_data["split"].isin(["DEV_EN"])]
        return raw_data

    def _final_labels(self,x, level):
        labels = [0 for _ in range(len(self.label2id_dict[f"level_{level}"]))]
        if self.data_name == "EXIST":
            non_gbv = "NO" if level == 1 else "-"
            if x[non_gbv] == 3: # 6 annotators split on label
                return 99
            elif x[non_gbv] > 3: # Majority of annotators (4/6) labeled as non-GBV
                labels[0] = 1
                return labels
            else:
                if level == 1:
                    labels[1] = 1
                    return labels
                elif level == 2:
                    for k, v in x.items():
                        if k in self.label2id_dict[f"level_{level}"]:
                            if v >= 2: # EXIST task applies label if at least 2 annotators labeled it as such
                                labels[self.label2id_dict[f"level_{level}"][k]] = 1
                    labels[0] = 0 # ensure that the non-gbv label is set to 0 if any gbv label is present
            if sum(labels) == 0:
                return 99
        else:
            raise ValueError(f"Unsupported data_name: {self.data_name}. Currently only supports 'EXIST'.")
        return labels


    def _handle_labels(self, dataset):
        if self.data_name == "EXIST":
            column_names = {"level_1": "labels_task1_1", "level_2": "labels_task1_3"} # GBV and GBV subtype columns for EXIST
        else:
            raise ValueError(f"Unsupported data_name: {self.data_name}. Currently only supports 'EXIST'.")
        for level in [1, 2]:
            col = column_names[f"level_{level}"]
            print(f"Processing level {col} for {self.data_name}")
            dataset = dataset.dropna(subset=[col]) # drop rows with missing labels for this level
            if level == 1:
                column_label = "binary_labels"
            else:
                column_label = "category_labels"
            dataset[column_label] = dataset[col].apply(lambda x: x if isinstance(x, str) or isinstance(x, int) else (Counter(x) if isinstance(x[0], str) else Counter(flatten(x))))
            dataset[column_label] = dataset[column_label].apply(lambda x: self._final_labels(x, level))

        return dataset[["text", "binary_labels", "category_labels"]]
    
    def load_processed_data(self, clean=False, remove_99=True):
        ''' Function to load and process the raw data, returning a DataFrame with text and multi-hot encoded labels. '''
        raw_data = self._load_raw_data()
        raw_data.rename(columns={"tweet": "text"}, inplace=True) if "tweet" in raw_data.columns else None
        processed_data = self._handle_labels(raw_data)
        if clean:
            processed_data["text"] = processed_data["text"].apply(clean_text)
        process_data.reset_index(names="data_id", inplace=True) # add a unique identifier for each data point, which can be used for tracking during inference

        if remove_99:
            print(f"Number of binary label ties: {len(processed_data[processed_data["binary_labels"]==99])}")
            processed_data = processed_data[processed_data["binary_labels"] != 99]
            print(f"Number of category label ties: {len(processed_data[processed_data["category_labels"]==99])}")
            processed_data = processed_data[processed_data["category_labels"] != 99]

        return processed_data

