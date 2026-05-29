import os # must import before torch to have desired effect 
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from trl import SFTTrainer
import pandas as pd
from custom_code import DatasetPrepper, calculate_class_weights
import time
import sklearn
import shutil
from argparse import ArgumentParser
import optuna 
import itertools
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import re 
from functools import partial
import json 
import time
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModel
from typing import Any, Dict, List, Sequence, Optional
from torch.optim import AdamW
import math
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
import random
import logging 

parser = ArgumentParser()
parser.add_argument("--train", action="store_true", help="Whether to run training.")
args = parser.parse_args()

torch.cuda.empty_cache()

# Logger for report
#rep_logger = logging.getLogger("report_logger")
#rep_logger.addHandler(logging.FileHandler("report.log", mode="w"))

# Logger for debugging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='analysis.log', filemode='w')

'''
def compute_class_weights(labels, num_classes=6):
    """
    Calculate class weights for imbalanced datasets for a multilabel task.
    """
    print(f"Number of labels: {num_classes}")
    classes = range(num_classes) # for 6 classes in multilabel classification
    N = len(labels)
    class_weights: List[float] = [0.0 for _ in classes]

    for i, id in enumerate(classes):
        class_weights[i] = N / np.sum([1 for label in labels if label[id] == 1])
        class_weights[id] = np.float32(class_weights[id]) # convert to float32 for compatibility with model training

    print(f"Calculated class weights: {class_weights}")
    return class_weights
    '''


def set_seed(seed):
    torch.manual_seed(seed)  # Sets seed for CPU operations
    np.random.seed(seed)     # Sets seed for NumPy
    random.seed(seed)        # Sets seed for Python's random module
    
    if torch.cuda.is_available():  # GPU-specific settings
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disables non-deterministic optimizations

# Example usage
set_seed(42)

def compute_class_weights(labels, num_classes=6):
    N = len(labels)
    pos_counts = []
    for i in range(num_classes):
        #print(f"Class {i} positive count: {sum(label[i] for label in labels)}")
        pos_counts.append(sum(label[i] for label in labels))
    #print(f"Positive counts per class: {pos_counts}")
    neg_counts = [N - pos for pos in pos_counts]
    #class_weights = [N/ (num_classes * val) for val in pos_counts]
    #class_weights = (neg_counts / pos_counts).tolist()
    class_weights = [x/y for x,y in zip(neg_counts, pos_counts) if y > 0]
    print(f"Calculated class weights: {class_weights}")
    return class_weights


def flatten(xss):
    return [x for xs in xss for x in xs]

def final_labels(x: dict, level, label2id_dict):
    labels = [0 for _ in range(len(label2id_dict[f"level_{level}"]))]
    non_gbv = "NO" if level == 1 else "-"
    if x[non_gbv] == 3:
        return 99
    elif x[non_gbv] > 3:
        labels[0] = 1
        return labels
    else:
        if level == 1:
            labels[1] = 1
            return labels
        elif level == 3:
            for k, v in x.items():
                if k in label2id_dict[f"level_{level}"]:
                    if v >= 2:
                        labels[label2id_dict[f"level_{level}"][k]] = 1
            labels[0] = 0 # ensure that the non-gbv label is set to 0 if any gbv label is present
    if sum(labels) == 0:
        return 99
    return labels


def handle_labels(dataset, level: list, label2id_dict):
    label_list = []

    for l in level:
        col = f"labels_task1_{l}"
        print(f"Processing level {col} for EXIST")
        dataset = dataset.dropna(subset=[col]) # drop rows with missing labels for this level
        dataset[f"labels_{l}"] = dataset[col].apply(lambda x: Counter(x) if isinstance(x[0], str) else Counter(flatten(x)))
        label_list.append(f"labels_{l}")
        dataset[f"labels_{l}"] = dataset[f"labels_{l}"].apply(lambda x: final_labels(x, level=l, label2id_dict=label2id_dict))

    return dataset[["tweet"] + label_list]

class DataLoader:
    def __init__(self, file_path, split, label2id_dict, multilingual=False):
        self.split = split
        self.file_path = file_path
        self.train_data_name = "EXIST"
        self.label2id_dict = label2id_dict
        self.multilingual = multilingual

        self.raw_data = self.load_data()
        self.processed_data = self.preprocess_data(self.raw_data, [1,3], label2id_dict)
        self.processed_data = self.processed_data.rename(columns={"tweet": "text", "labels_1": "binary_labels", "labels_3": "category_labels"})
        self.processed_data["id_EXIST"] = self.processed_data.index # add a column for the original comment ID from the EXIST dataset for easier analysis later

    def load_data(self):
        if self.split == "train":
            raw_data = pd.read_json(self.file_path,orient='index')
            if self.multilingual:
                dataset = raw_data[(raw_data["split"] == "TRAIN_EN") | (raw_data["split"] == "TRAIN_ES")]
            else:
                dataset = raw_data[raw_data["split"] == "TRAIN_EN"]
        elif self.split == "dev":
            raw_data = pd.read_json(self.file_path,orient='index')
            dataset = raw_data[raw_data["split"] == "DEV_EN"]
        return dataset

    def preprocess_data(self, raw_data, level, label2id_dict):
        return handle_labels(raw_data, level=level, label2id_dict=label2id_dict)


class AppearanceMultiTaskClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        num_category_labels: int = 6,
        dropout: float = 0.1,
        lambda_binary: float = 1.0,
        lambda_category: float = 1.0,
        binary_loss_type: str = "ce",
        category_loss_type: str = "ce",
        focal_gamma_binary: float = 2.0,
        focal_gamma_category: float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_size, 2)
        self.category_head = nn.Linear(hidden_size, num_category_labels)
        self.lambda_binary = lambda_binary
        self.lambda_category = lambda_category
        self.binary_loss_type = binary_loss_type
        self.category_loss_type = category_loss_type
        self.focal_gamma_binary = focal_gamma_binary
        self.focal_gamma_category = focal_gamma_category

    def _pooled_output(self, encoder_outputs: Any) -> torch.Tensor:
        if getattr(encoder_outputs, "pooler_output", None) is not None:
            return encoder_outputs.pooler_output
        return encoder_outputs.last_hidden_state[:, 0]

    def _binary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        label_indices = labels.argmax(dim=1).long()
        if self.binary_loss_type == "focal":
            ce_loss = F.cross_entropy(logits, label_indices, weight=class_weights, reduction="none")
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, label_indices.unsqueeze(1)).squeeze(1).clamp(1e-4, 1 - 1e-4)
            return ((1 - pt) ** self.focal_gamma_binary) * ce_loss
        return F.cross_entropy(logits, label_indices, weight=class_weights, reduction="none")

    def _category_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.category_loss_type == "focal":
            if class_weights is not None:
                class_matrix = class_weights.unsqueeze(0).expand_as(logits)
            else:
                class_matrix = None
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_matrix, reduction="none")
            probs = torch.sigmoid(logits)
            pt = torch.where(labels >= 0.5, probs, 1 - probs).clamp(1e-4, 1 - 1e-4)
            focal_weight = (1 - pt) ** self.focal_gamma_category
            loss = focal_weight * bce_loss
            return loss 
        return F.cross_entropy(logits, labels, weight=class_weights, reduction="none")

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        binary_labels: torch.Tensor | None = None,
        category_labels: torch.Tensor | None = None,
        binary_class_weights: torch.Tensor | None = None,
        category_class_weights: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self._pooled_output(encoder_outputs))
        binary_logits = self.binary_head(pooled)
        category_logits = self.category_head(pooled)

        outputs: Dict[str, torch.Tensor] = {
            "binary_logits": binary_logits,
            "category_logits": category_logits,
        }

        if binary_labels is not None and category_labels is not None:
            binary_loss = self._binary_loss(binary_logits, binary_labels, binary_class_weights)
            #category_loss = F.cross_entropy(
            #    category_logits,
            #    category_labels,
            #    weight=category_class_weights,
            #    reduction="none",
            #)
            category_loss = self._category_loss(category_logits, category_labels, category_class_weights)
 

            outputs["loss_binary"] = binary_loss.mean()
            outputs["loss_category"] = category_loss.mean()
            outputs["loss"] = (
                self.lambda_binary * outputs["loss_binary"]
                + self.lambda_category * outputs["loss_category"]
            )

        return outputs

def create_data_loader(processed_data, tokenizer, batch_size=16):
    encodings = tokenizer(
        processed_data["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    binary_labels = torch.tensor(processed_data["binary_labels"].tolist())
    category_labels = torch.tensor(processed_data["category_labels"].tolist())
    comment_id = torch.tensor(processed_data["id_EXIST"].tolist())
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        binary_labels.to(torch.float),
        category_labels.to(torch.float),
        comment_id,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)    


def move_batch_to_device(batch, device: "cuda") -> Dict[str, Any]:
    tensor_keys = ["input_ids", "attention_mask", "binary_labels", "category_labels", "comment_id"]
    moved = defaultdict()
    #print(batch)
    for i, key in enumerate(tensor_keys):
        moved[key] = batch[i].to(device, non_blocking=True)
    return moved

def prob_to_label(probs, threshold=0.5):
    return (probs >= threshold).long()

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def gather_objects(local_object: Any) -> List[Any]:
    if not is_distributed():
        return [local_object]
    gathered: List[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_object)
    return gathered

def evaluate_model(model, loader, device) -> Dict[str, Any]:
    model.eval()
    local_records: List[Dict[str, Any]] = []
    loss_values: List[float] = []
    loss_binary_values: List[float] = []
    loss_category_values: List[float] = []

    autocast_enabled = device == "cuda"
    with torch.no_grad():
        for step, batch in enumerate(loader):
            moved = move_batch_to_device(batch, "cuda")
            with torch.autocast(device_type=device, enabled=autocast_enabled):
                outputs = model(
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                    binary_labels=moved["binary_labels"],
                    category_labels=moved["category_labels"],
                )
            binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)[:, 1].detach().cpu().tolist()
            category_probs = torch.sigmoid(outputs["category_logits"]).detach().cpu()
            category_pred = prob_to_label(category_probs).tolist()
            category_conf = category_probs.tolist()
            loss_values.append(float(outputs["loss"].detach().cpu().item()))
            loss_binary_values.append(float(outputs["loss_binary"].detach().cpu().item()))
            loss_category_values.append(float(outputs["loss_category"].detach().cpu().item()))

            for index, item in enumerate(zip(moved["comment_id"], moved["binary_labels"], moved["category_labels"])):
                local_records.append(
                    {
                        "comment_id": item[0],
                        "true_binary": 1 if item[1][0] == 0 else 0, # convert from multi-label format to binary format for easier analysis
                        "true_category": item[2],
                        "pred_binary_prob": float(binary_probs[index]),
                        "pred_category": category_pred[index],
                        "pred_category_confidence": category_conf[index],
                    }
                )

    gathered_records = gather_objects(local_records)
    gathered_loss_values = gather_objects(loss_values)
    gathered_loss_binary = gather_objects(loss_binary_values)
    gathered_loss_category = gather_objects(loss_category_values)

    flat_records = [item for shard in gathered_records for item in shard]
    flat_loss_values = [item for shard in gathered_loss_values for item in shard]
    flat_loss_binary = [item for shard in gathered_loss_binary for item in shard]
    flat_loss_category = [item for shard in gathered_loss_category for item in shard]
    return {
        "records": flat_records,
        "loss": float(sum(flat_loss_values) / max(len(flat_loss_values), 1)),
        "loss_binary": float(sum(flat_loss_binary) / max(len(flat_loss_binary), 1)),
        "loss_category": float(sum(flat_loss_category) / max(len(flat_loss_category), 1)),
    }


class EarlyStopping:
    def __init__(self, patience=200, min_delta=0.001, mode="min", warmup_steps=100):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.warmup_steps = warmup_steps  # don't check early stopping until this many steps
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_step = 0

    def step(self, score, current_step) -> bool:
        if current_step < self.warmup_steps:
            return False

        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score < self.best_score - self.min_delta) if self.mode == "min" \
                   else (score > self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_step = current_step
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

def train(model, train_dataset, val_dataset, label2id_dict, model_save_path=None, model_ref="microsoft/mdeberta-v3-base"):
    device = "cuda"

    print("No checkpoint found, starting from scratch.")

    binary_class_weights = compute_class_weights(train_dataset.binary_labels, num_classes=2)
    category_class_weights = compute_class_weights(train_dataset.category_labels, num_classes=len(label2id_dict["level_3"]))
    binary_weight_tensor = torch.tensor(binary_class_weights, dtype=torch.float32, device=device)
    category_weight_tensor = torch.tensor(category_class_weights, dtype=torch.float32, device=device)

    lr = 1e-5
    print(f"Using learning rate: {lr}")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scaler = torch.amp.GradScaler("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    #model.config.pad_token_id = tokenizer.pad_token_id

    train_loader = create_data_loader(train_dataset, tokenizer, batch_size=32)
    val_loader = create_data_loader(val_dataset, tokenizer, batch_size=32)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001, warmup_steps=100)
    best_val_loss = float("inf")
    global_step = 0


    for epoch in range(3):
        model.train()
        local_loss_sum = 0.0
        local_binary_loss_sum = 0.0
        local_category_loss_sum = 0.0
        local_step_count = 0
        progress_interval = max(1, math.ceil(len(train_loader) / 10))

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            moved = move_batch_to_device(batch, "cuda")
            outputs = model(
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                    binary_labels=moved["binary_labels"],
                    category_labels=moved["category_labels"],
                    binary_class_weights=binary_weight_tensor,
                    category_class_weights=category_weight_tensor)
            loss = outputs["loss"] 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if step % 20 == 0:
                validation = evaluate_model(model, val_loader, "cuda")
                val_loss = validation["loss"]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if model_save_path is not None:
                        torch.save(model.state_dict(), model_save_path)
                        print(f"New best model saved at step {step} with validation loss {val_loss:.4f}")

                if early_stopping.step(val_loss, global_step):
                    print(f"Early stopping triggered at global step {step} with validation loss {val_loss:.4f}. Best validation loss was {best_val_loss:.4f} at step {early_stopping.best_step}.")
                    model.load_state_dict(torch.load(model_save_path))
                    break

                #print(f"Step {step}: Training Loss: {loss.item():.4f}, Validation Loss: {validation['loss']:.4f}, binary loss: {validation['loss_binary']:.4f}, category loss: {validation['loss_category']:.4f}")

        validation = evaluate_model(model, val_loader, "cuda")
        print(f"Validation loss for epoch {epoch}: {validation['loss']:.4f}, binary loss: {validation['loss_binary']:.4f}, category loss: {validation['loss_category']:.4f}")

    torch.save(model.state_dict(), model_save_path)


def get_f1_score(records):
    f1_score = defaultdict(int)
    y_true_binary = [record["true_binary"] for record in records]
    y_pred_binary = [1 if record["pred_binary_prob"] >= 0.5 else 0 for record in records]
    f1_score["binary"] = sklearn.metrics.f1_score(y_true_binary, y_pred_binary)

    y_true_category = [record["true_category"].cpu() for record in records]
    y_pred_category = [record["pred_category"] for record in records]
    f1_score["category"] = sklearn.metrics.f1_score(y_true_category, y_pred_category, average=None)
    return f1_score

def clean_text(tweet):
    """Remove Twitter handles from text."""
    assert isinstance(tweet, str), f"Expected a string, but got {type(tweet)}, with value: {tweet}"
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'https?\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    #result = re.sub(r'<MENTION_[1-9]>', '', result)  # for AMI dataset 
    #result = re.sub(r'<URL>', '', result)  # for AMI dataset
    #result = re.sub(r'#', ' ', result)
    return result


def evaluate_and_save(model, model_path, dataset, device, model_ref="microsoft/mdeberta-v3-base"):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    loader = create_data_loader(dataset, AutoTokenizer.from_pretrained(model_ref), batch_size=16)
    evaluation = evaluate_model(model, loader, device)
    print(f"Evaluation results: Loss: {evaluation['loss']:.4f}, Binary Loss: {evaluation['loss_binary']:.4f}, Category Loss: {evaluation['loss_category']:.4f}")
    f1_score = get_f1_score(evaluation["records"])
    print(f"Binary F1 Score: {f1_score['binary']:.4f}, Category F1 Score: {f1_score['category']}")

def main(label2id_dict, do_train):
    #model_ref = "FacebookAI/xlm-roberta-large"
    #model_ref = "microsoft/mdeberta-v3-base"
    #model_ref = 'annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal'
    #model_ref = "MilaNLProc/njh-classifier"
    model_ref = "NLP-LTU/bertweet-large-sexism-detector"
    #model_ref = "cardiffnlp/twitter-roberta-base-hate-latest"

    multilingual = True if "xlm" in model_ref else True if "mdeberta" in model_ref else False
    print(f"Using model reference: {model_ref}")

    # Load the train dataset
    train_data = DataLoader("./data/EXIST 2025 Tweets Dataset/training/EXIST2025_training.json", split="train", label2id_dict=label2id_dict, multilingual=True).processed_data
    train_data["text"] = train_data["text"].apply(clean_text) # clean the text by removing Twitter handles and URLs for better model performance
    print(train_data.head())

    #print(train_data["binary_labels"].value_counts())
    #print(train_data["category_labels"].value_counts())

    test_data = DataLoader("./data/EXIST 2025 Tweets Dataset/dev/EXIST2025_dev.json", split="dev", label2id_dict=label2id_dict, multilingual=True).processed_data
    #print(test_data.head())
    #print(test_data["binary_labels"].value_counts())
    #print(test_data["category_labels"].value_counts())

    print(f"Length of train dataset before removing ties: {len(train_data)}")
    train_data = train_data[train_data["binary_labels"] != 99]
    print(f"Length of train dataset after removing binary ties: {len(train_data)}")
    train_data = train_data[train_data["category_labels"] != 99]
    print(f"Length of train dataset after removing category ties: {len(train_data)}")

    train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=42, stratify=train_data["binary_labels"])

    test_data = test_data[test_data["binary_labels"] != 99]
    test_data = test_data[test_data["category_labels"] != 99]
    #test_data["text"] = test_data["text"].apply(clean_text) # clean the text by removing Twitter handles and URLs for better model performance


    model = AppearanceMultiTaskClassifier(
        model_ref,
        num_category_labels=6,
        dropout=0.1,
        lambda_binary=1.0,
        lambda_category=1.0, # OG 1.0
        binary_loss_type="focal",#"ce",
        category_loss_type="focal",
        focal_gamma_binary=1.0,
        focal_gamma_category=2.0,
    ).to("cuda")

    print(vars(model))

    if do_train:
        train(model, train_data, val_data, label2id_dict, model_save_path=f"{model_ref.split("/")[-1]}.pt", model_ref=model_ref)

    evaluate_and_save(model, f"{model_ref.split("/")[-1]}.pt", test_data, "cuda", model_ref=model_ref)


if __name__ == "__main__":
    label2id_dict = {"level_1": {"NO":0, "YES":1}, "level_2": {"-":0, "DIRECT":1, "JUDGEMENTAL":2, "REPORTED":3}, "level_3": {"-":0, "IDEOLOGICAL-INEQUALITY":1, "STEREOTYPING-DOMINANCE":2, "OBJECTIFICATION":3, "SEXUAL-VIOLENCE":4, "MISOGYNY-NON-SEXUAL-VIOLENCE":5}}

    main(label2id_dict, args.train)

