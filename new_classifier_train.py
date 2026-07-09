import os # must import before torch to have desired effect 
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import sklearn
from argparse import ArgumentParser
import optuna
from collections import defaultdict
from sklearn.model_selection import train_test_split
import re 
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import Any, Dict, List
from torch.optim import AdamW
import math
import random
import logging 
from tqdm import tqdm

import ast

from huggingface_hub import PyTorchModelHubMixin



parser = ArgumentParser()
parser.add_argument("--train", action="store_true", help="Whether to run training.")
parser.add_argument("--train_data_name", type=str, default="EXIST", help="Name of the training dataset. Options: EXIST")
parser.add_argument("--train_data_path", nargs="?", type=str, const="../DHDC/data/EXIST 2025 Tweets Dataset/training/EXIST2025_training.json", help="Path to the training dataset.")
parser.add_argument("--val_data_path", type=str, nargs="?", const="../DHDC/data/EXIST 2025 Tweets Dataset/dev/EXIST2025_dev.json", help="Path to the validation dataset.")
parser.add_argument("--inf", action="store_true", help="Whether to run inference with the trained model.")
parser.add_argument("--inf_data_path", type=str, nargs="?", const="/home/eddie/DHDC/exp10_mixed_weak_gold_deberta-v3-small_predictions_bluesky_posts_RecollectionApr28_cleaned.csv", help="Path to the dataset for inference.")
parser.add_argument("--resume_inf", action="store_true", help="Whether to resume inference.")
parser.add_argument("--model", type=str, default="NLP-LTU/bertweet-large-sexism-detector", help="Model reference for the Hugging Face model to use. Options: FacebookAI/xlm-roberta-large, microsoft/mdeberta-v3-base, annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal, MilaNLProc/njh-classifier, NLP-LTU/bertweet-large-sexism-detector, cardiffnlp/twitter-roberta-base-hate-latest")
args = parser.parse_args()

# Logger for report
#rep_logger = logging.getLogger("report_logger")
#rep_logger.addHandler(logging.FileHandler("report.log", mode="w"))

# Logger for debugging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='analysis.log', filemode='w')

def set_seed(seed):
    torch.manual_seed(seed)  # Sets seed for CPU operations
    np.random.seed(seed)     # Sets seed for NumPy
    random.seed(seed)        # Sets seed for Python's random module
    
    if torch.cuda.is_available():  # GPU-specific settings
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disables non-deterministic optimizations

def compute_class_weights(labels, num_classes=6):
    N = len(labels)
    pos_counts = []
    for i in range(num_classes):
        #print(f"Class {i} positive count: {sum(label[i] for label in labels)}")
        pos_counts.append(sum(label[i] for label in labels))
    #print(f"Positive counts per class: {pos_counts}")
    neg_counts = [N - pos for pos in pos_counts]
    class_weights = [x/y for x,y in zip(neg_counts, pos_counts) if y > 0]
    print(f"Calculated class weights: {class_weights}")
    return class_weights

def flatten(xss):
    return [x for xs in xss for x in xs]

# ----------------------------------
# Data loading and processing
# ----------------------------------

def clean_text(tweet):
    """Remove Twitter handles from text."""
    assert isinstance(tweet, str), f"Expected a string, but got {type(tweet)}, with value: {tweet}"
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'https?\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'<MENTION_[1-9]>', '', result)  # for AMI dataset 
    result = re.sub(r'<URL>', '', result)  # for AMI dataset
    #result = re.sub(r'#', ' ', result)
    return result

def create_data_loader(processed_data, tokenizer, batch_size=16, inf=False):
    encodings = tokenizer(
        processed_data["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    if not inf:
        binary_labels = torch.tensor(processed_data["binary_labels"].tolist())
        category_labels = torch.tensor(processed_data["category_labels"].tolist())
        comment_id = torch.tensor(processed_data["data_id"].tolist())
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            binary_labels.to(torch.float),
            category_labels.to(torch.float),
            comment_id,
        )
    else:
        #comment_id = torch.tensor(processed_data["comment_id"].tolist())
        comment_id = processed_data["comment_id"].tolist()  # keep comment IDs as a list of ints for inference, since we won't be moving them to device or using them in tensors
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            comment_id,
        )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def set_multilingual(model_ref):
    if model_ref in ["FacebookAI/xlm-roberta-large", "microsoft/mdeberta-v3-base", "annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal", "NLP-LTU/bertweet-large-sexism-detector"]: # bertweet performs better on multilingual training data, despite being an English-language model 
        return True
    return False

class CustomDataLoader:
    def __init__(self, data_name, data_path, split, label2id_dict, multilingual=False):
        self.data_name = data_name
        self.data_path = data_path
        self.split = split
        self.label2id_dict = label2id_dict
        self.multilingual = multilingual

    def load_raw_data(self):
        print(self.data_path)
        if self.data_path.endswith(".json"):
            raw_data = pd.read_json(self.data_path, orient='index')
        elif self.data_path.endswith(".csv") or self.data_path.endswith(".tsv"):
            raw_data = pd.read_csv(self.data_path, sep='\t' if self.data_path.endswith(".tsv") else ',')
        if self.multilingual and self.data_name == "EXIST":
            if self.split == "train":
                raw_data = raw_data[raw_data["split"].isin(["TRAIN_EN", "TRAIN_ES"])]
            else:
                raw_data = raw_data[raw_data["split"].isin(["DEV_EN"])]
        return raw_data

    def load_processed_data(self, level_list, clean=False):
        self.level_list = level_list 
        raw_data = self.load_raw_data()
        raw_data.rename(columns={"tweet": "text"}, inplace=True) if "tweet" in raw_data.columns else None
        processed_data = self.handle_labels(raw_data)
        assert "labels_1" in processed_data.columns and "labels_3" in processed_data.columns, f"Expected columns 'labels_1' and 'labels_3' in processed data, but got {processed_data.columns}"
        processed_data.rename(columns={"labels_1": "binary_labels", "labels_3": "category_labels"}, inplace=True)
        if clean:
            processed_data["text"] = processed_data["text"].apply(clean_text)
        processed_data["data_id"] = processed_data.index.tolist()  # add a unique identifier for each data point, which can be used for tracking during inference
        return processed_data

    def final_labels(self,x, level):
        labels = [0 for _ in range(len(self.label2id_dict[f"level_{level}"]))]
        if self.data_name == "EXIST":
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
                        if k in self.label2id_dict[f"level_{level}"]:
                            if v >= 2:
                                labels[self.label2id_dict[f"level_{level}"][k]] = 1
                    labels[0] = 0 # ensure that the non-gbv label is set to 0 if any gbv label is present
            if sum(labels) == 0:
                return 99
        elif self.data_name == "AMI":
            index = self.label2id_dict[f"level_{level}"][str(x)]
            labels[index] = 1
        return labels


    def handle_labels(self, dataset):
        label_list = []
        if self.data_name == "EXIST":
            col_list = [f"labels_task1_{i}" for i in range(1,4)]
        elif self.data_name in ["AMI"]:
            col_list = ["misogynous", "target", "misogyny_category"]
        print(col_list)
        for level in self.level_list:
            label_list.append(f"labels_{level}")
            col = col_list[level-1]
            print(f"Processing level {col} for {self.data_name}")
            dataset = dataset.dropna(subset=[col]) # drop rows with missing labels for this level
            dataset[f"labels_{level}"] = dataset[col].apply(lambda x: x if isinstance(x, str) or isinstance(x, int) else (Counter(x) if isinstance(x[0], str) else Counter(flatten(x))))
            dataset[f"labels_{level}"] = dataset[f"labels_{level}"].apply(lambda x: self.final_labels(x, level))

        return dataset[["text"] + label_list]


# ----------------------------------
# Define model 
# ----------------------------------

class GBVMultiTaskClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_name: str,
        *,
        num_category_labels: int = 6,
        dropout: float = 0.1,
        lambda_binary: float = 1.0,
        lambda_category: float = 1.0,
        binary_loss_type: str = "focal",
        category_loss_type: str = "focal",
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
        assert self.binary_loss_type == "focal", f"Code currently only supports focal loss for binary classification, but got {self.binary_loss_type}"
        ce_loss = F.cross_entropy(logits, label_indices, weight=class_weights, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, label_indices.unsqueeze(1)).squeeze(1).clamp(1e-4, 1 - 1e-4)
        return ((1 - pt) ** self.focal_gamma_binary) * ce_loss

    def _category_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.category_loss_type == "focal", f"Code currently only supports focal loss for category classification, but got {self.category_loss_type}"
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
            category_loss = self._category_loss(category_logits, category_labels, category_class_weights)

            #only include category loss for GBV items
            binary_positive_mask = binary_labels.argmax(dim=1).float()
            masked_category_loss = (category_loss.mean(dim=1) * binary_positive_mask)
 

            outputs["loss_binary"] = binary_loss.mean()
            outputs["loss_category"] = masked_category_loss.sum() / (binary_positive_mask.sum().clamp(min=1) )
            outputs["loss"] = (
                self.lambda_binary * outputs["loss_binary"]
                + self.lambda_category * outputs["loss_category"]
            )

        return outputs



# ----------------------------------
# Define training loop  
# ----------------------------------


def move_batch_to_device(batch, device: str = "cuda", inf=False) -> Dict[str, Any]:
    if not inf:
        tensor_keys = ["input_ids", "attention_mask", "binary_labels", "category_labels"]
    else:
        tensor_keys = ["input_ids", "attention_mask"]
    moved = defaultdict()
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

def predict_labels(outputs):
    binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)[:, 1].detach().cpu()
    binary_pred_mask_bool = (binary_probs >= 0.5)  # 1D bool tensor
    category_probs = torch.sigmoid(outputs["category_logits"]).detach().cpu()
    
    # zero out category predictions for non-GBV instances
    # category_probs = category_probs * binary_pred_mask_bool.unsqueeze(1).float()

    # Normalize outputs to native Python types
    binary_probs_list = [float(x) for x in binary_probs.tolist()]
    category_pred_tensor = prob_to_label(category_probs)
    category_pred = [list(map(int, row)) for row in category_pred_tensor.tolist()]
    # enforce non-GBV -> category 0
    category_pred = [pred if binary_pred_mask_bool[i].item() == True else [1] + [0]*(len(pred)-1) for i, pred in enumerate(category_pred)]
    category_conf = [list(map(float, row)) for row in category_probs.tolist()]

    return binary_pred_mask_bool, binary_probs_list, category_pred, category_conf

def evaluate_model(model, loader, device) -> Dict[str, Any]:
    model.eval()
    local_records: List[Dict[str, Any]] = []
    loss_values: List[float] = []
    loss_binary_values: List[float] = []
    loss_category_values: List[float] = []

    autocast_enabled = device == "cuda"
    with torch.no_grad():
        for step, batch in enumerate(loader):
            moved = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device, enabled=autocast_enabled):
                outputs = model(
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                    binary_labels=moved["binary_labels"],
                    category_labels=moved["category_labels"],
                )

            binary_pred_mask_bool, binary_probs_list, category_pred, category_conf = predict_labels(outputs)

            loss_values.append(float(outputs["loss"].detach().cpu().item()))
            loss_binary_values.append(float(outputs["loss_binary"].detach().cpu().item()))
            loss_category_values.append(float(outputs["loss_category"].detach().cpu().item()))

            for index, item in enumerate(zip(batch[-1], moved["binary_labels"], moved["category_labels"])):
                # normalize record fields to plain Python types
                #comment_id = int(item[0].item()) if hasattr(item[0], 'item') else int(item[0])
                comment_id = item[0]
                true_binary = int(item[1].argmax().item())
                true_category = [int(x) for x in item[2].cpu().tolist()]

                local_records.append(
                    {
                        "comment_id": comment_id,
                        "true_binary": true_binary,
                        "true_category": true_category,
                        "pred_binary": int(binary_pred_mask_bool[index].item()),
                        "pred_binary_prob": binary_probs_list[index],
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
    #print(f"Flat records dtype: {type(flat_records)}, example record: {flat_records[0] if len(flat_records) > 0 else 'N/A'}")
    true_category_list = [item["true_category"] for item in flat_records]
    pred_category_list = [item["pred_category"] for item in flat_records]

    return {
        "records": flat_records,
        "loss": float(sum(flat_loss_values) / max(len(flat_loss_values), 1)),
        "loss_binary": float(sum(flat_loss_binary) / max(len(flat_loss_binary), 1)),
        "loss_category": float(sum(flat_loss_category) / max(len(flat_loss_category), 1)),
        "f1_category": sklearn.metrics.f1_score(
            true_category_list,
            pred_category_list,
            average="weighted",
        ),
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

def train(model, train_dataset, val_dataset, label2id_dict, best_save_path=None, model_ref="microsoft/mdeberta-v3-base", lr=6e-6, weight_decay=0.03, device="cuda"):
    print("No checkpoint found, starting from scratch.")

    binary_class_weights = compute_class_weights(train_dataset.binary_labels, num_classes=len(label2id_dict["level_1"]))
    category_class_weights = compute_class_weights(train_dataset.category_labels, num_classes=len(label2id_dict["level_3"]))
    binary_weight_tensor = torch.tensor(binary_class_weights, dtype=torch.float32, device=device)
    category_weight_tensor = torch.tensor(category_class_weights, dtype=torch.float32, device=device)

    print(f"Using learning rate: {lr}")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler(device=device)

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    #model.config.pad_token_id = tokenizer.pad_token_id

    train_loader = create_data_loader(train_dataset, tokenizer, batch_size=32)
    val_loader = create_data_loader(val_dataset, tokenizer, batch_size=32)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001, warmup_steps=100, mode="min")
    #early_stopping = EarlyStopping(patience=10, min_delta=0.01, warmup_steps=100, mode="max")

    best_val_loss = float("inf")
    best_val_f1 = float(0)
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
            moved = move_batch_to_device(batch, device)
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
                validation = evaluate_model(model, val_loader, device)
                val_loss = validation['loss']
                #print(f"Step {step}: Training Loss: {loss.item():.4f}, Validation Loss: {validation['loss']:.4f}, binary loss: {validation['loss_binary']:.4f}, category loss: {validation['loss_category']:.4f}")
                val_f1 = validation["f1_category"]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if best_save_path is not None:
                        torch.save(model.state_dict(), best_save_path)
                        print(f"New best model saved at step {step} with validation loss {val_loss:.4f}")

                #if val_f1 > best_val_f1:
                #    best_val_f1 = val_f1
                #    if best_save_path is not None:
                #        torch.save(model.state_dict(), best_save_path)
                #        print(f"New best model saved at step {step} with validation F1 {val_f1:.4f}")

                if early_stopping.step(val_loss, global_step):
                    print(f"Early stopping triggered at global step {step} with validation loss {val_loss:.4f}. Best validation loss was {best_val_loss:.4f} at step {early_stopping.best_step}.")
                    break

                #if early_stopping.step(val_f1, global_step):
                #    print(f"Early stopping triggered at global step {step} with validation F1 {val_f1:.4f}. Best validation F1 was {best_val_f1:.4f} at step {early_stopping.best_step}.")
                #    break

        validation = evaluate_model(model, val_loader, device)
        print(f"Validation loss for epoch {epoch}: {validation['loss']:.4f}, binary loss: {validation['loss_binary']:.4f}, category loss: {validation['loss_category']:.4f}")
        print(f"Validation F1 for epoch {epoch}: {validation['f1_category']:.4f}")

    # load best model at end 
    print("Loading best model from checkpoint for final evaluation.")
    model.load_state_dict(torch.load(best_save_path))
    #torch.save(model.state_dict(), model_save_path)
    return model, tokenizer


# ----------------------------------
# Define optimization class 
# ----------------------------------

class Optimizer:
    def __init__(self, dataset, device="cuda", model_ref="NLP-LTU/bertweet-large-sexism-detector", num_category_labels=6):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
        self.device = device
        self.model_ref = model_ref
        dataset = dataset.sample(frac=0.2, random_state=42).reset_index(drop=True) # shuffle dataset
        self.df_train, self.df_val = train_test_split(dataset, test_size=0.10, stratify=dataset["binary_labels"], random_state=42)
        self.train_loader = create_data_loader(self.df_train, self.tokenizer, batch_size=32)
        self.val_loader = create_data_loader(self.df_val, self.tokenizer, batch_size=32)

        binary_class_weights = compute_class_weights(self.df_train.binary_labels, num_classes=2)
        category_class_weights = compute_class_weights(self.df_train.category_labels, num_classes=len(label2id_dict["level_3"]))
        self.binary_weight_tensor = torch.tensor(binary_class_weights, dtype=torch.float32, device=self.device)
        self.category_weight_tensor = torch.tensor(category_class_weights, dtype=torch.float32, device=self.device)
        self.dataset = dataset

        self.num_category_labels = num_category_labels

    def objective(self, trial):
        """Optuna objective: builds a model with sampled hyperparameters, runs a few epochs and returns validation loss to minimize.
        Uses pruning and reports intermediate results back to Optuna.
        """
        # sample hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        #focal_gamma_category = trial.suggest_float("focal_gamma_category", 0.0, 3.0)
        focal_gamma_category = 2 

        # build model
        model = GBVMultiTaskClassifier(
            model_name=self.model_ref,
            num_category_labels=self.num_category_labels,
            dropout=0.1,
            lambda_binary=1.0,
            lambda_category=1.0,
            binary_loss_type="focal",
            category_loss_type="focal",
            focal_gamma_binary=1.0,
            focal_gamma_category=focal_gamma_category,
        ).to(self.device)

        

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler(enabled=(self.device == "cuda"))

        best_val_f1 = float(0)
        for epoch in range(3):
            model.train()
            for step, batch in enumerate(self.train_loader):
                optimizer.zero_grad(set_to_none=True)
                moved = move_batch_to_device(batch, self.device)
                with torch.autocast(device_type=(self.device if self.device == "cuda" else "cpu"), enabled=(self.device == "cuda")):
                    outputs = model(
                        input_ids=moved["input_ids"],
                        attention_mask=moved["attention_mask"],
                        binary_labels=moved["binary_labels"],
                        category_labels=moved["category_labels"],
                        binary_class_weights=self.binary_weight_tensor,
                        category_class_weights=self.category_weight_tensor,
                    )
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # evaluate at epoch end
            validation = evaluate_model(model, self.val_loader, self.device)
            #val_loss = validation["loss"]
            val_f1 = validation["f1_category"]
            print(val_f1)

            # report intermediate objective value to Optuna and allow pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

        return best_val_f1

    def run_search(self, n_trials: int = 30, direction: str = "maximize"):
        """Method to run an Optuna study for this Optimizer instance and return the Study object.
        """
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
        study.optimize(self.objective, n_trials=n_trials)
        print(f"Best value: {study.best_value}, best params: {study.best_params}")
        return study


# ----------------------------------
# Define evaluation and saving functions 
# ----------------------------------


class ICMCalculator:
    def __init__(self, num_labels, labels):
        self.num_labels = num_labels
        self.labels = labels
        assert len(self.labels) > 0, "Labels list cannot be empty."         
        counts = [sum(item[i] == 1 for item in labels) for i in range(num_labels)]
        
        probabilities = [count / len(labels) for count in counts]

        # laplace smoothing
        probabilities = [prob + (1/len(self.labels)) for prob in probabilities]

        self.gbv_parent_idx = num_labels
        self.root_idx = num_labels + 1

        gbv_parent_prob = sum(1 for item in labels if any(item[i] == 1 for i in range(1, num_labels))) / len(labels) + (1/len(self.labels)) 
        self.probabilities = probabilities + [gbv_parent_prob, 1.0] # add probabilities for GBV parent and root nodes
        #print(f"Calculated probabilities for ICM: {self.probabilities}")

    def ic(self,label):
        return - math.log(self.probabilities[label], 2)

    def los(self, label1, label2):
        if label1 == label2:
            return label1
        if 0 in (label1, label2):
            return self.root_idx # if either label is non-GBV, LSO is root
        return self.gbv_parent_idx # if both labels are GBV but different, LSO is GBV parent node
        
    def get_ic(self, item, depth=0):
        positives = [i for i,v in enumerate(item) if v == 1] if depth == 0 else item 

        #print(f"Positives: {positives}")
        if len(positives) == 0:
            return 0.0

        elif len(positives) == 1:
            return self.ic(positives[0])
        
        c1 = positives.pop(0)
        lso_concepts = [self.los(c1, ci) for ci in positives]
        #print(lso_concepts)

        return self.ic(c1) + self.get_ic(positives, depth=1) - self.get_ic(lso_concepts, depth=1)

    def calculate_icm(self, predictions):
        icm = 0
        for gold, pred in zip(self.labels, predictions):
            union = [1 if g + p > 0 else 0 for g, p in zip(gold, pred)]
            icm += 2 * self.get_ic(gold) + 2 * self.get_ic(pred) - 3 * self.get_ic(union)
        return icm / len(self.labels)


def get_f1_score(records):
    f1_score = defaultdict(int)
    y_true_binary = [record["true_binary"] for record in records]
    y_pred_binary = [1 if record["pred_binary_prob"] >= 0.5 else 0 for record in records]
    f1_score["binary"] = sklearn.metrics.f1_score(y_true_binary, y_pred_binary)

    y_true_category = [record["true_category"] for record in records]
    y_pred_category = [record["pred_category"] for record in records]
    f1_score["category"] = sklearn.metrics.f1_score(y_true_category, y_pred_category, average=None)
    f1_score["category_macro"] = sklearn.metrics.f1_score(y_true_category, y_pred_category, average="macro")
    return f1_score

def calculate_icm(gold_labels, pred_labels, num_category_labels=6):
    gold_standard_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm(gold_labels)
    majority_class_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm([[1,0,0,0,0,0] for _ in gold_labels]) # all predictions are non-GBV
    minority_class_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm([[0,0,0,0,1,0] for _ in gold_labels]) # all predictions are the most common GBV category
    predicted_icm = ICMCalculator(num_category_labels, gold_labels).calculate_icm(pred_labels)
    print(f"Gold standard ICM: {gold_standard_icm:.4f}, Majority class ICM: {majority_class_icm:.4f}, Minority class ICM: {minority_class_icm:.4f}, Predicted ICM: {predicted_icm:.4f}")


def evaluate_and_save(model, model_save_path=None, dataset=None, device="cuda", model_ref="NLP-LTU/bertweet-large-sexism-detector", train_data_name=None, tokenizer=None):
    loader = create_data_loader(dataset, AutoTokenizer.from_pretrained(model_ref), batch_size=16)
    evaluation = evaluate_model(model, loader, device)
    print(f"Evaluation results: Loss: {evaluation['loss']:.4f}, Binary Loss: {evaluation['loss_binary']:.4f}, Category Loss: {evaluation['loss_category']:.4f}")
    f1_score = get_f1_score(evaluation["records"])
    print(f"Binary F1 Score: {f1_score['binary']:.4f}, Category F1 Score: {f1_score['category']}, Category Macro F1 Score: {f1_score['category_macro']:.4f}")

    gold_labels = [record["true_category"] for record in evaluation["records"]]
    pred_labels = [record["pred_category"] for record in evaluation["records"]]

    if train_data_name == "EXIST":
        print("Calculating ICM for EXIST dataset...")
        calculate_icm(gold_labels, pred_labels)

    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model state dict saved at {model_save_path}")

    model.save_pretrained('gbv-model')
    tokenizer.save_pretrained('gbv-model')
    
    return model 


# ----------------------------------
# Define inference function
# ----------------------------------

def split_into_slices(sequence, slice_size):
    for i in range(0, len(sequence), slice_size):
        yield slice(i, i + slice_size)

def inference(model, inf_data, device, resume=False, tokenizer=None):
    print("Starting inference...")
    model.eval()

    autocast_enabled = device == "cuda"

    batch_size = 32
    save_steps = 1000
    csv_path = f"{args.inf_data_path.split('.')[-2]}_inference_predictions_temp.csv"
    columns = ["comment_id", "pred_binary", "pred_binary_prob", "pred_category", "pred_category_confidence"]

    print(f"Length of inference data: {len(inf_data)}")

    tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained("NLP-LTU/bertweet-large-sexism-detector")

    inf_data_loader = create_data_loader(inf_data, tokenizer, batch_size=batch_size, inf=True)
    print(f"Created inference data loader with {len(inf_data_loader)} batches.")

    if resume == True:
        print(f"Loading previous records from CSV...")
        all_records = pd.read_csv(csv_path).to_dict(orient="records")
        resume_step = len(all_records) // batch_size
        print(f"Resuming inference from step {resume_step}.")
    else:
        all_records: List[Dict[str, Any]] = []

    local_records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(inf_data_loader), total=len(inf_data_loader), desc="Inference"):
            if resume == True and step < resume_step:
                continue
            moved = move_batch_to_device(batch, device, inf=True)
            with torch.autocast(device_type=device, enabled=autocast_enabled):
                outputs = model(
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                )

            binary_pred_mask_bool, binary_probs_list, category_pred, category_conf = predict_labels(outputs)

            for index, item in enumerate(zip(batch[-1], category_pred)): # iterate over comment_id field in batch, always final field
                # normalize record fields to plain Python types
                comment_id, category = item

                record = {
                        "comment_id": comment_id,
                        "pred_binary": int(binary_pred_mask_bool[index].item()),
                        "pred_binary_prob": binary_probs_list[index],
                        "pred_category": category_pred[index],
                        "pred_category_confidence": category_conf[index],
                    }

                local_records.append(record)
                all_records.append(record)

            if step % save_steps == 0 or step == len(inf_data_loader) - 1:
                print(f"Processed first {step} steps. Saving most recent records to CSV...")
                df = pd.DataFrame(local_records, columns=["comment_id", "pred_binary", "pred_binary_prob", "pred_category", "pred_category_confidence"])
                df.to_csv(f"{args.inf_data_path.split('.')[-2]}_inference_predictions_temp.csv", mode="a", index=False, header=False if os.path.exists(csv_path) else True)
                local_records: List[Dict[str, Any]] = []

   
    #gathered_records = gather_objects(local_records)

    return all_records

def process_category_pred(pred_category_str, id2label_dict):
    """Convert prediction category indices to labels and pipe-separated string."""
    index_list = ast.literal_eval(pred_category_str)
    labels = [id2label_dict[i] for i, val in enumerate(index_list) if val == 1]
    pipe_str = "|".join(labels) if labels else "-"
    return labels, pipe_str

# ---------------------------------
# Prep for use by gbv-d-toxify
# ---------------------------------

class GBVWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = [torch.zeros_like(outputs["category_logits"][0]) for _ in range(outputs["category_logits"].shape[0])]
        for i,item in enumerate(zip(outputs["binary_logits"], outputs["category_logits"])):
            binary_logits, category_logits = item
            binary_probs = torch.softmax(binary_logits, dim=-1)[1]
            category_probs = torch.sigmoid(category_logits)
            # enforce non-GBV -> category 0
            if binary_probs > 0.5:
                prediction[i] = category_probs
            else:
                prediction[i] = torch.zeros_like(category_probs)
                prediction[i][0] = 1  # Set the first category (non-GBV) to 1
        return prediction

# ----------------------------------
# Define main function
# ----------------------------------

def main(label2id_dict, train_flag=False, train_data_name = None, train_data_path=None, val_data_path=None, model_ref="NLP-LTU/bertweet-large-sexism-detector"):
    print(f"Using model reference: {model_ref}")

    if args.train and args.inf:
        print("To confirm, model is being trained and then used for inference.")

    set_seed(42)

    multilingual = set_multilingual(model_ref)

    num_category_labels = len(label2id_dict["level_3"])

    test_data = CustomDataLoader(data_name = train_data_name, data_path = val_data_path,  split="dev", label2id_dict=label2id_dict, multilingual=True).load_processed_data([1, 3], clean=False)
    #print(test_data.head())
    #print(test_data["binary_labels"].value_counts())
    #print(test_data["category_labels"].value_counts())
    test_data = test_data[test_data["binary_labels"] != 99]
    test_data = test_data[test_data["category_labels"] != 99]

    focal_gamma_category = 2 #1.33 #0.7 #2.8 #3.0
    lr = 5e-6 #5e-6 #6e-6 #1.5e-5 #6e-6 #1e-5
    weight_decay = 0.01 #0.01 #0.09 #0.03 #0.01

    model = GBVMultiTaskClassifier(
        model_ref,
        num_category_labels=num_category_labels,
        dropout=0.1,
        lambda_binary=1.0,
        lambda_category=1.0, # OG 1.0
        binary_loss_type="focal",#"ce",
        category_loss_type="focal",
        focal_gamma_binary=1.0,
        focal_gamma_category=focal_gamma_category,
    ).to("cuda")

    

    print(vars(model))

    # Optimizer study
    #optuna_study = Optimizer(train_data, device="cuda", model_ref=model_ref, num_category_labels=num_category_labels).run_search(n_trials=20)
    #print(f"Best hyperparameters from Optuna study: {optuna_study.best_params}")
    #exit()


    if train_flag:
        # Load the train dataset
        train_data = CustomDataLoader(data_name=train_data_name, data_path = train_data_path, label2id_dict=label2id_dict, split="train", multilingual=multilingual).load_processed_data([1, 3], clean=True)
        print(train_data.head())
        #print(train_data["binary_labels"].value_counts())
        #print(train_data["category_labels"].value_counts())    

        print(f"Length of train dataset before removing ties: {len(train_data)}")
        train_data = train_data[train_data["binary_labels"] != 99]
        print(f"Length of train dataset after removing binary ties: {len(train_data)}")
        train_data = train_data[train_data["category_labels"] != 99]
        print(f"Length of train dataset after removing category ties: {len(train_data)}")

        train_data, val_data = train_test_split(train_data, test_size=0.05, stratify=train_data["binary_labels"])

        trained_model, tokenizer = train(model, train_data, val_data, label2id_dict, best_save_path=f"checkpoints/{model_ref.split('/')[-1]}_{train_data_name}.pt", model_ref=model_ref, lr=lr, weight_decay=weight_decay, device="cuda")

        model = evaluate_and_save(trained_model, model_save_path=f"classifier_state_dicts/{model_ref.split('/')[-1]}_{train_data_name}.pt", dataset=test_data, device="cuda", model_ref=model_ref, train_data_name=train_data_name, tokenizer=tokenizer)

    else:
        model.load_state_dict(torch.load(f"classifier_state_dicts/{model_ref.split('/')[-1]}_{train_data_name}.pt"))
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        model.to("cuda")

    if not args.inf:
        model = evaluate_and_save(model, dataset=test_data, device="cuda", model_ref=model_ref, train_data_name=train_data_name, tokenizer=tokenizer)

        model.eval()

        wrapped_model = GBVWrapper(model)
        wrapped_model.eval()

        dummy_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long)
        dummy_mask = torch.ones((1, 128), dtype=torch.long)
        torch.onnx.export(wrapped_model, (dummy_ids.to("cuda"), dummy_mask.to("cuda")), "gbv_model.onnx", input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "logits": {0: "batch_size"}}, opset_version=14)

        exit()


        dummy_input = tokenizer(["This is a test input.", "She is a silly bitch!", "I'd love to take her out to dinner", "What a twat."], return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        og_output = model(input_ids=dummy_input["input_ids"].to("cuda"), attention_mask=dummy_input["attention_mask"].to("cuda"))
        print(f"Output from original model: {og_output}")
        output = wrapped_model(dummy_input["input_ids"].to("cuda"), dummy_input["attention_mask"].to("cuda"))
        print(f"Output from wrapped model: {output}")

        cp_test_data = test_data.copy()
        cp_test_data.rename(columns={"category_labels": "labels"}, inplace=True)

        cp_test_data["text"] = cp_test_data["text"].apply(lambda x: tokenizer(x, return_tensors="pt", padding="max_length", truncation=True, max_length=128))

        predictions = []
        with torch.no_grad():
            for item in cp_test_data["text"]:
                outputs = wrapped_model(input_ids=item["input_ids"].to("cuda"), attention_mask=item["attention_mask"].to("cuda"))
                predictions.append(outputs)

        print(cp_test_data.columns)
        gold_labels = cp_test_data["labels"].to_list()

        for i, prediction in enumerate(predictions):
            prediction = prediction[0].tolist()
            prediction = [1 if p > 0.5 else 0 for p in prediction]
            predictions[i] = prediction


        print(f"Gold labels: {gold_labels[:10]}")
        print(f"Predictions: {predictions[:10]}")

        icm_score = calculate_icm(gold_labels, predictions)

    elif args.inf:
        inf_data = pd.read_csv(args.inf_data_path)
        inf_data = inf_data[["text", "comment_id"]] #.sample(n=1000).reset_index(drop=True)
        #inf_data = pd.DataFrame({"text": inf_data, "comment_id": [1, 2, 3]})
        predictions = inference(model, inf_data, device="cuda", resume=args.resume, tokenizer=tokenizer)
        predictions = pd.DataFrame(predictions, columns=["comment_id", "pred_binary", "pred_binary_prob", "pred_category", "pred_category_confidence"])
        labelled_texts = pd.merge(inf_data, pd.DataFrame(predictions), on="comment_id")
        print(labelled_texts.head())
        for item in predictions["pred_category"]:
            if item[0] == 1 and any(x == 1 for x in item[1:]):
                print(f"Found instance where non-GBV category is predicted as 1 along with other categories: {item}")
        #print(predictions["pred_category"][:10])

        sample = labelled_texts[["text", "pred_binary"]].groupby("pred_binary").sample(n=10)
        print(sample)
        
        # Process predictions and create category labels/pipes directly
        id2label_dict = {v: k for k, v in label2id_dict["level_3"].items()}
        
        labelled_texts[["pred_category_labels", "pred_category_pipe"]] = labelled_texts["pred_category"].apply(
            lambda x: pd.Series(process_category_pred(x, id2label_dict))
        )
        
        labelled_texts.to_csv("inference_predictions.csv", index=False)
        print(labelled_texts.head())
    

if __name__ == "__main__":
    if args.train_data_name == "EXIST":
        label2id_dict = {"level_1": {"NO":0, "YES":1}, "level_2": {"-":0, "DIRECT":1, "JUDGEMENTAL":2, "REPORTED":3}, "level_3": {"-":0, "IDEOLOGICAL-INEQUALITY":1, "STEREOTYPING-DOMINANCE":2, "OBJECTIFICATION":3, "SEXUAL-VIOLENCE":4, "MISOGYNY-NON-SEXUAL-VIOLENCE":5}}
        if not args.train_data_path:
            args.train_data_path = "../DHDC/data/EXIST 2025 Tweets Dataset/training/EXIST2025_training.json"
        if not args.val_data_path:
            args.val_data_path = "../DHDC/data/EXIST 2025 Tweets Dataset/dev/EXIST2025_dev.json"

    elif args.train_data_name == "AMI":
        label2id_dict = {"level_1": {"0":0, "1":1}, "level_2":{"0":0, "active":1, "passive":2}, "level_3": {"0":0, "discredit":1, "sexual_harassment":2, "stereotype":3, "dominance":4, "derailing":5}}
        if not args.train_data_path:
            args.train_data_path = "../DHDC/data/combined_training_anon.tsv"
        if not args.val_data_path:
            args.val_data_path = "../DHDC/data/en_testing_labeled_anon.tsv"



    main(label2id_dict, train_flag = args.train, train_data_name=args.train_data_name, train_data_path=args.train_data_path, val_data_path=args.val_data_path, model_ref=args.model)

