import os # must import before torch to have desired effect 
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from transformers import AutoTokenizer, AutoModel, AutoConfig
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
from typing import Any, Dict, List
from torch.optim import AdamW
import math
import random
import logging 
from tqdm import tqdm
import onnx
import ast
from huggingface_hub import PyTorchModelHubMixin
from handle_data import compute_class_weights, create_data_loader
from evaluate_model import _evaluate_model
from utils import move_batch_to_device

torch.cuda.empty_cache()


# ----------------------------------
# Define model 
# ----------------------------------

class GBVMultiTaskClassifier(nn.Module, PyTorchModelHubMixin):
    ''' Custom GBV Classifier with two heads, for binary and multi-label classification. '''
    def __init__(
        self,
        model_name: str,
        *,
        num_category_labels: int = 6,
        dropout: float = 0.1,
        lambda_binary: float = 1.0, # How much to weight the binary loss relative to the category loss
        lambda_category: float = 1.0, # How much to weight the category loss relative to the binary loss
        binary_loss_type: str = "focal",
        category_loss_type: str = "focal",
        focal_gamma_binary: float = 2.0, # Focal loss gamma for binary classification, determines how much to down-weight easy examples
        focal_gamma_category: float = 2.0, # Focal loss gamma for category classification, determines how much to down-weight easy examples
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
        if self.binary_loss_type != "focal":
            raise ValueError(f"Code currently only supports focal loss for binary classification, but got {self.binary_loss_type}")
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
        if self.category_loss_type != "focal":
            raise ValueError(f"Code currently only supports focal loss for category classification, but got {self.category_loss_type}")
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

            #only include category loss for items labeled as GBV (binary label 1)
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

def train(model, tokenizer, train_dataset, val_dataset, label2id_dict, best_save_path=None, lr=5e-6, weight_decay=0.01, device="cuda"):
    print("Training from scratch with hyperparameters:")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")

    binary_class_weights = compute_class_weights(train_dataset.binary_labels, num_classes=len(label2id_dict["level_1"]))
    category_class_weights = compute_class_weights(train_dataset.category_labels, num_classes=len(label2id_dict["level_2"]))
    binary_weight_tensor = torch.tensor(binary_class_weights, dtype=torch.float32, device=device)
    category_weight_tensor = torch.tensor(category_class_weights, dtype=torch.float32, device=device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device=device)
    
    train_loader = create_data_loader(train_dataset, tokenizer, batch_size=32)
    val_loader = create_data_loader(val_dataset, tokenizer, batch_size=32)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001, warmup_steps=100, mode="min")

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
                validation = _evaluate_model(model, val_loader, device)
                val_loss = validation['loss']

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if best_save_path is not None:
                        torch.save(model.state_dict(), best_save_path)
                        print(f"New best model saved at step {step} with validation loss {val_loss:.4f}")

                if early_stopping.step(val_loss, global_step):
                    print(f"Early stopping triggered at global step {step} with validation loss {val_loss:.4f}. Best validation loss was {best_val_loss:.4f} at step {early_stopping.best_step}.")
                    break

        validation = _evaluate_model(model, val_loader, device)
        print(f"Validation loss for epoch {epoch}: {validation['loss']:.4f}, binary loss: {validation['loss_binary']:.4f}, category loss: {validation['loss_category']:.4f}")
        print(f"Validation F1 for epoch {epoch}: {validation['f1_category']:.4f}")

    # load best model at end 
    print("Loading best model from checkpoint for final evaluation.")
    model.eval()
    model.load_state_dict(torch.load(best_save_path, map_location=device, weights_only=True)) if best_save_path is not None else None

    return model


# ----------------------------------
# Define optimization class 
# ----------------------------------

class Optimizer:
    def __init__(self, model, tokenizer, dataset, device="cuda", label2id_dict=None):
        self.device = device
        self.model = model 
        self.tokenizer = tokenizer
        self.dataset = dataset.sample(frac=0.2, random_state=42).reset_index(drop=True) # shuffle dataset
        self.df_train, self.df_val = train_test_split(self.dataset, test_size=0.10, stratify=self.dataset["binary_labels"], random_state=42)
        self.train_loader = create_data_loader(self.df_train, self.tokenizer, batch_size=32)
        self.val_loader = create_data_loader(self.df_val, self.tokenizer, batch_size=32)

        binary_class_weights = compute_class_weights(self.df_train.binary_labels, num_classes=2)
        category_class_weights = compute_class_weights(self.df_train.category_labels, num_classes=len(label2id_dict["level_2"]))
        self.binary_weight_tensor = torch.tensor(binary_class_weights, dtype=torch.float32, device=self.device)
        self.category_weight_tensor = torch.tensor(category_class_weights, dtype=torch.float32, device=self.device)

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
        self.model.to(self.device)
        self.model.focal_gamma_category = focal_gamma_category

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler(enabled=(self.device == "cuda"))

        best_val_f1 = float(0)
        for epoch in range(3):
            self.model.train()
            for step, batch in enumerate(self.train_loader):
                optimizer.zero_grad(set_to_none=True)
                moved = move_batch_to_device(batch, self.device)
                with torch.autocast(device_type=(self.device if self.device == "cuda" else "cpu"), enabled=(self.device == "cuda")):
                    outputs = self.model(
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
            validation = _evaluate_model(self.model, self.val_loader, self.device)
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


