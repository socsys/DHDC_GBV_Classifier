import os # must import before torch to have desired effect 
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding, BitsAndBytesConfig

from datasets import Dataset
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
        labels = torch.tensor(processed_data["labels"].tolist())
        comment_id = torch.tensor(processed_data["data_id"].tolist())
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            labels.to(torch.float),
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
        if 1 in self.level_list: assert "labels_1" in processed_data.columns, f"Expected column 'labels_1' in processed data, but got {processed_data.columns}"
        if 3 in self.level_list: assert "labels_3" in processed_data.columns, f"Expected column 'labels_3' in processed data, but got {processed_data.columns}"
        processed_data.rename(columns={"labels_1": "binary_labels" if 1 in self.level_list else "labels_1", "labels_3": "category_labels" if 3 in self.level_list else "labels_3"}, inplace=True)
        if clean:
            processed_data["text"] = processed_data["text"].apply(clean_text)
        processed_data["data_id"] = processed_data.index.tolist()  # add a unique identifier for each data point, which can be used for tracking during inference
        return processed_data

    def final_labels(self,x, level):
        labels = [0.0 for _ in range(len(self.label2id_dict[f"level_{level}"]))]
        if self.data_name == "EXIST":
            non_gbv = "NO" if level == 1 else "-"
            if x[non_gbv] == 3:
                return 99
            elif x[non_gbv] > 3:
                labels[0] = 1.0
                return labels
            else:
                if level == 1:
                    labels[1] = 1.0
                    return labels
                elif level == 3:
                    for k, v in x.items():
                        if k in self.label2id_dict[f"level_{level}"]:
                            if v >= 2:
                                labels[self.label2id_dict[f"level_{level}"][k]] = 1.0
                    labels[0] = 0.0 # ensure that the non-gbv label is set to 0 if any gbv label is present
            if sum(labels) == 0:
                return 99
        elif self.data_name == "AMI":
            index = self.label2id_dict[f"level_{level}"][str(x)]
            labels[index] = 1.0
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
# Define training loop  
# ----------------------------------


def move_batch_to_device(batch, device: str = "cuda", inf=False) -> Dict[str, Any]:
    if not inf:
        tensor_keys = ["input_ids", "attention_mask", "labels"]
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

def evaluate_model(model, loader, device) -> Dict[str, Any]:
    model.eval()
    local_records: List[Dict[str, Any]] = []
    loss_values: List[float] = []
    evaluation: Dict[str, Any] = {}

    autocast_enabled = device == "cuda"
    with torch.no_grad():
        for step, batch in enumerate(loader):
            moved = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device, enabled=autocast_enabled):
                outputs = model(
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                    labels=moved["labels"],
                )

                outputs["probs"] = torch.sigmoid(outputs["logits"])
                outputs["pred_labels"] = prob_to_label(outputs["probs"])

                for index, item in enumerate(zip(moved["labels"], outputs["pred_labels"])):
                    record = {
                        "true_labels": item[0].cpu().numpy().tolist(),
                        "pred_labels": item[1].cpu().numpy().tolist(),
                    }
                    local_records.append(record)

    evaluation["loss"] = outputs["loss"].item()
    evaluation["records"] = {"true": [record["true_labels"] for record in local_records], "pred": [record["pred_labels"] for record in local_records]}

    return evaluation 

def check_pred_error(prediction_list):
    for i, pred in enumerate(prediction_list):
        if pred[0] == 1 and any(x == 1 for x in pred[1:]):
            print(f"Error in prediction at index {i}: {pred}")


class CustomTrainer(Trainer):
    def __init__(self, weight_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_tensor = weight_tensor

    def compute_loss_function(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        class_matrix = self.weight_tensor.unsqueeze(0).expand_as(logits)

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=class_matrix, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(labels >= 0.5, probs, 1 - probs).clamp(1e-4, 1 - 1e-4)
        focal_weight = (1 - pt) ** 0.6 # gamma = 0.6 is best, 0.0425 ICM
        loss = focal_weight * bce_loss
        loss = loss.mean()  # average over the batch
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return self.compute_loss_function(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)

def custom_eval(evalpred):
    logits, labels = evalpred
    probs = torch.sigmoid(torch.tensor(logits))
    pred_labels = (probs >= 0.5).long()
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits), torch.tensor(labels).float(), reduction="mean")
    f1 = sklearn.metrics.f1_score(labels, pred_labels, average="macro")
    return {"loss": bce_loss.item(), "f1": f1}

def train(model, train_dataset, val_dataset, label2id_dict, best_save_path=None, model_ref="microsoft/mdeberta-v3-base", lr=6e-6, weight_decay=0.03, device="cuda"):
    print("No checkpoint found, starting from scratch.")

    class_weights = compute_class_weights(train_dataset.labels, num_classes=len(label2id_dict["level_3"]))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    print(f"Using learning rate: {lr}")

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    #model.config.pad_token_id = tokenizer.pad_token_id

    train_loader = create_data_loader(train_dataset, tokenizer, batch_size=32)
    val_loader = create_data_loader(val_dataset, tokenizer, batch_size=32)

    train_dataset = Dataset.from_pandas(train_dataset)
    val_dataset = Dataset.from_pandas(val_dataset)

    tokenized_train = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=256))
    tokenized_val = val_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=256))


    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=50,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        label_names=["labels"],
        num_train_epochs=3,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        
    )

    trainer = CustomTrainer(
        model=model,
        weight_tensor=weight_tensor,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    model = trainer.model


    torch.save(model.state_dict(), best_save_path)
    return model, tokenizer 


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

def get_value_counts(list_of_lists):
    for i in range(len(list_of_lists[0])):
        counts = Counter(tuple(record[i] for record in list_of_lists))
        print(f"Value counts for index {i}: {counts}")

class ONNXWRapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["logits"]

def evaluate_and_save(model, model_save_path=None, dataset=None, device="cuda", model_ref="NLP-LTU/bertweet-large-sexism-detector", train_data_name=None, tokenizer=None):
    loader = create_data_loader(dataset, tokenizer, batch_size=16)

    evaluation = evaluate_model(model, loader, device)
    print(f"Evaluation results: Loss: {evaluation['loss']:.4f}")

    gold_labels = [record for record in evaluation["records"]["true"]]
    pred_labels = [record for record in evaluation["records"]["pred"]]

    get_value_counts(gold_labels)
    get_value_counts(pred_labels)

    check_pred_error(pred_labels)

    print(f"Gold labels examples: {gold_labels[:10]}")
    print(f"Predicted labels examples: {pred_labels[:10]}")

    if train_data_name == "EXIST":
        print("Calculating ICM for EXIST dataset...")
        calculate_icm(gold_labels, pred_labels)

    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model state dict saved at {model_save_path}")

    f1_score = sklearn.metrics.f1_score(gold_labels, pred_labels, average="weighted")
    print(f"Weighted F1 score: {f1_score:.4f}")

    model.eval()

    #model.save_pretrained("gbv-model-simple")
    #model.push_to_hub("gbv-model")
    #tokenizer.save_pretrained("gbv-model-simple")
    #tokenizer.push_to_hub("gbv-model")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    quantized_model = AutoModelForSequenceClassification.from_pretrained("gbv-model-simple", quantization_config=quantization_config)

    quantized_model.cpu()

    quantized_model.save_pretrained("gbv-model-quantized")
    tokenizer.save_pretrained("gbv-model-quantized")

    '''
    onnx_model = ORTModel.from_pretrained(
        "./gbv-model-quantized", 
        export=True
    )
    onnx_model.save_pretrained("gbv-model-quantized")

    exit()
    
    wrapped_model = ONNXWRapper(quantized_model)
    wrapped_model.eval()




    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128), dtype=torch.long).to(device)  # batch size 1, sequence length 128
    #dummy_attention_mask = torch.ones(1, 128, dtype=torch.long).to(device)
    #example_input = (dummy_input_ids, dummy_attention_mask) # input_ids and attention_mask
    onnx_program = torch.onnx.export(wrapped_model, dummy_input_ids, "gbv-model-quantized/gbv-model.onnx", dynamo=False, opset_version=17)#, input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes={
    #    "input_ids": {0: "batch_size", 1: "sequence_length"},
    #    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    #    "logits": {0: "batch_size"},
    #})
    
    exit()

    quantize_dynamic(
        model_input="gbv-model-quantized/gbv-model.onnx",
        model_output="gbv-model-quantized/gbv-model.quant.onnx",
        weight_type=QuantType.QInt8
    )
    '''

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

def process_category_pred(pred_category, id2label_dict):
    """Convert prediction category indices to labels and pipe-separated string."""
    #index_list = ast.literal_eval(pred_category_str)
    index_list = pred_category
    labels = [id2label_dict[i] for i, val in enumerate(index_list) if val == 1]
    pipe_str = "|".join(labels) if labels else "-"
    return labels, pipe_str


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

    test_data = CustomDataLoader(data_name = train_data_name, data_path = val_data_path,  split="dev", label2id_dict=label2id_dict, multilingual=True).load_processed_data([ 3], clean=False)
    #print(test_data.head())
    #print(test_data["binary_labels"].value_counts())
    #print(test_data["category_labels"].value_counts())
    test_data.rename(columns={"category_labels": "labels"}, inplace=True)
    test_data = test_data[test_data["labels"] != 99]
    print(f"Length of test dataset after removing ties: {len(test_data)}")

    model = AutoModelForSequenceClassification.from_pretrained(model_ref, num_labels=num_category_labels, problem_type="multi_label_classification", ignore_mismatched_sizes=True).to("cuda")
    model.config.id2label = {v: k for k, v in label2id_dict["level_3"].items()}
    model.config.label2id = label2id_dict["level_3"]

    lr = 1e-6
    weight_decay = 0.01



    if train_flag:
        # Load the train dataset
        train_data = CustomDataLoader(data_name=train_data_name, data_path = train_data_path, label2id_dict=label2id_dict, split="train", multilingual=multilingual).load_processed_data([3], clean=True)
        #print(train_data["binary_labels"].value_counts())
        #print(train_data["category_labels"].value_counts())    
        train_data.rename(columns={"category_labels": "labels"}, inplace=True)
        train_data = train_data[train_data["labels"] != 99]
        print(f"Length of train dataset after removing ties: {len(train_data)}")
        print(train_data.head())


        train_data, val_data = train_test_split(train_data, test_size=0.05)#, stratify=train_data["labels"])

        trained_model, tokenizer = train(model, train_data, val_data, label2id_dict, best_save_path=f"checkpoints/{model_ref.split('/')[-1]}_{train_data_name}_simple.pt", model_ref=model_ref, lr=lr, weight_decay=weight_decay, device="cuda")

        model = evaluate_and_save(trained_model, model_save_path=f"classifier_state_dicts/{model_ref.split('/')[-1]}_{train_data_name}_simple.pt", dataset=test_data, device="cuda", model_ref=model_ref, train_data_name=train_data_name, tokenizer=tokenizer)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer)) 
        model.load_state_dict(torch.load(f"checkpoints/{model_ref.split('/')[-1]}_{train_data_name}_simple.pt"))
        model.to("cuda")

    if not args.inf:
        model = evaluate_and_save(model, dataset=test_data, device="cuda", model_ref=model_ref, train_data_name=train_data_name, tokenizer=tokenizer)
        print(model.config)

    elif args.inf:
        inf_data = pd.read_csv(args.inf_data_path)
        inf_data.rename(columns={"pure_text": "text", "item_id": "comment_id"}, inplace=True) # rename relevant text column to text for inference
        inf_data.dropna(subset=["text"], inplace=True)
        inf_data = inf_data[["text", "comment_id"]] #.sample(n=1000).reset_index(drop=True)
        print(inf_data.head())
        print(f"Length of inference dataset: {len(inf_data)}")
        #inf_data = pd.DataFrame({"text": inf_data, "comment_id": [1, 2, 3]})
        predictions = inference(model, inf_data, device="cuda", resume=args.resume_inf, tokenizer=tokenizer)
        print(f"Length of predictions: {len(predictions)}")
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

        labelled_texts.to_csv(f"{args.inf_data_path.split('.')[0]}_inference_predictions_final.csv", index=False)
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

