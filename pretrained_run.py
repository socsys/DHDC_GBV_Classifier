import os
import sys
import pickle
import re
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import json
import random
from nltk.tokenize import RegexpTokenizer
import torch
import sklearn 
import argparse
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from itertools import combinations
import itertools
import time
from custom_code import LLM, Classifier, Predictor, DatasetPrepper, ICMCalculator, calculate_class_weights, perform_evaluation, remove_handles, tokenize_function, evaluate_predictions
import huggingface_hub
import math 
import re 
from tqdm import tqdm

# Models to do:
# cardiffnlp/twitter-roberta-base-hate-multiclass-latest





original_df = pd.read_csv(f"for_peter/bluesky_posts_RecollectionApr28_combinedResults_afterFiltering.csv")
test_input = original_df.rename(columns={"cleaned_text": "text"})
test_input = test_input.dropna(subset=["text"])
test_input = test_input[test_input["item_type"] == "reply"]
test_input = test_input[test_input["author_type"] == "by_public"]
test_input = test_input[(test_input["duplicates"] == "no duplicates") | (test_input["duplicates"] == "first duplicate")]
test_input = test_input.sample(n=10000, random_state=42).reset_index(drop=True) 


#model_list = ["wetey/distilbert-base-uncased-measuring-hate-speech", "facebook/roberta-hate-speech-dynabench-r4-target"]
#model_list = ["NLP-LTU/bertweet-large-sexism-detector"]
#model_list = ["tum-nlp/bertweet-sexism"]
#model = "facebook/roberta-hate-speech-dynabench-r4-target"
#model_name = "r4"
#model = "NLP-LTU/bertweet-large-sexism-detector"
#model_name = "bertweet_sexism"

#model= "ISEGURA/roberta_edos_a"
#model_name = "edos_a"
'''
model="tum-nlp/bertweet-sexism"
model_name = "tum_bertweet_sexism"

def predict(model, model_name, test_input):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    inputs = tokenizer(test_input["text"].tolist(), return_tensors="pt", truncation=True, padding=True, max_length=512)
    print(f"{model} loaded")
    predictions = []
    completed = 0
    with open(f"for_peter/{model_name}_predictions.csv", "w") as f:
        with torch.no_grad():
            for batch in tqdm(range(0, len(test_input), 32)):
                batch_inputs = {k: v[batch:batch+32] for k, v in inputs.items()}
                outputs = model(**batch_inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).tolist()
                predictions.extend(prediction)
                completed += 32
                if completed % 3200 == 0:
                    print(f"Completed {completed}/{len(test_input)}")
                    pd.DataFrame({"item_id": test_input["item_id"][:completed], "predicted_label": predictions}).to_csv(f"for_peter/{model_name}_predictions.csv", index=False)
        
    #print(predictions)

    test_input["predicted_label"] = predictions
    print(test_input["predicted_label"].value_counts())

    female = test_input[test_input["gender"] == "Female"]
    print(100*(sum(female["predicted_label"])/len(female["predicted_label"])))

    male = test_input[test_input["gender"] == "Male"]
    print(100*( sum(male["predicted_label"])/len(male["predicted_label"])))

    return test_input


print(f"Evaluating {model}")
test_input = predict(model, model_name, test_input)
test_input[["item_id", "predicted_label"]].to_csv(f"for_peter/predictions_{model_name}.csv", index=False)
'''

#model = AutoModelForSequenceClassification.from_pretrained("ac8736/toxic-tweets-fine-tuned-distilbert")
#tokenizer = AutoTokenizer.from_pretrained("ac8736/toxic-tweets-fine-tuned-distilbert")
#model_name = "toxic_tweets_distilbert"

model = AutoModelForSequenceClassification.from_pretrained("MilaNLProc/njh-classifier")
tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/njh-classifier")
model_name = "njh_classifier"


def predict_multilabel(model, model_name, test_input, labels):
    print(test_input["text"].tolist()[:5])
    inputs = tokenizer(test_input["text"].tolist(), truncation=True, padding='max_length', return_tensors="pt", max_length=256)

    probs = torch.zeros((len(test_input), len(labels)))
    completed = 0
    predicted_labels = []
    with open(f"for_peter/{model_name}_predictions.csv", "w") as f:
        for batch in tqdm(range(0, len(test_input), 32)):
            batch_inputs = {k: v[batch:batch+32] for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**batch_inputs)
                predictions = torch.sigmoid(outputs.logits)*100
                probs[batch:batch+32] = predictions
            for i in range(batch, batch+32):
                if i >= len(test_input):
                    break
                predictions = []
                for j, label in enumerate(labels):
                    if probs[i][j] > 50:
                        predictions.append(label)
                if predictions == []:
                    predicted_labels.append("none")
                else:
                    predicted_labels.append(', '.join(predictions))
            completed += 32
            if completed % 3200 == 0:
                print(f"Completed {completed}/{len(test_input)}")
                print(f"Length of predicted_labels: {len(predicted_labels)}")
                print(f"Length of test_input: {len(test_input['item_id'][:completed])}")
                pd.DataFrame({"item_id": test_input["item_id"][:completed], "predicted_label": predicted_labels}).to_csv(f"for_peter/{model_name}_predictions.csv", index=False)


    test_input["predicted_label"] = predicted_labels

    print(test_input["predicted_label"].value_counts())

    return test_input

#labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
labels = [ "Profanity", "Insults","Char. Assassination","Outrage",
    "Discrimination",
    "Hostility",
    "Incivility",
    "Intolerance"
  ]

if not os.path.exists(f"for_peter/predictions_{model_name}.csv"):
    test_input = predict_multilabel(model, model_name, test_input, labels)
    test_input[["item_id", "predicted_label"]].to_csv(f"for_peter/predictions_{model_name}.csv", index=False)
else:
    test_input = pd.read_csv(f"for_peter/predictions_{model_name}.csv")

test_input = pd.merge(test_input, original_df[["item_id", "gender"]], on="item_id", how="left")

female = test_input[test_input["gender"] == "Female"]
counts = {}
for label in ["none"] + labels:
    counts[label] = 100*sum(female["predicted_label"].apply(lambda x: label in x))
counts = {k: v/len(female["predicted_label"]) for k, v in counts.items()}
print(counts) 

male = test_input[test_input["gender"] == "Male"]
male_counts = {}
for label in ["none"] + labels:
    male_counts[label] = 100*sum(male["predicted_label"].apply(lambda x: label in x))
male_counts = {k: v/len(male["predicted_label"]) for k, v in male_counts.items()}
print(male_counts)