# ----- In Draft, Does Not Currently Work ----- 


import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from new_classifier_train import AppearanceMultiTaskClassifier
import pandas as pd
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--inf_data", type=str, default="/home/eddie/DHDC/bluesky_posts_RecollectionApr28_cleaned.csv")
parser.add_argument("--model_ref", type=str, default="NLP-LTU/bertweet-large-sexism-detector", help="Hugging Face model reference for the pre-trained model to use.")
parser.add_argument("--model_path", type=str, default="classifier_state_dicts/bertweet-large-sexism-detector.pt", help="Path to the saved model state dict for inference.")
args = parser.parse_args()


def main():
    inf_data = args.inf_data
    df = pd.read_csv(inf_data)
    # Perform inference using the loaded DataFrame

    model = AppearanceMultiTaskClassifier(
        args.model_ref,
        num_category_labels=6,
        dropout=0.1,
        lambda_binary=1.0,
        lambda_category=1.0, # OG 1.0
        binary_loss_type="focal",#"ce",
        category_loss_type="focal",
        focal_gamma_binary=1.0,
        focal_gamma_category=2.0,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_ref)

    model.load_state_dict(torch.load(args.model_path))

    model.eval()
    with torch.no_grad():
        for index, row in df.iterrows():
            inputs = tokenizer(row["cleaned_text"], return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = model(**inputs)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            print(f"Predictions for row {index}: {preds}")

if __name__ == "__main__":
    main()
