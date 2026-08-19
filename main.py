from transformers import AutoTokenizer, AutoModel, AutoConfig
from export import export_model, normalize_merges
from gbv_multilabel_classifier import GBVMultiTaskClassifier, train, Optimizer
from utils import move_batch_to_device, tensor_to_number
from handle_data import create_data_loader, CustomDataLoader
from evaluate_model import predict_labels, evaluation
from sklearn.model_selection import train_test_split
import os 
from dotenv import load_dotenv
from argparse import ArgumentParser
import torch
import pandas as pd
import numpy as np
import random
import ast
from typing import Any, Dict, List
from tqdm import tqdm
import logging

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO, filename='gbv_model.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("===== Starting GBV Multi-Task Classifier script. =====")

parser = ArgumentParser()
parser.add_argument("--train", action="store_true", help="Whether to run training.")
parser.add_argument("--train_data_name", type=str, default="EXIST", help="Name of the training dataset. Options: EXIST")
parser.add_argument("--train_data_path", type=str, nargs="?", default=os.getenv("TRAIN_DATA", None), help="Path to the training dataset.") 
parser.add_argument("--no_eval", action="store_true", help="Whether to skip evaluation during training or inference. If set, no validation will be performed.")
parser.add_argument("--val_data_path", type=str, nargs = "?", help="Path to the validation dataset.", default=os.getenv("VAL_DATA", None)) 
parser.add_argument("--infr", action="store_true", help="Whether to run inference with the trained model.")
parser.add_argument("--infr_data_path", type=str, nargs="?", help="Path to the dataset for inference.", default=os.getenv("INFR_DATA", None)) 
parser.add_argument("--resume_infr", action="store_true", help="Whether to resume inference.")
parser.add_argument("--export", action="store_true", help="Whether to export the trained model to a format suitable for the DToxify Extension")
parser.add_argument(
    "-o", "--output", default=None,
    help="Output path (defaults to overwriting input, with a .bak backup)"
)
parser.add_argument(
    "--no_backup", action="store_true",
    help="Skip creating a .bak backup when overwriting in place"
    )

def set_seed(seed: int = 42):
    ''' Sets the random seed for reproducibility. '''
    torch.use_deterministic_algorithms(True, warn_only=False)

    torch.manual_seed(seed)  # Sets seed for CPU operations
    np.random.seed(seed)     # Sets seed for NumPy
    random.seed(seed)        # Sets seed for Python's random module
    
    if torch.cuda.is_available():  # GPU-specific settings
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False     # Disables non-deterministic optimizations

# ----------------------------------
# Define saving functions 
# ----------------------------------

def save_model(model, tokenizer, save_path):

    print(model.config)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved at {save_path}")


# ----------------------------------
# Define inference function
# ----------------------------------

def inference(model, tokenizer, inf_data=None, resume=False, device="cuda", infr_data_path=None):
    print("Starting inference...")
    if infr_data_path is None:
        raise ValueError("infr_data_path must be provided for inference.")
    model.eval()
    

    autocast_enabled = device == "cuda"
    batch_size = 32
    save_steps = 1000
    csv_path = f"{os.getenv('PROCESSED_DATA_DIR')}{infr_data_path.split('/')[-1].split('.')[-2]}_inference_predictions_temp.csv"

    inf_data.dropna(subset=["text"], inplace=True)
    inf_data.reset_index(drop=True, inplace=True)
    print(f"Length of inference dataset after dropping NaNs: {len(inf_data)}")

    inf_data_loader = create_data_loader(inf_data, tokenizer, batch_size=batch_size, infr=True)
    print(f"Created inference data loader with {len(inf_data_loader)} batches.")

    if resume == True:
        print(f"Loading previous records from CSV...")
        all_records = pd.read_csv(csv_path).to_dict(orient="records")
        resume_step = len(all_records) // batch_size
        print(f"Resuming inference from step {resume_step}.")
    else:
        ## Clear old temporary files
        if os.path.exists(csv_path):
            os.remove(csv_path)
        all_records: List[Dict[str, Any]] = []

    local_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(inf_data_loader), total=len(inf_data_loader), desc="Inference"):
            if resume == True and step < resume_step:
                continue
            moved = move_batch_to_device(batch, device, include_labels=False)
            with torch.autocast(device_type=device, enabled=autocast_enabled):
                outputs = model(
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                )

            binary_pred_mask_bool, binary_probs_list, category_pred, category_conf = predict_labels(outputs)

            for index, item in enumerate(zip(batch[-1], category_pred)): # iterate over data_id field in batch, always final field
                # normalize record fields to plain Python types
                data_id, category = item

                record = {
                        "data_id": tensor_to_number(data_id),
                        "pred_binary": int(binary_pred_mask_bool[index].item()),
                        "pred_binary_prob": binary_probs_list[index],
                        "pred_category": category_pred[index],
                        "pred_category_confidence": category_conf[index],
                    }

                local_records.append(record)
                all_records.append(record)

            if step % save_steps == 0 or step == len(inf_data_loader) - 1:
                print(f"Processed first {step} steps. Saving most recent records to CSV...")
                df = pd.DataFrame(local_records, columns=["data_id", "pred_binary", "pred_binary_prob", "pred_category", "pred_category_confidence"])
                df.to_csv(csv_path, mode="a", index=False, header=False if os.path.exists(csv_path) else True)
                local_records: List[Dict[str, Any]] = []

    print(f"Length of all_records: {len(all_records)}")

    return all_records

def process_category_pred(pred_category_str, id2label_dict):
    """Convert prediction category indices to labels and pipe-separated string."""
    if isinstance(pred_category_str, str):
        index_list = ast.literal_eval(pred_category_str)
    else:
        index_list = pred_category_str
    labels = [id2label_dict[i] for i, val in enumerate(index_list) if val == 1]
    pipe_str = "|".join(labels) if labels else "-"
    return labels, pipe_str

# ----------------------------------
# Define main function
# ----------------------------------

def main():
    model_ref = "NLP-LTU/bertweet-large-sexism-detector"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()

    if args.train_data_name == "EXIST":
        label2id_dict = {"level_1": {"NO":0, "YES":1}, "level_2": {"-":0, "IDEOLOGICAL-INEQUALITY":1, "STEREOTYPING-DOMINANCE":2, "OBJECTIFICATION":3, "SEXUAL-VIOLENCE":4, "MISOGYNY-NON-SEXUAL-VIOLENCE":5}}
    else:
        raise ValueError(f"Unsupported train_data_name: {args.train_data_name}. Please provide a valid dataset name.")

    print(f"Using model reference: {model_ref}")

    ## Set random seed for reproducibility
    seed = 51
    set_seed(seed)
    logger.info(f"Random seed set to {seed} for reproducibility.")

    num_category_labels = len(label2id_dict["level_2"])

    ## Define hyperparameters for training
    focal_gamma_category = 2 
    lr = 5e-6 
    weight_decay = 0.01 
    logger.info(f"Training hyperparameters: learning rate={lr}, weight decay={weight_decay}, focal gamma for category={focal_gamma_category}")

    model = GBVMultiTaskClassifier(
        model_ref,
        num_category_labels=num_category_labels,
        dropout=0.1,
        lambda_binary=1.0,
        lambda_category=1.0, 
        binary_loss_type="focal",
        category_loss_type="focal",
        focal_gamma_binary=1.0,
        focal_gamma_category=focal_gamma_category,
    ).to(device)

    logger.info(f"Model configuration: {model._hub_mixin_config}")

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    if not os.path.exists("classifier_state_dicts/"):
        os.makedirs("classifier_state_dicts/")
    state_dict_path = f"classifier_state_dicts/{model_ref.split('/')[-1]}_{args.train_data_name}_final_{seed}.pt"
    save_path = "gbv-model-final"
    logger.info(f"State dict path: {state_dict_path}, Save path for exported model: {save_path}")

    # print(vars(model))

    # Optimizer study
    #optuna_study = Optimizer(model, tokenizer, train_data, device=device, model_ref=model_ref, label2id_dict=label2id_dict).run_search(n_trials=20)
    #print(f"Best hyperparameters from Optuna study: {optuna_study.best_params}")
    #exit()

    ## Load validation data for evaluation or export evaluation
    if args.export or not args.no_eval:
        test_data = CustomDataLoader(data_name=args.train_data_name, data_path=args.val_data_path, split="test", label2id_dict=label2id_dict, multilingual=False).load_processed_data(clean=False)


    if args.train:
        logger.info(f"Training flag is set. Starting training with dataset: {args.train_data_name} from path: {args.train_data_path}")
        # Load the train dataset
        train_data = CustomDataLoader(data_name=args.train_data_name, data_path=args.train_data_path, label2id_dict=label2id_dict, split="train", multilingual=True).load_processed_data(clean=True) # Model trains best on multilingual data despite being monolingual model. 
        print(train_data.head())
        #print(train_data["binary_labels"].value_counts())
        #print(train_data["category_labels"].value_counts())    

        train_data, val_data = train_test_split(train_data, test_size=0.05, stratify=train_data["binary_labels"], random_state=42)

        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        trained_model = train(model, tokenizer, train_data, val_data, label2id_dict, best_save_path=f"checkpoints/{model_ref.split('/')[-1]}_{args.train_data_name}.pt", lr=lr, weight_decay=weight_decay, device=device)

        # save state dict for later use
        torch.save(trained_model.state_dict(), state_dict_path)
        print(f"Trained model state dict saved at {state_dict_path}")
        trained_model.to(device)
        logger.info(f"Trained model saved at {state_dict_path}.")

    else:
        model.load_state_dict(torch.load(state_dict_path, weights_only=True, map_location=device))
        model.to(device)
        trained_model = model
        logger.info(f"Loaded model state dict from {state_dict_path}.")

    if not args.no_eval:
        #print(test_data.head())
        #print(test_data["binary_labels"].value_counts())
        #print(test_data["category_labels"].value_counts())
        loss, f1_score, predicted_icm = evaluation(trained_model, tokenizer, dataset=test_data, device=device)
        logger.info(f"Evaluation completed.")
        logger.info(f"Loss: {loss['loss']:.4f}, Binary Loss: {loss['loss_binary']:.4f}, Category Loss: {loss['loss_category']:.4f}")
        logger.info(f"Binary F1 Score: {f1_score['binary']:.4f}, Category F1 Score: {f1_score['category']}, Category Macro F1 Score: {f1_score['category_macro']:.4f}")
        logger.info(f"Predicted ICM: {predicted_icm:.4f}")

    if args.export:
        logger.info(f"Export flag is set. Exporting model and tokenizer to {save_path}.")
        export_model(trained_model, tokenizer, save_path=save_path, test_dataset=test_data)
        normalize_merges(f"{save_path}/tokenizer.json", args.output, backup=not args.no_backup)
        logger.info(f"Exported model and tokenizer to {save_path}")

    if args.infr:
        logger.info(f"Inference flag is set. Starting inference with dataset from path: {args.infr_data_path}")
        inf_data = pd.read_csv(args.infr_data_path, dtype={"comment_id": str, "item_id": str, "data_id": str})
        print(f"Length of inference dataset: {len(inf_data)}")
        #inf_data = inf_data.sample(n=1000, random_state=42).reset_index(drop=True) ## Sample for testing 
        data_id_col = "comment_id" if "comment_id" in inf_data.columns else "item_id" if "item_id" in inf_data.columns else "data_id" # Update to match inf dataset labeling 
        inf_data.rename(columns={"cleaned_text": "text"}, inplace=True) if "cleaned_text" in inf_data.columns else None 
        inf_data.rename(columns={data_id_col: "item_id"}, inplace=True)

        try:
            inf_data["data_id"] = inf_data["item_id"].astype(int) # should be int for TensorDataset
        except Exception as e:
            print(f"Error converting item_id to int: {e}. Generating new unique data_id values.")
            new_ids = np.random.choice(range(1, 10**9), size=len(inf_data), replace=False)
            inf_data["data_id"] = new_ids

        ## Temporary id suitable for torch dataset
        #inf_data.reset_index(inplace=True, names=["data_id"])
        
        inf_data = inf_data[["item_id", "data_id", "text"]]

        predictions = inference(model, tokenizer, inf_data, resume=args.resume_infr, infr_data_path=args.infr_data_path)
        logger.info(f"Inference completed. Predictions generated for {len(predictions)} data points.")
        predictions = pd.DataFrame(predictions, columns=["data_id", "pred_binary", "pred_binary_prob", "pred_category", "pred_category_confidence"])
        predictions.drop_duplicates(subset=["data_id"], keep="first", inplace=True) # Handle any accidental duplication from resuming inference
        print(f"Length of predictions: {len(predictions)}")

        if any(predictions["pred_binary"] == 99):
            raise ValueError("Found predictions with binary label 99, which should not occur.")
        if any(predictions["pred_category"] == 99):
            raise ValueError("Found predictions with category label 99, which should not occur.")
        if len(predictions) > len(inf_data):
            raise ValueError(f"Number of predictions exceeds number of input data points. Length predictions = {len(predictions)}, Length input data = {len(inf_data)}")

        print(f"Predictions data_id dtype: {predictions['data_id'].dtype}, Inference data data_id dtype: {inf_data['data_id'].dtype}")

        labelled_texts = pd.merge(inf_data, pd.DataFrame(predictions), on="data_id")
        print(labelled_texts.head())

        for item in predictions["pred_category"]:
            if item[0] == 1 and any(x == 1 for x in item[1:]):
                print(f"Found instance where non-GBV category is predicted as 1 along with other categories: {item}")
        #print(predictions["pred_category"][:10])

        sample = labelled_texts[["text", "pred_binary"]].groupby("pred_binary").sample(n=10)
        print(sample)
        
        # Process predictions and create category labels/pipes directly
        id2label_dict = {v: k for k, v in label2id_dict["level_2"].items()}
        
        labelled_texts[["pred_category_labels", "pred_category_pipe"]] = labelled_texts["pred_category"].apply(
            lambda x: pd.Series(process_category_pred(x, id2label_dict))
        )

        print(f"Length of labelled_texts: {len(labelled_texts)}")
        print(labelled_texts.head())

        ## Remove temporary id used for torch dataset
        labelled_texts.drop("data_id", axis=1, inplace=True)
        labelled_texts.rename(columns={"item_id":"data_id"}, inplace=True)
        labelled_texts.to_csv(f"{os.getenv('PROCESSED_DATA_DIR')}{args.infr_data_path.split('/')[-1].split('.')[-2]}_predictions.csv", index=False)
        logger.info(f"Inference completed. Predictions saved to {os.getenv('PROCESSED_DATA_DIR')}{args.infr_data_path.split('/')[-1].split('.')[-2]}_predictions.csv")


if __name__ == "__main__":
    main()