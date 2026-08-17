from metrics import ICMCalculator, get_f1_score, calculate_icm
from export import export_model, normalize_merges
from gbv_multilabel_classifier import *
from handle_data import * 
from evaluate import *

parser = ArgumentParser()
parser.add_argument("--train", action="store_true", help="Whether to run training.")
parser.add_argument("--train_data_name", type=str, default="EXIST", help="Name of the training dataset. Options: EXIST")
parser.add_argument("--train_data_path", type=str, nargs="?", default="../DHDC/data/EXIST 2025 Tweets Dataset/training/EXIST2025_training.json", help="Path to the training dataset.") # Remove default for final sharing
parser.add_argument("--val_data_path", type=str, nargs = "?", help="Path to the validation dataset.", default="../DHDC/data/EXIST 2025 Tweets Dataset/dev/EXIST2025_dev.json") # Remove default for final sharing 
parser.add_argument("--infr", action="store_true", help="Whether to run inference with the trained model.")
parser.add_argument("--infr_data_path", type=str, nargs="?", help="Path to the dataset for inference.")
parser.add_argument("--resume_infr", action="store_true", help="Whether to resume inference.")
#parser.add_argument("--save", action="store_true", help="Whether to save the trained model.")
parser.add_argument("--export", action="store_true", help="Whether to export the trained model to a format suitable for the DToxify Extension")
parser.add_argument("--tokenizer_path", help="Path to tokenizer.json")
parser.add_argument(
    "-o", "--output", default=None,
    help="Output path (defaults to overwriting input, with a .bak backup)"
)
parser.add_argument(
    "--no-backup", action="store_true",
    help="Skip creating a .bak backup when overwriting in place"
    )
args = parser.parse_args()

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

def inference(model, tokenizer, inf_data, resume=False, device="cuda", infr_data_path=None):
    print("Starting inference...")
    if infr_data_path is None:
        raise ValueError("infr_data_path must be provided for inference.")
    model.eval()
    

    autocast_enabled = device == "cuda"
    batch_size = 32
    save_steps = 1000
    csv_path = f"../dhdc_stats/processed_data/{infr_data_path.split('/')[-1].split('.')[-2]}_inference_predictions_temp.csv"

    print(f"Length of inference data: {len(inf_data)}")
    inf_data.dropna(subset=["text"], inplace=True)
    inf_data.reset_index(drop=True, inplace=True)

    inf_data_loader = create_data_loader(inf_data, tokenizer, batch_size=batch_size, infr=True)
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
                        "data_id": data_id,
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
    #gathered_records = gather_objects(local_records)

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

def main(label2id_dict, train_flag=False, train_data_name = None, train_data_path=None, val_data_path=None, model_ref="NLP-LTU/bertweet-large-sexism-detector", device="cuda"):
    print(f"Using model reference: {model_ref}")

    if args.train and args.infr:
        print("To confirm, model is being trained and then used for inference.")

    seed = 51
    set_seed(seed)

    num_category_labels = len(label2id_dict["level_2"])

    focal_gamma_category = 2 
    lr = 5e-6 
    weight_decay = 0.01 

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
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    if not os.path.exists("classifier_state_dicts/"):
        os.makedirs("classifier_state_dicts/")
    state_dict_path = f"classifier_state_dicts/{model_ref.split('/')[-1]}_{train_data_name}_final_{seed}.pt"
    save_path = "gbv-model-final"

    # print(vars(model))

    # Optimizer study
    #optuna_study = Optimizer(model, tokenizer, train_data, device=device, model_ref=model_ref, num_category_labels=num_category_labels).run_search(n_trials=20)
    #print(f"Best hyperparameters from Optuna study: {optuna_study.best_params}")
    #exit()


    if train_flag:
        # Load the train dataset
        train_data = CustomDataLoader(data_name=train_data_name, data_path = train_data_path, label2id_dict=label2id_dict, split="train", multilingual=True).load_processed_data(clean=True) # Model trains best on multilingual data despite being monolingual model. 
        print(train_data.head())
        #print(train_data["binary_labels"].value_counts())
        #print(train_data["category_labels"].value_counts())    

        #print(f"Length of train dataset before removing ties: {len(train_data)}")
        train_data = train_data[train_data["binary_labels"] != 99]
        #print(f"Length of train dataset after removing binary ties: {len(train_data)}")
        train_data = train_data[train_data["category_labels"] != 99]
        #print(f"Length of train dataset after removing category ties: {len(train_data)}")

        train_data, val_data = train_test_split(train_data, test_size=0.05, stratify=train_data["binary_labels"], random_state=42)

        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        trained_model = train(model, tokenizer, train_data, val_data, label2id_dict, best_save_path=f"checkpoints/{model_ref.split('/')[-1]}_{train_data_name}.pt", lr=lr, weight_decay=weight_decay, device=device)

        # save state dict for later use
        torch.save(trained_model.state_dict(), state_dict_path)
        print(f"Trained model state dict saved at {state_dict_path}")
        trained_model.to(device)

    else:
        model.load_state_dict(torch.load(state_dict_path, weights_only=True, map_location=device))
        model.to(device)
        trained_model = model

    test_data = CustomDataLoader(data_name = train_data_name, data_path = val_data_path,  split="dev", label2id_dict=label2id_dict, multilingual=False).load_processed_data(clean=False)
    #print(test_data.head())
    #print(test_data["binary_labels"].value_counts())
    #print(test_data["category_labels"].value_counts())
    test_data = test_data[test_data["binary_labels"] != 99]
    test_data = test_data[test_data["category_labels"] != 99]
    evaluation(trained_model, tokenizer, dataset=test_data, device=device)


    if args.export:
        export_model(trained_model, tokenizer, save_path=save_path, test_dataset=test_data)
        normalize_merges(f"{save_path}/tokenizer.json", args.output, backup=not args.no_backup)


    if args.infr:
        inf_data = pd.read_csv(args.infr_data_path, dtype={"comment_id": str, "item_id": str, "data_id": str})
        print(f"Length of inference dataset: {len(inf_data)}")
        data_id_col = "comment_id" if "comment_id" in inf_data.columns else "item_id" if "item_id" in inf_data.columns else "data_id" # Update to match inf dataset labeling 
        inf_data.rename(columns={"cleaned_text": "text"}, inplace=True) if "cleaned_text" in inf_data.columns else None 
        inf_data = inf_data[[data_id_col, "text"]]
        inf_data.rename(columns={data_id_col: "data_id"}, inplace=True)
        ## For testing
        #inf_data = inf_data.sample(n=1000, random_state=42).reset_index(drop=True) 
        predictions = inference(model, tokenizer, inf_data, resume=args.resume_infr, infr_data_path=args.infr_data_path)
        predictions = pd.DataFrame(predictions, columns=["data_id", "pred_binary", "pred_binary_prob", "pred_category", "pred_category_confidence"])
        print(f"Length of predictions: {len(predictions)}")

        if any(predictions["pred_binary"] == 99):
            raise ValueError("Found predictions with binary label 99, which should not occur.")
        if any(predictions["pred_category"] == 99):
            raise ValueError("Found predictions with category label 99, which should not occur.")
        if len(predictions) > len(inf_data):
            raise ValueError("Number of predictions exceeds number of input data points.")

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

        labelled_texts.to_csv(f"processed_data/{args.infr_data_path.split('.')[-2]}_predictions.csv", index=False)
    

if __name__ == "__main__":
    if args.train_data_name == "EXIST":
        label2id_dict = {"level_1": {"NO":0, "YES":1}, "level_2": {"-":0, "IDEOLOGICAL-INEQUALITY":1, "STEREOTYPING-DOMINANCE":2, "OBJECTIFICATION":3, "SEXUAL-VIOLENCE":4, "MISOGYNY-NON-SEXUAL-VIOLENCE":5}}
    else:
        raise ValueError(f"Unsupported train_data_name: {args.train_data_name}. Please provide a valid dataset name.")

    main(label2id_dict, train_flag = args.train, train_data_name=args.train_data_name, train_data_path=args.train_data_path, val_data_path=args.val_data_path, model_ref="NLP-LTU/bertweet-large-sexism-detector")