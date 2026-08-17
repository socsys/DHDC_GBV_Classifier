import torch 
import torch.nn as nn
import numpy as np 
from huggingface_hub import PyTorchModelHubMixin
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime.quantization.quant_utils as quant_utils
import onnxruntime 
import onnx
from onnx import shape_inference
from onnxruntime.quantization.shape_inference import quant_pre_process
import os 
from metrics import * 
import json
import shutil
from pathlib import Path


# ---------------------------------
# Prep for use by gbv-d-toxify
# ---------------------------------

class GBVWrapper(nn.Module, PyTorchModelHubMixin):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.architectures = [
            "RobertaForSequenceClassification"
        ]

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        binary_logits = outputs["binary_logits"]
        category_logits = outputs["category_logits"]

        is_gbv = torch.argmax(binary_logits, dim=-1) == 1

        batch_size, num_labels = category_logits.shape
        ones = torch.ones(batch_size, 1, dtype=category_logits.dtype, device=category_logits.device)
        neg_ones = -torch.ones(batch_size, num_labels - 1, dtype=category_logits.dtype, device=category_logits.device)
        non_gbv_logits = torch.cat([ones, neg_ones], dim=1)

        final_logits = torch.where(is_gbv.unsqueeze(-1), category_logits, non_gbv_logits)

        return final_logits

def _evaluate_wrapped_model(model, tokenizer, dataset, device="cuda"):
    encodings = tokenizer(list(dataset["text"]), padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    labels = torch.tensor(dataset["category_labels"].tolist())
    dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        predictions = []
        for item in dataloader:
            outputs = model(input_ids=item[0].to(device), attention_mask=item[1].to(device))
            predictions.extend(outputs)

        gold_labels = labels.tolist()

        for i, prediction in enumerate(predictions):
            prediction = torch.sigmoid(prediction).detach().cpu()
            prediction = [1 if p >= 0.5 else 0 for p in prediction]
            predictions[i] = prediction

        print(f"Gold labels: {gold_labels[:10]}")
        print(f"Predictions: {predictions[:10]}")

        icm_score = calculate_icm(gold_labels, predictions)


def _save_model_for_extension(model, tokenizer, save_path, device="cuda"):
    dummy_ids = torch.randint(0, 1000, (1, 256), dtype=torch.long)
    dummy_mask = torch.ones((1, 256), dtype=torch.long)
    torch.onnx.export(model, (dummy_ids.to(device), dummy_mask.to(device)), f"{save_path}/gbv_model.onnx", input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}, "logits": {0: "batch_size"}}, opset_version=17, external_data=False, do_constant_folding=False, dynamo=False)

    #quant_utils.load_model_with_shape_infer = lambda model_path: onnx.load(f"{save_path}/gbv_model.onnx")
    #inferred = shape_inference.infer_shapes_path(f"{save_path}/gbv_model.onnx", f"{save_path}/gbv_model_inferred.onnx")

    quant_pre_process(
    input_model_path=f"{save_path}/gbv_model.onnx",
    output_model_path=f"{save_path}/gbv_model_preprocessed.onnx",
    auto_merge=True,
    skip_symbolic_shape=False,
    verbose=3,
    )

    quantize_dynamic(
        model_input=f"{save_path}/gbv_model_preprocessed.onnx",
        model_output=f"{save_path}/gbv_model_quant_temp.onnx",
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        weight_type=QuantType.QUInt8,
        per_channel=True
    )

    if not os.path.exists(f"{save_path}/onnx/"):
        os.makedirs(f"{save_path}/onnx/")

    model = onnx.load(f"{save_path}/gbv_model_quant_temp.onnx")
    model.ir_version = 8
    onnx.save_model(model, f"{save_path}/onnx/model_quantized.onnx")

    os.remove(f"{save_path}/gbv_model_quant_temp.onnx")


def _evaluate_exported_model(save_path, tokenizer, wrapped_model=None, device="cuda"):
    original = onnxruntime.InferenceSession(f"{save_path}/gbv_model_preprocessed.onnx")
    quantized = onnxruntime.InferenceSession(f"{save_path}/onnx/model_quantized.onnx")

    dummy_input = tokenizer(["This is a test input.", "She is a silly bitch!", "I'd love to take her out to dinner", "What a twat.", "This is a sentence.", "I want to kill her."], return_tensors="pt", max_length=256, padding="max_length", truncation=True)
    input_ids = dummy_input["input_ids"].numpy()
    attention_mask = dummy_input["attention_mask"].numpy()

    # create output from wrapped model
    dataset = torch.utils.data.TensorDataset(dummy_input["input_ids"], dummy_input["attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        wrapped_model_out = []
        for item in dataloader:
            outputs = wrapped_model(input_ids=item[0].to(device), attention_mask=item[1].to(device))
            wrapped_model_out.extend(outputs)

    orig_out = original.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    quant_out = quantized.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    for i in range(len(orig_out[0])):
        print(f"Wrapped model logits[{i}]:", wrapped_model_out[i])
        print(f"original logits[{i}]:", orig_out[0][i])
        print(f"quantized logits[{i}]:", quant_out[0][i])
        print(f"max abs diff[{i}]:", np.abs(orig_out[0][i] - quant_out[0][i]).max())

    os.remove(f"{save_path}/gbv_model_preprocessed.onnx")

def export_model(model, tokenizer, save_path, test_dataset=None):
    wrapped_model = GBVWrapper(model)
    wrapped_model.eval()

    # custom config 
    config = {
        "model_name": "gbv-detector",
        "num_labels": 6,
        "hidden_size": 1024,
        "id2label": {
            "0": "-",
            "1": "IDEOLOGICAL-INEQUALITY",
            "2": "STEREOTYPING-DOMINANCE",
            "3": "OBJECTIFICATION",
            "4": "SEXUAL-VIOLENCE",
            "5": "MISOGYNY-NON-SEXUAL-VIOLENCE"
        },
        "label2id": {
            "-": 0,
            "IDEOLOGICAL-INEQUALITY": 1,
            "STEREOTYPING-DOMINANCE": 2,
            "OBJECTIFICATION": 3,
            "SEXUAL-VIOLENCE": 4,
            "MISOGYNY-NON-SEXUAL-VIOLENCE": 5
        },
        "architectures": [
            "RobertaForSequenceClassification"
        ],
        "problem_type": "multi_label_classification",
        "model_type": "roberta",
        "max_position_embeddings": 514,
        "dtype": "float32"
        }

    wrapped_model.save_pretrained(save_path, config=config)
    tokenizer.save_pretrained(save_path)


    if test_dataset is not None:
        print("Evaluating wrapped model...")
        _evaluate_wrapped_model(wrapped_model, tokenizer, test_dataset)
    
    _save_model_for_extension(wrapped_model, tokenizer, save_path)
    print(f"Model exported to {save_path} for use in gbv-d-toxify.")

    _evaluate_exported_model(save_path, tokenizer, wrapped_model)

    model = onnx.load(f"{save_path}/onnx/model_quantized.onnx")

    for opset_import in model.opset_import:
        if opset_import.domain == 'ai.onnx.ml':
            opset_import.version = 3

    onnx.save(model, f"{save_path}/onnx/model_quantized.onnx")

    # check if model opset version has successfully changed
    model = onnx.load(f"{save_path}/onnx/model_quantized.onnx")
    for opset_import in model.opset_import:
        if opset_import.domain == 'ai.onnx.ml':
            print(f"Updated opset version for ai.onnx.ml: {opset_import.version}")



def normalize_merges(tokenizer_path: str, output_path: str = None, backup: bool = True):
    """
    Normalize the `merges` field in a tokenizer.json file.

    Newer tokenizer.json files store merges as pairs, e.g. ["t", "h"].
    Older consumers (like some transformers.js BPE implementations)
    expect merges as space-joined strings, e.g. "t h".

    Converts pair-format merges to string format; leaves string-format
    merges untouched. Only writes/backs up if a change is actually made.
    """
    tokenizer_path = Path(tokenizer_path)
    output_path = Path(output_path) if output_path else tokenizer_path

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", {})
    merges = model.get("merges")

    if merges is None:
        print("No 'merges' field found under data['model']['merges']. Nothing to do.")
        return

    if not merges:
        print("'merges' is empty. Nothing to do.")
        return

    sample = merges[0]

    if isinstance(sample, str):
        print("Merges are already in string format. No changes needed.")
        return

    if not isinstance(sample, list):
        raise TypeError(f"Unrecognized merge entry type: {type(sample)}")

    normalized = []
    for i, pair in enumerate(merges):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                f"Unexpected merge entry at index {i}: {pair!r} "
                f"(expected a 2-element list)"
            )
        normalized.append(" ".join(pair))

    # Only back up right before we actually overwrite the original file
    if backup and output_path == tokenizer_path:
        backup_path = tokenizer_path.with_suffix(tokenizer_path.suffix + ".bak")
        shutil.copy2(tokenizer_path, backup_path)
        print(f"Backup written to {backup_path}")

    model["merges"] = normalized
    data["model"] = model

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(normalized)} merges from pair format to string format.")
    print(f"Written to {output_path}")


