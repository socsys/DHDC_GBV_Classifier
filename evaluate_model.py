from handle_data import create_data_loader
from metrics import get_f1_score, calculate_icm
from utils import prob_to_label, move_batch_to_device, gather_objects
from typing import Any, Dict, List
import torch
import sklearn.metrics

def predict_labels(outputs):
    binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)[:, 1].detach().cpu()
    binary_pred_mask_bool = (binary_probs >= 0.5)  # 1D bool tensor
    category_probs = torch.sigmoid(outputs["category_logits"]).detach().cpu()
    
    binary_probs_list = [float(x) for x in binary_probs.tolist()]
    category_pred_tensor = prob_to_label(category_probs)
    category_pred = [list(map(int, row)) for row in category_pred_tensor.tolist()]
    
    # enforce non-GBV -> category 0
    category_pred = [pred if binary_pred_mask_bool[i].item() == True else [1] + [0]*(len(pred)-1) for i, pred in enumerate(category_pred)]
    category_conf = [list(map(float, row)) for row in category_probs.tolist()]

    return binary_pred_mask_bool, binary_probs_list, category_pred, category_conf

def _evaluate_model(model, loader, device) -> Dict[str, Any]:
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
                data_id = item[0]
                true_binary = int(item[1].argmax().item())
                true_category = [int(x) for x in item[2].cpu().tolist()]

                local_records.append(
                    {
                        "data_id": data_id,
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


def evaluation(model, tokenizer, dataset=None, device="cuda"):
    model.eval()
    loader = create_data_loader(dataset, tokenizer, batch_size=16)
    evaluation = _evaluate_model(model, loader, device)
    print(f"Evaluation results: Loss: {evaluation['loss']:.4f}, Binary Loss: {evaluation['loss_binary']:.4f}, Category Loss: {evaluation['loss_category']:.4f}")
    f1_score = get_f1_score(evaluation["records"])
    print(f"Binary F1 Score: {f1_score['binary']:.4f}, Category F1 Score: {f1_score['category']}, Category Macro F1 Score: {f1_score['category_macro']:.4f}")

    gold_labels = [record["true_category"] for record in evaluation["records"]]
    pred_labels = [record["pred_category"] for record in evaluation["records"]]

    print("Calculating ICM for EXIST dataset...")
    predicted_icm = calculate_icm(gold_labels, pred_labels)
    return evaluation, f1_score, predicted_icm
