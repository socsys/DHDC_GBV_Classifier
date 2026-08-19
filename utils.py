import torch
from typing import Any, Dict, List
import torch.distributed as dist


def move_batch_to_device(batch, device: str = "cuda", include_labels: bool=True) -> Dict[str, Any]:
    tensor_keys = (["input_ids", "attention_mask", "binary_labels", "category_labels"] if include_labels else ["input_ids", "attention_mask"])

    return {key: batch[i].to(device, non_blocking=True) for i, key in enumerate(tensor_keys)}

def prob_to_label(probs, threshold=0.5):
    return (probs >= threshold).long()

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def gather_objects(local_object: Any) -> List[Any]:
    if not is_distributed():
        return [local_object]
    gathered: List[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_object)
    return gathered

def tensor_to_number(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, str):
        x = x.replace("tensor(", "").replace(")", "").strip()
        try:
            return int(x)
        except ValueError:
            return x