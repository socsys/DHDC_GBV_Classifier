import torch
from collections import defaultdict
from typing import Any, Dict, List
import torch.distributed as dist


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
