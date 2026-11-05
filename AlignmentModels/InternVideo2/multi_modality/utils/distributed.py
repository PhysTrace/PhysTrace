# Trimmed from the upstream InternVideo2 utils/distributed.py: only criterions.py's
# `from utils.distributed import get_rank, get_world_size` is ever imported in this
# single-process inference path, and neither is actually called (both live inside
# loss-forward code paths this scorer never exercises). The distributed-training-only
# helpers (init_distributed_mode, GatherLayer, gather_tensor_along_batch*, ...) were dropped.
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
