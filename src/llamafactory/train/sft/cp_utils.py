"""Ulysses context-parallel primitives for v0 SFT trainer.

Copies of LlamaFactory v1's primitives, stripped of v1's plugin / singleton
infrastructure. Original sources:
  - SeqAllToAll4D / all_to_all_tensor:
      v1/plugins/model_plugins/parallelization/seq_comm.py
  - padding_and_split_data / sequence_parallel_loss:
      v1/plugins/model_plugins/parallelization/sequence_parallel.py

The intent is to keep v0's CP path minimal: no DeviceMesh abstraction leak,
no plugin registry, no DistributedInterface singleton. Callers pass a bare
`torch.distributed.ProcessGroup`.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Process-group accessors
# ---------------------------------------------------------------------------

_CP_GROUP: Optional[dist.ProcessGroup] = None


def set_cp_group(group: Optional[dist.ProcessGroup]) -> None:
    global _CP_GROUP
    _CP_GROUP = group


def get_cp_group() -> Optional[dist.ProcessGroup]:
    return _CP_GROUP


def get_cp_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    group = get_cp_group() if group is None else group
    return dist.get_world_size(group) if group is not None else 1


def get_cp_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    group = get_cp_group() if group is None else group
    return dist.get_rank(group) if group is not None else 0


# ---------------------------------------------------------------------------
# All-to-all primitive
# ---------------------------------------------------------------------------


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None]:
        return (
            None,
            all_to_all_tensor(grad_output[0], ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Input sharding
# ---------------------------------------------------------------------------


def padding_and_split_data(
    data: dict[str, Any],
    cp_group: dist.ProcessGroup,
    label_key: str = "labels",
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Pad rank>=2 tensors in `data` to a multiple of cp_world_size on the
    last dim, then chunk and keep this rank's shard.

    Pad values:
      - `label_key`          -> `ignore_index`
      - `loss_weights`       -> 0.0
      - everything else      -> 0
    """
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    for k, v in list(data.items()):
        if not isinstance(v, torch.Tensor) or v.ndim <= 1:
            continue
        # Global max length across CP ranks (all ranks should hold the same
        # length per-sample already, but follow v1's defensive pattern).
        data_len = torch.tensor(v.shape[-1], device=v.device, dtype=torch.int64)
        global_data_len = [torch.empty_like(data_len) for _ in range(cp_size)]
        dist.all_gather(global_data_len, data_len, group=cp_group)
        max_data_len = max(t.item() for t in global_data_len)
        pad_size = max_data_len - v.shape[-1] + (cp_size - max_data_len % cp_size) % cp_size
        if k == label_key:
            pad_value = ignore_index
        elif k == "loss_weights":
            pad_value = 0.0
        else:
            pad_value = 0
        padded = F.pad(v, (0, pad_size), value=pad_value)
        data[k] = torch.chunk(padded, chunks=cp_size, dim=-1)[cp_rank].contiguous()
    return data


# ---------------------------------------------------------------------------
# CP-aware cross-entropy
# ---------------------------------------------------------------------------


def sequence_parallel_loss_reduce(
    logits: Tensor,
    labels: Tensor,
    loss_weights: Tensor,
    cp_group: dist.ProcessGroup,
) -> Tensor:
    """CP-aware causal-LM loss.

    Each rank holds `logits` / `labels` / `loss_weights` for its seq shard
    (shapes: `(B, N_local, V)`, `(B, N_local)`, `(B, N_local)` — N_local is
    N_full / cp). We all-gather labels + loss_weights along seq dim within
    the CP group, shift by 1 for causal-LM, shard the shifted views back,
    compute per-shard log-probs, all-gather log-probs, and finally reduce.

    Returns a scalar loss.
    """
    batch_size, _ = labels.shape
    cp_world_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    logits = logits.float()

    # Gather labels (full seq = N_full) -> shift left -> re-shard per CP rank.
    global_labels = [torch.empty_like(labels) for _ in range(cp_world_size)]
    dist.all_gather(global_labels, labels, group=cp_group)
    labels_full = torch.cat(global_labels, dim=1).contiguous()
    shift_labels = labels_full[..., 1:].reshape(-1).contiguous()
    shift_labels = F.pad(shift_labels, (0, 1), value=-100)
    shift_labels = torch.chunk(shift_labels, chunks=cp_world_size, dim=-1)[cp_rank].contiguous()

    # Same for loss_weights.
    global_loss_weights = [torch.empty_like(loss_weights) for _ in range(cp_world_size)]
    dist.all_gather(global_loss_weights, loss_weights, group=cp_group)
    shift_loss_weights = torch.cat(global_loss_weights, dim=1).contiguous()
    shift_loss_weights = shift_loss_weights[..., 1:].contiguous()

    shift_logits = logits.view(shift_labels.size(0), -1).contiguous()

    # log_probs for this rank's seq shard; all-gather across CP to get full seq.
    log_probs_local = -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)
    global_log_probs = dist.nn.all_gather(log_probs_local, group=cp_group)
    log_probs_full = torch.cat(global_log_probs, dim=1).contiguous()
    # Drop the trailing position we padded above with -100.
    log_probs_full = log_probs_full[..., :-1].contiguous()

    loss = (-log_probs_full * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)
    return loss
