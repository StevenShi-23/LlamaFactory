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


_CP_DBG_COUNTER = {"a2a": 0, "pad": 0}


def _cp_dbg(tag: str, msg: str = "") -> None:
    """Rank-0 debug print, gated on env CP_DEBUG=1. Counter-based so we can
    see exactly how far through the first forward we got before any hang."""
    import os

    if os.environ.get("CP_DEBUG", "0") != "1":
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print(f"[CP_DBG][rank0] {tag} {msg}", flush=True)


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    seq_world_size = dist.get_world_size(group)
    _CP_DBG_COUNTER["a2a"] += 1
    n = _CP_DBG_COUNTER["a2a"]
    if n <= 6 or n % 60 == 0:
        _cp_dbg(
            f"a2a#{n:3d}",
            f"shape={tuple(local_input.shape)} scatter={scatter_dim} gather={gather_dim} ws={seq_world_size}",
        )
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    if n <= 6:
        _cp_dbg(f"a2a#{n:3d}", "calling dist.all_to_all...")
    dist.all_to_all(output_list, input_list, group=group)
    if n <= 6:
        _cp_dbg(f"a2a#{n:3d}", "dist.all_to_all returned")
    out = torch.cat(output_list, dim=gather_dim).contiguous()
    if n <= 6:
        _cp_dbg(f"a2a#{n:3d}", f"done out_shape={tuple(out.shape)}")
    return out


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
    _CP_DBG_COUNTER["pad"] += 1
    pn = _CP_DBG_COUNTER["pad"]
    if pn <= 3:
        _cp_dbg(f"pad#{pn}", f"keys={list(data.keys())} cp_size={cp_size}")
    for k, v in list(data.items()):
        if not isinstance(v, torch.Tensor) or v.ndim <= 1:
            continue
        # Global max length across CP ranks (all ranks should hold the same
        # length per-sample already, but follow v1's defensive pattern).
        data_len = torch.tensor(v.shape[-1], device=v.device, dtype=torch.int64)
        global_data_len = [torch.empty_like(data_len) for _ in range(cp_size)]
        if pn <= 3:
            _cp_dbg(f"pad#{pn}", f"all_gather len for key={k!r} shape={tuple(v.shape)}...")
        dist.all_gather(global_data_len, data_len, group=cp_group)
        if pn <= 3:
            _cp_dbg(f"pad#{pn}", f"all_gather returned for key={k!r}")
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
    ce_chunk_size: int = 2048,
) -> Tensor:
    """CP-aware causal-LM loss — local CE + scalar all-reduce.

    Key design: each CP rank computes its own CE contribution locally, then
    we all-reduce only TWO scalars (loss_sum, weight_sum). This avoids:
      - The 64 GiB fp32 logits materialisation (was causing OOM at cp=4 256K)
      - The N_full × B all-gather of log-probs (also large at 256K)

    The CE is chunked along the token dim so fp32 memory peaks at only
    ce_chunk_size × V × 4 bytes per chunk (2048 × 262144 × 4 ≈ 2 GiB).

    Correctness: rank r holds logits for sequence positions [r*N_local :
    (r+1)*N_local]. After the causal-LM shift, each rank's logits predict the
    NEXT token within its slice. Summing local losses across CP ranks gives
    exactly the full-sequence CE, so the scalar all-reduce is mathematically
    equivalent to gathering all log-probs and summing globally.
    """
    batch_size, _ = labels.shape
    cp_world_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    # Gather labels across CP to get the full-sequence label tensor, then
    # shift left by 1 for causal LM and re-shard back to this rank's slice.
    global_labels = [torch.empty_like(labels) for _ in range(cp_world_size)]
    dist.all_gather(global_labels, labels, group=cp_group)
    labels_full = torch.cat(global_labels, dim=1).contiguous()
    shift_labels_full = labels_full[..., 1:].reshape(-1).contiguous()
    shift_labels_full = F.pad(shift_labels_full, (0, 1), value=-100)
    shift_labels = torch.chunk(shift_labels_full, chunks=cp_world_size, dim=-1)[cp_rank].contiguous()
    del labels_full, shift_labels_full

    # Same gather+shift for loss_weights.
    global_loss_weights = [torch.empty_like(loss_weights) for _ in range(cp_world_size)]
    dist.all_gather(global_loss_weights, loss_weights, group=cp_group)
    shift_loss_weights_full = torch.cat(global_loss_weights, dim=1).contiguous()
    shift_loss_weights = shift_loss_weights_full[..., 1:].contiguous()
    del shift_loss_weights_full

    # Chunked local CE: process ce_chunk_size tokens at a time to keep peak
    # fp32 memory at ce_chunk_size × V × 4 B (≈ 2 GiB at chunk=2048, V=262K).
    n_tokens = shift_labels.size(0)  # B × N_local
    logits_flat = logits.view(n_tokens, -1)  # (N_local*B, V) bf16 — no copy
    shift_weights_flat = shift_loss_weights.view(-1)

    local_loss_sum = torch.tensor(0.0, dtype=torch.float64, device=logits.device)
    local_weight_sum = torch.tensor(0.0, dtype=torch.float64, device=logits.device)

    for start in range(0, n_tokens, ce_chunk_size):
        end = min(start + ce_chunk_size, n_tokens)
        chunk_fp32 = logits_flat[start:end].float()
        chunk_labels = shift_labels[start:end]
        chunk_weights = shift_weights_flat[start:end].float()
        log_probs_chunk = -F.cross_entropy(chunk_fp32, chunk_labels, reduction="none")
        local_loss_sum += (log_probs_chunk * chunk_weights).sum().double()
        local_weight_sum += chunk_weights.sum().double()
        del chunk_fp32, log_probs_chunk

    # All-reduce the two scalars across the CP group.
    # Using float64 to preserve precision across cp=8 ranks.
    stats = torch.stack([local_loss_sum, local_weight_sum])
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=cp_group)
    total_loss_sum, total_weight_sum = stats[0], stats[1]

    loss = (-total_loss_sum / (total_weight_sum + 1e-6)).float()
    return loss
