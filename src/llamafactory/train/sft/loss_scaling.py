"""Shared, framework-agnostic primitives for GA- and packing-invariant loss
scaling (per-token and per-sample).

The same helpers are used by:
  - the 1-attention-layer toy harness in tests (Phase 1 validation), and
  - ``CustomSeq2SeqTrainer`` for the real Gemma-4 / CP path (Phase 2).

Design (see plan): every objective is a weighted per-token sum
``L = (1/Z) * sum_t w_t * ce_t`` where only the per-token weight ``w_t`` and the
global normalizer ``Z`` change:

  - per_token : w_t = 1            , Z = total valid (shifted) tokens
  - per_sample: w_t = 1 / v_{s(t)} , Z = total number of samples

``Z`` is the ``num_items_in_batch`` that HuggingFace accumulates across the whole
gradient-accumulation (GA) window and (with our override) the data-parallel
group. Because each micro-batch returns ``(sum_t w_t ce_t) / Z`` and HF skips the
``/GA`` division when ``num_items_in_batch`` is set, summing the micro-batch
losses reconstructs ``L`` exactly -> GA-invariant. ``Z`` counts real tokens /
samples regardless of packing -> packing-invariant.

Only depends on ``torch``.
"""

from __future__ import annotations

from typing import Optional

import torch


IGNORE_INDEX = -100
PER_TOKEN = "per_token"
PER_SAMPLE = "per_sample"


def _cu_seqlens_to_pairs(cu_seqlens) -> list[tuple[int, int]]:
    cu = [int(x) for x in cu_seqlens]
    return list(zip(cu[:-1], cu[1:]))


def per_sample_shift_labels(
    labels: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Causal left-shift that respects sample boundaries.

    - ``cu_seqlens is None``: every row of ``labels`` (B, N) is one sample; this
      is the ordinary global left-shift (last column -> ignore).
    - ``cu_seqlens`` given (packing, B==1): shift within each ``[start, end)``
      segment and set each segment's last position to ``ignore_index`` so the
      end of sample A never predicts the first token of sample B.

    Returns a tensor shaped like ``labels``.
    """
    if cu_seqlens is None:
        out = torch.full_like(labels, ignore_index)
        out[:, :-1] = labels[:, 1:]
        return out

    if labels.dim() != 2 or labels.shape[0] != 1:
        raise ValueError("packed per-sample shift expects labels of shape (1, N)")
    flat = labels.reshape(-1)
    out = torch.full_like(flat, ignore_index)
    for start, end in _cu_seqlens_to_pairs(cu_seqlens):
        if end - start >= 2:
            out[start : end - 1] = flat[start + 1 : end]
        # out[end - 1] stays ignore_index (the boundary token has no in-sample target)
    return out.view_as(labels)


def loss_weights_from_shift(
    shift_labels: torch.Tensor,
    loss_reduction: str,
    cu_seqlens: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Per-token weights ``w_t`` (same shape as ``shift_labels``, float).

    per_token  -> 1.0 on valid (non-ignore) positions, 0.0 elsewhere.
    per_sample -> 1/v_s on valid positions of sample ``s`` (v_s = its valid count),
                  so summing within a sample gives that sample's mean CE.
    """
    valid = (shift_labels != ignore_index).to(torch.float32)
    if loss_reduction == PER_TOKEN:
        return valid
    if loss_reduction != PER_SAMPLE:
        raise ValueError(f"unknown loss_reduction: {loss_reduction!r}")

    if cu_seqlens is None:
        v = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return valid / v

    flat_valid = valid.reshape(-1)
    flat_w = torch.zeros_like(flat_valid)
    for start, end in _cu_seqlens_to_pairs(cu_seqlens):
        seg = flat_valid[start:end]
        vs = seg.sum().clamp_min(1.0)
        flat_w[start:end] = seg / vs
    return flat_w.view_as(valid)


def count_num_items(
    labels: torch.Tensor,
    loss_reduction: str,
    cu_seqlens: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Local ``num_items`` contribution for one micro-batch, counted over the
    per-sample *shifted* labels so numerator and denominator share a token set.

    per_token  -> number of valid shifted tokens.
    per_sample -> number of samples that contain >= 1 valid shifted token.
    Returns a 0-dim float tensor on ``labels.device``.
    """
    shift = per_sample_shift_labels(labels, cu_seqlens, ignore_index)
    valid = shift != ignore_index
    if loss_reduction == PER_TOKEN:
        return valid.sum().to(torch.float32)
    if loss_reduction != PER_SAMPLE:
        raise ValueError(f"unknown loss_reduction: {loss_reduction!r}")

    if cu_seqlens is None:
        return valid.any(dim=1).sum().to(torch.float32)
    flat_valid = valid.reshape(-1)
    n = 0
    for start, end in _cu_seqlens_to_pairs(cu_seqlens):
        if bool(flat_valid[start:end].any()):
            n += 1
    return torch.tensor(float(n), device=labels.device)


def weighted_token_ce_sum(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    weights: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """``sum_t w_t * ce_t`` over local tokens (no normalization), chunked along
    the token dim so fp32 peak is ``chunk_size * V * 4`` bytes. Differentiable.

    ``logits`` (B, N, V) aligned with ``shift_labels`` / ``weights`` (B, N).
    """
    vocab = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab)
    labels_flat = shift_labels.reshape(-1)
    w_flat = weights.reshape(-1)
    n = labels_flat.numel()
    total = logits.new_zeros(())
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ce = torch.nn.functional.cross_entropy(
            logits_flat[start:end].float(),
            labels_flat[start:end],
            ignore_index=ignore_index,
            reduction="none",
        )
        total = total + (ce * w_flat[start:end]).sum()
    return total


def token_ce_sums(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    weights: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One chunked CE pass returning ``(weighted_sum, ce_sum, valid_count)``:

    - ``weighted_sum = sum_t w_t * ce_t`` (differentiable; drives the loss)
    - ``ce_sum       = sum_t ce_t`` over valid tokens (for token-level global_ce)
    - ``valid_count`` = number of valid (non-ignore) tokens

    ``ce_sum`` / ``valid_count`` are detached (logging only).
    """
    vocab = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab)
    labels_flat = shift_labels.reshape(-1)
    w_flat = weights.reshape(-1)
    n = labels_flat.numel()
    weighted = logits.new_zeros(())
    ce_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ce = torch.nn.functional.cross_entropy(
            logits_flat[start:end].float(),
            labels_flat[start:end],
            ignore_index=ignore_index,
            reduction="none",
        )
        weighted = weighted + (ce * w_flat[start:end]).sum()
        ce_sum = ce_sum + ce.detach().sum()
    valid = (labels_flat != ignore_index).sum().to(torch.float32)
    return weighted, ce_sum, valid


def weighted_token_ce(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    weights: torch.Tensor,
    num_items: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """``(sum_t w_t * ce_t) / num_items`` with per-token CE (reduction='none').

    ``logits`` (B, N, V) are aligned with ``shift_labels`` (B, N) and ``weights``
    (B, N). The division by the *global* ``num_items`` makes the per-microbatch
    loss sum to the exact objective across the GA window.
    """
    weighted_sum = weighted_token_ce_sum(logits, shift_labels, weights, ignore_index, chunk_size=logits.shape[1] or 1)
    denom = num_items
    if not torch.is_tensor(denom):
        denom = torch.tensor(float(denom), device=logits.device)
    return weighted_sum / denom.to(weighted_sum.device).clamp_min(1.0)
