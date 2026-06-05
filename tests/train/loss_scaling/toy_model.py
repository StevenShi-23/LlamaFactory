"""Tiny 1-attention-layer causal LM + deterministic dataset for validating the
GA/packing-invariant loss scaling under real DP+GA.

Kept intentionally minimal (and HF-``PreTrainedModel`` compatible) so the
``transformers.Trainer`` integration path is exercised exactly as production.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class ToyConfig(PretrainedConfig):
    model_type = "toy_causal"

    def __init__(self, vocab_size=64, hidden_size=32, num_heads=4, max_pos=256, **kw):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_pos = max_pos
        super().__init__(**kw)


class ToyForCausalLM(PreTrainedModel):
    """Embedding -> 1 causal self-attention block -> MLP -> tied LM head.

    forward returns only ``logits``; the loss is computed by the trainer's
    ``compute_loss`` override (so token/sample weighting + num_items live in one
    place, shared with production). ``accepts_loss_kwargs=True`` makes HF set
    ``model_accepts_loss_kwargs=True`` and skip the ``/GA`` division.
    """

    config_class = ToyConfig
    main_input_name = "input_ids"
    accepts_loss_kwargs = True
    supports_gradient_checkpointing = False

    def __init__(self, config: ToyConfig):
        super().__init__(config)
        h = config.hidden_size
        self.embed = nn.Embedding(config.vocab_size, h)
        self.pos = nn.Embedding(config.max_pos, h)
        self.ln1 = nn.LayerNorm(h)
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.attn_out = nn.Linear(h, h, bias=False)
        self.ln2 = nn.LayerNorm(h)
        self.mlp = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
        self.num_heads = config.num_heads
        self.head_dim = h // config.num_heads
        self.post_init()

    def forward(self, input_ids, labels=None, cu_seqlens=None, num_items_in_batch=None, **kwargs):
        B, N = input_ids.shape
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos(pos_ids)

        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, N, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        x = x + self.attn_out(attn)
        x = x + self.mlp(self.ln2(x))
        logits = self.lm_head(x)

        out = {"logits": logits}
        if labels is not None:
            # HF-native per-token loss path (mirrors ForCausalLMLoss): global
            # left-shift, reduction="sum", divide by num_items_in_batch.
            import torch.nn.functional as F

            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
            shift_labels = labels[:, 1:].reshape(-1)
            if num_items_in_batch is not None:
                ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
                denom = num_items_in_batch
                if not torch.is_tensor(denom):
                    denom = torch.tensor(float(denom), device=ce.device)
                out["loss"] = ce / denom.to(ce.device)
            else:
                out["loss"] = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean")
        return out


def _gen_samples(num_samples, seq_len, vocab_size, mask_frac=0.5, seed=0):
    """Deterministic list of (ids, labels) with a masked prefix of varying
    length so different samples have different valid-token counts (=> per_token
    and per_sample reductions are genuinely distinct)."""
    from llamafactory.train.sft.loss_scaling import IGNORE_INDEX

    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(num_samples):
        n_mask = int(seq_len * mask_frac) + (i % max(1, seq_len // 4))
        n_mask = min(n_mask, seq_len - 1)
        ids = torch.randint(0, vocab_size, (seq_len,), generator=g)
        labels = ids.clone()
        labels[:n_mask] = IGNORE_INDEX
        out.append((ids, labels))
    return out


def make_dataset(num_samples, seq_len, vocab_size, mask_frac=0.5, seed=0):
    """Non-packed: one sample per example."""
    return [{"input_ids": ids, "labels": labels}
            for ids, labels in _gen_samples(num_samples, seq_len, vocab_size, mask_frac, seed)]


def make_packed_dataset(num_samples, samples_per_pack, seq_len, vocab_size, mask_frac=0.5, seed=0):
    """Packed: ``samples_per_pack`` of the SAME underlying samples concatenated
    into one example, with ``cu_seqlens`` marking the sample boundaries. Uses the
    identical samples as ``make_dataset`` so packed vs non-packed must agree."""
    assert num_samples % samples_per_pack == 0, "num_samples must be divisible by samples_per_pack"
    samples = _gen_samples(num_samples, seq_len, vocab_size, mask_frac, seed)
    packs = []
    for p in range(0, num_samples, samples_per_pack):
        group = samples[p : p + samples_per_pack]
        ids = torch.cat([s[0] for s in group])
        labels = torch.cat([s[1] for s in group])
        cu = torch.tensor([0] + [(j + 1) * seq_len for j in range(len(group))], dtype=torch.int32)
        packs.append({"input_ids": ids, "labels": labels, "cu_seqlens": cu})
    return packs
