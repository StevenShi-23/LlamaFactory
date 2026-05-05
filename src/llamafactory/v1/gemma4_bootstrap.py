"""Gemma-4 speedups + correctness patches, applied before any model load.

Three patches, ported from reasoning_distill/sft-simple/run_train.py (v0 path):

1. `torchaudio` stub — the gemma4-cp env has a pip record for torchaudio 2.7
   but no importable module. LF's data pipeline imports it eagerly; stub before
   that happens so passthrough SFT loads cleanly.

2. Triton GQA attention — registers the `triton_gqa` attention kernel (from
   gemma-triton-flash-attn, handles D=512 global + D=256 SWA layers), then
   forces Gemma4ForConditionalGeneration to use it. Without this the model
   silently falls back to eager/SDPA which materializes the N² attention
   scratch at each `full_attention` layer (10 × N² × 4 bytes fp32 upcast).
   At 8K × 32 heads × 10 layers that's ~100 GB — the primary cause of the
   Rung 1 v1 OOM at 133 GB/rank.

3. Liger fused-linear-CE — replaces Gemma4ForConditionalGeneration.forward
   loss branch with LigerForCausalLMLoss (softcap-aware since Liger main ≥
   Apr 2026). Streams (hidden_states × lm_head_weight → softcap → softmax →
   CE) in one triton kernel, skipping the (B, N, V) logits materialization.
   At 128K × 262K vocab that tensor alone is 67 GB before fp32 upcast.

4. `mm_token_type_ids` injection — Gemma4ForConditionalGeneration.forward
   requires `mm_token_type_ids` during training (modeling_gemma4.py:2026).
   Text-only SFT has no vision tokens; injecting zeros_like(input_ids) means
   every position is text (mm_token_type_ids=0 is the text-token sentinel).

Idempotent — safe to call multiple times.
"""
from __future__ import annotations

import os
import sys


_APPLIED = False


def _stub_torchaudio_if_missing() -> None:
    if "torchaudio" in sys.modules:
        return
    try:
        import torchaudio  # noqa: F401
    except ModuleNotFoundError:
        import importlib.machinery
        import types

        def _mk(name):
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            m.__loader__ = None
            return m

        stub = _mk("torchaudio")
        stub.__version__ = "0.0.0-stub"
        stub.__path__ = []
        sys.modules["torchaudio"] = stub
        for sub in ("transforms", "functional", "io", "compliance"):
            sys.modules[f"torchaudio.{sub}"] = _mk(f"torchaudio.{sub}")


def _strip_vision_audio_for_text_only() -> None:
    """Nullify `vision_config` and `audio_config` BEFORE Gemma4Model.__init__
    reads them (modeling_gemma4.py:2085, 2091), so vision_tower / audio_tower
    are never instantiated. For text-only SFT this saves ~1-2 GB of bf16
    weights per rank + avoids loading the corresponding checkpoint shards.

    Weights for vision/audio in the safetensors shards are simply not touched
    — from_pretrained's `state_dict.load` skips keys that have no target
    parameter (with a benign "unused keys" warning).
    """
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    except ImportError:
        return

    if getattr(Gemma4Config, "_text_only_patched", False):
        return

    _orig_post = Gemma4Config.__post_init__ if hasattr(Gemma4Config, "__post_init__") else None
    _orig_init = Gemma4Config.__init__

    # Opt-in via env var — text-only is the default here but keep the escape hatch.
    if os.environ.get("GEMMA4_TEXT_ONLY", "1") != "1":
        return

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Strip vision/audio after parent init so sub-configs are already parsed.
        if getattr(self, "vision_config", None) is not None:
            self.vision_config = None
        if getattr(self, "audio_config", None) is not None:
            self.audio_config = None
        # Force triton_gqa as the attention impl on config + text_config.
        # Belt-and-suspenders: also patched in Gemma4ForConditionalGeneration.__init__.
        self._attn_implementation = "triton_gqa"
        if hasattr(self, "text_config") and self.text_config is not None:
            self.text_config._attn_implementation = "triton_gqa"

    Gemma4Config.__init__ = _init
    Gemma4Config._text_only_patched = True
    if int(os.environ.get("RANK", "0")) == 0:
        print("[gemma4_bootstrap] vision_config + audio_config forced to None (text-only mode)", flush=True)


def _register_triton_flash_attn() -> None:
    try:
        from gemma_triton_flash_attn import (
            patch_transformers_5_5_4_flash_attn_key,
            register_triton_attention,
        )
    except ImportError:
        print("[gemma4_bootstrap] triton_gqa: gemma_triton_flash_attn not installed, skipping", flush=True)
        return

    patch_transformers_5_5_4_flash_attn_key()
    register_triton_attention()  # registers "triton_gqa" in ALL_ATTENTION_FUNCTIONS

    # Wrap the registered adapter on rank 0 to log first-call proof of routing.
    # Without this, there's no way to distinguish "triton_gqa was invoked but
    # didn't help" from "triton_gqa was never invoked, we silently fell back to eager".
    if int(os.environ.get("RANK", "0")) == 0:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        _adapter = ALL_ATTENTION_FUNCTIONS["triton_gqa"]
        _seen = {"n": 0}

        def _logged_adapter(module, query, key, value, *args, **kwargs):
            _seen["n"] += 1
            if _seen["n"] <= 3:
                B, Hq, N, D = query.shape
                print(f"[triton_gqa][rank0] call #{_seen['n']:3d} N={N} D={D} slide={kwargs.get('sliding_window', None)}", flush=True)
            return _adapter(module, query, key, value, *args, **kwargs)

        ALL_ATTENTION_FUNCTIONS["triton_gqa"] = _logged_adapter

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
    except ImportError:
        return

    if getattr(Gemma4ForConditionalGeneration, "_triton_gqa_patched", False):
        return

    _orig_init = Gemma4ForConditionalGeneration.__init__

    def _init(self, config, *args, **kwargs):
        config._attn_implementation = "triton_gqa"
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config._attn_implementation = "triton_gqa"
        return _orig_init(self, config, *args, **kwargs)

    Gemma4ForConditionalGeneration.__init__ = _init
    Gemma4ForConditionalGeneration._triton_gqa_patched = True
    if int(os.environ.get("RANK", "0")) == 0:
        print("[gemma4_bootstrap] triton_gqa registered + forced on Gemma4ForConditionalGeneration", flush=True)


def _patch_v1_base_trainer_fused_loss() -> None:
    """Replace BaseTrainer.compute_log_probs with a fused-linear-CE variant
    that calls Gemma-4's backbone (not the LM head) and runs LigerForCausalLMLoss
    directly on hidden_states — skipping the (B, N, V) logits materialization.

    Why patch `compute_log_probs` and not the model forward: v1's SFTTrainer
    does `(-log_probs * shift_loss_weights).sum() / shift_loss_weights.sum()`
    to apply per-token loss_weights (assistant-only loss for gemma4_passthrough).
    Returning log_probs of shape (B, N-1) preserves that behavior while skipping
    the expensive logits path.

    Non-gemma4 models still use the original implementation.
    """
    try:
        import torch
        import torch.nn.functional as F
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    except ImportError as e:
        print(f"[gemma4_bootstrap] Liger fused CE skipped: {e}", flush=True)
        return

    from llamafactory.v1.core.base_trainer import BaseTrainer

    if getattr(BaseTrainer, "_liger_ce_patched", False):
        return

    _orig = BaseTrainer.compute_log_probs

    def compute_log_probs(self, model, batch):
        """Fast-path for Gemma-4: compute log-probs via fused linear CE.

        Returns log_probs of shape (B, N-1) — same shape as the original.
        Falls through to the original implementation for non-Gemma-4 models.
        """
        # Unwrap DDP/FSDP2/DeepSpeed wrappers to inspect the underlying class.
        inner = model
        while hasattr(inner, "module"):
            inner = inner.module
        cls_name = type(inner).__name__
        if cls_name not in ("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"):
            return _orig(self, model, batch)

        batch_size, seq_len = batch["labels"].shape
        model_inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        labels = batch["labels"].to(self.device, non_blocking=True)
        # Run backbone only — we'll do LM head + CE fused.
        backbone_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
        if cls_name == "Gemma4ForConditionalGeneration":
            # Need to inject mm_token_type_ids when training (modeling_gemma4.py:2026).
            if inner.training and "mm_token_type_ids" not in backbone_inputs:
                backbone_inputs["mm_token_type_ids"] = torch.zeros_like(
                    backbone_inputs["input_ids"], dtype=torch.long
                )
            outputs = inner.model(**backbone_inputs, return_dict=True)
        else:
            outputs = inner.model(**backbone_inputs, return_dict=True)

        hidden_states = outputs.last_hidden_state  # (B, N, H)
        text_config = inner.config.get_text_config()
        softcap = getattr(text_config, "final_logit_softcapping", None)
        hidden_size = hidden_states.shape[-1]
        vocab_size = text_config.vocab_size
        lm_head_weight = inner.lm_head.weight

        # Shift for causal LM: hidden[t] predicts label[t+1].
        shift_hidden = hidden_states[..., :-1, :].contiguous().view(-1, hidden_size)  # (B*(N-1), H)
        shift_labels = labels[..., 1:].contiguous().view(-1)                           # (B*(N-1),)

        # Liger's underlying function returns a per-token CE loss tensor when
        # reduction="none". API: (hidden, weight, labels, bias, reduction, softcap, ...).
        # We want per-token log_probs = -CE_per_token, reshaped to (B, N-1).
        # Note: LigerFusedLinearCrossEntropyFunction.apply signature can drift
        # across versions. Use LigerForCausalLMLoss which is stable.
        # But LigerForCausalLMLoss returns a scalar — so go lower-level.
        per_token_loss = _liger_fused_ce_per_token(
            shift_hidden, lm_head_weight, shift_labels,
            vocab_size=vocab_size, softcap=softcap, ignore_index=-100,
        )
        log_probs = -per_token_loss.view(batch_size, -1)
        return log_probs

    BaseTrainer.compute_log_probs = compute_log_probs
    BaseTrainer._liger_ce_patched = True
    if int(os.environ.get("RANK", "0")) == 0:
        print("[gemma4_bootstrap] BaseTrainer.compute_log_probs patched to use Liger fused CE (Gemma-4)", flush=True)


def _liger_fused_ce_per_token(hidden_states, lm_head_weight, labels, *, vocab_size, softcap, ignore_index):
    """Call Liger's low-level fused-linear-CE with reduction='none' to get
    per-token CE. Handles softcap if supported; falls back to chunked manual
    path if Liger's API doesn't match.
    """
    import inspect
    import torch
    import torch.nn.functional as F

    # Try LigerFusedLinearCrossEntropyFunction.apply first.
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction as LF
        sig = inspect.signature(LF.apply)
        # Supported signature (Liger main, Apr 2026):
        # apply(_input, weight, target, bias, ce_weight, ignore_index,
        #      lse_square_scale, label_smoothing, reduction, softcap, return_z_loss, ...)
        # We want reduction='none' and softcap.
        out = LF.apply(
            hidden_states,
            lm_head_weight,
            labels,
            None,               # bias
            None,               # ce_weight
            ignore_index,
            0.0,                # lse_square_scale
            0.0,                # label_smoothing
            "none",             # reduction
            softcap,            # softcap
            False,              # return_z_loss
        )
        # Returns (loss, z_loss) or just loss depending on return_z_loss
        if isinstance(out, tuple):
            loss = out[0]
        else:
            loss = out
        # With reduction='none', loss is per-token (flat shape = labels.shape).
        # Positions with label=ignore_index have loss=0 (Liger sets them to 0).
        return loss
    except Exception as e:
        # Fallback: manual chunked path (same as run_train.py v0 but vectorized).
        CHUNK = 4096
        M = hidden_states.shape[0]
        out = torch.zeros(M, dtype=torch.float32, device=hidden_states.device)
        for start in range(0, M, CHUNK):
            end = min(start + CHUNK, M)
            h = hidden_states[start:end]
            l = labels[start:end]
            logits = F.linear(h, lm_head_weight)
            if softcap is not None:
                logits = (logits / softcap).tanh() * softcap
            per_tok = F.cross_entropy(logits.float(), l, ignore_index=ignore_index, reduction="none")
            out[start:end] = per_tok
        return out


def apply() -> None:
    """Apply all Gemma-4 bootstrap patches. Idempotent.

    Memory profiling lives in a separate module (``mem_profiler``) opted in
    via ``MEM_PROFILE=1``; see sft_trainer.py for the install site.
    """
    global _APPLIED
    if _APPLIED:
        return
    _stub_torchaudio_if_missing()
    _strip_vision_audio_for_text_only()
    _register_triton_flash_attn()
    _patch_v1_base_trainer_fused_loss()  # patches BaseTrainer.compute_log_probs
    # NOTE: mm_token_type_ids injection happens inside the patched compute_log_probs.
    _APPLIED = True


_APPLIED_V0 = False


def apply_v0() -> None:
    """Gemma-4 bootstrap subset for the v0 SFT trainer.

    Same as `apply()` minus `_patch_v1_base_trainer_fused_loss` (which targets
    v1's BaseTrainer). v0 handles loss in `CustomSeq2SeqTrainer.compute_loss`
    and does not need the Liger fused-CE patch to be installed at bootstrap
    time.

    Idempotent across both `apply()` and `apply_v0()`.
    """
    global _APPLIED_V0
    if _APPLIED_V0 or _APPLIED:
        return
    _stub_torchaudio_if_missing()
    _strip_vision_audio_for_text_only()
    _register_triton_flash_attn()
    _APPLIED_V0 = True
