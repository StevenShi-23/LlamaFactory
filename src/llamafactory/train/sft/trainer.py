# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


# Length (tokens) of the all-ignore dummy pad pack. Kept short to minimise wasted
# forward compute and a multiple of 8 (== the collator's pad_to_multiple_of) so no
# extra collator padding is added; it does NOT need to match cutoff_len.
DUMMY_PACK_LEN = 16


class _AllIgnoreDummyDataset(torch.utils.data.Dataset):
    """Wrap a map-style dataset to expose one extra trailing index (``len == N + 1``).

    Index ``N`` (``dummy_index``) returns a short, all-``IGNORE_INDEX`` "dummy pack".
    Paired with :class:`_DummyPadDistributedSampler`, this implements "method 1"
    padding: when the number of (packed) sequences is not divisible by the
    data-parallel degree, each replica's index list is padded to equal length with
    this sentinel instead of re-appending real packs from the front (torch
    ``DistributedSampler``'s default ``indices += indices[:pad]``).

    The dummy has all-ignore labels, so the per-token / per-sample loss counts it as
    0 items (``num_items``), 0 weighted CE sum and 0 gradient — a true no-op for the
    loss math, costing only a tiny wasted forward. The existing NaN-safe guards in
    ``_compute_loss_scaled`` / ``_compute_loss_cp`` (``has_valid`` / ``clamp_min``)
    absorb the resulting 0/0.
    """

    def __init__(self, dataset, pad_token_id: Optional[int] = None, length: int = DUMMY_PACK_LEN):
        self.dataset = dataset
        self.num_real = len(dataset)
        self.length = max(1, int(length))
        self.pad_token_id = int(pad_token_id) if pad_token_id is not None else 0

    @property
    def dummy_index(self) -> int:
        return self.num_real

    def __len__(self) -> int:
        return self.num_real + 1

    def __getitem__(self, index):
        if index == self.num_real:
            # Plain 0/1 attention mask (all ones) — NOT a neat-packing sample-id
            # mask — so cu_seqlens recovery treats it as one short sample. No
            # images/videos/audios/packing_params: the collator pops those with a
            # default of None, and per_device_train_batch_size == 1 means this
            # never shares a batch with a real (packed) example.
            return {
                "input_ids": [self.pad_token_id] * self.length,
                "attention_mask": [1] * self.length,
                "labels": [IGNORE_INDEX] * self.length,
            }
        return self.dataset[index]


class _DummyPadDistributedSampler(torch.utils.data.Sampler):
    """``DistributedSampler`` whose padding indices are an all-ignore dummy (method 1).

    Mirrors ``torch.utils.data.DistributedSampler`` (drop_last=False) — each replica
    gets exactly ``ceil(num_real / num_replicas)`` indices so every replica runs the
    same number of steps (collective symmetry preserved) — but the indices needed to
    reach a multiple of ``num_replicas`` are the dataset's ``dummy_index`` rather than
    wrapped-around real indices. This removes the small bias of double-counting the
    head pack(s), especially with ``disable_shuffling``.
    """

    def __init__(self, num_real: int, num_replicas: int, rank: int, dummy_index: int,
                 shuffle: bool = False, seed: int = 0):
        self.num_real = int(num_real)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dummy_index = int(dummy_index)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(self.num_real / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_real, generator=g).tolist()
        else:
            indices = list(range(self.num_real))

        pad = self.total_size - len(indices)
        if pad > 0:  # method 1: pad with the all-ignore dummy, not front-wrapped reals
            indices += [self.dummy_index] * pad

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)

        self.finetuning_args = finetuning_args

        # GA/packing-invariant loss scaling via HF's num_items_in_batch path.
        # When active, `compute_loss` consumes a globally-counted num_items and
        # HF skips its /gradient_accumulation_steps division (gated on
        # `model_accepts_loss_kwargs and num_items_in_batch is not None`). This
        # supersedes the old `processor -> model_accepts_loss_kwargs=False`
        # downgrade (transformers 5.5.4 no longer special-cases processors).
        # Disabled for the custom DFT/EAFT/ASFT losses, which keep their own path.
        self.loss_reduction = getattr(finetuning_args, "loss_reduction", "per_token")
        self._use_loss_scaling = not (
            finetuning_args.use_dft_loss
            or finetuning_args.use_eaft_loss
            or finetuning_args.use_asft_loss
        )
        if self._use_loss_scaling:
            self.model_accepts_loss_kwargs = True
        elif processor is not None:
            # avoid wrong loss under gradient accumulation for the custom losses
            self.model_accepts_loss_kwargs = False
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

        # Ulysses context parallelism (Gemma-4 long-context path).
        self.cp_group = None
        self._cp_ce_sum_accum = 0.0
        self._cp_valid_accum = 0
        self.cp_mesh = None

        # Per-step throughput logging (loss/grad_norm are already logged by HF;
        # we add wall-clock step_time and global tokens/sec). `_tokens_local_accum`
        # sums UNIQUE input tokens this rank contributes over the GA window — for
        # CP, only cp-rank 0 counts (peers replicate the same sample), so the
        # world all-reduce in `log()` yields the global unique-token count.
        self._last_log_time = None
        self._tokens_local_accum = 0
        if finetuning_args.context_parallel_size > 1:
            import torch.distributed as dist
            from torch.distributed.device_mesh import init_device_mesh

            from .cp_utils import set_cp_group

            cp = finetuning_args.context_parallel_size
            ws = dist.get_world_size()
            self.cp_mesh = init_device_mesh("cuda", (ws // cp, cp), mesh_dim_names=("dp", "cp"))
            self.cp_group = self.cp_mesh["cp"].get_group()
            set_cp_group(self.cp_group)

            # True data-parallel degree, read from the mesh's `dp` dimension.
            # DeepSpeed/DDP averages gradients over the ENTIRE world
            # (dist.get_world_size()); only `dp_size` of those ranks hold
            # distinct samples. Every other rank — CP peers today, and TP/PP
            # ranks if the mesh is later extended to e.g. (dp, pp, tp, cp) —
            # redundantly processes the same sample, so their per-shard losses
            # must be SUMMED, not averaged. The loss-scaling factor
            # world_size/dp_size (computed in `_compute_loss_cp`) cancels that
            # extra averaging. Equals cp_size for the current (dp, cp) mesh and
            # generalizes to cp*tp*pp with no change to the loss code.
            self.dp_size = self.cp_mesh["dp"].size()

            cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
            n_heads = cfg.num_attention_heads
            n_kv = cfg.num_key_value_heads
            if n_heads % cp != 0:
                raise ValueError(f"num_attention_heads ({n_heads}) not divisible by context_parallel_size ({cp}).")
            if not (n_kv % cp == 0 or cp % n_kv == 0):
                raise ValueError(
                    f"num_key_value_heads ({n_kv}) is incompatible with context_parallel_size ({cp}): "
                    "need n_kv % cp == 0 OR cp % n_kv == 0 (the latter replicates KV heads)."
                )

            # Swap triton_gqa -> CP variant. Use varlen_ulysses when packing is active.
            data_collator = kwargs.get("data_collator", None)
            use_packing = getattr(data_collator, "neat_packing", False)
            if use_packing:
                from gemma_triton_flash_attn import register_triton_attention_varlen_ulysses
                register_triton_attention_varlen_ulysses(self.cp_group, name="triton_gqa_varlen_ulysses")
                attn_name = "triton_gqa_varlen_ulysses"
            else:
                from gemma_triton_flash_attn import register_triton_attention_ulysses
                register_triton_attention_ulysses(self.cp_group, name="triton_gqa_ulysses")
                attn_name = "triton_gqa_ulysses"

            if getattr(self.model.config, "_attn_implementation", None) == "triton_gqa":
                self.model.config._attn_implementation = attn_name
            if hasattr(self.model.config, "text_config") and self.model.config.text_config is not None:
                if getattr(self.model.config.text_config, "_attn_implementation", None) == "triton_gqa":
                    self.model.config.text_config._attn_implementation = attn_name

            logger.info_rank0(
                f"Context parallelism enabled: cp_size={cp}, dp_size={ws // cp}, "
                f"attn_implementation={attn_name}, packing={use_packing}."
            )

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        import time

        import torch.distributed as dist

        if self._cp_valid_accum > 0:
            logs["global_ce"] = round(self._cp_ce_sum_accum / self._cp_valid_accum, 4)
            self._cp_ce_sum_accum = 0.0
            self._cp_valid_accum = 0

        # Per-step wall time + throughput. Only on training-step logs (which carry
        # "loss"); skip the final summary log so step_time isn't polluted by the
        # save/eval tail. `log()` is called symmetrically on all ranks at the same
        # steps, so the all-reduce below cannot deadlock.
        now = time.perf_counter()
        if "loss" in logs and self._last_log_time is not None:
            step_time = now - self._last_log_time
            local = torch.tensor(
                float(self._tokens_local_accum), device=self.accelerator.device
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local, op=dist.ReduceOp.SUM)
            global_tokens = local.item()
            logs["step_time"] = round(step_time, 3)
            if step_time > 0 and global_tokens > 0:
                logs["tokens_per_sec"] = round(global_tokens / step_time, 1)
        if "loss" in logs:
            self._last_log_time = now
            self._tokens_local_accum = 0

        super().log(logs, start_time)

    @override
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _dist_sharding_spec(self) -> Optional[tuple[int, int, bool]]:
        """``(num_replicas, rank, shuffle)`` for the active training sampler, or
        ``None`` when no ``DistributedSampler`` is used (single-process / HF default).

        - CP>1 : shard across ``dp_size`` (CP peers share a DP rank and get the same
          sample via the broadcast in ``_compute_loss_cp``), shuffle iff not disabled.
        - ``disable_shuffling`` + distributed : shard across the full world, no shuffle.
        """
        import torch.distributed as dist

        if self.cp_group is not None:
            cp_size = dist.get_world_size(self.cp_group)
            world_size = dist.get_world_size()
            return (world_size // cp_size, dist.get_rank() // cp_size, not self.finetuning_args.disable_shuffling)

        if self.finetuning_args.disable_shuffling:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                return (dist.get_world_size(), dist.get_rank(), False)
            return None  # single process -> SequentialSampler, no padding needed

        return None  # HF default (RandomSampler + accelerate sharding)

    def _needs_dummy_pad(self, spec: Optional[tuple[int, int, bool]]) -> bool:
        """A dummy pad pack is needed only when the dataset size is not divisible by
        the number of replicas (otherwise ``DistributedSampler`` adds no padding).
        Gated by ``LF_DUMMY_PAD`` (default on); a no-op contribution so safe generally.
        """
        if os.environ.get("LF_DUMMY_PAD", "1") != "1":
            return False
        if spec is None or spec[0] <= 1 or self.train_dataset is None:
            return False
        return len(self.train_dataset) % spec[0] != 0

    def get_train_dataloader(self):
        spec = self._dist_sharding_spec()
        use_dummy = self._needs_dummy_pad(spec)

        # Non-CP without a remainder keeps HF's standard (accelerate-prepared)
        # dataloader untouched. Everything else builds a raw DataLoader: all CP>1
        # (existing behavior) and the remainder case, where the sampler must be able
        # to emit the all-ignore dummy index (method 1) instead of letting torch's
        # DistributedSampler / accelerate duplicate a real pack from the front.
        if self.cp_group is None and not use_dummy:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader

        dataset = self.train_dataset
        if use_dummy:
            pad_id = getattr(self.processing_class, "pad_token_id", None)
            dataset = _AllIgnoreDummyDataset(dataset, pad_token_id=pad_id)
            logger.info_rank0(
                f"Dummy-pack padding enabled: {len(self.train_dataset)} packs not divisible by "
                f"{spec[0]} replica(s); padding each replica with one all-ignore dummy pack "
                "(method 1) instead of duplicating a real pack."
            )

        sampler = self._get_train_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

    @override
    def _get_train_sampler(self, train_dataset=None, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        # Test hook: fixed-order sampler cycles [0,1,2,...,N-1] regardless of
        # world_size or CP. Set CP_TEST_FIXED_SAMPLER=1 to enable.
        if os.environ.get("CP_TEST_FIXED_SAMPLER") == "1":
            from torch.utils.data import Sampler

            class _FixedOrder(Sampler):
                def __init__(self, ds):
                    self.n = len(ds)
                def __iter__(self):
                    i = 0
                    while True:
                        yield i % self.n
                        i += 1
                def __len__(self):
                    return self.n * 1000

            return _FixedOrder(self.train_dataset)

        if train_dataset is None:
            train_dataset = self.train_dataset

        # CP-aware / disable_shuffling sharding. CP shards across dp_size (CP peers
        # share a DP rank); disable_shuffling shards across the full world. When the
        # dataset is the dummy-padded wrapper, emit the all-ignore dummy index for
        # the padding slots; otherwise fall back to torch's DistributedSampler.
        spec = self._dist_sharding_spec()
        if spec is not None:
            num_replicas, rank, do_shuffle = spec
            if isinstance(train_dataset, _AllIgnoreDummyDataset):
                return _DummyPadDistributedSampler(
                    num_real=train_dataset.num_real,
                    num_replicas=num_replicas,
                    rank=rank,
                    dummy_index=train_dataset.dummy_index,
                    shuffle=do_shuffle,
                    seed=self.args.seed,
                )
            return torch.utils.data.DistributedSampler(
                train_dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=do_shuffle,
                seed=self.args.seed,
            )

        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(train_dataset)

        return super()._get_train_sampler(train_dataset, *args, **kwargs)

    @staticmethod
    def _cu_seqlens_from_batch(batch: dict) -> Optional["torch.Tensor"]:
        """Recover packing sample boundaries from a neat-packing 2D attention
        mask of sample-ids (e.g. [1,1,2,2,2,0]). Returns int32 cu_seqlens over
        the flattened sequence, or None when not packed (one sample per row)."""
        am = batch.get("attention_mask")
        labels = batch.get("labels")
        if am is None or not torch.is_tensor(am) or am.dim() != 2:
            return None
        if labels is not None and labels.shape[0] != 1:
            return None  # multi-row batch: each row is its own sample
        idx = am.reshape(-1)
        if int(idx.max()) <= 1:
            return None  # plain 0/1 mask, not neat-packing sample ids
        boundaries = [0]
        for uid in idx[idx != 0].unique(sorted=True).tolist():
            boundaries.append(boundaries[-1] + int((idx == uid).sum()))
        return torch.tensor(boundaries, dtype=torch.int32, device=am.device)

    @override
    def _get_num_items_in_batch(self, batch_samples, device=None):
        """Global ``num_items`` (Z) over the GA window: valid shifted tokens
        (per_token) or samples (per_sample), counted on per-sample-shifted
        labels and reduced over the DATA-parallel group only (CP peers hold the
        same data via broadcast, so we divide the world sum by cp_size)."""
        if not getattr(self, "_use_loss_scaling", False):
            return super()._get_num_items_in_batch(batch_samples, device)

        import torch.distributed as dist

        from .loss_scaling import count_num_items

        total = None
        for batch in batch_samples:
            # Duck-type: batches may be a plain dict OR a transformers
            # BatchEncoding (a UserDict, for which isinstance(batch, dict) is
            # False -- counting it as a dict would silently zero num_items and
            # leave the loss unnormalized).
            try:
                labels = batch["labels"]
            except (KeyError, TypeError, IndexError):
                continue
            if labels is None:
                continue
            cu = self._cu_seqlens_from_batch(batch)
            c = count_num_items(labels, self.loss_reduction, cu)
            total = c if total is None else total + c
        if total is None:
            return None
        if device is not None:
            total = total.to(device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            cp = dist.get_world_size(self.cp_group) if self.cp_group is not None else 1
            if cp > 1:
                total = total / cp
        return total

    def _scale_loss(self, local_weighted_sum, num_items_in_batch, display_value=None):
        """``grad_loss = (local_weighted_sum / Z) * world``.

        The optimizer must see exactly ``(1/Z) * d/dθ Σ_t w_t·ce_t`` for ANY
        (dp, cp, ga) layout. With ``num_items`` set, HF skips its own
        ``/gradient_accumulation_steps`` division, so the per-microbatch weighted
        CE sums add up across the GA window; DeepSpeed only averages grads by
        ``1/world`` (it does NOT re-apply ``1/gas`` here), so multiplying by
        ``world`` reconstructs the global ``(1/Z) Σ_t w_t·ce_t`` independent of
        (dp, cp, ga).

        When ``display_value`` (a per-token-mean CE) is given, an identity trick
        keeps the gradient flowing through the scaled term while the *logged*
        loss reads as that mean (divided by gas because HF SUMS the GA-window
        micro-batch losses when its /GA division is skipped).
        """
        import torch.distributed as dist

        world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        gas = getattr(self, "current_gradient_accumulation_steps", None) or self.args.gradient_accumulation_steps or 1
        if num_items_in_batch is None:
            num_items_in_batch = local_weighted_sum.new_tensor(1.0)
        denom = num_items_in_batch
        if not torch.is_tensor(denom):
            denom = local_weighted_sum.new_tensor(float(denom))
        denom = denom.to(local_weighted_sum.device).clamp_min(1.0)
        scaled = local_weighted_sum / denom * world

        if display_value is None:
            return scaled
        disp = display_value if torch.is_tensor(display_value) else local_weighted_sum.new_tensor(float(display_value))
        return scaled - scaled.detach() + disp.detach().to(scaled.device) / gas

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.cp_group is not None:
            return self._compute_loss_cp(model, inputs, *args, **kwargs)

        # Non-CP: each rank holds a distinct sample, so every rank's tokens are
        # unique. Accumulate over the GA window for throughput logging.
        if "input_ids" in inputs:
            self._tokens_local_accum += int(inputs["input_ids"].numel())

        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        if not self._use_loss_scaling:
            # DFT/EAFT custom losses keep HF's native path.
            return super().compute_loss(model, inputs, *args, **kwargs)

        return self._compute_loss_scaled(model, inputs, *args, **kwargs)

    def _compute_loss_scaled(self, model, inputs, *args, **kwargs):
        """Non-CP per_token / per_sample loss via global num_items scaling."""
        from .loss_scaling import (
            loss_weights_from_shift,
            per_sample_shift_labels,
            token_ce_sums,
        )

        num_items = kwargs.get("num_items_in_batch", None)
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        cu = self._cu_seqlens_from_batch({"labels": labels, "attention_mask": inputs.get("attention_mask")})
        shift_labels = per_sample_shift_labels(labels, cu)
        weights = loss_weights_from_shift(shift_labels, self.loss_reduction, cu)

        if self.loss_reduction == "per_token":
            # Efficient: Liger fused returns the local mean CE; recover CE_sum.
            inputs["shift_labels"] = shift_labels
            outputs = model(**inputs)
            valid = (shift_labels != IGNORE_INDEX).sum().float()
            if outputs.loss is not None:
                # NaN-safe: an all-ignore micro-batch (e.g. a dummy pad pack) makes
                # the model's mean CE 0/0 -> NaN. `torch.where` masks it to 0 and
                # routes zero gradient to `outputs.loss`, so the dummy contributes
                # nothing (mirrors the guard in `_compute_loss_cp`).
                has_valid = valid > 0
                safe = torch.where(has_valid, outputs.loss, torch.zeros_like(outputs.loss))
                local_sum = safe * valid
            else:
                local_sum, _, _ = token_ce_sums(outputs.logits, shift_labels, weights)
        else:  # per_sample: need per-token CE -> logits path (no fused loss)
            inputs.pop("shift_labels", None)
            outputs = model(**inputs)
            if getattr(outputs, "logits", None) is None:
                raise RuntimeError("per_sample loss requires logits; Liger fused must be bypassed for labels-less forward")
            local_sum, _, _ = token_ce_sums(outputs.logits, shift_labels, weights)

        disp = local_sum.detach() / weights.sum().clamp_min(1.0)  # token/sample-mean CE for logging
        loss = self._scale_loss(local_sum, num_items, display_value=disp)
        return (loss, outputs) if kwargs.get("return_outputs", False) else loss

    def _compute_loss_cp(self, model, inputs, *args, **kwargs):
        """CP-aware loss. Pads+splits inputs on the seq dim across `self.cp_group`,
        runs the forward (attention does Ulysses all-to-all internally), and
        reduces CE loss across CP ranks.

        For Gemma-4 with `use_bidirectional_attention == "vision"`, the image-
        group state is computed on the FULL seq BEFORE the split, otherwise
        each rank's shard loses the group context. This path is a no-op when
        the installed `gemma_triton_flash_attn` does not expose the vision-
        group helpers (e.g., the cookbook build).
        """
        import torch.distributed as dist

        from .cp_utils import _cp_dbg, padding_and_split_data
        from .loss_scaling import loss_weights_from_shift, per_sample_shift_labels, token_ce_sums

        _cp_dbg("loss_cp", f"enter; input_keys={list(inputs.keys())}")

        # (a) Compute Gemma-4 vision-group state on full seq pre-split, if applicable.
        # The helpers only exist on the fork build of gemma_triton_flash_attn; on
        # the cookbook build the import fails and we fall through (no-op).
        token = None
        image_group_state = None
        try:
            from gemma_triton_flash_attn.hf_integration import (
                _compute_image_group_state,
                _image_group_state as image_group_state,
            )

            mmt = inputs.get("mm_token_type_ids", None)
            text_cfg = (
                model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
            )
            if getattr(text_cfg, "use_bidirectional_attention", None) == "vision" and mmt is not None:
                token = image_group_state.set(_compute_image_group_state(mmt))
        except ImportError:
            pass

        try:
            # (b) Replicate data across CP group: HF's DistributedSampler gives
            # each rank a DIFFERENT sample, but all CP ranks must process the
            # SAME sample (just different seq-position shards). Broadcast rank 0's
            # inputs to all other ranks within the CP group.
            # Real data has variable-length samples, so tensors have different shapes
            # across ranks. First broadcast shape, then allocate matching buffers.
            cp_src = dist.distributed_c10d.get_global_rank(self.cp_group, 0)
            cp_local_rank = dist.get_rank(self.cp_group)
            inputs = dict(inputs)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    shape_tensor = torch.tensor(v.shape, dtype=torch.long, device=v.device)
                    dist.broadcast(shape_tensor, src=cp_src, group=self.cp_group)
                    if cp_local_rank != 0:
                        v = torch.empty(shape_tensor.tolist(), dtype=v.dtype, device=v.device)
                        inputs[k] = v
                    dist.broadcast(v, src=cp_src, group=self.cp_group)

            # Throughput accounting: the full (pre-split) sample is now replicated
            # on every CP rank. Count it ONCE per CP group (cp-rank 0 only) so the
            # world all-reduce in `log()` doesn't multiply by cp_size.
            if cp_local_rank == 0 and "input_ids" in inputs:
                self._tokens_local_accum += int(inputs["input_ids"].numel())

            # Extract cu_seqlens from attention_mask before splitting (for varlen+packing)
            varlen_token = None
            attn_impl = getattr(self.model.config, "_attn_implementation", "")
            is_varlen = "varlen" in (attn_impl or "")
            pre_split_cu = None
            if is_varlen:
                attn_mask = inputs.get("attention_mask")
                if attn_mask is not None and attn_mask.dim() == 2:
                    from gemma_triton_flash_attn.hf_integration import (
                        cu_seqlens_from_2d_indices,
                    )
                    pre_split_cu, _ = cu_seqlens_from_2d_indices(attn_mask)

            if "position_ids" not in inputs:
                seq_len = inputs["input_ids"].shape[-1]
                inputs["position_ids"] = torch.arange(
                    seq_len, device=inputs["input_ids"].device
                ).unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)

            # Per-sample causal shift BEFORE splitting: respects packing sample
            # boundaries (each sample's last token -> IGNORE so it never predicts
            # the next sample's first token) and leaves no gap at CP shard
            # boundaries. cu_full = None when not packing (each row is a sample).
            # Both HF's ForCausalLMLoss and Liger skip internal shifting when
            # shift_labels is provided.
            labels_full = inputs.pop("labels")           # (B, N_full)
            cu_full = pre_split_cu if (is_varlen and pre_split_cu is not None) else None
            shift_labels_full = per_sample_shift_labels(labels_full, cu_full)  # (B, N_full)
            inputs["shift_labels"] = shift_labels_full
            if self.loss_reduction == "per_sample":
                # Per-token weights 1/v_s; split alongside labels (padded with 0).
                inputs["loss_weights"] = loss_weights_from_shift(shift_labels_full, "per_sample", cu_full)

            _cp_dbg("loss_cp", "before padding_and_split_data")
            inputs = padding_and_split_data(inputs, self.cp_group, label_key="shift_labels", ignore_index=IGNORE_INDEX)
            _cp_dbg("loss_cp", f"after padding_and_split_data; new_keys={list(inputs.keys())}")

            # Set varlen cu_seqlens AFTER split so we know the full padded length
            if is_varlen and pre_split_cu is not None:
                from gemma_triton_flash_attn.hf_integration import set_varlen_cu_seqlens
                cp_ws = dist.get_world_size(self.cp_group)
                N_local = inputs["input_ids"].shape[-1]
                N_full = N_local * cp_ws
                total_valid = int(pre_split_cu[-1].item())
                max_sl = max(int(pre_split_cu[i+1] - pre_split_cu[i]) for i in range(pre_split_cu.numel() - 1))
                # Append dummy sample for padding region (stable shape for grad checkpoint)
                if total_valid < N_full:
                    cu_final = torch.cat([pre_split_cu, torch.tensor([N_full], dtype=torch.int32, device=pre_split_cu.device)])
                    max_sl = max(max_sl, N_full - total_valid)
                else:
                    cu_final = pre_split_cu
                varlen_token = set_varlen_cu_seqlens(cu_final, max_sl)
                _cp_dbg("loss_cp", f"varlen: {pre_split_cu.numel()-1} samples, valid={total_valid}, N_full={N_full}, cu_len={cu_final.numel()}")

            shift_labels_local = inputs["shift_labels"]   # (B, N_local)
            w_local = inputs.pop("loss_weights", None)    # not a model input
            if self.loss_reduction == "per_sample":
                inputs.pop("shift_labels", None)          # return logits, not fused loss

            _cp_dbg("loss_cp", f"calling model(...) with input_keys={list(inputs.keys())}")
            outputs = model(**inputs)

            # NOTE: don't clear varlen_token here — backward (gradient checkpointing
            # recompute) needs the cu_seqlens ContextVar. It gets overwritten next step.

            # ── CP loss reduction (num_items-based, GA/packing-invariant) ──
            # Each CP rank holds a shard of one sample's sequence and computes the
            # weighted CE SUM over its shard. `_scale_loss` divides by the global
            # num_items (Z) and multiplies by `world`; DeepSpeed then averages
            # grads over the world, so summing across CP shards + DP + GA yields
            # exactly (1/Z) * sum_t w_t * ce_t -- independent of (dp, cp, ga).
            #   per_token : w_t = 1            , Z = global valid tokens
            #   per_sample: w_t = 1/v_{sample} , Z = global sample count
            # NaN-safe: a zero-valid shard contributes 0 (no 0/0).
            local_valid = (shift_labels_local != IGNORE_INDEX).sum()
            if self.loss_reduction == "per_token":
                if outputs.loss is not None:
                    # Liger fused returned the shard's MEAN CE; recover CE_sum.
                    has_valid = local_valid > 0
                    safe = torch.where(has_valid, outputs.loss, torch.zeros_like(outputs.loss))
                    local_weighted_sum = safe * local_valid.float()
                    local_ce_sum = local_weighted_sum
                else:
                    weights = (shift_labels_local != IGNORE_INDEX).float()
                    local_weighted_sum, local_ce_sum, _ = token_ce_sums(
                        outputs.logits, shift_labels_local, weights
                    )
            else:  # per_sample
                if getattr(outputs, "logits", None) is None:
                    raise RuntimeError("per_sample CP loss requires logits (Liger fused must be bypassed)")
                local_weighted_sum, local_ce_sum, _ = token_ce_sums(
                    outputs.logits, shift_labels_local, w_local
                )

            # Token-level global_ce (mean CE), reduced across CP — used both for
            # logging and as the displayed loss value via the identity trick.
            with torch.no_grad():
                global_ce_sum = local_ce_sum.detach().clone().float()
                global_valid = local_valid.detach().clone().float()
                dist.all_reduce(global_ce_sum, op=dist.ReduceOp.SUM, group=self.cp_group)
                dist.all_reduce(global_valid, op=dist.ReduceOp.SUM, group=self.cp_group)
                global_ce = global_ce_sum / global_valid.clamp_min(1.0)
            self._cp_ce_sum_accum += global_ce_sum.item()
            self._cp_valid_accum += int(global_valid.item())

            num_items = kwargs.get("num_items_in_batch", None)
            loss = self._scale_loss(local_weighted_sum, num_items, display_value=global_ce)

            if kwargs.get("return_outputs", False):
                return loss, outputs
            return loss
        finally:
            if token is not None and image_group_state is not None:
                image_group_state.reset(token)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
