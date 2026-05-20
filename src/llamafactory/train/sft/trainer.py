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
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
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
        if finetuning_args.context_parallel_size > 1:
            import torch.distributed as dist
            from torch.distributed.device_mesh import init_device_mesh

            from .cp_utils import set_cp_group

            cp = finetuning_args.context_parallel_size
            ws = dist.get_world_size()
            self.cp_mesh = init_device_mesh("cuda", (ws // cp, cp), mesh_dim_names=("dp", "cp"))
            self.cp_group = self.cp_mesh["cp"].get_group()
            set_cp_group(self.cp_group)

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
        if self._cp_valid_accum > 0:
            logs["global_ce"] = round(self._cp_ce_sum_accum / self._cp_valid_accum, 4)
            self._cp_ce_sum_accum = 0.0
            self._cp_valid_accum = 0
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

    def get_train_dataloader(self):
        if self.cp_group is None:
            return super().get_train_dataloader()

        import torch.distributed as dist
        from torch.utils.data import DataLoader

        sampler = self._get_train_sampler()
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
        return dl

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
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

        # CP-aware sampler: only dp_size ranks contribute unique samples.
        # All CP ranks within a group get the same sample (via broadcast in
        # _compute_loss_cp). The sampler must shard data across dp_size, not
        # world_size, so each DP rank gets 1/dp_size of the dataset.
        if self.cp_group is not None:
            import torch.distributed as dist

            cp_size = dist.get_world_size(self.cp_group)
            world_size = dist.get_world_size()
            dp_size = world_size // cp_size
            dp_rank = dist.get_rank() // cp_size
            do_shuffle = not self.finetuning_args.disable_shuffling
            return torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=dp_size,
                rank=dp_rank,
                shuffle=do_shuffle,
                seed=self.args.seed,
            )

        if self.finetuning_args.disable_shuffling:
            import torch.distributed as dist

            if dist.is_initialized() and dist.get_world_size() > 1:
                return torch.utils.data.DistributedSampler(
                    self.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False,
                )
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.cp_group is not None:
            return self._compute_loss_cp(model, inputs, *args, **kwargs)

        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

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

        from .cp_utils import _cp_dbg, padding_and_split_data, sequence_parallel_loss_reduce

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

            # Pre-shift labels globally BEFORE splitting, so each CP rank's
            # shard has the correct next-token targets with no boundary gap.
            # Both HF's ForCausalLMLoss and Liger's LigerForCausalLMLoss skip
            # internal shifting when shift_labels is provided.
            labels_full = inputs.pop("labels")           # (B, N_full)
            shift_labels_full = torch.cat([
                labels_full[:, 1:],
                torch.full((labels_full.shape[0], 1), IGNORE_INDEX,
                           dtype=labels_full.dtype, device=labels_full.device),
            ], dim=-1)                                   # (B, N_full)
            inputs["shift_labels"] = shift_labels_full

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

            _cp_dbg("loss_cp", f"calling model(...) with input_keys={list(inputs.keys())}")
            outputs = model(**inputs)

            # NOTE: don't clear varlen_token here — backward (gradient checkpointing
            # recompute) needs the cu_seqlens ContextVar. It gets overwritten next step.

            # Liger fused CE: `outputs.loss` is the local mean CE (no logits materialized).
            # If Liger didn't fire (e.g. fallback path), outputs.logits is non-None.
            if outputs.loss is not None:
                local_loss = outputs.loss
            else:
                # Fallback: Liger not active, compute chunked CE from logits.
                loss_weights = (shift_labels_local != IGNORE_INDEX).float()
                local_loss = sequence_parallel_loss_reduce(
                    outputs.logits, shift_labels_local, loss_weights, self.cp_group
                )
                if kwargs.get("return_outputs", False):
                    return local_loss, outputs
                return local_loss

            # ── CP loss reduction ─────────────────────────────────────
            # Each CP rank holds a shard of one sample's sequence.
            # scaled_loss = local_mean_CE × (local_valid / global_valid)
            # gives each rank its fraction; sum across CP = global_ce.
            #
            # × cp_size corrects for DS ZeRO-3 averaging gradients over
            # world_size (= dp × cp) instead of dp_size. Without it the
            # gradient is 1/cp_size too small. Proof:
            #   DS gives: (1/(dp×cp)) × Σ_j [cp × d(mean_CE_j)/dθ]
            #           = (1/dp) × Σ_j d(mean_CE_j)/dθ   ← correct
            #
            # NaN-safe: if a shard has zero valid tokens, its contribution
            # is zeroed out to prevent Liger's 0/0 NaN from propagating.
            cp_size = dist.get_world_size(self.cp_group)
            local_valid = (shift_labels_local != IGNORE_INDEX).sum()
            global_valid = local_valid.clone().float()
            dist.all_reduce(global_valid, op=dist.ReduceOp.SUM, group=self.cp_group)

            has_valid = local_valid > 0
            safe_loss = torch.where(has_valid, local_loss, torch.zeros_like(local_loss))
            local_ce_sum = safe_loss * local_valid.float()
            weight = local_valid.float() / (global_valid + 1e-8)
            scaled_loss = safe_loss * weight * cp_size

            with torch.no_grad():
                global_ce_sum = local_ce_sum.detach().clone()
                dist.all_reduce(global_ce_sum, op=dist.ReduceOp.SUM, group=self.cp_group)
                global_ce = global_ce_sum / (global_valid + 1e-8)

            # Accumulate CE sum and valid count for per-token logging.
            self._cp_ce_sum_accum += global_ce_sum.item()
            self._cp_valid_accum += int(global_valid.item())

            # Identity trick: gradient flows through scaled_loss, but
            # .item() reports global_ce for wandb/logging.
            loss = scaled_loss - scaled_loss.detach() + global_ce.detach()

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
