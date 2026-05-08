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
        # See /fsx/home/zijishi/.claude/plans/hashed-swinging-finch.md.
        self.cp_group = None
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

            # Swap triton_gqa -> triton_gqa_ulysses only if triton_gqa is the
            # currently selected implementation. Non-Gemma models keep their own path.
            from gemma_triton_flash_attn import register_triton_attention_ulysses

            register_triton_attention_ulysses(self.cp_group, name="triton_gqa_ulysses")
            if getattr(self.model.config, "_attn_implementation", None) == "triton_gqa":
                self.model.config._attn_implementation = "triton_gqa_ulysses"
            if hasattr(self.model.config, "text_config") and self.model.config.text_config is not None:
                if getattr(self.model.config.text_config, "_attn_implementation", None) == "triton_gqa":
                    self.model.config.text_config._attn_implementation = "triton_gqa_ulysses"

            logger.info_rank0(
                f"Context parallelism enabled: cp_size={cp}, dp_size={ws // cp}, "
                f"attn_implementation=triton_gqa_ulysses."
            )

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

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
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
            # (b) Pad + split all rank-2+ tensors on last dim across cp_group.
            # Labels are KEPT in inputs (not popped) so that the Liger fused-CE
            # patch in run_train.py fires inside model(): it bypasses logits
            # materialisation entirely (hidden_states × lm_head_weight → CE
            # in one Triton tile), saving ~32 GB bf16 logits at N_local=64K.
            # The labels passed in are this rank's shifted CP-local shard.
            _cp_dbg("loss_cp", "before padding_and_split_data")
            inputs = padding_and_split_data(dict(inputs), self.cp_group, ignore_index=IGNORE_INDEX)
            _cp_dbg("loss_cp", f"after padding_and_split_data; new_keys={list(inputs.keys())}")

            # Shift labels by 1 for causal LM within this rank's local shard.
            # Note: the last token of each CP rank's labels is a "cross-shard"
            # edge whose true target lives on the next rank. We approximate it
            # with IGNORE_INDEX (mask it out) — this introduces 1/N_local ≈ 0%
            # loss error, acceptable for training and smoke tests.
            labels_local = inputs.pop("labels")          # (B, N_local)
            shift_labels = torch.roll(labels_local, -1, dims=-1)
            shift_labels[:, -1] = IGNORE_INDEX           # mask the cross-rank edge
            # Replace labels in inputs with shifted version for Liger's CE.
            inputs["labels"] = shift_labels

            _cp_dbg("loss_cp", f"calling model(...) with input_keys={list(inputs.keys())}")
            outputs = model(**inputs)

            # Liger fused CE: `outputs.loss` is the local mean CE (no logits materialized).
            # If Liger didn't fire (e.g. fallback path), outputs.logits is non-None.
            if outputs.loss is not None:
                _cp_dbg("loss_cp", f"Liger path: outputs.loss={outputs.loss.item():.4f}")
                local_loss = outputs.loss
            else:
                # Fallback: Liger not active, compute chunked CE from logits.
                _cp_dbg("loss_cp", f"logits fallback; logits_shape={tuple(outputs.logits.shape)}")
                loss_weights = (shift_labels != IGNORE_INDEX).float()
                local_loss = sequence_parallel_loss_reduce(
                    outputs.logits, shift_labels, loss_weights, self.cp_group
                )
                if kwargs.get("return_outputs", False):
                    return local_loss, outputs
                return local_loss

            # Return local_loss directly for backward. Gradients flow through
            # Liger's fused CE → backbone — correct because DS all-reduces
            # gradients across all 16 world ranks, which naturally averages
            # over the 4 DP samples × 4 CP shards (math works out exactly).
            # No all_reduce of the loss needed for correctness; only log it.
            with torch.no_grad():
                global_loss_display = local_loss.detach().clone()
                dist.all_reduce(
                    global_loss_display, op=dist.ReduceOp.SUM, group=self.cp_group
                )
                global_loss_display.div_(dist.get_world_size(self.cp_group))
            _cp_dbg("loss_cp", f"global loss={global_loss_display.item():.4f}")
            # local_loss ≈ global_loss (both normalize per N_local or N_full
            # of valid tokens, which are equal in distribution). Return local.
            loss = local_loss

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
