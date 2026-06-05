"""Single-process diagnostic: GA-invariance only (no DDP).

Vary (per_device_bs, ga) at constant global batch and compare the accumulated
gradient. Isolates HF's num_items + /GA-skip from any DDP grad-combine scaling.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from transformers import Trainer, TrainingArguments

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))

from toy_model import ToyConfig, ToyForCausalLM, make_dataset  # noqa: E402
from llamafactory.train.sft.loss_scaling import (  # noqa: E402
    count_num_items,
    loss_weights_from_shift,
    per_sample_shift_labels,
    weighted_token_ce,
)


class T(Trainer):
    def __init__(self, *a, loss_reduction="per_token", **k):
        super().__init__(*a, **k)
        self.loss_reduction = loss_reduction
        self._captured = None

    def _get_num_items_in_batch(self, batch_samples, device=None):
        total = None
        for b in batch_samples:
            if "labels" not in b:
                continue
            c = count_num_items(b["labels"], self.loss_reduction)
            total = c if total is None else total + c
        return total

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        shift = per_sample_shift_labels(labels)
        w = loss_weights_from_shift(shift, self.loss_reduction)
        logits = model(input_ids=inputs["input_ids"])["logits"]
        if num_items_in_batch is None:
            num_items_in_batch = w.sum().clamp_min(1.0)
        loss = weighted_token_ce(logits, shift, w, num_items_in_batch)
        return (loss, {"logits": logits}) if return_outputs else loss

    def _get_train_sampler(self, *a, **k):
        return torch.utils.data.SequentialSampler(self.train_dataset)

    def create_optimizer(self):
        opt = super().create_optimizer()
        real = opt.step
        tr = self

        def step(*a, **k):
            tr._capture()
            return real(*a, **k)

        opt.step = step
        return opt

    def _capture(self):
        if self._captured is not None:
            return
        base = self.accelerator.unwrap_model(self.model)
        parts = [(p.grad.detach().reshape(-1).float() if p.grad is not None else torch.zeros(p.numel()))
                 for _, p in sorted(base.named_parameters(), key=lambda x: x[0])]
        v = torch.cat(parts)
        self._captured = {"grad": v, "grad_norm": v.norm().item()}


def run(mode, bs, ga, global_bs=8, seq_len=24, vocab=64, seed=1234):
    torch.manual_seed(seed)
    model = ToyForCausalLM(ToyConfig(vocab_size=vocab))
    ds = make_dataset(global_bs, seq_len, vocab, seed=0)
    args = TrainingArguments(
        output_dir="/tmp/diag_ga", per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga, max_steps=1, learning_rate=0.0,
        max_grad_norm=0.0, logging_strategy="no", save_strategy="no", report_to=[],
        remove_unused_columns=False, dataloader_num_workers=0, seed=seed,
    )
    tr = T(model=model, args=args, train_dataset=ds, loss_reduction=mode)
    tr.train()
    return tr._captured


def main():
    global_bs = 8
    for mode in ["per_token", "per_sample"]:
        print(f"\n=== single-process GA invariance: mode={mode} global_bs={global_bs} ===")
        ref = None
        for bs, ga in [(1, 8), (2, 4), (4, 2), (8, 1)]:
            res = run(mode, bs, ga, global_bs)
            if ref is None:
                ref = res
                print(f"  bs={bs} ga={ga}: grad_norm={res['grad_norm']:.6f} (ref)")
            else:
                md = (ref["grad"] - res["grad"]).abs().max().item()
                close = torch.allclose(ref["grad"], res["grad"], atol=1e-5, rtol=1e-4)
                print(f"  bs={bs} ga={ga}: grad_norm={res['grad_norm']:.6f} max|dgrad|={md:.2e} [{'OK' if close else 'MISMATCH'}]")


if __name__ == "__main__":
    main()
