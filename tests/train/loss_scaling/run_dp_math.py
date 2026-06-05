"""One (mode, packing, dp) config of the DP-invariance check, implementing the
target mechanism explicitly (framework-agnostic):

  per microbatch: backprop the RAW weighted CE *sum* (no /Z, no /GA)
  accumulate local grads across the GA window
  all_reduce(SUM) grads across DP ranks
  all_reduce(SUM) the item count -> global Z
  final_grad = reduced_grad / Z

This is exactly "sum loss per microbatch, accumulate, all-reduce, divide by the
global token/sample count at the end". Because it is the same global sum merely
partitioned differently across (dp, ga), the result must be identical for every
(dp, ga) at fixed global batch -- which is what we assert. It validates the
shift / weight / counting logic and its DP+GA composition, independent of any
DDP/DeepSpeed averaging convention (that is validated on the real stack in the
grid). CPU/gloo by default.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))

from toy_model import ToyConfig, ToyForCausalLM, make_dataset, make_packed_dataset  # noqa: E402
from llamafactory.train.sft.loss_scaling import (  # noqa: E402
    count_num_items,
    loss_weights_from_shift,
    per_sample_shift_labels,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["per_token", "per_sample"])
    ap.add_argument("--packing", type=int, default=0)
    ap.add_argument("--global_bs", type=int, required=True, help="number of SAMPLES in the global batch")
    ap.add_argument("--seq_len", type=int, default=24)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--samples_per_pack", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_seed", type=int, default=1234)
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    world = dist.get_world_size()
    rank = dist.get_rank()

    torch.manual_seed(args.model_seed)
    model = ToyForCausalLM(ToyConfig(vocab_size=args.vocab))
    # Make params identical across ranks (defensive; same seed already matches).
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model.train()

    # Build the SAME global set of samples on every rank, then shard strided so
    # the union across ranks is exactly the global batch (no overlap, no drop).
    if args.packing:
        examples = make_packed_dataset(
            args.global_bs, args.samples_per_pack, args.seq_len, args.vocab, seed=0
        )
    else:
        examples = make_dataset(args.global_bs, args.seq_len, args.vocab, seed=0)
    shard = examples[rank::world]

    model.zero_grad(set_to_none=True)
    local_items = torch.zeros((), dtype=torch.float64)
    for ex in shard:
        input_ids = ex["input_ids"].unsqueeze(0)
        labels = ex["labels"].unsqueeze(0)
        cu = ex.get("cu_seqlens")
        shift = per_sample_shift_labels(labels, cu)
        w = loss_weights_from_shift(shift, args.mode, cu)
        logits = model(input_ids=input_ids)["logits"]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            shift.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        loss = (ce * w.reshape(-1)).sum()  # RAW weighted sum, no normalization
        loss.backward()
        local_items += count_num_items(labels, args.mode, cu).double()

    # Reduce grads (SUM) and item count (SUM) across DP, then divide by global Z.
    parts = []
    for _, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        parts.append(g.detach().reshape(-1).clone())
    grad = torch.cat(parts)
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_items, op=dist.ReduceOp.SUM)
    Z = local_items.clamp_min(1.0)
    final = (grad / Z).float()

    if rank == 0:
        torch.save({"grad": final, "grad_norm": final.norm().item(), "Z": float(Z)}, args.out)
        print(f"[run_dp_math] mode={args.mode} packing={args.packing} dp={world} "
              f"Z={float(Z):.0f} grad_norm={final.norm().item():.6f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
