"""CPU-only regression test for the SFT data *double-sharding* bug.

Trigger (non-CP path): ``context_parallel_size == 1`` + ``disable_shuffling`` +
``world_size > 1`` + ``len(dataset) % world_size == 0``. In that case the trainer's
``_dist_sharding_spec()`` returns ``(world, rank, False)`` and ``_get_train_sampler``
injects an explicit ``torch.utils.data.DistributedSampler(num_replicas=world)``.

Before the fix, ``get_train_dataloader`` only diverted to the raw DataLoader when
``cp_group is not None``, so this non-CP case fell through to HF's native path and
``accelerator.prepare`` re-sharded the *already sharded* sampler. The two stages
are:

    stage 1: torch ``DistributedSampler(num_replicas=world, rank=r)`` -> shards once
    stage 2: accelerate ``BatchSamplerShard(num_processes=world)``     -> shards AGAIN

Running BOTH stages (the bug) makes each rank see only ``len/world**2`` samples and
the union across ranks covers just ``len/world`` unique indices -- a world-fold data
loss. The fix routes ``spec is not None`` through the raw DataLoader, which uses the
explicit ``DistributedSampler`` ONLY (no ``accelerator.prepare`` / no
``BatchSamplerShard``), so the union across ranks covers every index exactly once.

This test drives the REAL ``accelerate.data_loader.BatchSamplerShard`` on top of a
real ``DistributedSampler`` for each rank -- no GPU and no ``torch.distributed``
runtime needed. It passes on the fixed code and would have caught the bug.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, DistributedSampler

WORLD = 8  # nproc_per_node in the repro (cp1 -> dp8)
N = 128  # len(dataset); N % WORLD == 0 is the bug's trigger
BATCH = 1  # per_device_train_batch_size in the repro config


class _RangeDataset:
    """Minimal map-style dataset whose item == its index, so collected sample
    values *are* dataset indices (lets us check coverage directly)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def _fixed_indices_for_rank(dataset, world: int, rank: int) -> list[int]:
    """FIXED raw-DataLoader path: the explicit DistributedSampler ONLY
    (disable_shuffling -> shuffle=False), exactly what the trainer builds when
    ``spec is not None`` and accelerate is bypassed."""
    return list(DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False))


def _buggy_indices_for_rank(dataset, world: int, rank: int, batch: int) -> list[int]:
    """BUGGY native path: DistributedSampler THEN accelerate BatchSamplerShard,
    i.e. the explicit sampler re-sharded by ``accelerator.prepare``."""
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False)
    # PyTorch auto-wraps (sampler, batch_size) into this BatchSampler inside DataLoader;
    # accelerate then re-shards that batch_sampler in prepare_data_loader.
    batch_sampler = BatchSampler(sampler, batch_size=batch, drop_last=False)
    sharded = BatchSamplerShard(
        batch_sampler, num_processes=world, process_index=rank, split_batches=False, even_batches=True
    )
    collected: list[int] = []
    for batch_indices in sharded:
        collected.extend(batch_indices)
    return collected


def _real_coverage(per_rank: list[list[int]]) -> Counter:
    """Count visits per *real* dataset index summed across ranks (BatchSamplerShard
    may wrap-pad with duplicate/oob indices for even batches; ignore those)."""
    counter: Counter = Counter()
    for idxs in per_rank:
        counter.update(i for i in idxs if 0 <= i < N)
    return counter


def test_fixed_path_covers_every_index_exactly_once():
    ds = _RangeDataset(N)
    per_rank = [_fixed_indices_for_rank(ds, WORLD, r) for r in range(WORLD)]
    cov = _real_coverage(per_rank)

    assert len(cov) == N, f"fixed path should reach all {N} indices, got {len(cov)}"
    assert sum(cov.values()) == N, f"fixed path should emit exactly {N} samples, got {sum(cov.values())}"
    assert set(cov.values()) == {1}, f"every index exactly once; got counts {sorted(set(cov.values()))}"
    # each rank gets a full, equal 1/world shard
    assert all(len(idxs) == N // WORLD for idxs in per_rank), [len(x) for x in per_rank]


def test_buggy_double_shard_loses_data():
    ds = _RangeDataset(N)
    per_rank = [_buggy_indices_for_rank(ds, WORLD, r, BATCH) for r in range(WORLD)]
    cov = _real_coverage(per_rank)

    # double sharding: union covers only len/world unique indices (world-fold loss)
    assert len(cov) < N, f"expected data loss but covered all {N} indices"
    assert len(cov) == N // WORLD, f"expected {N // WORLD} unique indices covered, got {len(cov)}"
    # each rank ends up with len/world**2 real samples
    per_rank_real = [sum(1 for i in idxs if 0 <= i < N) for idxs in per_rank]
    assert all(c == N // (WORLD * WORLD) for c in per_rank_real), per_rank_real


def test_fixed_recovers_world_times_more_than_buggy():
    ds = _RangeDataset(N)
    fixed = _real_coverage([_fixed_indices_for_rank(ds, WORLD, r) for r in range(WORLD)])
    buggy = _real_coverage([_buggy_indices_for_rank(ds, WORLD, r, BATCH) for r in range(WORLD)])
    assert len(fixed) == WORLD * len(buggy), (len(fixed), len(buggy))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"ALL SAMPLER COVERAGE TESTS PASSED (world={WORLD}, N={N}: fixed=128/128 unique, buggy={N // WORLD}/{N})")
