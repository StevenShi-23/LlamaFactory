"""CPU-only checks that the raw (non-accelerate) training DataLoader used by the
manually-sharded path -- ``llamafactory.train.sft.trainer._EpochAwareDataLoader``
-- matches the native HF / accelerate-prepared loader in the three ways that the
manual path would otherwise lose by bypassing ``accelerator.prepare``:

  req #1  per-epoch reshuffle : ``DataLoader.set_epoch(epoch)`` forwards to the
          sampler, so HF's ``if hasattr(train_dataloader, "set_epoch"):
          train_dataloader.set_epoch(epoch)`` reshuffles every epoch when shuffle
          is on, and stays deterministic when shuffle is off (disable_shuffling).
  req #3  seedable           : the sampler order is fully determined by
          ``(seed, epoch)`` -- same (seed, epoch) -> identical order, different
          seed/epoch -> different order.
  req #2  last-step          : ``accelerator.gradient_state.end_of_dataloader``
          (a process-wide singleton shared with the loader) flips to ``True`` on
          the final batch -- the flag transformers' trainer reads for
          ``is_last_step = self.accelerator.gradient_state.end_of_dataloader``.

Coverage (logic reused from ``test_sampler_coverage.py``): across ranks every
real index appears exactly once (the dummy-pad slots are excluded).

No GPU and no ``torch.distributed`` runtime are needed -- the samplers take an
explicit ``num_replicas`` / ``rank``. pytest may be absent, so the file is also
directly runnable:  ``python test_dataloader_parity.py``.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU even on a GPU box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from accelerate.state import GradientState
from torch.utils.data import DistributedSampler

from llamafactory.train.sft.trainer import _DummyPadDistributedSampler, _EpochAwareDataLoader

WORLD = 8  # nproc_per_node (cp1 -> dp8), matches test_sampler_coverage.py
NUM_REAL = 130  # 130 % 8 != 0 -> exercises method-1 dummy padding
N_DIVISIBLE = 128  # 128 % 8 == 0 -> torch DistributedSampler needs no padding


class _RangeDataset:
    """item == index, so a collected batch value *is* the dataset index. No bounds
    check (matches test_sampler_coverage.py), so the dummy index (== num_real) is
    fetchable and surfaces as its own value for filtering."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def _dummy_sampler(num_real: int, rank: int, shuffle: bool, seed: int):
    return _DummyPadDistributedSampler(
        num_real=num_real, num_replicas=WORLD, rank=rank, dummy_index=num_real, shuffle=shuffle, seed=seed
    )


def _torch_sampler(dataset, rank: int, shuffle: bool, seed: int):
    return DistributedSampler(dataset, num_replicas=WORLD, rank=rank, shuffle=shuffle, seed=seed)


def _order_via_loader(dataset, sampler, epoch: int) -> list[int]:
    """Per-rank index order produced by the REAL ``_EpochAwareDataLoader`` after
    ``loader.set_epoch(epoch)`` -- exercises req #1's forwarding end-to-end (loader
    -> sampler), the same call path as transformers' per-epoch ``set_epoch``."""
    loader = _EpochAwareDataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda b: b,  # identity: batch of 1 int -> [i]
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    assert hasattr(loader, "set_epoch"), "HF gates set_epoch on hasattr(train_dataloader, 'set_epoch')"
    loader.set_epoch(epoch)
    order: list[int] = []
    for batch in loader:
        order.extend(batch)
    return order


def _sampler_order(make_sampler, epoch: int) -> list[list[int]]:
    """Per-rank order straight from the sampler at a given epoch (no loader)."""
    out = []
    for r in range(WORLD):
        s = make_sampler(r)
        s.set_epoch(epoch)
        out.append(list(s))
    return out


def _assert_real_coverage_once(per_rank: list[list[int]], num_real: int) -> None:
    counter: Counter = Counter()
    for idxs in per_rank:
        counter.update(i for i in idxs if 0 <= i < num_real)
    assert len(counter) == num_real, f"covered {len(counter)}/{num_real} real indices"
    assert sum(counter.values()) == num_real, f"emitted {sum(counter.values())} real samples, want {num_real}"
    assert set(counter.values()) == {1}, f"each real index exactly once; got counts {sorted(set(counter.values()))}"


# ── req #1 (a): shuffle ON -> per-rank order CHANGES across epochs (via loader) ──
def test_set_epoch_reshuffles_when_shuffle_on():
    ds = _RangeDataset(NUM_REAL)
    e0 = [_order_via_loader(ds, _dummy_sampler(NUM_REAL, r, True, 1234), 0) for r in range(WORLD)]
    e1 = [_order_via_loader(ds, _dummy_sampler(NUM_REAL, r, True, 1234), 1) for r in range(WORLD)]
    assert e0 != e1, "set_epoch(0) vs set_epoch(1) must change the order when shuffle is on"
    changed = sum(1 for a, b in zip(e0, e1) if a != b)
    assert changed == WORLD, f"every rank should reshuffle across epochs, only {changed}/{WORLD} changed"
    _assert_real_coverage_once(e0, NUM_REAL)  # coverage preserved each epoch
    _assert_real_coverage_once(e1, NUM_REAL)


# ── req #1 (c): shuffle OFF -> order IDENTICAL across epochs (deterministic) ──
def test_set_epoch_deterministic_when_shuffle_off():
    ds = _RangeDataset(NUM_REAL)
    e0 = [_order_via_loader(ds, _dummy_sampler(NUM_REAL, r, False, 1234), 0) for r in range(WORLD)]
    e1 = [_order_via_loader(ds, _dummy_sampler(NUM_REAL, r, False, 1234), 1) for r in range(WORLD)]
    assert e0 == e1, "disable_shuffling (shuffle=False) order must be identical across epochs"
    _assert_real_coverage_once(e0, NUM_REAL)


# ── req #3 (b): reproducible by (seed, epoch); seed-sensitive (both samplers) ──
def test_reproducible_and_seed_sensitive():
    ds_div = _RangeDataset(N_DIVISIBLE)
    sampler_factories = {
        "dummy_pad": lambda seed: (lambda r: _dummy_sampler(NUM_REAL, r, True, seed)),
        "torch_distributed": lambda seed: (lambda r: _torch_sampler(ds_div, r, True, seed)),
    }
    for name, make in sampler_factories.items():
        same_a = _sampler_order(make(7), epoch=2)
        same_b = _sampler_order(make(7), epoch=2)
        other_seed = _sampler_order(make(8), epoch=2)
        assert same_a == same_b, f"{name}: same (seed=7, epoch=2) must be identical (reproducible)"
        assert same_a != other_seed, f"{name}: different seed must change the order"


# ── req #1 (a) at sampler level for both samplers: epoch changes order ──
def test_epoch_changes_order_both_samplers():
    ds_div = _RangeDataset(N_DIVISIBLE)
    d0 = _sampler_order(lambda r: _dummy_sampler(NUM_REAL, r, True, 5), 0)
    d1 = _sampler_order(lambda r: _dummy_sampler(NUM_REAL, r, True, 5), 1)
    assert d0 != d1, "dummy_pad: different epoch must change order (shuffle on)"
    t0 = _sampler_order(lambda r: _torch_sampler(ds_div, r, True, 5), 0)
    t1 = _sampler_order(lambda r: _torch_sampler(ds_div, r, True, 5), 1)
    assert t0 != t1, "torch_distributed: different epoch must change order (shuffle on)"


# ── coverage (d): each real index exactly once across ranks (both samplers) ──
def test_coverage_across_ranks():
    for shuffle in (False, True):
        per_rank = _sampler_order(lambda r: _dummy_sampler(NUM_REAL, r, shuffle, 3), 0)
        _assert_real_coverage_once(per_rank, NUM_REAL)
        # every rank runs the same number of steps (collective symmetry preserved)
        assert len({len(x) for x in per_rank}) == 1, [len(x) for x in per_rank]

    ds_div = _RangeDataset(N_DIVISIBLE)
    for shuffle in (False, True):
        per_rank = _sampler_order(lambda r: _torch_sampler(ds_div, r, shuffle, 3), 0)
        cov = Counter(i for idxs in per_rank for i in idxs)
        assert len(cov) == N_DIVISIBLE and set(cov.values()) == {1}, sorted(set(cov.values()))


# ── req #2: gradient_state.end_of_dataloader flips True on the final batch ──
def test_end_of_dataloader_flips_on_last_batch():
    gs = GradientState()  # same shared singleton the loader registers with
    ds = _RangeDataset(NUM_REAL)
    sampler = _dummy_sampler(NUM_REAL, 0, False, 0)
    n_batches = len(list(_dummy_sampler(NUM_REAL, 0, False, 0)))  # ceil(130/8) == 17
    loader = _EpochAwareDataLoader(
        ds, batch_size=1, sampler=sampler, collate_fn=lambda b: b, num_workers=0, pin_memory=False, drop_last=False
    )
    assert gs.end_of_dataloader is False, "not iterating yet -> False"

    flags = []
    for _ in loader:
        flags.append(bool(gs.end_of_dataloader))

    assert len(flags) == n_batches, f"expected {n_batches} batches, got {len(flags)}"
    assert flags[-1] is True, "end_of_dataloader must be True on the final batch (is_last_step)"
    assert not any(flags[:-1]), "end_of_dataloader must be False before the final batch"
    assert gs.end_of_dataloader is False, "must reset (in_dataloader False) after iteration ends"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(
        f"ALL DATALOADER PARITY TESTS PASSED (world={WORLD}, num_real={NUM_REAL}: "
        "set_epoch reshuffles on / deterministic off, seedable, end_of_dataloader flips last)"
    )
