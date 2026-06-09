"""CPU-only regression test for the SFT train-sampler ``data_seed`` *parity* bug.

Trigger: ``data_seed`` is set to something other than ``seed``. HF's native cp=1
path shuffles its data sampler with ``data_seed`` (falling back to ``seed`` only
when ``data_seed is None``) -- via accelerate's ``SeedableRandomSampler`` /
``DataLoaderConfiguration.data_seed`` (``TrainingArguments.data_seed`` docs:
"If not set, ... use the same seed as ``seed``"). The custom cp>1 / manually-sharded
path in ``llamafactory.train.sft.trainer.CustomSeq2SeqTrainer._get_train_sampler``
used to hardcode ``seed=self.args.seed`` for both ``torch.utils.data.DistributedSampler``
and ``_DummyPadDistributedSampler``. So whenever ``data_seed != seed`` the two paths
shuffled with *different* seeds and produced *different* data orders -- cp=1 and cp>1
diverged on the exact same config.

The fix resolves an ``effective_data_seed = data_seed if data_seed is not None else
seed`` (mirroring HF) and seeds BOTH custom samplers with it, so cp=1 and the
(fixed) cp>1 path shuffle identically.

This test reproduces the order math at the sampler level (no Trainer, no model):
  * cp=1 global order == HF's ``SeedableRandomSampler`` / ``RandomSampler`` yield:
    ``g.manual_seed(resolve(seed, data_seed) + epoch)`` -> ``randperm(N)``.
  * cp>1 global order == the interleave ("union") across dp ranks of
    ``DistributedSampler(range(N), num_replicas=dp, rank=r, shuffle=True,
    seed=resolve(seed, data_seed))`` with ``set_epoch(epoch)`` -- which strides the
    SAME randperm, so the dp ranks reassemble exactly that global order.
With ``data_seed != seed`` the FIXED cp>1 order matches cp=1, the BUGGY cp>1 order
(seeded with ``seed``) does not, and with ``data_seed is None`` both already agree.

No GPU and no ``torch.distributed`` runtime are needed -- the samplers take an
explicit ``num_replicas`` / ``rank`` (like the sibling tests). pytest may be absent,
so the file is also directly runnable:  ``python test_sampler_seed_parity.py``.
"""

from __future__ import annotations

import os
import sys


os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU even on a GPU box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import torch
from torch.utils.data import DistributedSampler

from llamafactory.train.sft.trainer import _DummyPadDistributedSampler


SEED = 42  # TrainingArguments.seed (the model/global seed)
DATA_SEED = 1234  # TrainingArguments.data_seed (distinct -> exposes the bug)
N = 128  # dataset size; N % DP == 0 -> DistributedSampler needs no padding
DP = 4  # data-parallel degree (e.g. world=8, cp=2 -> dp=4)
EPOCHS = (0, 1, 2)  # per-epoch reshuffle must stay in lockstep too


def resolve(seed: int, data_seed: int | None) -> int:
    """HF-style data-sampler seed: ``data_seed`` when set, else fall back to ``seed``.

    Mirrors the ``effective_data_seed`` line added to
    ``CustomSeq2SeqTrainer._get_train_sampler`` and HF/accelerate's
    ``data_seed if data_seed is not None else seed``.
    """
    return data_seed if data_seed is not None else seed


def _cp1_global_order(n: int, seed: int, epoch: int) -> list[int]:
    """The cp=1 (HF native) global order -- exactly what HF's data sampler yields.

    ``SeedableRandomSampler`` / ``RandomSampler`` produce ``randperm(n)`` from a generator
    seeded with ``seed + epoch`` (accelerate adds the epoch to ``initial_seed``).
    """
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    return torch.randperm(n, generator=g).tolist()


def _cp_per_rank_orders(n: int, dp: int, seed: int, epoch: int, sampler: str = "distributed") -> list[list[int]]:
    """Per-rank index streams for the cp>1 / manually-sharded path (one list per dp rank).

    ``sampler="distributed"`` -> ``torch.utils.data.DistributedSampler``;
    ``sampler="dummy_pad"``   -> ``_DummyPadDistributedSampler`` (the other sampler the
    fix touches; with ``n % dp == 0`` it adds no padding and strides the same randperm).
    """
    out: list[list[int]] = []
    for r in range(dp):
        if sampler == "distributed":
            s: object = DistributedSampler(range(n), num_replicas=dp, rank=r, shuffle=True, seed=seed)
        elif sampler == "dummy_pad":
            s = _DummyPadDistributedSampler(
                num_real=n, num_replicas=dp, rank=r, dummy_index=n, shuffle=True, seed=seed
            )
        else:
            raise ValueError(sampler)
        s.set_epoch(epoch)
        out.append(list(s))
    return out


def _reconstruct_global(per_rank: list[list[int]], dp: int) -> list[int]:
    """Reassemble the global order from the dp ranks' strided slices.

    ``DistributedSampler`` hands rank ``r`` the slice ``global[r::dp]``, so interleaving
    the ranks index-by-index (``global[i*dp + r] = per_rank[r][i]``) is the inverse of
    that stride and recovers the full global order (the "union" across dp ranks).
    """
    num_samples = len(per_rank[0])
    assert all(len(x) == num_samples for x in per_rank), [len(x) for x in per_rank]
    order: list[int] = []
    for i in range(num_samples):
        for r in range(dp):
            order.append(per_rank[r][i])
    return order


# ── resolve(): data_seed when set (even 0!), else fall back to seed ──
def test_resolve_prefers_data_seed_else_seed():
    assert resolve(42, 1234) == 1234  # data_seed set -> use it
    assert resolve(42, 0) == 0  # data_seed=0 is "set" (not None) -> honored, NOT treated as falsy
    assert resolve(42, None) == 42  # data_seed None -> fall back to seed
    assert resolve(7, None) == 7
    assert resolve(7, 7) == 7


# ── KEY parity: with data_seed != seed, FIXED cp>1 (seed=data_seed) matches cp=1 ──
def test_cp1_and_fixed_cp2_match_with_distinct_data_seed():
    s = resolve(SEED, DATA_SEED)
    assert s == DATA_SEED, "data_seed is set, so resolve() must pick data_seed"
    for epoch in EPOCHS:
        cp1 = _cp1_global_order(N, s, epoch)
        per_rank = _cp_per_rank_orders(N, DP, s, epoch, sampler="distributed")
        cp2 = _reconstruct_global(per_rank, DP)
        assert cp2 == cp1, f"epoch {epoch}: fixed cp>1 global order must match cp=1 once both use data_seed"
        # per-rank groupings line up: each rank is exactly the cp=1 randperm strided by dp
        for r in range(DP):
            assert per_rank[r] == cp1[r::DP], (
                f"epoch {epoch} rank {r}: DistributedSampler must stride the cp=1 randperm"
            )
        # coverage: the reassembled order is a permutation of every index exactly once
        assert sorted(cp2) == list(range(N)), f"epoch {epoch}: cp>1 must cover every index exactly once"


# ── regression sanity: the BUGGY cp>1 (hardcoded seed) DIFFERS from cp=1 (data_seed) ──
def test_buggy_cp2_diverges_from_cp1():
    # If this ever passes (buggy == cp1) the parity test above would be vacuous.
    for epoch in EPOCHS:
        cp1 = _cp1_global_order(N, resolve(SEED, DATA_SEED), epoch)  # data_seed=1234
        buggy = _reconstruct_global(
            _cp_per_rank_orders(N, DP, SEED, epoch, sampler="distributed"),
            DP,  # BUG: seeded with seed=42
        )
        assert buggy != cp1, (
            f"epoch {epoch}: buggy cp>1 (seed) must differ from cp=1 (data_seed) -- test is meaningful"
        )


# ── already-good case: data_seed is None -> resolve to seed, so cp1 == cp2 ──
def test_data_seed_none_already_matches():
    s = resolve(SEED, None)
    assert s == SEED, "data_seed=None must fall back to seed"
    for epoch in EPOCHS:
        cp1 = _cp1_global_order(N, s, epoch)
        cp2 = _reconstruct_global(_cp_per_rank_orders(N, DP, s, epoch, sampler="distributed"), DP)
        assert cp2 == cp1, f"epoch {epoch}: with data_seed=None cp=1 and cp>1 already agree"


# ── the fix touches BOTH samplers: _DummyPadDistributedSampler must honor it too ──
def test_dummy_pad_sampler_also_honors_resolved_seed():
    # N % DP == 0 -> _DummyPadDistributedSampler adds no dummy padding and strides the
    # same randperm as DistributedSampler, so the resolved seed reproduces cp=1 and the
    # buggy `seed` diverges -- mirroring the second line changed by the fix.
    s = resolve(SEED, DATA_SEED)
    for epoch in EPOCHS:
        cp1 = _cp1_global_order(N, s, epoch)
        fixed = _reconstruct_global(_cp_per_rank_orders(N, DP, s, epoch, sampler="dummy_pad"), DP)
        buggy = _reconstruct_global(_cp_per_rank_orders(N, DP, SEED, epoch, sampler="dummy_pad"), DP)
        assert fixed == cp1, f"epoch {epoch}: dummy-pad sampler seeded with data_seed must match cp=1"
        assert buggy != cp1, f"epoch {epoch}: dummy-pad sampler seeded with the buggy seed must diverge"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(
        f"ALL SAMPLER SEED PARITY TESTS PASSED (seed={SEED}, data_seed={DATA_SEED}, N={N}, dp={DP}: "
        "cp=1 and fixed cp>1 shuffle identically once both resolve data_seed; buggy `seed` diverges)"
    )
