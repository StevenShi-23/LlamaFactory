"""CPU-only checks for the pack-counted global-batch sampler (KEEP + DUMMY-PAD).

Covers the parse-time derivation
``llamafactory.hparams.parser.derive_pack_counted_grad_accum`` and the runtime behaviour of
the pack-counted ``global_batch_size_in_packs`` feature, which reuses the EXISTING dummy-pad machinery
``llamafactory.train.sft.trainer._DummyPadDistributedSampler`` (method-1 padding) combined with
the derived ``gradient_accumulation_steps = N // dp``.

With offline packing one dataset row == one pack, so ``global_batch_size_in_packs`` (``N``) packs per
optimizer step means each epoch runs exactly ``ceil(P / N)`` steps: the first ``P // N`` steps
are full ``N``-pack global batches and the trailing partial batch keeps its ``rem = P % N`` real
packs. Every real pack is therefore trained EXACTLY ONCE per epoch -- nothing is dropped and
nothing is duplicated. The ``pack_last_batch_pad`` flag selects how the trailing partial batch is
padded (both modes -> ``ceil(P/N)`` steps, every real sample once; they differ only in dummies):
  * ``"dp"`` (default): pad up to dp-divisibility only (``ceil(P/dp)*dp - P`` dummies, minimal;
    last step is ``ceil(rem/dp)*dp`` packs). pad_to_multiple defaults to ``dp``.
  * ``"global"``: pad up to a full ``N`` packs (``ceil(P/N)*N - P`` dummies; last step is ``N``
    packs, Megatron-style uniform N). Implemented via ``_DummyPadDistributedSampler(pad_to_multiple=N)``.

Math (mirrors how HF groups micro-batches): ``_DummyPadDistributedSampler`` builds ONE global
order ``[shuffled P reals] + [dummies-at-end]`` (same seed on every rank) and shards it
``order[rank::dp]``; HF groups ``ga = N/dp`` consecutive micro-batches per optimizer step, so the
union of each rank's window ``[k*ga:(k+1)*ga]`` is global positions ``[k*N:(k+1)*N)``. Hence
``ceil(ceil(P/dp)/(N/dp)) = ceil(P/N)`` steps, every real pack once, dummies (== ``dummy_index``)
contribute 0 ``num_items``. No bespoke sampler is needed.

No GPU and no ``torch.distributed`` runtime are needed -- the sampler takes an explicit
``num_replicas`` / ``rank``. pytest may be absent, so the file is also directly runnable:
``python test_pack_counted_sampler.py``.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter


os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU even on a GPU box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from llamafactory.hparams.parser import derive_pack_counted_grad_accum
from llamafactory.train.sft.trainer import _DummyPadDistributedSampler


# (P packs, N global_batch_size_in_packs in PACKS, dp replicas); N % dp == 0 is a feature precondition.
# Cover P % N != 0 (partial last batch), P % dp != 0 (real dummies), exact, and P < N.
PACK_CASES = [
    (200, 64, 8),   # P%N=8 !=0, P%dp=0  -> partial last batch, 0 dummies (rem divisible by dp)
    (130, 64, 8),   # P%N=2 !=0, P%dp=2  -> partial last batch, 6 dummies
    (192, 64, 8),   # exact: P%N=0       -> all full batches, 0 dummies, no partial step
    (40, 64, 8),    # P < N              -> single (partial) step, 0 dummies
    (131, 64, 4),   # cp peer (dp=4): P%N=3, P%dp=3 -> partial last batch, 1 dummy
    (100, 24, 4),   # P%N=4 !=0, P%dp=0  -> partial last batch, 0 dummies
]


def _per_rank_orders(
    p: int, dp: int, dummy_index: int, shuffle: bool, seed: int, epoch: int, pad_to_multiple: int | None = None
) -> list[list[int]]:
    """Per-rank index streams from the REAL dummy-pad sampler (one per dp replica).

    ``pad_to_multiple=None`` -> pad-to-dp (default); ``pad_to_multiple=N`` -> pad-to-N (global mode).
    """
    out: list[list[int]] = []
    for r in range(dp):
        s = _DummyPadDistributedSampler(
            num_real=p, num_replicas=dp, rank=r, dummy_index=dummy_index, shuffle=shuffle, seed=seed,
            pad_to_multiple=pad_to_multiple,
        )
        s.set_epoch(epoch)
        out.append(list(s))
    return out


def _optimizer_step_batches(per_rank: list[list[int]], ga: int) -> list[list[int]]:
    """Reconstruct each optimizer step's GLOBAL batch.

    HF accumulates ``ga`` consecutive micro-batches per rank per optimizer step, so step ``k``'s
    global batch is the union across ranks of each rank's window ``[k*ga:(k+1)*ga]``. All ranks
    have the same number of micro-batches (collective symmetry), so the last window is equally
    short on every rank.
    """
    num_samples = len(per_rank[0])
    assert all(len(x) == num_samples for x in per_rank), [len(x) for x in per_rank]
    steps = math.ceil(num_samples / ga)
    batches: list[list[int]] = []
    for k in range(steps):
        batch: list[int] = []
        for r in range(len(per_rank)):
            batch.extend(per_rank[r][k * ga : (k + 1) * ga])  # slicing auto-caps the final window
        batches.append(batch)
    return batches


def _assert_pack_counted(p: int, n: int, dp: int, shuffle: bool, seed: int, epoch: int, pad_mode: str = "dp") -> None:
    """Assert the keep + dummy-pad behavior for one (P, N, dp) case under ``pad_mode``.

    Both modes run ``ceil(P/N)`` steps with identical full steps and train every real pack
    exactly once; they differ ONLY in the trailing partial batch:
      * ``"dp"``    : last step == ``ceil(rem/dp)*dp`` packs (minimal dummies, ``ceil(P/dp)*dp - P``).
      * ``"global"``: last step == a full ``N`` packs (``ceil(P/N)*N - P`` dummies).
    """
    assert n % dp == 0, f"feature precondition: N({n}) % dp({dp}) == 0"
    assert pad_mode in ("dp", "global")
    ga = n // dp
    # ga must equal the parse-time derivation (cp=1 -> world_size==dp, micro==1).
    assert ga == derive_pack_counted_grad_accum(n, world_size=dp, context_parallel_size=1, per_device_train_batch_size=1)

    dummy_index = p  # _AllIgnoreDummyDataset.dummy_index == num_real == P
    pad_to_multiple = n if pad_mode == "global" else None  # global -> pad up to a full N
    per_rank = _per_rank_orders(p, dp, dummy_index, shuffle=shuffle, seed=seed, epoch=epoch, pad_to_multiple=pad_to_multiple)
    batches = _optimizer_step_batches(per_rank, ga)

    full = p // n
    rem = p % n
    expected_steps = math.ceil(p / n)
    pad_mult = n if pad_mode == "global" else dp
    expected_dummies = math.ceil(p / pad_mult) * pad_mult - p  # 0 when P % pad_mult == 0

    # (1) exactly ceil(P / N) optimizer steps -- SAME in both modes.
    assert len(batches) == expected_steps, (pad_mode, p, n, dp, len(batches), expected_steps)

    # (2) steps 0 .. P//N-1 are FULL N real-pack global batches (no dummies, no repeats) -- SAME in both modes.
    for k in range(full):
        b = batches[k]
        assert len(b) == n, (pad_mode, p, n, dp, k, len(b))
        assert all(0 <= i < p for i in b), f"{pad_mode}: step {k} must be all real packs"
        assert len(set(b)) == n, f"{pad_mode}: step {k} must have no repeated pack"

    # (3) the trailing partial batch keeps `rem` real packs + all-ignore dummies (count varies by mode).
    if rem != 0:  # rem == P when P < N (full == 0) -> the single step is the partial one
        last = batches[-1]
        reals = [i for i in last if i != dummy_index]
        dummies = [i for i in last if i == dummy_index]
        assert len(reals) == rem, (pad_mode, p, n, dp, len(reals), rem)
        assert len(set(reals)) == rem, "rem real packs in the last batch must be distinct"
        assert all(i == dummy_index for i in dummies), "dummy entries must equal dummy_index (== P)"
        assert len(dummies) == expected_dummies, (pad_mode, p, n, dp, len(dummies), expected_dummies)
        if pad_mode == "global":
            assert len(last) == n, f"global: last step must be a full N batch, got {len(last)}"
            assert len(dummies) == n - rem, (p, n, dp, len(dummies), n - rem)
        else:
            assert len(last) == math.ceil(rem / dp) * dp, "dp: last step size == ceil(rem/dp)*dp (<= N)"
            assert len(last) <= n
    else:
        assert len(batches) == full, "exact P%N==0 -> no partial step"

    # (4) every REAL pack appears EXACTLY once across ranks (none dropped, none duplicated) -- SAME in both modes.
    real_cov = Counter(i for r in per_rank for i in r if i != dummy_index)
    assert len(real_cov) == p, (pad_mode, p, n, dp, len(real_cov))
    assert set(real_cov.values()) == {1}, sorted(set(real_cov.values()))
    # total dummies emitted across ranks == the pad count for this mode.
    total_dummies = sum(1 for r in per_rank for i in r if i == dummy_index)
    assert total_dummies == expected_dummies, (pad_mode, p, n, dp, total_dummies, expected_dummies)
    # collective symmetry: every rank runs the same number of micro-batches (and thus steps).
    assert len({len(r) for r in per_rank}) == 1, [len(r) for r in per_rank]


# ── ceil(P/N) steps, full N-pack batches, rem+dummies last, every real pack once (shuffle on/off) ──
def test_pack_counted_keep_and_dummy_pad_all_cases():
    for (p, n, dp) in PACK_CASES:
        for shuffle in (False, True):
            _assert_pack_counted(p, n, dp, shuffle=shuffle, seed=1, epoch=0)


# ════════════════════ pad-to-N: pack_last_batch_pad="global" (selectable option) ════════════════════
# Same ceil(P/N) steps and every real pack exactly once, but the trailing partial batch is padded
# up to a FULL N packs (rem real + (N-rem) dummy) instead of only to dp-divisibility. Reuses the
# generalized _DummyPadDistributedSampler via the optional pad_to_multiple=N.

# ── global mode, all cases: ceil(P/N) steps, full-N last batch, every real pack once (shuffle on/off) ──
def test_pack_counted_global_mode_all_cases():
    for (p, n, dp) in PACK_CASES:
        for shuffle in (False, True):
            _assert_pack_counted(p, n, dp, shuffle=shuffle, seed=1, epoch=0, pad_mode="global")


# ── global mode: the last step is a full N batch = rem real + (N-rem) dummy ──
def test_global_mode_last_batch_is_full_n():
    p, n, dp = 130, 64, 8
    ga = n // dp
    per_rank = _per_rank_orders(p, dp, p, shuffle=True, seed=3, epoch=0, pad_to_multiple=n)
    batches = _optimizer_step_batches(per_rank, ga)
    assert len(batches) == math.ceil(p / n) == 3
    rem = p % n  # 2
    last = batches[-1]
    assert len(last) == n, "global mode pads the last step to a full N packs"
    assert sum(1 for i in last if i != p) == rem
    assert sum(1 for i in last if i == p) == n - rem  # 62 dummies
    total_dummies = sum(1 for r in per_rank for i in r if i == p)
    assert total_dummies == math.ceil(p / n) * n - p == n - rem == 62
    real_cov = Counter(i for r in per_rank for i in r if i != p)
    assert len(real_cov) == p and set(real_cov.values()) == {1}


# ── KEY: global mode needs dummies even when P % dp == 0 but P % N != 0 (dp mode needs none) ──
def test_global_mode_needs_dummies_when_p_div_dp_not_n():
    p, n, dp = 200, 64, 8  # 200 % 8 == 0 (dp-divisible) but 200 % 64 == 8 != 0
    dp_dummies = math.ceil(p / dp) * dp - p
    assert dp_dummies == 0, "dp mode: rem already divisible by dp -> zero dummies"
    global_dummies = math.ceil(p / n) * n - p
    assert global_dummies == 56, "global mode: pad last batch up to a full N -> dummies required"
    per_rank = _per_rank_orders(p, dp, p, shuffle=True, seed=9, epoch=0, pad_to_multiple=n)
    assert sum(1 for r in per_rank for i in r if i == p) == global_dummies
    # both modes: still ceil(P/N) steps and every real pack exactly once
    _assert_pack_counted(p, n, dp, shuffle=True, seed=9, epoch=0, pad_mode="dp")
    _assert_pack_counted(p, n, dp, shuffle=True, seed=9, epoch=0, pad_mode="global")


# ── generalized sampler shapes: global pads to ceil(P/N)*N (divisible by dp); default still pads to dp ──
def test_global_mode_sampler_shapes_and_backward_compat():
    p, n, dp = 131, 64, 4
    g = _DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=0, dummy_index=p, shuffle=False, pad_to_multiple=n)
    assert g.total_size == math.ceil(p / n) * n == 192
    assert g.total_size % dp == 0, "total_size must stay divisible by num_replicas"
    assert g.num_samples == g.total_size // dp == 48
    assert len(list(g)) == g.num_samples
    # default (pad_to_multiple=None) is byte-identical to the original pad-to-dp behavior
    d = _DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=0, dummy_index=p, shuffle=False)
    assert d.total_size == math.ceil(p / dp) * dp == 132
    assert d.num_samples == math.ceil(p / dp) == 33
    # explicitly passing pad_to_multiple=num_replicas matches the default
    e = _DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=0, dummy_index=p, shuffle=False, pad_to_multiple=dp)
    assert (e.total_size, e.num_samples) == (d.total_size, d.num_samples)


# ── global mode preserves cp peer-sharing: same dp rank identical; diff dp ranks disjoint reals ──
def test_global_mode_cp_peer_sharing():
    p, n, dp = 130, 64, 4
    a = list(_DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=2, dummy_index=p, shuffle=True, seed=5,
                                         pad_to_multiple=n))
    b = list(_DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=2, dummy_index=p, shuffle=True, seed=5,
                                         pad_to_multiple=n))
    assert a == b, "CP peers (same dp rank) must receive identical streams in global mode"
    reals = [
        {i for i in _DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=r, dummy_index=p, shuffle=True,
                                                seed=5, pad_to_multiple=n) if i != p}
        for r in range(dp)
    ]
    for i in range(dp):
        for j in range(i + 1, dp):
            assert reals[i].isdisjoint(reals[j]), f"dp ranks {i},{j} must hold disjoint real packs"
    assert len(set().union(*reals)) == p, "every real pack covered exactly once across dp ranks"


# ── P < N is valid now (no raise): exactly one step covering every real pack once ──
def test_p_less_than_n_yields_single_step():
    p, n, dp = 40, 64, 8
    ga = n // dp
    per_rank = _per_rank_orders(p, dp, p, shuffle=True, seed=2, epoch=0)
    batches = _optimizer_step_batches(per_rank, ga)
    assert len(batches) == 1 == math.ceil(p / n)
    real_cov = Counter(i for r in per_rank for i in r if i != p)
    assert len(real_cov) == p and set(real_cov.values()) == {1}


# ── real dummies appear only in the final partial batch when P % dp != 0 ──
def test_dummies_only_in_final_partial_batch():
    p, n, dp = 130, 64, 8  # P%dp=2 -> ceil(130/8)*8-130 = 6 dummies
    ga = n // dp
    per_rank = _per_rank_orders(p, dp, p, shuffle=True, seed=4, epoch=0)
    batches = _optimizer_step_batches(per_rank, ga)
    expected_dummies = math.ceil(p / dp) * dp - p
    assert expected_dummies == 6
    for b in batches[:-1]:  # every full step is dummy-free
        assert all(i != p for i in b), "no dummy may appear before the final partial batch"
    last_dummies = sum(1 for i in batches[-1] if i == p)
    assert last_dummies == expected_dummies
    assert sum(1 for i in batches[-1] if i != p) == p % n == 2  # rem real packs


# ── reproducible by (seed, epoch); per-epoch reshuffle changes order but keeps exactly-once ──
def test_reshuffle_changes_order_preserves_exactly_once():
    p, n, dp = 130, 64, 8
    e0 = _per_rank_orders(p, dp, p, shuffle=True, seed=7, epoch=0)
    e0_again = _per_rank_orders(p, dp, p, shuffle=True, seed=7, epoch=0)
    e1 = _per_rank_orders(p, dp, p, shuffle=True, seed=7, epoch=1)
    assert e0 == e0_again, "same (seed, epoch) must reproduce the exact per-rank order"
    assert e0 != e1, "different epoch must reshuffle the order (shuffle on)"
    for e in (e0, e1):  # exactly-once preserved across epochs
        real_cov = Counter(i for r in e for i in r if i != p)
        assert len(real_cov) == p and set(real_cov.values()) == {1}
        _assert_pack_counted(p, n, dp, shuffle=True, seed=7, epoch=0)


# ── cp>1 peer-sharing: same dp rank -> identical stream; different dp ranks -> disjoint reals ──
def test_cp_peer_sharing_same_dp_rank_identical_diff_disjoint():
    p, dp = 130, 4  # e.g. world=8, cp=2 -> dp=4 (peer-sharing is independent of N)
    peer_a = list(_DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=1, dummy_index=p, shuffle=True, seed=5))
    peer_b = list(_DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=1, dummy_index=p, shuffle=True, seed=5))
    assert peer_a == peer_b, "CP peers (same dp rank) must receive identical packs"

    reals = [
        {i for i in _DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=r, dummy_index=p, shuffle=True, seed=5)
         if i != p}
        for r in range(dp)
    ]
    for i in range(dp):
        for j in range(i + 1, dp):
            assert reals[i].isdisjoint(reals[j]), f"dp ranks {i},{j} must hold disjoint real packs"
    union: set[int] = set().union(*reals)
    assert len(union) == p, "every real pack covered exactly once across dp ranks"


# ── parse-time derivation: divisibility + micro!=1 hard-errors, and valid derivations ──
def test_derive_grad_accum_hard_errors_and_values():
    # micro != 1 -> error
    try:
        derive_pack_counted_grad_accum(64, world_size=8, context_parallel_size=1, per_device_train_batch_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("per_device_train_batch_size != 1 must raise")

    # N indivisible by dp_size (world=8, cp=1 -> dp=8; 60 % 8 != 0) -> error
    try:
        derive_pack_counted_grad_accum(60, world_size=8, context_parallel_size=1, per_device_train_batch_size=1)
    except ValueError:
        pass
    else:
        raise AssertionError("global_batch_size_in_packs indivisible by dp_size must raise")

    # non-sft stage -> error
    try:
        derive_pack_counted_grad_accum(64, 8, 1, 1, stage="dpo")
    except ValueError:
        pass
    else:
        raise AssertionError("non-sft stage must raise")

    # valid derivations: gas = N // (dp_size * micro)
    assert (
        derive_pack_counted_grad_accum(64, world_size=8, context_parallel_size=2, per_device_train_batch_size=1) == 16
    )
    assert (
        derive_pack_counted_grad_accum(64, world_size=8, context_parallel_size=1, per_device_train_batch_size=1) == 8
    )
    assert (
        derive_pack_counted_grad_accum(64, world_size=1, context_parallel_size=1, per_device_train_batch_size=1) == 64
    )


# ── guard: the batch is counted in PACKS, so packing must be enabled (else a clear hard-error) ──
def test_derive_grad_accum_requires_packing():
    # packing off -> ValueError even for an otherwise-valid (divisible, micro==1, sft) config.
    try:
        derive_pack_counted_grad_accum(
            64, world_size=8, context_parallel_size=1, per_device_train_batch_size=1, packing=False
        )
    except ValueError as e:
        assert "pack" in str(e).lower(), f"error must explain packing is required, got: {e}"
    else:
        raise AssertionError("`global_batch_size_in_packs` set without packing must raise")

    # packing on (the default) -> the same config derives gradient_accumulation_steps normally.
    assert (
        derive_pack_counted_grad_accum(
            64, world_size=8, context_parallel_size=1, per_device_train_batch_size=1, packing=True
        )
        == 8
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(
        "ALL PACK-COUNTED SAMPLER TESTS PASSED (keep + dummy-pad; both pad modes: ceil(P/N) steps, "
        "full N-pack batches, every sample exactly once, cp peer-sharing -- 'dp' pads last to "
        "ceil(rem/dp)*dp, 'global' pads last to a full N)"
    )
