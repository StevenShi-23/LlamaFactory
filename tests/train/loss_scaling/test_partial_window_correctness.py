"""CPU-only proof that the pad-to-dp smaller last optimizer step is loss-normalized correctly.

The infra under test is the EXISTING per-GA-window loss-normalization
(``num_items_in_batch`` counting + ``_scale_loss``).

Setup under test (pad-to-dp, pack-counted ``global_batch_size_in_packs``): an epoch runs
``ceil(P/N)`` optimizer steps. The first ``P//N`` are full ``N``-pack windows; the
trailing window has ``rem = P % N`` REAL packs plus all-ignore DUMMY packs up to
``ceil(rem/dp)*dp`` (``_DummyPadDistributedSampler`` method-1 padding). When
``rem < dp`` some ranks hold ONLY a dummy pack (local count 0, local sum 0).

The claim verified here (per_token AND per_sample): the accumulated, world-averaged
gradient of that smaller window equals the TRUE per-token / per-sample mean over
just the ``rem`` real packs -- identical to a full window with the same real packs
and no dummies, and independent of how many dummies were added or that some ranks
hold only a dummy.

Why it holds, straight from the infra (line refs are the current working tree):

  - ``loss_scaling.count_num_items`` (loss_scaling.py L104-131): counts on the
    per-sample *shifted* labels -- per_token = valid shifted tokens (L119-120),
    per_sample = segments/rows with >=1 valid shifted token (L124-131). An
    all-IGNORE pack shifts to all-IGNORE -> ``valid`` all False -> 0.

  - ``CustomSeq2SeqTrainer._get_num_items_in_batch`` (trainer.py L650-687): sums
    ``count_num_items`` over the window's micro-batches (L664-677), then (only when
    ``dist`` is initialized) all-reduce SUM across the world and ``/cp`` (L682-687).
    => ``Z`` for the window = real tokens/samples only; dummies add 0; the smaller
    window simply yields a smaller ``Z``. In the no-dist single-process branch the
    sum is returned verbatim (no all-reduce, no ``/cp``) -- the branch hit below.

  - ``CustomSeq2SeqTrainer._scale_loss`` (trainer.py L689-720):
    ``grad_loss = local_weighted_sum / Z * world`` (L715). HF skips its
    ``/gradient_accumulation_steps`` when ``num_items`` is set, so the per-microbatch
    weighted-CE sums ADD across the GA window; DeepSpeed/DDP then averages grads by
    ``1/world`` only (L694-698, doc L341-349). Net optimizer-visible gradient over a
    window across all ranks = ``(1/Z) d/dθ Σ_t w_t·ce_t`` -- independent of (dp, cp,
    ga) and of any dummy packs (which contribute 0 to both ``Σ w·ce`` and ``Z``).

No GPU and no ``torch.distributed`` runtime are needed: the GA window / DP ranks are
simulated explicitly (the per-rank window count uses the REAL
``_get_num_items_in_batch`` no-dist branch; the cross-rank all-reduce SUM and the
DeepSpeed ``1/world`` average are modeled per the cited lines). pytest may be absent,
so the file is also directly runnable: ``python test_partial_window_correctness.py``.
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace


os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU even on a GPU box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import torch

from llamafactory.train.sft.loss_scaling import (
    IGNORE_INDEX,
    count_num_items,
    loss_weights_from_shift,
    per_sample_shift_labels,
)
from llamafactory.train.sft.trainer import (
    DUMMY_PACK_LEN,
    CustomSeq2SeqTrainer,
    _AllIgnoreDummyDataset,
    _DummyPadDistributedSampler,
)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal building blocks
# ──────────────────────────────────────────────────────────────────────────────
class _RangeDataset:
    """item == index (base for _AllIgnoreDummyDataset; len>=1)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return i


def _cu_from(batch: dict):
    """Recover cu_seqlens exactly as the trainer does, for production-faithful counts.

    Uses ``CustomSeq2SeqTrainer._cu_seqlens_from_batch`` (trainer.py L631-648, a
    @staticmethod), so tests count over the SAME sample boundaries as production.
    """
    return CustomSeq2SeqTrainer._cu_seqlens_from_batch(batch)


def _make_fake_trainer(reduction: str):
    """Build the smallest object that drives the REAL trainer methods (no-dist branch).

    Drives ``_get_num_items_in_batch`` / ``_scale_loss`` through their single-process
    (no-dist) branch. Only the attributes those two methods touch are provided:
      _get_num_items_in_batch: _use_loss_scaling (L656), loss_reduction (L676),
                               _cu_seqlens_from_batch (L675), cp_group (L684).
      _scale_loss            : current_gradient_accumulation_steps + args
                               .gradient_accumulation_steps (L708).
    """
    fake = SimpleNamespace()
    fake._use_loss_scaling = True
    fake.loss_reduction = reduction
    fake.cp_group = None
    fake._cu_seqlens_from_batch = CustomSeq2SeqTrainer._cu_seqlens_from_batch  # staticmethod -> plain fn
    fake.current_gradient_accumulation_steps = None
    fake.args = SimpleNamespace(gradient_accumulation_steps=1)
    return fake


def _real_window_Z(reduction: str, micro_batches: list[dict]) -> float:
    """Compute Z for one window via the REAL ``_get_num_items_in_batch`` no-dist branch.

    See trainer.py L650-687. Returns 0.0 for an all-dummy window (count, not None).
    """
    fake = _make_fake_trainer(reduction)
    z = CustomSeq2SeqTrainer._get_num_items_in_batch(fake, micro_batches)
    return 0.0 if z is None else float(z)


# ── micro-batch (one pack) builders ───────────────────────────────────────────
def _single_sample_mb(ce_values: list[float]) -> dict:
    """Make one non-packed pack whose SHIFTED valid tokens carry the given CE values.

    1 row == 1 sample. labels=[IGNORE, t, t, ...] (len k+1) -> after the global
    left-shift the first ``k`` positions are valid, last is IGNORE. ``ce`` is aligned
    to the shifted positions (CE on the k valid, 0 on the trailing IGNORE).
    """
    k = len(ce_values)
    labels = torch.tensor([[IGNORE_INDEX] + [5] * k])          # (1, k+1)
    attention_mask = torch.ones((1, k + 1), dtype=torch.long)  # plain 0/1 -> cu is None
    ce = torch.tensor([list(ce_values) + [0.0]], dtype=torch.float32)
    return {"labels": labels, "attention_mask": attention_mask, "ce": ce}


def _dummy_mb() -> dict:
    """Make the REAL all-ignore dummy pad pack the sampler emits.

    Via ``_AllIgnoreDummyDataset.__getitem__`` (trainer.py L86-98): all-IGNORE labels,
    plain all-ones mask, DUMMY_PACK_LEN tokens. CE is all-zero (a dummy never
    contributes loss).
    """
    ds = _AllIgnoreDummyDataset(_RangeDataset(1), pad_token_id=0)
    item = ds[ds.dummy_index]
    labels = torch.tensor([item["labels"]])
    attention_mask = torch.tensor([item["attention_mask"]])
    ce = torch.zeros((1, len(item["labels"])), dtype=torch.float32)
    return {"labels": labels, "attention_mask": attention_mask, "ce": ce}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  count_num_items correctness  (loss_scaling.py L104-131)
# ──────────────────────────────────────────────────────────────────────────────
def test_count_per_token_counts_valid_shifted_tokens():
    # labels=[IGN,5,5,5] -> shift=[5,5,IGN] (global drop-last) -> 3 valid tokens.
    labels = torch.tensor([[IGNORE_INDEX, 5, 5, 5]])
    assert int(count_num_items(labels, "per_token")) == 3
    # one valid token: labels=[IGN,5] -> shift=[5,IGN] -> 1.
    assert int(count_num_items(torch.tensor([[IGNORE_INDEX, 5]]), "per_token")) == 1


def test_count_per_sample_counts_samples_with_a_valid_token():
    # 1 row with >=1 valid shifted token -> 1 sample.
    labels = torch.tensor([[IGNORE_INDEX, 5, 5, 5]])
    assert int(count_num_items(labels, "per_sample")) == 1


def test_count_all_ignore_pack_is_zero_both_modes():
    labels = torch.full((1, DUMMY_PACK_LEN), IGNORE_INDEX)
    assert int(count_num_items(labels, "per_token")) == 0
    assert int(count_num_items(labels, "per_sample")) == 0


def test_count_real_dummy_pack_is_zero():
    """The actual dummy pad pack the sampler emits counts as 0 items (both modes)."""
    mb = _dummy_mb()
    cu = _cu_from(mb)
    assert cu is None  # plain 0/1 mask -> not neat-packing -> one short sample (L643-644)
    assert int(count_num_items(mb["labels"], "per_token", cu)) == 0
    assert int(count_num_items(mb["labels"], "per_sample", cu)) == 0


def test_count_mixed_multirow_batch():
    # 3 rows: valid, all-ignore, valid. per_token = 2 + 0 + 1 = 3; per_sample = 2 rows.
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 5, 5],          # shift [5,5,IGN] -> 2 valid
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],  # all ignore -> 0
            [IGNORE_INDEX, 7, IGNORE_INDEX],  # shift [7,IGN,IGN] -> 1 valid
        ]
    )
    assert int(count_num_items(labels, "per_token")) == 3
    assert int(count_num_items(labels, "per_sample")) == 2


def test_count_with_cu_seqlens_neat_packing():
    # neat-packing: 1 row, sample ids [1,1,1,2,2,2,2] -> cu=[0,3,7]; 2 samples packed.
    # seg0=[IGN,5,5]    -> shift[5,5,IGN]   -> 2 valid
    # seg1=[IGN,5,5,5]  -> shift[5,5,5,IGN] -> 3 valid
    labels = torch.tensor([[IGNORE_INDEX, 5, 5, IGNORE_INDEX, 5, 5, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2]])
    cu = _cu_from({"labels": labels, "attention_mask": attention_mask})
    assert cu.tolist() == [0, 3, 7]
    assert int(count_num_items(labels, "per_token", cu)) == 5      # 2 + 3
    assert int(count_num_items(labels, "per_sample", cu)) == 2     # both segments valid
    # a packed sample that shifts to all-ignore is NOT counted (per_sample).
    labels2 = torch.tensor([[IGNORE_INDEX, 9, 9, 9, 9]])  # sample ids [1, 2,2,2,2]
    am2 = torch.tensor([[1, 2, 2, 2, 2]])
    cu2 = _cu_from({"labels": labels2, "attention_mask": am2})
    assert cu2.tolist() == [0, 1, 5]            # seg0 len1 -> shift IGN (0 valid); seg1 -> 3 valid
    assert int(count_num_items(labels2, "per_token", cu2)) == 3
    assert int(count_num_items(labels2, "per_sample", cu2)) == 1   # only seg1 counts


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Z over a GA window ignores dummies  (real _get_num_items_in_batch, L650-687)
# ──────────────────────────────────────────────────────────────────────────────
def test_window_Z_ignores_dummy_packs():
    """A window of [real packs + dummy packs] yields Z == the real-only count.

    Holds for both reductions, using the REAL _get_num_items_in_batch (no-dist branch).
    """
    real = [_single_sample_mb([4.0, 8.0]), _single_sample_mb([2.0, 2.0, 8.0]), _single_sample_mb([10.0])]
    dummies = [_dummy_mb(), _dummy_mb()]

    # per_token: real valid tokens = 2 + 3 + 1 = 6.
    assert _real_window_Z("per_token", real) == 6.0
    assert _real_window_Z("per_token", real + dummies) == 6.0           # dummies add nothing
    assert _real_window_Z("per_token", dummies + real) == 6.0           # order-independent

    # per_sample: 3 real samples.
    assert _real_window_Z("per_sample", real) == 3.0
    assert _real_window_Z("per_sample", real + dummies) == 3.0

    # an all-dummy window counts to 0 (the clamp_min in _scale_loss absorbs 0/0).
    assert _real_window_Z("per_token", dummies) == 0.0
    assert _real_window_Z("per_sample", dummies) == 0.0


def test_smaller_window_yields_smaller_Z():
    """The trailing (smaller) window's Z is exactly the count over its real packs."""
    full = [_single_sample_mb([1.0]) for _ in range(4)] + [_single_sample_mb([1.0, 1.0])]
    # full window: 4 packs of 1 valid token + 1 pack of 2 = 6 tokens, 5 samples.
    assert _real_window_Z("per_token", full) == 6.0
    assert _real_window_Z("per_sample", full) == 5.0
    # a smaller trailing window of just the last 2 real packs (+ a dummy):
    smaller = full[-2:] + [_dummy_mb()]
    assert _real_window_Z("per_token", smaller) == 3.0   # 1 + 2 real tokens
    assert _real_window_Z("per_sample", smaller) == 2.0  # 2 real samples


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Normalization invariance (the core claim)
# ──────────────────────────────────────────────────────────────────────────────
def _scaled_grad_term(local_weighted_sum, Z: float, world: int):
    """Mirror ``_scale_loss``'s gradient term verbatim: ``lws / clamp_min(Z, 1) * world``.

    See trainer.py L711-715. (The display identity trick on L719-720 is
    grad-transparent, so it is irrelevant to the optimizer step; see
    test_scale_loss_matches_real_method_world1 which checks both against the real
    method.)
    """
    denom = max(float(Z), 1.0)
    return local_weighted_sum / denom * world


def _simulate_optimizer_grad(
    ranks: list[list[dict]],
    reduction: str,
    world: int,
    *,
    use_global_Z: bool = True,
    use_world_factor: bool = True,
) -> float:
    """Replay one optimizer step over a GA window split across ``dp`` ranks.

    Returns the FINAL optimizer-visible gradient w.r.t a scalar θ. Mirrors the
    production data flow exactly:
      * Z_global = Σ_rank (rank-window count via the REAL _get_num_items_in_batch,
        L676-677) -- the all-reduce SUM in L682-683 with cp=1 (L684-686 -> /1).
      * Each micro-batch's loss = _scale_loss(local_weighted_sum, Z_global) =
        local_weighted_sum / Z_global * world (L715); HF SUMS them over the window
        (num_items set -> no /gas), realized here as repeated .backward() into one
        per-rank θ (one model replica per rank).
      * DeepSpeed/DDP averages grads across the world by 1/world (doc L341-349,
        L694-698): final = (1/world) Σ_rank θ_rank.grad.

    The surrogate ``local_weighted_sum = θ · Σ_t w_t·ce_t`` is linear in θ, so its
    gradient is exactly ``Σ_t w_t·ce_t`` -- isolating the normalization math from any
    model. ``w_t`` and ``Z`` come from the real primitives (loss_scaling.py).

    The two flags are negative controls (default off): ``use_global_Z=False`` divides
    by each rank's LOCAL window count (i.e. no all-reduce); ``use_world_factor=False``
    drops the ``* world`` of L715.
    """
    per_rank_Z = [_real_window_Z(reduction, mbs) for mbs in ranks]
    Z_global = sum(per_rank_Z)

    final = 0.0
    for mbs, local_Z in zip(ranks, per_rank_Z):
        Z = Z_global if use_global_Z else local_Z
        theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
        for mb in mbs:
            cu = _cu_from(mb)
            shift = per_sample_shift_labels(mb["labels"], cu)
            w = loss_weights_from_shift(shift, reduction, cu).to(torch.float64)
            local_weighted_sum = theta * (w * mb["ce"].to(torch.float64)).sum()
            loss = _scaled_grad_term(local_weighted_sum, Z, world if use_world_factor else 1)
            loss.backward()
        final += float(theta.grad) if theta.grad is not None else 0.0
    return final / world  # DeepSpeed 1/world average across the world


def _true_mean_grad(packs: list[dict], reduction: str) -> float:
    """Compute the objective's TRUE gradient over the real packs: ``(Σ_t w_t·ce_t) / Z``.

    Linear in θ. Computed straight from the real primitives over the real packs only
    -- the value every layout must reproduce.
    """
    num = 0.0
    Z = 0.0
    for pk in packs:
        cu = _cu_from(pk)
        shift = per_sample_shift_labels(pk["labels"], cu)
        w = loss_weights_from_shift(shift, reduction, cu).to(torch.float64)
        num += float((w * pk["ce"].to(torch.float64)).sum())
        Z += float(count_num_items(pk["labels"], reduction, cu))
    return num / max(Z, 1.0)


def _pad_to_dp_layout(real_packs: list[dict], dp: int) -> list[list[dict]]:
    """Build the per-rank micro-batch lists for ONE pad-to-dp window.

    Exactly as ``_DummyPadDistributedSampler`` does (trainer.py L130-144): global
    order = [real packs] + [dummy]*pad, pad = ceil(rem/dp)*dp - rem (minimal),
    sharded ``order[rank::dp]``. ga = ceil(rem/dp) micro-batches per rank.
    """
    rem = len(real_packs)
    total = math.ceil(rem / dp) * dp
    order = list(real_packs) + [_dummy_mb() for _ in range(total - rem)]
    return [order[r::dp] for r in range(dp)]


# The 3 real packs shared by all layouts (single-sample, distinct valid-token counts
# so per_token != per_sample). Known SHIFTED-token CE values; valid counts are powers
# of 2 (2, 4, 1) so the per_sample weight 1/v_s is EXACT in fp32 (loss_scaling.py L85
# computes weights in float32), keeping the hand-true value exact:
#   pack0: [4, 8]        (v=2, sum 12, mean 6)
#   pack1: [2, 2, 4, 8]  (v=4, sum 16, mean 4)
#   pack2: [10]          (v=1, sum 10, mean 10)
# per_token : Z=7, Σce=38            -> true mean = 38/7
# per_sample: Z=3, Σmeans=6+4+10=20  -> true mean = 20/3
_PACK_CE = [[4.0, 8.0], [2.0, 2.0, 4.0, 8.0], [10.0]]
_HAND_TRUE = {"per_token": 38.0 / 7.0, "per_sample": 20.0 / 3.0}


def _check_invariance(reduction: str) -> None:
    real_packs = [_single_sample_mb(ce) for ce in _PACK_CE]
    rem = len(real_packs)  # 3

    # The primitive-computed truth (the value every layout must reproduce), cross-
    # checked against the by-hand rational. Both agree to float64 eps because the
    # chosen valid counts make the fp32 weights exact.
    true_mean = _true_mean_grad(real_packs, reduction)
    assert abs(true_mean - _HAND_TRUE[reduction]) < 1e-9, (reduction, true_mean, _HAND_TRUE[reduction])

    # (a) full window: rem real packs alone on a single rank (dp=1, ga=rem, world=1).
    grad_a = _simulate_optimizer_grad([list(real_packs)], reduction, world=1)

    # (b) pad-to-dp window: rem real + dummies up to ceil(rem/dp)*dp (dp=2 -> 1 dummy).
    dp_b = 2
    ranks_b = _pad_to_dp_layout(real_packs, dp_b)
    assert sum(len(r) for r in ranks_b) == math.ceil(rem / dp_b) * dp_b  # 4 micro-batches (3 real + 1 dummy)
    grad_b = _simulate_optimizer_grad(ranks_b, reduction, world=dp_b)

    # (c) rem < dp edge: dp=4 > rem=3, 1 micro-batch/rank, the dp-rem=1 extra rank
    #     holds ONLY a dummy (local Z=0, local sum=0). world=4.
    dp_c = 4
    ranks_c = _pad_to_dp_layout(real_packs, dp_c)
    assert [len(r) for r in ranks_c] == [1, 1, 1, 1]                 # 1 micro-batch per rank
    assert _real_window_Z(reduction, ranks_c[3]) == 0.0             # rank 3 is dummy-only
    grad_c = _simulate_optimizer_grad(ranks_c, reduction, world=dp_c)

    # The core assertion: all three layouts reproduce the TRUE mean over the real packs.
    for tag, g in (("full", grad_a), ("pad-to-dp", grad_b), ("rem<dp", grad_c)):
        assert abs(g - true_mean) < 1e-9, f"{reduction} {tag}: {g} != true {true_mean}"


def test_scale_loss_matches_real_method_world1():
    """Ground the simulator against the REAL ``_scale_loss`` for world=1 (no dist).

    Checks ``_scaled_grad_term`` and the display identity trick both equal
    ``CustomSeq2SeqTrainer._scale_loss``.
    """
    fake = _make_fake_trainer("per_token")  # gas=1
    Z = torch.tensor(4.0)

    lws = torch.tensor(7.0, requires_grad=True)
    real = CustomSeq2SeqTrainer._scale_loss(fake, lws, Z)            # world=1 -> 7/4
    assert abs(float(real.detach()) - 7.0 / 4.0) < 1e-12
    assert abs(float(real.detach()) - float(_scaled_grad_term(lws.detach(), 4.0, world=1))) < 1e-12

    # display path: value reads as disp/gas, gradient still flows as d(scaled).
    lws2 = torch.tensor(7.0, requires_grad=True)
    disp = torch.tensor(1.25)
    real_disp = CustomSeq2SeqTrainer._scale_loss(fake, lws2, Z, display_value=disp)
    assert abs(float(real_disp.detach()) - 1.25) < 1e-12            # disp/gas, gas=1
    real_disp.backward()
    assert abs(float(lws2.grad) - 1.0 / 4.0) < 1e-12               # d(lws/Z*world)=1/Z, world=1

    # clamp_min(1.0) guards an all-dummy (Z=0) window: 0/0 -> finite.
    z0 = CustomSeq2SeqTrainer._scale_loss(fake, torch.tensor(0.0), torch.tensor(0.0))
    assert float(z0) == 0.0


def test_normalization_invariance_per_token():
    _check_invariance("per_token")


def test_normalization_invariance_per_sample():
    _check_invariance("per_sample")


def test_rem_less_than_dp_edge_with_negative_controls():
    """Spell out the subtle case: rem=3 real packs across dp=4 ranks (rank 3 dummy-only).

    The correct global-Z + world-scaling recovers the true mean; naive variants
    (local-Z, or dropping the world factor) do NOT -- proving the test has teeth and
    that BOTH ingredients are required.
    """
    for reduction in ("per_token", "per_sample"):
        real_packs = [_single_sample_mb(ce) for ce in _PACK_CE]
        true_mean = _true_mean_grad(real_packs, reduction)
        assert abs(true_mean - _HAND_TRUE[reduction]) < 1e-9
        ranks = _pad_to_dp_layout(real_packs, dp=4)

        # dummy-only rank really is local-empty (0 count, and 0 weighted sum).
        assert _real_window_Z(reduction, ranks[3]) == 0.0

        # CORRECT: global Z (all-reduce SUM) + world factor.
        correct = _simulate_optimizer_grad(ranks, reduction, world=4)
        assert abs(correct - true_mean) < 1e-9, (reduction, correct, true_mean)

        # NEGATIVE CONTROL 1: per-rank LOCAL Z (no all-reduce) -> wrong (sum of
        # per-pack means, not the global token/sample mean).
        local_z = _simulate_optimizer_grad(ranks, reduction, world=4, use_global_Z=False)
        assert abs(local_z - true_mean) > 1e-3, (reduction, "local-Z unexpectedly correct", local_z)

        # NEGATIVE CONTROL 2: drop the *world factor of L715 -> off by 1/world.
        no_world = _simulate_optimizer_grad(ranks, reduction, world=4, use_world_factor=False)
        assert abs(no_world - true_mean / 4.0) < 1e-9, (reduction, no_world)
        assert abs(no_world - true_mean) > 1e-3


def test_pad_to_dp_window_invariant_across_dp_for_fixed_real_packs():
    """Same rem real packs across many (dp, #dummies) layouts -> identical gradient.

    The pad-to-dp window is invariant to dp and to the dummy count.
    """
    for reduction in ("per_token", "per_sample"):
        real_packs = [_single_sample_mb(ce) for ce in _PACK_CE]  # rem=3
        true_mean = _true_mean_grad(real_packs, reduction)
        grads = []
        for dp in (1, 2, 3, 4, 5):  # dp=3 -> 0 dummies; dp in {2,4,5} -> dummies; dp=5>rem
            ranks = _pad_to_dp_layout(real_packs, dp)
            grads.append(_simulate_optimizer_grad(ranks, reduction, world=dp))
        for dp, g in zip((1, 2, 3, 4, 5), grads):
            assert abs(g - true_mean) < 1e-9, (reduction, dp, g, true_mean)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Sampler tied to loss: last window's real packs == trailing rem; Z matches.
# ──────────────────────────────────────────────────────────────────────────────
def _per_rank_orders(p: int, dp: int, dummy_index: int) -> list[list[int]]:
    """Per-rank index streams from the REAL dummy-pad sampler (shuffle off).

    Shuffle off -> deterministic order [0..P-1] + dummies.
    """
    return [
        list(_DummyPadDistributedSampler(num_real=p, num_replicas=dp, rank=r, dummy_index=dummy_index, shuffle=False))
        for r in range(dp)
    ]


def _optimizer_step_batches(per_rank: list[list[int]], ga: int) -> list[list[int]]:
    """Reconstruct each optimizer step's global batch from the per-rank streams.

    Union over ranks of each rank's window ``[k*ga:(k+1)*ga]`` (the same grouping HF
    uses; cf. the sibling pack-counted test).
    """
    num_samples = len(per_rank[0])
    steps = math.ceil(num_samples / ga)
    return [
        [i for r in range(len(per_rank)) for i in per_rank[r][k * ga : (k + 1) * ga]]
        for k in range(steps)
    ]


def test_last_window_real_packs_are_trailing_rem_and_Z_matches():
    """Tie the sampler to the loss: the last window's real packs are the trailing rem.

    End-to-end-ish: ``_DummyPadDistributedSampler`` (P, dp) + ga=N/dp. The last
    optimizer step's REAL packs are exactly the trailing ``rem`` packs, and feeding
    their labels through ``count_num_items`` gives the window's expected Z. Includes
    the rem<dp edge (P=10, N=8, dp=4 -> rem=2 < dp).
    """
    # pack i (real) carries (i % 3) + 1 valid shifted tokens -> known per_token count.
    def real_pack(i: int) -> dict:
        return _single_sample_mb([1.0] * ((i % 3) + 1))

    for (P, N, dp) in [(10, 8, 4), (11, 8, 2), (200, 64, 8)]:
        assert N % dp == 0, "feature precondition"
        ga = N // dp
        rem = P % N
        full = P // N
        dummy_index = P  # _AllIgnoreDummyDataset.dummy_index == num_real == P

        per_rank = _per_rank_orders(P, dp, dummy_index)
        batches = _optimizer_step_batches(per_rank, ga)
        assert len(batches) == math.ceil(P / N)

        last = batches[-1]
        real_idx = sorted(i for i in last if i != dummy_index)
        # the last window's real packs are EXACTLY the trailing rem packs [P-rem, P).
        assert real_idx == list(range(full * N, P)), (P, N, dp, real_idx)
        assert len(real_idx) == rem

        # rem < dp edge: at least one rank in the last window holds ONLY a dummy.
        if rem < dp:
            last_per_rank = [per_rank[r][(len(batches) - 1) * ga : len(batches) * ga] for r in range(dp)]
            dummy_only = [r for r in range(dp) if all(i == dummy_index for i in last_per_rank[r])]
            assert len(dummy_only) == dp - rem, (P, N, dp, dummy_only)

        # Feed the trailing real packs' labels through the loss counter -> window Z.
        last_real_mbs = [real_pack(i) for i in real_idx]
        expected_pt = sum((i % 3) + 1 for i in real_idx)  # per_token tokens
        assert _real_window_Z("per_token", last_real_mbs) == float(expected_pt)
        assert _real_window_Z("per_sample", last_real_mbs) == float(rem)  # rem samples
        # adding the window's dummies does not change Z.
        n_dummies = math.ceil(rem / dp) * dp - rem
        with_dummies = last_real_mbs + [_dummy_mb() for _ in range(n_dummies)]
        assert _real_window_Z("per_token", with_dummies) == float(expected_pt)
        assert _real_window_Z("per_sample", with_dummies) == float(rem)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(
        "ALL PARTIAL-WINDOW CORRECTNESS TESTS PASSED "
        "(count_num_items; window-Z ignores dummies; per_token & per_sample "
        "normalization invariant across full / pad-to-dp / rem<dp layouts; "
        "sampler's last window == trailing rem real packs -> expected Z)"
    )
