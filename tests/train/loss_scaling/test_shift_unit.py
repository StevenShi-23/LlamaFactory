"""Fast, single-process correctness checks for the per-sample shift / weights /
counting primitives (no distributed runtime). Asserts the packing boundary fix:
the last token of each packed sample must NOT predict the next sample's first
token.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import torch

from llamafactory.train.sft.loss_scaling import (
    IGNORE_INDEX,
    count_num_items,
    loss_weights_from_shift,
    per_sample_shift_labels,
)


def test_nonpacked_shift_is_global():
    labels = torch.tensor([[10, 11, 12, 13]])
    shift = per_sample_shift_labels(labels, cu_seqlens=None)
    assert shift.tolist() == [[11, 12, 13, IGNORE_INDEX]], shift.tolist()


def test_packed_shift_respects_boundaries():
    # two samples of length 3 packed: [a0 a1 a2 | b0 b1 b2]
    labels = torch.tensor([[10, 11, 12, 20, 21, 22]])
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    shift = per_sample_shift_labels(labels, cu_seqlens=cu)
    # within sample A: 10->11, 11->12, 12->IGNORE (must NOT predict b0=20)
    # within sample B: 20->21, 21->22, 22->IGNORE
    assert shift.tolist() == [[11, 12, IGNORE_INDEX, 21, 22, IGNORE_INDEX]], shift.tolist()


def test_packed_equals_concat_of_per_sample_shifts():
    s1 = torch.tensor([[5, 6, 7, 8]])
    s2 = torch.tensor([[1, 2, 3, 4]])
    packed = torch.cat([s1, s2], dim=1)
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    got = per_sample_shift_labels(packed, cu)
    want = torch.cat([per_sample_shift_labels(s1), per_sample_shift_labels(s2)], dim=1)
    assert torch.equal(got, want), (got, want)


def test_per_sample_weights_sum_to_num_samples():
    # 2 samples, different valid counts after shift
    labels = torch.tensor([[IGNORE_INDEX, 11, 12, 20, 21, 22]])
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    shift = per_sample_shift_labels(labels, cu)  # [11,12,IGN, 21,22,IGN]
    w = loss_weights_from_shift(shift, "per_sample", cu)
    # each sample's weights sum to 1 -> total == num samples (2)
    assert abs(w.sum().item() - 2.0) < 1e-6, w
    # per_token weights are the valid mask
    wt = loss_weights_from_shift(shift, "per_token", cu)
    assert wt.sum().item() == (shift != IGNORE_INDEX).sum().item()


def test_counts():
    labels = torch.tensor([[IGNORE_INDEX, 11, 12, 20, 21, 22]])
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    # shift -> [11,12,IGN,21,22,IGN] : 4 valid tokens, 2 samples
    assert int(count_num_items(labels, "per_token", cu)) == 4
    assert int(count_num_items(labels, "per_sample", cu)) == 2
    # a sample that becomes all-ignore must not be counted
    labels2 = torch.tensor([[30, IGNORE_INDEX, 12, 13]])  # one sample, len4
    cu2 = torch.tensor([0, 1, 4], dtype=torch.int32)  # seg0=[30] -> shift IGN (len1), seg1 valid
    assert int(count_num_items(labels2, "per_sample", cu2)) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print("ALL SHIFT UNIT TESTS PASSED")
