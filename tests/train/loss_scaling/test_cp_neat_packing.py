# Copyright 2025 the LlamaFactory team.
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

"""CPU-only checks for the CP boundary-aware-packing invariant.

Covers ``llamafactory.hparams.parser.validate_cp_packing``, the parse-time guard
that makes boundary-aware (block-diagonal) attention COMPULSORY under context
parallelism by HARD-ERRORING on plain ``packing`` without ``neat_packing``.

The invariant (the footgun being fixed): under CP, plain ``packing`` (without
``neat_packing``) carries NO per-sample boundaries in the batch -- the collated
attention mask is all-ones -- so the Ulysses attention would silently attend
across packed samples (cross-sample leak). Rather than silently rewriting the
config, the parser REJECTS that combination at arg-parse time with a
``ValueError``; the user must set ``neat_packing: true`` or disable packing.
Non-CP behavior and the no-packing path are unchanged (nothing is rejected).

No GPU / no torch.distributed runtime needed (pure boolean logic). The Triton
kernel-selection path itself (trainer.py) is GPU-only and is NOT exercised here.
``pytest`` is required (``pytest.raises``); the file is also directly runnable:
``python test_cp_neat_packing.py``.
"""

from __future__ import annotations

import os
import sys

import pytest


os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU even on a GPU box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from llamafactory.hparams.parser import validate_cp_packing


def test_cp_plain_packing_is_rejected():
    # THE FOOTGUN, now a HARD ERROR: packing on, neat_packing off, CP on. Plain packing
    # emits an all-ones mask (no per-sample boundaries), so the CP varlen kernel cannot
    # isolate packed samples -> cross-sample leak. Rejected at parse time, never promoted.
    for cp in (2, 4):
        with pytest.raises(ValueError):
            validate_cp_packing(context_parallel_size=cp, packing=True, neat_packing=False)


def test_cp_neat_packing_is_allowed():
    # Already boundary-aware: the batch carries per-sample boundaries, so this is fine
    # under CP (with packing explicitly on, or left as the stage default None).
    assert validate_cp_packing(context_parallel_size=4, packing=True, neat_packing=True) is None
    assert validate_cp_packing(context_parallel_size=2, packing=None, neat_packing=True) is None


def test_cp_no_packing_is_allowed():
    # No packing under CP: one real sample per sequence, no boundaries to cross, so there
    # is nothing to leak across -- accepted regardless of neat_packing.
    assert validate_cp_packing(context_parallel_size=4, packing=None, neat_packing=False) is None
    assert validate_cp_packing(context_parallel_size=4, packing=False, neat_packing=False) is None


def test_non_cp_plain_packing_is_allowed():
    # CP off (size 1): plain packing without neat_packing is still allowed -- the leak is a
    # CP-only concern, so this guard must NOT touch non-CP behavior.
    assert validate_cp_packing(context_parallel_size=1, packing=True, neat_packing=False) is None
    assert validate_cp_packing(context_parallel_size=1, packing=True, neat_packing=True) is None
    assert validate_cp_packing(context_parallel_size=1, packing=None, neat_packing=False) is None


def test_error_message_is_actionable():
    # The rejection must tell the user how to fix it (set neat_packing or disable packing)
    # and why (the CP varlen kernel cannot isolate plain-packed samples).
    with pytest.raises(ValueError) as excinfo:
        validate_cp_packing(context_parallel_size=2, packing=True, neat_packing=False)
    msg = str(excinfo.value)
    assert "neat_packing" in msg
    assert "context_parallel_size" in msg
    assert "triton_gqa_varlen_ulysses" in msg


if __name__ == "__main__":
    test_cp_plain_packing_is_rejected()
    test_cp_neat_packing_is_allowed()
    test_cp_no_packing_is_allowed()
    test_non_cp_plain_packing_is_allowed()
    test_error_message_is_actionable()
    print("ok")
