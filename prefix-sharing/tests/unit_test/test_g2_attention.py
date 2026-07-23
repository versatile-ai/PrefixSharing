"""Tests for B2 — G2 attention store/expand helper functions.

Covers _g2_store_with_kwargs, _g2_store_per_sequence.
Mac-testable — uses synthetic tensors and mock context.

NOTE: Tests using _g2_expand_attn_output removed (function deleted in v2 design).
Will be rewritten in Task 2 with _g2_kv_store_or_expand.
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION,
    G2AttentionStore,
    PrefixActivationSlotId,
)
from prefix_sharing.integrations.g2_attention import (
    _g2_store_per_sequence,
    _g2_store_with_kwargs,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_plan(*, batch_size, prefix_lens, original_lengths):
    """Build a PrefixSharingPlan with structural metadata patched in."""
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner

    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    input_ids = [list(range(s)) for s in original_lengths]
    plan = planner.plan(input_ids)

    kept_lengths_q = [ol - pl for ol, pl in zip(original_lengths, prefix_lens)]
    provider_index = [-1] * batch_size
    for i, pl in enumerate(prefix_lens):
        if pl > 0:
            provider_index[i] = 0

    object.__setattr__(plan, "batch_size", batch_size)
    object.__setattr__(plan, "original_lengths", list(original_lengths))
    object.__setattr__(plan, "prefix_lens", list(prefix_lens))
    object.__setattr__(plan, "kept_lengths_q", kept_lengths_q)
    object.__setattr__(plan, "provider_index", provider_index)
    return plan


@dataclass
class MockContext:
    """Minimal context duck-type carrying store + layout + plan."""

    store: G2AttentionStore
    layout: PackedBatchLayout
    plan: object


def _slot_id(plan, layer_id=3, batch_idx=0, tp_rank=0):
    """Build a slot ID using the plan's actual forward_id / micro_batch_id."""
    return PrefixActivationSlotId(
        plan.forward_id, plan.micro_batch_id, layer_id,
        batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank,
    )


# ── _g2_store_with_kwargs ────────────────────────────────────────────


def test_g2_store_with_kwargs():
    """StoredG2Activation → store → load round-trip preserves all fields."""
    from prefix_sharing.core.prefix_store import StoredG2Activation

    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[5])
    slot = _slot_id(plan)
    kv = torch.randn(5, 512, requires_grad=True)
    cmp = torch.randn(0, 512)
    data = StoredG2Activation(kv=kv, kv_compress=cmp, stored_len=5)

    _g2_store_with_kwargs(store, slot, data)

    loaded = store.load(slot)
    assert loaded.kv is kv
    assert loaded.kv_compress is cmp
    assert loaded.stored_len == 5
    assert loaded.kv.requires_grad  # autograd preserved


# ── _g2_store_per_sequence ───────────────────────────────────────────


def test_g2_store_per_sequence():
    """Provider's valid kv prefix stored; non-provider skipped."""
    store = G2AttentionStore()
    # seq0: provider, valid_len=6; seq1: non-provider, valid_len=4
    # padded: seq0=6, seq1=4 → total=10
    plan = _make_plan(batch_size=2, prefix_lens=[0, 3], original_lengths=[6, 8])
    layout = PackedBatchLayout.from_valid_lengths([6, 4])
    ctx = MockContext(store=store, layout=layout, plan=plan)

    kv = torch.arange(10 * 512, dtype=torch.float32).reshape(10, 512)

    _g2_store_per_sequence(ctx, layout, plan, layer_id=3, tp_rank=0, field="kv", tensor=kv)

    # seq0 (provider, batch_idx=0): should be stored
    slot0 = _slot_id(plan, batch_idx=0)
    assert store.contains(slot0)
    entry0 = store.load(slot0)
    assert entry0.kv.shape == (6, 512)  # valid_len=6
    assert torch.equal(entry0.kv[:, 0], torch.arange(6, dtype=torch.float32) * 512)

    # seq1 (non-provider, batch_idx=1): should NOT be stored
    slot1 = _slot_id(plan, batch_idx=1)
    assert not store.contains(slot1)


def test_g2_store_per_sequence_none_tensor():
    """None tensor → no-op, no store calls."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 0], original_lengths=[6, 4])
    layout = PackedBatchLayout.from_valid_lengths([6, 4])
    ctx = MockContext(store=store, layout=layout, plan=plan)

    _g2_store_per_sequence(ctx, layout, plan, layer_id=3, tp_rank=0, field="kv_compress", tensor=None)

    assert store.size == 0


# NOTE: incremental store with compressed fields (kv_compress, indexer_k)
# requires _compute_cmp_lengths for per-sequence split.  Will be covered
# in Task 2 (_g2_kv_store_or_expand tests).


# NOTE: _g2_expand_attn_output and transitive reuse tests removed — function
# deleted in v2 design. Will be rewritten in Task 2 with _g2_kv_store_or_expand.
