"""Tests for B2 — G2 attention store/expand helper functions.

Covers _g2_store_with_kwargs, _g2_store_per_sequence, _g2_expand_attn_output.
Mac-testable — uses synthetic tensors and mock context.
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
    _g2_expand_attn_output,
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


def test_g2_store_per_sequence_incremental():
    """Same slot called twice (kv → attn_o) preserves both fields.

    Uses kv + attn_o (same seqlen) because kv_compress has different
    per-sequence lengths (valid_len // ratio).  CMP KV incremental store
    will be tested in B3 when _compute_cmp_lengths is integrated.
    """
    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[7])
    layout = PackedBatchLayout.from_valid_lengths([7])
    ctx = MockContext(store=store, layout=layout, plan=plan)

    kv = torch.randn(7, 512)
    attn_o = torch.randn(7, 4, 512)

    _g2_store_per_sequence(ctx, layout, plan, layer_id=3, tp_rank=0, field="kv", tensor=kv)
    _g2_store_per_sequence(ctx, layout, plan, layer_id=3, tp_rank=0, field="attn_o", tensor=attn_o)

    slot = _slot_id(plan, batch_idx=0)
    entry = store.load(slot)
    assert entry.kv.shape == (7, 512)
    assert entry.attn_o.shape == (7, 4, 512)


# ── _g2_expand_attn_output ───────────────────────────────────────────


def test_g2_expand_attn_output():
    """Reuser attn_o expanded with provider prefix."""
    store = G2AttentionStore()
    from prefix_sharing.core.prefix_store import StoredG2Activation

    # Provider seq0: valid=6, stored attn_o with prefix available
    # Reuser seq1: valid=4, suffix-only attn_o, prefix_len=3
    plan = _make_plan(batch_size=2, prefix_lens=[0, 3], original_lengths=[6, 7])
    layout = PackedBatchLayout.from_valid_lengths([6, 4])
    ctx = MockContext(store=store, layout=layout, plan=plan)

    # Pre-store provider attn_o (seq0, len=6, n_heads=4, head_dim=16)
    provider_attn = torch.randn(6, 4, 16)
    slot0 = _slot_id(plan, batch_idx=0)
    _g2_store_with_kwargs(store, slot0, StoredG2Activation(attn_o=provider_attn, stored_len=6))

    # Suffix-only attn_o from suffix forward
    # seq0: valid=6 → provider keeps its own
    # seq1: valid=4 → reuser needs expansion
    suffix_o = torch.cat([torch.randn(6, 4, 16), torch.randn(4, 4, 16)], dim=0)

    result = _g2_expand_attn_output(ctx, layout, plan, layer_id=3, tp_rank=0, o=suffix_o)

    # After expansion:
    # seq0: 6 (unchanged)
    # seq1: 3 (prefix) + 4 (suffix) = 7
    assert result.shape[0] == 6 + 7  # 13

    # seq0 prefix part unchanged
    assert torch.equal(result[:6], suffix_o[:6])

    # seq1: first 3 from provider prefix
    assert torch.equal(result[6:9], provider_attn[:3])
    # seq1: last 4 from suffix
    assert torch.equal(result[9:13], suffix_o[6:10])


def test_transitive_reuse_store():
    """Reuser A's expanded attn_o stored for Reuser B to use."""
    store = G2AttentionStore()
    from prefix_sharing.core.prefix_store import StoredG2Activation

    # Chain: seq0(provider) → seq1(reuser A) → seq2(reuser B, shares seq1's prefix)
    # seq0: valid=8, provider
    # seq1: valid=3, reuser of seq0, prefix_len=5
    # seq2: valid=2, reuser of seq1, prefix_len=6 (from seq1's expanded P+S)
    plan = _make_plan(batch_size=3, prefix_lens=[0, 5, 6], original_lengths=[8, 8, 8])
    layout = PackedBatchLayout.from_valid_lengths([8, 3, 2])
    ctx = MockContext(store=store, layout=layout, plan=plan)

    # Store provider attn_o (seq0)
    provider_attn = torch.randn(8, 4, 16)
    _g2_store_with_kwargs(store, _slot_id(plan, batch_idx=0),
                          StoredG2Activation(attn_o=provider_attn, stored_len=8))

    # Pre-create slot for seq1 (simulating Hook C which stores kv/kv_compress).
    # Without this, _g2_expand_attn_output's transitive reuse is skipped.
    store.store(_slot_id(plan, batch_idx=1), stored_len=5)

    # Suffix-only attn_o
    suffix_o = torch.cat([
        torch.randn(8, 4, 16),  # seq0 (provider)
        torch.randn(3, 4, 16),  # seq1 (reuser A suffix)
        torch.randn(2, 4, 16),  # seq2 (reuser B suffix)
    ], dim=0)

    result = _g2_expand_attn_output(ctx, layout, plan, layer_id=3, tp_rank=0, o=suffix_o)

    # seq1 expanded: 5 + 3 = 8
    # seq2 expanded: 6 + 2 = 8
    assert result.shape[0] == 8 + 8 + 8  # 24

    # Verify seq1's expanded attn_o was stored (transitive reuse)
    slot1 = _slot_id(plan, batch_idx=1)
    assert store.contains(slot1)
    expanded_seq1 = store.load(slot1)
    assert expanded_seq1.attn_o.shape == (8, 4, 16)
    assert expanded_seq1.stored_len == 8
