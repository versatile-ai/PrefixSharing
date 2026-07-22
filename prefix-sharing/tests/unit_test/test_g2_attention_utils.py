"""Tests for B1 — G2 attention backend utility functions.

Covers _split_by_cu_seqlens, _compute_cmp_lengths, _adjust_cu_seqlens_for_batch.
Pure PyTorch — no MindSpeed dependency.
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.g2_attention_utils import (
    _adjust_cu_seqlens_for_batch,
    _compute_cmp_lengths,
    _split_by_cu_seqlens,
)
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.planner import PrefixSharingPlan


# ── helpers ──────────────────────────────────────────────────────────


def _make_plan(*, batch_size, prefix_lens, original_lengths):
    """Build a minimal PrefixSharingPlan with structural metadata set.

    The planner requires input_ids for trie detection; we bypass that by
    constructing a plan and patching its structural fields directly.
    """
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner

    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    # Use dummy input_ids — structural fields are overridden below.
    input_ids = [list(range(s)) for s in original_lengths]
    plan = planner.plan(input_ids)

    kept_lengths_q = [ol - pl for ol, pl in zip(original_lengths, prefix_lens)]
    provider_index = [-1] * batch_size
    for i, pl in enumerate(prefix_lens):
        if pl > 0:
            provider_index[i] = 0  # all reusers share seq0

    object.__setattr__(plan, "batch_size", batch_size)
    object.__setattr__(plan, "original_lengths", list(original_lengths))
    object.__setattr__(plan, "prefix_lens", list(prefix_lens))
    object.__setattr__(plan, "kept_lengths_q", kept_lengths_q)
    object.__setattr__(plan, "provider_index", provider_index)

    return plan


@dataclass
class MockPackedSeqParams:
    """Minimal Megatron packed_seq_params compatible dataclass."""

    cu_seqlens_kv: list[int]
    cu_seqlens_cmp_kv: list[int] | None = None


@dataclass
class MockPackedSeqParamsPadded:
    """Megatron variant with cu_seqlens_kv_padded (MindSpeed >= 2.x)."""

    cu_seqlens_kv_padded: list[int]
    cu_seqlens_kv: list[int]  # both present
    cu_seqlens_cmp_kv: list[int] | None = None


# ── _split_by_cu_seqlens ─────────────────────────────────────────────


def test_split_by_cu_seqlens_basic():
    """Split a packed tensor and verify per-sequence padded shapes."""
    # 2 sequences: seq0 padded_len=6, seq1 padded_len=4 → total=10
    padded_lengths = [6, 4]
    tensor = torch.arange(10 * 8, dtype=torch.float32).reshape(10, 8)

    rows = _split_by_cu_seqlens(tensor, padded_lengths)
    assert len(rows) == 2
    assert rows[0].shape == (6, 8)
    assert rows[1].shape == (4, 8)
    # Verify content: first row contains first 6 positions
    assert torch.equal(rows[0][:, 0], torch.arange(6, dtype=torch.float32) * 8)


def test_split_by_cu_seqlens_length_mismatch_raises():
    """Raise ValueError when sum(padded_lengths) != tensor.shape[0]."""
    padded_lengths = [5, 5]  # sum=10
    tensor = torch.zeros(9, 8)  # length=9 ≠ 10

    with pytest.raises(ValueError):
        _split_by_cu_seqlens(tensor, padded_lengths)


def test_split_by_cu_seqlens_empty_list_returns_empty():
    """Empty lengths → empty list."""
    assert _split_by_cu_seqlens(torch.zeros(0, 8), []) == []


# ── _compute_cmp_lengths ─────────────────────────────────────────────


def test_compute_cmp_lengths_basic():
    """Compressed lengths = valid_lengths // compress_ratio."""
    layout = PackedBatchLayout.from_valid_lengths([256, 128])
    lengths = _compute_cmp_lengths(layout, compress_ratio=128, kv_compress_shape_0=3)
    # 256//128=2, 128//128=1
    assert lengths == [2, 1]


def test_compute_cmp_lengths_mismatch_raises():
    """Assertion fails when computed sum != actual kv_compress dim=0."""
    layout = PackedBatchLayout.from_valid_lengths([256, 128])
    # computed = 2+1=3, but we pass 5 → mismatch
    with pytest.raises(AssertionError):
        _compute_cmp_lengths(layout, compress_ratio=128, kv_compress_shape_0=5)


def test_compute_cmp_lengths_non_divisible_raises():
    """Non-divisible valid_length raises assertion (compressor must align)."""
    layout = PackedBatchLayout.from_valid_lengths([255])  # 255 // 128 = 1
    # computed sum = 1, but actual kv_compress may be 2 if compressor pads
    # Pass a mismatched value to trigger the assert.
    with pytest.raises(AssertionError):
        _compute_cmp_lengths(layout, compress_ratio=128, kv_compress_shape_0=2)


# ── _adjust_cu_seqlens_for_batch ─────────────────────────────────────


def test_adjust_cu_seqlens_basic():
    """Reuser offset correctly shifts subsequent cu_seqlens_kv entries."""
    # 3 sequences: provider(8), reuser(5), provider(6)
    # prefix_lens:  [0, 3, 0]  →  reuser seq1 has 3 prefix tokens
    plan = _make_plan(
        batch_size=3,
        prefix_lens=[0, 3, 0],
        original_lengths=[8, 8, 6],
    )
    params = MockPackedSeqParams(
        cu_seqlens_kv=[0, 8, 13, 19],  # padded: seq0=8, seq1=5, seq2=6
    )

    result = _adjust_cu_seqlens_for_batch(params, plan, compress_ratio=128)

    # After adjustment: seq1's prefix_len=3 →  all subsequent +3
    # seq0: 0-8 (unchanged), seq1: 8-13+3=16, seq2: 13+3=16 to 19+3=22
    assert result.cu_seqlens_kv == [0, 8, 16, 22]


def test_adjust_cu_seqlens_with_cmp_kv():
    """cu_seqlens_cmp_kv also adjusted, with compress_ratio division."""
    plan = _make_plan(
        batch_size=2,
        prefix_lens=[0, 256],  # reuser seq1 has 256 prefix → cmp_offset = 256//128 = 2
        original_lengths=[256, 256],
    )
    params = MockPackedSeqParams(
        cu_seqlens_kv=[0, 256, 512],
        cu_seqlens_cmp_kv=[0, 2, 4],  # 256//128=2 per seq
    )

    result = _adjust_cu_seqlens_for_batch(params, plan, compress_ratio=128)

    # kv: seq1 offset = 256 → seq2 from 512 to 768
    assert result.cu_seqlens_kv == [0, 256, 768]
    # cmp: seq1 cmp_offset = 256//128=2 → seq2 from 4 to 6
    assert result.cu_seqlens_cmp_kv == [0, 2, 6]


def test_adjust_cu_seqlens_padded_attr_priority():
    """cu_seqlens_kv_padded takes priority over cu_seqlens_kv."""
    plan = _make_plan(
        batch_size=2,
        prefix_lens=[0, 3],
        original_lengths=[8, 8],
    )
    # Both attributes present — padded wins
    params = MockPackedSeqParamsPadded(
        cu_seqlens_kv_padded=[0, 8, 16],
        cu_seqlens_kv=[0, 8, 13],  # should be ignored
    )

    result = _adjust_cu_seqlens_for_batch(params, plan, compress_ratio=128)

    # padded version adjusted: [0, 8, 19]
    assert result.cu_seqlens_kv_padded == [0, 8, 19]


def test_adjust_cu_seqlens_none_returns_none():
    """None input → None output."""
    assert _adjust_cu_seqlens_for_batch(None, None, compress_ratio=128) is None


def test_adjust_cu_seqlens_no_reuser_unchanged():
    """No reusers → cu_seqlens unchanged."""
    plan = _make_plan(
        batch_size=2,
        prefix_lens=[0, 0],
        original_lengths=[8, 6],
    )
    params = MockPackedSeqParams(cu_seqlens_kv=[0, 8, 14])

    result = _adjust_cu_seqlens_for_batch(params, plan, compress_ratio=128)

    assert result.cu_seqlens_kv == [0, 8, 14]
    # Should be a new instance (not in-place)
    assert result is not params
