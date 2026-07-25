"""Unit tests for align_prefix_lens_to_compression."""

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import (
    PrefixSharingPlanner,
    align_prefix_lens_to_compression,
)


def test_no_op_for_no_compress():
    """compress_ratios empty or None → no changes."""
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])

    orig_prefix = list(plan.prefix_lens)
    orig_kept = list(plan.kept_lengths_q)
    orig_cu = list(plan.cu_seqlens_q)
    orig_ranges = list(plan.input_keep_ranges)

    align_prefix_lens_to_compression(plan, [])

    assert plan.prefix_lens == orig_prefix
    assert plan.kept_lengths_q == orig_kept
    assert plan.cu_seqlens_q == orig_cu
    assert plan.input_keep_ranges == orig_ranges


def test_no_op_for_ratio_le_1():
    """compress_ratios all ≤ 1 → no changes."""
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([[1, 2, 3, 4], [1, 2, 3, 5]])

    orig_prefix = list(plan.prefix_lens)
    align_prefix_lens_to_compression(plan, [0, 1])

    assert plan.prefix_lens == orig_prefix


def test_align_prefix_rounds_down():
    """P=130, r=128 → P=128."""
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    # 2 seqs, common prefix [1,2,3,...,130] of len 130
    seq0 = list(range(200))
    seq1 = list(range(130)) + [999 + i for i in range(70)]

    plan = planner.plan([seq0, seq1])
    # Manually check prefix_lens from trie detection
    assert plan.prefix_lens[1] == 130  # seq1 shares first 130 tokens with seq0

    align_prefix_lens_to_compression(plan, [128])

    assert plan.prefix_lens[1] == 128  # rounded down
    assert plan.prefix_lens[0] == 0    # provider unchanged


def test_align_prefix_rebuilds_kept_lengths():
    """P=130→128, kept_lengths_q: seq1 from 70→72."""
    seq0 = list(range(200))
    seq1 = list(range(130)) + [999 + i for i in range(70)]  # original_len=200
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([seq0, seq1])

    # seq0 (provider): original=200, P=0, kept=200
    # seq1 (reuser): original=200, P=130, kept=70
    assert plan.kept_lengths_q[0] == 200
    assert plan.kept_lengths_q[1] == 70

    align_prefix_lens_to_compression(plan, [128])

    # After align: P=128 → seq1 kept = 200-128 = 72
    assert plan.kept_lengths_q[0] == 200  # unchanged
    assert plan.kept_lengths_q[1] == 72   # updated


def test_align_prefix_rebuilds_keep_ranges():
    """P=130→128, input_keep_ranges for reuser: (130,200)→(128,200)."""
    seq0 = list(range(200))
    seq1 = list(range(130)) + [999 + i for i in range(70)]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([seq0, seq1])

    # seq1 (reuser) keep range: (P, original) = (130, 200)
    assert plan.input_keep_ranges[1] == (130, 200)

    align_prefix_lens_to_compression(plan, [128])

    assert plan.input_keep_ranges[1] == (128, 200)  # start moved earlier
    assert plan.label_keep_ranges[1] == (128, 200)
    assert plan.loss_mask_keep_ranges[1] == (128, 200)


def test_align_prefix_rebuilds_cu_seqlens_q():
    """P=130→128, cu_seqlens_q reflects new kept_lengths."""
    seq0 = list(range(200))
    seq1 = list(range(130)) + [999 + i for i in range(70)]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([seq0, seq1])

    # Before: cumsum([200, 70]) = [0, 200, 270]
    assert plan.cu_seqlens_q == [0, 200, 270]

    align_prefix_lens_to_compression(plan, [128])

    # After: cumsum([200, 72]) = [0, 200, 272]
    assert plan.cu_seqlens_q == [0, 200, 272]


def test_align_prefix_multiple_ratios():
    """Max ratio (128) used for alignment."""
    seq0 = list(range(20))
    seq1 = list(range(6)) + [99 + i for i in range(14)]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([seq0, seq1])

    # P=6, max_ratio=128, 128>6 → rounds to 0
    align_prefix_lens_to_compression(plan, [0, 4, 128])

    assert plan.prefix_lens[1] == 0  # rounded down to nearest multiple of 128


def test_already_aligned_unchanged():
    """P=128, r=128 → no change."""
    seq0 = list(range(256))
    seq1 = list(range(128)) + [999 + i for i in range(128)]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([seq0, seq1])

    # P=128 already aligned
    assert plan.prefix_lens[1] == 128

    orig_kept = list(plan.kept_lengths_q)
    align_prefix_lens_to_compression(plan, [128])

    assert plan.prefix_lens[1] == 128  # unchanged
    assert plan.kept_lengths_q == orig_kept  # all derived fields unchanged
