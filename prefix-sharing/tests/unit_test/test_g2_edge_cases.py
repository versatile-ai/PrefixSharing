"""Task 1 — Edge case and boundary tests for G2 attention store/expand logic.

Mac-testable — uses synthetic tensors, mock context, and mock
packed_seq_params.  Supplements the coverage in test_g2_attention.py
and test_g2_attention_utils.py.
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.g2_attention_utils import (
    _adjust_cu_seqlens_for_batch,
    _compute_cmp_lengths,
    _split_by_cu_seqlens,
)
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION,
    G2AttentionStore,
    PrefixActivationSlotId,
    StoredG2Activation,
)
from prefix_sharing.integrations.g2_attention import (
    _g2_kv_store_or_expand,
    _g2_store_with_kwargs,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_plan(*, batch_size, prefix_lens, original_lengths):
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner

    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    input_ids = [list(range(s)) for s in original_lengths]
    plan = planner.plan(input_ids)

    kept_lengths_q = [ol - pl for ol, pl in zip(original_lengths, prefix_lens)]
    provider_index = [-1] * batch_size
    is_provider = [True] * batch_size
    for i, pl in enumerate(prefix_lens):
        if pl > 0:
            provider_index[i] = 0
            is_provider[i] = False  # reuser
        # if pl == 0: stays True (provider)

    object.__setattr__(plan, "batch_size", batch_size)
    object.__setattr__(plan, "original_lengths", list(original_lengths))
    object.__setattr__(plan, "prefix_lens", list(prefix_lens))
    object.__setattr__(plan, "kept_lengths_q", kept_lengths_q)
    object.__setattr__(plan, "provider_index", provider_index)
    object.__setattr__(plan, "is_provider", is_provider)
    return plan


@dataclass
class MockContext:
    store: G2AttentionStore
    packed_batch_layout: PackedBatchLayout
    prefix_sharing_plan: object
    parallel_info: object = None

    def __post_init__(self):
        if self.parallel_info is None:
            from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
            self.parallel_info = MegatronParallelInfo()


def _slot(plan, batch_idx=0, tp_rank=0):
    return PrefixActivationSlotId(
        plan.forward_id, plan.micro_batch_id, 0,
        batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)


def _expand_and_return(ctx, kv, kv_compress=None, indexer_k=None,
                       compress_topk_idxs=None, packed_seq_params=None,
                       compress_ratio=128, attn_module=None,
                       start_pos=0, kv_allgather=False, sequence_parallel=False):
    """Shorthand for calling _g2_kv_store_or_expand."""
    result = _g2_kv_store_or_expand(
        ctx, kv, kv_compress, indexer_k,
        compress_topk_idxs, packed_seq_params,
        compress_ratio, attn_module,
        start_pos, kv_allgather, sequence_parallel)
    return result[:5]  # drop compress_topk_score


# ──────────────────────────────────────────────────────────────────────
# Group 1: multi-sequence mixed batches
# ──────────────────────────────────────────────────────────────────────


def test_multi_provider_multi_reuser():
    """2 providers + 3 reusers with cross-reuse across different provider indices.

    Layout:
      seq0: provider_A, full 512
      seq1: provider_B, full 512  (different prefix from seq0)
      seq2: reuser of seq0, P=128, S=128  (kept_length=128)
      seq3: reuser of seq0, P=256, S=128
      seq4: reuser of seq1, P=128, S=256
    """
    store = G2AttentionStore()
    batch_size = 5
    prefix_lens = [0, 0, 128, 256, 128]
    original_lengths = [512, 512, 256, 384, 384]
    plan = _make_plan(batch_size=batch_size, prefix_lens=prefix_lens,
                      original_lengths=original_lengths)
    # Override provider_index for multi-provider layout
    object.__setattr__(plan, "provider_index", [0, 1, 0, 0, 1])
    object.__setattr__(plan, "is_provider", [True, True, False, False, False])
    # kept_lengths_q: suffix lengths
    object.__setattr__(plan, "kept_lengths_q",
                       [512, 512, 128, 128, 256])

    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    ctx = MockContext(store=store, packed_batch_layout=layout,
                      prefix_sharing_plan=plan)

    # Pre-populate provider stores — these are the same tensors that
    # go into the packed kv as the providers' "forward output".
    pA_kv = torch.randn(512, 512)
    pB_kv = torch.randn(512, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=pA_kv, stored_len=512))
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=1),
                          StoredG2Activation(kv=pB_kv, stored_len=512))

    # Build packed KV — providers use the SAME tensors that are in the store.
    # (In real execution, the store is populated from the forward output,
    # so the packed kv for providers IS what gets stored.)
    suffix_kvs = [
        pA_kv,       # seq0 provider: same as pre-stored
        pB_kv,       # seq1 provider: same as pre-stored
        torch.randn(128, 512),  # seq2 reuser suffix
        torch.randn(128, 512),  # seq3 reuser suffix
        torch.randn(256, 512),  # seq4 reuser suffix
    ]
    kv = torch.cat(suffix_kvs, dim=0)

    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    # seq0: 512 (unchanged)
    # seq1: 512 (unchanged)
    # seq2: P+S = 128+128 = 256
    # seq3: P+S = 256+128 = 384
    # seq4: P+S = 128+256 = 384
    assert result_kv.shape[0] == 512 + 512 + 256 + 384 + 384

    # seq2 reuses seq0[:128]
    offset2 = 512 + 512
    assert torch.equal(result_kv[offset2:offset2 + 128], pA_kv[:128])

    # seq3 reuses seq0[:256]
    offset3 = offset2 + 256
    assert torch.equal(result_kv[offset3:offset3 + 256], pA_kv[:256])

    # seq4 reuses seq1[:128]
    offset4 = offset3 + 384
    assert torch.equal(result_kv[offset4:offset4 + 128], pB_kv[:128])

    # Verify transitive store: seq2 and seq3 store expanded data for deeper reuse
    assert store.contains(_slot(plan, batch_idx=2))
    assert store.load(_slot(plan, batch_idx=2)).stored_len == 256


def test_pass_through_sequence():
    """A sequence that is neither provider nor reuser passes through unchanged.

    This can occur when a sequence has prefix_len=0 but is_provider=False
    (e.g. after alignment rounds down a short prefix to zero).  The else
    branch in _g2_kv_store_or_expand should pass the row through.
    """
    store = G2AttentionStore()
    plan = _make_plan(batch_size=3, prefix_lens=[0, 256, 0],
                      original_lengths=[128, 256, 64])
    # seq2: not provider (is_provider=False), prefix_len=0 → not reuser either
    object.__setattr__(plan, "provider_index", [0, 0, -1])
    object.__setattr__(plan, "is_provider", [True, False, False])
    object.__setattr__(plan, "kept_lengths_q", [128, 128, 64])

    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    ctx = MockContext(store=store, packed_batch_layout=layout,
                      prefix_sharing_plan=plan)

    # Pre-populate provider
    p_kv = torch.randn(128, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=128))

    # Packed: [p(128), reuser_suffix(128), pass_through(64)] = 320
    kv = torch.cat([p_kv, torch.randn(128, 512), torch.randn(64, 512)], dim=0)
    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    # seq0: 128, seq1: 128+128=256, seq2: 64 (pass-through)
    assert result_kv.shape[0] == 128 + 256 + 64
    # seq2 should be unchanged (neither stored nor expanded)
    assert torch.equal(result_kv[128 + 256:], kv[128 + 128:])


# ──────────────────────────────────────────────────────────────────────
# Group 2: cu_seqlens boundary cases
# ──────────────────────────────────────────────────────────────────────


def test_cu_seqlens_cmp_kv_adjust():
    """cu_seqlens_cmp_kv is offset by prefix_len // compress_ratio."""
    @dataclass
    class MockPsp:
        cu_seqlens_kv: list
        cu_seqlens_cmp_kv: list

    plan = _make_plan(batch_size=2, prefix_lens=[0, 256],
                      original_lengths=[256, 256])
    object.__setattr__(plan, "kept_lengths_q", [256, 128])
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    psp = MockPsp(
        cu_seqlens_kv=[0, 256, 384],          # original: seq0=256, seq1=128
        cu_seqlens_cmp_kv=[0, 2, 3],          # cmp: 256//128=2, 128//128=1
    )

    result = _adjust_cu_seqlens_for_batch(psp, plan, compress_ratio=128)

    # seq1 is reuser with prefix_len=256
    # cu_seqlens_kv: seq1 [256:] offset by +256 → [0, 256, 640]
    assert result.cu_seqlens_kv == [0, 256, 640]
    # cu_seqlens_cmp_kv: offset by 256//128=2 → [0, 2, 5]
    assert result.cu_seqlens_cmp_kv == [0, 2, 5]


def test_cu_seqlens_kv_padded_priority():
    """When cu_seqlens_kv_padded exists, it takes priority over cu_seqlens_kv.

    g2_attention_utils.py:130 checks for cu_seqlens_kv_padded first.
    """
    @dataclass
    class MockPspBoth:
        cu_seqlens_kv: list
        cu_seqlens_kv_padded: list
        cu_seqlens_cmp_kv: list

    plan = _make_plan(batch_size=2, prefix_lens=[0, 256],
                      original_lengths=[256, 256])
    object.__setattr__(plan, "kept_lengths_q", [256, 128])
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    psp = MockPspBoth(
        cu_seqlens_kv=[0, 256, 384],          # unpadded (should be ignored)
        cu_seqlens_kv_padded=[0, 256, 384],    # padded (should be used)
        cu_seqlens_cmp_kv=[0, 2, 3],
    )

    result = _adjust_cu_seqlens_for_batch(psp, plan, compress_ratio=128)

    # cu_seqlens_kv_padded should be adjusted (priority path)
    assert result.cu_seqlens_kv_padded == [0, 256, 640]
    # cu_seqlens_kv should remain unchanged (not the priority attribute)
    assert result.cu_seqlens_kv == [0, 256, 384]
    # cu_seqlens_cmp_kv also adjusted
    assert result.cu_seqlens_cmp_kv == [0, 2, 5]

    # Verify the original object is unchanged (dataclasses.replace semantics)
    assert psp.cu_seqlens_kv_padded == [0, 256, 384]
    assert psp.cu_seqlens_kv == [0, 256, 384]


# ──────────────────────────────────────────────────────────────────────
# Group 3: error paths and empty boundaries
# ──────────────────────────────────────────────────────────────────────


def test_compute_cmp_lengths_mismatch():
    """AssertionError when computed sum of cmp lengths != actual shape[0]."""
    layout = PackedBatchLayout.from_valid_lengths([256, 128])
    # valid // 128: 256//128=2, 128//128=1 → sum=3
    # Passing shape_0=5, which ≠ 3 → should raise AssertionError
    with pytest.raises(AssertionError, match="mismatch"):
        _compute_cmp_lengths(layout, compress_ratio=128, kv_compress_shape_0=5)


def test_store_after_close():
    """Operations on a closed store raise RuntimeError."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[6])
    slot = _slot(plan, batch_idx=0)

    # Store works when open
    _g2_store_with_kwargs(store, slot,
                          StoredG2Activation(kv=torch.randn(6, 512), stored_len=6))
    assert store.contains(slot)

    store.close()

    # contains() is a passive lookup — it doesn't call _ensure_open()
    # and won't raise after close.  That's by design.
    with pytest.raises(RuntimeError, match="closed"):
        store.load(slot)
    with pytest.raises(RuntimeError, match="closed"):
        _g2_store_with_kwargs(store, slot,
                              StoredG2Activation(kv=torch.randn(6, 512), stored_len=6))


def test_empty_batch():
    """batch_size=0 produces empty results without error."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=0, prefix_lens=[], original_lengths=[])
    object.__setattr__(plan, "batch_size", 0)
    object.__setattr__(plan, "is_provider", [])
    object.__setattr__(plan, "prefix_lens", [])
    object.__setattr__(plan, "kept_lengths_q", [])
    object.__setattr__(plan, "provider_index", [])

    layout = PackedBatchLayout.from_valid_lengths([])
    ctx = MockContext(store=store, packed_batch_layout=layout,
                      prefix_sharing_plan=plan)

    kv = torch.zeros(0, 512)
    result_kv, result_cmp, result_idxk, _, _ = _expand_and_return(ctx, kv)

    assert result_kv.shape == (0, 512)
    assert result_cmp is None
    assert result_idxk is None
    assert store.size == 0


def test_prefix_len_zero_reuser():
    """prefix_len=0 reuser (after alignment rounds down) passes through.

    When align_prefix_lens_to_compression rounds a short prefix to 0,
    the sequence becomes a non-reuser (is_reuser returns False because
    prefix_len == 0).  It should pass through unchanged.
    """
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 0],
                      original_lengths=[128, 64])
    # seq1: is_provider=False, prefix_len=0 → neither provider nor reuser
    object.__setattr__(plan, "provider_index", [0, 0])
    object.__setattr__(plan, "is_provider", [True, False])
    object.__setattr__(plan, "kept_lengths_q", [128, 64])

    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    ctx = MockContext(store=store, packed_batch_layout=layout,
                      prefix_sharing_plan=plan)

    p_kv = torch.randn(128, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=128))

    kv = torch.cat([p_kv, torch.randn(64, 512)], dim=0)
    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    # seq0: 128, seq1: 64 — no expansion (prefix_len=0 → not a reuser)
    assert result_kv.shape[0] == 128 + 64
    # seq1 entries not stored (prefix_len=0, not a reuser, not a provider)
    assert not store.contains(_slot(plan, batch_idx=1))


def test_max_prefix_len():
    """Reuser whose prefix_len == valid_len (no suffix) gets full provider KV.

    When a reuser's entire sequence is a prefix (suffix_len=0), the cat
    produces [provider[:P], empty] = provider[:P].  This can happen when
    a reuser has the same sequence length as its prefix.
    """
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 128],
                      original_lengths=[128, 128])
    object.__setattr__(plan, "kept_lengths_q", [128, 0])  # seq1: suffix=0

    layout = PackedBatchLayout.from_valid_lengths([128, 0])
    ctx = MockContext(store=store, packed_batch_layout=layout,
                      prefix_sharing_plan=plan)

    p_kv = torch.randn(128, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=128))

    # seq1 has kept_length=0 — no suffix tensor to pack
    kv = p_kv  # only seq0
    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    # seq0: 128 + seq1: 128 (P+S=128+0) = 256 ... wait
    # Actually with kept_length_q[1]=0, the reuser row from _split_by_cu_seqlens
    # is an empty tensor. So cat(provider[:128], empty) = provider[:128] (128 rows).
    # seq0: 128, seq1: 128 → total 256
    assert result_kv.shape[0] == 128 + 128
    # seq1 is entirely prefix — its expanded KV equals provider[:128]
    assert torch.equal(result_kv[128:256], p_kv[:128])

    # seq1 stores its expanded data for transitive reuse
    assert store.contains(_slot(plan, batch_idx=1))
    assert store.load(_slot(plan, batch_idx=1)).stored_len == 128
