"""Task 2 — _g2_kv_store_or_expand core logic tests.

Mac-testable — uses synthetic tensors and mock context.
Does NOT cover topk recomputation (Task 3) or cu_seqlens adjust (Task 3).
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    """integration 代码在 reuser-load 探针(VAL-DBG)里无条件调
    torch.distributed.get_rank();单测 = 单进程语义(mock 无 PG)→ 替身 rank 0。"""
    import torch.distributed as _dist
    monkeypatch.setattr(_dist, "get_rank", lambda group=None: 0)

from prefix_sharing.backends.packed_layout import PackedBatchLayout
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
    stats: object = None

    def __post_init__(self):
        if self.parallel_info is None:
            from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
            self.parallel_info = MegatronParallelInfo()


def _slot(plan, batch_idx=0, tp_rank=0):
    return PrefixActivationSlotId(
        plan.forward_id, plan.micro_batch_id, 0,  # layer_id=0 matches default in _g2_kv_store_or_expand
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
    return result[:5]  # drop compress_topk_score (not used in unit tests)


# ── Provider ──────────────────────────────────────────────────────────


def test_provider_store_kv():
    """Provider kv stored with correct valid_len slice."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[6])
    layout = PackedBatchLayout.from_valid_lengths([6])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    kv = torch.randn(6, 512)
    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    slot = _slot(plan, batch_idx=0)
    assert store.contains(slot)
    e = store.load(slot)
    assert e.kv.shape == (6, 512)
    assert torch.equal(e.kv, kv[:6])  # provider returns unchanged
    assert torch.equal(result_kv, kv)


def test_provider_store_kv_compress():
    """Provider kv_compress stored (aligned, compress_ratio=128)."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[256])
    layout = PackedBatchLayout.from_valid_lengths([256])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    kv = torch.randn(256, 512)
    cmp = torch.randn(2, 512)  # 256//128=2
    _, result_cmp, _, _, _ = _expand_and_return(ctx, kv, kv_compress=cmp, compress_ratio=128)

    slot = _slot(plan, batch_idx=0)
    e = store.load(slot)
    assert e.kv_compress.shape == (2, 512)
    assert torch.equal(result_cmp, cmp)


def test_provider_store_indexer_k():
    """Provider indexer_k stored (aligned, compress_ratio=4)."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[8])
    layout = PackedBatchLayout.from_valid_lengths([8])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    kv = torch.randn(8, 512)
    idxk = torch.randn(2, 1, 128)  # 8//4=2
    _, _, result_idxk, _, _ = _expand_and_return(
        ctx, kv, indexer_k=idxk, compress_ratio=4)

    slot = _slot(plan, batch_idx=0)
    e = store.load(slot)
    assert e.indexer_k.shape == (2, 1, 128)
    assert torch.equal(result_idxk, idxk)


def test_non_reuser_not_stored():
    """Sequences without reuser identity (no sharing) store as providers."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 0], original_lengths=[6, 6])
    layout = PackedBatchLayout.from_valid_lengths([6, 6])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    kv = torch.randn(12, 512)
    _expand_and_return(ctx, kv)

    # Both are providers (no sharing): each stores its own data
    assert store.size == 2


def test_provider_returns_original():
    """Provider path returns input tensors unchanged."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[128])
    layout = PackedBatchLayout.from_valid_lengths([128])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    kv = torch.randn(128, 512)
    r_kv, r_cmp, r_idxk, r_topk, r_psp = _expand_and_return(ctx, kv)

    assert torch.equal(r_kv, kv)  # same values (provider returns original)
    assert r_cmp is None
    assert r_idxk is None
    assert r_topk is None
    assert r_psp is None


# ── Reuser ────────────────────────────────────────────────────────────


def test_reuser_expand_kv():
    """Reuser expands kv: cat(provider[:P], reuser[:S]) → shape[0]=P+S."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 128], original_lengths=[256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128])  # S=128
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    p_kv = torch.randn(256, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=256))

    kv = torch.cat([p_kv, torch.randn(128, 512)], dim=0)
    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    # seq0: 256 (unchanged), seq1: 128+128 = 256
    assert result_kv.shape[0] == 256 + 256
    # seq1 prefix part = provider[:128]
    assert torch.equal(result_kv[256:384], p_kv[:128])
    # seq1 suffix part = reuser suffix
    assert torch.equal(result_kv[384:512], kv[256:384])


def test_reuser_expand_kv_compress():
    """Reuser expands kv_compress: cat(provider[:P//r], reuser[:S//r])."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 256], original_lengths=[256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128])  # S=128
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)
    r = 128

    p_kv = torch.randn(256, 512)
    p_cmp = torch.randn(2, 512)  # 256//128=2
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, kv_compress=p_cmp, stored_len=256))

    kv = torch.cat([p_kv, torch.randn(128, 512)], dim=0)
    cmp = torch.cat([p_cmp, torch.randn(1, 512)], dim=0)  # 256//128=2 + 128//128=1 = 3
    _, result_cmp, _, _, _ = _expand_and_return(ctx, kv, kv_compress=cmp, compress_ratio=r)

    # seq0: 2, seq1: 2+1 = 3
    assert result_cmp.shape[0] == 2 + 3
    # seq1 prefix = provider[:2]
    assert torch.equal(result_cmp[2:4], p_cmp[:2])


def test_reuser_expand_indexer_k():
    """Reuser expands indexer_k: cat(provider[:P//4], reuser[:S//4])."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 8], original_lengths=[8, 8])
    layout = PackedBatchLayout.from_valid_lengths([8, 4])  # S=4
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)
    r = 4

    p_kv = torch.randn(8, 512)
    p_idxk = torch.randn(2, 1, 128)  # 8//4=2
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, indexer_k=p_idxk, stored_len=8))

    kv = torch.cat([p_kv, torch.randn(4, 512)], dim=0)
    idxk = torch.cat([p_idxk, torch.randn(1, 1, 128)], dim=0)  # 8//4=2 + 4//4=1
    _, _, result_idxk, _, _ = _expand_and_return(ctx, kv, indexer_k=idxk, compress_ratio=r)

    # seq0: 2, seq1: 2+1 = 3
    assert result_idxk.shape[0] == 2 + 3
    assert torch.equal(result_idxk[2:4], p_idxk[:2])


def test_reuser_preserves_own_suffix():
    """Expanded reuser suffix portion matches original input."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 128], original_lengths=[256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128])  # S=128
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    p_kv = torch.randn(256, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=256))

    suffix_kv = torch.randn(128, 512)
    kv = torch.cat([p_kv, suffix_kv], dim=0)
    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    # seq1 suffix: last 128 rows of result (after seq0's 256 rows)
    assert torch.equal(result_kv[384:512], suffix_kv)


def test_transitive_reuse():
    """Reuser A's expanded data stored for Reuser B to reuse."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=3, prefix_lens=[0, 128, 256], original_lengths=[256, 256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128, 128])  # S1=128, S2=128
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    p_kv = torch.randn(256, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=256))

    kv = torch.cat([p_kv, torch.randn(128, 512), torch.randn(128, 512)], dim=0)
    result_kv, _, _, _, _ = _expand_and_return(ctx, kv)

    # seq0: 256, seq1: 128+128 = 256, seq2: 256+128 = 384
    assert result_kv.shape[0] == 256 + 256 + 384

    # seq1's expanded data should be stored for seq2 (transitive reuse)
    slot1 = _slot(plan, batch_idx=1)
    assert store.contains(slot1)
    e = store.load(slot1)
    assert e.kv.shape == (256, 512)  # P+S=128+128
    assert e.stored_len == 256


# ── Edge cases ────────────────────────────────────────────────────────


def test_compress_ratio_le1_no_cmp():
    """compress_ratio≤1: kv_compress=None → no error."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=1, prefix_lens=[0], original_lengths=[6])
    layout = PackedBatchLayout.from_valid_lengths([6])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    kv = torch.randn(6, 512)
    r_kv, r_cmp, r_idxk, _, _ = _expand_and_return(ctx, kv, kv_compress=None, compress_ratio=1)
    assert r_kv.shape == (6, 512)
    assert r_cmp is None


def test_indexer_k_none_for_ratio128():
    """indexer_k=None → pass through unchanged."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 256], original_lengths=[256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    p_kv = torch.randn(256, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=256))

    kv = torch.cat([p_kv, torch.randn(128, 512)], dim=0)
    _, _, r_idxk, _, _ = _expand_and_return(ctx, kv, indexer_k=None, compress_ratio=128)

    assert r_idxk is None


def test_aligned_prefix_assert():
    """Non-aligned prefix raises AssertionError (Phase 1)."""
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 3], original_lengths=[128, 128])
    layout = PackedBatchLayout.from_valid_lengths([128, 125])  # P=3, S=125
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)

    p_kv = torch.randn(128, 512)
    _g2_store_with_kwargs(store, _slot(plan, batch_idx=0),
                          StoredG2Activation(kv=p_kv, stored_len=128))

    kv = torch.cat([p_kv, torch.randn(125, 512)], dim=0)
    with pytest.raises(AssertionError, match="aligned prefix"):
        _expand_and_return(ctx, kv, compress_ratio=128)
