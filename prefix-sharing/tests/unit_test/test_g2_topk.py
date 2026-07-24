"""Task 3 — Topk recomputation + cu_seqlens adjust tests."""

from dataclasses import dataclass
import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION, G2AttentionStore, PrefixActivationSlotId, StoredG2Activation)
from prefix_sharing.integrations.g2_attention import _g2_kv_store_or_expand, _g2_store_with_kwargs


def _make_plan(*, batch_size, prefix_lens, original_lengths):
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True))
    plan = planner.plan([list(range(s)) for s in original_lengths])
    kept = [ol - pl for ol, pl in zip(original_lengths, prefix_lens)]
    pidx = [-1] * batch_size; is_p = [True] * batch_size
    for i, pl in enumerate(prefix_lens):
        if pl > 0: pidx[i] = 0; is_p[i] = False
    for attr, val in [("batch_size", batch_size), ("original_lengths", list(original_lengths)),
                       ("prefix_lens", list(prefix_lens)), ("kept_lengths_q", kept),
                       ("provider_index", pidx), ("is_provider", is_p)]:
        object.__setattr__(plan, attr, val)
    return plan


@dataclass
class MockContext:
    store: G2AttentionStore; packed_batch_layout: object; prefix_sharing_plan: object; parallel_info: object = None
    def __post_init__(self):
        if self.parallel_info is None:
            from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
            self.parallel_info = MegatronParallelInfo()


def _slot(plan, batch_idx=0):
    return PrefixActivationSlotId(plan.forward_id, plan.micro_batch_id, 0,
                                   batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, 0)


# ── ratio=128 ────────────────────────────────────────────────────────

Captured128 = []


class Mock128Module:
    def __init__(self): self.compress_ratio = 128; self.layer_number = 1; self.window_size = None; self.indexer = None
    def get_compress_topk_idxs(self, ratio, bsz, seqlen, start_pos=0, offset=0, cp_shard=False):
        Captured128.append(dict(ratio=ratio, bsz=bsz, seqlen=seqlen, start_pos=start_pos))
        return torch.zeros(bsz, seqlen, seqlen // ratio, dtype=torch.int64)


def test_ratio128_called_with_expanded_seqlen():
    """Ratio=128: get_compress_topk_idxs called with expanded seqlen."""
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    Captured128.clear()
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 256], original_lengths=[256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)
    p_kv = torch.randn(256, 512)
    _g2_store_with_kwargs(store, _slot(plan, 0), StoredG2Activation(kv=p_kv, stored_len=256))
    kv = torch.cat([p_kv, torch.randn(128, 512)], dim=0)
    # topk with bsz=2, q_len=max(256,128)=256, topk_len=expanded//128=3
    topk = torch.zeros(2, 256, 3, dtype=torch.int64)

    _g2_kv_store_or_expand(ctx, kv, None, None, topk, None, 128, Mock128Module(), 0, False, False)

    assert len(Captured128) >= 1
    # Called for reuser at batch_idx=1 with expanded seqlen
    assert Captured128[-1]["start_pos"] == 0
    # seqlen should be expanded: prefix_len + q_global = 256 + 128 = 384
    assert Captured128[-1]["seqlen"] == 384


def test_ratio128_start_pos_passthrough():
    """Ratio=128: start_pos passed to get_compress_topk_idxs."""
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    Captured128.clear()
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 256], original_lengths=[256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)
    p_kv = torch.randn(256, 512)
    _g2_store_with_kwargs(store, _slot(plan, 0), StoredG2Activation(kv=p_kv, stored_len=256))
    kv = torch.cat([p_kv, torch.randn(128, 512)], dim=0)
    topk = torch.zeros(2, 256, 3, dtype=torch.int64)

    _g2_kv_store_or_expand(ctx, kv, None, None, topk, None, 128, Mock128Module(), start_pos=4096, kv_allgather=False, sequence_parallel=False)

    assert Captured128[-1]["start_pos"] == 4096


# ── ratio=4 ──────────────────────────────────────────────────────────

Captured4 = []


class Mock4Indexer:
    compress_ratio = 4; index_topk = 512; use_fused_lightning_indexer = False
    def forward_with_scores_compress(self, x, q, k, w, mask=None, packed_seq_params=None,
                                      start_pos=0, index_topk=512, offset=0, compress_ratio=4):
        Captured4.append(dict(k_shape=k.shape))
        K = k.shape[0]; topk = min(index_topk, K)
        S = q.shape[0] if hasattr(q, 'shape') else 1
        return (torch.zeros(2, S, topk, dtype=torch.int64),
                torch.rand(2, S, topk))
    def post_process_index(self, i, s): return i, s
    def all_gather_qk_weight_kvallgather(self, q, k, w): return q, k, w


class Mock4Module:
    def __init__(self): self.compress_ratio = 4; self.layer_number = 1; self.window_size = None; self.indexer = Mock4Indexer()


def test_ratio4_expanded_k_passed():
    """Ratio=4: expanded indexer_k passed to forward_with_scores_compress."""
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    Captured4.clear()
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 8], original_lengths=[8, 8])
    layout = PackedBatchLayout.from_valid_lengths([8, 4])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)
    p_kv = torch.randn(8, 512); p_idxk = torch.randn(2, 1, 128)
    _g2_store_with_kwargs(store, _slot(plan, 0), StoredG2Activation(kv=p_kv, indexer_k=p_idxk, stored_len=8))
    kv = torch.cat([p_kv, torch.randn(4, 512)], dim=0)
    idxk = torch.cat([p_idxk, torch.randn(1, 1, 128)], dim=0)
    topk = torch.zeros(2, 8, 512, dtype=torch.int64)

    mock_q = torch.randn(4, 64, 128)  # S=4, n_heads=64, head_dim=128
    mock_w = torch.randn(4, 64)
    mock_x = torch.randn(4, 1, 4096)
    _g2_kv_store_or_expand(ctx, kv, None, idxk, topk, None, 4, Mock4Module(), 0, False, False,
                           query_index=mock_q, indexer_weights=mock_w, dsa_hidden=mock_x)

    # forward_with_scores_compress should be called with expanded k
    assert len(Captured4) >= 1
    # P//4+S//4 = 8//4+4//4 = 2+1 = 3
    assert Captured4[-1]["k_shape"] == (3, 1, 128)


# ── cu_seqlens ───────────────────────────────────────────────────────


def test_cu_seqlens_adjust_in_expand():
    """cu_seqlens_kv shifted by prefix_len for reuser."""
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    @dataclass
    class MockPsp:
        cu_seqlens_kv: list
    store = G2AttentionStore()
    plan = _make_plan(batch_size=2, prefix_lens=[0, 256], original_lengths=[256, 256])
    layout = PackedBatchLayout.from_valid_lengths([256, 128])
    ctx = MockContext(store=store, packed_batch_layout=layout, prefix_sharing_plan=plan)
    p_kv = torch.randn(256, 512)
    _g2_store_with_kwargs(store, _slot(plan, 0), StoredG2Activation(kv=p_kv, stored_len=256))
    kv = torch.cat([p_kv, torch.randn(128, 512)], dim=0)
    psp = MockPsp(cu_seqlens_kv=[0, 256, 384])
    topk = torch.zeros(2, 256, 3, dtype=torch.int64)

    result = _g2_kv_store_or_expand(
        ctx, kv, None, None, topk, psp, 128, Mock128Module(), 0, False, False)
    result_psp = result[4]  # packed_seq_params

    assert result_psp.cu_seqlens_kv == [0, 256, 640]
