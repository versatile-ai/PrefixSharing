"""Task 1 — StoredG2Activation (4 fields) + G2AttentionStore tests."""

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION,
    G2AttentionStore,
    PrefixActivationSlotId,
    StoredG2Activation,
)


def _slot(forward_id=1, micro_batch_id=2, layer_id=3, batch_idx=0, tp_rank=0):
    return PrefixActivationSlotId(
        forward_id, micro_batch_id, layer_id,
        batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)


# ── lifecycle ────────────────────────────────────────────────────────


def test_store_lifecycle():
    """Store → load → fields correct. Close blocks access."""
    store = G2AttentionStore()
    slot = _slot()
    kv = torch.randn(7, 512, requires_grad=True)
    cmp = torch.randn(0, 512)
    idxk = torch.randn(1, 1, 128)

    store.store(slot, kv=kv, kv_compress=cmp, indexer_k=idxk, stored_len=7)
    e = store.load(slot)

    assert isinstance(e, StoredG2Activation)
    assert e.kv is kv
    assert e.kv_compress is cmp
    assert e.indexer_k is idxk
    assert e.stored_len == 7
    assert e.kv.requires_grad

    # Type guard
    bad = PrefixActivationSlotId(1, 2, 3, 4, "wrong_type", 0)
    with pytest.raises(ValueError, match="g2_attention"):
        store.store(bad, stored_len=1)

    # stored_len guard
    with pytest.raises(ValueError, match="stored_len"):
        store.store(slot, stored_len=-1)

    # Close
    store.close()
    assert store.closed
    with pytest.raises(RuntimeError):
        store.load(slot)


def test_store_overwrite_reject():
    """Duplicate store without overwrite raises KeyError."""
    store = G2AttentionStore()
    slot = _slot()
    store.store(slot, kv=torch.randn(3, 512), stored_len=3)
    with pytest.raises(KeyError):
        store.store(slot, kv=torch.randn(3, 512), stored_len=3)


def test_store_overwrite_merge():
    """overwrite=True merges: kv → kv_compress preserves kv."""
    store = G2AttentionStore()
    slot = _slot()
    kv = torch.randn(5, 512)
    cmp = torch.randn(0, 512)

    store.store(slot, kv=kv, stored_len=5)
    store.store(slot, kv_compress=cmp, stored_len=5, overwrite=True)
    e = store.load(slot)

    assert e.kv is kv          # 保持
    assert e.kv_compress is cmp  # 新增
    assert e.indexer_k is None   # 未设置


# ── indexer_k ────────────────────────────────────────────────────────


def test_store_indexer_k():
    """indexer_k stored and loaded correctly."""
    store = G2AttentionStore()
    slot = _slot()
    idxk = torch.randn(2, 1, 128)

    store.store(slot, kv=torch.randn(8, 512), indexer_k=idxk,
                kv_compress=torch.randn(2, 512), stored_len=8)
    e = store.load(slot)

    assert e.indexer_k.shape == (2, 1, 128)
    assert e.indexer_k is idxk


def test_store_indexer_k_none():
    """indexer_k=None (ratio=128 layer) — no error, field stays None."""
    store = G2AttentionStore()
    slot = _slot()

    store.store(slot, kv=torch.randn(8, 512), stored_len=8)
    e = store.load(slot)

    assert e.kv is not None
    assert e.indexer_k is None
    assert e.kv_compress is None


# ── merge ─────────────────────────────────────────────────────────────


def test_merge_g2_fields():
    """_merge_g2_fields updates one field, preserves others including indexer_k."""
    from prefix_sharing.backends.g2_attention_utils import _merge_g2_fields

    kv = torch.randn(7, 512)
    cmp = torch.randn(0, 512)
    idxk = torch.randn(1, 1, 128)

    # From None
    r1 = _merge_g2_fields(None, "kv", kv)
    assert r1.kv is kv
    assert r1.stored_len == 7

    # Add kv_compress
    r2 = _merge_g2_fields(r1, "kv_compress", cmp)
    assert r2.kv is kv
    assert r2.kv_compress is cmp

    # Add indexer_k
    r3 = _merge_g2_fields(r2, "indexer_k", idxk)
    assert r3.kv is kv
    assert r3.kv_compress is cmp
    assert r3.indexer_k is idxk
    assert r3.stored_len == 7  # max(7, 7, 7)


def test_merge_transformer_deleted():
    """_merge_g2_transformer_fields should not exist — was deleted."""
    from prefix_sharing.backends import g2_attention_utils
    assert not hasattr(g2_attention_utils, '_merge_g2_transformer_fields'), \
        "_merge_g2_transformer_fields should have been deleted"
