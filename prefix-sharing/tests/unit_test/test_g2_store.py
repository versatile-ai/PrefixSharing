"""Tests for Group A — G2AttentionStore + StoredG2Activation.

These tests validate the data storage layer for DeepSeek V4 prefix sharing.
They run in pure PyTorch and do not require MindSpeed runtime.
"""

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.g2_attention_utils import (
    _merge_g2_fields,
    _merge_g2_transformer_fields,
)
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION,
    G2AttentionStore,
    PrefixActivationSlotId,
    StoredG2Activation,
)
from prefix_sharing.integrations.context import _create_store


# ── helpers ──────────────────────────────────────────────────────────


def _make_slot_id(
    forward_id=1,
    micro_batch_id=2,
    layer_id=3,
    batch_idx=0,
    tp_rank=0,
):
    return PrefixActivationSlotId(
        forward_id,
        micro_batch_id,
        layer_id,
        batch_idx,
        PREFIX_STATE_TYPE_G2_ATTENTION,
        tp_rank,
    )


def _make_runtime_state(model_type="deepseek4"):
    """Minimal runtime state duck-type for _create_store testing."""

    class State:
        pass

    state = State()
    state.model_type = model_type
    return state


# ── lifecycle ────────────────────────────────────────────────────────


def test_g2_store_lifecycle():
    """Store/load/overwrite/close lifecycle with type guard."""
    store = G2AttentionStore()
    slot_id = _make_slot_id()

    kv = torch.randn(7, 512, requires_grad=True)
    store.store(slot_id, kv=kv, stored_len=7)

    entry = store.load(slot_id)
    assert isinstance(entry, StoredG2Activation)
    assert entry.kv is kv
    assert entry.kv.requires_grad
    assert entry.kv_compress is None
    assert entry.attn_o is None
    assert entry.stored_len == 7

    # 不允许重复 overwrite（除非显式 overwrite=True）
    with pytest.raises(KeyError):
        store.store(slot_id, kv=kv, stored_len=7)

    # overwrite=True 允许更新
    kv2 = torch.randn(7, 512)
    store.store(slot_id, kv=kv2, stored_len=7, overwrite=True)
    assert store.load(slot_id).kv is kv2

    # type guard：slot 类型必须是 g2_attention
    bad_slot = PrefixActivationSlotId(1, 2, 3, 4, "wrong_type", 0)
    with pytest.raises(ValueError, match="g2_attention"):
        store.store(bad_slot, kv=kv, stored_len=7)

    # stored_len guard
    with pytest.raises(ValueError, match="stored_len"):
        store.store(slot_id, kv=kv, stored_len=-1)

    # close 后不可 store / load
    store.close()
    assert store.closed
    with pytest.raises(RuntimeError):
        store.load(slot_id)
    with pytest.raises(RuntimeError):
        store.store(slot_id, kv=kv, stored_len=7)


def test_g2_store_incremental():
    """Incremental store: same slot, three calls (kv → kv_compress → attn_o)."""
    store = G2AttentionStore()
    slot_id = _make_slot_id()

    kv = torch.randn(7, 512, requires_grad=True)
    cmp = torch.randn(0, 512)  # ratio>1 时可能为空
    attn_o = torch.randn(7, 64, 512, requires_grad=True)

    # 第一次 store: kv
    store.store(slot_id, kv=kv, stored_len=7)
    e1 = store.load(slot_id)
    assert e1.kv is kv
    assert e1.kv_compress is None
    assert e1.attn_o is None
    assert e1.stored_len == 7

    # 第二次 store: kv_compress (overwrite=True 保留已有 kv)
    store.store(slot_id, kv_compress=cmp, stored_len=7, overwrite=True)
    e2 = store.load(slot_id)
    assert e2.kv is kv  # 第一次的 kv 还在
    assert e2.kv_compress is cmp
    assert e2.stored_len == 7

    # 第三次 store: attn_o
    store.store(slot_id, attn_o=attn_o, stored_len=7, overwrite=True)
    e3 = store.load(slot_id)
    assert e3.kv is kv
    assert e3.kv_compress is cmp
    assert e3.attn_o is attn_o
    assert e3.stored_len == 7


# ── merge helpers ─────────────────────────────────────────────────────


def test_merge_g2_fields():
    """_merge_g2_fields updates one field at a time, preserves others."""
    kv = torch.randn(7, 512)
    cmp = torch.randn(0, 512)
    attn_o = torch.randn(7, 64, 512)

    # from None
    r1 = _merge_g2_fields(None, "kv", kv)
    assert isinstance(r1, StoredG2Activation)
    assert r1.kv is kv
    assert r1.stored_len == 7

    # add kv_compress
    r2 = _merge_g2_fields(r1, "kv_compress", cmp)
    assert r2.kv is kv
    assert r2.kv_compress is cmp
    assert r2.attn_o is None

    # add attn_o, stored_len 取 max
    r3 = _merge_g2_fields(r2, "attn_o", attn_o)
    assert r3.kv is kv
    assert r3.kv_compress is cmp
    assert r3.attn_o is attn_o
    assert r3.stored_len == 7  # max(7, 7)


def test_merge_g2_transformer_fields():
    """_merge_g2_transformer_fields updates 3 transformer fields at once."""
    kv = torch.randn(7, 512)
    residual = torch.randn(7, 4, 4096)
    post = torch.randn(7, 4)
    comb = torch.randn(7, 4, 4)

    # from existing attention-only entry
    existing = StoredG2Activation(kv=kv, stored_len=7)
    merged = _merge_g2_transformer_fields(
        existing,
        residual_prefix=residual,
        post_prefix=post,
        comb_prefix=comb,
        valid_len=7,
    )
    assert merged.kv is kv  # attention 字段保留
    assert merged.kv_compress is None
    assert merged.attn_o is None
    assert merged.residual_prefix is residual  # transformer 字段更新
    assert merged.post_prefix is post
    assert merged.comb_prefix is comb
    assert merged.stored_len == 7

    # from None（transformer patch 在 attention patch 之前执行）
    fresh = _merge_g2_transformer_fields(
        None,
        residual_prefix=residual,
        post_prefix=post,
        comb_prefix=comb,
        valid_len=7,
    )
    assert fresh.residual_prefix is residual
    assert fresh.post_prefix is post
    assert fresh.comb_prefix is comb
    assert fresh.kv is None
    assert fresh.stored_len == 7


# ── factory ───────────────────────────────────────────────────────────


def test_create_store_deepseek4():
    """_create_store returns G2AttentionStore for model_type='deepseek4'."""
    store_ds = _create_store(_make_runtime_state("deepseek4"))
    assert isinstance(store_ds, G2AttentionStore)

    # 默认 model_type 返回 PrefixAttentionStore（向后兼容）
    store_default = _create_store(_make_runtime_state("text_only_causal_lm"))
    from prefix_sharing.core.prefix_store import PrefixAttentionStore

    assert isinstance(store_default, PrefixAttentionStore)

    # 未知 model_type 也返回 PrefixAttentionStore（安全回退）
    store_unknown = _create_store(_make_runtime_state("unknown"))
    assert isinstance(store_unknown, PrefixAttentionStore)
