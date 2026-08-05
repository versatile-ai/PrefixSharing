"""DeepSeek4 G2 attention store/expand helpers.

Per-sequence functions that split packed tensors, iterate over batch
indices, and call into :class:`G2AttentionStore` for provider storage
and reuser expansion.  Framework-agnostic — no MindSpeed dependency.
"""

from __future__ import annotations

import os as _os

import torch

from prefix_sharing.backends.g2_attention_utils import (
    _adjust_cu_seqlens_for_batch,
    _compute_cmp_lengths,
    _merge_g2_fields,
    _split_by_cu_seqlens,
)
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION,
    PrefixActivationSlotId,
    StoredG2Activation,
)

_PS_DEBUG = _os.environ.get("PS_DEBUG", "0") == "1"


def _g2_store_with_kwargs(store, slot_id, data):
    """Store *data* (a :class:`StoredG2Activation`) into *store*.

    Fields are passed explicitly as keyword arguments to
    :meth:`G2AttentionStore.store`, avoiding the type-safety issues of
    ``**dict`` unpacking with a frozen dataclass.

    Args:
        store: :class:`G2AttentionStore`.
        slot_id: :class:`PrefixActivationSlotId`.
        data: :class:`StoredG2Activation` — all fields are forwarded.
    """
    store.store(
        slot_id,
        kv=data.kv,
        kv_compress=data.kv_compress,
        indexer_k=data.indexer_k,
        stored_len=data.stored_len,
        overwrite=True,
    )


def _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank, field, tensor):
    """Split packed *tensor* and store provider rows into the G2 store.

    Each provider's ``valid_row[:valid_len]`` is merged with any existing
    entry for the same slot (incremental store: kv → kv_compress → indexer_k).

    Args:
        ctx: Runtime context with ``.store`` (G2AttentionStore).
        layout: :class:`PackedBatchLayout`.
        plan: :class:`PrefixSharingPlan`.
        layer_id: int.
        tp_rank: int.
        field: One of ``"kv"``, ``"kv_compress"``, ``"indexer_k"``.
        tensor: Packed tensor ``[total_padded, ...]``, or ``None``
            (no-op).
    """
    if tensor is None:
        return

    rows = _split_by_cu_seqlens(tensor, layout.padded_lengths)
    for batch_idx, row in enumerate(rows):
        if not plan.is_provider[batch_idx]:
            continue
        valid_len = layout.valid_lengths[batch_idx]
        valid_row = row[:valid_len]
        slot_id = PrefixActivationSlotId(
            plan.forward_id,
            plan.micro_batch_id,
            layer_id,
            batch_idx,
            PREFIX_STATE_TYPE_G2_ATTENTION,
            tp_rank,
        )
        existing = ctx.store.load(slot_id) if ctx.store.contains(slot_id) else None
        merged = _merge_g2_fields(existing, field, valid_row)
        _g2_store_with_kwargs(ctx.store, slot_id, merged)


def _g2_padded_store_or_replace(
    ctx, kv, kv_compress, indexer_k,
    compress_topk_idxs, packed_seq_params, compress_topk_score,
    compress_ratio, attention_module, layout, plan, layer_id, tp_rank,
):
    """Store/replace for padded batch [S, B, D] format.

    Unlike the packed path which concatenates variable-length sequences,
    padded batch keeps all sequences at the same length S. Each sequence
    is in kv[:, batch_idx, :].

    Provider: store kv[:valid_len, provider_idx, :] into G2AttentionStore.
    Reuser:   replace kv[:prefix_len, reuser_idx, :] with stored provider KV.
    Tensors are returned with the same shape — no length change.
    """
    seq_len = kv.shape[0]

    for batch_idx in range(layout.batch_size):
        valid_len = layout.valid_lengths[batch_idx]

        if plan.is_provider[batch_idx]:
            provider_kv = kv[:valid_len, batch_idx, :].clone()
            provider_cmp = None
            if kv_compress is not None and compress_ratio > 1:
                cmp_len = valid_len // compress_ratio
                provider_cmp = kv_compress[:cmp_len, batch_idx, :].clone() \
                    if kv_compress.ndim == 3 else kv_compress[:cmp_len].clone()
            provider_idxk = None
            if indexer_k is not None and compress_ratio > 1:
                idxk_len = valid_len // compress_ratio
                if indexer_k.ndim >= 3 and indexer_k.shape[1] > 1:
                    provider_idxk = indexer_k[:idxk_len, batch_idx].clone()
                else:
                    provider_idxk = indexer_k[:idxk_len].clone()

            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            _g2_store_with_kwargs(ctx.store, slot_id, StoredG2Activation(
                kv=provider_kv, kv_compress=provider_cmp,
                indexer_k=provider_idxk, stored_len=valid_len))

        elif plan.is_reuser(batch_idx):
            prefix_len = plan.prefix_lens[batch_idx]
            provider_idx = plan.provider_index[batch_idx]
            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                provider_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            provider = ctx.store.load(slot_id)

            kv[:prefix_len, batch_idx, :] = provider.kv[:prefix_len]

            if kv_compress is not None and compress_ratio > 1 and provider.kv_compress is not None:
                cmp_p = prefix_len // compress_ratio
                if kv_compress.ndim == 3 and kv_compress.shape[1] > 1:
                    kv_compress[:cmp_p, batch_idx, :] = provider.kv_compress[:cmp_p]
                else:
                    kv_compress[:cmp_p] = provider.kv_compress[:cmp_p]

    # ── Stats reporting ──────────────────────────────────────────
    if ctx.stats is not None:
        _store_count = sum(1 for i in range(layout.batch_size) if plan.is_provider[i])
        _reuse_count = sum(1 for i in range(layout.batch_size) if plan.is_reuser(i))
        _stored_tokens = sum(
            layout.valid_lengths[i] for i in range(layout.batch_size) if plan.is_provider[i])
        _reused_prefix_tokens = sum(
            plan.prefix_lens[i] for i in range(layout.batch_size) if plan.is_reuser(i))
        _expanded_kv_tokens = sum(layout.valid_lengths)  # padded: total valid lengths unchanged
        ctx.stats.record_attention_kv_build(
            layer_id=layer_id, store_count=_store_count,
            reuse_count=_reuse_count, reuse_hit_count=_reuse_count,
            reuse_miss_count=0,
            stored_tokens=_stored_tokens,
            reused_prefix_tokens=_reused_prefix_tokens,
            expanded_kv_tokens=_expanded_kv_tokens,
            valid_q_tokens=sum(layout.valid_lengths),
            padded_q_tokens=sum(layout.padded_lengths),
        )

    return (kv, kv_compress, indexer_k,
            compress_topk_idxs, packed_seq_params, compress_topk_score)


def _g2_kv_store_or_expand(
    ctx,
    kv: torch.Tensor,
    kv_compress: torch.Tensor | None,
    indexer_k: torch.Tensor | None,
    compress_topk_idxs,
    packed_seq_params,
    compress_ratio: int,
    attention_module,
    start_pos: int,
    kv_allgather: bool,
    sequence_parallel: bool,
    *,
    query_index=None,            # ratio=4: q_r from Phase 2
    indexer_weights=None,        # ratio=4: w_r from Phase 2
    dsa_hidden=None,             # ratio=4: dsa_hidden from Phase 2
    attention_mask=None,         # ratio=4: forward_with_scores_compress mask
    compress_topk_score=None,    # ratio=4: updated by re-scoring
):
    """Provider store / Reuser expand for all key-side data.

    Called in the patched forward between Phase 3 and Phase 4.
    Splits packed tensors, iterates batch indices, and branches by
    provider/reuser identity.

    Returns expanded (kv, kv_compress, indexer_k, compress_topk_idxs,
    packed_seq_params).  For providers fields are unchanged.
    Topk recomputation and cu_seqlens adjustment deferred to Task 3.
    """
    layout = ctx.packed_batch_layout
    plan = ctx.prefix_sharing_plan
    tp_rank = ctx.parallel_info.tp_rank
    layer_id = attention_module.layer_number if attention_module is not None else 0

    # Empty batch (all sequences trimmed away) → return empty tensors unchanged.
    if layout.batch_size == 0:
        return (kv, kv_compress, indexer_k,
                compress_topk_idxs, packed_seq_params, compress_topk_score)

    # Padded batch (BSND): kv is [S, B, D] with B > 1.
    # Each sequence already has full KV for all positions (no trim).
    # Store provider prefix KV, then replace reuser prefix KV in-place.
    # Tensor shape stays [S, B, D] — no length change.
    _is_padded = kv.ndim == 3 and kv.shape[1] > 1
    if _is_padded:
        return _g2_padded_store_or_replace(
            ctx, kv, kv_compress, indexer_k,
            compress_topk_idxs, packed_seq_params, compress_topk_score,
            compress_ratio, attention_module, layout, plan, layer_id, tp_rank)

    # Split raw KV by padded lengths (matching Q path)
    kv_rows = _split_by_cu_seqlens(kv, layout.padded_lengths)

    # Split compressed fields by cmp lengths (valid//ratio)
    has_cmp = kv_compress is not None and compress_ratio > 1
    has_idxk = indexer_k is not None and compress_ratio > 1

    cmp_rows = None
    idxk_rows = None
    if has_cmp:
        cmp_lengths = _compute_cmp_lengths(layout, compress_ratio, kv_compress.shape[0])
        cmp_rows = _split_by_cu_seqlens(kv_compress, cmp_lengths)
    if has_idxk:
        idxk_lengths = _compute_cmp_lengths(layout, compress_ratio, indexer_k.shape[0])
        idxk_rows = _split_by_cu_seqlens(indexer_k, idxk_lengths)

    new_kv: list[torch.Tensor] = []
    new_cmp: list[torch.Tensor] = []
    new_idxk: list[torch.Tensor] = []

    for batch_idx in range(layout.batch_size):
        valid_len = layout.valid_lengths[batch_idx]

        if plan.is_provider[batch_idx]:
            # ── Provider: store ──
            valid_kv = kv_rows[batch_idx][:valid_len]
            valid_cmp = cmp_rows[batch_idx][:valid_len // compress_ratio] if cmp_rows else None
            valid_idxk = idxk_rows[batch_idx][:valid_len // compress_ratio] if idxk_rows else None

            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            if _PS_DEBUG:
                print(f"[PS_DEBUG] Provider store: batch_idx={batch_idx} valid_len={valid_len} kv_shape={valid_kv.shape} slot={slot_id}")
            _g2_store_with_kwargs(ctx.store, slot_id, StoredG2Activation(
                kv=valid_kv, kv_compress=valid_cmp, indexer_k=valid_idxk,
                stored_len=valid_len))

            new_kv.append(kv_rows[batch_idx])
            if cmp_rows:
                new_cmp.append(cmp_rows[batch_idx])
            if idxk_rows:
                new_idxk.append(idxk_rows[batch_idx])

        elif plan.is_reuser(batch_idx):
            # ── Reuser: expand ──
            prefix_len = plan.prefix_lens[batch_idx]
            if compress_ratio > 1:
                assert prefix_len % compress_ratio == 0, (
                    f"Phase 1 requires aligned prefix: "
                    f"prefix_len={prefix_len}, compress_ratio={compress_ratio}")

            provider_idx = plan.provider_index[batch_idx]
            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                provider_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            if _PS_DEBUG:
                print(f"[PS_DEBUG] Reuser load: batch_idx={batch_idx} prefix_len={prefix_len} provider_idx={provider_idx} slot={slot_id}")
                print(f"[PS_DEBUG] Store contains slot: {ctx.store.contains(slot_id)}")
            provider = ctx.store.load(slot_id)
            if _PS_DEBUG:
                print(f"[PS_DEBUG] Loaded provider kv: shape={provider.kv.shape} stored_len={provider.stored_len}")

            # Expand kv
            expanded_kv = torch.cat([
                provider.kv[:prefix_len],
                kv_rows[batch_idx][:valid_len]], dim=0)
            if _PS_DEBUG:
                print(f"[PS_DEBUG] Expanded kv: {expanded_kv.shape} (prefix={prefix_len} + suffix={valid_len})")
            new_kv.append(expanded_kv)

            # Expand kv_compress
            expanded_cmp = None
            if cmp_rows is not None:
                cmp_p = prefix_len // compress_ratio
                cmp_s = valid_len // compress_ratio
                expanded_cmp = torch.cat([
                    provider.kv_compress[:cmp_p],
                    cmp_rows[batch_idx][:cmp_s]], dim=0)
                new_cmp.append(expanded_cmp)

            # Expand indexer_k
            expanded_idxk = None
            if idxk_rows is not None:
                idxk_p = prefix_len // compress_ratio
                idxk_s = valid_len // compress_ratio
                expanded_idxk = torch.cat([
                    provider.indexer_k[:idxk_p],
                    idxk_rows[batch_idx][:idxk_s]], dim=0)
                new_idxk.append(expanded_idxk)

            # THD format (bsz=1): map batch_idx to tensor batch dim 0
            _topk_batch_idx = 0 if compress_topk_idxs is not None and compress_topk_idxs.shape[0] == 1 else batch_idx

            # Recompute topk for expanded key space
            if compress_topk_idxs is not None and compress_ratio > 1:
                if hasattr(attention_module, 'indexer') and attention_module.indexer is not None:
                    # ratio=4: re-score with expanded indexer_k
                    if expanded_idxk is not None and query_index is not None:
                        new_topk, new_score = attention_module.indexer.forward_with_scores_compress(
                            x=dsa_hidden, q=query_index, k=expanded_idxk, weights=indexer_weights,
                            mask=attention_mask, packed_seq_params=packed_seq_params,
                            start_pos=start_pos, index_topk=attention_module.indexer.index_topk,
                            offset=0, compress_ratio=compress_ratio)
                        q_len_local = valid_len
                        topk_len = new_topk.shape[-1]
                        compress_topk_idxs[_topk_batch_idx, :q_len_local, :topk_len] = \
                            new_topk[_topk_batch_idx, :q_len_local, :]
                        if compress_topk_score is not None:
                            compress_topk_score[_topk_batch_idx, :q_len_local, :topk_len] = \
                                new_score[_topk_batch_idx, :q_len_local, :]
                else:
                    # ratio=128: recompute by position with expanded seqlen
                    tp_size = 1
                    cp_size = 1
                    try:
                        from megatron.core import parallel_state
                        tp_size = parallel_state.get_tensor_model_parallel_world_size()
                        cp_size = parallel_state.get_context_parallel_world_size()
                    except (ImportError, RuntimeError, AssertionError):
                        pass
                    q_len_local = valid_len
                    q_len = q_len_local * tp_size if sequence_parallel else q_len_local
                    q_len_global = q_len * cp_size if cp_size > 1 else q_len
                    expanded_seqlen = prefix_len + q_len_global
                    bsz = compress_topk_idxs.shape[0]
                    new_idxs = attention_module.get_compress_topk_idxs(
                        compress_ratio, bsz, expanded_seqlen,
                        start_pos=start_pos, offset=0, cp_shard=kv_allgather)
                    topk_len = min(new_idxs.shape[-1],
                                   compress_topk_idxs.shape[-1])
                    # get_compress_topk_idxs returns LOCAL indices [0, seqlen//r).
                    # Add provider's CMP block offset to convert to global packed indices.
                    # Provider CMP blocks before this reuser = global offset
                    _cmp_offset = sum(cmp_lengths[:batch_idx]) if cmp_rows else 0
                    _new_idxs = new_idxs[_topk_batch_idx, -q_len_local:, :topk_len].clone()
                    _new_idxs[_new_idxs >= 0] += _cmp_offset
                    compress_topk_idxs[_topk_batch_idx, :q_len_local, :topk_len] = _new_idxs

            # Store back for transitive reuse
            own_slot = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            _g2_store_with_kwargs(ctx.store, own_slot, StoredG2Activation(
                kv=expanded_kv, kv_compress=expanded_cmp,
                indexer_k=expanded_idxk, stored_len=prefix_len + valid_len))

        else:
            # Non-provider, non-reuser — pass through unchanged
            new_kv.append(kv_rows[batch_idx][:valid_len])
            if cmp_rows:
                new_cmp.append(cmp_rows[batch_idx][:valid_len // compress_ratio])
            if idxk_rows:
                new_idxk.append(idxk_rows[batch_idx][:valid_len // compress_ratio])

    # ── Stats reporting ──────────────────────────────────────────
    if ctx.stats is not None:
        _store_count = sum(1 for i in range(layout.batch_size) if plan.is_provider[i])
        _reuse_count = sum(1 for i in range(layout.batch_size) if plan.is_reuser(i))
        _stored_tokens = sum(
            layout.valid_lengths[i] for i in range(layout.batch_size) if plan.is_provider[i])
        _reused_prefix_tokens = sum(
            plan.prefix_lens[i] for i in range(layout.batch_size) if plan.is_reuser(i))
        _expanded_kv_tokens = sum(
            (plan.prefix_lens[i] + layout.valid_lengths[i]) if plan.is_reuser(i)
            else layout.valid_lengths[i] for i in range(layout.batch_size))
        ctx.stats.record_attention_kv_build(
            layer_id=layer_id, store_count=_store_count,
            reuse_count=_reuse_count, reuse_hit_count=_reuse_count,
            reuse_miss_count=0,
            stored_tokens=_stored_tokens,
            reused_prefix_tokens=_reused_prefix_tokens,
            expanded_kv_tokens=_expanded_kv_tokens,
            valid_q_tokens=sum(layout.valid_lengths),
            padded_q_tokens=sum(layout.padded_lengths),
        )

    # Adjust cu_seqlens for reusers (offsets all subsequent entries)
    if packed_seq_params is not None:
        if _PS_DEBUG:
            print(f"[PS_DEBUG] Before adjust: cu_seqlens_q={packed_seq_params.cu_seqlens_q}, cu_seqlens_kv={packed_seq_params.cu_seqlens_kv}")
        packed_seq_params = _adjust_cu_seqlens_for_batch(
            packed_seq_params, plan, compress_ratio)
        if _PS_DEBUG:
            print(f"[PS_DEBUG] After adjust: cu_seqlens_q={packed_seq_params.cu_seqlens_q}, cu_seqlens_kv={packed_seq_params.cu_seqlens_kv}")

    result_kv = torch.cat(new_kv, dim=0)
    if _PS_DEBUG:
        print(f"[PS_DEBUG] result_kv shape: {result_kv.shape} (was {kv.shape})")
    result_cmp = torch.cat(new_cmp, dim=0) if new_cmp else (kv_compress if has_cmp else None)
    result_idxk = torch.cat(new_idxk, dim=0) if new_idxk else (indexer_k if has_idxk else None)

    return (result_kv, result_cmp, result_idxk,
            compress_topk_idxs, packed_seq_params, compress_topk_score)
