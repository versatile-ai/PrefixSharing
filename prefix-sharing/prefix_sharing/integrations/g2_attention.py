"""DeepSeek4 G2 attention store/expand helpers.

Per-sequence functions that split packed tensors, iterate over batch
indices, and call into :class:`G2AttentionStore` for provider storage
and reuser expansion.  Framework-agnostic — no MindSpeed dependency.
"""

from __future__ import annotations

import torch

from prefix_sharing.backends.g2_attention_utils import (
    _merge_g2_fields,
    _split_by_cu_seqlens,
)
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION,
    PrefixActivationSlotId,
    StoredG2Activation,
)


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
        attn_o=data.attn_o,
        residual_prefix=data.residual_prefix,
        post_prefix=data.post_prefix,
        comb_prefix=data.comb_prefix,
        indexer_score=data.indexer_score,
        stored_len=data.stored_len,
        overwrite=True,
    )


def _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank, field, tensor):
    """Split packed *tensor* and store provider rows into the G2 store.

    Each provider's ``valid_row[:valid_len]`` is merged with any existing
    entry for the same slot (incremental store: kv → kv_compress → attn_o).

    Args:
        ctx: Runtime context with ``.store`` (G2AttentionStore).
        layout: :class:`PackedBatchLayout`.
        plan: :class:`PrefixSharingPlan`.
        layer_id: int.
        tp_rank: int.
        field: One of ``"kv"``, ``"kv_compress"``, ``"attn_o"``,
            ``"indexer_score"``.
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


def _g2_expand_attn_output(ctx, layout, plan, layer_id, tp_rank, o):
    """Expand reuser attention outputs with provider prefix ``attn_o``.

    Provider rows are kept as-is.  Reuser rows prepend the provider's
    prefix slice of ``attn_o``.  Expanded rows are stored back for
    transitive reuse (deeper reusers in the same batch).

    Args:
        ctx: Runtime context.
        layout: :class:`PackedBatchLayout`.
        plan: :class:`PrefixSharingPlan`.
        layer_id: int.
        tp_rank: int.
        o: Packed suffix-only attention output
            ``[total_padded, n_local, head_dim]``.

    Returns:
        Expanded ``attn_o`` ``[total_padded + sum(reuser_prefix_lens), …]``.
    """
    rows = _split_by_cu_seqlens(o, layout.padded_lengths)
    expanded: list[torch.Tensor] = []

    for batch_idx, row in enumerate(rows):
        valid_len = layout.valid_lengths[batch_idx]
        if not plan.is_reuser(batch_idx):
            expanded.append(row[:valid_len])
            continue

        provider_idx = plan.provider_index[batch_idx]
        prefix_len = plan.prefix_lens[batch_idx]
        slot_id = PrefixActivationSlotId(
            plan.forward_id,
            plan.micro_batch_id,
            layer_id,
            provider_idx,
            PREFIX_STATE_TYPE_G2_ATTENTION,
            tp_rank,
        )
        provider_data = ctx.store.load(slot_id)
        expanded_row = torch.cat(
            [provider_data.attn_o[:prefix_len], row[:valid_len]], dim=0
        )
        expanded.append(expanded_row)

        # Store back for transitive reuse.
        own_slot_id = PrefixActivationSlotId(
            plan.forward_id,
            plan.micro_batch_id,
            layer_id,
            batch_idx,
            PREFIX_STATE_TYPE_G2_ATTENTION,
            tp_rank,
        )
        try:
            existing = ctx.store.load(own_slot_id)
            merged = _merge_g2_fields(existing, "attn_o", expanded_row)
            _g2_store_with_kwargs(ctx.store, own_slot_id, merged)
        except KeyError:
            # Hook C may not have created a slot yet (e.g. compress_ratio≤1).
            pass

    return torch.cat(expanded, dim=0)
