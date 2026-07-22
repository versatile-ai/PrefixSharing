"""DeepSeek4 G2 attention utility functions.

Tensor manipulation helpers for splitting, merging, and adjusting
G2 attention data structures.  Framework-agnostic — no MindSpeed
dependency.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from prefix_sharing.core.prefix_store import StoredG2Activation


def _merge_g2_fields(
    existing: StoredG2Activation | None,
    field: str,
    new_tensor: Any,
) -> StoredG2Activation:
    """Return a new StoredG2Activation with *field* updated to *new_tensor*.

    Since :class:`StoredG2Activation` is a frozen dataclass, every merge
    creates a fresh instance.  All other fields retain their values from
    *existing* (or stay ``None`` when *existing* is ``None``).
    ``stored_len`` is set to ``max(existing.stored_len, new_tensor.shape[0])``.
    """
    if existing is None:
        return StoredG2Activation(
            **{field: new_tensor},
            stored_len=new_tensor.shape[0],
        )
    return StoredG2Activation(
        kv=new_tensor if field == "kv" else existing.kv,
        kv_compress=new_tensor if field == "kv_compress" else existing.kv_compress,
        attn_o=new_tensor if field == "attn_o" else existing.attn_o,
        residual_prefix=(
            new_tensor if field == "residual_prefix" else existing.residual_prefix
        ),
        post_prefix=new_tensor if field == "post_prefix" else existing.post_prefix,
        comb_prefix=new_tensor if field == "comb_prefix" else existing.comb_prefix,
        indexer_score=(
            new_tensor if field == "indexer_score" else existing.indexer_score
        ),
        stored_len=max(existing.stored_len, new_tensor.shape[0]),
    )


def _merge_g2_transformer_fields(
    existing: StoredG2Activation | None,
    *,
    residual_prefix: Any,
    post_prefix: Any,
    comb_prefix: Any,
    valid_len: int,
) -> StoredG2Activation:
    """Merge transformer fields (residual/post/comb) while retaining attention fields.

    Unlike :func:`_merge_g2_fields`, this updates three fields at once, which
    avoids multiple store calls when the transformer patch writes its data.
    When *existing* is ``None`` (transformer patch runs before attention patch),
    a fresh entry with only transformer fields is returned.
    """
    if existing is None:
        return StoredG2Activation(
            residual_prefix=residual_prefix,
            post_prefix=post_prefix,
            comb_prefix=comb_prefix,
            stored_len=valid_len,
        )
    return StoredG2Activation(
        kv=existing.kv,
        kv_compress=existing.kv_compress,
        attn_o=existing.attn_o,
        residual_prefix=residual_prefix,
        post_prefix=post_prefix,
        comb_prefix=comb_prefix,
        indexer_score=existing.indexer_score,
        stored_len=max(existing.stored_len, valid_len),
    )


# ── B1: packed tensor utilities ──────────────────────────────────────


def _split_by_cu_seqlens(tensor: torch.Tensor, padded_lengths: list[int]) -> list[torch.Tensor]:
    """Split a packed tensor into per-sequence rows by *padded_lengths*.

    Args:
        tensor: ``[total_padded, ...]`` packed tensor.
        padded_lengths: Padded length of each sequence.  Must sum to
            ``tensor.shape[0]``.

    Returns:
        One tensor per sequence, each with shape ``[padded_len, ...]``.

    Raises:
        ValueError: If ``sum(padded_lengths) != tensor.shape[0]``.
    """
    if not padded_lengths:
        return []
    if sum(padded_lengths) != tensor.shape[0]:
        raise ValueError(
            f"sum(padded_lengths)={sum(padded_lengths)} != "
            f"tensor.shape[0]={tensor.shape[0]}"
        )
    return list(torch.split(tensor, padded_lengths, dim=0))


def _compute_cmp_lengths(
    layout: Any,
    compress_ratio: int,
    kv_compress_shape_0: int,
) -> list[int]:
    """Compute per-sequence compressed-KV padded lengths.

    ``kv_compress`` has dim=0 ``sum(valid_len // ratio)``, which differs
    from the Q-path ``padded_lengths``.  This helper derives the per-row
    lengths from ``valid_lengths`` and asserts they sum to the actual
    tensor length (to catch compressor-internal padding mismatches).

    Args:
        layout: :class:`PackedBatchLayout` whose ``valid_lengths`` are used.
        compress_ratio: Compression ratio (e.g. 128).
        kv_compress_shape_0: Actual ``kv_compress.shape[0]`` for validation.

    Returns:
        Per-sequence compressed lengths ``[valid_0 // ratio, …]``.

    Raises:
        AssertionError: If the computed sum does not match *kv_compress_shape_0*.
    """
    lengths = [vl // compress_ratio for vl in layout.valid_lengths]
    computed_sum = sum(lengths)
    assert computed_sum == kv_compress_shape_0, (
        f"CMP KV length mismatch: computed={computed_sum} "
        f"!= actual={kv_compress_shape_0}. "
        f"compressor may apply TP padding — adjust _compute_cmp_lengths accordingly"
    )
    return lengths


def _adjust_cu_seqlens_for_batch(
    packed_seq_params: Any | None,
    plan: Any,
    compress_ratio: int,
) -> Any | None:
    """Return a new *packed_seq_params* with reuser ``cu_seqlens_kv`` shifted.

    Uses :func:`dataclasses.replace` to create a new instance — the
    original is never mutated.

    Both ``cu_seqlens_kv`` (or ``cu_seqlens_kv_padded`` when present)
    and ``cu_seqlens_cmp_kv`` (when present) are adjusted.

    Args:
        packed_seq_params: Megatron ``packed_seq_params`` dataclass, or ``None``.
        plan: :class:`PrefixSharingPlan` with ``prefix_lens`` and ``batch_size``.
        compress_ratio: Compression ratio for ``cu_seqlens_cmp_kv`` offset
            computation (``cmp_offset = prefix_len // ratio``).

    Returns:
        A new dataclass instance, or ``None`` when *packed_seq_params* is ``None``.
    """
    if packed_seq_params is None:
        return None

    # cu_seqlens_kv_padded takes priority (MindSpeed >= 2.x may define both).
    kv_attr = (
        "cu_seqlens_kv_padded"
        if hasattr(packed_seq_params, "cu_seqlens_kv_padded")
        else "cu_seqlens_kv"
    )
    old_cu_kv: list[int] = list(getattr(packed_seq_params, kv_attr))

    for batch_idx in range(plan.batch_size):
        if not plan.is_reuser(batch_idx):
            continue
        offset = plan.prefix_lens[batch_idx]
        for i in range(batch_idx + 1, len(old_cu_kv)):
            old_cu_kv[i] += offset

    new_params = replace(packed_seq_params, **{kv_attr: old_cu_kv})

    # cu_seqlens_cmp_kv — same logic with compress_ratio division.
    if hasattr(packed_seq_params, "cu_seqlens_cmp_kv") and packed_seq_params.cu_seqlens_cmp_kv is not None:
        old_cu_cmp: list[int] = list(packed_seq_params.cu_seqlens_cmp_kv)
        for batch_idx in range(plan.batch_size):
            if plan.is_reuser(batch_idx):
                cmp_offset = plan.prefix_lens[batch_idx] // compress_ratio
                for i in range(batch_idx + 1, len(old_cu_cmp)):
                    old_cu_cmp[i] += cmp_offset
        new_params = replace(new_params, cu_seqlens_cmp_kv=old_cu_cmp)

    return new_params
