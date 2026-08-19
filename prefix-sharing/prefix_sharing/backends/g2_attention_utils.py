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
        indexer_k=new_tensor if field == "indexer_k" else existing.indexer_k,
        stored_len=max(existing.stored_len, new_tensor.shape[0]),
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
    """Compute per-sequence compressed-KV lengths, adjusted for TP padding.

    With TP>1 and ``sequence_parallel=True`` the compressor TP-pads each
    rank's output before ``gather_from_sp_cp`` concatenates them.  The
    total ``kv_compress.shape[0]`` may be larger than ``sum(valid//ratio)``.
    We distribute the excess padding to the last sequence's length so that
    ``_split_by_cu_seqlens`` can split the tensor without error.

    Args:
        layout: :class:`PackedBatchLayout` whose ``valid_lengths`` are used.
        compress_ratio: Compression ratio (e.g. 128).
        kv_compress_shape_0: Actual ``kv_compress.shape[0]``.

    Returns:
        Per-sequence compressed lengths.  The last entry may include TP
        padding so the sum matches *kv_compress_shape_0*.
    """
    lengths = [vl // compress_ratio for vl in layout.valid_lengths]
    computed_sum = sum(lengths)
    remainder = kv_compress_shape_0 - computed_sum
    if remainder > 0:
        lengths[-1] += remainder
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

    import torch

    # [PS-fix8] sparse MLA 读的是非 padded 的 cu_seqlens_kv,而模型两个字段都设置了。
    # 只调 padded(旧优先级逻辑)会让注意力拿到未调整的 [0,2304,2688],
    # 与展开后的 KV 表(4608)不匹配 → cmp cu_seqlens 越界 → 稀疏注意力 NaN。
    # 因此两个字段必须同时调整。
    def _shift_cu(raw: torch.Tensor) -> torch.Tensor:
        vals: list[int] = raw.tolist()  # [PS-fix6] 必须是 Python int 副本
        for batch_idx in range(plan.batch_size):
            if not plan.is_reuser(batch_idx):
                continue
            offset = plan.prefix_lens[batch_idx]
            for i in range(batch_idx + 1, len(vals)):
                vals[i] += offset
        return torch.tensor(vals, dtype=raw.dtype, device=raw.device)

    old_cu_kv_raw = getattr(packed_seq_params, "cu_seqlens_kv", None)
    if old_cu_kv_raw is None:
        return packed_seq_params  # nothing to adjust
    _kv_is_tensor = isinstance(old_cu_kv_raw, torch.Tensor)
    old_cu_kv: list[int] = old_cu_kv_raw.tolist()

    for batch_idx in range(plan.batch_size):
        if not plan.is_reuser(batch_idx):
            continue
        offset = plan.prefix_lens[batch_idx]
        for i in range(batch_idx + 1, len(old_cu_kv)):
            old_cu_kv[i] += offset

    new_kwargs = {"cu_seqlens_kv":
        torch.tensor(old_cu_kv, dtype=old_cu_kv_raw.dtype, device=old_cu_kv_raw.device)
        if _kv_is_tensor else old_cu_kv}
    _padded_raw = getattr(packed_seq_params, "cu_seqlens_kv_padded", None)
    if _padded_raw is not None:
        new_kwargs["cu_seqlens_kv_padded"] = _shift_cu(_padded_raw)
    new_params = replace(packed_seq_params, **new_kwargs)

    # cu_seqlens_cmp_kv — same logic with compress_ratio division.
    if hasattr(packed_seq_params, "cu_seqlens_cmp_kv") and packed_seq_params.cu_seqlens_cmp_kv is not None:
        old_cu_cmp_raw = packed_seq_params.cu_seqlens_cmp_kv
        _cmp_is_tensor = isinstance(old_cu_cmp_raw, torch.Tensor)
        # [PS-fix6] 同上:list(tensor) → 标量视图原地改;必须 tolist()
        old_cu_cmp: list[int] = old_cu_cmp_raw.tolist()
        for batch_idx in range(plan.batch_size):
            if plan.is_reuser(batch_idx):
                cmp_offset = plan.prefix_lens[batch_idx] // compress_ratio
                for i in range(batch_idx + 1, len(old_cu_cmp)):
                    old_cu_cmp[i] += cmp_offset
        new_params = replace(new_params, cu_seqlens_cmp_kv=
            torch.tensor(old_cu_cmp, dtype=old_cu_cmp_raw.dtype, device=old_cu_cmp_raw.device)
            if _cmp_is_tensor else old_cu_cmp)

    return new_params
