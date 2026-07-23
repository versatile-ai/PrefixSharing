"""Attention backend adapters."""

from prefix_sharing.backends.base import BackendCapabilities, PrefixAttentionBackend, PrefixDeltanetBackend
from prefix_sharing.backends.block_causal_mask import build_block_causal_mask
from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin, FlashBackendValidationError
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
from prefix_sharing.backends.g2_attention_utils import (
    _adjust_cu_seqlens_for_batch,
    _compute_cmp_lengths,
    _merge_g2_fields,
    _split_by_cu_seqlens,
)
from prefix_sharing.backends.torch_ref import TorchReferenceBackend

__all__ = [
    "BackendCapabilities",
    "FlashAttentionMixin",
    "FlashBackendValidationError",
    "GpuFlashAttentionBackend",
    "NpuFlashAttentionBackend",
    "PrefixAttentionBackend",
    "PrefixDeltanetBackend",
    "TorchReferenceBackend",
    "_adjust_cu_seqlens_for_batch",
    "_compute_cmp_lengths",
    "_merge_g2_fields",
    "_split_by_cu_seqlens",
    "build_block_causal_mask",
    "get_backend_instance",
]
