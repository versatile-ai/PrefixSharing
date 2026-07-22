"""Framework-independent prefix sharing semantics."""

from prefix_sharing.core.batch_trim import TrimmedBatch, trim_batch, trim_inputs, trim_labels, trim_loss_masks
from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.observability import PrefixSharingLayerStats, PrefixSharingStats
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PREFIX_STATE_TYPE_DELTANET_STATE,
    PREFIX_STATE_TYPE_G2_ATTENTION,
    G2AttentionStore,
    PrefixActivationSlotId,
    PrefixActivationStore,
    PrefixAttentionStore,
    PrefixDeltanetStore,
    StoredAttentionKV,
    StoredDeltanetState,
    StoredG2Activation,
)
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner

__all__ = [
    "G2AttentionStore",
    "PREFIX_STATE_TYPE_ATTENTION_KV",
    "PREFIX_STATE_TYPE_DELTANET_STATE",
    "PREFIX_STATE_TYPE_G2_ATTENTION",
    "PrefixActivationSlotId",
    "PrefixActivationStore",
    "PrefixAttentionStore",
    "PrefixDeltanetStore",
    "PrefixDetectionResult",
    "PrefixLastRestoreSpec",
    "PrefixReuseSpec",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixSharingLayerStats",
    "PrefixSharingPlan",
    "PrefixSharingPlanner",
    "PrefixSharingStats",
    "StoredAttentionKV",
    "StoredDeltanetState",
    "StoredG2Activation",
    "TriePrefixDetector",
    "TrimmedBatch",
    "trim_batch",
    "trim_inputs",
    "trim_labels",
    "trim_loss_masks",
]
