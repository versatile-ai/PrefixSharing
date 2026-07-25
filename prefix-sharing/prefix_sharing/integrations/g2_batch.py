"""DeepSeek V4 standalone pretrain training integration.

Wraps MindSpeed pretrain ``forward_step`` with prefix detection,
batch trimming, runtime context injection, and prefix-last restore.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import (
    PrefixSharingPlanner,
    align_prefix_lens_to_compression,
)
from prefix_sharing.core.batch_trim import trim_batch
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState


def _build_runtime_state(plan, layout, model_type="deepseek4") -> PrefixSharingRuntimeState:
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=layout,
        parallel_info=get_megatron_parallel_info(),
        model_type=model_type,
    )


def wrap_forward_step(
    original_forward_step,
    ps_config: PrefixSharingConfig,
    *,
    get_batch_fn=None,
):
    """Wrap MindSpeed pretrain ``forward_step`` with prefix sharing.

    The wrapped function intercepts batch data before model forward,
    detects shared prefixes, trims reuser sequences to suffix-only,
    injects runtime context for the attention patch, and restores
    prefix-last logprobs after forward.

    Args:
        original_forward_step: MindSpeed pretrain ``forward_step``
            function ``(data_iterator, model) → (output, loss_func)``.
        ps_config: Validated :class:`PrefixSharingConfig`.
        get_batch_fn: Optional override for extracting tensors from
            ``data_iterator``.  Defaults to importing
            ``megatron.training.training.get_batch``.

    Returns:
        Wrapped ``forward_step`` with prefix sharing logic.
    """

    if get_batch_fn is None:
        def _default_get_batch(data_iterator):
            from megatron.training.training import get_batch
            return get_batch(data_iterator)
        get_batch_fn = _default_get_batch

    planner = PrefixSharingPlanner(ps_config)

    def wrapped(data_iterator, model):
        # 1. Get batch
        batch = get_batch_fn(data_iterator)
        if batch is None:
            return original_forward_step(data_iterator, model)

        tokens, labels, loss_mask, attention_mask, position_ids = batch

        # 2. Detect prefixes
        input_ids_list = _extract_input_ids(tokens, attention_mask)
        plan = planner.plan(input_ids_list)

        if not plan.has_sharing:
            return original_forward_step(data_iterator, model)

        # 3. Align prefix_lens to compress_ratio (DeepSeek V4 only)
        try:
            from megatron.training import get_args
            compress_ratios = getattr(get_args(), "compress_ratios", None)
            if compress_ratios:
                align_prefix_lens_to_compression(plan, compress_ratios)
                if not plan.has_sharing:  # alignment may have zeroed all prefix_lens
                    return original_forward_step(data_iterator, model)
        except (ImportError, RuntimeError):
            pass

        # 4. Trim batch (reuser → suffix-only)
        trimmed = trim_batch(
            tokens, labels, loss_mask, attention_mask, position_ids, plan)
        trimmed_tokens, trimmed_labels, trimmed_loss_mask, trimmed_attn_mask, trimmed_pos = trimmed

        # 5. Build layout and runtime state
        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
        state = _build_runtime_state(plan, layout)

        # 5. Build trimmed batch and run forward
        output = _run_with_context(
            original_forward_step, state, model,
            trimmed_tokens, trimmed_labels, trimmed_loss_mask,
            trimmed_attn_mask, trimmed_pos, data_iterator)

        # 6. Restore prefix-last logprobs
        output = _restore_prefix_last(output, plan, layout)

        return output

    return wrapped


def _extract_input_ids(tokens, attention_mask):
    """Extract per-sequence input_ids from padded batch tensors."""
    input_ids_list = []
    for i in range(tokens.shape[0]):
        valid = attention_mask[i].nonzero(as_tuple=False).flatten()
        if len(valid) > 0:
            input_ids_list.append(tokens[i, valid].detach().cpu().tolist())
    return input_ids_list


def _run_with_context(original_forward_step, state, model,
                       tokens, labels, loss_mask, attn_mask, pos_ids,
                       data_iterator):
    """Run forward inside prefix sharing context.

    Temporarily replaces ``get_batch`` to return pre-computed batch
    so that ``original_forward_step`` picks up the trimmed data.
    """
    import megatron.training.training as mtt
    original_get_batch = mtt.get_batch

    def trimmed_get_batch(data_iterator):
        return tokens, labels, loss_mask, attn_mask, pos_ids

    mtt.get_batch = trimmed_get_batch
    try:
        with prefix_sharing_runtime_context(state) as ctx:
            return original_forward_step(data_iterator, model)
    finally:
        mtt.get_batch = original_get_batch


def _restore_prefix_last(output, plan, layout):
    """Restore prefix-last logprobs — standalone pretrain adapter.

    NOTE: ``restore_reuser_prefix_columns_2d`` expects verl-format output dict
    with ``log_probs``/``entropy`` keys and callable vocab-logprob functions.
    Standalone pretrain outputs ``(output_tensor, loss_func)``.
    This adapter will be completed when the full standalone training loop
    is integrated (8-card E2E test phase).
    """
    if not plan.prefix_last_restore:
        return output
    # TODO: implement standalone restore counterpart
    # - reconstruct per-token logprobs from packed output_tensor
    # - bulk-copy interior prefix columns
    # - recompute prefix-last using saved provider logits + reuser label
    return output
