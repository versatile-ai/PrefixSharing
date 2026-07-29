"""Minimal precision test — verify A≈B on NPU with THD packed format."""
import pytest
import torch
import types, os

torch = pytest.importorskip("torch")

from prefix_sharing.setup.patches.mindspeed_deepseek4.attention import patch_g2_attention
from prefix_sharing.core.prefix_store import G2AttentionStore
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


@pytest.fixture
def _npu_only():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("torch_npu not available")


@pytest.mark.parametrize("compress_ratio", [0, 128])
def test_ps_no_sharing_noop(init_distributed, megatron_args, compress_ratio, seqlen):
    """When no sharing is detected, PS path should be a no-op (A≈C)."""
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH",
                          "/usr/local/Ascend/vendors/custom_transformer")

    # Plan: all unique sequences — no sharing
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    input_ids = [list(range(128)), list(range(1000, 1128)), list(range(2000, 2128))]
    plan = planner.plan(input_ids)

    if plan.has_sharing:
        pytest.skip("Unexpected sharing detected")

    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    store = G2AttentionStore()
    pinfo = MegatronParallelInfo()

    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan, attention_backend=None,
        packed_batch_layout=layout, parallel_info=pinfo,
        model_type="deepseek4",
    )

    # Build attention module
    from mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention import (
        get_deepseek4_self_attn_submodules, DeepSeek4SelfAttention)
    from megatron.core.transformer import TransformerConfig
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    get_cuda_rng_tracker(inference_rng_tracker=True)
    config = TransformerConfig(
        hidden_size=megatron_args.hidden_size,
        num_attention_heads=megatron_args.num_attention_heads,
        num_layers=1, gated_linear_unit=False,
        apply_rope_fusion=False, use_cpu_initialization=True,
        normalization="RMSNorm", layernorm_epsilon=1e-6,
    )
    for a, v in [("use_fused_rmsnorm", True), ("layernorm_zero_centered_gamma", False)]:
        if not hasattr(config, a) or getattr(config, a) is None:
            object.__setattr__(config, a, v)

    sm = get_deepseek4_self_attn_submodules(
        qk_layernorm=True, mla_mm_split=False, enable_dsa_indexer=False,
        compressor=(compress_ratio > 1))

    # Create model once, baseline first
    megatron_args.compress_ratios = [compress_ratio]
    torch.manual_seed(42)
    attn = DeepSeek4SelfAttention(config=config, submodules=sm, layer_number=1)
    attn.eval()
    attn = attn.to("npu").to(torch.float16)

    # Build THD input
    total_q = sum(plan.kept_lengths_q)
    hidden = torch.randn(total_q, 1, megatron_args.hidden_size,
                         device="npu", dtype=torch.float16)

    from dataclasses import dataclass
    @dataclass
    class _Psp:
        cu_seqlens_q: object; cu_seqlens_kv: object; cu_seqlens_kv_padded: object = None
        cu_seqlens_cmp_kv: object = None; max_seqlen_q: int = 0; max_seqlen_kv: int = 0
        qkv_format: str = "thd"

    psp = _Psp(
        cu_seqlens_q=torch.tensor(plan.cu_seqlens_q, dtype=torch.int32, device="npu"),
        cu_seqlens_kv=torch.tensor(plan.cu_seqlens_q, dtype=torch.int32, device="npu"),
        max_seqlen_q=plan.max_seqlen_q, max_seqlen_kv=plan.max_seqlen_kv,
    )

    # RoPE
    rope_hd = megatron_args.rope_head_dim
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, rope_hd, 2, device="npu").float() / rope_hd))
    t = torch.arange(4096, device="npu").float()
    fre = torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))

    # Path A: baseline (original forward)
    with torch.no_grad():
        o_A, _ = attn(hidden_states=hidden, attention_mask=None,
                       rotary_pos_emb=[fre, fre], start_pos=0, packed_seq_params=psp)
    assert not torch.isnan(o_A).any(), "Baseline has NaN"

    # Path B: patch no-op (context empty → falls through to original_forward)
    attn.forward = types.MethodType(patch_g2_attention(type(attn).forward), attn)
    with torch.no_grad():
        o_B, _ = attn(hidden_states=hidden, attention_mask=None,
                       rotary_pos_emb=[fre, fre], start_pos=0, packed_seq_params=psp)
    assert not torch.isnan(o_B).any(), "Patched no-op has NaN"

    # A≈B (should be bitwise identical for compress_ratio=0, close for ratio>0)
    max_diff = (o_A.float() - o_B.float()).abs().max().item()
    assert max_diff < 1e-3, f"A≠B: max diff = {max_diff}"

    # Path C: PS activated (no sharing → should also be no-op)
    with prefix_sharing_runtime_context(state) as ctx:
        with torch.no_grad():
            o_C, _ = attn(hidden_states=hidden, attention_mask=None,
                           rotary_pos_emb=[fre, fre], start_pos=0, packed_seq_params=psp)
    assert not torch.isnan(o_C).any(), "PS activated has NaN"
    max_diff_C = (o_A.float() - o_C.float()).abs().max().item()
    assert max_diff_C < 1e-3, f"A≠C: max diff = {max_diff_C}"
