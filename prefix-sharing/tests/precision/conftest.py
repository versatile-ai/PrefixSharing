"""Shared fixtures for PrefixSharing DeepSeek V4 precision tests.

Fixtures are organised in three tiers:

Tier 1 — Mac (always available)
    Plan / layout / store construction from batch_composition descriptions.
    Does not require NPU or Megatron.

Tier 2 — NPU single-card
    torch.distributed init, Megatron args injection, single-layer
    DeepSeek4SelfAttention construction with random weights.

Tier 3 — NPU multi-card
    Same as Tier 2 but launched via torchrun; parallel_state already
    initialised by the launcher.

All Megatron / torch_npu imports are lazy (inside fixture bodies) so the
module is importable on any machine.
"""

from __future__ import annotations

import pytest

# ──────────────────────────────────────────────────────────────────────
# Tier 1: Mac-compatible plan / layout helpers
# ──────────────────────────────────────────────────────────────────────


# ── batch_composition → concrete sequences ───────────────────────────


def _build_sequences(
    batch_composition: str,
    seqlen: int,
    prefix_len_pct: float,
) -> dict:
    """Translate a *batch_composition* label into concrete sequence definitions.

    Returns a dict with keys:
        batch_size, original_lengths, prefix_lens, provider_index, is_provider,
        expected_suffix_lengths, description

    Definitions (seqlen=512, prefix_len_pct=0.5 → P=256)::

        "1p1r":
            seq0: provider, full 512
            seq1: reuser,  P=256, S=256  (reuses seq0[:256])

        "1p3r":
            seq0: provider, full 512
            seq1: reuser,  P=256, S=256  (reuses seq0[:256])
            seq2: reuser,  P=256, S=256  (reuses seq0[:256])
            seq3: reuser,  P=256, S=256  (reuses seq0[:256])

        "2p3r_mixed":
            seq0: provider_A,  full 512
            seq1: provider_B,  full 512  (different prefix from seq0)
            seq2: reuser (seq0), P=384, S=128  (longer prefix from seq0)
            seq3: reuser (seq0), P=256, S=256  (shorter prefix from seq0)
            seq4: reuser (seq1), P=128, S=384  (different provider)

    The helper generates synthetic token IDs: provider_i uses tokens
    ``[100*i + j for j in range(L)]`` so each provider has a visually
    distinct ID range.  The detector picks up the shared prefix naturally
    when two sequences share the same first *P* tokens.
    """
    P_default = int(seqlen * prefix_len_pct)

    if batch_composition == "1p1r":
        return {
            "batch_size": 2,
            "input_ids": [
                list(range(seqlen)),                     # seq0: 0..511
                list(range(P_default)) + list(range(1000, 1000 + seqlen - P_default)),  # seq1: shared prefix + unique suffix
            ],
            "expected_prefix_lens": [0, P_default],
            "expected_provider_index": [0, 0],
            "expected_is_provider": [True, False],
            "expected_suffix_lengths": [seqlen, seqlen - P_default],
            "description": f"1p1r: P={P_default}, S={seqlen-P_default}",
        }

    elif batch_composition == "1p3r":
        P1, P2, P3 = P_default, P_default, P_default
        return {
            "batch_size": 4,
            "input_ids": [
                list(range(seqlen)),                     # seq0: provider
                list(range(P1)) + list(range(1000, 1000 + seqlen - P1)),  # seq1
                list(range(P2)) + list(range(2000, 2000 + seqlen - P2)),  # seq2
                list(range(P3)) + list(range(3000, 3000 + seqlen - P3)),  # seq3
            ],
            "expected_prefix_lens": [0, P1, P2, P3],
            "expected_provider_index": [0, 0, 0, 0],
            "expected_is_provider": [True, False, False, False],
            "expected_suffix_lengths": [seqlen,
                                        seqlen - P1, seqlen - P2, seqlen - P3],
            "description": f"1p3r: all reuse seq0, P={P_default}",
        }

    elif batch_composition == "2p3r_mixed":
        # P values per the plan: seq0=0, seq1=0, seq2=384, seq3=256, seq4=128
        P2, P3, P4 = 384, 256, 128
        return {
            "batch_size": 5,
            "input_ids": [
                list(range(seqlen)),                     # seq0: provider_A
                list(range(1000, 1000 + seqlen)),        # seq1: provider_B (different prefix)
                list(range(P2)) + list(range(2000, 2000 + seqlen - P2)),  # seq2: reuses seq0[:384]
                list(range(P3)) + list(range(3000, 3000 + seqlen - P3)),  # seq3: reuses seq0[:256]
                list(range(1000, 1000 + P4)) + list(range(4000, 4000 + seqlen - P4)),  # seq4: reuses seq1[:128]
            ],
            "expected_prefix_lens": [0, 0, P2, P3, P4],
            "expected_provider_index": [0, 1, 0, 0, 1],
            "expected_is_provider": [True, True, False, False, False],
            "expected_suffix_lengths": [seqlen, seqlen,
                                        seqlen - P2, seqlen - P3, seqlen - P4],
            "description": f"2p3r_mixed: P=[0,0,{P2},{P3},{P4}] from two providers",
        }

    else:
        raise ValueError(f"Unknown batch_composition: {batch_composition}")


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def seqlen():
    """Default sequence length for precision tests."""
    return 512


@pytest.fixture
def batch_spec(batch_composition: str, seqlen: int, prefix_len_pct: float):
    """Resolve a *batch_composition* label to concrete sequence definitions."""
    return _build_sequences(batch_composition, seqlen, prefix_len_pct)


@pytest.fixture
def g2_plan(batch_spec):
    """Build a :class:`PrefixSharingPlan` from *batch_spec*.

    Uses the real planner with real input_ids (synthetic token ranges).
    This works on Mac — no Megatron needed.
    """
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner

    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True),
    )
    plan = planner.plan(batch_spec["input_ids"])
    return plan


@pytest.fixture
def g2_store():
    """An empty :class:`G2AttentionStore`."""
    from prefix_sharing.core.prefix_store import G2AttentionStore

    return G2AttentionStore()


# ──────────────────────────────────────────────────────────────────────
# Tier 2: NPU-dependent fixtures (lazy imports)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def init_distributed():
    """Initialise torch.distributed + Megatron parallel_state.

    Single-card: manually call init_process_group and initialize_model_parallel.
    Multi-card (torchrun): already initialised; this fixture is a no-op when
    torch.distributed.is_initialized() returns True.

    Sets ``ASCEND_CUSTOM_OPP_PATH`` for CANN 9.1 pre-built sparse_flash_mla.
    """
    import os
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH",
                          "/usr/local/Ascend/vendors/custom_transformer")

    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(
                backend="hccl",
                world_size=1,
                rank=0,
            )
        except Exception:
            try:
                torch.distributed.init_process_group(
                    backend="gloo",
                    world_size=1,
                    rank=0,
                )
            except Exception:
                pytest.skip("Cannot init process_group (no hccl, no gloo)")

    try:
        from megatron.core import parallel_state

        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
    except ImportError:
        pytest.skip("megatron.core not available — NPU environment required")
    except Exception as exc:
        pytest.skip(f"Megatron init failed: {exc}")


@pytest.fixture
def megatron_args(init_distributed, compress_ratio: int, seqlen: int):
    """Inject minimal Megatron args for DeepSeek4SelfAttention.

    Sets ``megatron.training.global_args._GLOBAL_ARGS`` so that
    ``get_args()`` inside the attention module returns a usable object.

    Key fields (from DeepSeek4SelfAttention.__init__):
        - qk_head_dim, rope_head_dim
        - q_lora_rank, o_lora_rank
        - hidden_size, num_attention_heads, o_groups
        - g2_window_size
        - compress_ratios (per-layer)
        - use_sparse_flash_attn, use_fused_rmsnorm
        - use_fused_lightning_indexer_loss, use_g2_indexer_loss
    """
    try:
        from megatron.training import get_args as _original_get_args
    except ImportError:
        pytest.skip("megatron.training not available — NPU environment required")

    import torch

    # Build a minimal args namespace that satisfies DeepSeek4SelfAttention.__init__
    class _MinimalArgs:
        qk_head_dim = 512
        rope_head_dim = 64
        q_lora_rank = 1024
        o_lora_rank = 1024
        hidden_size = 4096
        num_attention_heads = 64
        o_groups = 8
        g2_window_size = 128
        compress_ratios = [compress_ratio]  # single layer
        use_sparse_flash_attn = True
        use_fused_rmsnorm = True
        use_fused_lightning_indexer_loss = False  # disable for precision testing
        use_g2_indexer_loss = False
        use_fp8_padding = False
        use_causal_mask = True
        add_qkv_bias = False
        seq_length = seqlen
        max_position_embeddings = seqlen
        attention_dropout = 0.0
        hidden_dropout = 0.0
        num_layers = 1
        context_parallel_size = 1
        context_parallel_algo = "none"
        transformer_impl = "transformer_engine"
        norm_epsilon = 1e-6
        norm_eps = 1e-6
        swiglu = False
        untie_embeddings_and_output_weights = False
        init_method_std = 0.02
        params_dtype = torch.float16

    # Patch the global args — Megatron stores args in a module-level _GLOBAL_ARGS
    try:
        from megatron.training import global_vars as _args_mod
        _args_mod._GLOBAL_ARGS = _MinimalArgs()
    except (ImportError, AttributeError):
        try:
            import megatron.training as _mtt
            _mtt._GLOBAL_ARGS = _MinimalArgs()
        except (ImportError, AttributeError):
            # Last resort: monkey-patch get_args to return our object
            import megatron.training
            megatron.training.get_args = lambda: _MinimalArgs()

    return _MinimalArgs()


@pytest.fixture
def make_attention_module(megatron_args, compress_ratio: int):
    """Construct a single-layer :class:`DeepSeek4SelfAttention` with random weights.

    Returns a factory function ``make(compress_ratio, layer_number=1)``
    so that different compress_ratio values can be tested with fresh weights.
    """

    def _make(ratio: int | None = None, layer_number: int = 1):
        import torch

        _ratio = ratio if ratio is not None else compress_ratio

        try:
            from megatron.core.transformer.transformer_config import TransformerConfig
        except ImportError:
            pytest.skip("megatron.core not available")

        # Update compress_ratios for this layer_number
        megatron_args.compress_ratios = [_ratio]
        megatron_args.num_layers = layer_number

        config = TransformerConfig(
            num_layers=megatron_args.num_layers,
            hidden_size=megatron_args.hidden_size,
            num_attention_heads=megatron_args.num_attention_heads,
            num_query_groups=megatron_args.num_attention_heads,
            kv_channels=megatron_args.qk_head_dim,
            layernorm_epsilon=megatron_args.norm_epsilon,
            attention_dropout=megatron_args.attention_dropout,
            hidden_dropout=megatron_args.hidden_dropout,
            add_qkv_bias=megatron_args.add_qkv_bias,
            gated_linear_unit=megatron_args.swiglu,
            seq_length=megatron_args.seq_length,
            max_position_embeddings=megatron_args.max_position_embeddings,
            apply_rope_fusion=False,
            use_cpu_initialization=True,  # avoid NPU device init
        )

        try:
            from mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention import (
                DeepSeek4SelfAttention,
            )
        except ImportError:
            pytest.skip("mindspeed_llm not available — NPU container required")

        attn = DeepSeek4SelfAttention(
            config=config,
            submodules=None,
            layer_number=layer_number,
        )

        # Move to NPU in FP16 — sparse_flash_mla kernel requires FP16/BF16
        device = "cpu"
        try:
            import torch_npu
            if torch.npu.is_available():
                device = "npu:0"
                attn = attn.to(device).to(torch.float16)
        except (ImportError, RuntimeError):
            pass

        return attn

    return _make


@pytest.fixture
def make_batch_inputs(megatron_args, g2_plan, seqlen: int):
    """Construct packed attention inputs from a :class:`PrefixSharingPlan`.

    Returns a dict with keys:
        hidden_states, attention_mask, rotary_pos_emb, packed_seq_params

    All tensors have ``requires_grad=True`` where applicable.
    """
    import torch

    hidden_size = megatron_args.hidden_size

    def _make():
        # THD format: batch_size=1, cu_seqlens handle sequence boundaries
        total_q = sum(g2_plan.kept_lengths_q)
        hidden = torch.randn(total_q, 1, hidden_size,
                            dtype=torch.float16, requires_grad=True)

        # Attention mask — shape depends on Megatron convention
        # For packed sequences: [1, 1, total_q, total_kv] or bool mask
        total_kv = sum(g2_plan.original_lengths)
        attn_mask = torch.zeros(1, 1, total_q, total_kv)

        # rotary_pos_emb — freqs_cis for NPU
        # DeepSeek4SelfAttention expects rotary_pos_emb as a tuple
        #   (local_freqs_cis, global_freqs_cis) for compress_ratio > 1
        rotary_pos_emb = (
            torch.randn(seqlen, megatron_args.rope_head_dim // 2, 2),
            torch.randn(seqlen, megatron_args.rope_head_dim // 2, 2),
        )

        # packed_seq_params — Megatron dataclass
        try:
            from megatron.core.packed_seq_params import PackedSeqParams
            psp = PackedSeqParams(
                cu_seqlens_q=torch.tensor(g2_plan.cu_seqlens_q, dtype=torch.int32),
                cu_seqlens_kv=torch.tensor(g2_plan.cu_seqlens_q, dtype=torch.int32),  # trimmed, Hook adjusts
                cu_seqlens_q_padded=None,
                cu_seqlens_kv_padded=None,
                max_seqlen_q=torch.tensor(g2_plan.max_seqlen_q, dtype=torch.int32),
                max_seqlen_kv=torch.tensor(g2_plan.max_seqlen_kv, dtype=torch.int32),
                qkv_format="thd",
            )
        except ImportError:
            # Fallback: minimal dataclass — dataclasses.replace() in
            # _adjust_cu_seqlens_for_batch requires a real dataclass instance
            # (SimpleNamespace would raise TypeError).
            from dataclasses import dataclass

            @dataclass
            class _MinimalPackedSeqParams:
                cu_seqlens_kv: list
                cu_seqlens_q: list
                cu_seqlens_kv_padded: list | None = None
                cu_seqlens_cmp_kv: list | None = None

            psp = _MinimalPackedSeqParams(
                cu_seqlens_kv=g2_plan.cu_seqlens_q,  # trimmed, Hook adjusts
                cu_seqlens_q=g2_plan.cu_seqlens_q,
            )

        return {
            "hidden_states": hidden,
            "attention_mask": attn_mask,
            "rotary_pos_emb": rotary_pos_emb,
            "packed_seq_params": psp,
        }

    return _make
