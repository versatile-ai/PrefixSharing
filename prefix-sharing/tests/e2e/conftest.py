"""E2E shared fixtures: distributed init, Megatron args, model factory, mock data."""

import os
import types

import pytest
import torch


# ── Session fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def init_distributed():
    """Initialise torch.distributed + Megatron parallel_state.

    Single-card: manual init.  Multi-card (torch.distributed.launch):
    already initialised.
    """
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH",
                          "/usr/local/Ascend/vendors/custom_transformer")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29600")

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    import torch_npu  # noqa: F401 — verify NPU available

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hccl")

    from megatron.core import parallel_state

    # Read TP/CP from env (set by torch.distributed.launch) or default to 1
    tp = int(os.environ.get("TP_SIZE", 1))
    cp = int(os.environ.get("CP_SIZE", 1))

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )

    yield

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        parallel_state.destroy_model_parallel()


@pytest.fixture(scope="session")
def set_e2e_args(init_distributed):
    """Inject minimal Megatron training args for DeepSeek4Model."""
    from megatron.training.global_vars import set_args
    from argparse import Namespace

    from mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention import (
        DeepSeek4SelfAttention,
    )

    # We can't import all symbol-heavy modules through minimal args.
    # Use a _LazyArgs that returns None for unknown fields (same as Task 2).
    class _LazyArgs(Namespace):
        _defaults = {
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "seq_length": 512,
            "max_position_embeddings": 512,
            "add_bias_linear": False,
            "add_qkv_bias": False,
            "use_cpu_initialization": True,
            "bf16": True,
            "fp16": False,
            "params_dtype": torch.bfloat16,
            "use_fused_rmsnorm": True,
            "use_fused_lightning_indexer_loss": False,
            "use_g2_indexer_loss": False,
            "use_sparse_flash_attn": True,
            "context_parallel_size": int(os.environ.get("CP_SIZE", 1)),
            "context_parallel_algo": "",
            "tensor_model_parallel_size": int(os.environ.get("TP_SIZE", 1)),
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "use_moe": False,
            "num_experts": 1,
            "num_moe_experts": 1,
            "moe_router_topk": 0,
            "moe_router_num_experts": 1,
            "moe_grouped_gemm": False,
            "make_vocab_size_divisible_by": 128,
            "padded_vocab_size": 32000,
            "num_layers": 3,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "ffn_hidden_size": 11008,
            "gated_linear_unit": True,
            "transformer_impl": "local",
            "position_embedding_type": "rope",
            "no_rope_freqs_schedule": False,
            "original_seq_len": 0,
            "enable_mhc": False,
            "untie_embeddings_and_output_weights": False,
            "schedules_method": "",
            "use_global_aux_loss": False,
            "overlap_p2p_comm": False,
            "spec": "",
            "recompute_granularity": None,
            "recompute_method": None,
            "recompute_num_layers": None,
            "activations_checkpoint_method": None,
            "activations_checkpoint_granularity": None,
            "activations_checkpoint_num_layers": None,
            "distribute_saved_activations": False,
            "data_parallel_random_init": False,
            "sequence_parallel": False,
            "use_distributed_optimizer": False,
            "train_iters": 1,
            "eval_iters": 1,
            "use_mtp": False,
            "mtp_num_layers": None,
            "num_experts_list": None,
            "moe_permute_fusion": False,
            "enable_dsa_indexer": False,
            "mla_mm_split": False,
        }

        def __getattr__(self, name):
            if name in self._defaults:
                return self._defaults[name]
            return None

    args = _LazyArgs(
        qk_head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        g2_window_size=128,
        compress_ratios=[0, 128, 128],
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_factor=40,
        beta_fast=32,
        beta_slow=1,
        rope_scaling_original_max_position_embeddings=4096,
        norm_eps=1e-6,
    )
    set_args(args)
    return args
