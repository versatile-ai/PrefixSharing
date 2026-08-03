"""Model factory for E2E precision tests.

Builds a small DeepSeek4Model (3 layers, random weights) for loss-level
comparison between baseline and PS path.
"""

import sys
import types

import torch


def _ensure_mindspeed_adaptor():
    """Mock mindspeed_llm.megatron_adaptor if not available.

    The adaptor registers MindSpeed patches into Megatron on import.
    For our minimal test we only need the model to build — we don't
    need the full training-side patches.
    """
    mod_name = "mindspeed_llm.megatron_adaptor"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)


def _ensure_features_manager():
    """Mock features_manager to satisfy mindspeed_llm import chain."""
    mod_name = "mindspeed_llm.features_manager"
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        m.FEATURES_LIST = []
        sys.modules[mod_name] = m


def make_e2e_model():
    """Create a 3-layer DeepSeek4Model with random weights.

    Uses ``use_cpu_initialization=True`` so no real checkpoint is needed.
    Returns the model on CPU; call ``.to("npu").to(torch.bfloat16)`` to
    move to NPU.

    The model has:
    - 3 layers: compress_ratios = [0, 128, 128]
    - MoE disabled (num_experts=1, topk=0)
    - hidden=4096, heads=64, seq_len=512
    """
    import os

    from megatron.training.global_vars import get_args

    _ensure_mindspeed_adaptor()
    _ensure_features_manager()

    args = get_args()

    # Use the container's pretrain_deepseek4 model_provider
    # Add the pretrain script's directory to path
    sys.path.insert(0, "/workspace-verl/verl")

    from pretrain_deepseek4 import model_provider

    model = model_provider(pre_process=True, post_process=True)
    model.eval()
    return model
