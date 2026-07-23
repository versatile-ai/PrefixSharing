"""MindSpeed DeepSeek4 standalone pretrain patch set.

Patch 目标:
    DeepSeek4SelfAttention.forward → prefix-sharing KV store/expand hook

所有业务逻辑由 integrations 层处理，本 patch set 只负责 thin wrapper 编排。
"""

from prefix_sharing.setup.registry import PatchSpec
from .attention import patch_g2_attention

PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention",
        target_getter=lambda mod: (
            getattr(mod, "DeepSeek4SelfAttention"), "forward"),
        patch_factory=patch_g2_attention,
        description="DeepSeek4SelfAttention.forward → prefix-sharing KV store/expand",
    ),
]
