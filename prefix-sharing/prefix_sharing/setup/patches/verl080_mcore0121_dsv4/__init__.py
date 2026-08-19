"""verl 0.8.0.dev + megatron-core 0.12.1 + MindSpeed-LLM（DeepSeek V4 容器）

v25 容器配套：deepseek-rl:910b-cann9.1-vllm0.23-v25
(verl=0.8.0.dev, megatron-core=0.12.1, mindspeed 版本不可探测)

Patch 目标：
1. MegatronEngineWithLMHead.forward_step → 微批次重组 + runtime context 注入
   （MindSpeedLLMEngineWithLMHead 继承该方法，DSv4 mindspeed 引擎同样命中）
2. DeepSeek4SelfAttention.forward         → G2 KV store/expand hook
   （DSv4 专属，替代 verl080 里的 mcore Attention patch——DSv4 不经过 mcore Attention）
3. vocab_parallel_log_probs_from_logits  → 自动 logprob restore
4. no_padding_2_padding                  → PS 物理裁剪后修正序列长度
   (module-level + 所有 from...import 引用)

与 verl080_mcore0161_ms0160 的差异：
- 去掉 mcore Attention.forward patch（Qwen 系标准注意力专属）
- 加入 DeepSeek4SelfAttention.forward patch（来自 mindspeed_deepseek4 patch set）

所有业务逻辑由 integrations 层处理，本 patch set 只负责 thin wrapper 编排。
"""

from prefix_sharing.setup.registry import PatchSpec
from .forward_step import patch_verl_forward_step
from .g2_attention import patch_g2_attention
from .vocab_logprobs import patch_megatron_vocab
from .nopadding import patch_no_padding_2_padding

# no_padding_2_padding 被 4 个模块用 from...import 直接引用：
#   verl.workers.utils.padding           — 原定义模块
#   verl.workers.utils.losses            — ppo_loss 内调用
#   verl.trainer.distillation.losses     — distillation 内调用
#   verl.trainer.ppo.ray_trainer         — trainer 侧调用
# from...import 创建的是模块级属性，setattr 可以更新。
# 必须对每个引用模块都 patch，否则该模块的局部引用仍指向原函数。
_NOPADDING_PATCH_MODULES = [
    "verl.workers.utils.padding",
    "verl.workers.utils.losses",
    "verl.trainer.distillation.losses",
    "verl.trainer.ppo.ray_trainer",
]

def patch_verl_forward_backward_batch(original_forward_backward_batch):
    """[PS-fix14 改动A] 批末(或异常路径)关闭 PS runtime ctx。

    forward_step patch(改动B)把 ctx 的生命周期改为手动管理,ctx 不再随
    forward_step 返回而关闭;最后一个 micro-batch 的 ctx 在此处收尾,
    并覆盖异常路径(正常路径的 ctx 已在下一 mb 入口被关闭,这里幂等)。
    """
    def patched_forward_backward_batch(self, data, loss_function, forward_only=False):
        try:
            return original_forward_backward_batch(
                self, data, loss_function, forward_only)
        finally:
            try:
                from prefix_sharing.integrations.context import _ps_close_context
                _ps_close_context()
            except Exception:
                pass
    return patched_forward_backward_batch


PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "MegatronEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=patch_verl_forward_step,
        description="MegatronEngineWithLMHead.forward_step → "
                    "micro-batch reorg + context (verl 0.8.0 engine)",
    ),
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "MegatronEngineWithLMHead"),
            "forward_backward_batch",
        ),
        patch_factory=patch_verl_forward_backward_batch,
        description="MegatronEngineWithLMHead.forward_backward_batch → "
                    "batch-end ctx close (fix14: ctx 覆盖 backward 重放)",
    ),
    PatchSpec(
        module_name="mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention",
        target_getter=lambda mod: (
            getattr(mod, "DeepSeek4SelfAttention"),
            "forward",
        ),
        patch_factory=patch_g2_attention,
        description="DeepSeek4SelfAttention.forward → G2 KV store/expand "
                    "(DeepSeek V4, 替代 mcore Attention patch)",
    ),
    # verl080 算 logprob 的真正调用点在 transformer_impl 的 logits_processor
    # 闭包里（本容器为 transformer_impl.py:916），该名字是模块加载时
    # from...import 绑定的局部引用。只 patch 源模块
    # verl.utils.megatron.tensor_parallel 无法命中（setattr 源模块属性不会改
    # transformer_impl 的局部引用），必须直接 patch transformer_impl 模块属性。
    #
    # 注意：不要同时 patch 源模块。restore 侧重算 prefix-last logp 时
    # （forward_step.py 内 from verl.utils.megatron.tensor_parallel import）
    # 需要拿原始函数；若源模块被 patch，重算会误入 patched_fn——传入 logits 仅
    # [1, V//tp]，而 index.provider_1d_pos 是全局 packed 偏移，切片为空会触发
    # vocab_logprobs.py 的 empty-slice RuntimeError。
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (mod, "vocab_parallel_log_probs_from_logits"),
        patch_factory=patch_megatron_vocab,
        description="vocab_parallel_log_probs → auto logprob restore (verl 0.8.0)",
    ),
] + [
    PatchSpec(
        module_name=mod_name,
        target_getter=lambda mod: (mod, "no_padding_2_padding"),
        patch_factory=patch_no_padding_2_padding,
        description=f"no_padding_2_padding in {mod_name} → PS trimming-aware",
    )
    for mod_name in _NOPADDING_PATCH_MODULES
]
