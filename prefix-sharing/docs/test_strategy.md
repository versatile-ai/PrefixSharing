# DeepSeek V4 PrefixSharing 测试策略

> **原则**：TDD 优先，逐层验证。每个功能组完成开发并通过全部验证用例后，再进入下一组。

## 环境矩阵

| 环境 | 资源 | 可用性 | 适用场景 |
|------|------|:--:|------|
| **Mac 本地** | CPU-only, PyTorch 2.12 | 随时 | 纯张量逻辑、store 生命周期、工具函数 |
| **NPU 单卡** | 1×910B3, 64GB HBM | 需要 NPU 服务器 | 单层/减层 attention 等价性验证 |
| **NPU 8 卡** | 8×910B3, TP=1/PP=1/EP=1 | 需要 NPU 服务器 | 完整 43 层模型端到端精度验证 |

**NPU 服务器**：`192.168.0.112`（通过 `190.92.241.16` 跳板）
**容器**：`verl-qwen-prefix-baseline`（verl-9.0.0 + MindSpeed + Megatron-LM + PyTorch 2.9.0）

## 各组测试策略

### 总览

```
A ──→ B1 ──→ B2 ──→ B3 ──→ B4 ──→ C ──→ D ──→ E
 ✅     ✅     ✅    ⬜      ⬜      ⬜     ⬜     ⬜
```

| 组 | Mac 测试 | NPU 单卡测试 | NPU 8 卡测试 | 所需模型 |
|----|:--:|:--:|:--:|------|
| A | 5 | — | — | 无 |
| B1 | 3 | — | — | 无 |
| B2 | 6 | — | — | 无 |
| B3 | 3 | 5 | — | 1 层 Attention（ratio=128） |
| B4 | 4 | 3 | — | 4 层模型（ratio≤1/4/128） |
| C | 3 | 7 | — | 2-3 层含 ratio=4 + DSA Indexer |
| D | 2 | 4 | — | 2-3 层含 MHC |
| E | — | — | 6 | 完整 43 层（关闭 MoE） |
| **合计** | **26** | **19** | **6** | |

### 功能组 A：数据存储层

**Mac**（不依赖 MindSpeed）：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_g2_store_lifecycle` | store/load/overwrite/close + type guard |
| 2 | `test_g2_store_incremental` | 三次增量 store 后完整数据保留 |
| 3 | `test_merge_g2_fields` | 逐字段更新，旧字段保留 |
| 4 | `test_merge_g2_transformer_fields` | attention 字段保留，transformer 字段更新 |
| 5 | `test_create_store_deepseek4` | 工厂按 model_type 选择正确 store 类型 |

### 功能组 B1：工具函数扩展

**Mac**（纯张量操作）：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_split_by_cu_seqlens` | packed tensor 拆分 + length mismatch |
| 2 | `test_compute_cmp_lengths` | per-sequence 压缩长度计算 + 断言 |
| 3 | `test_adjust_cu_seqlens_for_batch` | cu_seqlens_kv/cmp 偏移 + padded 优先级 |

### 功能组 B2：存储/扩展辅助函数

**Mac**（mock context + 合成张量）：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_g2_store_with_kwargs` | StoredG2Activation → store 往返 |
| 2 | `test_g2_store_per_sequence` | provider kv 正确存入，non-provider 跳过 |
| 3 | `test_g2_store_per_sequence_none` | None tensor no-op |
| 4 | `test_g2_store_per_sequence_incremental` | 两次 store（kv → attn_o）字段保留 |
| 5 | `test_g2_expand_attn_output` | reuser attn_o 拼接 provider prefix |
| 6 | `test_transitive_reuse_store` | Reuser A 扩展后 Reuser B 复用 |

### 功能组 B3：KV 扩展 + topk 调整

**Mac**（mock attention_module，验证 shape 逻辑）：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_kv_expand_shape` | KV/CMP KV cat 后 dim=0 正确 |
| 2 | `test_topk_adjust_shape` | mock get_compress_topk_idxs，验证取值范围 |
| 3 | `test_expand_kv_and_adjust_return` | 四元组返回值的 shape 一致性 |

**NPU 单卡**（1 层 Attention，ratio=128，随机权重，**seqlen=512**）：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 4 | `test_kv_expansion_equivalence` | Provider 全量 vs Reuser KV 拼接 | 输出一致（误差 < 1e-12） |
| 5 | `test_cmp_kv_expansion` | CMP KV 扩展后长度正确 | dim=0 = sum((P+S)//ratio) |
| 6 | `test_topk_adjust_static` | expanded seqlen 重生成 indices | 指向正确 cmp KV 范围 |
| 7 | `test_topk_adjust_start_pos` | 续训 start_pos > 0 | start_pos 透传正确 |
| 8 | `test_second_rope_symmetry` | 2nd RoPE 位置对称 | 扩展后 o 与 baseline 一致 |

> **seqlen=512 而非 128 的原因**：compress_ratio=128 时，如果 prefix_len 或 suffix_len < 128，对应的 `kv_compress` slice 为 `[0, ...]` 空张量，无法验证 CMP KV 拼接逻辑。
> 设 P=256, S=256：
> - provider CMP: 256//128 = **2 个 entry**（非空）
> - reuser CMP: `cat(provider[:256//128=2], suffix[:256//128=2])` = **4 个 entry**（可验证拼接）
> 这确保 `test_cmp_kv_expansion` 真正覆盖了 CMP KV 扩展逻辑。

### 前置：B3 NPU 可行性验证

**最大风险**：`DeepSeek4SelfAttention` 能否脱离完整 43 层模型独立实例化？

从 MindSpeed 实际代码分析：

```python
# g2_attention.py line 93-118
def __init__(self, config: TransformerConfig, submodules, layer_number, ...):
    args = get_args()                              # ← 需要 Megatron args 全局初始化
    self.head_dim = args.qk_head_dim               # ← 依赖 args 中的字段
    self.layer_number = layer_number + \
        get_transformer_layer_offset(self.config)   # ← 依赖 PP config
    self.compress_ratio = args.compress_ratios[
        self.layer_number - 1]                      # ← 按层索引 compress_ratio
```

三个风险点：

| 风险 | 说明 | 影响 |
|------|------|------|
| `get_args()` | 需要 Megatron 全局 args 初始化 | 单卡测试需先初始化 `parallel_state` + args |
| `get_transformer_layer_offset()` | 依赖 PP config | PP=1 时返回 0，可控 |
| `compress_ratios[self.layer_number - 1]` | 按层索引，需确保数组长度足够 | 单层测试时 `layer_number=1`，只需 `compress_ratios[0]` 就位 |

**建议**：在写 B3 NPU 测试代码**之前**，先在容器中执行一次快速 proof-of-concept：

```python
# PoC: 确认单层 DeepSeek4SelfAttention 可实例化
import torch
from megatron.training import get_args, _set_args
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig

# 1. 初始化 distributed（单卡）
torch.distributed.init_process_group(backend="hccl", world_size=1, rank=0)
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

# 2. 设置 args（需要包含 compress_ratios）
# 3. 构造 TransformerConfig
# 4. 实例化 DeepSeek4SelfAttention
# 5. 跑一次 forward → 确认 self.sparse_attention() 不报错
```

如果单层 `DeepSeek4SelfAttention` 独立实例化失败，回退方案：用**最少层数的完整 TransformerLayer**（1 层含 Attention + 空 MLP），此时 config 和 args 初始化路径与生产环境一致，成功率最高。

### 功能组 B4：Fork Forward + PatchSpec

**Mac**（compat_matrix 配置验证）：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_compat_entry_parse` | CompatEntry 字段值正确 |
| 2 | `test_patch_set_id_unique` | mindspeed_deepseek4 三元组不冲突 |
| 3 | `test_patch_set_exists` | PATCH_SET 可 import |
| 4 | `test_patch_spec_target` | PatchSpec 格式正确 |

**NPU 单卡**（4 层模型，覆盖 ratio≤1/128/4，随机权重，seqlen=128）：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 5 | `test_skip_layer_attention` | ratio=4+indexer 层走 original_forward | 输出与 baseline 一致 |
| 6 | `test_ratio128_full_flow` | ratio=128 层 provider+reuser 完整 fork | Hook A/B/C/D 全部命中 |
| 7 | `test_transitive_reuse` | A reuse B, B reuse C | 深度 reuser 输出一致 |

### 功能组 C：ratio=4 DSA Indexer

**Mac**（mock DSA indexer 逻辑）：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_ratio4_index_offset` | Step 1 offset 后 indices 指向正确范围 |
| 2 | `test_ratio4_provider_store_shape` | provider indexer_score shape 验证 |
| 3 | `test_ratio4_skip_coordination` | attention + transformer 双 patch 协调 |

**NPU 单卡**（2-3 层含 ratio=4 + DSA Indexer，随机权重）：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 4 | `test_ratio4_baseline_parity` | Layer 1 skip 下输出与 baseline 一致 | 误差 < 1e-12 |
| 5 | `test_ratio4_no_interference` | ratio=4 skip 不影响相邻 ratio=128 层 | 共享正确 |
| 6 | `test_ratio4_provider_store` | Provider kv/kv_compress/score 正确存入 | load 完整 |
| 7 | `test_ratio4_reuser_baseline` | Layer 2 Reuser 仍独立计算 | 与 baseline 一致 |
| 8 | `test_ratio4_index_offset` | Step 1 offset 后有效值 < (P+S)//4 | 边界正确 |
| 9 | `test_ratio4_expand_equivalence` | KV/CMP KV 扩展后 attention 输出一致 | 误差 < 1e-12 |
| 10 | `test_ratio4_transitive_reuse` | 深度 reuser 输出一致 | 输出一致 |

### 功能组 D：Transformer 注入

**Mac**（mock transformer data + 合成张量）：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_g2_store_transformer_data_shape` | provider residual/post/comb 存储 shape 正确 |
| 2 | `test_g2_expand_transformer_data_shape` | reuser 扩展后 cat 顺序正确 |

**NPU 单卡**（2-3 层含 MHC，随机权重）：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 3 | `test_residual_expansion` | Reuser residual 扩展后 MHC Post 输出一致 | 误差 < 1e-12 |
| 4 | `test_post_comb_expansion` | post/comb 同步扩展 | 同上 |
| 5 | `test_mhc_disabled` | MHC 不启用时跳过 | 无异常 |
| 6 | `test_transformer_store_provider` | Provider 数据完整存入 store | load 正确 |

### 功能组 E：训练流程集成

**NPU 8 卡**（完整 43 层模型，关闭 MoE，TP=1/PP=1/EP=1/CP=1）：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 1 | `test_e2e_loss_parity` | ENABLE=0 vs =1 | loss 完全一致 |
| 2 | `test_e2e_logprob_parity` | 逐 token logprob 对比 | 完全一致 |
| 3 | `test_e2e_gradient_flow` | Provider KV 梯度正确回传 | grad norm 一致 |
| 4 | `test_prefix_last_restore` | Reuser prefix-last logprob | 误差 < 1e-12 |
| 5 | `test_wrap_forward_step` | 无共享前缀时回退 | 与 baseline 一致 |
| 6 | `test_mixed_ratios` | 同一 batch 含 ratio=0/4/128 | 正确 skip + 正确共享 |

## 模型规模策略

| 组 | 模型 | 大小 | 原因 |
|----|------|------|------|
| B3 | 1 层 `DeepSeek4SelfAttention`（ratio=128）| ~100MB | 只测单层 KV 扩展等价性 |
| B4 | 4 层模型（覆盖 ratio≤1/4/128）| ~500MB | 测 fork forward + skip 层协调 |
| C | 2-3 层含 ratio=4 + DSA Indexer | ~300MB | DSA Indexer 路径专项 |
| D | 2-3 层含 MHC | ~300MB | MHC residual/post/comb 扩展 |
| E | 完整 43 层（关闭 MoE）| ~5GB | 端到端精度红线 |

**全部使用随机权重**：因为每个测试都是对比"同一模型、同一输入、不同路径"的输出一致性，权重值本身不影响结论。

**减层模型关键配置**（B4）：

```python
# DeepSeek V4 使用 MLA（Multi-head Latent Attention），不是标准 MHA/MQA。
# KV 通过 q_lora_rank/kv_lora_rank 压缩，没有 num_key_value_heads 概念。

# Megatron args 关键字段（摘自 g2_attention.py __init__ 实际引用）：
#   args.qk_head_dim = 512           # QK head 维度（total = nope + rope）
#   args.rope_head_dim = 64          # RoPE 部分维度
#   args.q_lora_rank = 1024          # Q 的 LoRA rank
#   args.o_lora_rank = 1024          # 输出的 LoRA rank
#   args.hidden_size = 4096          # 隐藏层维度
#   args.num_attention_heads = 64    # 注意力头数
#   args.o_groups = 8                # 输出投影分组
#   args.g2_window_size = 128        # 滑动窗口大小
#   args.compress_ratios = [0, 0, 128, 4]  # 每层压缩比（长度 = num_layers）
#   args.use_sparse_flash_attn = True       # 使用稀疏 flash attention

# DeepSeek4SelfAttention 的 compress_ratio 来自：
#   self.compress_ratio = args.compress_ratios[self.layer_number - 1]

# 注意：实际实例化依赖 Megatron 的 initialize() + get_args() 全局路径，
# 不能用常规 dataclass 构造。具体方式在可行性验证中确认。
```

## 测试执行流程

```
本地 Mac（开发 + Mac 测试）
    │  commit → push
    ▼
NPU 服务器（可行性验证）
    │  确认 DeepSeek4SelfAttention 可独立实例化
    │  确认 self.sparse_attention() NPU kernel 可调用
    │  确认 compress_ratios / layer_number / get_transformer_layer_offset 行为
    ▼
NPU 服务器（B3/B4/C/D NPU 单卡测试）
    │  减层模型 + 随机权重 + seqlen=512
    │  全部 NPU 测试通过
    ▼
NPU 8 卡（E 组端到端测试）
    │  精度红线全部达标
    ▼
准出：DeepSeek V4 PrefixSharing 功能完成
```

## 精度红线

| 指标 | 要求 | 适用组 |
|------|------|:--:|
| Attention 输出 | 与 baseline 完全一致（误差 < 1e-12）| B3, B4, C |
| MHC Post 输出 | 与 baseline 完全一致 | D |
| Loss | 与 baseline 完全一致 | E |
| Logprob (per token) | 与 baseline 完全一致 | E |
| Gradient norm | 与 baseline 完全一致 | E |
| 参数更新量 | 与 baseline 完全一致 | E |

## 测试统计

| 环境 | A | B1 | B2 | B3 | B4 | C | D | E | 合计 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Mac | 5 | 3 | 6 | 3 | 4 | 3 | 2 | — | **26** |
| NPU 单卡 | — | — | — | 5 | 3 | 7 | 4 | — | **19** |
| NPU 8 卡 | — | — | — | — | — | — | — | 6 | **6** |
| **合计** | **5** | **3** | **6** | **8** | **7** | **10** | **6** | **6** | **51** |
