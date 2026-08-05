# PrefixSharing DeepSeek V4 项目状态

> **日期**：2026-08-05
> **分支**：`feature/deepseek4-prefix-sharing`

## 1. 项目目标

在 DeepSeek V4 预训练中实现 micro-batch 内 prefix KV 复用。同一 batch 内多个 sequence 共享前缀时，provider 计算一次前缀 KV，reuser 复用，消除冗余计算。

## 2. 架构

```
prefix_sharing/
├── core/
│   ├── prefix_store.py          ← G2AttentionStore + StoredG2Activation (4 fields)
│   └── planner.py               ← + align_prefix_lens_to_compression (方案A)
├── backends/
│   └── g2_attention_utils.py    ← split/cmp/cu_seqlens/topk adjust/merge
├── integrations/
│   ├── g2_attention.py           ← _g2_kv_store_or_expand (packed) + _g2_padded_store_or_replace (padded)
│   ├── g2_batch.py               ← wrap_forward_step (padded 模式)
│   └── verl_mcore.py             ← build_prefix_sharing_micro_batch_verl080 + restore
├── setup/patches/mindspeed_deepseek4/
│   ├── __init__.py                ← 1 个 PatchSpec
│   └── attention.py               ← 1 插入点 (Phase 3 ↔ Phase 4) + IdentityOp 防护
└── setup/patches/verl080_mcore0161_ms0160/
    └── forward_step.py            ← patch_verl_forward_step
```

**双路径设计**：
| 格式 | 检测条件 | 操作 | PS 方法 |
|------|---------|------|------|
| Padded (BSND) | `kv.ndim==3 and kv.shape[1]>1` | In-place replace | `_g2_padded_store_or_replace` |
| Packed (THD) | `kv.ndim==3 and kv.shape[1]==1` | Cat expand | `_g2_kv_store_or_expand` |

**核心原理**：Attention 输出长度 = Q 长度 = S。Reuser 全程 suffix-only，只在最终输出边界 restore prefix logprob。

## 3. 已完成

### 核心功能

| Task | 内容 | 状态 |
|------|------|:--:|
| A | StoredG2Activation + G2AttentionStore | ✅ |
| B | Attention ratio=128 (Fork forward + Hook + topk/cu_seqlens) | ✅ |
| C | Attention ratio=4 DSA Indexer (完整动态 topk) | ✅ |
| D | ~~Transformer 注入~~ | 不需要 (v2 设计) |
| E | wrap_forward_step + verl forward_step patch | ✅ |

### 精度测试

| 阶段 | 环境 | 配置数 | 结果 |
|------|------|:--:|------|
| 单卡等价性 | NPU 单卡, BF16 | 18 | bitwise (max_diff=0) |
| Padded 多卡 (TP/CP) | NPU 多卡 | 5 | context-only bitwise, KV-replace ~1e-4 |
| Packed Expand | NPU 单卡, BF16 | 5 | allclose(1e-3~2e-3) |
| DSA Indexer (ratio=4) 单序列 | NPU 单卡, BF16 | 2 | 单序列 bitwise |
| **DeepSeek4Model E2E PS (全ratio)** | **NPU 单卡, BF16** | **4** | **全层 matches_expected** |

**Mac 测试**：225 passed, 0 skipped
**NPU 测试**：30/30 passed

### DeepSeek4Model E2E PS 验证矩阵 (2026-08-05)

| 层配置 | 模型 | 结果 |
|------|------|:--:|
| 1 层 `[128]` | 351M | ✅ 1/1 matches_expected |
| 3 层 `[0, 128, 128]` | 922M | ✅ 3/3 matches_expected |
| **4 层 `[0, 128, 4, 128]`** (完整 DSA) | **1.2B** | ✅ **4/4 matches_expected** |

每层统计：`store_count=1 reuse_count=1 stored_tokens=256 reused_prefix_tokens=128`

## 4. 决策记录

| # | 问题 | 决策 | 实施 |
|---|------|------|:--:|
| 1 | Attention 输出是 P+S 还是 S | S (Q 决定) | 删 Hook D、TransformerLayer patch |
| 2 | ratio=4 是跳过还是统一处理 | 统一处理 | 和 ratio=128 同一个 Hook |
| 3 | 压缩边界冲突 | 方案 A: prefix_len 对齐 r | `align_prefix_lens_to_compression()` |
| 4 | compress_topk_score 回传 | 加进返回值 | 6 元组 |
| 5 | Packed vs Padded 两条路径 | `kv.ndim==3 and shape[1]>1` 检测分派 | `_g2_padded_store_or_replace` (新增) |
| 6 | Padded 模式不需要 trim | Store/replace 后全量 Q 等价基线 | 删除 `g2_batch.py` trim 逻辑 |
| 7 | Packed cu_seqlens_kv 初始化 | 使用 trimmed 长度 `plan.cu_seqlens_q`，Hook 调整为 expanded | 测试构造修正 |
| 8 | compat_matrix 容器匹配 | 新增 verl=0.8.0.dev + mc=0.12.1 条目 | `mindspeed_deepseek4` patch set |
| 9 | ratio=4 DSA Indexer args | 补全 `kv_compress`/`index_head_dim`/`index_n_heads` 等字段 | `_LazyArgs` 扩展 |
| 10 | TP=2 单层不可行 | `LinearNoTP` 不分片，`n_local_heads` 不匹配 | 需 Megatron TP 完整初始化 |
| 11 | DSA Indexer `rope_head_dim` 不匹配 | 测试设置 32，DSA 期望 64 (MLA qk_rope_head_dim) | `rope_head_dim=64` |
| 12 | DSA Indexer `kv_compress` 默认 False | 不构建 `kv_compressor` 但 forward 无条件调用 | `kv_compress=True` |
| 13 | patched_forward IdentityOp 检查 | Fork 的 MindSpeed 代码 `is not None` 对 IdentityOp 误判 | 加 `isinstance(self.indexer, IdentityOp)` 防护 |

## 5. 代码变更汇总

| 文件 | 变更 | 说明 |
|------|------|------|
| `integrations/g2_attention.py` | +92 | `_g2_padded_store_or_replace` + PS_DEBUG + padded dispatch + 双路径 stats 上报 |
| `integrations/g2_batch.py` | -66/+28 | 删除 trim 逻辑, monkey-patch `__main__.get_batch` |
| `backends/g2_attention_utils.py` | +12 | tensor 类型保持 + None 检查 |
| `setup/patches/.../attention.py` | +55 | capture_intermediates + TND 3D/4D 修复 + **IdentityOp 防护 + import** |
| `tests/precision/test_packed_expand.py` | +652 | Packed expand NPU 验证 |
| `tests/unit_test/test_g2_edge_cases.py` | 新增 | 9 个边界用例 |
| `docs/reports/precision_test_report.md` | 新增 | 全量精度报告 |
| **容器** `/MindSpeed-LLM/.../g2_attention_kernel.py` | +1 | `from __future__ import annotations`（修复 NPU triton 不可用时 `tl.constexpr` 崩溃） |

## 6. 待完成

| 项 | 优先级 | 说明 |
|----|:--:|------|
| Packed + TP/CP | 中 | TP=2 阻塞（LinearNoTP 不分片，需 Megatron TP 初始化）；CP=2 待测 |
| E2E 8 卡精度红线 | 中 | loss/logprob/grad 与 baseline 一致性，需完整模型权重和训练数据 |
| standalone restore | 低 | `_restore_prefix_last` 适配 pretrain 格式；padded 模式不需要 |
| verl E2E forward (MegatronEngineWithLMHead) | 低 | compat+activation+planning+trim+context 已验证，差异仅在入口层 |

## 7. 测试配置参考（DeepSeek4Model E2E）

以下参数在容器 `deepseek-verify` (192.168.0.2, CANN 9.1) 上验证通过：

```
--num-layers 4
--hidden-size 2048
--num-attention-heads 32
--ffn-hidden-size 5504
--num-experts 8
--position-embedding-type g2
--spec mindspeed_llm.tasks.models.spec.deepseek4_spec layer_spec
--enable-dsa-indexer
--swiglu --no-bias-swiglu-fusion --disable-bias-linear
--no-gradient-accumulation-fusion
```

关键 DeepSeek4 特定参数（`_ds_fields`）：
```python
"qk_head_dim": 256, "rope_head_dim": 64,  # MLA: rope_head_dim 必须匹配 DSA Indexer
"q_lora_rank": 512, "o_lora_rank": 512, "o_groups": 4,
"g2_window_size": 64, "compress_ratios": [0, 128, 4, 128],
"compress_rope_theta": 10000.0, "rope_theta": 10000.0,
"rope_factor": 40, "beta_fast": 32, "beta_slow": 1,
"rope_scaling_original_max_position_embeddings": 4096, "norm_eps": 1e-6,
"enable_dsa_indexer": True, "kv_compress": True,
```

## 8. 经验教训

1. **根本假设必须源码验证**：v1 假定 attention output = P+S，导致整套错误设计。查源码纠正为 output = Q = S，设计大幅简化。

2. **三件套版本匹配是关键**：mindspeed/megatron-core/mindspeed_llm 版本不兼容是最耗时的阻碍。最终用 CANN 9.1 镜像 `deepseek-rl:910b-cann9.1-vllm0.23-v23-sparse` 解决。

3. **测试盲区要及时补**：`align_prefix_lens_to_compression` 和 `wrap_forward_step` 发布后才发现 bug。

4. **Packed 格式的 batch_size=1**：THD 格式下 hidden tensor batch_size 必须是 1，cu_seqlens 处理序列边界。

5. **NPU non-determinism 不可消除**：bf16 GEMM 对不同 matrix size 使用不同 tile 策略，packed 路径的 trim 改变了 total length。精度对比需区分"PS 错误"和"硬件 non-det"。

6. **模拟测试参数必须与模型架构匹配**：`rope_head_dim` 和 `kv_compress` 等参数不是随意设置的，必须与 MindSpeed 模型 spec 内部期望一致。`rope_head_dim=64`（MLA qk_rope_head_dim）和 `kv_compress=True`（DSA Indexer 的 kv_compressor 构建条件）都是模型架构决定的。

7. **DSA Indexer 有两个代码路径**：`kv_compress=True` 构建 `kv_compressor`，`kv_compress=False` 构建 `wk` + `k_norm`。但 `forward_with_index_compress` 只在 `kv_compress=True` 路径下测试过——这是 CANN 9.1 容器中 MindSpeed-LLM 代码的实际状态。
