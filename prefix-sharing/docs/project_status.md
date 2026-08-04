# PrefixSharing DeepSeek V4 项目状态

> **日期**：2026-08-04
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
│   ├── g2_attention.py           ← _g2_kv_store_or_expand + _g2_padded_store_or_replace (双路径)
│   └── g2_batch.py               ← wrap_forward_step (padded 模式, monkey-patch __main__.get_batch)
└── setup/patches/mindspeed_deepseek4/
    ├── __init__.py                ← 1 个 PatchSpec
    └── attention.py               ← 1 插入点 (Phase 3 和 Phase 4 之间) + capture_intermediates
```

**双路径设计**：`_g2_kv_store_or_expand` 根据 tensor shape 自动分派：

| 格式 | 检测条件 | 操作 | PS 方法 |
|------|---------|------|------|
| Padded (BSND) | `kv.ndim==3 and kv.shape[1]>1` | In-place replace，shape 不变 | `_g2_padded_store_or_replace` |
| Packed (THD) | `kv.ndim==2` or `shape[1]==1` | Cat expand，total 变长 | 原 packed expand 路径 |

**核心原理**：Attention 输出长度 = Q 长度 = S。Reuser 全程 suffix-only，只在最终输出边界 restore prefix logprob。

**Store 存储**：3 种 key-side 数据 (kv, kv_compress, indexer_k) + stored_len。

**Hook**：在 `g2_attention.py` Phase 3 (Compressed KV) 和 Phase 4 (Sparse Attention) 之间插入，Provider 存/Reuser 扩。

## 3. 已完成

### 核心功能 (Task 1-5)

| Task | 内容 | 状态 |
|------|------|:--:|
| 1 | StoredG2Activation 裁剪 (8→4 fields) | ✅ |
| 2 | _g2_kv_store_or_expand 核心逻辑 | ✅ |
| 3 | Topk 重算 + cu_seqlens 调整 | ✅ |
| 4 | Patch 注入 + Compat Matrix | ✅ |
| 5 | wrap_forward_step 框架 | ✅ |

### 精度测试（28 配置, 100% 通过）

| 阶段 | 环境 | 配置数 | 结果 | 报告 |
|------|------|:--:|------|------|
| 单卡等价性 | NPU 单卡, BF16 | 18 | bitwise (max_diff=0) | report_1_2_single_card.md |
| Padded 多卡训练 | NPU 多卡, TP/CP | 5 | context-only bitwise, KV-replace ~1e-4 | precision_test_report.md §3.2 |
| Packed Expand | NPU 单卡, BF16 | 5 | allclose(1e-3~2e-3) | precision_test_report.md §3.3 |
| DSA Indexer (ratio=4) | NPU 单卡, BF16 | 2 | 单序列 bitwise; packed sharing NaN | 本次更新 |

**Mac 测试**：225 passed, 0 skipped
**NPU 测试**：30/30 passed

### 关键发现

- **NPU non-determinism**：packed 路径 trim 改变 total length，NPU bf16 GEMM tile 策略不同，导致 ~1e-3 级差异。控制实验确认差异来源为硬件行为，不可消除。
- **Padded 精度更高**：padded 不改变 total length，无 NPU non-det 影响，provider 和 reuser 全部 bitwise identical。

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

## 5. 待完成

| 项 | 优先级 | 说明 |
|----|:--:|------|
| **verl E2E forward 对比** | 高 | DeepSeek4Model + verl batch → base vs PS loss。compat+activation+planning 已验证，差最后一步 |
| Packed + TP/CP | 中 | TP=2 阻塞（LinearNoTP 不分片，需 Megatron TP 初始化）；CP=2 待测 |
| E2E 8 卡精度红线 | 中 | loss/logprob/grad 与 baseline 一致性，需完整模型权重和训练数据 |
| standalone restore | 低 | `_restore_prefix_last` 适配 pretrain 格式；padded 模式不需要 |
| ratio=4 packed sharing | 低 | sparse_flash_mla 兼容问题，非 Hook 逻辑问题 |

## 6. 经验教训

1. **根本假设必须源码验证**：v1 假定 attention output = P+S，导致整套错误设计。查源码纠正为 output = Q = S，设计大幅简化。

2. **三件套版本匹配是关键**：mindspeed/megatron-core/mindspeed_llm 版本不兼容是最耗时的阻碍。最终用 CANN 9.1 镜像 `deepseek-rl:910b-cann9.1-vllm0.23-v23-sparse` 解决。

3. **测试盲区要及时补**：`align_prefix_lens_to_compression` 和 `wrap_forward_step` 发布后才发现 bug。

4. **Packed 格式的 batch_size=1**：THD 格式下 hidden tensor batch_size 必须是 1，cu_seqlens 处理序列边界。batch_size=plan.batch_size 会导致 2nd RoPE 失败。

5. **NPU non-determinism 不可消除**：bf16 GEMM 对不同 matrix size 使用不同 tile 策略，packed 路径的 trim 改变了 total length。精度对比需区分"PS 错误"和"硬件 non-det"。

6. **Padded vs Packed 双路径**：padded 精度更高（bitwise），packed 更灵活（支持 variable-length trim）。两种场景都需要覆盖。

## 7. 代码变更汇总

| 文件 | 变更 | 说明 |
|------|------|------|
| `integrations/g2_attention.py` | +92 | `_g2_padded_store_or_replace` + PS_DEBUG + padded dispatch |
| `integrations/g2_batch.py` | -66/+28 | 删除 trim 逻辑, monkey-patch `__main__.get_batch` |
| `backends/g2_attention_utils.py` | +12 | tensor 类型保持 + None 检查 |
| `setup/patches/.../attention.py` | +53 | capture_intermediates + TND 3D/4D 修复 |
| `tests/precision/test_packed_expand.py` | +652 | Packed expand NPU 验证 |
| `tests/unit_test/test_g2_edge_cases.py` | 新增 | 9 个边界用例 |
| `docs/reports/precision_test_report.md` | 新增 | 全量精度报告 |
