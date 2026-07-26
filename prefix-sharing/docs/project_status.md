# PrefixSharing DeepSeek V4 项目状态

> **日期**：2026-07-26
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
│   ├── g2_attention.py           ← _g2_kv_store_or_expand (核心)
│   └── g2_batch.py               ← wrap_forward_step
└── setup/patches/mindspeed_deepseek4/
    ├── __init__.py                ← 1 个 PatchSpec
    └── attention.py               ← 1 插入点 (gather 后, sparse_attention 前)
```

**核心原理**：Attention 输出长度 = Q 长度 = S。Reuser 全程 suffix-only，只在最终输出边界 restore prefix logprob。

**Store 存储**：3 种 key-side 数据 (kv, kv_compress, indexer_k) + stored_len。

**Hook**：在 `g2_attention.py` Phase 3 (Compressed KV) 和 Phase 4 (Sparse Attention) 之间插入，Provider 存/Reuser 扩。

## 3. 已完成 (Task 1-5)

| Task | 内容 | 测试 | NPU 验证 |
|------|------|:--:|:--:|
| 1 | StoredG2Activation 裁剪 (8→4 fields) | 7 Mac | — |
| 2 | _g2_kv_store_or_expand 核心逻辑 | 13 Mac | — |
| 3 | Topk 重算 + cu_seqlens 调整 | 5 Mac | ratio=0 forward OK |
| 4 | Patch 注入 + Compat Matrix | 5 Mac | 6/6 通过 |
| 5 | wrap_forward_step 框架 + 文档 | — | — |

**Mac 测试**：216 passed, 0 skipped
**NPU 测试**：6/6 passed (patch + forward + store + output parity)

## 4. 决策记录

| # | 问题 | 决策 | 实施 |
|---|------|------|:--:|
| 1 | Attention 输出是 P+S 还是 S | S (Q 决定) | 删 Hook D、TransformerLayer patch、D 组 |
| 2 | ratio=4 是跳过还是统一处理 | 统一处理 | 和 ratio=128 同一个 Hook |
| 3 | 压缩边界冲突 | 方案 A: prefix_len 对齐 r | `align_prefix_lens_to_compression()` |
| 4 | compress_topk_score 回传 | 加进返回值 | 6 元组 |
| 5 | restore 签名错误 | 注解 TODO | standalone restore 待 8 卡 E2E |

## 5. 待完成

| 项 | 优先级 | 说明 |
|----|:--:|------|
| **精度测试 (ratio=128/4)** | 高 | 5 个 NPU 等价性场景 |
| **8 卡 E2E 精度红线** | 高 | loss/logprob/grad 与 baseline 一致性 |
| standalone restore 实现 | 中 | `_restore_prefix_last` 适配 pretrain 格式 |
| CP>1 支持 | 中 | 全局 layout, cp_rank, cu_seqlens 适配 |
| indexer_k 全量测试 (ratio=4) | 低 | 当前 Mock 测试已 pass, 需要真 Indexer |

## 6. 经验教训

1. **根本假设必须源码验证**：v1 假定 attention output = P+S，导致整套错误设计。查源码纠正为 output = Q = S，设计大幅简化。

2. **三件套版本匹配是关键**：mindspeed/megatron-core/mindspeed_llm 版本不兼容是最耗时的阻碍。用 `core_r0.16.0` 线 + PYTHONPATH + 少量源码 patch 可行。

3. **测试盲区要及时补**：`align_prefix_lens_to_compression` 和 `wrap_forward_step` 发布后才发现 bug，因为完全没有测试覆盖。

4. **先讨论清楚再写文档**：CP+压缩边界问题讨论了多轮才搞明白条件一二的交集逻辑。提前写文档反而制造混乱。

5. **改 prefix_len 对齐是最简单的方案**：避免 comp_boundary 存储、重建逻辑、CP 通信。一行代码消掉整个问题类。
