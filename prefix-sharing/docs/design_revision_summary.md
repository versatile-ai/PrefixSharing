# DeepSeek V4 PrefixSharing 方案修正总结

> **日期**：2026-07-23

## 1. 原方案的错误

### 1.1 Attention 输出长度

原方案假定 KV 扩展到 P+S 后，attention output 也变为 P+S。错误。

**Attention 的输出长度 = Query 长度。** 扩展 KV 只让每个 query token 看到更长历史，不增加输出 token 数：

```
Attention(Q[S], KV[P+S]) = Output[S]
```

这个错误引出了全部不需要的设计：Hook D、`_g2_expand_attn_output`、`residual/post/comb_prefix` 存储、TransformerLayer patch、功能组 D。

### 1.2 ratio=4 的 fallback

原方案在 fork forward 内检测 `compress_ratio==4 and indexer` → 走 `original_forward`。但此时 batch 已被裁为 suffix-only，Indexer 算出的 key space 可能为空，prefix context 已不可恢复。

正确做法：ratio=4 与 ratio=128 统一处理——扩展 key-side 数据（包括 `indexer_k`），reuser 用自己的 query 重算 topk。

## 2. 修正后的方案

整个方案只有两部分：**一个 typed store + 一个插入点。**

### 2.1 Typed Store：StoredG2Activation

Provider 计算 prefix 时，在 attention 内部产生以下数据。这些数据需要存下来供 reuser 复用：

| 字段 | Shape | 来源 | 用途 |
|------|-------|------|------|
| `kv` | `[s, b, 512]` | `linear_kv → layernorm → RoPE → gather` | sparse_flash_mla 的 raw KV 输入，SWA attention |
| `kv_compress` | `[s//r, b, 512]` | `attention.compressor(hidden_states, ...)` | sparse_flash_mla 的压缩 KV 输入，HCA/CSA coarse attention |
| `indexer_k` | `[s//4, b, 1, 128]` | `indexer.kv_compressor(hidden_states, ...)` 即 `forward_with_index_compress` 返回的 k | DSA Indexer 重新打分时需要的 key embeddings（仅 ratio=4 层） |
| `stored_len` | `int` | valid_len | 取出时确定 slice 范围 |

三个张量分别来自三个不同的模块实例，形状和用途各不相同。reuser 把它们和自己的 suffix 数据拼接，得到完整的 key-side 空间。

### 2.2 插入点

原始 `g2_attention.py` forward 的结构：

```
Phase 1: Q/KV 投影 + RoPE                     → kv 就绪
Phase 2: compress_topk_idxs 生成                → topk + indexer_k 就绪
Phase 3: Compressed KV                         → kv_compress 就绪

    ╔══════════════ 插入点 ═══════════════╗
    ║ provider: store(kv, kv_compress, idxk)║
    ║ reuser:   expand + recompute topk    ║
    ╚══════════════════════════════════════╝

Phase 4: Sparse Attention  ← 消费扩后的数据
Phase 5: RoPE + Output Projection
```

**安全性**：插入点之前所需数据全部就绪，之后 `sparse_attention` 立即消费。Phase 4 末尾的 DSA loss 块虽然修改 `compress_topk_idxs` 和 `kv_compress`，但在 attention **之后**，不影响。Phase 5 的 `rearrange(s=q_len)` 中 `q_len` 是 suffix 长度 S，与 attention 输出 [S, ...] 天然匹配。

```python
# 插入点代码结构
def _g2_kv_store_or_expand(ctx, kv, kv_compress, indexer_k,
                            compress_topk_idxs, packed_seq_params):
    for batch_idx in range(batch_size):
        if plan.is_provider(batch_idx):
            store(kv=kv[:valid_len],
                  kv_compress=cmp[:valid_len//r],
                  indexer_k=idxk[:valid_len//4])
        elif plan.is_reuser(batch_idx):
            p = ctx.store.load(provider_slot)
            kv = cat(p.kv[:P], reuser_kv[:S])
            kv_compress = cat(p.kv_compress[:P//r], reuser_cmp[:S//r])
            indexer_k = cat(p.indexer_k[:P//4], reuser_idxk[:S//4])
            compress_topk_idxs = recompute_topk(...)
            packed_seq_params = adjust_cu_seqlens(...)
    return kv, kv_compress, indexer_k, compress_topk_idxs, packed_seq_params
```

### 2.3 ratio=128 与 ratio=4 的区别

两类层走完全相同的 key-side 扩展逻辑。唯一的区别在 `recompute_topk` 这一步：

| ratio | recompute_topk |
|-------|---------------|
| ≤1 | 不需要 |
| 128 | `attention_module.get_compress_topk_idxs(expanded_seqlen)` — 纯位置重算 |
| 4 | reuser 用自己的 query × 扩展后的 indexer_k → `forward_with_scores_compress` 重新打分 |

ratio=4 时重算 topk 的过程：

```python
# Phase 2 原始代码已跑过 forward_with_index_compress(suffix_hidden)，
# 产生了 q_r, w_r, k_r（k_r 可能为空）。q_r 和 w_r 是对的，直接复用。
# 插入点把 k_r 替换为 expanded_k 后重新打分：

compress_topk_idxs = self.indexer.forward_with_scores_compress(
    dsa_hidden,    # ← Phase 2 产物，复用
    q_r,           # ← Phase 2 产物，复用（suffix query，正确）
    expanded_k,    # ← 插入点替换！cat(provider.indexer_k[:P//4], k_r)
    w_r,           # ← Phase 2 产物，复用
    ...)
# 覆盖 Phase 2 算错的 compress_topk_idxs，Phase 4 读到的是正确的
```

## 3. 差异分析修正

| # | 差异 | 原分析 | 修正 |
|---|------|--------|------|
| 1 | MLA KV 压缩向量 | 需新建 typed store | **不变** |
| 2 | 融合算子不可替换 | 只能 pre/post hook | **不变** |
| 3 | 两次 RoPE | Hook D 存/扩 attn_o | **删除**。扩的是 KV，不是 attn_o |
| 4 | 稀疏数据结构 | topk/cu_seqlens 同步扩展 | **不变**，增加 indexer_k |
| 5 | MHC residual shape | 需 TransformerLayer patch | **删除**。attn_o=S 与 residual 对齐 |
| 6 | 层间异构 | ratio=4 跳过 | **重写**。与 ratio=128 统一，不跳过 |

## 4. 模块映射

```
prefix_sharing/
├── core/prefix_store.py          ← StoredG2Activation (4 fields) + G2AttentionStore
├── backends/g2_attention_utils.py ← split/cmp/cu_seqlens adjust
├── integrations/
│   ├── g2_attention.py            ← _g2_kv_store_or_expand
│   └── g2_batch.py                ← wrap_forward_step
└── setup/patches/mindspeed_deepseek4/
    ├── __init__.py                 ← 1 个 PatchSpec
    └── attention.py               ← 1 个插入点
```

删除：`g2_transformer.py`、`transformer.py` patch、`_g2_expand_attn_output`、`_merge_g2_transformer_fields`、`_is_skip_layer`。

## 5. 对已完成代码的影响

| 文件 | 操作 |
|------|------|
| `StoredG2Activation` | 8→4 fields：删 attn_o/residual/post/comb/indexer_score，保留 kv/kv_compress/stored_len，加 indexer_k |
| `G2AttentionStore.store()` | 参数同步修改 |
| `_merge_g2_fields` | 分支同步修改 |
| `_merge_g2_transformer_fields` | 删除 |
| `_g2_expand_attn_output` | 删除 |
| `_g2_store_per_sequence` | 字段处理同步修改 |
| B1 工具函数 | 不变 |
| 测试文件 | 重写 |

## 6. 执行计划

| 任务 | 时间 | 内容 |
|------|:--:|------|
| A — Store | 0.5 天 | StoredG2Activation + G2AttentionStore 裁剪 |
| B — Hook | 2 天 | _g2_kv_store_or_expand + topk adjust + 1 插入点 |
| E — 集成 | 1 天 | wrap_forward_step + prefix-last restore |
| **合计** | **3.5 天** | |

## 7. 核心原则

1. **suffix-only 不变量**：reuser 全程保持 suffix-only，仅最终输出边界 restore
2. **只扩 key-side**：扩 KV / CMP KV / indexer_k；不扩 Q / attn_o / residual / hidden
3. **KV 不 detach**
4. **精度一致性优先**
