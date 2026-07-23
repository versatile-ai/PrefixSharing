# DeepSeek V4 PrefixSharing 集成设计 v2

> **日期**：2026-07-23
> **参考源码**：`mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py`、`mindspeed_llm/tasks/models/transformer/dsa_indexer.py`

## 1. 核心原理

DeepSeek V4 的 Attention 在 forward 过程中产生三种 key-side 数据：

```
Phase 1: Q/KV 投影 + RoPE
  → kv          [s, b, 512]

Phase 2: compress_topk_idxs 生成 (ratio>1 时)
  → indexer_k   [s//4, b, 1, 128]   (仅 ratio=4)

Phase 3: Compressed KV (ratio>1 时)
  → kv_compress [s//r, b, 512]

Phase 4: sparse_attention(q, kv, kv_compress, topk, ...)
```

Prefix Sharing 需要做的是：**在 Phase 3 和 Phase 4 之间插入一个 Hook 点，Provider 存入这三种数据，Reuser 读出、扩展、适配后替换原变量，再送入 Phase 4。**

```
Phase 1-3 (原始代码照跑)
    │
    ╔═══ Hook: Store / Expand ═══╗
    ║ Provider → store             ║
    ║ Reuser   → load + expand     ║
    ║          → recompute topk    ║
    ║          → adjust cu_seqlens ║
    ╚══════════════════════════════╝
    │
Phase 4-5 (原始代码照跑，消费替换后的数据)
```

Reuser 全程保持 suffix-only [S]，只在最终训练输出边界做 prefix-last restore。

## 2. 方案总览

| 部分 | 内容 |
|------|------|
| A — Typed Store | `StoredG2Activation` + `G2AttentionStore` |
| B — Hook 插入 | `_g2_kv_store_or_expand` + 1 个 patch 点 |
| E — 训练集成 | `wrap_forward_step` + prefix-last restore |

## 3. Typed Store

### 3.1 StoredG2Activation

Provider 在 forward 中产生的三种 key-side 数据，需缓存供 reuser 复用：

| 字段 | Shape | 来源 | 用途 |
|------|-------|------|------|
| `kv` | `[s, b, 512]` | `linear_kv → layernorm → RoPE → gather` | sparse_flash_mla 的 raw KV，SWA attention |
| `kv_compress` | `[s//r, b, 512]` | `self.compressor(hidden, start_pos, freqs, params)` | sparse_flash_mla 的 compressed KV，HCA/CSA coarse attention |
| `indexer_k` | `[s//4, b, 1, 128]` | `self.indexer.forward_with_index_compress(hidden).k` — float embedding | DSA Indexer 重打分时的 key embeddings |
| `stored_len` | `int` | valid_len | 取出时确定 slice 范围 |

`indexer_k` 仅在 `compress_ratio=4` 且 `self.indexer is not None` 的层存储，其他层为 None。

`kv_compress` 仅在 `compress_ratio>1` 的层存储。

三个数据来自三个不同的模块实例，不可互相替代。

### 3.2 G2AttentionStore

**文件**：`prefix_sharing/core/prefix_store.py`（修改）

遵循现有 `PrefixActivationStore` base + typed wrapper 模式，与 `PrefixAttentionStore`、`PrefixDeltanetStore` 并列。增量存储（同一 slot 多次 store，merge 已有字段）。

### 3.3 辅助函数

**文件**：`prefix_sharing/integrations/g2_attention.py`

| 函数 | 作用 |
|------|------|
| `_g2_store_with_kwargs` | `StoredG2Activation` 各字段显式传入 `G2AttentionStore.store()` |

**文件**：`prefix_sharing/backends/g2_attention_utils.py`

| 函数 | 作用 |
|------|------|
| `_split_by_cu_seqlens` | 按 padded_lengths 拆分 packed tensor |
| `_compute_cmp_lengths` | 计算 per-sequence 压缩 KV 长度 |
| `_adjust_cu_seqlens_for_batch` | 用 `dataclasses.replace` 重建 `packed_seq_params`，reuser 偏移 prefix_len |
| `_merge_g2_fields` | 基于已有 entry 创建更新单个字段的新实例（frozen dataclass 不可原地改） |

## 4. Hook 插入

### 4.1 插入位置

原始 `g2_attention.py` forward 的 Phase 3（Compressed KV）之后、Phase 4（Sparse Attention）之前。

```
Phase 1: Q 投影 + RoPE
Phase 1: KV 投影 + RoPE                               → kv
Phase 2: compress_topk_idxs 生成                       → topk, indexer_k
Phase 3: self.compressor(...)                          → kv_compress

    ╔══════════ 插入点 ═══════════╗
    ║ _g2_kv_store_or_expand()   ║
    ╚══════════════════════════════╝

Phase 4: self.sparse_attention(q, kv, kv_compress, topk, ...)
Phase 5: 2nd RoPE + Output Projection
```

**安全性**：插入点之前所有数据已就绪。Phase 4 立即消费。Phase 4 末尾的 DSA loss 块虽修改 `compress_topk_idxs` 和 `kv_compress`，但在 attention **之后**。Phase 5 的 `rearrange(s=q_len)` 中 `q_len=S`，与 attention 输出 [S, ...] 匹配。**不会被覆盖。**

**最小化 fork**：patch 复制 Phase 1-3 的编排代码（~40 行）以获取变量访问权，不修改任何计算逻辑。MindSpeed 升级时 diff 对照原始 `g2_attention.py` 即可。

### 4.2 Patch 注入

**文件**：`prefix_sharing/setup/patches/mindspeed_deepseek4/attention.py`

引入仅 ~5 行：

```python
def patched_forward(self, hidden_states, attention_mask, rotary_pos_emb, ...):
    # Phase 1-3: 原始代码照跑 (g2_attention.py line 361-477)
    #   其中包括:
    #     kv = gather_from_sp_cp(self.kv_layernorm(kv_compressed))
    #     query_index, key_index, weights, dsa_hidden = \
    #         self.indexer.forward_with_index_compress(...)
    #     kv_compress = gather_from_sp_cp(self.compressor(...))
    ...

    # ═══ 插入点 ═══
    ctx = current_prefix_sharing_context()
    if ctx is not None and isinstance(ctx.store, G2AttentionStore):
        # key_index 是 Phase 2 的产物，重命名为语义更清晰的 indexer_k
        indexer_k = key_index if self.compress_ratio == 4 else None

        kv, kv_compress, indexer_k, compress_topk_idxs, packed_seq_params = \
            _g2_kv_store_or_expand(
                ctx, kv, kv_compress, indexer_k,
                compress_topk_idxs, packed_seq_params,
                self.compress_ratio, self, start_pos,
                self.kv_allgather, self.config.sequence_parallel)
    # ═══════════════

    # Phase 4-5: 原始代码照跑 (g2_attention.py line 481-580)
    o = self.sparse_attention(q, kv, kv_compress, compress_topk_idxs, ...)
    ...
```

### 4.3 _g2_kv_store_or_expand

**文件**：`prefix_sharing/integrations/g2_attention.py`

按 batch 身份分支——拆分 packed tensor，逐序列处理。最终拼回 extended packed tensors。

Provider：
1. `_split_by_cu_seqlens` 拆分 packed tensor
2. 截 `[:valid_len]` 有效部分
3. `_merge_g2_fields` 增量更新 → `_g2_store_with_kwargs` 存入

Reuser：
1. `ctx.store.load(provider_slot)` 读出 provider 数据
2. `cat(provider_data[:P], row[:valid])` 拼接扩展 kv / kv_compress / indexer_k
3. 重算 `compress_topk_idxs`（见 §4.4）
4. `_adjust_cu_seqlens_for_batch` 调整 cu_seqlens
5. 回存 `own_slot` 供 transitive reuse

### 4.4 Topk 重算

| ratio | 重算方式 |
|-------|---------|
| ≤1 | 不需要 |
| 128 | `self.get_compress_topk_idxs(expanded_seqlen)` — 纯位置重算 |
| 4 | Reuser 自己的 q × 扩展后的 indexer_k → `self.indexer.forward_with_scores_compress` 重新打分 |

ratio=4 的时序（精确到变量）：

```
Phase 2 (原始代码, suffix-only [S]):
  query_index, key_index, weights, dsa_hidden = \
      self.indexer.forward_with_index_compress(hidden[S], q_compressed[S], ...)
  # query_index: [S, b, 64, 128]   ← suffix query embedding, 复用
  # key_index:   [S//4, b, 1, 128]  ← suffix key embedding, 可能为空
  # weights:     [S, b, 64]         ← suffix weight, 复用
  # key_index 只编码了 suffix 这 S 个 token 的压缩表示，不包含 provider prefix

  compress_topk_idxs = self.indexer.forward_with_scores_compress(
      ..., q=query_index, k=key_index, w=weights, ...)
  # 此时 topk 基于 suffix-only key space 算出，对 reuser 不完整

Phase 3: Compressed KV

=== 插入点 ===
  indexer_k = key_index   # 重命名，语义清晰
  # Provider: store(indexer_k) — 存 [P//4, b, 1, 128]
  # Reuser:
  expanded_k = cat(provider.indexer_k[:P//4], indexer_k[:S//4])
  #              ↑ 包含 provider prefix 的 CMP key embeddings
  #                                        ↑ suffix 的 (可能为空)
  # expanded_k = [(P+S)//4, b, 1, 128] — 完整 key space
  #
  compress_topk_idxs = self.indexer.forward_with_scores_compress(
      ..., q=query_index, k=expanded_k, w=weights, ...)
  # 用 reuser 自己的 q + w，对完整 key 空间重新打分 → 正确的 topk
```

**`key_index` (即 `indexer_k`) 的 dtype 确认**：

- 源码 `dsa_indexer.py:456`：`k = self.kv_compressor(x, ...).unsqueeze(2)` — 这是 Linear 投影 + 压缩后的 **float32/float16 张量**，不是整数索引
- 源码 `dsa_indexer.py:375`：`def forward_with_scores_compress(x, q, k, weights, ...)` — `k` 是函数参数，外部传入即可
- **实现前确认**：在容器内执行 `print(key_index.dtype, key_index.shape)` 验证（预期 `torch.float16/bfloat16`, `[S//4, b, 1, 128]`）

### 4.5 Compressed Boundary

当 prefix_len 不是 compress_ratio 的整数倍时，存在跨边界压缩块：

```
P=3, S=3, r=4:
  P//r + S//r = 0+0 = 0
  (P+S)//r    = 6//4 = 1   ← 一个压缩块跨越了 prefix/suffix 边界
  cat(provider[:0], suffix[:0]) = 空  ← 丢失
```

**Phase 1**：断言 `P % compress_ratio == 0`，只支持对齐 prefix。大部分实际场景 prefix_len 是 128 的倍数，不受影响。

**Phase 2**：provider 存储最后 `r - P%r` 个 token 的 raw hidden。Reuser 用 provider tail + suffix head 重跑 `self.compressor()` 得到正确的跨边界压缩块。

### 4.6 实现注意事项：packed 内的 dim 混排

DeepSeek V4 只走 THD packed 格式，所有张量在 dim=0 维度上拼接。但在 packed 内部，不同张量的序列维位置不同：

| 张量 | Shape | 序列维 | split/cat 操作 |
|------|-------|--------|:--:|
| `kv` | `[total_tokens, b, 512]` | dim=0 | ✅ 直接操作 |
| `kv_compress` | `[total_cmp, b, 512]` | dim=0 | ✅ 直接操作 |
| `indexer_k` | `[total_idxk, b, 1, 128]` | dim=0 | ✅ 直接操作 |
| `compress_topk_idxs` | `[b, total_q, topk]` | **dim=1** | ❌ 不能走 split/cat |

`compress_topk_idxs` 的 dim=0 是 batch，dim=1 才是序列。如果对它调 `_split_by_cu_seqlens` 会按 batch 拆而非按序列拆。**处理方式：topk 走重算（§4.4），不走 split/store/expand。**

`_split_by_cu_seqlens` 仅用于 dim=0 为序列维的张量（kv / kv_compress / indexer_k）。`_g2_kv_store_or_expand` 内部对这三种张量和 topk 走不同分支。

## 5. 训练流程集成

**文件**：`prefix_sharing/integrations/g2_batch.py`

`wrap_forward_step()` 包装原始 forward_step：
1. 前缀检测 + 裁剪 batch（复用现有 planner + trim）
2. `prefix_sharing_runtime_context` 包裹模型 forward
3. prefix-last restore（复用现有 `restore_reuser_prefix_columns_2d`）

并行约束：Phase 1 CP=1。

## 6. 核心不变量

1. **suffix-only**：reuser 全程保持 suffix-only，仅最终输出边界做 prefix-last restore
2. **只扩 key-side**：扩 kv / kv_compress / indexer_k；不扩 q / attn_o / residual / hidden
3. **KV 不 detach**
4. **精度一致性优先**
5. **Phase 1 对齐 prefix**：`P % compress_ratio == 0`

## 7. 文件清单

### 新增

| 文件 | 内容 |
|------|------|
| `core/prefix_store.py` | `StoredG2Activation` + `G2AttentionStore`（修改已有文件） |
| `backends/g2_attention_utils.py` | `_split_by_cu_seqlens`、`_compute_cmp_lengths`、`_adjust_cu_seqlens_for_batch`、`_merge_g2_fields` |
| `integrations/g2_attention.py` | `_g2_kv_store_or_expand`、`_g2_store_with_kwargs` |
| `integrations/g2_batch.py` | `wrap_forward_step()` |
| `setup/patches/mindspeed_deepseek4/__init__.py` | PATCH_SET (1 个 PatchSpec) |
| `setup/patches/mindspeed_deepseek4/attention.py` | 1 个插入点 |
| `tests/unit_test/test_g2_store.py` | Store 测试 |
| `tests/unit_test/test_g2_attention_utils.py` | 工具函数测试 |
| `tests/unit_test/test_g2_attention.py` | 插入点逻辑测试 |

### 修改

| 文件 | 修改 |
|------|------|
| `core/__init__.py` | 导出 `StoredG2Activation` 等 |
| `backends/__init__.py` | 导出 `_split_by_cu_seqlens` 等 |
| `integrations/context.py` | `_create_store()` 工厂 |
| `integrations/verl_mcore.py` | `PrefixSharingRuntimeState.model_type` |
| `setup/compat_matrix.py` | MindSpeed DeepSeek4 条目 |
