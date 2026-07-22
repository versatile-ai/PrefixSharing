# PrefixSharing → DeepSeek V4 集成设计方案

> **文档定位**：本文档描述在 **PrefixSharing 项目**（`/Users/kevin/code/PrefixSharing`）中新增 DeepSeek V4 支持的完整设计方案。通过 monkey-patch 方式注入 MindSpeed-LLM 的 DeepSeek V4 训练流程，不改动 MindSpeed 源码。

## 1. 需求

### 1.1 背景

**PrefixSharing** 是一个在 RL 训练 pipeline 中实现 micro-batch 内 prefix KV 复用的框架。当前已支持标准 Megatron SelfAttention（Qwen2.5/3.5 等），需要扩展到 MindSpeed 的 **DeepSeek V4 Flash** 模型训练。

### 1.2 目标

在 DeepSeek V4 预训练/续训场景中，当同一 micro-batch 内多条序列共享前缀时，provider 序列计算一次前缀 KV，reuser 序列直接复用，消除冗余计算。

### 1.3 核心原则

1. **精度一致性 > 性能**：启用 prefix sharing 后的 logprob / loss / 梯度必须与 baseline 完全一致
2. **KV 不 detach**：缓存 prefix KV 时保留完整 autograd 计算图
3. **One-Forward + KV Injection + Prefix-Last Restore**
4. **Monkey-patch 注入，不改动 MindSpeed 源码**：遵循 PrefixSharing 现有注入模式
5. **方案 A（Fork forward 编排逻辑）先行**：长远可向 MindSpeed 提 PR 加入显式 Hook（方案 B）

## 2. 分析

### 2.1 DeepSeek V4 G2 Attention 与标准 Attention 的差异

在适配 DeepSeek V4 之前，PrefixSharing 已支持标准 Megatron SelfAttention（Qwen2.5/3.5 等）。DeepSeek V4 的 G2 Attention 与标准 Attention 存在以下 **6 个根本性差异**，导致每个维度都需要特殊处理：

#### 差异 1：MLA KV 是压缩向量，非多头张量

| | 标准 Attention | DeepSeek V4 G2 |
|---|---|---|
| KV 投影 | `Linear(h, 2*n_kv_heads*head_dim)` → 分离的 K、V | `LinearNoTP(h, 512)` → 单一向量 |
| KV Shape | `[s, num_kv_heads, head_dim]` | `[s, 512]`（无 head 维度） |

**设计影响**：`PrefixAttentionStore` 的 `StoredAttentionKV(key_tensor, value_tensor)` 模型不适用 → 新建 `G2AttentionStore` + `StoredG2Activation`（第 4.1 节）。

#### 差异 2：融合算子不可替换

| | 标准 Attention | DeepSeek V4 G2 |
|---|---|---|
| 注意力计算 | `flash_attn_varlen_func`（通用 GPU） | `self.sparse_attention()` / `self.sparse_attention_with_indexer_loss()`（实例方法，NPU 专用） |

**设计影响**：不能像标准 attention patch 那样"提取 QKV → 自己做 attention 计算"。必须让 MindSpeed 的 `self.sparse_attention()` 正常运行，PrefixSharing 只在调用前后做 **pre/post hook**（扩展入参、扩展返回值）→ 采用**方案 A（Fork forward 编排逻辑）**（第 5.1 节）。

#### 差异 3：两次 RoPE，attn_o 捕获点在中间

| | 标准 Attention | DeepSeek V4 G2 |
|---|---|---|
| RoPE 次数 | 1 次（Q/K 前） | 2 次（Q 前 + output 后，第二次用 `apply_rotary_emb(..., inverse=True)`） |
| RoPE 函数 | `apply_rotary_pos_emb` | `apply_rotary_emb`（不同函数，不接收 cu_seqlens，按 packed 位置索引 freqs） |

**设计影响**：第二次 RoPE 必须在 attn_o 扩展**之前**各自执行，确保每部分的 encode/decode 位置对称（详见第 2.4 节 RoPE 分析）。Provider 存储 post-RoPE attn_o，Reuser 拼接两边均已逆 RoPE 过的 prefix + suffix。

#### 差异 4：稀疏注意力有额外数据结构

| | 标准 Attention | DeepSeek V4 G2 |
|---|---|---|
| 额外输入 | 无 | `kv_compress`（压缩 KV）、`compress_topk_idxs`（稀疏索引） |
| 压缩阈值 | — | `compress_ratio > 1`（ratio=1 等价于无压缩） |
| Compressor | 无 | `self.compressor(hidden_states, start_pos, local_freqs_cis, packed_seq_params)`（4 参数） |

**设计影响**：扩展 KV 时，`kv_compress` 和 `compress_topk_idxs` 也必须同步扩展/调整；`cu_seqlens_kv` 和 `cu_seqlens_cmp_kv` 需反映 expanded 长度（第 5.2-5.3 节）。

#### 差异 5：MHC 带来的 Residual Shape 不匹配

| | 标准 Attention | DeepSeek V4 G2 |
|---|---|---|
| 额外模块 | 无 | MHC Pre / MHC Post（Sinkhorn 归一化） |
| Hidden shape | `[s, b, h]` | `[s, b, 4, h]`（4 个 chunk） |
| Residual 语义 | attention 的 residual = 输入 hidden | residual = MHC Pre 的输入（含 4 chunk 维度） |

**关键认知**：MHC Pre 是 **per-position** 操作（Sinkhorn 在每个位置独立做 `[4,4]` 归一化），suffix-only 输入直接跑，结果等价。

**真正的问题**：Attention 内部 KV 扩展后返回 `[P+S, ...]`，但 residual/post/comb 仍是 `[suffix, ...]`，shape 不匹配。需要在 Attention 返回后从 provider store 扩展 residual/post/comb → 需要**额外 patch TransformerLayer._forward_attention()**（第 7.1 节）。

**注意**：`_forward_mlp` **不需要 patch**——`_forward_attention` 已将 reuser 的 hidden_states 扩展到 `[P+S, ...]`，进入 `_forward_mlp` 时已是完整长度。

#### 差异 6：层间异构 + DSA Indexer Loss

| compress_ratio | Indexer | sparse_attention 路径 | Phase 1 |
|---|---|---|---|
| ≤1（前 2 层等价） | 无 | `self.sparse_attention()` | ✅ 完整支持 |
| 128（大部分层） | 无（静态 topk） | `self.sparse_attention()` | ✅ 完整支持 |
| 4（少量层） | DSA Indexer | `self.sparse_attention_with_indexer_loss()` 或 `self.sparse_attention()` + DSA loss | 渐进开发：Layer 1 跳过 → Layer 2 Provider 存储 → Layer 3 Reuser 扩展（第 6 节） |

此外，ratio=4 路径有 `DSAIndexerLossAutoScaler.apply(o, loss)` 对 attention output 做 in-place 修改。ratio=4 的 `compress_topk_idxs` 由 `self.indexer.forward_with_scores_compress()` 基于 hidden states 生成（非纯位置），无法像 ratio=128 那样用 expanded seqlen 简单重算——这是 ratio=4 需要独立渐进设计的根本原因。

### 2.2 差异对 PrefixSharing 的影响

从六个差异出发，分析 PrefixSharing 现有模块的复用情况：

```
prefix-sharing/prefix_sharing/
├── core/          → 框架无关：TriePrefixDetector, PrefixSharingPlanner, PrefixSharingPlan, PrefixAttentionStore
├── backends/      → 硬件执行：TorchReferenceBackend, PackedBatchLayout
├── integrations/  → 框架适配：verl_mcore, megatron_runtime, context
└── setup/patches/ → Monkey-patch：PatchRegistry, import hook
```

**可直接复用**：

| 模块 | 差异关联 | 说明 |
|------|----------|------|
| `core/config.py` | — | 配置验证，扩展 `model_type="deepseek4"` |
| `core/prefix_detector.py` | — | 前缀检测，完全通用 |
| `core/planner.py` | — | 执行计划，完全通用 |
| `core/prefix_store.py` → base class | 差异 1 | `PrefixActivationStore` 生命周期，继承复用 |
| `backends/packed_layout.py` | — | Packed 布局管理，直接复用 |
| `integrations/context.py` | 差异 1 | Runtime context，扩展 store 类型切换 |
| `integrations/verl_mcore.py` → restore 函数 | — | `restore_reuser_prefix_columns_2d()` 与 attention 类型无关 |
| `integrations/parallel_info.py` | — | 并行拓扑信息，直接复用 |

**不能直接复用 / 需新增**：

| 模块 | 差异关联 | 原因 |
|------|----------|------|
| `integrations/megatron_runtime.py` | 差异 2 | 针对标准 SelfAttention，完全替换 attention 计算 |
| `integrations/megatron_attention.py` | 差异 2 | Patch 目标为 `megatron.core.transformer.attention.SelfAttention` |
| `backends/torch_ref.py` | 差异 1、4 | 针对标准多头 K/V/Q 格式，不适用于 MLA |
| **新建** `core/prefix_store.py` → G2 扩展 | 差异 1、5 | `G2AttentionStore` + `StoredG2Activation`（第 4.1 节） |
| **新建** `integrations/g2_attention.py` | 差异 2、3、4 | Fork forward + 4 Hook 函数（第 5.1 节） |
| **新建** `integrations/g2_transformer.py` | 差异 5 | TransformerLayer residual 扩展（第 7.1 节） |
| **新建** `backends/g2_attention_utils.py` | 差异 4、6 | topk/cu_seqlens 调整工具函数（第 5.2-5.3 节） |

### 2.3 差异对 MindSpeed 集成的影响

本节从差异出发，分析在 MindSpeed 代码中需要注入的位置（Hook 点）以及关键 API 行为。

#### 2.3.1 Hook 点分析（差异 2、3、4 的集成位置）

基于 MindSpeed 实际代码（`mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py:360-582`）：

```
输入: hidden_states [q_len_local, b, 4096]

Phase 1: Q/KV 投影 + RoPE (line 386-418)
  q_compressed  = self.linear_q(hidden_states)
  kv_compressed = self.linear_kv(hidden_states)
  q = self.q_layernorm → self.linear_q_up_proj → view → RMSNorm
  q = apply_rotary_emb(q[..., -rope_dim:], global_freqs_cis)  ← self.get_freqs_cis(..., get_global=True)
  kv = self.kv_layernorm(kv_compressed)
  kv = apply_rotary_emb(kv[..., -rope_dim:], local_freqs_cis)  ← self.get_freqs_cis(..., get_global=False)
  kv = gather_from_sp_cp(kv)                                    ← Hook A: KV 就绪

Phase 2: compress_topk_idxs 生成 (line 421-468)
  offset = 0 if self.use_sparse_flash_attn else kv.size(0)
  - ratio=4:  self.indexer.forward_with_scores_compress(...)    ← DSA Indexer 路径
  - ratio>1 且无 indexer: self.get_compress_topk_idxs(...)       ← 静态路径（有 @lru_cache）
  - ratio≤1:  compress_topk_idxs = None

Phase 3: Compressed KV (line 472-477)
  kv_compress = self.compressor(hidden_states, start_pos, local_freqs_cis, packed_seq_params)
  kv_compress = gather_from_sp_cp(kv_compress)                  ← Hook B: CMP KV 就绪

Phase 4: Sparse Attention (line 481-555)
  路径A (use_smla_with_slig): o = self.sparse_attention_with_indexer_loss(...)
  路径B (default):            o = self.sparse_attention(q, kv, kv_compress,
                                  compress_topk_idxs, self.attn_sink, self.softmax_scale,
                                  self.compress_ratio, q_len_global, packed_seq_params)
                              ← Hook C (pre): 扩展 KV/CMP KV, 调整 indices/cu_seqlens
                              + DSA indexer loss 计算 (line 514-555)

Phase 5a: 第二次 RoPE（在扩展前执行） (line 557-560)
  o = apply_rotary_emb(o[..., -rope_dim:], global_freqs_cis, inverse=True)  ← 逆 RoPE
                              ← Hook D: Provider 存 post-RoPE attn_o；Reuser 扩展 attn_o
                                （拼接两边都已 RoPE 过的 prefix + suffix）

Phase 5b: Output Projection (line 562-582)
  o = rearrange → einsum(w_woa)
  core_attn_out, bias = self.linear_o_up_proj(o.flatten(2))
  return core_attn_out, bias
```

**4 个 Hook 点**：

| Hook | 位置 | Provider | Reuser |
|------|------|----------|--------|
| A | `gather_from_sp_cp(kv)` 后 | 存 `kv` 到 store | — |
| B | `gather_from_sp_cp(kv_compress)` 后 | 存 `kv_compress` 到 store | — |
| C | `self.sparse_attention()` 调用前 | — | 扩展 kv/kv_compress，调整 indices/cu_seqlens |
| D | 第二次 RoPE 后、rearrange 前 | 存 post-RoPE `attn_o` 到 store | 扩展 attn_o（拼接两边都已 RoPE 过的 prefix + suffix） |

**关键 API 确认**（基于实际代码）：

| 项目 | 实际代码 | 备注 |
|------|----------|------|
| 稀疏 attention | `self.sparse_attention(q, kv, kv_compress, compress_topk_idxs, self.attn_sink, self.softmax_scale, self.compress_ratio, q_len_global, packed_seq_params)` | 实例方法，非独立函数 |
| Indexer loss 路径 | `self.sparse_attention_with_indexer_loss(q, kv, kv_compress, compress_topk_idxs, self.attn_sink, self.softmax_scale, self.compress_ratio, q_len_global, query_index, key_index, weights, packed_seq_params)` | 额外接收 indexer 中间结果 |
| Q RoPE | `self.get_freqs_cis(start_pos, q_len_local, get_global=True)` → `apply_rotary_emb(q[..., -rope_dim:], global_freqs_cis)` | 实例方法生成 freqs |
| KV RoPE | `self.get_freqs_cis(start_pos, q_len_local, get_global=False)` → `apply_rotary_emb(kv[..., -rope_dim:], local_freqs_cis)` | local freqs |
| Output RoPE | `apply_rotary_emb(o[..., -rope_dim:], global_freqs_cis, inverse=True)` | 第三参数 True 表示逆 RoPE |
| Compressor | `self.compressor(hidden_states, start_pos, local_freqs_cis, packed_seq_params)` | 4 参数 |
| Topk 生成 | `self.get_compress_topk_idxs(ratio, bsz, q_len_global, start_pos, offset, cp_shard)` | 实例方法，有 `@lru_cache(maxsize=2)` |
| KV gather | `gather_from_sp_cp(kv)` | 独立函数，CP=1 时 no-op |
| Output 投影 | `core_attn_out, bias = self.linear_o_up_proj(o.flatten(2))` | bias 由此产生 |
| offset 变量 | `offset = 0 if self.use_sparse_flash_attn else kv.size(0)` | 影响 topk 索引偏移 |
| q_len 计算 | `q_len = q_len_local * tp_size if sequence_parallel else q_len_local` | 局部 vs 全局长度 |

#### 2.3.2 RoPE packed 格式索引行为（差异 3 的关键分析）

**为什么 Hook D（attn_o 存储/扩展）必须放在第二次 RoPE 之后，而非之前。**

DeepSeek4 的 RoPE 实现与标准 Megatron 不同。标准 Megatron 的 `apply_rotary_pos_emb` 接收 `cu_seqlens` 参数，在 THD packed 格式下按 per-sequence 绝对位置索引 freqs。但 DeepSeek4 的 `apply_rotary_emb` **不接收 `cu_seqlens`**，而是按 packed tensor 的 dim=0 位置逐元素索引 `freqs_cis`：

```python
# g2_attention.py 中的 RoPE 调用模式
q = q.transpose(0, 1)                                    # [s, b, n, d] → [b, s, n, d]
q[..., -rope_dim:] = apply_rotary_emb(q[...], freqs_cis) # freqs_cis 按位置 0,1,2,... 索引
q = q.transpose(0, 1)
```

其中 `freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=True)`，长度为 `q_len_global`（packed Q 的全局长度），覆盖绝对位置 `[start_pos, start_pos + q_len_global)`。

**核心结论**：`apply_rotary_emb` 在第 `i` 个 packed 位置使用的频率是 `freqs_cis[i]`（对应于绝对位置 `start_pos + i`），**而非**该 token 在原始序列中的 `position_id`。因此：

- RoPE 的 **编码（forward）和解码（inverse）必须作用在相同 packed 位置**，使用相同 freqs，才能正确抵消。
- RoPE 的正确性不依赖于 token 的语义位置，只依赖于 encode/decode 位置对称。

**为什么 Hook D 必须在第二次 RoPE 之后**：

```
方案 A（旧 — 错误）：RoPE 在扩展之后
  Reuser suffix 的 1st RoPE 用 freqs[full_len_0 : full_len_0+suffix] 编码
  concat 后 2nd RoPE 用 freqs[0 : prefix+suffix] 解码
  → suffix 的编码位置和解码位置不同 → 精度错误！

方案 B（新 — 正确）：RoPE 在扩展之前
  Provider prefix 的 1st RoPE 用 freqs[0:prefix] 编码
    → sparse_attention → 2nd RoPE 用相同 freqs[0:prefix] 解码 → Hook D 存 post-RoPE attn_o
  Reuser suffix 的 1st RoPE 用 freqs[full_len_0:full_len_0+suffix] 编码
    → sparse_attention → 2nd RoPE 用相同 freqs 解码 → Hook D 扩展：cat(provider 已解码 prefix, reuser 已解码 suffix)
  → 每部分的 encode/decode 在同一 forward 中位置对称 → 精度正确！
```

**一个潜在的误解**：有人可能认为 suffix 的逆 RoPE 用了 "错误的" 位置（`[full_len_0, ...)` 而非 `[prefix_len, ...)`），但这不是错误——因为正向 RoPE 也用了相同的 "错误" 位置。RoPE 的数学性质是：`RoPE⁻¹(RoPE(x, p), p) = x`，只要 `p` 一致。`p` 不需要等于语义上的 position_id，只要 encode 和 decode 用同一个 `p` 即可。

**额外的正确性保证**：即使 RoPE 实现改为使用绝对 `position_ids`（而非 packed 位置），方案 B 仍然正确——因为 provider prefix 和 reuser suffix 在各自 forward 中的 `position_ids` 与原始序列一致（batch 裁剪保留了 `position_ids`）。


## 3. 总体架构与设计

第 2 节的六个差异驱动了五组功能设计。每个功能组可独立开发、独立验证。

#### 追溯表

| 差异 | 对 PrefixSharing 的影响 | 设计要点 | 功能组 |
|------|------------------------|----------|--------|
| 1 MLA KV 压缩向量 | 需新建 typed store | `G2AttentionStore` + `StoredG2Activation`，遵循 base + wrapper 模式 | **A. 数据存储** |
| 2 融合算子不可替换 | 不能替换 attention，只能 pre/post hook | Fork forward 编排逻辑（方案 A），调用原 `self.sparse_attention()` | **B. Attention — ratio=128** |
| 3 两次 RoPE | Hook D 必须在 2nd RoPE 后执行 | RoPE encode/decode 位置对称原理（见 2.3.2），扩展 post-RoPE attn_o | **B. Attention — ratio=128** |
| 4 稀疏数据结构 | 需同步扩展 `kv_compress` / `compress_topk_idxs` / `cu_seqlens` | `_adjust_topk_indices_for_batch`、`_adjust_cu_seqlens_for_batch`、`_compute_cmp_lengths` | **B. Attention — ratio=128** |
| 5 MHC residual shape | Attention 返回 [P+S] 但 residual 是 [suffix] | TransformerLayer patch：Attention 后扩展 residual/post/comb | **D. Transformer 注入** |
| 6 层间异构 | ratio=4 DSA Indexer 需独立设计 | 渐进三层：Layer 1 Fork 覆盖 + skip、Layer 2 Provider 存储、Layer 3 Reuser index 调整 + 扩展（3.4） | **C. Attention — ratio=4** |

#### 功能组分解

```
┌─────────────────────────────────────────────────────────────────┐
│  A. 数据存储层（差异 1）                                [4]    │
│  ├─ core/prefix_store.py       G2AttentionStore, StoredG2Act.   │
│  ├─ integrations/context.py    _create_store() 工厂             │
│  └─ 验证: store 生命周期, merge 正确性                           │
├─────────────────────────────────────────────────────────────────┤
│  B. Attention 注入 — ratio=128 静态路径（差异 2、3、4） [5]    │
│  ├─ setup/patches/.../attention.py   Fork forward + 4 Hook      │
│  ├─ integrations/g2_attention.py     pre/post hook 辅助函数      │
│  ├─ backends/g2_attention_utils.py   topk/cu_seqlens 调整       │
│  └─ 验证: KV 扩展等价性, topk 正确性                            │
├─────────────────────────────────────────────────────────────────┤
│  C. Attention 注入 — ratio=4 DSA Indexer（差异 6）     [6]    │
│  ├─ 3.4.1 关键差异                                              │
│  ├─ 3.4.2-3.4.4 渐进三层（Fork覆盖 → Provider存储 → Reuser扩展）│
│  └─ 验证: ratio=4 逐层验证                                       │
├─────────────────────────────────────────────────────────────────┤
│  D. Transformer 注入（差异 5）                         [7]    │
│  ├─ setup/patches/.../transformer.py   Residual 扩展            │
│  ├─ integrations/g2_transformer.py     store/expand 辅助函数     │
│  └─ 验证: MHC Post 输出一致性                                  │
├─────────────────────────────────────────────────────────────────┤
│  E. 训练流程集成（全局）                              [8]    │
│  ├─ integrations/g2_batch.py     wrap_forward_step()            │
│  ├─ (复用) verl_mcore.py         prefix-last restore            │
│  └─ 验证: 端到端 loss/logprob/grad 精度                         │
└─────────────────────────────────────────────────────────────────┘
```

#### 模块映射

```
PrefixSharing 项目（prefix-sharing/prefix_sharing/）
│
├── core/
│   ├─ prefix_store.py  ← [A] 新增 G2AttentionStore + StoredG2Activation
│   └─ config.py        ← [A] 扩展 model_type + validate 白名单
│
├── backends/
│   └─ g2_attention_utils.py  ← [B] 新增 _split_by_cu_seqlens,
│                                _merge_g2_fields, _adjust_topk_indices_for_batch,
│                                _adjust_cu_seqlens_for_batch, _compute_cmp_lengths
│
├── integrations/
│   ├─ g2_attention.py   ← [B/C] 新增 _g2_store_per_sequence,
│   │                            _g2_expand_kv_and_adjust, _g2_expand_attn_output
│   ├─ g2_transformer.py ← [D] 新增 _g2_store_transformer_data,
│   │                            _g2_expand_transformer_data
│   ├─ g2_batch.py       ← [E] 新增 wrap_forward_step()
│   └─ context.py        ← [A] 修改 _create_store + 签名变更
│
└── setup/patches/mindspeed_deepseek4/
    ├─ __init__.py        ← PATCH_SET 定义
    ├─ attention.py       ← [B/C] DeepSeek4SelfAttention.forward
    └─ transformer.py     ← [D] TransformerLayer._forward_attention

MindSpeed-LLM（不修改源码）
  mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention
    └─ DeepSeek4SelfAttention.forward          ← attention patch [B/C]
  mindspeed_llm.core.transformer.transformer_layer
    └─ TransformerLayer._forward_attention     ← transformer patch [D]
```

## 4. 功能组 A：数据存储层（差异 1 — MLA KV 压缩向量）

> **差异回顾**：G2 的 KV 是单一压缩向量 `[s, 512]`，非分离的多头 K/V 张量。现有 `PrefixAttentionStore` 的 `StoredAttentionKV(key_tensor, value_tensor)` 模型不适用。

**设计组件**：
- `G2AttentionStore` + `StoredG2Activation`：新型 typed store，存储 kv/kv_compress/attn_o/residual/post/comb
- `_create_store()` 工厂：根据 `model_type="deepseek4"` 创建 G2AttentionStore
- `_merge_g2_fields` / `_merge_g2_transformer_fields`：frozen dataclass 增量更新

详见下方 3.2.1（Store 定义）和 3.2.2（Context 扩展）。

### 4.1 G2AttentionStore

新建 G2 专用 store，遵循现有 `PrefixActivationStore` base + typed wrapper 模式。新增到 `prefix-sharing/prefix_sharing/core/prefix_store.py`：

```python
PREFIX_STATE_TYPE_G2_ATTENTION = "g2_attention"


@dataclass(frozen=True)
class StoredG2Activation:
    """DeepSeek4 G2 MLA prefix activation — 一次存储覆盖所有扩展需求。

    与 StoredAttentionKV 的核心区别:
    - kv 是 MLA 压缩后的单一张量 [s, 512]，不是分离的 K/V 多头张量
    - 额外携带 kv_compress / attn_o / residual_prefix / post_prefix / comb_prefix

    所有字段均可为 None（增量存储时每次只更新部分字段）。
    """
    kv: Any | None = None              # [prefix_len, 512]
    kv_compress: Any | None = None     # [prefix_len//ratio, 512]
    attn_o: Any | None = None          # [prefix_len, n_local, 512]
    residual_prefix: Any | None = None # [prefix_len, 4, 4096] (MHC)
    post_prefix: Any | None = None     # [prefix_len, 4]       (MHC)
    comb_prefix: Any | None = None     # [prefix_len, 4, 4]    (MHC)
    indexer_score: Any | None = None   # [prefix_len//ratio, index_topk] — DSA Indexer 相关性分数（ratio=4 专用，Layer 2+）
    stored_len: int = 0  # 存储的完整长度（provider=valid_len, reuser=prefix_len+valid_len），非可共享前缀长度


class G2AttentionStore(PrefixActivationStore):
    """DeepSeek4 G2 attention 专用 typed store。

    与 PrefixAttentionStore / PrefixDeltanetStore 并列，
    遵循相同的 base + typed wrapper 模式。
    """

    def store(
        self, slot_id: PrefixActivationSlotId, *,
        kv=None, kv_compress=None, attn_o=None,
        residual_prefix=None, post_prefix=None, comb_prefix=None,
        indexer_score=None,
        stored_len: int, overwrite: bool = False,
    ) -> None:
        if slot_id.prefix_state_type != PREFIX_STATE_TYPE_G2_ATTENTION:
            raise ValueError(
                "G2AttentionStore requires prefix_state_type='g2_attention'")
        if stored_len < 0:
            raise ValueError("stored_len must be >= 0")
        self.store_entry(slot_id, entry=StoredG2Activation(
            kv=kv, kv_compress=kv_compress, attn_o=attn_o,
            residual_prefix=residual_prefix,
            post_prefix=post_prefix, comb_prefix=comb_prefix,
            indexer_score=indexer_score,
            stored_len=stored_len,
        ), overwrite=overwrite)

    def load(self, slot_id: PrefixActivationSlotId) -> StoredG2Activation:
        entry = self.load_entry(slot_id)
        if not isinstance(entry, StoredG2Activation):
            raise TypeError(
                f"Stored prefix state is not G2 activation for {slot_id}")
        return entry
```

**增量更新辅助函数**（`prefix-sharing/prefix_sharing/backends/g2_attention_utils.py`）：

```python
def _merge_g2_fields(existing, field, new_tensor):
    """基于已有 StoredG2Activation 创建更新单个字段的新实例。

    由于 StoredG2Activation 是 frozen dataclass，每次合并都创建新实例。
    返回 StoredG2Activation 实例（非 dict），调用方通过 _g2_store_with_kwargs 存入。
    """
    if existing is None:
        return StoredG2Activation(
            **{field: new_tensor}, stored_len=new_tensor.shape[0])
    return StoredG2Activation(
        kv=new_tensor if field == "kv" else existing.kv,
        kv_compress=new_tensor if field == "kv_compress" else existing.kv_compress,
        attn_o=new_tensor if field == "attn_o" else existing.attn_o,
        residual_prefix=(new_tensor if field == "residual_prefix"
                         else existing.residual_prefix),
        post_prefix=new_tensor if field == "post_prefix" else existing.post_prefix,
        comb_prefix=new_tensor if field == "comb_prefix" else existing.comb_prefix,
        indexer_score=(new_tensor if field == "indexer_score"
                       else existing.indexer_score),
        stored_len=max(existing.stored_len, new_tensor.shape[0]),
    )


def _merge_g2_transformer_fields(existing, *, residual_prefix, post_prefix,
                                  comb_prefix, valid_len):
    """合并 transformer 字段（residual/post/comb），保留已有 attention 字段。

    与 _merge_g2_fields 不同，此函数同时更新 3 个字段，避免多次 store。
    """
    if existing is None:
        return StoredG2Activation(
            residual_prefix=residual_prefix,
            post_prefix=post_prefix, comb_prefix=comb_prefix,
            stored_len=valid_len)
    return StoredG2Activation(
        kv=existing.kv, kv_compress=existing.kv_compress, attn_o=existing.attn_o,
        residual_prefix=residual_prefix,
        post_prefix=post_prefix, comb_prefix=comb_prefix,
        indexer_score=existing.indexer_score,
        stored_len=max(existing.stored_len, valid_len),
    )


def _compute_cmp_lengths(layout, compress_ratio, kv_compress_shape_0):
    """计算压缩 KV 的 per-sequence padded lengths。

    kv_compress 的 dim=0 长度是 sum(valid // ratio)，不能用 Q 路径的 padded_lengths。
    从 valid_lengths 计算，并通过运行时断言检测 compressor 内部对齐导致的长度偏差。
    """
    lengths = [vl // compress_ratio for vl in layout.valid_lengths]
    assert sum(lengths) == kv_compress_shape_0, \
        (f"CMP KV length mismatch: computed={sum(lengths)} "
         f"!= actual={kv_compress_shape_0}. "
         f"compressor may apply TP padding — adjust _compute_cmp_lengths accordingly")
    return lengths
```

**存储时序**（同一 micro-batch，同一层内，按 batch 顺序）：

```
Provider (batch_idx=0):
  1. TransformerLayer: store(residual_prefix, post_prefix, comb_prefix)
  2. Attention Hook A:  store(kv)
  3. Attention Hook B:  store(kv_compress)
  4. 第二次 RoPE 后:     store(post-RoPE attn_o)

Reuser (batch_idx=1, provider=0):
  1. TransformerLayer: MHC Pre (suffix-only, 正常执行)
  2. Attention Hook C:  load(provider kv, kv_compress)   → 扩展 KV/CMP KV
                        store(own kv, kv_compress)        → transitive reuse
  3. 第二次 RoPE 后:     load(provider post-RoPE attn_o) → 扩展 attn_o
                        store(own attn_o)                → transitive reuse
  4. TransformerLayer:  load(provider residual, post, comb) → 扩展 residual
```

### 4.2 Runtime Context 扩展

**修改 `PrefixSharingRuntimeContext`**（`prefix-sharing/prefix_sharing/integrations/context.py`）：

```python
# 1. 类型签名：store 从 PrefixAttentionStore 放宽为 base class
@dataclass(init=False)
class PrefixSharingRuntimeContext:
    ...
    store: PrefixActivationStore  # 原为 PrefixAttentionStore
    ...

    def __init__(self, runtime_state: Any, store: PrefixActivationStore) -> None:
        # 原签名: store: PrefixAttentionStore
        ...

# 2. prefix_sharing_runtime_context() 中 store 创建改为工厂函数
@contextmanager
def prefix_sharing_runtime_context(
    prefix_sharing_runtime_state: Any | None,
) -> Iterator[PrefixSharingRuntimeContext | None]:
    if prefix_sharing_runtime_state is None:
        yield None
        return

    store = _create_store(prefix_sharing_runtime_state)  # 原为 PrefixAttentionStore()
    ...


def _create_store(runtime_state) -> PrefixActivationStore:
    """根据 model_type 创建对应的 store 实例。"""
    model_type = getattr(runtime_state, 'model_type', 'text_only_causal_lm')
    if model_type == 'deepseek4':
        from prefix_sharing.core.prefix_store import G2AttentionStore
        return G2AttentionStore()
    from prefix_sharing.core.prefix_store import PrefixAttentionStore
    return PrefixAttentionStore()
```

**修改 `PrefixSharingRuntimeState`**（`prefix-sharing/prefix_sharing/integrations/verl_mcore.py`）：

```python
@dataclass(frozen=True)
class PrefixSharingRuntimeState:
    prefix_sharing_plan: PrefixSharingPlan
    attention_backend: Any
    packed_batch_layout: PackedBatchLayout
    parallel_info: MegatronParallelInfo
    kept_position_ids: Any | None = None
    model_type: str = "text_only_causal_lm"  # 新增：触发 _create_store 选择
```

### 4.3 验证用例（功能组 A — 数据存储层）

本组验证不依赖 MindSpeed 运行时，可在纯 PyTorch 环境下执行。

| 测试 | 内容 | 预期 |
|------|------|------|
| `test_g2_store_lifecycle` | store/load/overwrite/close 生命周期 | load 返回正确的 StoredG2Activation；close 后 store/load 抛异常 |
| `test_g2_store_incremental` | 同一 slot 分三次 store（kv → kv_compress → attn_o） | 最终 load 返回完整数据，merge 正确 |
| `test_merge_g2_fields` | `_merge_g2_fields` 逐个字段更新 | 新字段覆盖，旧字段保留，stored_len 正确 |
| `test_merge_g2_transformer_fields` | `_merge_g2_transformer_fields` 同时更新 3 个字段 | attention 字段保留，transformer 字段更新 |
| `test_create_store_deepseek4` | `model_type="deepseek4"` 时 `_create_store` 返回 `G2AttentionStore` | isinstance 检查通过 |

## 5. 功能组 B：Attention 注入 — ratio=128 静态路径（差异 2、3、4）

> **差异回顾**：`self.sparse_attention()` 不可替换，需 fork forward 做 pre/post hook。RoPE 需在扩展前执行（位置对称）。`kv_compress`/`compress_topk_idxs`/`cu_seqlens` 需同步扩展。

**设计组件**：
- Fork forward + 4 Hook（3.3.1）
- compress_topk_idxs 调整（3.3.2）
- packed_seq_params 调整（3.3.3）
- Attention Mask（3.3.4）
- Skip 层协调（3.3.5）

### 5.1 Fork Forward + 4 Hook

**策略**：Monkey-patch `DeepSeek4SelfAttention.forward`。context 未激活时直接调原始 forward；context 激活时，fork 编排逻辑在 4 个 Hook 点插入 prefix sharing 操作。

**关键设计**：Fork 的是**编排逻辑**，不是底层实现。`self.linear_q()`、`self.compressor()`、`self.sparse_attention()` 等仍调用 MindSpeed 原方法。

**前置步骤**：实现前先读取目标 MindSpeed 版本的 `g2_attention.py`，确认 forward 的精确返回值签名和内部调用的参数列表，防止遗漏。

**维护策略**：Fork 的是编排逻辑（~40 行），底层方法 `self.linear_q()`、`self.sparse_attention()` 等仍调 MindSpeed 原实现。MindSpeed 升级时 diff 对照 `g2_attention.py` 的编排变化即可，内部实现重构不受影响。

```python
# setup/patches/mindspeed_deepseek4/attention.py

def patch_g2_attention(original_forward):
    """创建 DeepSeek4SelfAttention.forward 的 patch wrapper。"""

    def patched_forward(self, hidden_states, attention_mask, rotary_pos_emb,
                        start_pos=0, packed_seq_params=None, attention_bias=None,
                        inference_context=None, rotary_pos_cos=None,
                        rotary_pos_sin=None, sequence_len_offset=None):

        ctx = current_prefix_sharing_context()
        if ctx is None or not isinstance(ctx.store, G2AttentionStore):
            return original_forward(
                self, hidden_states, attention_mask, rotary_pos_emb,
                start_pos=start_pos, packed_seq_params=packed_seq_params,
                attention_bias=attention_bias, inference_context=inference_context,
                rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
                sequence_len_offset=sequence_len_offset)

        # Phase 1: ratio=4 + DSA Indexer 层跳过（不共享），走原始 forward
        if self.compress_ratio == 4 and self.indexer is not None:
            return original_forward(
                self, hidden_states, attention_mask, rotary_pos_emb,
                start_pos=start_pos, packed_seq_params=packed_seq_params,
                attention_bias=attention_bias, inference_context=inference_context,
                rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
                sequence_len_offset=sequence_len_offset)

        plan = ctx.plan
        layout = ctx.packed_batch_layout
        layer_id = self.layer_number
        tp_rank = ctx.parallel_info.tp_rank

        # 捕获顶层变量，供内部 per-sequence 函数使用
        _start_pos = start_pos
        _kv_allgather = self.kv_allgather
        _seq_parallel = self.config.sequence_parallel
        _self_attn = self  # attention module 实例

        q_len_local, bsz, _ = hidden_states.shape
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        cp_size = parallel_state.get_context_parallel_world_size()
        q_len = q_len_local * tp_size if self.config.sequence_parallel else q_len_local
        q_len_global = q_len * cp_size if cp_size > 1 else q_len

        # ═══ Phase 1: Q/KV 投影 + RoPE ═══
        self.freqs_cis = rotary_pos_emb[0] if self.compress_ratio > 1 else rotary_pos_emb[1]
        self.freqs_cis = self.freqs_cis[start_pos : start_pos + q_len_global]
        if self.kv_allgather:
            self.freqs_cis = permute_cp_shard(self.freqs_cis, reorder=False)

        q_compressed = self.linear_q(hidden_states)
        kv_compressed = self.linear_kv(hidden_states)

        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(q_len, bsz, self.n_local_heads, -1)

        # Q RMSNorm + RoPE
        args = get_args()
        if args.use_fused_rmsnorm:
            nD = q.shape[-1]
            norm_gamma = torch.ones(nD, device=q.device, dtype=torch.float32)
            q = torch_npu.npu_rms_norm(q, gamma=norm_gamma, epsilon=self.config.layernorm_epsilon)[0]
        else:
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.config.layernorm_epsilon)

        q = q.transpose(0, 1)
        global_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=True)
        local_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=False)
        q[..., -self.rope_head_dim:] = apply_rotary_emb(q[..., -self.rope_head_dim:], global_freqs_cis)
        q = q.transpose(0, 1)

        # KV Layernorm + RoPE
        kv = self.kv_layernorm(kv_compressed)
        kv = kv.transpose(0, 1)
        kv[..., -self.rope_head_dim:] = apply_rotary_emb(kv[..., -self.rope_head_dim:], local_freqs_cis)
        kv = kv.transpose(0, 1)
        if self.config.sequence_parallel or self.kv_allgather:
            kv = gather_from_sp_cp(kv)

        # ═══ Hook A: Provider 存 KV ═══
        _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank,
                               field="kv", tensor=kv)

        # ═══ Phase 2: compress_topk_idxs ═══
        compress_topk_idxs = None
        if self.compress_ratio > 1:
            offset = 0 if self.use_sparse_flash_attn else kv.size(0)
            if self.indexer is not None:
                query_index, key_index, weights, dsa_hidden_states = \
                    self.indexer.forward_with_index_compress(
                        hidden_states.detach(), q_compressed.detach(),
                        start_pos, local_freqs_cis, packed_seq_params)
                query_index, key_index, weights = \
                    self.indexer.all_gather_qk_weight_kvallgather(
                        query_index, key_index, weights)
                dsa_indexer_context = torch.no_grad() \
                    if args.use_fused_lightning_indexer_loss else nullcontext()
                with dsa_indexer_context:
                    compress_topk_idxs, compress_topk_score = \
                        self.indexer.forward_with_scores_compress(
                            dsa_hidden_states, query_index, key_index, weights,
                            attention_mask, packed_seq_params, start_pos,
                            self.indexer.index_topk, offset, self.indexer.compress_ratio)
                    compress_topk_idxs, compress_topk_score = \
                        self.indexer.post_process_index(compress_topk_idxs, compress_topk_score)
            else:
                compress_topk_idxs = self.get_compress_topk_idxs(
                    self.compress_ratio, bsz, q_len_global, start_pos, offset,
                    self.kv_allgather)

        # ═══ Phase 3: Compressed KV ═══
        kv_compress = None
        if self.compress_ratio > 1:
            kv_compress = self.compressor(
                hidden_states, start_pos, local_freqs_cis, packed_seq_params)
            if kv_compress is not None:
                if self.config.sequence_parallel or self.kv_allgather:
                    kv_compress = gather_from_sp_cp(kv_compress)

        # ═══ Hook B: Provider 存 CMP KV ═══
        _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank,
                               field="kv_compress", tensor=kv_compress)

        # ═══ Hook C: Reuser 扩展 KV/CMP KV + 调整 indices/cu_seqlens ═══
        kv, kv_compress, compress_topk_idxs, packed_seq_params = \
            _g2_expand_kv_and_adjust(
                ctx, layout, plan, layer_id, tp_rank,
                kv, kv_compress, compress_topk_idxs,
                packed_seq_params, self.compress_ratio,
                attention_module=_self_attn,
                start_pos=_start_pos,
                kv_allgather=_kv_allgather,
                sequence_parallel=_seq_parallel)

        self.attn_sink = self.attn_sink.to(hidden_states.device)

        # ═══ Phase 4: Sparse Attention ═══
        use_smla_with_slig = (
            self.indexer is not None
            and args.use_g2_indexer_loss
            and torch.is_grad_enabled()
            and args.use_fused_lightning_indexer_loss
        )
        if use_smla_with_slig:
            o = self.sparse_attention_with_indexer_loss(
                q, kv, kv_compress, compress_topk_idxs,
                self.attn_sink, self.softmax_scale, self.compress_ratio,
                q_len_global, query_index, key_index, weights, packed_seq_params)
        else:
            o = self.sparse_attention(
                q, kv, kv_compress, compress_topk_idxs,
                self.attn_sink, self.softmax_scale, self.compress_ratio,
                q_len_global, packed_seq_params)
            # DSA indexer loss (line 514-555 of original code)
            if (args.use_g2_indexer_loss and self.compress_ratio > 1
                    and self.indexer is not None and torch.is_grad_enabled()):
                compress_topk_idxs_adj = (
                    torch.where(compress_topk_idxs == -1, compress_topk_idxs,
                                compress_topk_idxs - offset)
                    if offset != 0 else compress_topk_idxs)
                if tp_size > 1:
                    total_query = gather_from_tensor_model_parallel_region(
                        q.view(*q.shape[:2], -1))
                    total_query = total_query.view(*q.shape[:2], -1, q.shape[-1])
                else:
                    total_query = q
                if len(kv_compress.shape) == 3:
                    kv_compress_exp = kv_compress.unsqueeze(2)
                else:
                    kv_compress_exp = kv_compress
                main_attn_dist = get_attn_scores(
                    total_query.detach(), kv_compress_exp.detach(),
                    attention_mask, self.n_local_heads * tp_size,
                    self.softmax_scale, allgather_q=True)
                loss = compute_dsa_indexer_loss_dsv4(
                    main_attn_dist, compress_topk_score,
                    compress_topk_idxs_adj, args.indexer_loss_coeff,
                    cmp_ratio=self.compress_ratio)
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss, self.layer_number, self.config.num_layers,
                    avg_group=parallel_state.get_tensor_and_context_parallel_group())
                o = DSAIndexerLossAutoScaler.apply(o, loss)

        # ═══ Phase 5a: 第二次 RoPE（在扩展前执行，freqs 覆盖 suffix-only 的 o） ═══
        # 必须在 Hook D 扩展之前执行，因为 global_freqs_cis 长度为 q_len_global
        # （suffix-only），扩展后 o.shape[0] = P+S 会超出 freqs 范围导致索引越界。
        o = o.transpose(0, 1)
        o_rotated = o.clone()
        o_rotated[..., -self.rope_head_dim:] = apply_rotary_emb(
            o[..., -self.rope_head_dim:], global_freqs_cis, True)
        o = o_rotated.transpose(0, 1)
        # o 现在已完成第二次 RoPE，dim=0 仍为 suffix-only 长度

        # ═══ Hook D (pre): Provider 存 post-RoPE attn_o ═══
        _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank,
                               field="attn_o", tensor=o)

        # ═══ Hook D (post): Reuser 扩展 post-RoPE attn_o ═══
        o = _g2_expand_attn_output(ctx, layout, plan, layer_id, tp_rank, o)
        # o 扩展后 dim=0 = P+S，两边都已正确做过 2nd RoPE

        # ═══ Phase 5b: Rearrange + Output Projection ═══

        o = rearrange(o, 's b (g h) d -> s b g (h d)',
                      s=o.shape[0],  # 动态取扩展后实际长度（reuser 场景下 > q_len）
                      b=bsz,
                      g=self.n_groups // self.world_size,
                      h=self.n_heads // self.n_groups,
                      d=self.head_dim)
        weight_woa = rearrange(
            self.linear_o_down_proj.weight,
            '(g l) (d h)->g l (d h)',
            d=self.head_dim // self.n_groups,
            l=self.o_lora_rank, h=self.n_heads, g=self.n_local_groups)
        o = torch.einsum("sbgd,gld->sbgl", o, weight_woa)

        # bias 来源：self.linear_o_up_proj 返回 (output, bias)
        core_attn_out, bias = self.linear_o_up_proj(o.flatten(2))
        return core_attn_out, bias

    return patched_forward
```

**核心辅助函数**（`prefix-sharing/prefix_sharing/integrations/g2_attention.py`）：

```python
import torch
from prefix_sharing.core.prefix_store import (
    PrefixActivationSlotId, PREFIX_STATE_TYPE_G2_ATTENTION,
    G2AttentionStore, StoredG2Activation)
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.backends.g2_attention_utils import (
    _split_by_cu_seqlens, _merge_g2_fields, _compute_cmp_lengths,
    _adjust_topk_indices_for_batch, _adjust_cu_seqlens_for_batch)


def _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank, field, tensor):
    """逐序列存储 provider 数据到 G2AttentionStore。

    将 packed tensor 按 padded_lengths 拆分，对每个 provider row
    增量存入 store。同一 slot 支持多次调用（先存 kv，再存 kv_compress，再存 attn_o）。
    """
    if tensor is None:
        return
    rows = _split_by_cu_seqlens(tensor, layout.padded_lengths)
    for batch_idx, row in enumerate(rows):
        if not plan.is_provider(batch_idx):
            continue
        valid_len = layout.valid_lengths[batch_idx]
        valid_row = row[:valid_len]
        slot_id = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id,
            batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
        existing = ctx.store.load(slot_id) if ctx.store.contains(slot_id) else None
        new_data = _merge_g2_fields(existing, field, valid_row)
        _g2_store_with_kwargs(ctx.store, slot_id, new_data)


def _g2_expand_kv_and_adjust(ctx, layout, plan, layer_id, tp_rank,
                              kv, kv_compress, compress_topk_idxs,
                              packed_seq_params, compress_ratio, *,
                              attention_module, start_pos, kv_allgather,
                              sequence_parallel):
    """逐序列扩展 reuser 的 KV/CMP KV + 调整 indices/cu_seqlens。

    对 packed tensor 逐序列处理：
    - Provider: 保持原样
    - Reuser: 从 G2AttentionStore 取 provider prefix，拼接在 suffix 前面。
      扩展后回存到 store（支持 transitive reuse）。

    attention_module: DeepSeek4SelfAttention 实例，用于调 get_compress_topk_idxs。
    start_pos, kv_allgather: 从 fork forward 透传，供 topk 调整函数使用。
    """
    kv_rows = _split_by_cu_seqlens(kv, layout.padded_lengths)
    
    # CMP KV 的 dim=0 长度是 sum(valid // ratio)，不能复用 layout.padded_lengths。
    # 使用 packed_seq_params.cu_seqlens_cmp_kv 或从 valid_lengths 计算。
    if kv_compress is not None and compress_ratio > 1:
        cmp_lengths = _compute_cmp_lengths(layout, compress_ratio, kv_compress.shape[0])
        cmp_rows = _split_by_cu_seqlens(kv_compress, cmp_lengths)
    else:
        cmp_rows = []

    expanded_kv, expanded_cmp = [], []
    for batch_idx in range(layout.batch_size):
        valid_len = layout.valid_lengths[batch_idx]
        if not plan.is_reuser(batch_idx):
            expanded_kv.append(kv_rows[batch_idx][:valid_len])
            if cmp_rows:
                cmp_valid_len = valid_len // compress_ratio
                expanded_cmp.append(cmp_rows[batch_idx][:cmp_valid_len])
            continue

        provider_idx = plan.provider_index[batch_idx]
        prefix_len = plan.prefix_lens[batch_idx]
        slot_id = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id,
            provider_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
        provider_data = ctx.store.load(slot_id)

        # 扩展 KV
        expanded_kv_row = torch.cat(
            [provider_data.kv[:prefix_len], kv_rows[batch_idx][:valid_len]], dim=0)
        expanded_kv.append(expanded_kv_row)

        # 扩展 CMP KV（slice 长度用 valid_len // ratio，不是 valid_len）
        expanded_cmp_row = None
        if kv_compress is not None and compress_ratio > 1:
            cmp_prefix_len = prefix_len // compress_ratio
            cmp_valid_len = valid_len // compress_ratio
            expanded_cmp_row = torch.cat(
                [provider_data.kv_compress[:cmp_prefix_len],
                 cmp_rows[batch_idx][:cmp_valid_len]], dim=0)
            expanded_cmp.append(expanded_cmp_row)

        # 回存 expanded KV（transitive reuse）
        own_slot_id = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id,
            batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
        _g2_store_with_kwargs(ctx.store, own_slot_id, StoredG2Activation(
            kv=expanded_kv_row,
            kv_compress=expanded_cmp_row,
            attn_o=None,  # 稍后在 Hook D 存
            residual_prefix=None, post_prefix=None, comb_prefix=None,
            stored_len=prefix_len + valid_len,
        ))

    new_kv = torch.cat(expanded_kv, dim=0)
    new_cmp = torch.cat(expanded_cmp, dim=0) if expanded_cmp else None
    new_indices = _adjust_topk_indices_for_batch(
        compress_topk_idxs, plan, layout, compress_ratio,
        attention_module=attention_module, start_pos=start_pos,
        kv_allgather=kv_allgather, sequence_parallel=sequence_parallel)
    new_packed = _adjust_cu_seqlens_for_batch(
        packed_seq_params, plan, compress_ratio)

    return new_kv, new_cmp, new_indices, new_packed


def _g2_expand_attn_output(ctx, layout, plan, layer_id, tp_rank, o):
    """逐序列扩展 reuser 的 attention output。

    Reuser 扩展后回存 attn_o（transitive reuse）。
    """
    rows = _split_by_cu_seqlens(o, layout.padded_lengths)
    expanded = []
    for batch_idx, row in enumerate(rows):
        valid_len = layout.valid_lengths[batch_idx]
        if not plan.is_reuser(batch_idx):
            expanded.append(row[:valid_len])
            continue
        provider_idx = plan.provider_index[batch_idx]
        prefix_len = plan.prefix_lens[batch_idx]
        slot_id = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id,
            provider_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
        provider_data = ctx.store.load(slot_id)
        expanded_row = torch.cat(
            [provider_data.attn_o[:prefix_len], row[:valid_len]], dim=0)
        expanded.append(expanded_row)

        # 回存 expanded attn_o（transitive reuse）
        # 前提：Hook C 已为当前 reuser 创建了 slot（含 kv/kv_compress）。
        # 如果 slot 不存在（compress_ratio≤1 时 Hook C 可能未创建），跳过回存。
        own_slot_id = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id,
            batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
        try:
            existing = ctx.store.load(own_slot_id)
            merged = _merge_g2_fields(existing, "attn_o", expanded_row)
            _g2_store_with_kwargs(ctx.store, own_slot_id, merged)
        except KeyError:
            pass  # Hook C 未创建 slot，transitive reuse 不适用

    return torch.cat(expanded, dim=0)


def _g2_store_with_kwargs(store, slot_id, data):
    """将 StoredG2Activation 或 dict 显式传参存入 store。

    G2AttentionStore.store() 接受 keyword-only arguments，
    StoredG2Activation 是 frozen dataclass，显式传参避免 **dict 解包的类型安全问题。
    """
    if isinstance(data, dict):
        store.store(slot_id, **data, overwrite=True)
    else:
        store.store(slot_id,
            kv=data.kv, kv_compress=data.kv_compress, attn_o=data.attn_o,
            residual_prefix=data.residual_prefix,
            post_prefix=data.post_prefix, comb_prefix=data.comb_prefix,
            indexer_score=data.indexer_score,
            stored_len=data.stored_len, overwrite=True)
```

### 5.2 compress_topk_idxs 调整

`_adjust_topk_indices_for_batch` 调用 `_adjust_topk_indices_for_batch`，后者对 ratio=128 层重新生成 topk indices，对 ratio≤1 层返回原值（无压缩）。

> **注意**：`self.get_compress_topk_idxs` 有 `@lru_cache(maxsize=2)`。Reuser 的 expanded seqlen 产生新 cache key，不影响 provider 的缓存（各占 1 slot）。

```python
def _adjust_topk_indices_for_batch(compress_topk_idxs, plan, layout, compress_ratio, *,
                                    attention_module, start_pos, kv_allgather,
                                    sequence_parallel):
    """调整 packed 中所有 reuser 序列的 compress_topk_idxs。

    ratio≤1: 无压缩，返回原值。
    ratio>1 (无 indexer): 用 expanded seqlen 重新生成。
    ratio>1 (有 indexer): Phase 1 跳过——此函数不会被调用
                          （fork forward 在 ratio=4 层走 original_forward）。

    原地修改 compress_topk_idxs 并返回同一引用。
    """
    if compress_topk_idxs is None or compress_ratio <= 1:
        return compress_topk_idxs
    # ratio=128 静态路径
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_size = parallel_state.get_context_parallel_world_size()
    bsz = compress_topk_idxs.shape[0]

    # 为每个 reuser 序列独立重建 topk indices
    for batch_idx in range(plan.batch_size):
        if not plan.is_reuser(batch_idx):
            continue
        prefix_len = plan.prefix_lens[batch_idx]
        q_len_local = layout.valid_lengths[batch_idx]
        q_len = q_len_local * tp_size if sequence_parallel else q_len_local
        q_len_global = q_len * cp_size if cp_size > 1 else q_len
        expanded_seqlen = prefix_len + q_len_global
        new_idxs = attention_module.get_compress_topk_idxs(
            compress_ratio, bsz, expanded_seqlen,
            start_pos=start_pos, offset=0, cp_shard=kv_allgather)
        # 原地覆盖该 reuser 对应的行（取 suffix Q 对应的行 + 最后 P+S 列）
        compress_topk_idxs[batch_idx, :q_len_local, :] = \
            new_idxs[batch_idx, -q_len_local:, :]

    return compress_topk_idxs  # 与入参同一引用（原地修改）
```

**限制（Phase 1）**：

1. **单 prefix_len**：当不同 reuser 的 `prefix_len` 不同时，统一用 expanded seqlen 生成的 indices 在列区间上不完全精确。**同组共享前缀场景下所有 reuser 的 prefix_len 相同，此限制不触发。** 混合多前缀长度的 batch 需 Phase 2 完善。

2. **Per-sequence vs total-packed 语义**：`q_len_local = layout.valid_lengths[batch_idx]` 是单条序列的 TP-local suffix 长度，而原始 `get_compress_topk_idxs` 调用接收的是 total-packed 全局 `q_len_global`。本函数将 per-sequence expanded seqlen 传入，依赖 `get_compress_topk_idxs` 内部将 seqlen 视为全局长度并按 `bsz` 均分——这基于该函数是纯位置相关 mask 的假设（ratio=128 静态路径下成立）。实现时需通过单元测试验证。

### 5.3 packed_seq_params 调整

Megatron 的 `packed_seq_params` 是 dataclass，**不能原地修改**。使用 `dataclasses.replace()` 创建新实例：

```python
from dataclasses import replace

def _adjust_cu_seqlens_for_batch(packed_seq_params, plan, compress_ratio):
    """调整 packed_seq_params 中所有 reuser 序列的 cu_seqlens_kv。

    遍历 plan 的所有 batch index，对每个 reuser 将其 cu_seqlens_kv
    增加对应的 prefix_len。
    """
    if packed_seq_params is None:
        return None

    kv_attr = ('cu_seqlens_kv_padded' if hasattr(packed_seq_params, 'cu_seqlens_kv_padded')
               else 'cu_seqlens_kv')
    old_cu_kv = list(getattr(packed_seq_params, kv_attr))

    # 对每个 reuser 序列，其后的所有 cu_seqlens 均增加 prefix_len
    for batch_idx in range(plan.batch_size):
        if plan.is_reuser(batch_idx):
            offset = plan.prefix_lens[batch_idx]
            for i in range(batch_idx + 1, len(old_cu_kv)):
                old_cu_kv[i] += offset

    new_params = replace(packed_seq_params, **{kv_attr: old_cu_kv})

    # cu_seqlens_cmp_kv 同理（如果存在）
    if hasattr(packed_seq_params, 'cu_seqlens_cmp_kv'):
        old_cu_cmp = list(packed_seq_params.cu_seqlens_cmp_kv)
        for batch_idx in range(plan.batch_size):
            if plan.is_reuser(batch_idx):
                cmp_offset = plan.prefix_lens[batch_idx] // compress_ratio
                for i in range(batch_idx + 1, len(old_cu_cmp)):
                    old_cu_cmp[i] += cmp_offset
        new_params = replace(new_params, cu_seqlens_cmp_kv=old_cu_cmp)

    return new_params
```

### 5.4 Attention Mask

THD packed 格式下，`npu_sparse_flash_mla` 用 `cu_seqlens` 隐式确定序列边界和 causal 关系，**不需要显式扩展 mask**。KV 扩展后只需调整 `cu_seqlens_kv`（第 5.3 节），mask 自动适配。

BSND 格式（`packed_seq_params=None`）需要单独分析，Phase 1 以 THD 为主。

### 5.5 Skip 层的跨 Patch 协调（Layer 1 专用）

Layer 1 中 ratio=4 层跳过共享。**attention patch 和 transformer patch 必须协调**：

- Attention patch：检测到 `self.compress_ratio == 4 and self.indexer is not None` 时，走 `original_forward`（不 fork，不存 kv/attn_o）。不设标记——标记方案有时序问题（attention patch 的 `finally` 在 `self.self_attention()` 返回时即执行，而 transformer patch 后续的 expand 逻辑还没跑）。
- Transformer patch：自行判断——检查 `self.self_attention.compress_ratio == 4 and self.self_attention.indexer is not None`，若为真则跳过 residual/post/comb 的 store 和 expand。**不依赖 attention patch 的标记**，直接从 attention module 的属性读取。

同一层内所有序列共享同一个 attention module，判断对整层生效。

**⚠️ Recompute 风险**：Megatron 的 Recompute 机制可能在以下两个层面影响 prefix sharing：

1. **MHC Recompute**（`mhc_recompute`）：通过 `RecomputeInputWrap` / `RecomputeOutputWrap` 包裹 `_forward_attention` 的输入/输出。在此路径下 `_forward_attention` 只被调用一次，`_is_skip_layer()` 的属性检查在整个 `forward()` 期间结果不变。

2. **Megatron Activation Recompute**：可能重跑整个 `TransformerLayer.forward()`，包括 `_forward_attention`。此时：
   - Transformer patch 的 residual store 会重复执行，但 store 使用 merge 路径 + `overwrite=True`，重复 store 不会产生错误结果（幂等写入）。
   - Skip 层判断采用**属性检查**（`_is_skip_layer` 直接读 `self.self_attention.compress_ratio/indexer`），天然无状态——每次 forward 实时读取属性，不依赖 transient flag，Recompute 场景下自动正确。
   - 如果 Recompute 导致 `hidden_states` 被重新计算，provider 存储的 residual/post/comb 可能与第一次 forward 不同（autograd 内部状态差异），但这是 Recompute 本身的语义——recompute 前后的 tensor 数值应一致。

Phase 1 按此假设实现，Recompute 场景通过集成测试验证。

**为什么不会跨层串扰**：store key 包含 `layer_id`，不同层的数据存储在不同的 key 下。skip 层（如 layer 3，ratio=4）不会存 kv/attn_o，下一层（layer 4，ratio=128）reuser 取数据时用的是 `layer_id=4` 的 key，不会读到 layer 3 的数据。

### 5.6 验证用例（功能组 B — Attention 注入）

本组验证 KV 扩展、indices 调整、skip 层行为的正确性。

| 测试 | 内容 | 预期 |
|------|------|------|
| `test_kv_expansion_equivalence` | Provider 完整计算 vs Reuser KV 拼接（ratio=128） | attention 输出一致（误差 < 1e-6） |
| `test_cmp_kv_expansion` | CMP KV 扩展后长度正确 | kv_compress cat 后 dim=0 = sum((P+S)//ratio) |
| `test_topk_adjust_static` | ratio=128 静态路径：expanded seqlen 重新生成 indices | 调整后 indices 指向正确的 expanded cmp KV 位置 |
| `test_topk_adjust_start_pos` | 续训场景 start_pos > 0 时 indices 调整正确 | 同 `test_topk_adjust_static`，start_pos 透传 |
| `test_cu_seqlens_adjust` | 扩展后 cu_seqlens_kv 反映 reuser prefix_len 偏移 | cu_seqlens_kv[i] = 原值 + sum(prefix_lens[:i]) |
| `test_skip_layer_attention` | ratio=4 + indexer 层走 original_forward | fork 逻辑不执行，输出与 baseline 一致 |
| `test_skip_layer_coordination` | ratio=4 层 transformer patch 也跳过 store/expand | `_is_skip_layer` 返回 True，store/expand 不执行 |
| `test_transitive_reuse` | Reuser A 的 expanded KV 被 Reuser B 复用 | Reuser B 的 attention 输出与直接扩展一致 |
| `test_second_rope_symmetry` | 2nd RoPE 在扩展前执行，encode/decode 位置对称 | 扩展后 o 与 baseline full-sequence o 一致 |

## 6. 功能组 C：Attention 注入 — ratio=4 DSA Indexer（差异 6）

> **差异回顾**：ratio=4 层使用 DSA Indexer 生成 `compress_topk_idxs`（基于 hidden states，非纯位置），无法用 expanded seqlen 重算。采用三层渐进开发（见 6.1-6.4），每层可独立验证。
### 6.1 与 ratio=128 的关键差异

| | ratio=128 静态 | ratio=4 DSA Indexer |
|---|---|---|
| topk 生成 | `self.get_compress_topk_idxs()` 纯位置 | `self.indexer.forward_with_scores_compress()` 基于 hidden states |
| 可重算 | ✅ 用 expanded seqlen | ❌ 依赖 suffix hidden states |
| 额外产物 | 无 | `compress_topk_score`（相关性分数） |
| attention 路径 | `self.sparse_attention()` | `self.sparse_attention_with_indexer_loss()` 或 `self.sparse_attention()` + DSA loss |
| topk shape | `[b, s, s//ratio]` | `[b, s, index_topk=512]`（固定 topk，非等比） |

### 6.2 Layer 1: Fork 覆盖（Phase 1a — 当前实现）

**目标**：ratio=4 层在 fork forward 中正确执行，但 prefix sharing 逻辑完全跳过。

- Attention patch：`_is_skip_layer()` 检测到 ratio=4 + indexer → 走 `original_forward`，不存不扩展
- Transformer patch：`_is_skip_layer()` 返回 True → 跳过 residual store/expand
- Fork forward 代码中**保留完整 ratio=4 路径**（indexer 调用 + `sparse_attention_with_indexer_loss` + DSA loss），确保未来 Layer 3 可直接启用

**StoredG2Activation 预留字段**（为 Layer 2 准备）：

```python
@dataclass(frozen=True)
class StoredG2Activation:
    ...
    indexer_score: Any | None = None  # [prefix_len//ratio, index_topk] — DSA Indexer 相关性分数
```

Layer 1 中此字段始终为 None。

**验证**：

| 测试 | 内容 |
|------|------|
| `test_ratio4_baseline_parity` | ratio=4 层 fork 覆盖下输出与 baseline 一致 |
| `test_ratio4_skip_coordination` | attention + transformer 双 patch 均跳过 store/expand |
| `test_ratio4_no_interference` | ratio=4 skip 不影响相邻 ratio=128 层的共享 |

### 6.3 Layer 2: Provider 侧存储（Phase 1b）

**目标**：Provider 在 ratio=4 层存储 kv、kv_compress、compress_topk_score。Reuser 仍独立计算。

在 fork forward 中，ratio=4 层不再走 `original_forward`，而是走 fork 路径但**只执行 provider store，不执行 reuser expand**：

```python
# 在 patched_forward 中，ratio=4 层的处理
if self.compress_ratio == 4 and self.indexer is not None:
    # 执行完整 fork（含 indexer 调用），获取 compress_topk_score
    ...  # Phase 1-3: Q/KV/CMP KV/indexer（同 ratio=128）

    # Hook A/B: Provider 存 kv/kv_compress（同 ratio=128）
    _g2_store_per_sequence(..., field="kv", tensor=kv)
    _g2_store_per_sequence(..., field="kv_compress", tensor=kv_compress)

    # 新增：Provider 存 indexer_score
    _g2_store_per_sequence(..., field="indexer_score", tensor=compress_topk_score)

    # Hook C: Reuser 不扩展（Layer 2 不启用）
    # Hook D: Reuser 不扩展

    # Phase 4-5: sparse_attention + RoPE + output（同 ratio=128）
    ...
```

**注意**：ratio=4 层 Reuser 不走 expand，因此 KV 仍是 suffix-only。`sparse_attention` 以 suffix-only KV 调用，结果与 baseline 一致。Provider 的 store 为 Layer 3 做准备。

**验证**：

| 测试 | 内容 |
|------|------|
| `test_ratio4_provider_store` | Provider 存储的 kv/kv_compress/score 完整正确 |
| `test_ratio4_reuser_baseline` | Reuser 仍独立计算，输出与 baseline 一致 |

### 6.4 Layer 3: Reuser 侧 Index 调整 + 扩展（Phase 2）

**目标**：Reuser 利用 provider 的 `compress_topk_score` 调整 indices，实现完整的 KV/CMP KV 扩展。

**Index 调整策略**：

```python
def _adjust_topk_indices_indexer(compress_topk_idxs, compress_ratio,
                                  prefix_len, provider_score):
    """
    compress_topk_idxs: [b, suffix_len, index_topk=512] — reuser 的 indices
    provider_score:     [b, prefix_len//ratio, index_topk] — provider 的 DSA 分数

    策略（两阶段）：
    1. Offset: reuser 自己的 indices 整体偏移 cmp_offset = prefix_len // ratio
    2. 补充:   将 provider prefix cmp KV 中分数最高的 k 个位置，
               插入到 adjusted 中值为 -1（mask）的位置
    """
    cmp_offset = prefix_len // compress_ratio

    # Step 1: offset reuser's own indices
    adjusted = torch.where(
        compress_topk_idxs >= 0,
        compress_topk_idxs + cmp_offset,
        compress_topk_idxs)  # [b, suffix_len, index_topk]

    # Step 2: 从 provider score 中为每个 suffix 位置选取 prefix cmp KV 的 top-k
    # provider_score: [b, prefix_len//ratio, index_topk]
    # 沿 cmp_kv 维度取 mean（或 max），得到每个 cmp KV 位置的整体相关性
    prefix_relevance = provider_score.mean(dim=-1)  # [b, prefix_len//ratio]
    _, top_prefix_idx = torch.topk(prefix_relevance, k=min(K, cmp_offset), dim=-1)
    # top_prefix_idx: [b, K] — prefix cmp KV 中最相关的 K 个位置

    # 将 top_prefix_idx 插入到 adjusted 中值为 -1 的位置
    for batch_idx in range(adjusted.shape[0]):
        mask_positions = (adjusted[batch_idx] == -1).nonzero(as_tuple=True)
        num_to_fill = min(len(mask_positions[0]), K)
        # 在 mask 位置中填入 top prefix cmp KV 位置
        ...

    return adjusted
```

**待研究**：Step 2 的最佳策略（top-k 选取方式、K 值选择、分数聚合方式）需要通过精度验证确定。可能的简化方案是只用 Step 1（offset），不补充 provider prefix cmp KV 位置——如果 `index_topk=512` 足够大，reuser 自己的 indices + offset 可能已覆盖重要位置。

**注意**：`index_topk` 是固定的 512，不随 seqlen 变化（与 ratio=128 的 `s//ratio` 不同）。这意味着即使不补充 provider 位置，offset 后的 indices 仍然指向有效的 cmp KV 范围。

**验证**：

| 测试 | 内容 |
|------|------|
| `test_ratio4_index_adjust_offset` | Step 1 offset 后 indices 指向正确的 expanded cmp KV 范围 |
| `test_ratio4_expand_equivalence` | KV/CMP KV 扩展后 attention 输出与 baseline 一致 |
| `test_ratio4_indexer_loss` | 扩展后 DSA indexer loss 计算正确（如有） |

### 6.5 验证用例（功能组 C — ratio=4 DSA Indexer）

本组验证逐层叠加，每层通过后再进入下一层。

| 层 | 测试 | 内容 | 预期 |
|----|------|------|------|
| 1 | `test_ratio4_baseline_parity` | Layer 1 skip 下 ratio=4 层输出与 baseline 一致 | 误差 < 1e-6 |
| 1 | `test_ratio4_no_interference` | ratio=4 skip 不影响相邻 ratio=128 层的共享 | ratio=128 层共享正确 |
| 2 | `test_ratio4_provider_store` | Provider kv/kv_compress/indexer_score 正确存入 store | load 返回完整数据 |
| 2 | `test_ratio4_reuser_baseline` | Layer 2 下 Reuser 仍独立计算，输出不变 | 与 baseline 一致 |
| 3 | `test_ratio4_index_offset` | Step 1 offset 后 indices 指向正确 expanded cmp KV 范围 | 所有有效值 < (P+S)//4 |
| 3 | `test_ratio4_expand_equivalence` | KV/CMP KV 扩展后 attention 输出与 baseline 一致 | 误差 < 1e-6 |
| 3 | `test_ratio4_transitive_reuse` | ratio=4 层的 transitive reuse 正确 | 深度 reuser 输出一致 |

## 7. 功能组 D：Transformer 注入（差异 5）

> **差异回顾**：Attention 返回 `[P+S, ...]` 但 residual/post/comb 是 `[suffix, ...]`。需在 Attention 后扩展。

**设计组件**：TransformerLayer._forward_attention patch + per-sequence store/expand（3.4.1）。

### 7.1 TransformerLayer Patch：Residual/Post/Comb 扩展

> **Recompute 兼容性**：MHC Recompute 通过 `RecomputeInputWrap` / `RecomputeOutputWrap` 包裹，`_forward_attention` 只被调用一次。Megatron Activation Recompute 可能重跑整个 `TransformerLayer.forward()` ——此时 residual store 会重复执行，但 merge 路径 + `overwrite=True` 保证幂等写入，不产生错误结果。Phase 1 通过集成测试验证。

**关键认知**：MHC Pre 是 **per-position** 操作（Sinkhorn 在每个位置独立做 `[4,4]` 归一化），suffix-only 输入直接跑，结果等价。

**真正的问题**：THD packed 格式下 `_forward_attention` 每层只调用一次，`hidden_states` 是 `[total_tokens, b, ...]`，所有序列混在一起。不存在"当前是第几个 batch index"。Attention 返回 `[P+S, ...]` 但 residual/post/comb 仍是 `[suffix, ...]`。

**方案**：与 attention patch 同样的 per-sequence 模式——按 cu_seqlens 拆分 → 遍历所有 batch index → 对 provider 存、对 reuser 扩展 → 拼回。

```
Reuser _forward_attention (per-sequence view):
  hidden [suffix_len, b, 4, 4096]
    │
    ▼ MHC Pre (per-position, suffix-only 直接跑 ✓)
  residual = hidden   ← [suffix_len, b, 4, 4096]
  hidden, post, comb  ← [suffix_len, ...]
    │
    ▼ LayerNorm (suffix-only)
    │
    ▼ SelfAttention (内部 KV 扩展到 P+S)
  attn_output  ← [P+S, b, 4096]        ← 比 residual 长！
    │
    ▼ 扩展 residual/post/comb (从 provider G2AttentionStore)
  residual ← cat([provider_residual[:P], residual]) → [P+S, b, 4, 4096]
  post     ← cat([provider_post[:P], post])         → [P+S, b, 4]
  comb     ← cat([provider_comb[:P], comb])         → [P+S, b, 4, 4]
    │
    ▼ self_attn_bda(attn_output, residual)          ← shape 匹配 ✓
    │
    ▼ MHC Post(residual, post, comb)                ← 全部 full length ✓
  output [P+S, b, 4, 4096]
```

```python
# setup/patches/mindspeed_deepseek4/transformer.py

def patch_g2_transformer_attention(original_forward_attention):
    """Patch TransformerLayer._forward_attention。

    THD packed 格式下每层只调用一次，所有序列按 cu_seqlens 拆分后逐行处理。

    rotary_pos_emb 透传：TransformerLayer 直接将 rotary_pos_emb 作为 kwarg
    传入 self.self_attention()，attention forward 内部自行解包（基于实际代码确认）。
    Transformer patch 只做透明透传，不修改格式。
    """

    def patched_forward_attention(self, hidden_states, attention_mask, context=None,
                                   context_mask=None, rotary_pos_emb=None,
                                   rotary_pos_cos=None, rotary_pos_sin=None,
                                   attention_bias=None, inference_context=None,
                                   packed_seq_params=None, sequence_len_offset=None,
                                   input_ids=None, *,
                                   inference_params=None, recompute_info=None):
        ctx = current_prefix_sharing_context()
        is_g2 = ctx is not None and isinstance(ctx.store, G2AttentionStore)

        # ── MHC Pre (packed 直接跑, per-position 对所有序列正确) ──
        residual = hidden_states
        post, comb = None, None
        hidden_states = self.attn_mhc(
            hidden_states, mhc_stage='pre',
            recompute_info=recompute_info, module='attention')
        if isinstance(hidden_states, tuple):
            hidden_states, post, comb = hidden_states[0], hidden_states[1], hidden_states[2]

        # ── 逐序列存 provider residual/post/comb ──
        if is_g2 and ctx.plan.has_sharing and not _is_skip_layer(self):
            _g2_store_transformer_data(
                ctx, self.layer_number, residual, post, comb)

        # ── Input LayerNorm ──
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states)
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # ── Self Attention ──
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        if recompute_info:
            recompute_info.attention_output_with_bias = attention_output_with_bias[0]

        # ── 逐序列扩展 reuser residual/post/comb → 拼回 packed ──
        if is_g2 and ctx.plan.has_sharing and not _is_skip_layer(self):
            residual, post, comb = _g2_expand_transformer_data(
                ctx, self.layer_number, residual, post, comb)

        # ── self_attn_bda (shape 已匹配) ──
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0])
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout)

        # ── MHC Post (residual/post/comb 已扩展) ──
        hidden_states = self.attn_mhc(
            hidden_states, mhc_stage='post',
            residual=residual, post=post, comb=comb,
            recompute_info=recompute_info, module='attention')

        return hidden_states

    return patched_forward_attention
```

**辅助函数**（`prefix-sharing/prefix_sharing/integrations/g2_transformer.py`）：

```python
import torch
from prefix_sharing.core.prefix_store import (
    PrefixActivationSlotId, PREFIX_STATE_TYPE_G2_ATTENTION,
    StoredG2Activation, G2AttentionStore)
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.backends.g2_attention_utils import (
    _split_by_cu_seqlens, _merge_g2_transformer_fields)
from prefix_sharing.integrations.g2_attention import _g2_store_with_kwargs


def _is_skip_layer(transformer_layer):
    """判断当前层是否为 ratio=4 + DSA Indexer skip 层。

    直接从 attention module 的属性读取，不依赖 transient flag。
    """
    attn = transformer_layer.self_attention
    return attn.compress_ratio == 4 and attn.indexer is not None


def _g2_store_transformer_data(ctx, layer_id, residual, post, comb):
    """拆分 packed → 遍历 provider → 存 residual/post/comb prefix。

    使用 merge 路径（先 load existing → merge residual fields → store back），
    避免 overwrite=True + kv=None 在时序变化时静默覆盖 attention patch 已存的数据。
    """
    layout = ctx.packed_batch_layout
    plan = ctx.plan
    tp_rank = ctx.parallel_info.tp_rank

    r_rows = _split_by_cu_seqlens(residual, layout.padded_lengths)
    p_rows = _split_by_cu_seqlens(post, layout.padded_lengths) if post is not None else []
    c_rows = _split_by_cu_seqlens(comb, layout.padded_lengths) if comb is not None else []

    for batch_idx in range(layout.batch_size):
        if not plan.is_provider(batch_idx):
            continue
        valid_len = layout.valid_lengths[batch_idx]
        slot_id = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id,
            batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
        # merge 路径：保留已有 kv/kv_compress/attn_o，只更新 residual 字段
        existing = ctx.store.load(slot_id) if ctx.store.contains(slot_id) else None
        merged = _merge_g2_transformer_fields(
            existing,
            residual_prefix=r_rows[batch_idx][:valid_len] if r_rows else None,
            post_prefix=p_rows[batch_idx][:valid_len] if p_rows else None,
            comb_prefix=c_rows[batch_idx][:valid_len] if c_rows else None,
            valid_len=valid_len,
        )
        _g2_store_with_kwargs(ctx.store, slot_id, merged)


def _g2_expand_transformer_data(ctx, layer_id, residual, post, comb):
    """拆分 packed → 遍历 reuser → 拼接 provider prefix → 拼回 packed。"""
    layout = ctx.packed_batch_layout
    plan = ctx.plan
    tp_rank = ctx.parallel_info.tp_rank

    r_rows = _split_by_cu_seqlens(residual, layout.padded_lengths)
    p_rows = _split_by_cu_seqlens(post, layout.padded_lengths) if post is not None else []
    c_rows = _split_by_cu_seqlens(comb, layout.padded_lengths) if comb is not None else []

    exp_r, exp_p, exp_c = [], [], []
    for batch_idx in range(layout.batch_size):
        valid_len = layout.valid_lengths[batch_idx]
        if not plan.is_reuser(batch_idx):
            exp_r.append(r_rows[batch_idx][:valid_len])
            if p_rows: exp_p.append(p_rows[batch_idx][:valid_len])
            if c_rows: exp_c.append(c_rows[batch_idx][:valid_len])
            continue

        provider_idx = plan.provider_index[batch_idx]
        prefix_len = plan.prefix_lens[batch_idx]
        slot_id = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id,
            provider_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
        provider_data = ctx.store.load(slot_id)

        exp_r.append(torch.cat(
            [provider_data.residual_prefix[:prefix_len],
             r_rows[batch_idx][:valid_len]], dim=0))
        if p_rows:
            exp_p.append(torch.cat(
                [provider_data.post_prefix[:prefix_len],
                 p_rows[batch_idx][:valid_len]], dim=0))
        if c_rows:
            exp_c.append(torch.cat(
                [provider_data.comb_prefix[:prefix_len],
                 c_rows[batch_idx][:valid_len]], dim=0))

    new_residual = torch.cat(exp_r, dim=0)
    new_post = torch.cat(exp_p, dim=0) if exp_p else post
    new_comb = torch.cat(exp_c, dim=0) if exp_c else comb
    return new_residual, new_post, new_comb
```

### 7.2 验证用例（功能组 D — Transformer 注入）

| 测试 | 内容 | 预期 |
|------|------|------|
| `test_residual_expansion` | Reuser residual 扩展后 MHC Post 输出与 baseline 一致 | 误差 < 1e-6 |
| `test_post_comb_expansion` | post/comb 同步扩展后 MHC Post 正确 | 同上 |
| `test_mhc_disabled` | MHC 不启用时 residual/post/comb 为 None，扩展跳过 | 无异常，输出与 baseline 一致 |
| `test_transformer_store_provider` | Provider 的 residual/post/comb 正确存入 store | load 返回完整数据 |

## 8. 功能组 E：训练流程集成（全局）

> **定位**：A-D 组解决模型 forward **内部**的问题（store、attention KV 扩展、residual 扩展）。E 组解决训练循环**外部**的问题——把 prefix sharing 接入 MindSpeed 的 pretrain 入口，让整个流程跑起来。
>
> 具体包括：
> 1. **Batch 裁剪**：在 `forward_step` 调用前检测前缀、裁剪 batch（复用 PrefixSharing 现有 planner + batch trim 流程）
> 2. **Context 注入**：用 `prefix_sharing_runtime_context` 包裹模型 forward，使 A-D 组的 patch 能读取 runtime state
> 3. **MindSpeed 入口适配**：`forward_step` 是函数引用沿 `pretrain()` → `train()` → `train_step()` 传递，无法用 PatchRegistry 自动拦截，需 `wrap_forward_step()` 显式包装
> 4. **Prefix-Last Restore**：直接复用 `restore_reuser_prefix_columns_2d()`（与 attention 类型无关）
> 5. **并行约束**：Phase 1 仅支持 CP=1

核心改动量为用户侧 **1 行**：`pretrain(..., wrap_forward_step(forward_step, ps_config))`。

### 8.1 训练入口集成：wrap_forward_step()

Batch 裁剪、context 注入、prefix-last restore 均在此 wrapper 中完成。

**复用 PrefixSharing 现有流程**：前缀检测 → 计划 → 裁剪 batch（Provider 保留全部，Reuser 只保留 suffix）→ 构建 runtime state → 在 context 内执行 forward → prefix-last restore。

裁剪细节：

```
原始:
  seq_0 (provider): [A, B, C, D, E, F]  len=6
  seq_1 (reuser):   [A, B, C, G, H]     len=5, prefix_len=3

裁剪后:
  seq_0 (provider): [A, B, C, D, E, F]  len=6  (不变)
  seq_1 (reuser):   [G, H]              len=2  (只保留 suffix)
```

`position_ids`：reuser 的 suffix tokens 保持原始 position_ids（从 3 开始），这对 RoPE 正确性至关重要。

**MindSpeed 训练入口适配**：

MindSpeed 的 `pretrain_deepseek4.py:forward_step` 签名与 verl 不同——是直接的训练脚本入口而非 Engine 方法：

```python
# pretrain_deepseek4.py:217
def forward_step(data_iterator, model: DeepSeek4Model):
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels, loss_mask=loss_mask)
    return output_tensor, partial(loss_func, loss_mask)
```

**调用链确认**（基于实际代码）：

```
pretrain_deepseek4.main()
  → pretrain(..., forward_step)                        # mindspeed_llm.training.training:430
    → train(..., forward_step_func, ...)               # :581
      → train_step(forward_step_func, ...)             # megatron.training.training (Megatron)
        → forward_step_func(data_iterator, model)      # 逐 micro-batch 调用
```

`forward_step` 作为函数引用沿调用链传递，无法用 PatchRegistry 自动拦截（不像类方法可通过 import hook 在模块加载时替换）。

**方案**：提供 `wrap_forward_step()` 工具函数，用户在 `main()` 中 wrap 后传入：

```python
# pretrain_deepseek4.py main() — 用户侧改动（1 行）
from prefix_sharing.integrations.g2_batch import wrap_forward_step

def main():
    ps_config = PrefixSharingConfig.from_raw(args.prefix_sharing_config)
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             wrap_forward_step(forward_step, ps_config))  # ← 改动点
```

`wrap_forward_step()` 的逻辑（`integrations/g2_batch.py`）：
1. 包装原始 `forward_step`：前缀检测 → 计划 → 裁剪 batch → 构建 runtime state
2. 在 `prefix_sharing_runtime_context` 内执行原始 forward
3. prefix-last restore

此方式对独立训练脚本最干净——无需 patch 训练基础设施。Attention 和 TransformerLayer 的 patch 仍走 PatchRegistry 自动安装。

**Prefix-Last Restore**：直接复用 PrefixSharing 现有的 `restore_reuser_prefix_columns_2d()` 和 `restore_via_2d_unfold_verl080()`。Restore 逻辑与 attention 类型无关，在 `wrap_forward_step()` 的 post-forward 阶段调用。

**并行约束（Phase 1）**：CP=1。`gather_from_sp_cp(kv)` 在 CP=1 时是 no-op。CP>1 需后续扩展。

**⚠️ 边界处理**：

| 场景 | 处理 |
|------|------|
| 全 batch 无共享前缀 | 跳过所有逻辑，返回原始 batch |
| Reuser suffix_len=0 | Prefix-last restore 不执行，直接复制 provider logprobs |
| 续训（start_pos > 0） | start_pos 从 fork forward 入参透传到所有内部调用 |

### 8.2 验证用例（功能组 E — 训练流程集成）

本组需要 MindSpeed 运行时环境（NPU + DeepSeek4 模型）。

| 测试 | 内容 | 预期 |
|------|------|------|
| `test_e2e_loss_parity` | ENABLE_PREFIX_SHARING=0 vs =1，相同输入 | loss 完全一致（误差 < 1e-6） |
| `test_e2e_logprob_parity` | 逐 token logprob 对比 | 完全一致 |
| `test_e2e_gradient_flow` | Provider KV 的梯度通过 reuser 正确回传 | grad norm 一致 |
| `test_prefix_last_restore` | Reuser prefix-last logprob 与独立计算一致 | 误差 < 1e-6 |
| `test_wrap_forward_step` | `wrap_forward_step` 在无共享前缀时返回原始结果 | 与 baseline 一致 |
| `test_mixed_ratios` | 同一 batch 包含 ratio=0,4,128 层 | skip 层正确跳过，非 skip 层正确共享 |


## 9. 场景覆盖

### 9.1 并行策略

| 组合 | Phase 1 | 说明 |
|------|---------|------|
| TP=2, PP=1, EP=128, CP=1 | ✅ | 主要验证目标 |
| TP=1, PP=1, EP=8, CP=1 | ✅ | 小规模测试 |
| PP>1, VPP=0 | ⚠️ | 需验证 P2P 通信 |
| VPP>0 | ❌ | 不支持 |
| CP>1 | ❌ | 不支持 |

### 9.2 层类型

| compress_ratio | Indexer | Phase 1 |
|----------------|---------|---------|
| ≤1（前 2 层等价） | 无 | ✅ 完整支持 |
| 128（大部分层） | 无 (static) | ✅ 完整支持 |
| 4（少量层） | DSA Indexer | ⏭ 跳过（attention + transformer 双 patch 协调） |

### 9.3 MHC / MTP

| 状态 | Phase 1 |
|------|---------|
| MHC 启用 | ✅ 扩展 residual/post/comb |
| MHC 不启用 | ✅ residual/post/comb 为 None，跳过 |
| MTP 不启用 | ✅ Phase 1 目标 |
| MTP 启用 | ⏭ Phase 2 |


## 10. 文件清单与改动量估算

### 10.1 新增文件（均在 `prefix-sharing/prefix_sharing/` 下）

| 文件 | 行数 | 说明 |
|------|------|------|
| `backends/g2_attention_utils.py` | ~150 | `_split_by_cu_seqlens()`, `_merge_g2_fields()`, `_merge_g2_transformer_fields()`, `_compute_cmp_lengths()`, `_adjust_topk_indices_for_batch()`, `_adjust_cu_seqlens_for_batch()` |
| `integrations/g2_attention.py` | ~200 | `_g2_store_per_sequence()`, `_g2_expand_kv_and_adjust()`, `_g2_expand_attn_output()`, `_g2_store_with_kwargs()` |
| `integrations/g2_transformer.py` | ~120 | `_g2_store_transformer_data()`, `_g2_expand_transformer_data()` |
| `integrations/g2_batch.py` | ~100 | `wrap_forward_step()`：batch 准备 + context + restore |
| `setup/patches/mindspeed_deepseek4/__init__.py` | ~45 | PATCH_SET + compat_matrix 条目 |
| `setup/patches/mindspeed_deepseek4/attention.py` | ~15 | PatchSpec → `DeepSeek4SelfAttention.forward`；实现代码在 integrations |
| `setup/patches/mindspeed_deepseek4/transformer.py` | ~15 | PatchSpec → `TransformerLayer._forward_attention`；实现代码在 integrations |

**注意**：`forward_step` 不通过 PatchRegistry 安装，而是通过 `wrap_forward_step()` 工具函数在 `main()` 中显式包装（见第 3.6 节）。

**小计：~680 行**

### 10.2 修改文件（均在 `prefix-sharing/prefix_sharing/` 下）

| 文件 | 行数 | 修改内容 |
|------|------|----------|
| `core/config.py` | ~10 | `model_type` 支持 `"deepseek4"`；validate() 改为白名单模式 |
| `core/prefix_store.py` | ~60 | 新增 `StoredG2Activation`, `G2AttentionStore`, `PREFIX_STATE_TYPE_G2_ATTENTION` |
| `integrations/context.py` | ~15 | `PrefixSharingRuntimeContext.store` 类型 → base class；新增 `_create_store()`；`prefix_sharing_runtime_context()` 使用工厂 |
| `integrations/verl_mcore.py` | ~5 | `PrefixSharingRuntimeState` 新增 `model_type` 字段 |
| `setup/compat_matrix.py` | ~5 | 新增 MindSpeed DeepSeek4 版本条目 |

**小计：~95 行**

### 10.3 PatchSpec 定义

```python
# setup/patches/mindspeed_deepseek4/__init__.py

from prefix_sharing.setup.registry import PatchSpec
from .attention import patch_g2_attention
from .transformer import patch_g2_transformer_attention

PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention",
        target_getter=lambda mod: (
            getattr(mod, "DeepSeek4SelfAttention"), "forward"),
        patch_factory=patch_g2_attention,
        description="DeepSeek4SelfAttention.forward → fork forward + 4 Hook",
    ),
    PatchSpec(
        module_name="mindspeed_llm.core.transformer.transformer_layer",
        target_getter=lambda mod: (
            getattr(mod, "TransformerLayer"), "_forward_attention"),
        patch_factory=patch_g2_transformer_attention,
        description="TransformerLayer._forward_attention → residual/post/comb 扩展",
    ),
]
```

### 10.4 compat_matrix 条目

```python
CompatEntry(
    verl=None,
    megatron_core="0.16.1",
    mindspeed="2.2.0",  # 需确认实际版本
    patch_set_id="mindspeed_deepseek4",
    notes="DeepSeek V4 standalone pretrain (non-verl path)",
)
```

### 10.5 依赖关系

所有新代码在 PrefixSharing 项目内部。MindSpeed 仅作为运行时 import 目标，**不修改 MindSpeed 源码**。

## 11. 验证方案

### 11.1 单元测试

| 测试 | 内容 |
|------|------|
| `test_g2_store.py` | G2AttentionStore 的 store/load/overwrite/close 生命周期；`_merge_g2_fields` |
| `test_g2_kv_expansion.py` | Provider 完整计算 vs Reuser KV 拼接，attention 输出一致性 |
| `test_g2_topk_adjust.py` | compress_topk_idxs 调整正确性（ratio=128），含续训 start_pos>0 场景 |
| `test_g2_residual_expansion.py` | MHC residual/post/comb 扩展后 MHC Post 输出一致性 |
| `test_g2_prefix_last_restore.py` | Reuser prefix-last logprob 与独立计算一致 |
| `test_g2_gradient_flow.py` | Provider KV/attn_o/residual 的梯度通过 reuser 正确回传（含 transitive reuse） |
| `test_g2_skip_layer.py` | ratio=4 skip 层两个 patch 的协调行为 |

### 11.2 端到端测试

```bash
# 小规模：8卡，TP=1, PP=1, EP=1（关闭 MoE）
bash tests/poc/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh

# 对比 ENABLE_PREFIX_SHARING=0 vs =1，误差 < 1e-6
# 构造共享前缀数据：每个 micro-batch 前 4 条序列共享 75% 前缀
```

### 11.3 精度红线

| 指标 | 要求 |
|------|------|
| Loss | 与 baseline 完全一致 |
| Logprob (per token) | 与 baseline 完全一致 |
| Gradient norm | 与 baseline 完全一致 |
| 参数更新量 | 与 baseline 完全一致 |

## 12. 实施阶段

按功能组顺序逐个开发，每组完成开发并**通过全部验证用例**后再进入下一组。

```
A (数据存储) ──→ B (ratio=128) ──→ C (ratio=4) ──→ D (Transformer) ──→ E (训练入口)
   2天              3天               3天              2天                 2天
```

### 12.1 功能组 A：数据存储层（~2 天）

**依赖**：无

**开发**：
1. `core/prefix_store.py`：新增 `StoredG2Activation`、`G2AttentionStore`、`_merge_g2_fields`
2. `integrations/context.py`：`_create_store()` 工厂 + `PrefixSharingRuntimeContext` 签名放宽
3. `integrations/verl_mcore.py`：`PrefixSharingRuntimeState` 新增 `model_type`

**验证**（纯 PyTorch，无需 MindSpeed 运行时）：
- store 生命周期、增量存储、merge 正确性、`_create_store("deepseek4")` 返回 G2AttentionStore

**准出**：A 组全部验证用例（5 项）通过 ✓

### 12.2 功能组 B：Attention 注入 — ratio=128（~3 天）

**依赖**：A 组（需要 G2AttentionStore）

**开发**：
1. `backends/g2_attention_utils.py`：`_split_by_cu_seqlens`、`_compute_cmp_lengths`、`_adjust_topk_indices_for_batch`、`_adjust_cu_seqlens_for_batch`
2. `integrations/g2_attention.py`：`_g2_store_per_sequence`、`_g2_expand_kv_and_adjust`、`_g2_expand_attn_output`
3. `setup/patches/mindspeed_deepseek4/attention.py`：fork forward + PatchSpec
4. `setup/patches/mindspeed_deepseek4/__init__.py`：PATCH_SET + compat_matrix
5. Fork forward 代码中包含 ratio=4 路径，但当前通过 `_is_skip_layer()` 跳过（为 C 组预留）

**验证**：
- KV 扩展等价性、CMP KV 扩展、topk 调整（含续训 start_pos>0）、cu_seqlens 调整、skip 层行为、transitive reuse、2nd RoPE 位置对称

**准出**：B 组全部验证用例（9 项）通过 ✓，ratio≤1 和 ratio=128 层 prefix sharing 精度与 baseline 一致

### 12.3 功能组 C：Attention 注入 — ratio=4 DSA Indexer（~3 天）

**依赖**：A 组（store）+ B 组（fork forward + attention patch 基础设施）

**开发**：渐进三层叠加

| 层 | 内容 | 验证 |
|----|------|------|
| Layer 1 | Fork 覆盖确认：ratio=4 路径代码在 fork 中正确执行，`_is_skip_layer()` 跳过共享 | `test_ratio4_baseline_parity`、`test_ratio4_no_interference` |
| Layer 2 | `StoredG2Activation.indexer_score` 字段启用，provider 存储 kv/kv_compress/score | `test_ratio4_provider_store`、`test_ratio4_reuser_baseline` |
| Layer 3 | `_adjust_topk_indices_indexer`：offset + provider score 补充，reuser KV/CMP KV 扩展 | `test_ratio4_index_offset`、`test_ratio4_expand_equivalence`、`test_ratio4_transitive_reuse` |

**准出**：C 组全部验证用例（7 项）通过 ✓，ratio=4 层 prefix sharing 精度与 baseline 一致

### 12.4 功能组 D：Transformer 注入（~2 天）

**依赖**：A 组（store）+ B/C 组（attention patch 已安装，KV 扩展机制已验证）

**开发**：
1. `integrations/g2_transformer.py`：`_g2_store_transformer_data`、`_g2_expand_transformer_data`
2. `setup/patches/mindspeed_deepseek4/transformer.py`：`_is_skip_layer()` + PatchSpec

**验证**：
- residual/post/comb 扩展等价性、MHC 不启用路径、provider store 正确性

**准出**：D 组全部验证用例（4 项）通过 ✓，MHC Post 输出与 baseline 一致

### 12.5 功能组 E：训练流程集成（~2 天）

**依赖**：A-D 组全部就绪

**开发**：
1. `integrations/g2_batch.py`：`wrap_forward_step()` 实现
2. 端到端集成：MindSpeed pretrain 入口 + context 注入 + restore

**验证**（需 MindSpeed + NPU 运行时）：
- 端到端 loss/logprob/grad 精度、prefix-last restore、wrap_forward_step 无共享前缀回退、混合 ratio 层

**准出**：E 组全部验证用例（6 项）通过 ✓，端到端精度红线全部达标

### 12.6 后续（~2 周）

- MTP 层适配
- PP>1 适配
- Profiling 量化收益
- 向 MindSpeed 提 PR 加入方案 B 的显式 Hook 点，逐步替换方案 A 的 fork
