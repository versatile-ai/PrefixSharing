# CP + Prefix Sharing 联合设计方案

> **日期**：2026-07-24
> **基础方案**：`prefix_sharing_deepseek4_design_v2.md`（CP=1）
> **本文**：CP>1 时 PS 的适配方案

## 1. CP 与 PS 的关系

### 1.1 两者的共性

CP 和 PS 本质上做的是同一件事：**减少 MLP 计算。**

```
CP 的思路:
  把序列长度切到多个 rank → 每个 rank 只算一部分 Q → Q 少了 → MLP 少了
  代价: 需要跨 rank 重建 KV Cache → gather_from_sp_cp

PS 的思路:
  裁掉 reuser 的 prefix Q → Q 少了 → MLP 少了
  代价: 需要跨 sequence 重建 KV Cache → store/expand
```

两者异曲同工——都是在 Q 上做减法，在 KV 上加回来。叠加后效果叠加：

```
无 CP, 无 PS:  Q=25, MLP=25
有 PS, CP=1:   Q=18, MLP=18   (PS 裁了 7 个 prefix token)
有 PS, CP=2:   Q=9/卡, MLP=9/卡 (CP 切半)
```

Attention 的 Q×KV 计算总量不变（只是分布式并行），真正节省的是 MLP。

### 1.2 两者的互补

CP 和 PS 在 KV 补齐这个环节配合紧密：

```
CP 补齐跨 rank KV:  gather_from_sp_cp(kv_local) → kv_global
PS 补齐跨 seq KV:   expand(provider_prefix + reuser_suffix) → expanded_kv

两者都在 Phase 1-3 和 Phase 4(Attention) 之间完成。
gather 在前(Phase 1 末尾), expand 在后(Hook)。
sparse_attention 收到的是双重补齐后的 KV。
```

CP 的 gather 让跨 rank 访问 KV 的问题消失了。但这里有一个容易误解的地方需要明确：**Store 是 per-rank 的 Python 内存对象，不是跨 rank 共享的。** 那 rank 1 上的 reuser 怎么拿到 rank 0 上的 provider 数据？完整逻辑链：

```
CP 切分 packed tensor
  → 每个 rank 拿到 hidden 的一部分
  → 每个 rank 各自跑 forward（Phase 1-3）
  → gather_from_sp_cp → 所有 rank 都有全量 KV(18)
  → 每个 rank 各自执行 Hook
     → 每个 rank 各自 store provider 的 kv[:valid_len]
        rank 0: store(seq0=[a..h], seq2=[m..r])
        rank 1: store(seq0=[a..h], seq2=[m..r])     ← 和 rank 0 完全一样！
  → 每个 rank 各自 load provider 数据
     rank 1 上的 reuser 需要 seq0 的 prefix KV
     → load(seq0_slot) → [a..h]                     ← 从本地取，不跨 rank
```

**关键**：每个 rank 独立 store，但因为 gather 之后全量 KV 相同 + 全局 layout 相同，store 的内容完全相同。reuser load 时从本地取，不需要跨 rank 通信。**gather 让每个 rank 独立 store 的数据一致，load 自然能从本地取到。**

## 2. KV Cache 重建方案：基于 MindSpeed gather 原语

### 2.1 当前 CP 机制

MindSpeed 使用 `gather_from_sp_cp` 做 CP KV 重建：

```
Phase 1: linear_kv(hidden_local) → kv_local [local_chunk, b, 512]
         gather_from_sp_cp(kv_local) → kv_global [total_tokens, b, 512]
```

每个 rank 都持有一份全量 KV。内存 = CP_size 倍，实现最简单。

### 2.2 PS 适配：在 gather 之后 Hook

```
Phase 1:  kv_local → gather → kv_global[18]
Phase 2:  compress_topk_idxs (可能涉及 gather)
Phase 3:  compressor(hidden_local) → cmp_local → gather → cmp_global

    ╔══════ Hook: _g2_kv_store_or_expand ══════╗
    ║ Provider: store(kv_global, cmp_global)    ║
    ║ Reuser:   load → expand → [25]            ║
    ╚════════════════════════════════════════════╝

Phase 4:  sparse_attention(Q_local[9], KV_expanded[25], ...)
```

Hook 在 gather 之后——KV 全量已经就位。PS 的 store/expand 只做拼接，不引入新的跨 rank 通信。

### 2.3 关键改动：全局 layout

CP>1 时 `packed_seq_params` 中的 `padded_lengths` 是 CP-local 的——只描述本 rank 的 chunk。但 Hook 拆分的是 gather 后的全量 KV。

**必须用 trim 时确定的全局 layout 来拆分：**

```python
# trim 阶段(CP 切分之前)保存:
state = PrefixSharingRuntimeState(
    ...
    packed_batch_layout=PackedBatchLayout(
        padded_lengths=[8, 2, 6, 2],     # ← 全局
        valid_lengths=[8, 2, 6, 2],
        cu_seqlens=[0, 8, 10, 16, 18],
    ),
)

# Hook 阶段(每个 CP rank):
layout = ctx.packed_batch_layout          # 全局, 所有 rank 相同
_split_by_cu_seqlens(kv_global[18], layout.padded_lengths)
# → [seq0:8, seq1:2, seq2:6, seq3:2]  ✅
```

## 3. suffix 跨 rank 分布及处理

### 3.1 问题

CP 按 token 数切分 packed tensor，不感知序列边界。suffix 可能被切到不同 rank：

```
Reuser seq1 suffix [x,y]:
  Rank 0: [x]   ← suffix 第一半
  Rank 1: [y]   ← suffix 第二半
```

### 3.2 处理

**不需要特殊处理。** gather 后两个 rank 都是全量 KV。Hook 用全局 layout 拆分，两个 rank 都拿到完整的 suffix [x,y]。

```
Rank 0 用全局 layout:
  row_1 = [x,y]          ← 完整 suffix（虽然本 rank 只产生了 [x]）
  expand = cat(provider[:4], [x,y]) = [a,b,c,d, x,y]

Rank 1 用全局 layout:
  row_1 = [x,y]          ← 同上, gather 后全量
  expand = cat(provider[:4], [x,y]) = [a,b,c,d, x,y]

结果完全一致 ✅
```

### 3.3 Q 的处理

每个 rank 只算自己那部分 Q。CP-local 的 `cu_seqlens_q` 描述本 rank 的 Q chunk：

```
Rank 0: Q[9]  → cu_seqlens_q=[0, 8, 9]    ← seq0:8, seq1:1 ([x])
Rank 1: Q[9]  → cu_seqlens_q=[0, 1, 7, 9]  ← seq1:1 ([y]), seq2:6, seq3:2
```

Q 坐标系和 KV 坐标系独立。`sparse_attention` 接受独立的 `cu_seqlens_q` 和 `cu_seqlens_kv`。

## 4. Topk 计算——sparse attention 的正确性

### 4.1 topk 是 per-token 独立操作

`compress_topk_idxs` 的 shape 是 `[bsz, q_len, index_topk(512)]`——每个 query token 独立选自己的 top-512 个 CMP entry。不是全局操作。

```
ratio=4, 序列 [A..L] (12 tokens), 3 个 CMP entry [c0,c1,c2]

token A: score(A, c0) → topk=[0]
token E: score(E, c0), score(E, c1) → topk=[1,0]  
token I: score(I, c0), score(I, c1), score(I, c2) → topk=[2,1,0]
```

每个 token 对全量 CMP key space 独立打分。CP 把 Q 切成两块，两块各自算自己的 token——结果和全量 Q 一起算一致。

### 4.2 PS 的影响

PS 裁了 prefix 的 Q（不需要 prefix query 的 topk），但 CMP key space 是扩展后的全量。suffix Q 独立对完整 key space 打分：

```
正常: Q[0..11] × k[c0,c1,c2]     → topk for all 12 tokens
PS:   Q[8..11] × expanded_k[c0,c1,c2]  → topk only for suffix tokens
      ↑ suffix-only              ↑ 包含 provider 的 c0,c1
```

### 4.3 sparse_attention 的计算方式

不是标准 dense attention + mask。是 `sparse_flash_mla` 融合 kernel：

```
对每个 query token:
  1. 从 kv_compress 取出 topk 选中的 CMP entry
  2. 解压 CMP entry 为原始 token
  3. 加上 SWA sliding window 内的 raw KV
  4. Q × 这些 KV → attention score → output
```

**不物化全量 attention 矩阵。** PS 不改变 per-token attention 的计算模式——每个 suffix token 的稀疏 attention 计算方式和共享前相同。但 attention 总量因 Q 减少而降低（Q 从 25→18），见 §5.1 表格。

## 5. 系统效率分析

### 5.1 单卡视角（以示例 18 tokens, CP=2 为例）

> **注意**：Attn 列用 Q×KV 的 dense 等效值(Q_len × KV_len) 做定性对比。实际 `sparse_flash_mla` 是稀疏 kernel（topk + SWA），不物化全量矩阵，真实 FLOPs 远小于 dense 值。此处仅用于说明 CP/PS 的相对节省趋势。

```
                     Q/卡   KV/卡   Attn/卡(dense等效)  MLP/卡
无 PS, CP=1          25     25      625                 25
有 PS, CP=1          18     25      450                 18    ← PS 省
有 PS, CP=2           9     25      225                  9    ← CP 再省
```

从无 PS/CP=1 到有 PS/CP=2：Q↓64%, MLP↓64%, Attention 计算量↓64%。

**PS 额外内存开销 = 0**——expanded KV(25) 恰好等于原始全量(25)。PS 的 prefix 重复 + CP 的 gather 全量 = 无额外开销。

### 5.2 CP 内存代价

CP 的 gather 让每卡持有全量 KV(25)——这是 CP 本身的代价，不是 PS 引入的。`gather_from_sp_cp` 之后 KV 内存 = CP_size × 全量。

这个代价在所有 CP 路径都存在（无 PS 时也一样）。PS 不增加这个代价——因为 PS expand 后的 KV(25) 和原始全量(25) 一样大。

### 5.3 总结

CP + PS 叠加：
- Q 节省叠加：PS 裁 prefix + CP 切长度
- MLP 节省叠加：Q 省多少，MLP 省多少
- Attention 不变：只是分布式执行
- KV 内存：CP 的 gather 代价(CP_size×)，PS 不额外加
- 不引入新的跨 rank 通信

## 6. 实现改动清单

### 6.1 RuntimeState 增加全局 layout

```python
@dataclass(frozen=True)
class PrefixSharingRuntimeState:
    ...
    packed_batch_layout: PackedBatchLayout  # ← 已是全局面(trim 时确定, CP 切分前)
```

当前 `packed_batch_layout` 在 trim 时创建，此时 CP 尚未切分——已经是全局的。CP>1 时不变。

### 6.2 _g2_kv_store_or_expand 适配

```python
def _g2_kv_store_or_expand(ctx, kv, kv_compress, indexer_k, ...):
    layout = ctx.packed_batch_layout  # 全局, 所有 CP rank 一致
    
    # 用全局 padded_lengths 拆分全量 kv
    kv_rows = _split_by_cu_seqlens(kv, layout.padded_lengths)
    
    # provider/reuser 分支不变
    ...
    
    # cu_seqlens_kv 返回 expanded 全局版本
    return (result_kv, ..., compress_topk_idxs, packed_seq_params, ...)
```

### 6.3 cu_seqlens 适配

```python
# cu_seqlens_kv 用 expanded 全局
# cu_seqlens_q 用 CP-local (packed_seq_params 自带)
# 两者独立传入 sparse_attention，互不干扰
```

### 6.4 MegatronParallelInfo 增加 cp_rank

```python
@dataclass(frozen=True)
class MegatronParallelInfo:
    tp_rank: int = 0; tp_size: int = 1
    pp_rank: int = 0; pp_size: int = 1
    cp_rank: int = 0; cp_size: int = 1  # ← 新增
```

**cp_rank 不放入 SlotId**——gather 后全量 KV，所有 rank 存的一样，不需要区分。

### 6.5 不改的部分

- Store 结构不变——存的仍是 gather 后的全量 KV
- Hook 插入点不变——仍在 Phase 3 和 Phase 4 之间
- Topk 重算逻辑不变——ratio=128 纯位置，ratio=4 DSA 打分
- Prefix-last restore 不变

## 7. 延后事项

- CP + TP 组合验证（当前只考虑 CP 独立场景）
- CP + PP 组合验证
- 压缩边界 × CP 边界交叉（P%r≠0 时跨 rank 压缩块重建）
- Ring Attention 替代 gather（内存优化）
- **CP 对 ratio=4 Indexer 的影响（待确认）**：Phase 2 中 `forward_with_index_compress(hidden_states.detach(), ...)` 的 `hidden_states` 是 CP-local 的。`q_len_global = q_len * cp_size` 已经在 topk 参数中考虑了 CP 倍数，但 Indexer 的 `kv_compressor(hidden_local)` 只看到 CP-local hidden 就会只生成局部 k。需要确认 `all_gather_qk_weight_kvallgather` 是否在 gather 之后补全了 k——如果是，则 indexer_k 和 CP=1 一致；如果不是，则 CP>1 时 indexer_k 的内容和 shape 可能出问题。在实现 CP 之前需在容器内打印 `key_index.shape` 验证。
