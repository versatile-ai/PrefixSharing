# Prefix-Sharing 支持 DeepSeek V4 — 堵点梳理

> 日期：2026-07-13
> 目标：在标准 Megatron 引擎上，让 PS 支持 DeepSeek V4（CSA + HCA + MQA + SWA）

---

## 一、Batch 格式

### 1.1 当前 PS 只支持 THD

```python
# megatron_runtime.py:36
if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
    raise RuntimeError("prefix sharing requires packed_seq_params.qkv_format='thd'")
```

- THD 即 packed / varlen 格式，数据在 `preprocess_thd_engine` 中压平为 1D，用 `NestedTensor` 或 `cu_seqlens` 承载
- verl080 的 PS 路径内部大量依赖 NestedTensor 做物理裁剪（`_trim_nested_batch`）和 restore（`nested_to_2d` → 注入 → `fold_2d_to_nested`）

### 1.2 BSHD 不支持

BSHD 即标准 `[B, S, H, D]` padded 格式，靠 `attention_mask` 标记有效位。当前 PS 未实现 BSHD 路径：
- 无 `cu_seqlens`，无法做 packed 1D 坐标映射
- `build_kv` 和 `restore` 逻辑全基于 THD 假设
- 如果训练脚本走 BSHD，PS 入口直接崩溃

### 1.3 改造点

若需支持 BSHD，至少需：
1. 移除 `qkv_format == 'thd'` 硬断言
2. 重新实现 BSHD 下的物理裁剪（不能裁 NestedTensor，要裁 `attention_mask` 和 2D tensor）
3. `PackedBatchLayout` 全部替换为 2D 布局逻辑
4. restore 阶段从 `nested_to_2d` 改为直接在 2D padded 张量上注入

---

## 二、并行策略兼容性

| 并行 | 状态 | 说明 |
|------|------|------|
| **TP** | ✅ 已支持 | KV 注入/restore 已验证，dump 工具链支持 `_tp{r}` 分片 |
| **SP** | ✅ 已支持 | 序列维在 TP group 内切分，当前代码已处理 |
| **PP** | 🔴 有 bug | in-place restore 打断 autograd 图，反向卡死。修复方案：改为 out-of-place `torch.cat` 逐行重建 |
| **CP** | ❌ 待开发 | todo 尚未开发，PS 不支持 |
| **DP** | 🟡 负载不均 | 各 rank 裁剪后 token 数不同，导致 PPO mini-batch 同步等待 |
| **EP** | ✅ 无关 | EP 影响 MoE/FFN，PS 只操作 attention 层，路径不交叉 |

### 2.1 PP（Pipeline Parallel）— 已知 bug

- **根因**：`restore_reuser_prefix_columns_2d` 的 in-place 赋值打断 autograd 图
- **表现**：末 stage 反向报 `RuntimeError: modified by an inplace operation`；前序 stage 卡在 `recv_backward` 无限等待
- **修复**：in-place 赋值 → out-of-place `torch.cat` 逐行重建
- **注意**：TP 精度验证通常 `forward_only=True`，不触发反向，所以 bug 未暴露

### 2.2 DP（Data Parallel）— 负载均衡

- 不影响正确性，影响吞吐
- 不同 DP rank 的 batch 构成不同 → 裁剪后 token 数差异大
- 后续优化：基于前缀长度做 DP 分组负载均衡

### 2.3 CP（Context Parallel）— 未支持

- CP 对序列的切分方式与 SP 不同，PS 的 KV 注入/restore 需要额外适配
- 待 CP 代码路径稳定后再做 PS 适配

---

## 三、Attention 相关

### 3.1 DeepSeek V4 架构概览

| 维度 | 当前 PS（MLA） | DeepSeek V4 |
|------|---------------|-------------|
| 注意力类型 | Multi-head Latent Attention | **MQA** + CSA + HCA + Sliding Window |
| 压缩机制 | 低秩 KV 压缩 | 窗口级 KV 压缩（m=4） |
| 稀疏性 | 无 | **Lightning Indexer top-k 检索** |
| 位置编码 | 标准 RoPE | **部分 RoPE**（最后 64 维） |
| 残差连接 | 标准 | **mHC**（流型超连接） |
| 输出投影 | 标准 | **分组输出投影**（o_groups=8） |

![CSA 架构图](SWA+CSA.png)

### 3.2 CSA 缓存可复用性

| 层级 | 计算方式 | query-dependent | 可复用 |
|------|----------|-----------------|--------|
| Hidden States | 前一层输出 | ❌ | ✅ |
| Compressed KV Entries | $H \cdot W^{KV}$ → Compressor | ❌ | ✅ |
| Compressed Indexer Keys | $H \cdot W^{KV}$ → Compressor | ❌ | ✅ |
| Sliding Window KV | $H \cdot W^{KV}$（原始未压缩） | ❌ | ✅（需边界处理） |
| **Index Scores** | Indexer Queries · Keys 点积 + 加权和 | ✅ | ❌ |
| **Selected Compressed KV** | Top-k 选择 | ✅ | ❌ |
| **Core Attention** | MQA on Selected KV | ✅ | ❌ |

**结论**：PS 可复用前 4 层，后 3 层（Top-k 检索 + Core Attention）必须 reuser 自己算。

### 3.3 最大堵点：压缩块边界对齐

CSA 每 $m$ 个 token 压缩为一个块，若 trim 点不是 $m$ 的倍数，边界块跨越 provider/reuser：

```
Provider: [0, 1, 2, 3, 4, 5, 6, 7]  (prefix_len=8)
Reuser:               [6, 7, 8, 9]   (suffix)

m=4：
  Block 0: tokens 0-3
  Block 1: tokens 4-7  ← 跨越边界！包含 token 6,7（也是 reuser 的）
```

**CSA 还重叠压缩**：相邻块共享 token，trim 点落在重叠区域时多个块受影响。

| 方案 | 做法 | 代价 |
|------|------|------|
| A. 强制对齐 | prefix 长度必须是 $m$ 的倍数 | 限制太大，不现实 |
| B. 存 hidden states | 不存压缩 KV，reuser 自己做 Token-Level Compressor | 存储增加，无边界问题 |
| C. 部分块重算 | 完整块复用，跨越边界的最后一个块重新计算 | 最优，大部分复用 |

**对 build_kv 的影响**：当前 `torch.cat([provider_kv, reuser_kv])` 需改为：
1. 拼接压缩 KV（处理边界对齐）
2. 拼接索引 keys（处理边界对齐）
3. 拼接滑动窗口 KV（处理跨边界）
4. reuser 自己做 Top-k 检索
5. 做核心注意力（MQA）

### 3.4 SWA + HCA 的难点

![HCA 架构图](SWA+HCA.png)

#### 3.4.1 滑动窗口跨越 trim 边界

SWA 窗口 $n_{win}=128$。reuser 的前几个 token 的窗口会回看 provider 尾部：

```
Provider: [0, ..., 100]  (prefix_len=101)
Reuser:        [95, ..., 102]  (suffix)

Reuser token 101 的 SWA 需要 [93, ..., 100]
→ 93-100 在 provider，101 在 reuser → 跨边界
```

- 若 `prefix_len < n_win`，SWA 窗口会缺 token
- 解决：在 `provider_cache` 中额外存**最后 `n_win` 个 raw KV** 用于回填

#### 3.4.2 HCA 双路径对齐

HCA 局部路径（SWA-like）和全局路径（CSA）的 KV 拼接方式不同，最终门控融合：

```
HCA-local:  provider_raw_kv[-n_win:] + reuser_raw_kv
HCA-global: provider_comp_kv + reuser_comp_kv（需处理边界）
```

若两条路径的拼接边界不一致，门控融合时 token 对应关系错位。

#### 3.4.3 缓存结构膨胀

CSA 下 `provider_cache` 需同时存：

```python
provider_cache = {
    "full_compressed_kv":  [prefix_len/m, c],      # CSA 全局路径
    "full_indexer_keys":   [prefix_len/m, c_I],     # Top-k 检索
    "raw_kv_last_nwin":    [n_win, c],               # SWA / HCA-local 回填
    "hidden_states":       [prefix_len, d],          # 备选：重算用
}
```

#### 3.4.4 因果 mask 重算

SWA 的 mask 是带状（对角线附近 $n_{win}$ 宽度），不是下三角。PS 拼接后需重新构造跨越 trim 边界的带状 mask。

#### 3.4.5 总结

| 难点 | 说明 | 严重程度 |
|------|------|---------|
| 窗口跨越 trim 边界 | SWA 回看跨 provider/reuser | 🔴 高 |
| HCA 双路径对齐 | 局部/全局拼接边界不一致，门控融合错位 | 🔴 高 |
| 缓存结构膨胀 | 需同时存压缩 KV、索引 keys、尾部 raw KV | 🟡 中 |
| trim 长度限制 | 必须 `prefix_len >= n_win` | 🟡 中 |
| 因果 mask 重算 | 带状 mask 拼接后需重新构造 | 🟢 较低 |

### 3.5 MQA 与 PS 注入粒度

DSv4 Core Attention 是 MQA（单 KV 头），所有 query 头共享同一组 selected KV。PS 的 KV 注入粒度需要匹配：
- 当前 MLA 是多头的，注入按 head 维度拼接
- MQA 只有一个 KV 头，注入逻辑需调整，但复杂度反而降低（不用处理 head 维切分）

### 3.6 压缩 KV 与 PS 原始 KV 语义冲突

当前 PS 的 `provider_kv` 是原始 token 的 KV（经过 MLA 低秩压缩）。DSv4 的 prefix KV 是窗口级聚合后的压缩块，语义不同：
- PS 存的是逐 token 的 KV，CSA 存的是每 $m$ 个 token 聚合成的一个块
- 拼接时不能简单 `torch.cat`，需要处理压缩率对齐

---

> 本文档最后更新：2026-07-13
