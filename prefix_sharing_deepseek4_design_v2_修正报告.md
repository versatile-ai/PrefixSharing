# DeepSeek V4 PrefixSharing 集成设计 v2 独立审查与修正报告

> 被审文档：`D:\note\deepseek_PS\prefix_sharing_deepseek4_design_v2.md`  
> 基准：当前 PrefixSharing 源码、`D:\note\deepseek-v4\PS支持DeepSeekV4-堵点梳理.md`、论文/HF 方法  
> 审校日期：2026-07-24

---

## 0. 最终裁决

**v2 已从根本上纠正 v1 的方向性错误**，将架构收敛到 **suffix-only Q + 扩展 key-side history** 的正确数据流，并删除了 `attn_o`、Hook D、residual/post/comb、Transformer patch、provider score 复用等错误派生组件。因此，v2 不是“在错误架构上打补丁”，而是**一份方向正确但尚未完成关键细节的方案**。

但是，v2 仍存在若干会导致实现直接出错的硬核缺口，按阻塞程度排序：

1. **CSA 重叠压缩边界未处理**（v2 用 HCA 非重叠示例代替了 CSA 真实问题）。
2. **Phase 2 边界重算缺少 `raw_hidden` 字段**（store 字段与处理策略自相矛盾）。
3. **SWA 跨越 trim 边界处理缺失**（堵点文档中严重程度最高的问题未展开）。
4. **HCA ratio=128 的 topk 重算语义不清**（HCA 理论上是稠密注意力，不应有 top-k）。

在以上四项取得 MindSpeed 源码证据或明确设计决策之前，**不建议进入编码阶段**。

---

## 1. 审校方法

本次审校继续采用上一版修正报告确立的“独立必要性审查”原则：

1. 被审文档中的每个 claim 只是待验证假设，不是架构前提。
2. 以当前 PrefixSharing 真实数据流、堵点梳理文档、论文/HF 方法为优先证据。
3. 无法被证明为必要或无法被证明为正确的组件/字段/步骤，应明确标出并暂缓实施。
4. 原 `prefix_sharing_deepseek4_design_v2.md` 保持只读，本报告独立成文。

---

## 2. v2 相比 v1 的重大改进（应予保留）

| 维度 | v1 问题 | v2 处理 | 评价 |
|---|---|---|---|
| 数据流 | 扩展 `attn_o` / residual / hidden 到 full length | 固定 suffix-only Q/output/hidden | ✅ 正确 |
| 存储字段 | `attn_o`、`residual_prefix`、`post_prefix`、`comb_prefix` | 仅 `kv`、`kv_compress`、`indexer_k` | ✅ 必要且充分 |
| Hook | A/B/C/D 四个 Hook，D 专门扩展 output | 仅 Phase 3→4 一个插入点 | ✅ 最小化 |
| Transformer patch | 为补偿 full-output shape 而 patch `TransformerLayer` | 删除 | ✅ 避免架构污染 |
| CSA top-k | provider score 复用、填 `-1` 近似 | reuser 自己重算 score/top-k | ✅ 符合 query-dependent 约束 |
| fallback | trim 后 CSA original_forward skip | 删除 | ✅ 避免破坏性 trim 后回退 |

这些改进与上一版修正报告提出的“三不变量”一致：

1. reuser 的 Q、attention output、projected output、residual/mHC、下一层 hidden 均保持 suffix-only。
2. 只扩展 attention history（raw/compressed KV、indexer keys、SWA tail、boundary state），不扩 query-side output。
3. capability/fallback 必须在任何破坏性 trim 前完成。

---

## 3. 严重问题（不解决不能实现）

### 3.1 CSA 重叠压缩边界：v2 只按 HCA 非重叠压缩处理

**问题描述**

v2 §4.5 以 `P=3, S=3, r=4` 为例说明压缩块边界问题：

```
P//r + S//r = 0 + 0 = 0
(P+S)//r    = 6//4 = 1
```

该例子假设的是非重叠压缩。但 **CSA（ratio=4）是重叠压缩**，每个压缩块 $C_i^{Comp}$ 同时依赖前窗口的后 $m$ 个 token 和后窗口的前 $m$ 个 token，实际 receptive field 为 $2m = 8$ 个 token（见论文 §2.1.1 公式 11-12）。

因此 CSA 的边界问题比 v2 描述的更复杂：

- trim 点若落在重叠区，会同时影响**多个**压缩块。
- 仅存储“最后 `r - P%r` 个 token 的 raw hidden”无法覆盖 $2m$ 的 receptive field。
- provider 尾部与 suffix 头部拼接后，需要重跑的压缩块数量取决于 trim 点在重叠窗口中的具体位置。

**为什么这是个错误**

v2 把 CSA 的边界问题降格为 HCA 的非重叠问题，会导致实现时低估需要重算的数据量和存储字段。实际运行中若 `P % 4 != 0`，拼接出的 `kv_compress` 会在边界处出现数值错误，且该错误会沿层传播。

**修正建议**

为 CSA 单独建立边界模型：

1. 明确 CSA 压缩器 receptive field = $2m$ token。
2. 根据 trim 点位置，计算需要重算的所有压缩块索引。
3. 存储 provider 尾部至少 $2m - 1$ 个 raw hidden（或 raw KV，取决于 MindSpeed 压缩器输入）。
4. reuser 重跑 compressor 时输入 `provider_tail_hidden + suffix_head_hidden`，得到跨越边界的完整压缩块。

在取得 MindSpeed `g2_attention.py` / `dsa_indexer.py` 实际循环边界前，CSA 非对齐前缀应作为**不支持场景**被 capability gate 拦截。

---

### 3.2 Phase 2 边界重算缺少 `raw_hidden` 字段

**问题描述**

v2 §4.5 提出两阶段策略：

- **Phase 1**：断言 `P % compress_ratio == 0`，只支持对齐 prefix。
- **Phase 2**：provider 存储最后 `r - P%r` 个 token 的 raw hidden，reuser 用 provider tail + suffix head 重跑 `self.compressor()` 得到跨边界压缩块。

但 v2 §3.1 的 `StoredG2Activation` 字段只有：

| 字段 | 形状 |
|---|---|
| `kv` | `[s, b, 512]` |
| `kv_compress` | `[s//r, b, 512]` |
| `indexer_k` | `[s//4, b, 1, 128]` |
| `stored_len` | `int` |

**没有 raw hidden 字段**。Phase 1 和 Phase 2 同时存在是自相矛盾的：

- 若严格执行 Phase 1 断言，Phase 2 永远不会触发，应删除。
- 若要支持 Phase 2，必须新增字段（且字段长度需按 3.1 节的 receptive field 重新计算）。

**修正建议**

二选一：

**方案 A（推荐 Phase 1 先行）**：
- 永久断言 `P % compress_ratio == 0`。
- 删除 Phase 2 及其相关描述。
- 在 capability gate 中明确拦截非对齐 prefix。
- 业务上要求 prefix_len 按 128（HCA）和 4（CSA）对齐，或 planner 自动向下取整到最近对齐点。

**方案 B（后续扩展）**：
- 新增 `hidden_tail` 字段，长度为压缩器 receptive field（CSA 为 $2m$，HCA 为 $m$）。
- 明确该字段仅在非对齐层非 None。
- reuser 重跑 compressor 时使用 `provider.hidden_tail + suffix.hidden_head`。

当前 v2 混合两种方案，应清理为单一策略。

---

### 3.3 SWA 跨越 trim 边界处理缺失

**问题描述**

堵点文档 §3.4.1 将 SWA 窗口跨越 trim 边界标为“🔴 高”：

```
Provider: [0, ..., 100]  (prefix_len=101)
Reuser:        [95, ..., 102]  (suffix)

Reuser token 101 的 SWA 需要 [93, ..., 100]
→ 93-100 在 provider，101 在 reuser → 跨边界
```

v2 只在 §3.1 表格中说明 `kv` 用于“sparse_flash_mla 的 raw KV，SWA attention”，但**完全没有讨论**：

1. reuser 前几个 token 的 SWA 需要回看 provider 尾部 `n_win` 个 raw KV。
2. `prefix_len < n_win` 时窗口缺 token 的处理。
3. 带状 causal mask 拼接后如何重建。
4. HCA 双路径（local SWA-like + global compressed）拼接边界不一致导致门控融合错位的问题。

**为什么这是个错误**

SWA 不是可选优化，而是 DeepSeek V4 保证因果性的必要分支。若 provider 没有提供足够 tail，reuser 前几个 token 的 attention 会访问被 trim 掉的 token，导致数值和语义错误。

**修正建议**

增加 SWA 专门章节，明确：

1. **存储字段**：provider 除 `kv` 外，还需存储 `raw_kv_tail`，长度为 `min(prefix_len, n_win)`（HCA-local 与 CSA/HCA 的 SWA 补充分支共用）。
2. **位置与 mask**：tail 的语义位置、窗口起始位置、带状 mask 如何构造。
3. **双路径对齐**：HCA 的 local 路径用 `provider_raw_kv_tail + reuser_raw_kv`，global 路径用 `provider_kv_compress + reuser_kv_compress`；两条路径扩展后必须在 token 维度对齐，再输入门控融合。
4. **prefix_len < n_win**：_capability gate_ 应允许但记录可用 tail 长度小于窗口；attention kernel 的 mask 必须只覆盖实际存在的 token，不能假设窗口满。

---

### 3.4 HCA ratio=128 的 topk 重算语义不清

**问题描述**

v2 §4.4 表格：

| ratio | 重算方式 |
|---|---|
| 128 | `self.get_compress_topk_idxs(expanded_seqlen)` — 纯位置重算 |
| 4 | Reuser 自己的 q × 扩展后的 indexer_k → 重新打分 |

但论文 §2.2 明确说明 **HCA 是稠密注意力，不做稀疏 top-k 选择**：

> “HCA：更激进压缩……但**不做稀疏注意力**，保持稠密。”

如果 HCA 没有 top-k，v2 的 `get_compress_topk_idxs(expanded_seqlen)` 调用是什么语义？可能情况：

1. **最可能**：v2 把 CSA 的函数名误用到 HCA。HCA 不需要 compress_topk_idxs，应传 `None` 或空张量。
2. **次可能**：MindSpeed 实现中 HCA 有某种位置索引（如 compressed attention 的 block index），但不是 top-k。需要说明其物理含义。
3. **小概率**：论文与实现不一致，HCA 实际有 top-k。需要源码证据。

**修正建议**

在本地或容器中跑 MindSpeed DeepSeek V4 单步 forward，对 HCA 层打印：

- `compress_topk_idxs is not None` 是否成立
- 若为 None，v2 应删除 ratio=128 的 topk 分支
- 若不为 None，需说明其 shape/dtype 及与 CSA topk 的区别

在取得证据前，应在 capability gate 中将 HCA 的 `compress_topk_idxs` 处理标为“待验证”，不要直接套用 CSA 逻辑。

---

## 4. 中等问题（影响正确性，需在实现前澄清）

### 4.1 DSA loss 梯度边界未澄清

v2 不变量 3 提出“KV 不 detach”，这与当前 PrefixSharing `prefix_store.py:41` 的“Never detach”设计一致。但 DeepSeek V4 的 DSA loss 在 Phase 4 之后**修改** `kv_compress` 和 `compress_topk_idxs`（v2 §4.1 已注意到这一点），需要明确：

- DSA loss 是作用在整个扩展序列（provider prefix + suffix）还是仅 suffix？
- 若仅 suffix，provider prefix 部分的 `kv_compress` 梯度必须阻塞（例如对 provider tail 部分做 stop-gradient）。
- 若全局，则梯度自然流过 provider compressor，数学上正确但内存开销更大。

**修正建议**：在 §5 训练集成中增加“DSA loss 作用域与梯度边界”小节，给出明确决策。

---

### 4.2 位置编码 / 部分 RoPE 未讨论

DeepSeek V4 的关键细节：

- **部分 RoPE**：仅对向量最后 64 维施加。
- **双 RoPE 系统**：`main`（theta=10000）和 `compress`（theta=160000）。
- 压缩器使用 compress RoPE。

v2 没有讨论：

1. provider KV 在 Phase 1 后已带 provider 位置编码，reuser suffix KV 已带 suffix 位置编码，拼接后是否自然连续？
2. 压缩器重跑边界块时，compress RoPE 的 position 如何从 provider 尾部连续到 suffix 头部？
3. 部分 RoPE 的 64 维范围在扩展后是否需要特殊处理？

**修正建议**：增加 RoPE 连续性证明或实验验证章节。

---

### 4.3 mHC 未讨论

v2 完全没有提及 mHC。虽然 mHC 大概率是 token-local（A/B/C 映射由当前 token 残差状态生成），但作为 DeepSeek V4 的核心组件，应在文档中明确：

- mHC 不需要跨 provider/reuser 传递状态。
- reuser suffix hidden 经过 mHC 后与 prefix 的 mHC 状态无关。
- 这样可彻底排除 v1 中“扩展 residual/post/comb”的诱惑。

---

### 4.4 Patch 大小仍可压缩

v2 §4.1 说 patch 需要“复制 Phase 1-3 的编排代码（~40 行）以获取变量访问权”。实际上可以尝试更小侵入的方式：

- patch `self.compressor.forward` 捕获 `kv_compress` 输出
- patch `self.indexer.forward_with_index_compress` 捕获 `key_index`
- 在 `sparse_attention` 调用前插入单行 hook

如果必须复制 40 行，v2 需要说明哪些变量无法通过局部 hook 获取。

---

### 4.5 训练集成过于简略

v2 §5 只列了 `wrap_forward_step()` 和“prefix-last restore”，没有讨论：

- verl actor 与 standalone pretrain 的 restore 差异（当前 `verl_mcore.py` 已区分）。
- logprob/entropy/loss 调用链。
- 辅助 loss（DSA、MoE 负载均衡）如何纳入。
- 最终数值验证的 A≈B≈C 基线（A=无 patch 完整 forward，B=patch+完整 fallback，C=trim+PS）。

**修正建议**：扩展 §5，复用上一版修正报告 §7 的 restore 分类。

---

## 5. 轻微问题 / 待确认

### 5.1 `stored_len` 字段单位混乱

`kv`、`kv_compress`、`indexer_k` 的有效长度分别是 `P`、`P//r`、`P//4`，用单一 `stored_len` 无法同时表达。建议改为：

- 保留 `prefix_len`（语义长度）
- 增加 `valid_kv_len`、`valid_cmp_len`、`valid_idx_len` 等 per-field 长度

或至少说明 `stored_len` 指语义 prefix_len，各张量按自身压缩率推导 slice。

---

### 5.2 `indexer_k` 压缩率假设

v2 假设 indexer_k 压缩率固定为 4（`[s//4, b, 1, 128]`）。需要确认 MindSpeed 中 indexer 的压缩步长是否一定等于 CSA 压缩率，还是独立配置。

---

### 5.3 CP=1 是阶段性限制还是最终限制？

v2 只说“并行约束：Phase 1 CP=1”。CP 不支持是 PrefixSharing 当前已知限制，但 DSv4 长上下文场景对 CP 需求更强，应说明这是 Phase 1 临时限制，并给出后续开放计划。

---

### 5.4 测试覆盖不足

v2 §7 测试计划应增加：

- CSA/HCA 边界非对齐（`P % r != 0`，含 CSA 重叠）
- SWA 窗口跨越（`prefix_len < n_win` 和 `>= n_win`）
- 数值等价性：PS ON vs OFF、A≈B≈C
- verl 集成端到端

---

## 6. 推荐修正后的组件矩阵

| 组件/字段 | v2 当前状态 | 推荐状态 | 理由 |
|---|---|---|---|
| `StoredG2Activation.kv` | 保留 | ✅ 保留 | raw KV，用于 SWA 和 attention |
| `StoredG2Activation.kv_compress` | 保留 | ✅ 保留 | HCA/CSA 压缩历史 |
| `StoredG2Activation.indexer_k` | 保留 | ✅ 保留 | CSA indexer key space |
| `StoredG2Activation.raw_hidden_tail` | 无 | ⚠️ 条件新增 | 仅当支持非对齐 prefix 时需要 |
| `StoredG2Activation.raw_kv_tail` | 无 | ⚠️ 必须新增 | SWA 窗口回填 |
| `StoredG2Activation.stored_len` | 单一 int | ❌ 改为 per-field | 不同压缩率长度不同 |
| `_g2_kv_store_or_expand` | 一个函数 | ✅ 保留主体 | 但需分 HCA/CSA/SWA 分支 |
| ratio=128 topk 重算 | 保留 | ❌ 删除或待验证 | HCA 不应有 top-k |
| CSA 重叠边界处理 | 缺失 | ❌ 必须补充 | 当前示例只覆盖 HCA |
| SWA 跨边界处理 | 缺失 | ❌ 必须补充 | 堵点最高优先级 |
| Patch 大小 | ~40 行 | ⚠️ 尝试更小 | 用局部 hook 替代整段复制 |
| mHC 说明 | 缺失 | ⚠️ 补充声明 | 证明无需扩展 residual |
| RoPE/位置编码 | 缺失 | ⚠️ 补充验证 | 双 RoPE + 部分 RoPE |

---

## 7. 实施前必须完成的取证清单

在动笔写代码前，必须锁定以下证据（建议在容器内单步 forward 并打印）：

1. **`g2_attention.py` Phase 1-4 变量打印**
   - `kv` shape/dtype/device
   - `kv_compress` shape/dtype/device
   - `key_index`（即 v2 的 `indexer_k`）shape/dtype/device
   - `compress_topk_idxs` shape/dtype/device
   - HCA 层 `compress_topk_idxs` 是否为 None

2. **压缩器 receptive field**
   - CSA compressor 输入窗口实际是多少 token？是否为 $2m$ 重叠？
   - HCA compressor 输入窗口是否为 $m$ 非重叠？
   - 压缩器输入是 hidden 还是 raw KV？

3. **SWA 实现细节**
   - `n_win` 实际值
   - SWA 是独立分支还是 fused 在 `sparse_attention` 内部？
   - mask 是 kernel 内生成还是外部传入？

4. **DSA loss 调用链**
   - DSA loss 输入是否包含 provider prefix 的压缩块？
   - 反向时梯度是否应传回 provider compressor？

5. **RoPE 调用链**
   - 压缩器是否调用 compress RoPE？
   - provider 和 reuser 的 position ids 在拼接后是否连续？

---

## 8. 结论

`prefix_sharing_deepseek4_design_v2.md` 是一份**方向正确、架构收敛、但仍缺关键细节**的方案。它成功删除了 v1 中大量错误派生的组件，但在 CSA 重叠边界、SWA 跨越边界、HCA topk 语义、DSA 梯度边界等硬核问题上仍停留在简化假设或未讨论状态。

**下一步行动**：不要立即编码，先完成第 7 节的 MindSpeed 源码取证，然后根据证据修订 v2 的 §4.5（边界）、§4.4（HCA topk）、§3.1/§3.2（store 字段，增加 `raw_kv_tail` 和可选 `raw_hidden_tail`），并补充 SWA 专门章节。取证完成后，本报告可升级为最终可实施的 v3 基线。

---

*原 `prefix_sharing_deepseek4_design_v2.md` 保持只读，未修改。*
