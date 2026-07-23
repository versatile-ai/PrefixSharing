# DeepSeek V4 PrefixSharing 独立必要性审查与纠错报告

> **被审设计**：`D:\note\deepseek_PS\prefix_sharing_deepseek4_design.md`  
> **主要问题基准**：`D:\note\deepseek-v4\PS支持DeepSeekV4-堵点梳理.md`  
> **方法依据**：`deepseek_v4_论文方法与公式.md`、`deepseek_v4代码概览.md`  
> **项目依据**：`D:\Project\jackie-main\prefix-0501\prefix-sharing` 当前源码  
> **审查日期**：2026-07-22

> [!IMPORTANT]
> 本报告不是对原 AI 设计的补充解释，也不以保全其 A/B/C/D/E 功能组为目标。原设计只是一组待审查的主张；每个组件必须先证明对 DeepSeek V4 PrefixSharing 的正确闭环确有必要，才能进入推荐方案。

---

## 0. 最终裁决

原设计不能通过局部修补进入实现。它最根本的错误是：

> **把应该始终保持 suffix-only 的 query-side 输出恢复成 full sequence，再用额外 store 和 Transformer patch 修补由此制造的 shape 问题。**

正确主线应固定为：

```text
suffix Q
+ provider prefix 的可复用 attention history
+ reuser 自己的 suffix history
→ suffix attention output
→ suffix projected output
→ suffix residual / mHC / next-layer hidden
→ 仅在最终训练输出边界 restore
```

而不是：

```text
suffix Q
→ 拼 provider attn_o 得到 full output
→ 扩 residual/post/comb
→ 每层恢复 full hidden
```

### 0.1 直接删除

以下内容不进入推荐方案：

- `attn_o` provider store；
- Hook D 及 `_g2_expand_attn_output()`；
- `residual_prefix`、`post_prefix`、`comb_prefix`；
- `_g2_store_transformer_data()`、`_g2_expand_transformer_data()`；
- 仅为上述 full-output 扩展服务的 TransformerLayer patch；
- provider `indexer_score` 复用、均值/最大值聚合和填 `-1` 方案；
- HCA ratio=128 “static top-k”默认抽象；
- trim 后 ratio=4/CSA 调 `original_forward(suffix-only)`；
- 中间层恢复 `[P+S]` hidden 的设计与测试预期。

### 0.2 必须重写

- DeepSeek V4 typed state；
- HCA/CSA/SWA history assembler；
- compressed boundary/halo；
- model-level pre-trim capability gate；
- semantic/packed/compressed position contract；
- standalone pretrain 与 verl actor 两套训练 adapter；
- packed metadata、restore、recompute 和并行能力声明。

### 0.3 必须先取证

- 目标 MindSpeed-LLM repo/commit 与文件 hash；
- HCA/CSA/SWA 实际分派；
- compressor、indexer、kernel、RoPE、mask、mHC 的真实 API；
- THD/BSHD/TP/SP/PP/CP 下的 tensor axis 与 shape；
- standalone loss 和 recompute 调用链。

---

## 1. 审查方法与证据优先级

### 1.1 证据顺序

| 优先级 | 证据 | 用途 |
|---:|---|---|
| 1 | 当前 PrefixSharing 真实源码和测试 | 判断现有数据流、API、layout、生命周期 |
| 2 | `PS支持DeepSeekV4-堵点梳理.md` | 提供问题线索；结论仍需与当前源码和方法资料交叉核验 |
| 3 | DeepSeek V4 论文/HF 方法资料 | 判断 HCA、CSA、SWA、Indexer、RoPE、mHC 的算法语义 |
| 4 | 锁定版本的目标 MindSpeed 源码与运行 trace | 决定最终 hook、shape、kernel 和训练 contract |
| 5 | 静态推导 | 证明数学反例、shape 不变量和生命周期矛盾 |

原设计本身不属于事实证据。

### 1.2 状态标记

- **已确认**：由当前源码或明确方法公式直接支持；
- **条件成立**：方向合理，但需目标 MindSpeed API/trace 验证；
- **待取证**：本地资料不足，不能写成实施设计；
- **删除**：违反基本不变量或由错误前提派生。

### 1.3 对堵点文档的必要纠偏

`PS支持DeepSeekV4-堵点梳理.md` 正确识别了 compressed boundary、SWA 跨 trim、CSA query-dependent indexer 等关键问题，但不是不可质疑的最终事实：

1. 当前 checkout 已有通用 `BatchedBatchLayout`、BSHD backend 和相关 runtime 分支；因此“当前仓库完全没有 BSHD 路径”已过时。但这**不代表 DeepSeek V4 已支持 BSHD**。
2. `prefix_len < window` 并非天然不支持；所需 provider raw tail 应是 `min(prefix_len, required_left_history)`。
3. HCA global 是 HCA dense compressed history，不是 CSA；HCA 和 CSA 都可能伴随 SWA/local raw history，具体组合需目标源码确认。

---

## 2. 三个不可破坏的数据流不变量

### 2.1 不变量一：reuser 的 Q 和层输出保持 suffix-only

当前 planner 已区分：

- Q 路径：`kept_lengths_q`；
- KV 路径：`expanded_lengths_kv`。

依据：

- `prefix_sharing/core/planner.py:246-305`；
- `prefix_sharing/backends/torch_ref.py:178-218`。

对 reuser，应始终满足：

| 张量/状态 | token 轴 |
|---|---:|
| layer input hidden | `S` |
| attention Q / index query | `S` |
| raw/compressed history | `P+S` 或其压缩表示 |
| attention output | `S` |
| output projection | `S` |
| residual / mHC post | `S` |
| next-layer hidden | `S` |

核心关系是：

```text
Q length = S
KV/history length = P+S（或对应 compressed entries）
Attention output length = Q length = S
```

论文也是按每个 query token `t` 定义输出 `o_{t,i}`：

- `deepseek_v4_论文方法与公式.md:163-177`；
- `deepseek_v4_论文方法与公式.md:207-217`。

### 2.2 不变量二：只扩 history，不扩 query-side output

可以复用/扩展：

- HCA complete compressed entries；
- CSA compressed KV；
- CSA compressed indexer keys；
- SWA raw KV tail；
- compressor boundary raw input/halo；
- position、mask、layout metadata。

不能作为 prefix history 扩展：

- Q / index query；
- attention output；
- output projection 输入；
- residual/post/comb；
- next-layer hidden；
- provider query-dependent score/top-k。

当前标准 PrefixSharing 路径也只组装 K/V history，不扩 attention output：

- `integrations/megatron_runtime.py:110-161`；
- `backends/torch_ref.py:52-218`。

### 2.3 不变量三：fallback 必须发生在 trim 前

唯一安全顺序：

```text
完整原始 batch
  → 扫描整个模型和当前运行配置
  → 判断所有 sequence-mixing layers 是否都可保持 suffix-only 正确性
  ├─ 全部支持：允许 trim，进入优化路径
  ├─ 有未支持项：保持完整 batch，走 original forward
  └─ 配置/API 不匹配：fail-fast
```

禁止：

```text
先 trim
→ 到 ratio=4/未支持层
→ original_forward(suffix-only)
```

后者不是 fallback，而是丢失 prefix context。

### 2.4 restore 仅发生在最终框架边界

- verl actor：恢复最终 per-token logprob/entropy 语义；
- standalone pretrain：重建 unreduced token loss、prefix-last label、numerator/denominator 和辅助 loss；
- attention、TransformerLayer 和 mHC 中间不恢复 full hidden。

---

## 3. 原设计组件必要性矩阵

### 3.1 核心组件

| 原设计组件 | 判定 | 纠错结论 |
|---|---|---|
| Prefix detector / planner | **保留并扩展** | 继续负责 provider/reuser/prefix_len；增加 attention kind、boundary、capability 信息 |
| 破坏性 suffix trim | **有条件保留** | 仅当 model-level preflight 全部通过后才能执行 |
| 完整输入 fallback | **必要，前移** | 必须在 trim 前决定，不能 layer-local 临时 fallback |
| DeepSeek typed state | **必须重写** | 按 HCA/CSA/SWA/boundary/indexer-key/raw-tail 拆分 |
| 完整 prefix raw KV | **通常不必要** | SWA 只需实际窗口尾部；boundary 重算保存 compressor 所需最小 raw 输入 |
| compressed KV | **必要** | HCA/CSA history 的核心；不能直接按 `P//r` 切片拼接 |
| compressed indexer keys | **CSA 必要，原设计遗漏** | reuser 对完整 key 空间重新评分 |
| raw KV window tail | **SWA 必要，原设计遗漏** | suffix 前部 query 回看 provider 尾部 |
| compressor boundary state / halo | **必要，原设计遗漏** | 重算跨 trim 的 HCA/CSA entries |
| semantic position/mask/layout | **必要，原设计不足** | 保证 RoPE、窗口、压缩条目和 packed 坐标一致 |
| provider indexer score | **删除** | score 依赖 provider query，不能替代 reuser query |
| provider top-k | **删除** | top-k 依赖 reuser 当前 query |
| `attn_o` | **删除** | query-side output，不是可复用 history |
| residual/post/comb prefix state | **删除** | 由错误的 full-output 扩展派生 |
| 单一 `stored_len` | **删除** | 混合 token、block、query 等不同单位 |

### 3.2 Hook 与 patch

| 原设计 hook/patch | 判定 | 正确替代 |
|---|---|---|
| Hook A：存完整 raw KV | **重写** | 只捕获 SWA tail 和 boundary 所需 raw state |
| Hook B：存 compressed KV | **保留语义、重写实现** | 同时捕获 CSA indexer keys、entry positions、boundary metadata |
| Hook C：attention 前注入 | **必要、彻底重写** | 按 HCA/CSA/SWA 分支组装 history 和 metadata |
| Hook D：存/拼 `attn_o` | **删除** | attention output 保持 suffix-only |
| “必须四个 Hook” | **删除** | 先定义 semantic capture/assemble 事件，再由真实源码决定最小 hook |
| 完整 fork attention forward | **有条件** | 只有目标实现无稳定 extension point 时才考虑，并锁 source hash |
| Transformer `_forward_attention` patch | **删除主线** | 若仅用于 residual/full-output 扩展则完全不需要 |

### 3.3 HCA / CSA 特定组件

| 组件 | 判定 | 原因 |
|---|---|---|
| HCA ratio=128 static top-k | **删除** | 方法上 HCA 无 Lightning Indexer，使用 dense compressed attention |
| `_adjust_topk_indices_for_batch` 用于 HCA | **删除默认设计** | 若目标 NPU 需要 indices，先确认其只是 mask/layout 元数据，不得称为 HCA top-k |
| `prefix_len//128` 直接增加 cmp offset | **删除** | 非对齐 prefix 会产生跨界 block |
| CSA provider score mean/max | **删除** | 改变 query-dependent indexer 算法 |
| CSA 只 offset suffix indices | **删除** | 不会把 provider keys 加入候选 |
| CSA exact reuser score/top-k | **必要** | suffix query 对完整 provider+suffix index-key 空间重新计算 |

### 3.4 训练和基础设施

| 组件 | 判定 | 纠错结论 |
|---|---|---|
| prefix-last restore 数学思想 | **有条件保留** | 最终输出仍需恢复，但调用接口和 autograd 必须按框架重写 |
| standalone 直接复用 verl restore | **删除** | `(output_tensor, loss_func)` 与 `dict(log_probs,entropy)` 不是同一 contract |
| `wrap_forward_step` | **待入口取证** | 需要 prepare/context/restore 编排，但具体接入点取决于真实 `get_batch()` |
| 当前 in-place restore | **必须重写** | PP/autograd 风险，改为 out-of-place |
| PatchRegistry | **独立工程前置** | 修复 pending 覆盖、disable、幂等；不属于 attention 数学架构 |
| TP/SP/PP/CP 已支持声明 | **重写** | 标准 attention 能力不能自动继承到 DSv4/NPU adapter |
| 通用 BSHD 能力 | **已存在基础设施** | 不等于 DSv4 BSHD optimized path 已支持 |

---

## 4. 明确删除决定

## 4.1 删除 `attn_o` store 与 Hook D

原设计：

- `StoredG2Activation.attn_o`：`prefix_sharing_deepseek4_design.md:341-358`；
- provider 捕获与 reuser 拼接：`755-761`；
- `_g2_expand_attn_output()`：`901-936`。

删除理由：

1. `attn_o` 是当前 query 的 attention 结果，不是 query-independent history；
2. reuser Q 是 suffix-only，所以 output 也是 suffix-only；
3. prefix token 的 layer output 已由 provider 行计算，不应复制进 reuser 层间 hidden；
4. suffix query 对 prefix 的依赖已经通过 expanded history 建立；
5. 拼 `attn_o` 会错误地把 `S` 变成 `P+S`，破坏下一层 suffix-only contract。

删除内容：

- `attn_o` 字段及 merge/store/load；
- Hook D；
- `_g2_expand_attn_output()`；
- expanded `attn_o` transitive store；
- full-length output projection 逻辑；
- 相关测试。

## 4.2 删除 residual/post/comb 扩展与 Transformer patch

原设计 `1237-1460` 的整个功能组建立在“attention 返回 `P+S`、residual 只有 `S`”这一错误前提上。

正确路径：

```text
suffix residual
→ suffix mHC pre
→ attention(Q=S, history=P+S)
→ suffix attention output
→ suffix BDA / mHC post
→ suffix next-layer hidden
```

因此删除：

- `residual_prefix`；
- `post_prefix`；
- `comb_prefix`；
- `_g2_store_transformer_data()`；
- `_g2_expand_transformer_data()`；
- 仅为 shape 补偿的 TransformerLayer patch；
- 所有期待 `[P+S]` residual/mHC output 的测试。

mHC 仍需目标源码确认 pre/post 是否严格 token-local、tuple、RNG 和 recompute contract；但取证目标是“如何保持 suffix-only”，不是“如何扩 prefix residual”。

## 4.3 删除 provider score 补洞方案

删除：

- `indexer_score` reusable state；
- provider score mean/max；
- 用 provider candidates 填 `-1`；
- “只 offset 也可能足够”；
- 将这些近似称为 baseline parity。

CSA score 由当前 reuser suffix query 与 compressed indexer keys 共同决定：

- `deepseek_v4_论文方法与公式.md:139-161`；
- `PS支持DeepSeekV4-堵点梳理.md:84-96`。

可复用的是 key-side 表示，不是 provider query 的 score。

## 4.4 删除 HCA static top-k 默认设计

方法资料明确：

- HCA 使用非重叠重压缩；
- 无 Lightning Indexer；
- 对可见 compressed history 做 dense MQA。

依据：`deepseek_v4_论文方法与公式.md:187-227`。

若目标 MindSpeed 内部用 index tensor 描述 HCA causal block 可见范围，必须先由真实源码证明，并称为 kernel metadata，不能直接写成 HCA sparse top-k。

## 4.5 删除 trim 后的 ratio=4 skip/fallback

原设计先在入口删除 prefix，再在 ratio=4 层调用 original forward：

- trim：`1471-1503`；
- skip：`579-586,1055-1062`。

这会让 original forward 只看到 suffix，不是安全 fallback。正确处理只能是：

- 在 trim 前发现 CSA 未支持，整个 batch 保持 full input；或
- 完成 CSA exact history/indexer 支持后再 trim。

---

## 5. 真正需要实现的最小状态

## 5.1 公共计划与身份

```text
PrefixExecutionPlan
  forward_id / microbatch_id
  provider_idx / reuser_idx / prefix_len / suffix_len
  original semantic positions
  full/suffix logical row metadata
  attention kind per layer
  boundary/window capability
  training restore specification
```

cache identity 至少包含：

```text
forward_id, microbatch_id, layer_id,
provider_idx, prefix_len, attention_kind/path,
tp/cp ownership, state version
```

## 5.2 HCA state

```text
HCAHistoryState
  reusable complete compressed entries
  boundary raw compressor input
  compressed entry semantic positions/offsets
  dense causal visibility metadata
  raw SWA tail（若该实现包含 local 分支）
```

HCA 的核心问题是非对齐边界，不是 static top-k：

\[
\lfloor P/r\rfloor + \lfloor S/r\rfloor
\ne \lfloor(P+S)/r\rfloor
\]

例如 `P=3,S=3,r=4`，分段压缩为 0 块，完整序列为 1 块。跨界块必须由 provider tail + suffix head 重算。

## 5.3 CSA state

```text
CSAHistoryState
  reusable compressed KV
  reusable compressed indexer keys
  compressor overlap/left halo
  compressed entry semantic positions
  causal/top-k mask metadata
  raw SWA tail（若该实现包含 local 分支）
```

reuser 必须自算：

```text
suffix index query
→ 对完整 provider+suffix index keys 评分
→ exact top-k
→ selected compressed KV
→ core attention
→ indexer loss
```

## 5.4 SWA state

```text
SWAHistoryState
  provider raw KV tail
  tail semantic positions
  valid tail length
  window size/dilation/causal convention
  row/window mask metadata
```

尾部长度不是固定要求 `prefix_len >= window`，而是：

```text
raw_tail_len = min(prefix_len, required_left_history)
```

## 5.5 Compressed boundary state

```text
CompressedBoundaryState
  complete reusable entry boundary
  raw hidden / projection input halo
  receptive-field definition
  suffix head requirement
  crossing entry positions/count
  padding/tail policy
```

组装流程：

```text
provider complete entries
+ provider boundary raw state
+ suffix raw head
→ 重算所有 crossing entries
→ suffix complete entries
→ 完整 semantic-order history
```

不能再使用通用：

```python
prefix_len // ratio
valid_len // ratio
cat(provider_cmp, suffix_cmp)
```

## 5.6 mHC state

默认：

```text
provider prefix mHC reusable state = 无
```

只需取证：

- pre/post 是否 token-local；
- 是否存在跨 token reduction；
- tuple/recompute/dropout contract；
- PP stage contract。

只有真实源码证明有跨 token 状态，才新增对应 state。

## 5.7 训练恢复状态

verl actor：

```text
provider prefix token outputs
provider prefix-last logits
reuser labels
original row lengths/masks
```

standalone pretrain：

```text
provider unreduced prefix logits/losses
provider prefix-last logits
reuser labels and causal shift
original loss_mask and denominator
auxiliary loss components
DP reduction metadata
```

---

## 6. 正确的 suffix-only 数据流

## 6.1 模型前置阶段

```text
原始完整 batch
    │
    ├─ detect shared prefix
    ├─ build immutable plan
    ├─ 扫描所有 sequence-mixing layers
    ├─ 核验 attention kind / boundary / SWA / indexer
    ├─ 核验 layout / position / parallel / recompute / training adapter
    │
    ├─ 任一必要能力未支持
    │      └─ 保持完整 batch → original forward
    │
    └─ 全部支持
           └─ provider 保持完整
              reuser 裁为 suffix
```

## 6.2 每层数据流

```text
Provider row                            Reuser row
full layer hidden                      suffix-only hidden
      │                                      │
      ├─ token-local mHC pre / norm          ├─ token-local mHC pre / norm
      ├─ Q                                   ├─ suffix Q
      ├─ raw/compressed history              ├─ own suffix history
      └─ publish reusable history            │
                                             ▼
                                  load provider history
                                             │
                  ┌──────────────────────────┼─────────────────────────┐
                  │                          │                         │
             SWA/local                 HCA/global                CSA/global
       raw provider tail          complete compressed       compressed KV + keys
       + suffix raw KV            entries + boundary       + overlap halo
       + window mask              recompute                + suffix score/top-k
                  └──────────────────────────┬─────────────────────────┘
                                             │
                                  expanded history only
                                             │
                           attention(Q=suffix, history=expanded)
                                             │
                                  suffix attention output
                                             │
                              suffix inverse Partial RoPE
                                             │
                              suffix grouped projection
                                             │
                             suffix BDA / mHC post
                                             │
                                suffix next-layer hidden
```

全程不出现：

```text
provider attn_o 拼接
prefix residual/post/comb 拼接
中间层 full hidden restore
```

## 6.3 Transitive reuse

若 row1 reuse row0，同时又是 row2 的 provider，row1 发布的是其**已组装完成的历史 state**：

```text
row0 prefix reusable history
+ row1 own suffix history
+ 必要 boundary 重算
→ row1 expanded reusable history
→ row2 按自己的 prefix_len 切片/重算
```

不发布 row1 prefix `attn_o` 或 residual。

## 6.4 最终恢复

```text
final suffix outputs
    │
    ├─ verl actor
    │    → out-of-place logprob/entropy restore
    │
    └─ standalone pretrain
         → unreduced token loss reconstruction
         → prefix-last label-specific CE
         → denominator + auxiliary loss + DP reduction
```

---

## 7. 必须重写的架构组件

## 7.1 Model-level capability gate

必须在 trim 前返回：

```text
OPTIMIZED_SUFFIX_ONLY
FULL_INPUT_FALLBACK(reason)
UNSUPPORTED_FAIL_FAST(reason)
```

只要存在以下任一情况，就不能 trim：

- 某种 attention family 未支持；
- CSA 完整 index-key 候选空间无法构建；
- HCA/CSA boundary 无法重算；
- SWA raw tail/mask 未闭合；
- semantic position 无法证明；
- kernel 不支持 suffix Q + expanded history；
- source hash/signature 不匹配；
- recompute、dropout、PP/CP/MTP 超出 capability；
- 训练 adapter 未闭合。

## 7.2 Semantic history assemblers

建议按语义而非原 Hook 字母组织：

- `HCAHistoryAssembler`；
- `CSAHistoryAssembler`；
- `SWAHistoryAssembler`；
- `FullInputFallback`。

统一接口只负责 history-side：

```text
assemble_history(provider_state, suffix_state, semantic_positions)
→ expanded history
→ mask/packed metadata
→ publish transitive history
```

## 7.3 Position 与 layout contract

至少区分：

| 坐标 | 含义 |
|---|---|
| semantic token position | token 在原始序列中的位置 |
| packed physical offset | THD/BSHD tensor 的物理寻址位置 |
| compressed entry position | compressor 定义的 block/entry 位置 |
| continuation/start position | 续训或 cache continuation 起点 |

必须证明：

- suffix Q 与 provider prefix K/history 使用 baseline 相同的相对语义位置；
- SWA raw tail 和 suffix 位于同一语义坐标系；
- compressed entry position 来自真实 compressor contract；
- packed offset 不能替代 semantic position；
- output inverse RoPE 只处理 suffix query output，不能修复已经错误的 Q/K score。

## 7.4 Packed metadata

THD 下必须区分 logical sequence、physical batch、packed token、compressed entry 和 head axis。不得用 `hidden_states.shape[1]` 的 `bsz` 直接代表 planner batch。

`PackedSeqParams` 必须保持真实 Tensor dtype/device/contiguous，并从最终 Q/raw/compressed history layout 完整构建：

```text
cu[-1] == corresponding tensor physical length
cu monotonic
max_seqlen == max(diff(cu))
padded >= valid
all indices are in bounds and use the correct coordinate space
```

## 7.5 Framework adapters

### VerlMegatronAdapter

- verl batch/context；
- final-only logprob/entropy restore；
- prefix-last logits；
- out-of-place restore；
- NestedTensor/2D/BSHD contract。

### StandalonePretrainAdapter

- 在真实 `get_batch()` 后接入；
- 同步处理 tokens、labels、loss mask、positions、attention metadata；
- 暴露/重建 unreduced token loss；
- 主 loss numerator/denominator；
- indexer/MoE/MTP auxiliary losses；
- DP reduction 和每参数梯度。

不能直接把 verl restore 用在 `(output_tensor, loss_func)` 上。

## 7.6 PatchRegistry

若仍使用 import hook，应先修复：

- 同 module 多 pending spec 被 dict 覆盖；
- disable 不取消 pending hook；
- 重复 register/install 套 wrapper；
- partial failure 无原子 rollback。

这是集成基础设施前置条件，不应与 DeepSeek attention 数学混写。

---

## 8. 当前源码事实与待取证项

## 8.1 当前 PrefixSharing 已确认事实

- planner 已区分 suffix Q 与 expanded KV；
- 标准 reference/runtime 只扩 K/V history；
- store 默认不 detach；
- 当前已有通用 BSHD layout/backend/runtime 分支；
- 当前 context 仍固定使用标准 `PrefixAttentionStore`；
- 当前 verl restore 是 framework-specific，且含 in-place 赋值；
- activation recompute 可能发生在 forward context reset/store close 之后；
- PatchRegistry 存在上述生命周期问题。

这些事实能决定基础不变量，但不能证明 DSv4 HCA/CSA/SWA adapter 已存在。

## 8.2 目标 MindSpeed 必须取证

1. repo、commit、MindSpeed Core/Megatron/CANN 版本；
2. `g2_attention.py`、compressor、indexer、TransformerLayer 文件 hash；
3. HCA/CSA/SWA 实际分派；
4. SWA 是独立层、压缩 attention 的 local branch，或两者兼有；
5. compressor 尾块、padding、overlap、receptive field 和 entry position；
6. Q/raw KV/compressed KV/index-key/top-k/output 的 shape/axis/stride；
7. THD logical batch 与 physical batch；
8. mask、`PackedSeqParams` 和 compressed cumulative metadata；
9. RoPE、inverse RoPE、`start_pos` 的位置语义；
10. HCA 若使用 index tensor，其语义是 top-k 还是 causal/layout metadata；
11. mHC pre/post、dropout、recompute；
12. TP/SP/PP/CP collective 前后 shape；
13. standalone `get_batch()`、loss closure 和 auxiliary loss 调用链。

没有这些证据，具体 hook 名、类名、签名只可作为待取证示意。

---

## 9. 分阶段实施与准出条件

| 阶段 | 目标 | 准出条件 |
|---|---|---|
| **Phase 0** | 锁定目标源码和 trace | attention kind、axis、position、mask、compressor、训练 contract 均有证据 |
| **Phase 1** | model gate + full fallback | A 无 patch与 B patch+fallback 无副作用；fallback 全在 trim 前 |
| **Phase 2** | 最简单但完整的 suffix-only attention family PoC | 该 family baseline 的所有分支均覆盖；Q/output/hidden 都是 suffix-only |
| **Phase 3** | 非对齐 boundary + SWA + transitive state | HCA/CSA crossing entries、raw tail、mask、梯度正确 |
| **Phase 4** | CSA exact indexer | reuser score/top-k/indexer loss 基于完整候选空间重算 |
| **Phase 5** | 多层 + mHC suffix-only | 不存在 output/residual 扩展；逐层和每参数梯度通过 |
| **Phase 6** | verl 与 standalone adapters | 各自 final restore/loss/auxiliary/gradient contract 闭合 |
| **Phase 7** | 并行和训练特性 | TP、SP、DP、PP、recompute、dropout、CP、MTP 逐项 gate 后开放 |

> [!WARNING]
> Phase 2 不能预先假设“无 SWA HCA”。应从 Phase 0 选择目标实现中最简单但完整的真实 attention family；若 HCA baseline 自带 SWA，则同阶段必须实现 raw window tail。

首阶段限制建议：

```text
TP=1
SP off
PP=CP=1
dropout=0
all recompute off
MTP off
no unverified transitive reuse
```

---

## 10. 验证矩阵

## 10.1 数据流不变量

| 检查点 | Provider | Reuser | 必须成立 |
|---|---|---|---|
| layer input hidden | 自身 full/有效 token | suffix-only | reuser 不出现 prefix hidden |
| Q / index query | 自身 query | suffix-only | 不扩 Q |
| history | 自身 history | provider prefix + own suffix | 只扩 history |
| attention output | provider query length | suffix length | 不扩 output |
| residual/mHC/next hidden | provider query length | suffix length | 下一层继续 suffix-only |
| restore | 最终 adapter | 最终 adapter | 中间层不 restore |

## 10.2 Attention 与边界

| 维度 | 覆盖值 |
|---|---|
| attention kind | HCA、CSA、SWA、HCA+SWA、CSA+SWA |
| boundary | 对齐、`P mod r=1`、`r-1`、总长度短于一个块 |
| suffix | 1、短 suffix、跨多个块 |
| SWA | `P<W`、`P=W`、`P>W` |
| CSA query | 相同 prefix、不同 suffix query |
| prefix relation | direct、同 provider 多 prefix length、transitive |
| positions | `start_pos=0/>0`、无关 packed row 长度变化 |
| provider row | batch 首行、中间行 |

## 10.3 Capability / fallback

| 条件 | 预期 |
|---|---|
| 所有 adapter/boundary/position/layout 支持 | `OPTIMIZED_SUFFIX_ONLY`，之后才 trim |
| 任一 sequence-mixing layer 未支持 | `FULL_INPUT_FALLBACK`，禁止 trim |
| source hash/signature 不匹配 | fail-fast 或显式 full fallback |
| trim 后才发现不支持 | 测试失败，视为架构缺陷 |
| 通用 BSHD 可用但 DSv4 adapter 未确认 | 不得判定 optimized |
| `prefix_len<window` 且可构造 raw tail | 允许优化，不应单独拒绝 |

## 10.4 第一个数值发散点

依次比较：

1. plan/provider/prefix_len；
2. semantic positions；
3. raw history；
4. reusable compressed entries；
5. boundary/halo 重算；
6. indexer keys；
7. reuser index query；
8. score；
9. top-k；
10. SWA raw tail 与 mask；
11. RoPE 后 Q/K；
12. attention logits/probability；
13. suffix attention output；
14. suffix projected output；
15. suffix layer hidden；
16. final unreduced token output/loss；
17. auxiliary losses；
18. 每参数 gradient；
19. optimizer state/update；
20. 多步参数轨迹。

## 10.5 三基线

- **A**：无 PrefixSharing patch；
- **B**：安装 patch，但 gate 强制完整输入 fallback；
- **C**：启用限定优化。

顺序：先 A≈B，再 B≈C，最后才测性能。

---

## 11. 首阶段明确不承诺

- 通用 BSHD 已存在即可支持 DSv4；
- 任意 MindSpeed 版本；
- 非零 dropout 的逐 token RNG 等价；
- activation/mHC/input-LN recompute；
- PP/CP；
- MTP；
- 未验证的 TP/SP；
- 任意 transitive reuse；
- 任意同 provider 多 prefix length；
- BF16/NPU bitwise 一致。

精度应分层定义：FP32 reference、identity path、BF16/NPU baseline noise、逐层误差预算、每参数梯度、optimizer update、多步轨迹。

---

## 12. 最终推荐模块边界

```text
core/
  detector / planner
  model-level capability gate
  semantic prefix-state identities

integrations/deepseek_v4/
  HCA history assembler
  CSA history + exact indexer assembler
  SWA history assembler
  compressed boundary assembler
  position/layout adapter

integrations/frameworks/
  verl Megatron adapter
  standalone pretrain adapter

setup/
  version/source-hash guard
  lifecycle-safe patch registry
```

具体文件和 hook 必须在 Phase 0 取得目标源码后确定，不预先复制原设计的 A/B/C/D Hook 命名。

---

## 13. 最终结论

原设计中真正可保留的是：

- detector/planner 的共享关系；
- provider state 不 detach；
- typed state 的方向；
- 优先调用目标融合 kernel；
- 分阶段验证和保守能力限制。

必须整体删除的是：

```text
attn_o store / Hook D
→ residual/post/comb 扩展
→ Transformer full-output patch
```

以及：

```text
provider score 复用
HCA static top-k
trim 后 original-forward skip
```

真正需要新增的核心不是更多 query-side 激活，而是：

- HCA/CSA complete compressed history；
- CSA compressed indexer keys；
- SWA raw KV tail；
- compressor boundary raw state/halo；
- semantic position、mask 和 packed metadata；
- reuser suffix 上的 exact score/top-k；
- final-only framework-specific loss/output restore。

> **只有当整个模型所有 sequence-mixing layers 都能维持 `suffix Q + expanded reusable history → suffix output/hidden` 时，才允许执行 trim；否则必须在 trim 前保留完整输入并 fallback。**
