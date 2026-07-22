# DeepSeek V4 模型在 vLLM 中的实现分析

## 模型参数 (DeepSeek-V4-Flash)

| 参数 | 值 |
|---|---|
| `hidden_size` (D) | **4096** |
| `hc_mult` (M) | **4** |
| `num_attention_heads` (H) | **64** |
| `num_key_value_heads` | **1** (MQA) |
| `head_dim` | **512** (448 NoPE + 64 RoPE) |
| `q_lora_rank` | **1024** |
| `o_lora_rank` | **1024** |
| `o_groups` | **8** |
| `n_routed_experts` | **256** |
| `num_experts_per_tok` | **6** |
| `moe_intermediate_size` | **2048** |
| `index_topk` | **512** |
| `index_n_heads` | **64** |
| `index_head_dim` | **128** |
| `sliding_window` | **128** |
| `vocab_size` | **129280** |
| `num_hidden_layers` | **43** |
| `hc_sinkhorn_iters` | **20** |

`hc_mult3 = 2M + M² = 8 + 16 = 24`

---

## 一、MHC (Multi-head Latent Compression)

### 1.1 核心机制

MHC 将传统的**单条残差流**扩展为 **M=4 条并行残差流**。每个子层（Attention/FFN）只消费一条压缩后的**单流** `[N, D]`，子层输出被"写回" 4 条流中，4 条流之间通过 **Sinkhorn 正则化的可学习路由矩阵** 进行交叉混合。

MHC 与 KV Cache 在代码层面完全解耦。MHC 的职责域是流间混合与压缩/扩展，KV Cache 的职责域是 Attention 内部的存储与检索，两者没有交叉引用。

### 1.2 MHC Pre — 从 M 条流压缩为子层输入

**输入**: `residual [N, M, D]` — 上层的 M 条残差流

**权重**: `fn [hc_mult3, M*D] = [24, 16384]`，分为三组：

```
fn 的分区 (M=4, hc_mult3=24):

  fn[:4]      [4, 16384]   → pre_mix   每条流的"发言权"门控
  fn[4:8]     [4, 16384]   → post_mix  子层输出注入各流的强度
  fn[8:24]    [16, 16384]  → comb_mix  流间路由矩阵 (4×4)
```

**计算步骤**：

```python
x_flat = residual.view(N, M*D).float()         # [N, 16384]
mixes = x_flat @ fn.T                           # [N, 16384] @ [16384, 24] = [N, 24]
mixes = mixes * rsqrt(sqrsum / (M*D) + eps)    # RMSNorm-like

pre_logits = mixes[:, :4]                       # [N, 4]
pre_mix = sigmoid(pre_logits * scale[0] + base[:4]) + eps   # [N, 4]

post_logits = mixes[:, 4:8]                     # [N, 4]
post_mix = sigmoid(post_logits * scale[1] + base[4:8]) * 2.0  # [N, 4]

comb_logits = mixes[:, 8:24]                    # [N, 16]
comb_mix = softmax(comb_logits.view(N, 4, 4))   # [N, 4, 4]
for _ in range(20):                              # Sinkhorn 迭代
    comb_mix /= comb_mix.sum(dim=-1) + eps
    comb_mix /= comb_mix.sum(dim=-2) + eps
# → 近似双随机矩阵 [N, 4, 4]

layer_input = Σ pre_mix[:, i] * residual[:, i]  # [N, D]
```

**输出**:

| 变量 | Shape | 含义 |
|---|---|---|
| `layer_input` | `[N, D]` | 单流，送入子层 |
| `post_mix` | `[N, M, 1]` | 留存，等子层结束后做 Post |
| `comb_mix` | `[N, M, M]` | 留存，流间路由矩阵 |

### 1.3 MHC Post — 把子层输出"写回"多流残差

```python
mixed_residual = comb_mix @ residual             # [N,4,4] @ [N,4,D] = [N,4,D]
post_term = post_mix * x.unsqueeze(-2)            # [N,4,1] × [N,1,D] = [N,4,D]
new_residual = mixed_residual + post_term         # [N,4,D]
```

### 1.4 首层广播

首层输入是 embedding `[N, D]`，没有多流残差，使用 `mhc_pre_broadcast_tilelang`：

```python
fn_broadcast = fn.view(-1, M, D).sum(dim=1)     # [24, D]
mixes = x @ fn_broadcast.T                       # [N, D] @ [D, 24] = [N, 24]
# 同样拆分为 pre/post/comb_mix
residual_out = pre_mix.unsqueeze(-1) * x.unsqueeze(0)  # 广播: [N, 1, D] → [N, M, D]
layer_input = Σ pre_mix × residual_out           # [N, D]
```

### 1.5 层间 Fused Post→Pre

`mhc_fused_post_pre_tilelang` 将 Post(上一个子层) + Pre(下一个子层) 融合为一个 kernel。每层调用两次：

```
第一次: Post(上FFN) + Pre(当前Attn), 使用 hc_attn_fn  ← 首层例外，只有 Pre
第二次: Post(Attn) + Pre(当前FFN), 使用 hc_ffn_fn
```

`hc_attn_fn` 和 `hc_ffn_fn` 各自独立，形状均为 `[24, 16384]`。每层 MHC 参数约 3.14 MB。

### 1.6 完整数据流（含侧信道）

子层（Attn/FFN）永远只看到和产出 `[N, D]`，`[N, M, D]` 仅存在于 MHC 侧信道中：

```
                             input_ids [N]
                                  │
                                  │  embed_tokens
                                  ▼
                     ┌─────────────────────────┐
                     │  x [N, D]               │  ← 单流
                     │  D=4096                 │
                     └────────────┬────────────┘
                                  │
  residual=None                    │
  post_mix=None                    │
  res_mix=None                     │
                                  │
               ╔══════════════════▼══════════════════════╗
               ║  mhc_pre_broadcast_tilelang            ║  首层专用
               ║                                        ║
               ║  fn_broadcast [24, D]  = [24, 4096]   ║
               ║  mixes = x @ fn_broadcastᵀ  [N, 24]   ║
               ║                                        ║
               ║  ├─ pre_mix  [N, M]     = [N, 4]      ║  每条流的"发言权"
               ║  ├─ post_mix [N, M, 1]  = [N, 4, 1]   ║  子层输出注入强度
               ║  ├─ res_mix  [N, M, M]  = [N, 4, 4]   ║  流间路由 (双随机)
               ║  │                                      ║
               ║  ├─ residual [N, M, D]  = [N, 4, 4096] ║  ← 广播创建 4 流
               ║  └─ x [N, D]                           ║  ← 加权和压回单流
               ╚══════════════╤══════════════════════════╝
                              │
     ┌────────────────────────┼──────────────────────────────┐
     │  侧信道状态             │  主数据流                    │
     │  residual  [N, 4, 4096] │  x [N, 4096]                │
     │  post_mix  [N, 4, 1]    │                              │
     │  res_mix   [N, 4, 4]    │                              │
     └────────────────────────┼──────────────────────────────┘
                              │
                              │  Attention 子层: 只看到单流 [N, 4096]
                              ▼
               ┌──────────────────────────────────┐
               │  DeepseekV4Attention             │
               │                                  │
               │  fused_wqa_wkv: [4096]→[1536]   │
               │  ├─ qr [N, 1024]                │
               │  └─ kv [N, 512]                 │
               │                                  │
               │  wq_b: [1024]→[64×512]          │
               │  → q [N, 64, 512]               │
               │                                  │
               │  FlashMLA(q, swa_cache,          │
               │           sparse_kv_cache)       │
               │  → o [N, 64, 512]               │
               │                                  │
               │  wo_a: [64×512//8]→[8×1024]     │
               │  wo_b: [8×1024]→[4096]          │
               │  → x [N, 4096]                  │
               └────────────────┬─────────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────────┐
     │  residual  [N, 4, 4096]  │  x [N, 4096]  ← Attn 输出   │
     │  post_mix  [N, 4, 1]     │                              │
     │  res_mix   [N, 4, 4]     │                              │
     └──────────────────────────┼──────────────────────────────┘
                                │
               ╔════════════════▼══════════════════════════════╗
               ║  mhc_fused_post_pre_tilelang                 ║
               ║  (Post Attn + Pre FFN, 融合为一个 kernel)    ║
               ║                                             ║
               ║  ┌── Post ────────────────────────────────┐ ║
               ║  │ mixed = res_mix @ residual             │ ║
               ║  │   [N,4,4] @ [N,4,4096] → [N,4,4096]  │ ║
               ║  │ residual' = mixed + post_mix * x      │ ║
               ║  │   [N,4,4096] + [N,4,1]×[N,1,4096]    │ ║
               ║  └────────────────────────────────────────┘ ║
               ║                                             ║
               ║  ┌── Pre ─────────────────────────────────┐ ║
               ║  │ hc_ffn_fn [24, M×D] = [24, 16384]     │ ║
               ║  │ mixes = residual'.flat @ hc_ffn_fnᵀ   │ ║
               ║  │   [N,16384] @ [16384,24] → [N,24]     │ ║
               ║  │ → pre_mix' [N,4]                      │ ║
               ║  │ → post_mix' [N,4,1]                   │ ║
               ║  │ → res_mix'  [N,4,4]                   │ ║
               ║  │ x' = Σ pre_mix'ᵢ × residual'ᵢ        │ ║
               ║  │   [N,4096]                            │ ║
               ║  └────────────────────────────────────────┘ ║
               ╚══════════════╤═══════════════════════════════╝
                              │
     ┌────────────────────────┼──────────────────────────────┐
     │  residual' [N, 4, 4096]│  x' [N, 4096]  ← FFN 输入   │
     │  post_mix' [N, 4, 1]   │                              │
     │  res_mix'  [N, 4, 4]   │                              │
     └────────────────────────┼──────────────────────────────┘
                              │
                              │  MoE 子层: 只看到单流 [N, 4096]
                              ▼
               ┌──────────────────────────────────┐
               │  DeepseekV4MoE (MegaMoE)         │
               │                                  │
               │  gate: [4096] → [256]            │
               │  fused_topk_bias(topk=6)         │
               │  → topk_weights [N, 6]          │
               │  → topk_ids     [N, 6]          │
               │                                  │
               │  w13 [E, 2×2048, 2048]  MXFP4   │
               │  w2  [E, 4096, 1024]   MXFP4    │
               │  deep_gemm.fp8_fp4_mega_moe()   │
               │                                  │
               │  shared_expert (可选):           │
               │    gate_up [4096, 2×2048]        │
               │    down    [2048, 4096]          │
               │                                  │
               │  → x [N, 4096]                  │
               └────────────────┬─────────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────────┐
     │  residual' [N, 4, 4096]  │  x [N, 4096]  ← FFN 输出   │
     │  post_mix' [N, 4, 1]     │                              │
     │  res_mix'  [N, 4, 4]     │                              │
     └──────────────────────────┼──────────────────────────────┘
                                │
               ╔════════════════▼══════════════════════════════╗
               ║  下一层: mhc_fused_post_pre_tilelang         ║
               ║  (Post FFN + Pre Attn)                       ║
               ║  换用 hc_attn_fn [24, 16384]                ║
               ║                                             ║
               ║  ... 循环 43 层 ...                          ║
               ║  每层: Post + Attn + Post + Pre + FFN       ║
               ║  Attn 侧用 hc_attn_fn                        ║
               ║  FFN  侧用 hc_ffn_fn                         ║
               ╚══════════════╤═══════════════════════════════╝
                              │
     ┌────────────────────────┼──────────────────────────────┐
     │  residual  [N, 4, 4096]│  x [N, 4096]                │
     │  post_mix  [N, 4, 1]   │                              │
     │  res_mix   [N, 4, 4]   │                              │
     └────────────────────────┼──────────────────────────────┘
                              │
               ╔══════════════▼══════════════════════════════╗
               ║  mhc_post_tilelang                         ║
               ║                                           ║
               ║  mixed = res_mix @ residual               ║
               ║    [N,4,4] @ [N,4,4096] → [N,4,4096]    ║
               ║  hs = mixed + post_mix * x                ║
               ║    [N,4,4096]                             ║
               ║                                           ║
               ║  hs = hs.mean(dim=1)  →  [N, 4096]       ║
               ║  4 流简单平均坍缩为单流                    ║
               ╚══════════════╤════════════════════════════╝
                              │
                              │  hs [N, 4096]
                              ▼
               ┌──────────────────────────────────┐
               │  hc_head_fused_tilelang          │
               │  hc_head_fn [M, M×D] = [4,16384] │
               │  → [N, 4096]                     │
               │                                  │
               │  RMSNorm                         │
               │  lm_head  [4096] → [129280]      │
               │  → logits [N, 129280]            │
               └──────────────────────────────────┘


═══════════════════════════════════════════════════════════════

    [N, M, D] 存在于侧信道 (residual, post_mix, res_mix)
    [N, D]     存在于主数据流 (子层输入/输出)

    子层(Attn, FFN) 从头到尾只看到 [N, D], 完全感知不到 MHC 的存在

═══════════════════════════════════════════════════════════════
```

```
Embed(token) → x [N, D]
     │
     ▼  mhc_pre_broadcast
侧信道: residual [N, M, D], post_mix [N, M, 1], comb_mix [N, M, M]
主数据流: x [N, D]
     │
     ▼  Attention (只看到 [N, D])
主数据流: x [N, D]
     │
     ▼  mhc_fused_post_pre (Post Attn + Pre FFN, hc_ffn_fn)
侧信道更新, 主数据流: x [N, D]
     │
     ▼  FFN/MoE (只看到 [N, D])
主数据流: x [N, D]
     │
     ▼  mhc_fused_post_pre (Post FFN + Pre Attn, hc_attn_fn)
     ...  循环 43 层 ...
     │
     ▼  mhc_post_tilelang → mean(dim=1) → [N, D]
     │
     ▼  hc_head_fused_tilelang → [N, D] → RMSNorm → lm_head → logits
```

**关键**: 子层（Attn/FFN）从头到尾只看到 `[N, D]`，对 MHC 的多流结构完全透明。MHC 不扩展子层的计算量，只扩展残差记忆容量。

---

## 二、MLA Sparse Attention

### 2.1 三条投影路径总览

```
                         hidden [N, 4096]  bf16
                              │
         ┌────────────────────┼────────────────────────┐
         │                    │                        │
         ▼                    ▼                        │
   ┌──────────────┐    ┌──────────────┐               │
   │    Q 路径     │    │   KV 路径     │               │
   │  (低秩压缩)   │    │  (直接 MQA)   │               │
   └──────┬───────┘    └──────┬───────┘               │
          │                   │                        │
          ▼                   ▼                        │
   ┌──────────────┐    ┌──────────────┐               │
   │  fused_wqa_wkv  MergedColumnParallelLinear       │
   │  权重: [4096, 1024+512] = [4096, 1536]           │
   │  disable_tp=True (ReplicatedLinear)              │
   │                                                  │
   │  前半 1024 列 ─────────── 后半 512 列             │
   │       │                       │                  │
   │       ▼                       ▼                  │
   │  qr [N, 1024] bf16      kv [N, 512] bf16        │
   │  q_lora_rank              head_dim               │
   │       │                       │                  │
   │  RMSNorm(qr)             RMSNorm(kv)             │
   │       │                       │                  │
   │       ▼                       ▼                  │
   │  ┌──────────────┐    ┌──────────────┐            │
   │  │    wq_b      │    │   写入 SWA    │            │
   │  │ ColumnParallel│   │   Cache       │            │
   │  │ Linear       │    │ [blk,256,512] │            │
   │  │              │    └──────────────┘            │
   │  │ 权重:        │          │                    │
   │  │ [1024,       │     FlashMLA                │
   │  │  64×512]     │     在 kernel 内部           │
   │  │ = [1024,     │     读取 SWA K               │
   │  │   32768]     │          │                    │
   │  │              │          │                    │
   │  │ TP: 沿 head  │          │                    │
   │  │ 维度切分     │          │                    │
   │  └──────┬───────┘          │                    │
   │         │                  │                    │
   │         ▼                  │                    │
   │  Q [N, 64, 512] bf16       │                    │
   │  = [N, n_heads, head_dim] │                    │
   │         │                  │                    │
   │  ┌──────┴──────┐           │                    │
   │  │ nope  │ rope│           │                    │
   │  │ [448] │ [64]│           │                    │
   │  │ 不旋转│ RoPE│           │                    │
   │  └──────┴──────┘           │                    │
   │         │                  │                    │
   └─────────┼──────────────────┘                    │
             │                                       │
             │  ┌────────────────────────────────────┘
             │  │
             ▼  ▼
       ┌─────────────────────┐
       │      FlashMLA       │  ← MQA: 64 Q 头 × 1 KV
       │  Q [N,64,512]       │
       │  K [N,1,512]        │  ← 从 SWA cache + 压缩 cache 读取
       │  O [N,64,512]       │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────────────────────────────┐
       │               O 路径 (低秩分组压缩)           │
       │                                             │
       │  O [N, 64, 512] bf16                        │
       │       │                                     │
       │  按 G=8 组切分, 每组 8 个头:                  │
       │  head 0..7   → group 0                     │
       │  head 8..15  → group 1                     │
       │  ...                                        │
       │  head 56..63 → group 7                     │
       │       │                                     │
       │  per group: [N, 8×512] = [N, 4096]          │
       │       │                                     │
       │       ▼                                     │
       │  ┌──────────────────────────┐               │
       │  │    wo_a (bmm, 每组独立)   │               │
       │  │  ColumnParallelLinear    │               │
       │  │  is_bmm=True             │               │
       │  │  bmm_batch_size=8        │               │
       │  │                          │               │
       │  │  权重: [8, 4096, 1024]   │               │
       │  │  = [n_groups,            │               │
       │  │     heads_per_group      │               │
       │  │     × head_dim,          │               │
       │  │     o_lora_rank]         │               │
       │  │                          │               │
       │  │  TP: 沿 group 维度切分    │               │
       │  └──────────┬───────────────┘               │
       │             │                               │
       │  per group: [N, 1024]  bf16                 │
       │  = o_lora_rank                              │
       │             │                               │
       │  concat 8 groups: [N, 8192]                 │
       │  = n_groups × o_lora_rank                   │
       │             │                               │
       │             ▼                               │
       │  ┌──────────────────────────┐               │
       │  │    wo_b                  │               │
       │  │  RowParallelLinear       │               │
       │  │                          │               │
       │  │  权重: [8192, 4096]      │               │
       │  │  = [n_groups×            │               │
       │  │     o_lora_rank,         │               │
       │  │     hidden_size]         │               │
       │  │                          │               │
       │  │  TP: 沿 input 维度切分    │               │
       │  │  (+ all-reduce)          │               │
       │  └──────────┬───────────────┘               │
       │             │                               │
       │             ▼                               │
       │       output [N, 4096]  bf16                │
       │       = hidden_size                         │
       └─────────────────────────────────────────────┘
```

### 2.2 参数量对比

| 路径 | 权重矩阵 | Shape | 参数量 | 标准 MHA 参数量 | 节省 |
|---|---|---|---|---|---|
| Q | `fused_wqa_wkv` (前半) | `[4096, 1024]` | 4.2M | — | — |
| Q | `wq_b` | `[1024, 32768]` | 33.5M | — | — |
| Q 合计 | | | **37.8M** | 134M (`W_Q [4096, 32768]`) | **72%** |
| KV | `fused_wqa_wkv` (后半) | `[4096, 512]` | **2.1M** | 134M×2 (`W_K+W_V`) | **98%** |
| O | `wo_a` (bmm) | `[8, 4096, 1024]` | 33.5M | — | — |
| O | `wo_b` | `[8192, 4096]` | 33.5M | — | — |
| O 合计 | | | **67M** | 134M (`W_O [32768, 4096]`) | **50%** |
| **总计** | | | **107M** | **536M** | **80%** |

### 2.3 维度压缩率

```
              原始维度          压缩维度       压缩率

Q 路径    H × head_dim = 32768    q_lora_rank = 1024   32:1  (参数)
KV 路径   H × head_dim = 32768    head_dim = 512       64:1  (MQA: 1 KV 头 替代 64)
O 路径    H × head_dim = 32768    G × o_lora_rank=8192  4:1  (参数)
```

- **KV 路径最激进**: MQA 只保留 1 个 KV 头，KV Cache 只有标准 MHA 的 1/64
- **Q 和 O 更温和**: 低秩分解做参数压缩，但最终展开为 64 头做精确注意力

```
hidden [N, 4096]
     │
     ├── Q 路径 (低秩压缩):
     │   W_QA [4096, 1024]   → qr [N, 1024]  (压缩 Q, q_lora_rank)
     │   RMSNorm(qr)
     │   W_QB [1024, 64×512] → Q [N, 64, 512]
     │   参数量: 4.2M + 33.5M = 37.8M  (标准 MHA: 134M, 节省 72%)
     │
     ├── KV 路径 (无低秩压缩, 直接 MQA):
     │   W_KV [4096, 512]    → kv [N, 512]  (head_dim)
     │   RMSNorm(kv) → RoPE 后 64 维 → SWA Cache
     │   参数量: 2.1M  (MQA: 64 Q 头共享 1 KV)
     │
     └── O 路径 (低秩压缩, 分组):
         O [N, 64, 512] → 按 G=8 组切分, 每组 8 头
         per group: [N, 4096]
         W_OA [8, 4096, 1024] (bmm) → [N, 8, 1024]
         展平: [N, 8192]
         W_OB [8192, 4096] → [N, 4096]
         参数量: 33.5M + 33.5M = 67M  (标准 MHA: 134M, 节省 50%)
```

### 2.2 层类型分布

43 层按 `compress_ratios` 分为三种类型：

```
层 0, 1:   ratio=0→1  SWA-only   只用滑动窗口
层 2..38:  交替 C4A (偶数) 和 C128A (奇数)
层 39+:     ratio=0→1  SWA-only
```

| | SWA-only | C4A | C128A |
|---|---|---|---|
| 压缩率 | 无 | 4× | 128× |
| Indexer | 无 | **有** | **无** |
| Compressor coff | 无 | 2 (重叠) | 1 (无重叠) |
| KV Cache 份数 | 1 | **5** | 3 |

---

## 三、C4A — 4× 压缩 + Indexer 稀疏检索

### 3.1 为什么需要 Indexer

压缩后 token 数 = 序列长度 / 4。100K 序列 → 25K 压缩 token，全部做 512-dim 注意力仍然太贵。Indexer 用 128-dim 廉价扫描从 25K 中选出最相关的 512 个，主 Attention 只对这 512 个做精确计算。

### 3.2 Indexer 数据流

```
hidden [N, 4096]
     │
     ├→ indexer.wq_b(qr) → Q_idx [N, 64, 128]
     │   fused_indexer_q_rope_quant: RoPE + FP8 quant + scale 折叠进 weights
     │   → q_quant [N, 64, 128] fp8
     │
     ├→ indexer.weights_proj(hidden) → weights [N, 64]
     │   × q 量化 scale × 1/√128 × 1/√64
     │
     └→ indexer.compressor(compressed_kv_score)
           save_partial_states(kv, score+APE) → state_cache [blocks, 4, 512]
           compress(4→1) → indexer_k_cache [blocks, 64, 132]
           132 = 128(fp8 K) + 4(fp32 scale)
```

**Indexer 稀疏检索**:

```python
# Decode: paged 接口
logits = fp8_fp4_paged_mqa_logits(
    Q: q_quant [1, 64, 128] fp8,
    K: indexer_k_cache (paged, 1250 个压缩 token),
    weights: [1, 64]
)
64 头共享 1 套 K (MQA)
logits 展平为 [1, 64×1250] = [1, 80000]
topk → [1, 512]  局部索引 0..1249

# Prefill: 先 gather 再稠密 MQA
cp_gather_indexer_k_quant_cache(paged_cache → contiguous buffer)
mqa_logits(Q, K_contiguous, cu_seqlen)
top_k_per_row_prefill → 每行独立 top 512
```

### 3.3 主 Compressor 数据流

```
hidden [N, 4096]
     │
     └→ compressor.fused_wkv_wgate(hidden)  weight [4096, 2048]
          → kv_score [N, 2048] fp32
          split → kv [N, 1024] + score [N, 1024]
          1024 = coff × head_dim = 2 × 512

kv + score 存入 state_cache [blocks, 4, 2048]:
  每行 2048 bytes = kv(1024) + score+APE(1024)
  APE [4, 1024]: 可学习绝对位置编码, 按 position%4 查表加到 score 上

每积满 4 个连续 token (position%4==3), 触发压缩:
  start = position - 7, 收集 8 个条目 (4 当前 + 4 重叠)
  head_offset: 0 (读 kv 前半) → 重叠窗口 → score 部分
              512 (读 kv 后半) → 当前窗口 → kv 部分
  score = softmax(8 entries, dim=0)  [8, 512]
  compressed = Σ score[i] × kv[i]   [512]
  → RMSNorm(nope 448) → RoPE(rope 64, pos=压缩窗口首 token) → FP8 quant(block=64)
  → 写入 compressed_kv_cache: 448(fp8 nope) + 128(bf16 rope) + 8(ue8m0 scale) = 584 bytes
```

### 3.4 压缩算法核心: 逐维度 softmax 加权求和

```
不是标量加权, 而是 512 个维度各有独立的 8 个 softmax 权重:

for d in 0..511:
    w_d = softmax([score[i,d] for i in 0..7])  # 8 个标量 → 和为 1
    compressed[d] = Σ w_d[i] × kv[i,d]

不同维度可以偏重不同位置的 token。
score 来自 W_score(hidden), 是内容相关的。
APE 来自可学习参数, 是位置相关的 (position%4 选行)。
```

### 3.5 C4A 层 KV Cache 清单 (5 份)

```
1. attn.swa_cache_layer.kv_cache          [blocks, 256, 512]
   最近 128 个原始 token 的完整 KV
   写入: _fused_qnorm_rope_kv_insert
   读取: FlashMLA (k_cache)

2. attn.kv_cache                          [blocks, 64, 584]
   压缩 KV Cache, 全部历史
   写入: 主 compressor.compress_norm_rope_store
   读取: FlashMLA (extra_k_cache)

3. attn.compressor.state_cache.kv_cache   [blocks, 4, 2048]
   压缩中间状态, coff×4=8 个有效条目
   写入: save_partial_states
   读取: compress_norm_rope_store

4. attn.indexer.k_cache.kv_cache          [blocks, 64, 132]
   Indexer K Cache, 全部历史 (128-dim 廉价版)
   写入: indexer.compressor
   读取: SparseAttnIndexer → fp8_paged_mqa_logits

5. attn.indexer.compressor.state_cache.kv_cache  [blocks, 4, 512]
   Indexer 压缩中间状态
   写入: save_partial_states
   读取: compress_norm_rope_store
```

---

## 四、C128A — 128× 压缩，全量 Attention

### 4.1 为什么不需要 Indexer

100K 序列 → 781 个压缩 token。128 SWA + 781 compressed = 909 token，全量 512-dim attention 只需 ~30M FLOPs，比 C4A 的 Indexer 扫描 (205M) + 精确计算 (21M) 更便宜。

### 4.2 压缩差异

```
coff = 1 (无重叠)
fused_wkv_wgate 权重: [4096, 1024] → kv [512] + score [512]
state_cache 每行: 1024 bytes (vs C4A 的 2048)
block_size: 8 (vs C4A 的 4)
每 128 token 触发一次压缩, 收集 128 个条目做 softmax 加权
```

### 4.3 主 Attention 索引来源

C128A 不做 Indexer 稀疏检索。压缩 KV 的索引由 metadata build 阶段的位置查表预计算：

```python
# Metadata Build 时:
_build_c128a_topk_metadata_kernel(positions, compress_ratio, block_table, ...)
  Decode token: position → (position+1)//128 个压缩 slot → 查 block_table → 全局 slot ID
  → c128a_global_decode_topk_indices [N, 1, max_compressed]
  → c128a_decode_topk_lens [N]

# 主 Attention 直接用:
flash_mla_with_kvcache(
    extra_k_cache=compressed_kv,
    extra_indices_in_kvcache=c128a_global_decode_topk_indices,  # 全部压缩 slot
    extra_topk_length=c128a_decode_topk_lens,
)
```

---

## 五、Decode vs Prefill

### 5.1 Indexer

```
Decode:
  paged_mqa_logits(Q_paged, K_paged, block_table)
  → logits [N, 64×1250] → topk [N, 512]

Prefill:
  cp_gather_indexer_k_quant_cache(paged → contiguous buffer)
  dense mqa_logits(Q, K_contiguous, cu_seqlen)   # 变长序列, 每行独立 context
  top_k_per_row_prefill(logits, cu_seqlen)         # 每行从其 context 中选 top 512
```

### 5.2 主 Attention

```
Decode:
  flash_mla_with_kvcache(q, k_cache=swa, extra_k_cache=cmp, extra_indices=topk)
  K 在 paged cache 中, 按索引直接访问

Prefill:
  Chunk 处理 (batch 中的序列分组):
    1. dequantize_gather(compressed_kv) → temp buffer 前半
    2. dequantize_gather(swa_kv)        → temp buffer 后半
    3. combine_topk_swa_indices         → 合并索引, 指向 temp buffer
    4. flash_mla_sparse_fwd(q, kv_buffer, combined_indices)
```

### 5.3 Compressor

逻辑完全相同，prefill 批量执行时可能同时触发多个压缩窗口边界。

---

## 六、Prefix Caching — 相同前缀的 KV Cache 复用

### 6.1 核心原则：所有 Cache 都是确定性的

深层压缩管线中每一步的输入都是确定的（相同的 token + 相同的模型权重 + 相同的 position），因此所有 Cache 都是确定性的产物：

```
步骤                    依赖                              相同前缀→相同结果
────────────────────    ────                              ────────────────
1. hidden_states        相同 token + 相同 model → 相同值    ✓
2. kv = W_kv(hidden)    只依赖 hidden + 权重                ✓
3. W_score(hidden)      只依赖 hidden + 权重                ✓
4. score += ape[pos%4]  只依赖 score + position             ✓
5. save_partial_states  kv + score+APE → state_cache        ✓
6. compress(softmax加权) 只依赖 state_cache 中的条目         ✓
7. RMSNorm + RoPE + FP8  只依赖值 + 权重 + position          ✓
```

每一行 state_cache 是确定的 → 压缩窗口内的条目是确定的 → 压缩 KV 是确定的 → Indexer K 是确定的。

### 6.2 哪些可以共享，哪些不能

```
序列 A: [prefix 0..99] [suffix_A 100..500]
序列 B: [prefix 0..99] [suffix_B 100..500]

                    prefix 部分 (共享)        suffix 部分 (不共享)
                    ─────────────────        ──────────────────

SWA KV pages        ✓ 完全一致               ✗ 各自计算
                    但会被 evict (仅保留      (suffix 逐渐覆盖 prefix)
                    最近 128 token)
                    
压缩 KV pages        ✓ 完全一致               ✗ 各自压缩
(slot 0..24)         压缩管线是确定性的        slot 25+ 不同
                    
Indexer K pages     ✓ 完全一致               ✗ 各自压缩
(slot 0..24)         同上                     slot 25+ 不同

主 State Cache      ✓ prefix 末 8 个条目      ✗ 各自累积
(token 96..99)       必须显式保存/恢复         (suffix 的中间状态)

Indexer State Cache ✓ prefix 末 8 个条目      ✗ 各自累积
(token 96..99)       必须显式保存/恢复
```

### 6.3 C4A 层 Prefix Caching 全景

```
序列 B 从 token 100 继续时, 需要恢复的全部状态:

  ┌──────────────────────────────────────────────────────────┐
  │  1. SWA KV pages              [blocks, 256, 512]        │
  │     prefix 部分可直接共享标准 paged KV cache 机制          │
  │     一致 ✓                                               │
  │                                                          │
  │  2. 压缩 KV pages             [blocks, 64, 584]         │
  │     slot 0..24 是 prefix 的, 可共享                       │
  │     一致 ✓                                               │
  │                                                          │
  │  3. 主 compressor state_cache tail                       │
  │     [blocks, 4, 2048]                                    │
  │     token 96..99 共 4 个条目 (coff×cr=8 的"重叠"部分)     │
  │     一致 ✓, 但必须显式保存                                 │
  │     如果丢失: compress(100..103) 重叠窗口读到垃圾值         │
  │     → 压缩 slot 25 错误 → 影响后续所有 token 的主 Attention │
  │                                                          │
  │  4. Indexer K Cache pages     [blocks, 64, 132]         │
  │     slot 0..24 是 prefix 的, 可共享                       │
  │     一致 ✓                                               │
  │                                                          │
  │  5. Indexer compressor state_cache tail                  │
  │     [blocks, 4, 512]                                     │
  │     token 96..99 共 4 个条目                               │
  │     一致 ✓, 但必须显式保存                                 │
  └──────────────────────────────────────────────────────────┘
```

### 6.4 C128A 层 Prefix Caching 全景

```
序列 B 从 token 100 继续时:

  ┌──────────────────────────────────────────────────────────┐
  │  1. SWA KV pages                                         │
  │  2. 压缩 KV pages (仅 slot 0, coff=1, 无重叠)             │
  │  3. 主 compressor state_cache                            │
  │     token 0..99 共 100 个条目                              │
  │     C128A 的压缩窗口需要凑满 128 token 才触发               │
  │     → 所有 100 个条目都必须保留                              │
  │     → ceil(100/8) = 13 个 state_cache page 需要保存        │
  │                                                          │
  │  如果前缀停在非 128 对齐位置, state_cache 累积了大量条目     │
  │  这些条目必须原样恢复, 否则 token 127 首次压缩结果就错了     │
  └──────────────────────────────────────────────────────────┘
```

### 6.5 Topk 不是缓存，是动态激活值

Indexer 的 topk 虽然输入 K 一致（prefix 部分），但**不同后缀会产生不同的 topk**：

```
Prefix 阶段 (token 0..99): 双方完全一致
  Indexer 扫的 K 相同 → Q·K^T 相同 → topk 相同 ✓

Suffix 阶段 (token 100+): 逐步分化

  token 100:
    context 中只有 prefix 的压缩 KV (slot 0..24)
    suffix 自己的压缩 KV 还没产生
    → 双方扫的 K 完全一致 → topk 一致 ✓

  token 103+:
    suffix_A 的压缩 KV slot 25 出现 (与 suffix_B 不同)
    suffix_B 的压缩 KV slot 25 出现 (与 suffix_A 不同)
    → 双方扫的 K 不同 → Q·K^T 不同 → topk 不同 ✗

  token 200+:
    SWA 中 prefix 已被 evict, 双方看到的完全不同
    → topk 完全不同 ✗
```

**topk 是每次 decode 现场计算的激活值，不写入持久缓存**：

```
持久 Cache (可共享, 引用计数管理):
  SWA KV pages         ──┐
  压缩 KV pages         ─┤ 相同前缀 → 相同值 → 共享
  Indexer K pages       ─┤
  State Cache pages     ─┘

临时激活 (不缓存, 每次重新计算):
  q_quant, logits, topk_indices  ─── 不同后缀 → 不同值 → 各自算
```

topk 是动态产出的"路标"，指着前方哪 512 个压缩 token 值得看。路标指向不同，但路标指的物理位置如果落在 prefix 区域——就是同一条路上的同一块牌子，已经建好了，共享。

### 6.6 与标准 Transformer 的对比

| 维度 | 标准 Transformer | DeepSeek V4 |
|---|---|---|
| 每 token KV 大小 | ~32 KB | ~1.3 KB (C4A) |
| KV 头数 | 64 (MHA) | 1 (MQA) |
| 长程 KV | 全量, 无压缩 | 分层: SWA(全量) + 压缩(稀疏) |
| 缓存类型数 | 1 (KV cache) | 1~5 (取决于层类型) |
| 写入 | 即时, 独立 | 即时(SWA) + 延迟(压缩, 需等窗口填满) |
| 窗口外 token | 保留到序列结束 | SWA 回收, 压缩 KV 永久保留 |
| Prefix 复用需保存的 state | 1 种 (KV page) | 3~5 种 (含 state_cache) |
| State Cache | 无 | C4A: 最近 8 token, C128A: 最近 128 token |
| Topk 是否一致 | N/A (无 Indexer) | prefix 部分一致, suffix 部分各自不同 |
| 当前 vLLM 是否支持 | ✓ (标准 prefix caching) | state_cache 保存/恢复待实现 |

---

## 七、关键参数速查

```
MHC:
  M = 4, hc_mult3 = 24
  hc_attn_fn [24, 16384], hc_ffn_fn [24, 16384]
  Sinkhorn 迭代 20 次

MLA (主 Attention):
  Q: hidden → qr [1024] → Q [64, 512]
  peak_dim = 640 (128 SWA + 512 sparse)
  padded_heads = 64 or 128 (FlashMLA FP8 kernel 要求)

Indexer (仅 C4A):
  Q: qr → Q_idx [64, 128] fp8
  K: indexer_k_cache, 1250 个压缩 token
  logits: [1, 80000] → topk 512

Compressor:
  C4A:  coff=2, fused_wkv_wgate [4096, 2048], state_cache [4, 2048]
  C128A: coff=1, fused_wkv_wgate [4096, 1024], state_cache [8, 1024]
  压缩算法: 逐维度 softmax 加权求和
  APE: [compress_ratio, coff*head_dim] 可学习绝对位置编码

Cache 大小:
  SWA:            512 bytes/token
  Compressed KV:  584 bytes/4 tokens ≈ 146 bytes/token (C4A)
                  584 bytes/128 tokens ≈ 4.6 bytes/token (C128A)
  Indexer K:      132 bytes/4 tokens = 33 bytes/token (C4A)
  State Cache:    仅保留最近 8 或 128 个条目
```

---

## 八、关键文件索引

| 文件 | 内容 |
|---|---|
| `vllm/models/deepseek_v4/nvidia/model.py` | 主模型: DeepseekV4ForCausalLM, DeepseekV4DecoderLayer, DeepseekV4MoE |
| `vllm/models/deepseek_v4/attention.py` | DeepseekV4Attention: MLA + Indexer + Compressor 的组装 |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | FlashMLA 后端: decode/prefill, C4A/C128A/SWA 分发 |
| `vllm/models/deepseek_v4/sparse_mla.py` | FlashMLA metadata 定义, C128A topk metadata kernel |
| `vllm/models/deepseek_v4/compressor.py` | DeepseekCompressor: state_cache 管理, compress 调度 |
| `vllm/model_executor/kernels/mhc/tilelang.py` | MHC Pre/Post/Fused 实现 |
| `vllm/model_executor/kernels/mhc/torch.py` | MHC PyTorch 参考实现 (阅读理解) |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | Indexer: Q 量化, K gather, MQA logits, topk |
| `vllm/models/deepseek_v4/common/ops/save_partial_states.py` | state_cache 写入 |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | 压缩 kernel: softmax 加权 + RMSNorm + RoPE + FP8 quant |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | Indexer Q RoPE + FP8 量化 kernel |
| `vllm/models/deepseek_v4/nvidia/mtp.py` | MTP 推测解码 |
| `vllm/transformers_utils/configs/deepseek_v4.py` | HuggingFace 配置解析 |
