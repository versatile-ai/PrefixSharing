# 压缩边界与 CP 交叉问题分析

> **目的**：梳理问题的来龙去脉，提出可选方案，供评审讨论。

## 1. 背景知识

### 1.1 Compressor 工作原理

DeepSeek V4 的 `self.compressor` 把每 r 个连续 token 的 hidden_states 压缩为一个 CMP entry：

```
r=4:

Token:    [A]    [B]    [C]    [D]    [E]    [F]    [G]    [H]
Hidden:    h0     h1     h2     h3     h4     h5     h6     h7
            │      │      │      │      │      │      │      │
            └──────┴──────┴──────┘      └──────┴──────┴──────┘
                     │                           │
              c0 = f(h0,h1,h2,h3)          c1 = f(h4,h5,h6,h7)

kv_compress = [c0, c1]      shape [8//4, b, 512] = [2, b, 512]
```

compressor 内部流程：

```
hidden [s, b, 4096]
    │
    ├── wkv (Linear):  → kv_proj  [s, b, 512]    每个 token 独立投影
    │                         │
    │                    gather_from_sp_cp         ← CP=1 时无操作
    │                         │
    ├── wgate (Linear): → score    [s, b, 512]    每个 token 的门控权重
    │                         │
    │                    gather_from_sp_cp
    │                         │
    └── 分组: 每 r 个 token 一组
        kv_proj[0:r]  →  加权求和(score)  →  c0
        kv_proj[r:2r] →  加权求和(score)  →  c1
        ...

返回值: kv_compress = [c0, c1, c2, ...]     [s//r, b, 512]
```

**关键**：`c0` 由 `h0, h1, h2, h3` 四个 hidden 共同决定。改任何一个 token，`c0` 就变。压缩是不可逆的——不能从 `c0` 倒推出 `h0, h1, h2, h3`。

### 1.2 Prefix Sharing 的 Trim

PS 裁剪后，Provider 保留全量，Reuser 只保留 suffix：

```
Provider:  [A, B, C, D, E, F, G, H]     8 tokens, 完整序列
Reuser:    [U, V, W, X]                   4 tokens, suffix-only

前缀: [A, B, C, D] — 和 Provider 相同，但被 trim 裁掉了
分叉: 第 5 个位置, D → U
```

### 1.3 CP 的切分

CP 将 packed tensor 按 token 数近似均分到多个 rank：

```
CP=1:  18 tokens → [18, b, 4096]    全在一个 rank
CP=2:  18 tokens → [9, b, 4096] × 2 每个 rank 一半
```

## 2. 问题一：压缩边界问题（CP=1 时已存在）

### 2.1 问题描述

分叉点在压缩块**内部**时，Provider 的 CMP entry 对 Reuser 是错的：

```
r=4, P=3, S=4

Provider: [A, B, C, D, E, F, G, H]
Reuser:   [A, B, C, U, V, W, X]    ← 分叉在 token 3

压缩:
  Provider c0 = f(h0_A, h1_B, h2_C, h3_D)     ← D 的投影
  Reuser   c0'= f(h0_A, h1_B, h2_C, h3_U)     ← U 的投影

  c0 ≠ c0'   ← 第 4 个 token 的投影不同
```

Hook 拼 `cat(provider.kv_compress, reuser.kv_compress)` 时，provider 的 c0 包含的是 D 的信息，不是 U 的信息。Reuser 拿到的压缩块内容是错的。

```
受影响块 = 分叉点 token_index 所在的那个压缩窗口
         = tokens [ (P//r)*r : (P//r+1)*r ]
         
P=3, r=4: 受影响块 = tokens[0:4] = [A,B,C,D/U]
P=6, r=4: 受影响块 = tokens[4:8] = [E,F,G,H]
```

### 2.2 何时发生

```
P % r == 0  →  分叉点在块边界  →  没有问题 ✅
P % r != 0  →  分叉点在块内部  →  问题 ❌

示例:
  r=128, P=256  →  256%128=0  →  ✅
  r=128, P=200  →  200%128=72 →  ❌
  r=4,   P=6    →  6%4=2     →  ❌
  r=4,   P=8    →  8%4=0     →  ✅
```

### 2.3 影响范围

| 数据 | 压缩粒度 | 来源 | 受影响？ |
|------|---------|------|:--:|
| `kv` | per-token | `linear_kv(hidden)` | ✅ 不受影响, per-token |
| `kv_compress` | per-r-token | `compressor(hidden)` | ❌ 受影响 |
| `indexer_k` | per-r-token | `indexer.kv_compressor(hidden)` | ❌ 受影响 |

## 3. 问题二：CP 交叉问题（CP>1 时引入）

### 3.1 问题描述

CP 把序列切到多个 rank 后，每个 rank 只持有自己那部分 hidden。受影响块需要的 raw hidden 可能在不同的 CP rank 上：

```
r=4, P=5, CP=2

受影响块 tokens[4:8]: [E, F, G, U]  (5//4*4=4, 到 8)
  prefix 部分: [E, F]    (P%r=5%4=1... 不对, 我算错了)
```

让我重新算：

```
r=4, P=5

受影响块: tokens[(5//4)*4 : (5//4+1)*4] = tokens[4:8]
  prefix 部分: token 4     (P%r=5%4=1 个 token: E)

不对, P=5 表示 prefix 有 5 个 token: [A,B,C,D,E]
token 4 是 E, token 5 才是分叉点。
受影响块 tokens[4:8]: [E, U, V, W]

  prefix 部分: [E]        (1 个, 在 rank 0)
  suffix 部分: [U, V, W]   (3 个, 3 个都在 rank 1? 看 CP 怎么切)
```

不管具体怎么切。核心是：**Reuser 需要一个完整窗口的 hidden（r 个）。但 CP 切分后，这些 hidden 可能不在同一个 rank。**

```
Reuser: P=5, r=4, CP=2
受影响块 tokens[4:8]: 需要 [hE, hU, hV, hW]

     rank 0                     rank 1
     ──────                     ──────
有:   hA, hB, hC, hD, hE        hF, hG, hH (provider 部分)
      hU                        hV, hW (reuser 部分)

谁都不全。rank 0 缺 hV, hW。rank 1 缺 hE, hU。
必须交换 raw hidden 才能凑满 4 个去跑 compressor。
```

### 3.2 两个条件必须同时满足

```
   条件一: CP 切分点落在受影响块内部
            ↓
   CP 把受影响块需要的 hidden 分散到了不同 rank
            ↓
   条件二: 分叉点在压缩块内部 (P % r != 0)
            ↓
   需要重建边界压缩块 → 需要全量 hidden → 被 CP 切碎 → 拿不到
   
   两个条件缺一不可:
   条件一 ∧ 条件二 → 问题
   ¬条件一 ∨ ¬条件二 → 无问题
```

图解：

```
                    P % r ≠ 0     P % r == 0
                    (分叉在块内)   (分叉对齐)
                         │              │
CP 切在块内  ────────────┼──────────────┼────
                         │    ❌        │   ✅
                         │  两个条件    │  CP交叉存在
                         │  同时满足    │  但压缩块不需要重建
                         │  需要额外    │  (compressor自己
                         │  通信        │  的gather处理)
                         │              │
CP 切在块边界 ───────────┼──────────────┼────
                         │    ✅        │   ✅
                         │  每个rank    │  最理想
                         │  独立重建    │  不需要任何
                         │              │  额外处理
```

## 4. Prefix Sharing 的 Compression 重建流程

当条件一和条件二同时满足时，Reuser 需要重建边界压缩块。完整流程如下：

```
┌─ Provider (CP rank 0) ──────────────────────────────────┐
│                                                          │
│ hidden: [hA..hE]    ← CP-local, 只有 rank 0 的部分       │
│    │                                                     │
│    ├── linear_kv → kv (per-token) → Hook: store(kv)      │
│    │                                                     │
│    └── compressor(hidden)                                │
│          │                                               │
│          ├── wkv(hidden) → kv_proj[0:5]                  │
│          │   (Linear 投影, 5 个 token)                   │
│          │                                               │
│          ├── wgate(hidden) → score[0:5]                  │
│          │                                               │
│          ├── gather_from_sp_cp(kv_proj, score)           │
│          │   → 从 rank 0+1 收集全量投影                  │
│          │                                               │
│          └── 分组加权求和 → [c0, c1]                     │
│              Hook: store(kv_compress=c0,c1)              │
│                                                          │
│ Hook 额外操作:                                           │
│   对于 Reuser (P=5, r=4):                                │
│     受影响块 tokens[4:8] 中 prefix 部分 = token[4]      │
│     需要存 hE 的投影值 wkv(hE)                           │
│     → 但这个中间值已经被 compressor 消费了               │
│     → 如果存 raw hidden hE: 可以, 在 rank 0             │
│     → 如果存投影 wkv(hE): 也可以, 在 rank 0             │
│     但 rank 0 拿不到 hV, hW (在 rank 1)                │
└──────────────────────────────────────────────────────────┘

┌─ Reuser (CP rank 1) ───────────────────────────────────┐
│                                                          │
│ hidden: [hV, hW, hX]    ← CP-local, suffix 部分         │
│    │                                                     │
│    ├── linear_kv → kv (3 token)                         │
│    │                                                     │
│    └── compressor(hidden)                                │
│          Compressor 内部:                                │
│            wkv([hV,hW,hX]) → kv_proj[3]   ← 只有 suffix │
│            gather_from_sp_cp → 补齐 CP 切掉的部分        │
│                                                          │
│            由于 reuser 的 hidden 是 suffix-only，        │
│            compressor 只看到 suffix token 的投影。       │
│            provider prefix [wkv(A)..wkv(E)]             │
│            不在 reuser forward 的任何变量里。            │
│                                                          │
│            分组:                                         │
│              c1 = f(wkv(V),wkv(W),wkv(X), ?)            │
│              ← suffix 不足 r 个, 切出 0 个完整块        │
│                                                          │
│ Hook 拿到 reuser.kv_compress = [] (空)                  │
│ 需要拼: cat(provider.kv_compress, [])                    │
│   = [c0_provider, c1_provider]                          │
│      ↑ c0 没问题         ↑ c1 是 provider 的             │
│      (全在 prefix 内)     压缩了 [E,F,G,H]               │
│                           但 reuser 需要 [E,U,V,W]        │
│                           内容错了 ❌                      │
│                                                          │
│   需要重建 c1'_reuser:                                   │
│     需要: wkv(E), wkv(U), wkv(V), wkv(W)                │
│     wkv(U),wkv(V),wkv(W) — reuser 自己有（suffix的）    │
│     wkv(E) — 只在 provider forward 中产生过              │
│     → provider 需要提前存                                │
└──────────────────────────────────────────────────────────┘
```

**关键矛盾**：Reuser 的 forward 输入只有 suffix hidden。compressor 内部的 `gather_from_sp_cp` 只补齐 CP 切掉的部分，无法补齐 PS trim 裁掉的 provider prefix 的 hidden。重建边界块需要 `wkv(prefix_tail) + wkv(suffix_head)`，其中 `wkv(prefix_tail)` 只在 provider forward 的 compressor 内部产生过，Hook 拿不到。

## 5. 可选方案

### 5.1 方案 A：改 prefix_len 对齐 r（推荐）

**做法**：**在 `plan.plan()` 之后、trim 之前**，将 prefix_len 向下取整到 r 的整数倍。顺序至关重要——先修正 plan，再 trim。如果先 trim 再改 plan，`kept_lengths_q` 等衍生字段已不一致。

```python
# 正确顺序:
plan = planner.plan(sequences)          # Step 1: 检测前缀

for i in range(plan.batch_size):        # Step 2: 修正 prefix_len
    if plan.prefix_lens[i] % compress_ratio != 0:
        plan.prefix_lens[i] = (plan.prefix_lens[i] // compress_ratio) * compress_ratio

trimmed = trim_batch(batch, plan)       # Step 3: 用修正后的 plan 裁剪
```

P=130, r=128 → P=128。丢掉 2 个 token 的共享。

**数值验证**：以 P=128(r=128), S=128 验证 expanded kv_compress 与 baseline 一致。

```
Provider: 256 tokens → kv_compress = [c0, c1]        (256//128=2)
  c0 = f(tokens 0-127)
  c1 = f(tokens 128-255)

Reuser: suffix 128 tokens → kv_compress = [c0_reuser]  (128//128=1)
  c0_reuser = f(tokens 128-255)  ← 即 suffix 的投影

Baseline (全序列 256 tokens, 无 trim):
  c0_baseline = f(tokens 0-127)    ← 和 provider c0 一致
  c1_baseline = f(tokens 128-255)  ← 和 reuser c0_reuser 一致

Expanded: cat(provider.kv_compress[:1], reuser.kv_compress[:1])
        = [c0, c0_reuser]
        = [c0_baseline, c1_baseline]  ✅ 与 baseline 一致
```

c0_reuser 与 c1_baseline 一致的原因：G2 attention 的 causal 性质保证 suffix token 的 hidden 只依赖它之前的 token（包括 prefix），不依赖 prefix 之后的 token（即 suffix 内部独立）。trim 后 suffix hidden = full-seq hidden 在 suffix 位置的值，因此 `wkv(suffix[i])` 完全相同 → compressor 输出一致。

**效果**：

```
分叉点永远在压缩块边界 (P % r == 0)
  → 没有"分叉在块内"的情况
  → provider 的 c1 和 reuser 的 c1 是完全不同的块
  → 不需要重建任何压缩块
  → comp_boundary 机制不需要
  → CP 切分是否跨界不影响（压缩块不需要重建）
  → 条件二永远不成立 → 问题不出现
```

**代价**：
- 丢掉最多 `r-1` 个 token 的共享（r=128 时最多 127，r=4 时最多 3）
- 对于 seq_len=4096+ 的预训练场景，127 token ≈ 3%，可以接受

**优点**：
- 一行代码
- 零额外存储
- 零额外通信
- 不改变 Store 结构
- 不改变 Hook 逻辑
- 不依赖 CP 配置

### 5.2 方案 B：实现 comp_boundary + 跨 rank 交换 hidden

**做法**：

1. Provider 在 Hook Store 时为每个 Reuser 保存受影响块的投影值（`wkv(x)` 在 gather 之后的中间结果）
2. CP>1 时，跨 rank 交换这些投影值，让 Reuser 所在 rank 凑齐受影响块的全部 r 个投影
3. Reuser 在 Hook Expand 时用凑齐的投影值重建边界压缩块，替换 provider 的

**效果**：
- 不丢任何共享 token
- 支持任意 prefix_len

**代价**：
- 新增 `StoredG2Activation` 字段（comp_boundary）
- Provider Hook 增加遍历 Reuser 逻辑
- Reuser Hook 增加重建逻辑
- CP>1 时增加一次跨 rank 通信（gather 受影响块的投影值）
- 代码复杂度显著增加
- 测试覆盖：CP=1 非对齐 prefix、CP=2 非对齐 prefix

### 5.3 方案 C：Patch Compressor，截取 gather 后的全量投影值

**核心思路**：compressor 内部 `gather_from_sp_cp(kv_proj)` 已经把各 rank 的投影值拉成全量了——每个 rank 都有完整的 `[total_tokens, 512]`。如果能在这一步截出来存到 side channel，Reuser 重建边界块时直接用全量投影值，**不需要任何额外通信**。

**做法**：在 attention patch（已有）基础上，增加一个 compressor patch：

```python
# setup/patches/mindspeed_deepseek4/compressor.py (新增)

def patch_compressor(original_forward_tnd):
    def patched_forward_tnd(self, x, start_pos, freqs_cis, packed_seq_params):
        # ── 原始代码: wkv + wgate ──
        kv = self.wkv(x)
        score = self.wgate(x)
        
        # ── 原始代码: gather ──
        kv = gather_from_sp_cp(kv)        # ← 此时全量!
        score = gather_from_sp_cp(score)  # ← 此时全量!
        
        # ═══ 截取: 存到 side channel ═══
        ctx = current_prefix_sharing_context()
        if ctx is not None:
            ctx._compressor_proj = (kv, score)   # 全量投影, 所有 rank 一致
        
        # ── 原始代码: 分组加权求和, 返回 ──
        ... (不变)
    
    return patched_forward_tnd
```

Reuser 重建时从 side channel 取：

```python
# Reuser Hook Expand:
# 1. provider 在 forward 时已经通过 compressor patch 把全量投影存到 ctx 了
# 2. Reuser 自己 forward 时 compressor patch 也会存
# 3. 但我们需要的是 provider forward 时存的投影（含 prefix hidden 的投影）
#    结合 Reuser 自己计算 suffix 的 wkv/wgate
#    → 这个需要仔细设计 side channel 的生命周期

# 简化版: 只存边界块需要的投影
# 在 compressor patch 里, 如果发现当前是 provider 身份:
#   受影响块索引 = P // r  
#   保存: kv[block_start:block_end]  (这 r 个 token 的投影)

# Reuser:
#   取 provider 存的受影响块全量投影 (r 个 token, 全是 provider 视角的)
#   但 Reuser 需要的是 suffix 侧 token 的投影 (U/V/W 的 wkv, 不是 F/G/H)
#   → provider 的投影里没有 suffix token...
```

**等一下**——这个问题又绕回来了。compressor 的 gather 之后是全量投影，但那是 **Provider forward 时的全量投影**——`[wkv(A)..wkv(H)]`。Reuser forward 时 compressor 再次 gather 后的全量投影是 `[wkv(A)..wkv(C), wkv(U)..]` 的混合——因为 Reuser 只传了 suffix hidden 进 forward。

**关键洞察**：Reuser forward 时，compressor 收到的 hidden 是 suffix-only 的。`wkv(suffix_hidden)` 只能产出 suffix 部分的投影。gather 补齐的也只是 **CP 切分造成的缺失**，不是 **PS trim 造成的缺失**（provider prefix 的 hidden 根本没传进 Reuser 的 forward）。

所以 provider 需要在 Hook Store 时提前存 `wkv(prefix_tail)`。Reuser 拿自己的 `wkv(suffix_head)`（gather 后的） + provider 存的 → 拼成完整窗口。在 CP>1 时，provider 存的 `wkv` 是 CP-local 的？不对——provider forward 时 compressor 已经 gather 了，provider Hook 可以取到全量 `wkv`。

**纠正**：Provider Hook 在 compressor 返回后，可以从 ctx side channel 取到 gather 后的全量 `wkv` 投影，切出 `tail_hidden` 对应的那部分存起来。Reuser 重建时用自己的 `wkv(suffix_head)`（也是 gather 后全量） + provider 存的投影 → 拼成完整窗口。

**CP>1 不需要额外通信**——两个阶段的 gather 各自做了自己的全量：

```
Provider forward → compressor gather → 全量 wkv([A..H])
  → Hook: 存 wkv(prefix_tail) = wkv([E,F])   ← 从全量中截, 不跨 rank

Reuser forward → compressor gather → 全量 wkv([A..C, U..])
  → Hook: 自己的 wkv(suffix_head) = wkv([U,V])   ← 从全量中截
  → cat(provider.wkv([E,F]), reuser.wkv([U,V])) = [r 个投影]
  → 加权求和 → 正确 c1'
```

**代价**：
- 第二个 PatchSpec（compressor patch）
- side channel（`ctx._compressor_proj`）
- 约 30 行额外代码

### 5.4 方案对比

| 维度 | 方案 A (align P) | 方案 B (raw hidden) | 方案 C (compressor patch) |
|------|:--:|:--:|:--:|
| 代码改动 | 1 行 | ~50 行 | ~30 行 |
| Patch 数量 | 1 (attention) | 1 (attention) | 2 (attention + compressor) |
| Store 改动 | 无 | 新增字段 | 新增字段 (或 side channel) |
| 额外存储 | 0 | `P%r × 4096` | `P%r × 512` (wkv投影) |
| CP>1 额外通信 | 无 | 需要 | **不需要** (各自从 gather 后截) |
| 共享 token 损失 | ≤ r-1 | 0 | 0 |
| 适用 CP | 全部 | CP=1 直接, CP>1 需通信 | 全部 (CP 透明) |

> **存储说明**：方案 B 和 C 的 `P%r` 个值来自 provider 的 prefix tail（受影响块中 prefix 侧的部分）。Reuser 的 suffix head（r-P%r 个）由 Reuser 自己计算，不需要 provider 存储。因此存储量按 `P%r` 计算，不是 `r`。

## 6. 建议

**推荐方案 A**。理由：

1. 改动极小（1 行），风险极低
2. 不引入新的存储、通信、Store 字段
3. 长序列下共享 token 损失可忽略（r=128 丢最多 127 token ≈ 3%）
4. 保持 PS 和 CP 的解耦——PS 不需要知道 CP 的外部细节
5. 方案 B/C 在 CP>1 时都面临同样瓶颈（跨 rank 交换中间值），复杂度相近
6. 如果未来需要支持任意 prefix_len，再讨论方案 B 或 C（彼时 CP 实战经验积累后设计会更成熟）
