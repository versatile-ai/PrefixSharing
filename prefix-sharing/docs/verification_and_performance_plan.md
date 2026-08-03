# PrefixSharing DeepSeek V4 — 验证与性能测试方案

> **日期**：2026-07-26（方案），2026-08-03（执行完成）
> **状态**：精度验证已完成，性能测试延后
> **实际执行**：与方案差异较大，见 §0.3。完整结果见 `docs/reports/precision_test_report.md`

## 0. 背景与目标

### 0.1 已完成

核心代码（Store、Hook、Topk、Patch、wrap_forward_step、**padded 路径**）全部就绪。

### 0.2 精度验证已通过（28/28）

| 阶段 | 配置数 | 结果 |
|------|:--:|------|
| 单卡等价性 (ratio=0/128, packed) | 18 | bitwise |
| Padded 多卡 (TP=1/2, CP=1/2) | 5 | bitwise / ~1e-4 |
| Packed expand (单层, THD) | 5 | allclose(1e-3) |

### 0.3 总路线（实际执行 vs 原计划）

```
原计划                         实际执行
Task 1-2 (单卡)     ✅ ─── 同计划，18/18 bitwise
Task 3 (TP)         ⏸️ ─── 单层 TP 需要 Megatron 完整初始化，改在 E2E 阶段
Task 4 (CP)         ⏸️ ─── 同上
Task 5 (E2E)        🔀 ─── DeepSeek4Model 构建阻塞(PP配置)，改为 padded 多卡 + packed expand
阶段 2 (性能)       ⏸️ ─── 延后
```

---

## 1. 精度验证（阶段 1）

### 1.0 验证方法论

#### 三基线验证体系

每个测试场景必须经过三层对比，顺序执行，不可跳过：

```
A: 无 PS patch（纯 baseline）
   → 原始 MindSpeed forward，prefix-sharing 代码完全不加载
   
B: PS patch 安装但 gate 强制 full-input fallback  
   → patch 已挂载到 DeepSeek4SelfAttention.forward
   → ctx 为空（context 未设置）时 patched_forward 直接调 original_forward
   → 等价于：prerequisite 是 patch 本身不引入误差
   
C: PS patch 激活，走 suffix-only 优化
   → ctx 就位，provider store + reuser expand 全部执行
   → 等价于：PS 优化本身不引入误差
```

**验证顺序必须是 A≈B，再 B≈C**。如果 B 就不等于 A，说明 patch 本身（import、fork forward、中间变量）引入了误差——这不应该发生，也不是 PS 优化的问题。只有 A≈B 通过后，C 的偏差才能归因于 PS 优化路径。

**A≈B 的"patch no-op 模式"验证**：

```python
# B 路径：patch 安装，但 context 未激活 → 命中 patched_forward L37
# → 直接调 original_forward，输出应与 A 完全一致（bitwise）
# 在 1.2 中作为快速 smoke test 运行，不需要遍历全部参数组合
```

**B≈C 的"PS 优化等价"验证**：遍历全部参数组合，按以下判定标准对比。

**等价性定义**：同一模型、同一输入、同一随机种子下，路径 C（PS 激活）与路径 B（gate fallback = 全量 forward）的输出必须一致。

**对比粒度**：

| 级别 | 对比对象 | 判定标准 |
|------|---------|---------|
| Attention 输出 | `o` tensor in sparse_attention | `torch.allclose(atol=1e-5, rtol=1e-4)` |
| Layer 输出 | hidden_states after each layer | `torch.allclose(atol=1e-5, rtol=1e-4)` |
| Loss | scalar loss value | `abs(ps_loss - baseline_loss) < 1e-8` |
| Gradient | per-parameter grad | `torch.allclose(atol=1e-5, rtol=1e-4)` |

**为什么不是 bitwise**：DeepSeek V4 使用 `sparse_flash_mla` 融合 NPU kernel，PS 路径扩展 KV 后 kernel 内浮点运算顺序与 baseline 不同（cat 后的内存布局差异），bitwise 等价不可行。`atol=1e-5` 是合理的工程精度。

**随机权重即可**：因为是对比"同一模型、同一输入、不同路径"的相对一致性，权重值本身不影响结论。所有测试使用随机权重。

#### 逐级数值检查点

只比最终 loss 会掩盖问题——第一个不一致的点才是根因。按优先级从高到低逐级断言：

```
优先级 1 — 计划层（setup 阶段断言，非跨路径对比）：
  这些检查在 test setup 完成后、forward 之前执行。
  plan 是 A/B/C 三条路径共用的同一个实例（plan_A is plan_C），
  不需要"对比"两个 plan——只需验证字段符合预期值。

  1. plan.prefix_lens == expected_prefix_lens
  2. plan.provider_index == expected_provider_index
  3. plan.kept_lengths_q == expected_suffix_lengths
  4. plan.prefix_last_restore 的 label_value 指向正确的 reuser suffix 第一个 token
     （reuser 的 prefix-last 恢复位置：suffix 的第一个 token 的 label，不是 provider 的）

优先级 2 — 前处理层（期望 zero-diff，同一段代码）：
  5. linear_q output: allclose(ps_q_proj, baseline_q_proj)    # Q 路径不变
  6. RoPE 前 Q/K:     allclose(ps_q_pre_rope, baseline_q_pre_rope)
  7. RoPE 后 Q:       allclose(ps_q_roped, baseline_q_roped)  # Q 计算路径完全相同
  8. RoPE 后 KV:      allclose(ps_kv_roped, baseline_kv_roped) # KV 全量 gather 后一致
  9. kv_compress:     allclose(ps_cmp, baseline_cmp)           # compressor 输入不变

优先级 3 — 注意力层（允许 atol=1e-5，kernel 差异）：
  10. Phase 2 topk:   allclose(ps_topk, baseline_topk, atol=1e-5)
                      # ratio=128: 纯位置重算，可能与 baseline 不同
                      # ratio=4: 重新打分，期望覆盖原值
  11. attention logits: allclose(ps_attn_scores, baseline_attn_scores, atol=1e-5)
  12. attention output: allclose(ps_attn_o, baseline_attn_o, atol=1e-5)
  13. suffix output:  allclose(ps_o_suffix, baseline_o_suffix, atol=1e-5)
                      # 只对比 reuser suffix 部分（prefix 部分不存在于 PS 输出中）
  14. 2nd RoPE 后:    allclose(ps_o_rotated, baseline_o_rotated, atol=1e-5)

优先级 4 — 输出层：
  15. core_attn_out:  allclose(ps_final, baseline_final, atol=1e-5)
  16. bias:           allclose(ps_bias, baseline_bias, atol=1e-5)
```

**断言策略**：按优先级顺序执行。如果优先级 1 失败，停止后续检查并报告根因。优先级 2 失败可能表明 patch fork 代码与原始 forward 有差异（例如遗漏了某个 reshape / transpose）。优先级 3 失败才是 PS 优化路径本身的问题。

检查点 1-8 和 16 应在每个测试用例中执行。检查点 9-15 需要侵入 patched_forward 内部捕获中间张量——在 1.2 阶段通过抓取 patch 内部关键中间值实现（patch 代码中加入可选的 `capture_intermediates` 开关）。

### 1.1 单卡功能补测 (Mac)

**目标**：补充 Mac 环境下缺失的边界场景，不依赖 NPU。

**当前缺口**：
- Multi-provider / multi-reuser 混合 batch 的 store/expand 正确性
- `pass_through` 路径（既非 provider 也非 reuser）的覆盖
- `_adjust_cu_seqlens_for_batch` 对 `cu_seqlens_cmp_kv` 的偏移测试（现有测试只覆盖了 `cu_seqlens_kv`）
- `_compute_cmp_lengths` 对 mismatch 的断言触发（当前只有 happy path）

**新增测试文件**：`tests/unit_test/test_g2_edge_cases.py`

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_multi_provider_multi_reuser` | 2 provider + 3 reuser，验证各自 store/expand 正确 |
| 2 | `test_pass_through_sequence` | 非 provider 非 reuser 的序列 pass-through 不变 |
| 3 | `test_cu_seqlens_cmp_kv_adjust` | cu_seqlens_cmp_kv 按 compress_ratio 正确偏移 |
| 4 | `test_cu_seqlens_kv_padded_priority` | 同时有 cu_seqlens_kv_padded 和 cu_seqlens_kv 时，优先使用 padded 版本（验证 `g2_attention_utils.py:130` 的 priority 逻辑） |
| 5 | `test_compute_cmp_lengths_mismatch` | 长度不匹配时 assert 正确触发 |
| 6 | `test_store_after_close` | store close 后操作抛 RuntimeError |
| 7 | `test_empty_batch` | batch_size=0 时不报错 |
| 8 | `test_prefix_len_zero_reuser` | prefix_len=0 的 reuser（对齐后可能发生）处理正确 |
| 9 | `test_max_prefix_len` | prefix_len == original_len（reuser 无 suffix）的边界 |

**预期**：9 个新增 Mac 测试全部通过，不新增 NPU 依赖。

### 1.2 单卡精度等价性 (NPU 单卡)

**目标**：在 NPU 单卡上验证 ratio=128 和 ratio=4 路径的 forward/backward 输出与 baseline 一致。

**测试模型**：1 层 `DeepSeek4SelfAttention`（随机权重，seqlen=512）

- ratio=128：`compress_ratios[0] = 128`
- ratio=4：`compress_ratios[0] = 4`（含 DSA Indexer）

**测试文件**：`tests/precision/test_single_card_equivalence.py`

**参数矩阵**：

```python
@pytest.mark.parametrize("compress_ratio", [128, 4])
@pytest.mark.parametrize("prefix_len_pct", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("batch_composition", [
    "1p1r",         # 见下方定义
    "1p3r",         # 见下方定义
    "2p3r_mixed",   # 见下方定义
])
@pytest.mark.parametrize("checkpoint", ["forward", "backward"])
```

**batch_composition 详细定义**（seqlen=512，ratio=128，prefix_len_pct=0.5 → P=256）：

```
"1p1r":
  seq0(全量=512)           provider, prefix_len=0
  seq1(前缀=seq0[:256])   reuser,  prefix_len=256, suffix_len=256
  → 总 packed: seq0=512 + seq1=256(suffix) = 768

"1p3r":
  seq0(全量=512)           provider, prefix_len=0
  seq1(前缀=seq0[:256])   reuser,  prefix_len=256, suffix_len=256
  seq2(前缀=seq0[:256])   reuser,  prefix_len=256, suffix_len=256
  seq3(前缀=seq0[:256])   reuser,  prefix_len=256, suffix_len=256
  → 总 packed: seq0=512 + 3×256(suffix) = 1280
  → transitive reuse: seq1 扩展后回存，seq2/seq3 从 seq1 或 seq0 取

"2p3r_mixed":
  seq0(全量=512)           provider_A, prefix_len=0
  seq1(全量=512)           provider_B, prefix_len=0  (与 seq0 不同 prefix)
  seq2(前缀=seq0[:384])   reuser,  prefix_len=384, suffix_len=128  ← 复用 seq0 更长 prefix
  seq3(前缀=seq0[:256])   reuser,  prefix_len=256, suffix_len=256  ← 复用 seq0 较短 prefix
  seq4(前缀=seq1[:128])   reuser,  prefix_len=128, suffix_len=384  ← 复用 seq1 不同 provider
  → 覆盖：不同 provider 有不同 prefix 长度、同一 provider 被不同 prefix 长度复用
```

即：2 × 3 × 3 × 2 = **36 个参数组合**。

每个组合的核心逻辑：

```python
def _run_equivalence_test(ratio, prefix_len_pct, batch_comp, checkpoint):
    # 0. 构造输入：N 条序列，指定共享前缀关系（见上方 batch_composition 定义）
    # A. Baseline: original_forward（PS 代码完全不加载）
    # B. Patch no-op: 安装 patched_forward 但 ctx 为空 → 应 bitwise 等于 A
    # C. PS 激活: ctx 就位 → 按检查点 1-16 逐级与 B 对比
    #    检查点 1-8: 预期 zero-diff
    #    检查点 9-16: 预期 allclose(atol=1e-5)
    pass
```

**验证项**：

| 场景 | 验证目标 | 关键风险点 |
|------|---------|-----------|
| ratio=128 + forward | attention output 全等 | KV cat 后 sparse_flash_mla 计算一致性 |
| ratio=128 + backward | grad 全等 | prefix KV grad 从 reuser 正确回传 |
| ratio=4 + forward | attention output 全等 | expanded indexer_k 参与 forward_with_scores_compress 正确 |
| ratio=4 + backward | grad 全等 | DSA Indexer 的 grad 流经重新打分路径 |
| transitive reuse | 三层 reuser chain output 一致 | 中间 reuser 的 expanded data 正确回存 |
| different prefix_lens | 不同 P 值均正确 | P 对齐 compress_ratio 后 cat 边界正确 |

### 1.3 多卡 TP 精度 (NPU 多卡)

**目标**：验证 TP>1 时 PS 的 tensor-parallel 正确性。

**关键点**：TP 下 Q/KV 被切分到不同 rank，`gather_from_tensor_model_parallel_region` 和 `gather_from_sp_cp` 的调用顺序和范围需要验证。

**TP 特有的风险**：
- `packed_seq_params` 中的 `cu_seqlens` 是 TP-local 还是全局？
- `tp_rank` 放入 `PrefixActivationSlotId` 的隔离是否正确？
- TP 下 `kv_allgather` 和 `sequence_parallel` 的行为
- `g2_attention.py:98-99` 中 `gather_from_sp_cp(kv)` 之后的 kv 在各 TP rank 上是否一致

**测试文件**：`tests/precision/test_tp_equivalence.py`

**参数矩阵**：

```python
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("compress_ratio", [128, 4])
@pytest.mark.parametrize("batch_composition", ["1p1r", "1p3r"])
@pytest.mark.parametrize("sequence_parallel", [True, False])
```

共：3 × 2 × 2 × 2 = **24 个参数组合**。

**验证项**：

| # | 验证目标 |
|---|---------|
| 1 | TP 各 rank 的 attention output 与单卡 baseline 一致 |
| 2 | TP 各 rank 的 store 内容一致（gather 后全量） |
| 3 | TP + sequence_parallel 组合下 q_len 计算正确 |
| 4 | TP 各 rank 的 grad 一致且与 baseline 一致 |

### 1.4 多卡 CP 精度 (NPU 多卡)

**目标**：解除 `supported_cp_size=1` 限制，验证 CP>1 时 PS 的正确性。

**前置改动**：
1. 移除 `config.py:178` 的 CP size 硬门禁（改为 warn 或直接移除检查）
2. 确认 `packed_batch_layout` 在 CP 切分前创建（全局 layout）
3. 确认 `_split_by_cu_seqlens` 用全局 `padded_lengths` 拆分全量 KV

**验证清单**（对应 `cp_ps_design.md` §7 的 5 个延后项）：

| # | 延后项 | 验证方法 |
|---|--------|---------|
| 1 | CP + TP 组合 | `cp_size=2, tp_size=2` 下 forward 等价性 |
| 2 | CP + PP 组合 | `cp_size=2, pp_size=2` 下 store 生命周期不跨 PP stage |
| 3 | 压缩边界 × CP 交叉 | P%r≠0 但在 align 后 P%r=0（方案 A 已处理） |
| 4 | Ring Attention 替代 | 不在本轮范围（mark as future） |
| 5 | ratio=4 Indexer CP 行为 | 容器内打印 `key_index.shape` 验证 gather 后全量 |

**测试文件**：`tests/precision/test_cp_equivalence.py`

**参数矩阵**：

```python
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("compress_ratio", [128, 4])
@pytest.mark.parametrize("batch_composition", ["1p1r", "1p3r", "cross_rank"])
```

其中 `cross_rank` 场景：reuser 的 suffix 被 CP 切到多个 rank，验证 gather 后每个 rank 都能独立完成 expand 且结果一致。

共：2 × 2 × 3 = **12 个参数组合**。

**关键验证**：

| # | 验证目标 |
|---|---------|
| 1 | CP>1 时每个 rank 的 store 内容完全一致 |
| 2 | CP>1 时每个 rank 的 expand 结果完全一致 |
| 3 | Reuser 的 attention output 与 baseline 一致 |
| 4 | ratio=4 Indexer 的 `key_index` 在 gather 之后 shape 正确 |
| 5 | CP + TP 交叉场景的 output 一致性 |

### 1.5 端到端精度红线 (NPU 8 卡)

**目标**：完整 43 层模型（关闭 MoE），TP=1/PP=1/CP=1，验证 loss/logprob/grad 与 baseline 完全一致。

**测试文件**：`tests/precision/test_e2e_equivalence.py`

**前置条件**：
- 1.2 全部通过
- 1.3 全部通过
- 1.4 全部通过（或至少 CP=1 下的 TP 全通过）
- `wrap_forward_step` 可正确挂载到 MindSpeed pretrain 入口

**对比方法**：

```python
# 同一 batch 数据，跑两次：
# Run 1: ENABLE_PREFIX_SHARING=0 → loss_baseline, grad_baseline
# Run 2: ENABLE_PREFIX_SHARING=1 → loss_ps, grad_ps

assert abs(loss_ps - loss_baseline) < 1e-8
for name, param in model.named_parameters():
    assert torch.allclose(param.grad, baseline_grad[name], atol=1e-5)
```

**验证项**：

| # | 场景 | 验证目标 | 精度标准 |
|---|------|---------|---------|
| 1 | 无共享前缀 batch | PS 路径 == baseline（自动退避） | loss/grad 一致 |
| 2 | 有共享前缀 + ratio=128 层 | loss 一致 | abs(diff) < 1e-8 |
| 3 | 有共享前缀 + 混合 ratio | loss 一致（ratio=0/4/128 混合层） | abs(diff) < 1e-8 |
| 4 | Per-token logprob | 逐 token logprob 一致 | allclose(atol=1e-5) |
| 5 | Gradient norm | grad_norm 一致 | abs(diff) < 1e-8 |
| 6 | 多 micro-batch 累积 | 跨 micro-batch 的 grad 累积一致 | all param grads 一致 |

### 1.6 精度验证用例集产出

**目录结构**：

```
tests/
├── precision/                              # 新增
│   ├── conftest.py                         # 共享 fixtures（NPU 初始化、args、model 构造）
│   ├── test_single_card_equivalence.py     # 1.2
│   ├── test_tp_equivalence.py              # 1.3
│   ├── test_cp_equivalence.py              # 1.4
│   └── test_e2e_equivalence.py             # 1.5
└── unit_test/
    ├── test_g2_edge_cases.py               # 1.1 新增
    └── ...                                  # 现有 216 单测
```

**共享 fixtures (`conftest.py`)**：

```python
# 所有 NPU 精度测试共用的初始化逻辑
@pytest.fixture(scope="session")
def init_distributed():
    """初始化 torch.distributed + Megatron parallel_state + args"""
    ...

@pytest.fixture
def make_attention_module(compress_ratio):
    """构造单层 DeepSeek4SelfAttention（随机权重）"""
    ...

@pytest.fixture
def make_batch_inputs(batch_composition, prefix_len_pct, seqlen):
    """构造指定 prefix 关系的 batch 输入"""
    ...
```

**运行方式**：

```bash
# 单卡
PYTHONPATH=prefix-sharing pytest tests/precision/test_single_card_equivalence.py -v

# TP
torchrun --nproc_per_node=4 -m pytest tests/precision/test_tp_equivalence.py -v

# CP
torchrun --nproc_per_node=4 -m pytest tests/precision/test_cp_equivalence.py -v

# E2E
torchrun --nproc_per_node=8 -m pytest tests/precision/test_e2e_equivalence.py -v
```

---

## 2. 性能测试（阶段 2）

> **前提**：阶段 1 全部通过。性能测试不验证正确性（正确性已在阶段 1 保证），只回答"省了多少"和"开销在哪"。
>
> **平台**：所有性能测试在 **NPU (Ascend)** 上运行，不是 GPU (CUDA)。API 使用 `torch.npu.*` 和 `torch_npu.*`，不是 `torch.cuda.*`。

### 2.0 观测手段体系

性能测试有三个粒度的观测方法，从粗到细：

#### 层 1：Python 级计时（粗粒度，最基础）

```python
import time

def measure_timing(fn, warmup=10, iterations=100):
    for _ in range(warmup):
        fn()
    if torch.npu.is_available():
        torch.npu.synchronize()   # ← NPU，不是 torch.cuda
    
    times = []
    for _ in range(iterations):
        if torch.npu.is_available():
            torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.npu.is_available():
            torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    
    return {
        "median_ms": statistics.median(times) * 1000,
        "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000,
    }
```

**适用**：端到端 overhead 测量、阶段级耗时（store / expand / cat / topk 各环节占比）。
**局限**：只能看到 Python 侧耗时，看不到 NPU kernel 内部。

#### 层 2：torch_npu.profiler（中粒度，kernel 级）

```python
import torch_npu

def profile_with_torch_npu(fn, output_dir="./profiler_output"):
    """使用 torch_npu.profiler 采集 NPU kernel 级耗时。"""
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,    # CPU 端 op
        torch_npu.profiler.ProfilerActivity.NPU,    # NPU 端 kernel
    ]
    
    with torch_npu.profiler.profile(
        activities=activities,
        record_shapes=True,          # 记录 tensor shape
        profile_memory=True,         # 记录显存分配
        with_stack=True,             # 记录 Python 调用栈
    ) as prof:
        fn()
    
    # 导出 Chrome Trace 格式（可在 chrome://tracing 查看）
    prof.export_chrome_trace(f"{output_dir}/trace.json")
    
    # 导出表格格式（按耗时排序）
    table = prof.key_averages().table(
        sort_by="self_npu_time_total", row_limit=20)
    print(table)
    
    return prof
```

**适用**：定位哪个 NPU kernel 最慢、是否有 unexpected kernel launch、内存峰值在哪。
**局限**：`torch_npu.profiler` 需要 `torch_npu` 包（CANN 安装后内置）。

#### 层 3：msprof（细粒度，硬件级）

```bash
# msprof 是 CANN toolkit 自带的硬件 profiling CLI
# 可以采集 NPU 的 AICore/AIVector/bandwidth 利用率和 stall 原因

msprof --application="pytest tests/precision/test_single_card_equivalence.py::test_ratio128_1p1r_forward" \
       --output=/tmp/profiler_output \
       --npu-usage=on \
       --aicore-usage=on \
       --aic-metrics=MemoryBandwidth,L2Cache \
       --profile-level=level1
```

**适用**：深度优化阶段——分析算子瓶颈（是 compute-bound 还是 memory-bound）、aicore 利用率、HBM 带宽利用率。
**局限**：需要 CANN 环境变量配置妥当、仅在 Ascend 硬件可用、输出文件需用 `mindstudio-pro` 或 `msprof` 打开解析。

#### 层 4：MindStudio Profiler（综合分析，GUI 工具）

Ascend 官方 IDE MindStudio 内置 Profiler，可以同时展示：
- Timeline view（kernel 执行时间线）
- Flame graph（调用栈火焰图）
- Operator analysis（算子级耗时排序）
- Memory analysis（显存分配时间线）
- Communication analysis（HCCL 通信耗时）

**适用**：多卡 E2E 性能调优阶段，需要可视化分析时。
**局限**：需要 GUI 环境（VM/Mac 远程桌面到 NPU 服务器），不适合 CI 自动化。

#### 观测手段选择策略

| 场景 | 推荐手段 | 原因 |
|------|---------|------|
| 1.2-1.5 精度测试中的 overhead 初判 | 层 1：`time.perf_counter` | 简单快速，判断 Hook 开销 |
| 2.1 算子级——定位具体慢点 | 层 2：`torch_npu.profiler` | kernel 级精度，可导出 chrome trace |
| 2.2 流程级——吞吐分析 | 层 1 + 层 2 结合 | 先看大指标，异常时 profiling |
| 深度优化——分析 aicore 利用率 | 层 3：`msprof` | 硬件级指标 |
| 最终优化报告——可视化 + 决策 | 层 4：MindStudio | 综合展示 |

### 2.1 算子级基准

**目标**：测量 PS 每个环节的耗时和内存开销，定位性能瓶颈。

**测试脚本**：`tools/benchmark/operator_benchmark.py`

**测量方法论**：

```python
def measure(fn, warmup=10, iterations=100):
    """标准测量协议：warmup → 多次迭代 → 取 median。
    
    平台：NPU (Ascend)。所有同步/内存 API 使用 torch.npu.*。
    """
    # Warm-up：消除首次 kernel launch / cache miss 的影响
    for _ in range(warmup):
        fn()
    
    if torch.npu.is_available():
        torch.npu.synchronize()
    
    times = []
    for _ in range(iterations):
        if torch.npu.is_available():
            torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.npu.is_available():
            torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    
    # 使用 median 而非 mean——避免 outlier（GC、OS 调度、NPU driver 抖动）扭曲结果
    import statistics
    return {
        "median_ms": statistics.median(times) * 1000,
        "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000,
        "p99_ms": sorted(times)[int(len(times) * 0.99)] * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }
```

**为什么 median 而非 mean**：NPU kernel launch 时间受 driver 调度和硬件队列深度影响，偶尔出现数倍于正常的耗时。mean 会被 outlier 拉高，median 反映真实稳态性能。

**测量维度**：

| 环节 | 测量内容 | 关键参数 |
|------|---------|---------|
| Store | `_g2_store_with_kwargs` 耗时 | prefix_len, batch_size |
| Split | `_split_by_cu_seqlens` 耗时 | packed_total_tokens, batch_size |
| Cat | `torch.cat([provider_prefix, reuser_suffix])` 耗时 | P, S |
| Topk ratio=128 | `get_compress_topk_idxs` 耗时 | expanded_seqlen |
| Topk ratio=4 | `forward_with_scores_compress` 耗时 | expanded K size |
| cu_seqlens | `_adjust_cu_seqlens_for_batch` 耗时 | batch_size |
| 总 Hook | `_g2_kv_store_or_expand` 端到端耗时 | 所有参数组合 |
| 内存 | `torch.npu.max_memory_allocated()` / `torch.npu.memory_stats()` PS vs baseline | model_size, batch_size |

**输出**：以 Markdown 表格 + JSON 输出每个环节的 median/p95/p99 耗时和占比。

**参数扫描**：

```python
seq_lens = [512, 1024, 2048, 4096, 8192]
prefix_pcts = [0.25, 0.5, 0.75]
compress_ratios = [128, 4]
batch_sizes = [2, 4, 8]
```

### 2.2 流程级对比

**目标**：完整 pretrain 流程中 PS 开启 vs 关闭的吞吐/耗时对比。

**测试脚本**：`tools/benchmark/pipeline_benchmark.py`

**测量方法**：同上 `measure()` 协议（warmup=10, iterations=100, median）。

**测量维度**：

| 指标 | 说明 |
|------|------|
| Tokens/sec | 吞吐对比（PS vs baseline） |
| Forward time | 单步 forward 耗时 |
| Backward time | 单步 backward 耗时 |
| Peak memory | 峰值显存 |
| MLP 计算量 | reuser prefix Q 被裁后省了多少 MLP FLOPs |
| Attention 计算量 | 理论上 Q×KV 不变，但 sparse kernel 实际耗时是否变化 |

**对比矩阵**：

```python
cp_sizes = [1, 2, 4]
tp_sizes = [1, 2]
model_sizes = ["1_layer", "43_layers_no_moe"]
seq_lens = [2048, 4096, 8192]
```

**输出**：PS vs baseline 的耗时、吞吐、显存的百分比对比，以 Markdown 表格输出。

### 2.3 性能测试用例集产出

```
tools/
├── benchmark/
│   ├── operator_benchmark.py       # 算子级
│   ├── pipeline_benchmark.py       # 流程级
│   ├── report.py                   # 汇总对比输出
│   └── README.md                   # 运行说明
```

**运行方式**：

```bash
# 算子级
python tools/benchmark/operator_benchmark.py --ratio 128 --seqlen 4096

# 流程级 (8 卡)
torchrun --nproc_per_node=8 tools/benchmark/pipeline_benchmark.py \
    --model 43_layers --seqlen 4096 --cp 2
```

---

## 3. 阶段门禁与报告

### 3.1 门禁规则

每个阶段完成后，**必须先输出测试报告**，报告审核通过后，才能进入下一阶段。不允许跳过任何阶段。

```
阶段 i → 跑测试 → 输出报告 → 审核通过 → 阶段 i+1
                                    ↓ 不通过
                              修改 + 重新跑
```

**报告提交物**：每个阶段产出独立的 Markdown 报告文件，放在 `prefix-sharing/docs/reports/`。

### 3.2 报告模板

每个报告必须包含以下小节：

```markdown
# [阶段名称] 测试报告

> **日期**：YYYY-MM-DD
> **环境**：NPU 服务器 / 容器 / 机型 / torch 版本 / CANN 版本
> **执行人**：...

## 1. 测试概要

| 指标 | 值 |
|------|----|
| 总用例数 | N |
| 通过 | N_pass |
| 失败 | N_fail |
| 跳过 | N_skip |
| 执行时间 | T min |
| 通过率 | N_pass/N × 100% |

## 2. 环境信息

- 服务器 IP / 容器名
- torch 版本 (`torch.__version__`)
- torch_npu 版本
- CANN 版本 (`/usr/local/Ascend/ascend-toolkit/latest/version.cfg`)
- compress_ratios / num_layers / seqlen 等关键模型参数

## 3. 逐用例结果

| # | 用例名 | 参数 | 结果 | 关键指标 | 备注 |
|---|--------|------|:--:|------|------|
| 1 | test_xxx | ratio=128, P=256 | PASS | allclose max_diff=3e-6 | |
| 2 | test_yyy | ratio=4, P=128 | FAIL | plan 不一致 | 根因: ... |

## 4. 失败用例分析

（每个失败的用例）
- **根因**：
- **影响范围**：
- **修复方案**：

## 5. 精度红线状态

| 指标 | 标准 | 实际 | 判定 |
|------|------|------|:--:|
| A≈B patch no-op | bitwise | - | ✅/❌ |
| B≈C 检查点 1-8 | zero-diff | - | ✅/❌ |
| B≈C 检查点 9-16 | allclose(atol=1e-5) | max_diff=... | ✅/❌ |
| forward output | allclose(atol=1e-5) | - | ✅/❌ |
| backward grad | allclose(atol=1e-5) | - | ✅/❌ |
| loss | abs(diff) < 1e-8 | - | ✅/❌ |

## 6. 遗留问题

（此处记录已知但暂不阻塞进入下一阶段的问题，必须有明确理由）
```

### 3.3 准出条件

- 全部用例通过（pass rate = 100%）
- 精度红线全部达标
- 无未解释的失败用例
- 遗留问题（如有）有明确的不阻塞理由

### 3.4 工作排期

| 阶段 | 内容 | 预估工作量 | 依赖 | 准出 |
|------|------|:--:|------|------|
| 1.1 | 单卡功能补测 (Mac) | 0.5 天 | 无 | 报告 `report_1_1_mac.md` |
| 1.2 | 单卡精度等价性 (NPU 单卡) | 2 天 | 报告 1.1 | 报告 `report_1_2_single_card.md` |
| 1.3 | 多卡 TP 精度 (NPU 多卡) | 1.5 天 | 报告 1.2 | 报告 `report_1_3_tp.md` |
| 1.4 | 多卡 CP 精度 (NPU 多卡) | 1.5 天 | 报告 1.2 | 报告 `report_1_4_cp.md` |
| 1.5 | 端到端精度红线 (NPU 8 卡) | 2 天 | 报告 1.3 + 1.4 | 报告 `report_1_5_e2e.md` |
| 2.1 | 算子级性能基准 | 1 天 | 报告 1.5 | 报告 `report_2_1_operator.md` |
| 2.2 | 流程级性能对比 | 1 天 | 报告 2.1 | 报告 `report_2_2_pipeline.md` |
| **合计** | | **9.5 天** | | **7 份报告** |

---

## 4. 关键风险

| 风险 | 影响阶段 | 缓解措施 |
|------|:--:|------|
| NPU sparse_flash_mla 在 expanded KV 下精度偏差超预期 | 1.2 | 放宽 atol 到 1e-4；分析具体差异来源 |
| DSA Indexer 独立实例化失败（需要完整 43 层 config） | 1.2 | 用最小完整模型（2-3 层含 Indexer）代替单层 |
| TP 下 packed_seq_params 结构不同 | 1.3 | 容器内先 print 字段确认 |
| CP + TP 组合时 gather 行为与预期不符 | 1.4 | 先单独 CP、单独 TP，再组合 |
| 8 卡 E2E 环境不稳定（MindSpeed 版本、容器、NPU 驱动） | 1.5 | 先用 ratio=0 跑通 E2E 框架，再扩展到 128/4 |
| 性能测试需要稳定的 NPU profiling 工具 | 2 | 优先用层 1（`time.perf_counter` + `torch.npu.synchronize`）做粗粒度；`torch_npu.profiler` 需要确认 CANN 版本支持；`msprof` 需要 MindStudio 或 CANN toolkit 安装 |
| NPU kernel launch 调度抖动导致性能数据波动大 | 2 | 增加 iterations（100→200），报告 median + p95 + p99，对波动大的用例标注变异系数 |
