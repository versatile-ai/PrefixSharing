# Precision Verification — Task Breakdown

> **日期**：2026-07-26（方案），2026-08-03（被实际执行替代）
> **状态**：⛔ 作废——实际执行路径与方案差异较大，见 `docs/reports/precision_test_report.md` 和 `project_status.md`
> **父文档**：`verification_and_performance_plan.md`
> **范围**：阶段 1（精度验证）的任务拆分和实现细节（原始方案，仅供历史参考）
> **平台矩阵**：Mac (CPU PyTorch) → NPU 单卡 → NPU 多卡 (TP/CP) → NPU 8 卡

## 依赖关系

```
Task 0: infra (shared fixtures + capture_intermediates)
  └── Task 1 (1.1 Mac edge cases)  ← 可与 Task 0 并行
        │
        └── Task 2 (1.2 单卡精度)
              ├── Task 3 (1.3 TP)
              │     │
              │     └── Task 5 (1.5 E2E)
              │
              └── Task 4 (1.4 CP) ──┘
```

## 总览

| Task | 阶段 | 环境 | 用例数 | 新增文件 | 产出报告 |
|------|:--:|------|:--:|------|------|
| 0 | 基础设施 | Mac+NPU | — | 2 | — |
| 1 | 1.1 功能补测 | Mac | 9 | 1 | `report_1_1_mac.md` |
| 2 | 1.2 单卡精度 | NPU 单卡 | 36+2 | 2 | `report_1_2_single_card.md` |
| 3 | 1.3 TP 精度 | NPU 多卡 | 24 | 1 | `report_1_3_tp.md` |
| 4 | 1.4 CP 精度 | NPU 多卡 | 12+5 | 1 | `report_1_4_cp.md` |
| 5 | 1.5 E2E 精度 | NPU 8 卡 | 6 | 1 | `report_1_5_e2e.md` |

---

## Task 0：基础设施

### T0.1 shared fixtures (`conftest.py`)

**文件**：`tests/precision/conftest.py`（新建）

**内容**：所有 NPU 精度测试共用的 fixtures。

```python
# conftest.py

import pytest
import torch

# ── session scope: distributed init ──────────────────────────────────

@pytest.fixture(scope="session")
def init_distributed():
    """单卡: torch.distributed.init_process_group(backend=hccl, world_size=1, rank=0)
       多卡: 由 torchrun 自动初始化"""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="hccl", world_size=1, rank=0)
    from megatron.core import parallel_state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        )
    yield
    # cleanup: leave to next test / torchrun

# ── model fixtures ───────────────────────────────────────────────────

@pytest.fixture
def megatron_args():
    """构造 Megatron args 的最小集合（通过 setattr 注入全局 args）。

    关键字段（摘自已实现的 DeepSeek4SelfAttention.__init__）:
      - args.qk_head_dim = 512
      - args.rope_head_dim = 64
      - args.q_lora_rank / o_lora_rank
      - args.compress_ratios
      - args.hidden_size / num_attention_heads / o_groups / g2_window_size
    """
    ...

@pytest.fixture
def make_attention_module(megatron_args, compress_ratio):
    """构造单层 DeepSeek4SelfAttention（随机权重）。"""
    ...

@pytest.fixture
def make_batch_inputs():
    """根据 batch_composition 描述构造输入张量。"""
    ...
```

### T0.2 patch 增加 `capture_intermediates` 开关

**文件**：`setup/patches/mindspeed_deepseek4/attention.py`（修改）

**内容**：在 patched_forward 中加入可选的中间张量捕获，以支持检查点 9-15（Phase 2/3/4 中间值对比）。

```python
def patched_forward(self, hidden_states, ...):
    # ... Phase 1-3 (unchanged) ...
    
    # ── Optional: capture intermediates ──
    captured = {}  # type: dict[str, torch.Tensor] | None
    if getattr(ctx, 'capture_intermediates', False):
        captured['kv_after_rope'] = kv.detach()
        captured['kv_compress_raw'] = kv_compress.detach() if kv_compress is not None else None
        captured['compress_topk_idxs_phase2'] = compress_topk_idxs.detach() if compress_topk_idxs is not None else None
    
    # ═══════ Hook ═══════
    kv, kv_compress, indexer_k, compress_topk_idxs, packed_seq_params, ... = ...
    
    # ... Phase 4-5 (unchanged) ...
    
    if captured is not None:
        captured['o_raw'] = o.detach()
        captured['o_rotated'] = o_rotated.detach()
        captured['core_attn_out'] = core_attn_out.detach()
        captured['bias'] = bias.detach() if bias is not None else None
    
    if captured is not None:
        return core_attn_out, bias, captured
    return core_attn_out, bias
```

**要点**：
- `capture_intermediates` 通过 context 传递，默认 False，不影响现有代码
- 只在测试时开启，生产路径零开销
- 捕获的 tensor 全部 detach，不延长计算图

---

## Task 1：1.1 单卡功能补测 (Mac)

### 目标

补充 Mac 环境下缺失的边界场景，全部在 CPU-only 环境运行。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/unit_test/test_g2_edge_cases.py` | **新建** |

### 9 个测试用例

#### 组 1：多序列混合

| # | 测试 | 验证点 |
|---|------|--------|
| 1 | `test_multi_provider_multi_reuser` | 2 provider (不同 P) + 3 reuser（交叉复用），验证 store/expand 大小和内容 |
| 2 | `test_pass_through_sequence` | 非 provider 非 reuser 的序列（通过修改 plan 的 is_provider/is_reuser 字段模拟）pass-through 不变 |

#### 组 2：cu_seqlens 边界

| # | 测试 | 验证点 |
|---|------|--------|
| 3 | `test_cu_seqlens_cmp_kv_adjust` | cu_seqlens_cmp_kv 按 compress_ratio 偏移后，cmp entry 数量与 kv entry 对应 |
| 4 | `test_cu_seqlens_kv_padded_priority` | 同时有 cu_seqlens_kv_padded 和 cu_seqlens_kv 时，优先使用 padded 版本，且两者偏移一致 |

#### 组 3：错误路径 & 空边界

| # | 测试 | 验证点 |
|---|------|--------|
| 5 | `test_compute_cmp_lengths_mismatch` | computed_sum != actual 时 assert 触发，错误信息包含两个值 |
| 6 | `test_store_after_close` | close 后 store/load/contains 均抛 RuntimeError |
| 7 | `test_empty_batch` | batch_size=0 时所有函数返回空列表/None |
| 8 | `test_prefix_len_zero_reuser` | prefix_len=0 的 reuser expand 时 cat([empty, suffix]) = suffix |
| 9 | `test_max_prefix_len` | prefix_len == valid_len（reuser 无 suffix）时，cat([full_provider, empty]) = full_provider |

### 验证方式

```bash
cd prefix-sharing && python -m pytest tests/unit_test/test_g2_edge_cases.py -v
```

**准出条件**：9/9 通过。

### 产出

`docs/reports/report_1_1_mac.md`

---

## Task 2：1.2 单卡精度等价性 (NPU 单卡)

### 目标

在 NPU 单卡上完成 A/B/C 三基线对比，通过全部 16 个检查点。

### 前置

- [x] Task 0 完成（conftest.py + capture_intermediates）
- [x] Task 1 完成（Mac 边界场景全部通过）
- [ ] NPU 服务器就绪（`192.168.0.112`，容器 `verl-qwen-prefix-baseline`）
- [ ] 确认 `DeepSeek4SelfAttention` 可独立实例化（单层 + 随机权重，不走完整 43 层 config）

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/precision/test_single_card_equivalence.py` | **新建** |

### 测试矩阵（36 参数组合 + 2 smoke tests）

```python
@pytest.mark.parametrize("compress_ratio", [128, 4])
@pytest.mark.parametrize("prefix_len_pct", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("batch_composition", ["1p1r", "1p3r", "2p3r_mixed"])
@pytest.mark.parametrize("checkpoint", ["forward", "backward"])
```

### 验证流程（每个参数组合）

```
Step A: 无 PS patch 的 pure baseline
  → original_forward(hidden, mask, ...)
  → 保存: o_baseline, grad_baseline, 所有中间检查点值

Step B: PS patch 安装 + ctx 为空（patch no-op）
  → patched_forward(...) → original_forward(...)
  → 断言: o_b.bitwise_eq(o_a)     ← 应该完全 bitwise 一致
  → 断言: grad_b.bitwise_eq(grad_a)
  → 这个比较做一次 smoke test 验证即可，不需要遍历所有组合

Step C: PS patch 激活 + ctx 就位 + capture_intermediates=True
  → patched_forward(...) → 16 个检查点逐级对比

优先级 1 — plan 层（setup 阶段断言，非跨路径对比）：
  plan 是 A/B/C 三条路径共用的同一个实例（plan_A is plan_C），
  不需要"对比"，只需在 forward 之前验证字段符合预期：
  ✅ 1. prefix_lens == expected（根据 batch_composition + prefix_len_pct 计算）
  ✅ 2. provider_index == expected（哪个 batch 是 provider）
  ✅ 3. kept_lengths_q == expected_suffix_lengths
  ✅ 4. prefix_last_restore[*].label_value 指向正确的 reuser suffix 第一个 token

优先级 2 — 前处理层（期望 zero-diff）:
  ✅ 5. linear_q output
  ✅ 6. RoPE 前 Q/K
  ✅ 7. RoPE 后 Q
  ✅ 8. RoPE 后 KV

优先级 3 — 注意力层（allclose atol=1e-5）:
  ✅ 9. kv_compress
  ✅ 10. Phase 2 topk
  ✅ 11. attention logits
  ✅ 12. attention output
  ✅ 13. suffix output
  ✅ 14. 2nd RoPE 后

优先级 4 — 输出层（allclose atol=1e-5）:
  ✅ 15. core_attn_out
  ✅ 16. bias
```

每个检查点失败时，立即报告并中止该参数组合的后续检查。不中止整个测试（剩余组合继续跑，汇总所有失败）。

### 实现要点

```python
class SingleCardEquivalenceTest:
    """每个参数组合的 A/B/C 三基线对比测试器。
    
    核心方法:
      - run_baseline_A(): pure original_forward
      - run_noop_B(): patched_forward with ctx=None → assert bitwise_eq(A)
      - run_optimized_C(): patched_forward with ctx active → 16 checkpoints vs A
    """
    
    def _check_priority(self, priority, checks):
        """执行一组优先级检查，第一个失败后抛出，记录根因。"""
        for name, fn in checks:
            ok, detail = fn()
            if not ok:
                raise CheckpointFailure(
                    f"[P{priority}] {name} FAILED: {detail}")
    
    def run(self, ratio, prefix_len_pct, batch_comp, checkpoint):
        # Step A
        o_a, intermediates_a = self._run_pure_baseline()
        
        # Step B (smoke once) — can be cached after first run
        o_b, intermediates_b = self._run_patch_noop()
        assert _bitwise_equal(o_a, o_b), "B≠A: patch itself introduces error"
        
        # Step C — 16 checkpoints
        o_c, intermediates_c = self._run_ps_optimized(capture=True)
        
        # Priority 1 checks (plan level, setup-time — not cross-path comparison)
        # plan is the same instance for all A/B/C paths, verified before forward
        self._check_priority(1, [
            ("prefix_lens",      lambda: self._assert_eq(plan.prefix_lens, expected_prefix_lens)),
            ("provider_index",   lambda: self._assert_eq(plan.provider_index, expected_provider_index)),
            ("kept_lengths_q",   lambda: self._assert_eq(plan.kept_lengths_q, expected_suffix_lengths)),
            ("prefix_last_label", lambda: self._assert_label_value(plan.prefix_last_restore, expected_labels)),
        ])
        
        # Priority 2 checks (pre-processing, expect zero-diff)
        self._check_priority(2, [...])
        
        # Priority 3 checks (attention, allclose atol=1e-5)
        self._check_priority(3, [...])
        
        # Priority 4 checks (output, allclose atol=1e-5)
        self._check_priority(4, [...])
        
        # Backward (if applicable)
        if checkpoint == "backward":
            self._cmp_grads(grad_a, grad_c)
```

### 验证方式

```bash
# NPU 单卡
PYTHONPATH=prefix-sharing python -m pytest \
  tests/precision/test_single_card_equivalence.py -v --tb=short
```

### 准出条件

- 36/36 参数组合全部通过
- A≈B smoke test 通过（bitwise）
- 16 个检查点全部达标（检查点 1-8: zero-diff，检查点 9-16: allclose atol=1e-5）
- Backward 梯度全等（allclose atol=1e-5）

### 产出

`docs/reports/report_1_2_single_card.md`

---

## Task 3：1.3 多卡 TP 精度 (NPU 多卡)

### 目标

验证 TP>1 时 PS 的 tensor-parallel 正确性。

### 前置

- [x] Task 2 完成（单卡精度全部通过）

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/precision/test_tp_equivalence.py` | **新建** |

### TP 特有风险点（需验证）

| 风险 | 验证方法 |
|------|---------|
| packed_seq_params 的 cu_seqlens 是 TP-local 还是全局？ | `print(cu_seqlens)` + 对比各 rank |
| tp_rank 放入 SlotId 的隔离是否正确？ | 同一 batch_idx 不同 tp_rank 用不同 slot |
| kv_allgather=True + sequence_parallel=True 组合 | q_len × tp_size 计算正确 |
| gather_from_tensor_model_parallel_region 后数据一致性 | 各 rank 的 allgather 结果相同 |

### 测试矩阵（24 参数组合）

```python
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("compress_ratio", [128, 4])
@pytest.mark.parametrize("batch_composition", ["1p1r", "1p3r"])
@pytest.mark.parametrize("sequence_parallel", [True, False])
```

### 验证流程

每个参数组合验证：

```
1. 各 rank 各自 init（torchrun 分配 rank/word_size）
2. 各 rank 各自的 patched_forward（TP 下 Q/KV split 到各 rank）
3. 收集各 rank 的 attention output
4. 断言:
   ✅ 各 rank output shape 正确（考虑 TP split）
   ✅ 各 rank store 内容一致（gather 后全量 KV，数据相同）
   ✅ 各 rank 的 output 与单卡 baseline 一致（gather 后对比）
   ✅ backward grad 各 rank 一致
```

### 验证方式

```bash
# TP=2
torchrun --nproc_per_node=2 -m pytest tests/precision/test_tp_equivalence.py -v

# TP=4
torchrun --nproc_per_node=4 -m pytest tests/precision/test_tp_equivalence.py -v

# TP=8
torchrun --nproc_per_node=8 -m pytest tests/precision/test_tp_equivalence.py -v
```

### 准出条件

- 24/24 参数组合全部通过（三个 TP size 均通过）
- 各 rank store 内容一致（gather 后全量 KV）
- 梯度与单卡 baseline 一致

### 产出

`docs/reports/report_1_3_tp.md`

---

## Task 4：1.4 多卡 CP 精度 (NPU 多卡)

### 目标

解除 `supported_cp_size=1` 硬门禁，验证 CP>1 时 PS 的正确性。

### 前置

- [x] Task 2 完成（单卡精度全部通过）
- [x] NPU 多卡环境可访问

### 文件变更

| 文件 | 操作 |
|------|------|
| `core/config.py` | **修改**：`supported_cp_size: int = 1` → `supported_cp_size: int = 8`（或移除检查） |
| `tests/precision/test_cp_equivalence.py` | **新建** |

### 前置验证项（§7 延后项）

在参数化测试之前，先跑 5 个一次性验证：

| # | 验证项 | 方法 | 对应文档 |
|---|--------|------|---------|
| V1 | CP + TP 组合 | cp_size=2, tp_size=2 下 forward 等价性 | cp_ps_design.md §7 项1 |
| V2 | CP + PP 组合 | pp_size=2 下 store 生命周期不跨 PP stage | cp_ps_design.md §7 项2 |
| V3 | 压缩边界 × CP 交叉 | P%r≠0 但在 align 后 P%r=0（方案 A 已保证） | cp_ps_design.md §7 项3 |
| V4 | Ring Attention 替代 gather | **不在本轮范围** | cp_ps_design.md §7 项4 |
| V5 | ratio=4 Indexer CP 行为 | 容器内 print(key_index.shape) 确认 gather 后 shape | cp_ps_design.md §7 项5 |

### 测试矩阵（12 参数组合）

```python
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("compress_ratio", [128, 4])
@pytest.mark.parametrize("batch_composition", [
    "1p1r",         # 基础：1 provider + 1 reuser
    "1p3r",         # transitive reuse 在 CP 下
    "cross_rank",   # reuser suffix 跨 rank 分布
])
```

`cross_rank` 场景定义：构造一个 reuser 序列，其 suffix 被 CP 切到不同 rank。验证 gather 后每个 rank 的 expand 结果完全一致。

### 验证流程

```
1. CP 下各 rank 各自 forward
2. 断言: 各 rank 的 kv_after_gather 完全一致（gather_from_sp_cp 确保全量）
3. 断言: 各 rank 的 store 内容完全一致（相同数据 → 相同 store）
4. 断言: 各 rank 的 expand 结果完全一致
5. 断言: attention output 与 baseline（CP=1 单卡）一致
```

### 验证方式

```bash
# CP=2
torchrun --nproc_per_node=2 -m pytest tests/precision/test_cp_equivalence.py -v

# CP=4
torchrun --nproc_per_node=4 -m pytest tests/precision/test_cp_equivalence.py -v
```

### 准出条件

- 前置验证 V1-V3, V5 通过（V4 延后）
- 12/12 参数组合全部通过
- 各 rank store/expand/output 一致

### 产出

`docs/reports/report_1_4_cp.md`

---

## Task 5：1.5 端到端精度红线 (NPU 8 卡)

### 目标

完整 43 层模型（关闭 MoE），TP=1/PP=1/CP=1，验证 loss/logprob/grad 与 baseline 完全一致。

### 前置

- [x] Task 3 完成（TP 精度全部通过）
- [x] Task 4 完成（CP 精度全部通过）
- [ ] NPU 8 卡环境就绪
- [ ] `wrap_forward_step` 可正确挂载到 MindSpeed pretrain 入口

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/precision/test_e2e_equivalence.py` | **新建** |

### 6 个 E2E 场景

| # | 场景 | 模型 | 输入 | 验证目标 |
|---|------|------|------|---------|
| 1 | 无共享前缀 + PS 激活 | 完整 43 层 | 随机 batch (无前缀) | PS 自动退避 → loss/grad == baseline |
| 2 | 共享前缀 + ratio=128 层 | 完整 43 层 | 1p1r (P=S=256) | loss 一致（abs(diff) < 1e-8），grad 一致 |
| 3 | 共享前缀 + 混合 ratio | 完整 43 层 | 1p1r | ratio=0/4/128 混合层 loss/grad 一致 |
| 4 | Per-token logprob 一致性 | 完整 43 层 | 1p1r | 逐 token logprob allclose(atol=1e-5) |
| 5 | Gradient norm 一致性 | 完整 43 层 | 1p3r | grad_norm abs(diff) < 1e-8 |
| 6 | 多 micro-batch 累积 | 完整 43 层 | 连续 3 个 micro-batch | 累积 grad 与 baseline 一致 |

### 验证方式

```bash
# 8 卡 E2E
torchrun --nproc_per_node=8 -m pytest tests/precision/test_e2e_equivalence.py -v
```

### 准出条件

- 6/6 场景全部通过
- Loss/logprob/grad 全部达标
- 精度红线全部满足：
  - `abs(loss_ps - loss_baseline) < 1e-8`
  - per-token logprob: `allclose(atol=1e-5)`
  - per-param grad: `allclose(atol=1e-5)`
  - grad_norm: `abs(diff) < 1e-8`

### 产出

`docs/reports/report_1_5_e2e.md`

---

## 执行顺序与工时

```
Task 0  (infra)           ── 0.5天 ──┐
Task 1  (1.1 Mac)         ── 0.5天 ──┤ 可并行
                                      │
Task 2  (1.2 单卡)        ── 2.0天 ──┤ 依赖 Task 0 + 1
Task 3  (1.3 TP)          ── 1.5天 ──┤ 依赖 Task 2
Task 4  (1.4 CP)          ── 1.5天 ──┤ 依赖 Task 2
Task 5  (1.5 E2E)         ── 2.0天 ──┘ 依赖 Task 3 + 4
─────────────────────────────────────
合计                      8.0 天
```

**启动条件**：可以立刻在 Mac 上跑 Task 0 和 Task 1。Task 2-5 需要 NPU 服务器和容器就绪。
