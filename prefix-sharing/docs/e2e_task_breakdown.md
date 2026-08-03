# Task 5: E2E 精度仿真 — 任务拆分

> **日期**：2026-07-30
> **父文档**：`e2e_simulation_plan.md`
> **环境**：2 卡 910B3（`deepseek-verify` 容器）

## 总览

```
Phase 0: E2E 基础设施  (conftest + 模型工厂 + mock data)
  └── Phase 1: 单卡基线   (跑通 forward → loss, 不挂 PS)
        └── Phase 2: 单卡 PS   (wrap_forward_step, loss 对比)
              ├── Phase 3: TP=2    (分布式 + KV 分片)
              ├── Phase 4: CP=2    (跨 rank KV gather)
              └── Phase 5: TP=2+CP=2 (组合)
```

| Phase | 内容 | 卡数 | 新增用例 | 产出报告 |
|-------|------|:--:|:--:|------|
| 0 | 基础设施 | 1 | — | conftest.py + model_factory.py |
| 1 | 单卡基线 | 1 | 2 | report_5_1_baseline.md |
| 2 | 单卡 PS | 1 | 2 | report_5_2_single_ps.md |
| 3 | TP=2 | 2 | 2 | report_5_3_tp2.md |
| 4 | CP=2 | 2 | 2 | report_5_4_cp2.md |
| 5 | TP=2+CP=2 | 2 | 1 | report_5_5_tp2cp2.md |

---

## Phase 0: E2E 基础设施

### 目标

搭建可复用的 E2E 测试环境：模型工厂、mock 数据生成器、文件骨架。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/e2e/__init__.py` | 新建 |
| `tests/e2e/conftest.py` | 新建 — session fixture |
| `tests/e2e/model_factory.py` | 新建 — 模型工厂 |
| `tests/e2e/mock_data.py` | 新建 — mock 数据生成器 |

### 0.1 conftest.py — 单卡初始化

```python
@pytest.fixture(scope="session")
def init_distributed():
    """单卡：torch.distributed 初始化 + Megatron parallel_state.
    多卡 (torchrun)：自动识别 rank/world_size."""
    torch.distributed.init_process_group(backend="hccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        context_parallel_size=CP_SIZE,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )

@pytest.fixture(scope="session")
def set_e2e_args():
    """注入完整 Megatron args, 供 model_provider 和 forward_step 使用."""
    args = _MinimalArgs(**TRAINING_ARGS)
    set_args(args)
```

### 0.2 model_factory.py — 模型工厂

基于容器内 `pretrain_deepseek4.py` 的 `model_provider()`：

```python
def make_e2e_model(tp_size=1, cp_size=1, **overrides):
    """创建 3 层 DeepSeek4Model (随机权重, use_cpu_initialization=True)
    
    Args:
        tp_size, cp_size: TP/CP 配置
        **overrides: 覆盖默认 args (如 compress_ratios)
    """
    args = get_args()
    # 合并 overrides...
    config = core_transformer_config_from_args(args)
    transformer_layer_spec = get_gpt_layer_local_spec(...)
    return DeepSeek4Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        ...
    )

def make_e2e_model_fresh():
    """torch.manual_seed(42) + make_e2e_model() — 每次调返回相同权重."""
```

### 0.3 mock_data.py — Mock 数据

```python
def make_mock_batch(prefix_len=128, seq_len=384):
    """2 条序列, 共享 prefix_len 前缀, 对齐 compress_ratio=128
    
    Returns: (tokens, labels, loss_mask, attention_mask, position_ids)
    """
    P = prefix_len
    L = seq_len
    tokens = torch.tensor([
        list(range(L)),                          # seq0: provider
        list(range(P)) + list(range(1000, 1000+L-P)),  # seq1: reuser
    ])
    labels = F.pad(tokens[:, 1:], (0, 1), value=0)
    loss_mask = torch.ones_like(labels)
    attention_mask = torch.ones(B, 1, L, L)
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
    return tokens, labels, loss_mask, attention_mask, position_ids

class MockDataIterator:
    """torch 1.0, 返回 (tokens, labels, loss_mask, attention_mask, position_ids)."""
    def __init__(self, *batch): self.batch = batch
    def __iter__(self): return iter([self.batch])
    def __next__(self): return self.batch
```

### 准出条件

```bash
# 0.1 同步依赖
docker cp /Users/kevin/code/MindSpeed-LLM/pretrain_deepseek4.py deepseek-verify:/tmp/
docker cp /Users/kevin/code/MindSpeed-LLM/mindspeed_llm/core/models/deepseek4/ deepseek-verify:/tmp/mindspeed_llm/core/models/deepseek4/

docker exec deepseek-verify python3 -c "
import sys; sys.path.insert(0, '/tmp')
from pretrain_deepseek4 import model_provider, forward_step
from tests.e2e.model_factory import make_e2e_model
from tests.e2e.mock_data import make_mock_batch
model = make_e2e_model()
batch = make_mock_batch()
print('model layers:', len(model.decoder.layers))
print('batch tokens:', batch[0].shape)
"
```

- [ ] torchrun 不可用，Phase 3-5 改用 `python3 -m torch.distributed.launch --nproc_per_node=2`
- [ ] `pretrain_deepseek4.py` 已同步到容器
- [ ] 模型实例化成功（PP=1 避免 pipeline 错误）
- [ ] mock 数据 shape 正确（2×384）

---

## Phase 1: 单卡基线（TP=1, CP=1, 不挂 PS）

### 目标

跑通完整 forward → loss 流程，确认模型和 mock 数据在单卡下可正常运行。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/e2e/test_phase1_baseline.py` | 新建 |

### 两个测试用例

| # | 内容 | 验证 |
|---|------|------|
| 1 | 1 序列无共享：seq0 全量 forward → loss 非 NaN | E2E 框架可用 |
| 2 | 2 序列有共享 (P=128)：双序列全量 forward → loss 非 NaN | mock 数据可用 |

```python
def test_baseline_single_seq(init_distributed, set_e2e_args):
    model = make_e2e_model_fresh()
    tokens, labels, loss_mask = make_single_seq_batch()  # 1 seq
    output = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
    assert not torch.isnan(loss).any()

def test_baseline_two_seq(init_distributed, set_e2e_args):
    model = make_e2e_model_fresh()
    tokens, labels, loss_mask, _, _, pos_ids = make_mock_batch(P=128)
    output = model(tokens, pos_ids, torch.ones(2,1,384,384), labels=labels, loss_mask=loss_mask)
    assert not torch.isnan(loss).any()
```

### 运行

```bash
docker exec deepseek-verify PYTHONPATH=/tmp/prefix-sharing python3 -m pytest tests/e2e/test_phase1_baseline.py -v
```

### 准出条件

- [ ] 2/2 通过（loss 非 NaN）

---

## Phase 2: 单卡 PS（TP=1, CP=1, 挂 PS）

### 目标

首次验证 `wrap_forward_step` 全链路：batch trim → context → model forward → prefix-last restore。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/e2e/test_phase2_single_ps.py` | 新建 |

### 两个测试用例

| # | 内容 | 验证 |
|---|------|------|
| 1 | 无共享 (P=0)：PS path loss == baseline loss | wrap_forward_step fallback 正确 |
| 2 | 有共享 (P=128)：PS path loss == baseline loss | store/expand/restore 全链路 bitwise |

```python
def test_ps_no_sharing():
    """P=0 → plan.has_sharing=False → wrap_forward_step 退避到 original_forward_step."""
    model = make_e2e_model_fresh()
    batch = make_mock_batch(P=0)  # 无共享
    data_iter = MockDataIterator(*batch)
    
    loss_baseline = forward_step(data_iter, model)
    
    model2 = make_e2e_model_fresh()
    wrapped = wrap_forward_step(forward_step, ps_config, get_batch_fn=lambda it: it.next())
    loss_ps = wrapped(data_iter, model2)
    
    assert abs(loss_ps[0] - loss_baseline[0]) < 1e-8

def test_ps_with_sharing():
    """P=128, 共享前缀 → PS 全链路 (trim+store+expand+restore)."""
    model = make_e2e_model_fresh()
    batch = make_mock_batch(P=128)
    data_iter = MockDataIterator(*batch)
    
    loss_baseline = forward_step(data_iter, model)
    
    model2 = make_e2e_model_fresh()
    wrapped = wrap_forward_step(forward_step, ps_config, get_batch_fn=lambda it: it.next())
    loss_ps = wrapped(data_iter, model2)
    
    assert abs(loss_ps[0] - loss_baseline[0]) < 1e-8
```

### 准出条件

- [ ] 2/2 通过（loss 精确一致，1e-8）

---

## Phase 3: TP=2

### 目标

验证 PS 在 TP>1 下正确性。TP 下 Q/KV 被分片到 2 rank。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/e2e/test_phase3_tp2.py` | 新建 |

### 两个测试用例

| # | 内容 | 验证 |
|---|------|------|
| 1 | TP=2, 无共享：PS path loss == baseline | fallback 在 TP 下正确 |
| 2 | TP=2, 有共享 (P=128)：PS path loss == baseline | store/expand/restore 在 TP 分片下正确 |

```bash
python3 -m torch.distributed.launch --nproc_per_node=2 -m pytest tests/e2e/test_phase3_tp2.py -v
```

### 准出条件

- [ ] 2/2 通过（TP=2 下 loss 精确一致）

---

## Phase 4: CP=2

### 目标

验证 PS 在 CP>1 下正确性。CP 下 KV 被跨 rank gather。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/e2e/test_phase4_cp2.py` | 新建 |

### 两个测试用例

| # | 内容 | 验证 |
|---|------|------|
| 1 | CP=2, 无共享：PS path loss == baseline | fallback 在 CP 下正确 |
| 2 | CP=2, 有共享 (P=128)：PS path loss == baseline | gather 后 store/expand 正确 |

```bash
python3 -m torch.distributed.launch --nproc_per_node=2 -m pytest tests/e2e/test_phase4_cp2.py -v
```

### 准出条件

- [ ] 2/2 通过（CP=2 下 loss 精确一致）

---

## Phase 5: TP=2 + CP=2

### 目标

验证 PS 在 TP+CP 组合下正确性。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tests/e2e/test_phase5_tp2cp2.py` | 新建 |

### 一个测试用例

| # | 内容 | 验证 |
|---|------|------|
| 1 | TP=2, CP=2, 有共享 (P=128)：PS path loss == baseline | 组合场景全链路 |

```bash
python3 -m torch.distributed.launch --nproc_per_node=2 -m pytest tests/e2e/test_phase5_tp2cp2.py -v
```

### 准出条件

- [ ] 1/1 通过（TP+CP 组合下 loss 精确一致）

---

## 执行顺序

```
Phase 0  (infra)      ── 0.5天 ──┐
Phase 1  (baseline)   ── 0.5天 ──┤ 可并行
Phase 2  (单卡 PS)    ── 1.0天 ──┤ 依赖 Phase 0+1
Phase 3  (TP=2)       ── 1.0天 ──┤ 依赖 Phase 2
Phase 4  (CP=2)       ── 1.0天 ──┤ 依赖 Phase 2
Phase 5  (TP+CP)      ── 0.5天 ──┘ 依赖 Phase 3+4
─────────────────────────────────
合计                   3.5 天
```

## 准入依赖

- [ ] Phase 0 完成（E2E 基础设施就绪）
- [ ] Phase 1 完成（单卡基线通过）
- [ ] `wrap_forward_step` 接口确认（`get_batch_fn` 参数可用，或直接 monkey-patch `megatron.training.training.get_batch`）
- [ ] 2 卡可用（`torchrun --nproc_per_node=2` 正常启动）
