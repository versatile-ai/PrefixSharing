# Task 5: TP/CP 精度仿真测试方案 v2（单层 Attention）

> **日期**：2026-07-30
> **前置**：Task 2（单卡单层 forward bitwise）已通过
> **环境**：2 卡 910B3（192.168.0.2，容器 `deepseek-verify`）
> **策略**：基于 Task 2 的 `DeepSeek4SelfAttention` × 1 层，通过 `parallel_state` 控制 TP/CP，不依赖完整模型。

## 0. 与 v1 方案的区别

| | v1（废弃） | v2（当前） |
|------|------|------|
| 模型 | 3 层 DeepSeek4Model | 1 层 DeepSeek4SelfAttention |
| 构造方式 | pretrain_deepseek4.model_provider() | 直接 `DeepSeek4SelfAttention(config, submodules)` |
| 依赖 | transformer_engine, features_manager | 无额外依赖 |
| TP/CP 验证 | 需要完整模型 | parallel_state 初始化即可 |
| 可行性 | 阻塞（TE mock 打地鼠） | Task 2 已验证可行 |

## 1. 覆盖矩阵

```
               TP=1,CP=1    TP=2,CP=1    TP=1,CP=2    TP=2,CP=2
ratio=0         ✅ Task 2     Task 5.2      Task 5.3     Task 5.4
ratio=128       ✅ Task 2     Task 5.2      Task 5.3     Task 5.4
```

共 8 个参数组合（4 种并行配置 × 2 种压缩比），加上 2 个 baseline（无 PS 原始 forward），覆盖完整。

| 测试 | 验证内容 | 对应原计划 |
|------|------|------|
| Task 5.1 单卡 baseline | TP=1,CP=1 无 PS，确认框架可用 | Phase 1 |
| Task 5.2 TP=2 | 权重分片 + compressed attention + PS | Phase 3 |
| Task 5.3 CP=2 | KV 跨 rank gather + compressed attention + PS | Phase 4 |
| Task 5.4 TP=2+CP=2 | 分片+gather+compressed+PS | Phase 5 |

## 2. 测试方法

### 模型构造（复用 Task 2 conftest）

```python
# 与 Task 2 完全相同的方式构造单层 attention
sm = get_deepseek4_self_attn_submodules(
    qk_layernorm=True, mla_mm_split=False,
    enable_dsa_indexer=False, compressor=True,
)
attn = DeepSeek4SelfAttention(config=config, submodules=sm, layer_number=1)
```

### TP/CP 控制

```python
# conftest 中根据参数初始化 parallel_state
@pytest.fixture(scope="session")
def init_distributed(tp_size, cp_size):
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
```

`DeepSeek4SelfAttention.__init__()` 自动从 `parallel_state` 读取 `world_size` / `cp_size`，计算 `n_local_heads` / `n_local_groups`，权重自动分片。不需要任何额外代码。

### 数据构造（复用 Task 2 mock_data）

```python
# mock_batch: 2 序列共享 prefix, ratio 对齐
# ratio=0:  seqlen=256, P=[64, 128, 192]
# ratio=128: seqlen=512, P=[128, 256, 384]
tokens, labels, loss_mask, attn_mask, pos_ids = make_mock_batch(P=128, seqlen=512)
```

### 精度对比

```python
# A: baseline — 原始 forward，无 patch
torch.manual_seed(42)
attn_baseline = build_attention()
o_baseline, _ = attn_baseline(hidden_states=..., ...)

# B: patch no-op — context 为空，验证 patch 不引入误差
# Task 2 已验证 A≈B bitwise

# C: PS activated — patch + store/expand + sparse_attention
torch.manual_seed(42)
attn_ps = build_attention()
attn_ps.forward = types.MethodType(patch_g2_attention(type(attn_ps).forward), attn_ps)
with prefix_sharing_runtime_context(state) as ctx:
    o_ps, _ = attn_ps(hidden_states=..., ...)

# 对比
assert torch.allclose(o_ps, o_baseline, atol=1e-5, rtol=1e-4)
```

## 3. 参数矩阵

### ratio=0（无压缩，compressor=False）

| # | TP | CP | P | batch | 验证 |
|---|:--:|:--:|---|------|------|
| 1 | 1 | 1 | 64/128/192 | 1p1r/1p3r/2p3r_mixed | ✅ Task 2 |
| 2 | 2 | 1 | 128 | 1p1r/1p3r | attention output allclose |
| 3 | 1 | 2 | 128 | 1p1r/1p3r | attention output allclose |
| 4 | 2 | 2 | 128 | 1p1r | attention output allclose |

### ratio=128（压缩，compressor=True）

| # | TP | CP | P | batch | 验证 |
|---|:--:|:--:|---|------|------|
| 5 | 1 | 1 | 128/256/384 | 1p1r/1p3r/2p3r_mixed | ✅ Task 2 |
| 6 | 2 | 1 | 256 | 1p1r/1p3r | attention output allclose |
| 7 | 1 | 2 | 256 | 1p1r/1p3r | attention output allclose |
| 8 | 2 | 2 | 256 | 1p1r | attention output allclose |

### baseline（无 PS，验证框架）

| # | TP | CP | ratio | 验证 |
|---|:--:|:--:|:--:|------|
| 9 | 2 | 1 | 0 | 原始 forward 可执行，loss 非 NaN |
| 10 | 2 | 1 | 128 | 原始 forward 可执行，loss 非 NaN |
| 11 | 1 | 2 | 0 | 原始 forward 可执行，loss 非 NaN |
| 12 | 1 | 2 | 128 | 原始 forward 可执行，loss 非 NaN |

## 4. 文件结构

```
tests/e2e/
├── __init__.py
├── conftest.py              # 共享 fixture（复用 Task 2 的 Tier 1/2）
├── mock_data.py             # 复用 Task 2 的 mock_batch
├── model_factory.py         # DeepSeek4SelfAttention 工厂（精简版）
├── test_phase1_baseline.py  # 单卡 baseline（TP=1,CP=1, 不挂 PS）
├── test_phase2_single_ps.py # 单卡 PS（复用 Task 2）
├── test_phase3_tp2.py       # TP=2, CP=1 (4 参数组合)
├── test_phase4_cp2.py       # TP=1, CP=2 (4 参数组合)
└── test_phase5_tp2cp2.py    # TP=2, CP=2 (2 参数组合)
```

## 5. 准出条件

| Phase | 条件 |
|-------|------|
| Phase 1 | baseline 在 TP=2/CP=2 下可执行 |
| Phase 2 | PS 单卡 ratio=0+128 全部 bitwise（复验 Task 2） |
| Phase 3 | TP=2 下 PS output allclose baseline |
| Phase 4 | CP=2 下 PS output allclose baseline |
| Phase 5 | TP=2+CP=2 下 PS output allclose baseline |

## 6. 运行

```bash
# 单卡
docker exec deepseek-verify python3 -m pytest tests/e2e/test_phase1_baseline.py -v
docker exec deepseek-verify python3 -m pytest tests/e2e/test_phase2_single_ps.py -v

# TP=2 / CP=2 (2 卡)
python3 -m torch.distributed.launch --nproc_per_node=2 -m pytest tests/e2e/test_phase3_tp2.py -v
python3 -m torch.distributed.launch --nproc_per_node=2 -m pytest tests/e2e/test_phase4_cp2.py -v
python3 -m torch.distributed.launch --nproc_per_node=2 -m pytest tests/e2e/test_phase5_tp2cp2.py -v
```

## 7. 与 Task 2 的关系

Task 2 已覆盖 TP=1,CP=1 全部场景（ratio=0:9 用例, ratio=128:9 用例）。本方案新增 TP/CP 组合，不重复 Task 2 已覆盖的用例。

## 8. 可行性确认

| 风险 | 缓解 |
|------|------|
| TP=2 权重分片后 attention output 维度不对 | DeepSeek4SelfAttention 内部处理，Task 2 已验证单卡 |
| CP=2 gather 后 KV 长度不一致 | gather_from_sp_cp 在 attention.forward 内部调用 |
| 2 卡 torch.distributed.launch 启动 | 容器已验证 HCCL 可用 |
| 单层 attention 不能覆盖多层数据流 | 多层数据流放到 E2E 完整模型阶段（8 卡集群） |
