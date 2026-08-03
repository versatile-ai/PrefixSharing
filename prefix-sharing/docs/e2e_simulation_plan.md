# Task 5: E2E 精度仿真测试方案

> **日期**：2026-07-30
> **前置**：Task 2（单卡单层 forward bitwise）已通过
> **环境**：2 卡 910B3（覆盖单卡/TP=2/CP=2），容器 `deepseek-verify`

## 1. 目标

用**随机权重 + mock 数据**在 2 卡机器上完成 E2E 精度验证，覆盖：

| 维度 | 覆盖 |
|------|------|
| 多层数据流 | tokens → embed → layerN → lm_head → loss |
| 混合 ratio | ratio=128 层 + ratio=0 层 |
| prefix-last restore | PS path loss == baseline loss |
| TP=2 | Q/KV 分片下 PS 正确性 |
| CP=2 | 跨 rank KV gather 下 PS 正确性 |
| wrap_forward_step | batch trim → context → forward → restore 全链路 |

## 2. 模型配置

基于容器内 `pretrain_deepseek4.py` 的 `model_provider`，缩小为 3 层：

```python
TRANSFORMER_CONFIG = {
    "num_layers": 3,
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "num_query_groups": 64,
    "ffn_hidden_size": 11008,
    "gated_linear_unit": True,
    "num_moe_experts": 4,
    "moe_router_topk": 2,
    "moe_router_num_experts": 4,
    "seq_length": 512,
    "max_position_embeddings": 512,
    "use_cpu_initialization": True,
    "add_bias_linear": False,
    "normalization": "RMSNorm",
    "layernorm_epsilon": 1e-6,
}

TRAINING_ARGS = {
    "tensor_model_parallel_size": TP_SIZE,
    "context_parallel_size": CP_SIZE,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "micro_batch_size": 1,
    "global_batch_size": 1,
    "num_layers": 3,
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "ffn_hidden_size": 11008,
    "make_vocab_size_divisible_by": 128,
    "padded_vocab_size": 32000,
    "use_cpu_initialization": True,
    "bf16": True,
    "params_dtype": torch.bfloat16,
    # DeepSeek V4 specific
    "qk_head_dim": 512,
    "rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "g2_window_size": 128,
    "compress_ratios": [0, 128, 128],   # layer0=0, layer1=128, layer2=128
    "compress_rope_theta": 10000.0,
    "rope_theta": 10000.0,
    "rope_factor": 40,
    "beta_fast": 32,
    "beta_slow": 1,
    "rope_scaling_original_max_position_embeddings": 4096,
    "use_sparse_flash_attn": True,
    "use_fused_rmsnorm": True,
    "use_fused_lightning_indexer_loss": False,
    "use_g2_indexer_loss": False,
    "transformer_impl": "local",
    "position_embedding_type": "rope",
    "no_rope_freqs_schedule": False,
    "original_seq_len": 0,
    "enable_mhc": False,
}
```

**3 层选择理由**：
- layer 0: ratio=0 — 验证基础 attention 路径（无压缩）
- layer 1: ratio=128 — 验证 sparse_flash_mla + compressor
- layer 2: ratio=128 — 验证相邻 ratio=128 层间传递
- MoE=4 保持最小 Exert 配置

## 3. Mock 数据

```python
# 2 条序列，有共享前缀 P=192, total=384
# Prefix: 对齐 ratio=128 (192 = 128 + 64, 但 192%128=64 ≠ 0)
# 修正为 P=128: prefix_len=128, suffix_len=256 (384-128)
provider_tokens = [1, 2, 3, ..., 383, 384]              # 384 tokens
reuser_tokens  = [1, 2, ..., 128, 1001, ..., 1256]      # shared P=128 + unique suffix 256

# labels / loss_mask: 标准 causal LM
labels = tokens[:, 1:]  # [B, L-1]
loss_mask = torch.ones_like(labels)
```

## 4. 测试流程

每个参数组合（2 × 2 = 4 种）：

```python
@pytest.mark.parametrize("tp_size,cp_size", [(1,1), (2,1), (1,2), (2,2)])
def test_e2e_equivalence(tp_size, cp_size):
    # Step 1: init distribution (torchrun)
    
    # Step 2: build random-weight model via model_provider()
    
    # Step 3: create mock data_iter with 2 sequences
    
    # Step 4: baseline forward_step (no PS)
    loss_baseline = forward_step(data_iter, model)[0]
    
    # Step 5: PS forward_step (via wrap_forward_step)
    wrapped = wrap_forward_step(forward_step, ps_config, get_batch_fn=get_batch)
    loss_ps = wrapped(data_iter, model_ps)  # fresh model, same seed
    
    # Step 6: assert
    assert abs(loss_ps - loss_baseline) < 1e-8
```

### 关键细节

**Step 3 data_iter 构造**：mock 2 条序列的可迭代对象，返回 `(tokens, labels, loss_mask, attention_mask, position_ids)` 元组。

**Step 4 vs Step 5 模型独立**：每个参数组合创建两个独立模型实例（相同 seed），避免 PS path 修改模型状态影响 baseline。

**wrap_forward_step 集成**：`wrap_forward_step` 负责 prefix 检测、batch trim、context 注入和 prefix-last restore。在 E2E 中首次完整验证全链路。

**MoE 关闭**：`num_moe_experts=4` 但保持 TP/CP 简单，初版可以设 `moe_router_topk=0` 关闭 MoE gate。

## 5. 准出条件

| 测试 | 判定 |
|------|:--:|
| (TP=1, CP=1) loss_ps == loss_baseline | `abs(diff) < 1e-8` |
| (TP=2, CP=1) loss_ps == loss_baseline | `abs(diff) < 1e-8` |
| (TP=1, CP=2) loss_ps == loss_baseline | `abs(diff) < 1e-8` |
| (TP=2, CP=2) loss_ps == loss_baseline | `abs(diff) < 1e-8` |

## 6. 运行

```bash
# 单卡
PYTHONPATH=/tmp/prefix-sharing python3 tests/e2e/test_e2e_equivalence.py

# TP=2 / CP=2 (2 卡)
torchrun --nproc_per_node=2 -m pytest tests/e2e/test_e2e_equivalence.py -v
```

## 7. 文件变更

```
tests/e2e/
├── __init__.py
├── conftest.py                    # session fixture: torchrun init + model + data_iter
└── test_e2e_equivalence.py        # 4 参数组合测试
```

## 8. 可行性确认

| 风险 | 缓解 |
|------|------|
| 容器内 mindspeed_llm 需要完整环境 | 使用 `deepseek-verify` 容器，已验证 Task 2 |
| DeepSeek4Model 构造需要完整 arg 集合 | 从容器实际 `get_args()` 提取最小集合 |
| TP=2 需 2 卡 | 192.168.0.2 有 8 卡可用 |
| wrap_forward_step 调用 `get_batch` 内部 | 直接 override `megatron.training.training.get_batch` 或传 `get_batch_fn` |
| MoE overhead | 关闭 gate (`moe_router_topk=0`) 或使用 `num_moe_experts=1` |
| 3 层 + 512 seq 内存 | 约 4GB，单卡完全可承受 |

## 9. 与 Task 2 的关系

Task 2 验证了**单层 Hook 正确性**（store/expand/topk/cu_seqlens）。E2E 验证**多层数据流 + 训练框架集成**。两者互补：

```
Task 2: hidden → 1× attention → output          ✅ 已完成
E2E:   tokens → embed → 3× layers → lm_head     → 本方案
            → loss → prefix-last restore
```
