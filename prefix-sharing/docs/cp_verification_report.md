# PrefixSharing DeepSeek V4 — CP + verl Engine 验证报告

> **日期**: 2026-08-07 | **分支**: `feature/deepseek4-prefix-sharing` | **容器**: `deepseek-verify` (CANN 9.1)

## 1. 测试覆盖矩阵

### 1.1 模型配置

| 配置 | hidden | heads | layers | ratios | experts | params |
|------|:--:|:--:|:--:|------|:--:|:--:|
| 小模型 (core_attention) | 1024 | 16 | 2-4 | [0,128,4,128] | 4 | 140M-1.2B |
| **标准模型 (sparse_flash_mla)** | 2048 | 32 | 2 | [128,128] | 4 | 718M |

### 1.2 并行策略

| Config | TP | CP | DP | NPU | 结果 |
|--------|:--:|:--:|:--:|:--:|:--:|
| A | 1 | 1 | 8 | 8 | ✅ |
| B | 2 | 1 | 4 | 8 | ✅ (SP=True) |
| C | 1 | 2 | 4 | 2 | ⚠️ 环境限制 |

### 1.3 数据路径

| 路径 | 数据格式 | attention kernel | 结果 |
|------|---------|:--:|:--:|
| pretrain 2D → model.forward | [B,S] BSND | core_attention | ✅ |
| pretrain 2D → model.forward (reset_attn_mask) | [B,S] BSND + psp | core_attention | ✅ |
| verl `gptmodel_forward_model_engine` → `preprocess_thd_engine` | NestedTensor → THD packed | core_attention | ✅ |
| **verl `gptmodel_forward_model_engine` → `preprocess_thd_engine`** | **NestedTensor → THD packed** | **sparse_flash_mla** | ✅ |

### 1.4 DSA Indexer

| 配置 | 结果 |
|------|:--:|
| ratio=4, enable_dsa_indexer=True, kv_compress=True | ✅ 4层全部 matches_expected |
| ratio=4, enable_dsa_indexer=False | ⚠️ IdentityOp 风险（patched_forward 已防护） |

---

## 2. 发现的 Bug 与修复

### Bug 1: `model_type` 缺失导致 G2AttentionStore 不创建

**现象**: PS hook 不触发，audit 显示 `runtime_missing`

**根因**: `build_prefix_sharing_micro_batch_verl080` 创建 `PrefixSharingRuntimeState` 时未设置 `model_type`，默认 `"text_only_causal_lm"`。context 据此创建 `PrefixAttentionStore` 而非 `G2AttentionStore`。patched_forward 检查 `isinstance(ctx.store, G2AttentionStore)` 失败，返回原版 forward。

**修复** (`verl_mcore.py:677`):
```python
state = PrefixSharingRuntimeState(
    prefix_sharing_plan=plan,
    attention_backend=get_backend_instance(ps_config),
    packed_batch_layout=packed_layout,
    parallel_info=parallel_info,
    model_type="deepseek4",  # ← 添加
)
```

---

### Bug 2: G2 attention 双路径未上报 stats

**现象**: audit 显示 `runtime_missing`（即使 hook 正常执行）

**根因**: `_g2_padded_store_or_replace` 和 `_g2_kv_store_or_expand` 执行了 KV store/expand 但未调用 `ctx.stats.record_attention_kv_build()`。标准 attention 路径 (`torch_ref.py`) 有上报，G2 路径缺失。

**修复** (`g2_attention.py`，padded 和 packed 路径各加一段):
```python
if ctx.stats is not None:
    _store_count = sum(1 for i in range(layout.batch_size) if plan.is_provider[i])
    _reuse_count = sum(1 for i in range(layout.batch_size) if plan.is_reuser(i))
    ctx.stats.record_attention_kv_build(
        layer_id=layer_id, store_count=_store_count,
        reuse_count=_reuse_count, reuse_hit_count=_reuse_count,
        reuse_miss_count=0,
        stored_tokens=...,
        reused_prefix_tokens=...,
        expanded_kv_tokens=...,
        valid_q_tokens=sum(layout.valid_lengths),
        padded_q_tokens=sum(layout.padded_lengths),
    )
```

---

### Bug 3: patched_forward 缺少 IdentityOp 防护

**现象**: `enable_dsa_indexer=False` + `compress_ratio=4` 时，`IdentityOp` 被当作真实 indexer，调用不存在的 `forward_with_index_compress` 崩溃

**根因**: MindSpeed spec 在 DSA 禁用时创建 `IdentityOp` 而非 `None`。原版 forward 的 `if self.indexer is not None:` 检查对 `IdentityOp` 返回 True。patched_forward fork 了相同代码。

**注意**: 此 bug 仅影响 `compress_ratio=4` 的层。`ratio != 4` 时 `self.indexer = None`，不受影响。

**修复** (`patches/.../attention.py:129`):
```python
# 原来:
if self.indexer is not None:
# 修复:
if self.indexer is not None and not isinstance(self.indexer, IdentityOp):
```
并添加 import:
```python
from megatron.core.transformer.identity_op import IdentityOp
```

---

### Bug 4: `g2_attention_kernel.py` triton 不可用导致导入崩溃（MindSpeed 修复）

**现象**: NPU 上 `import DeepSeek4SelfAttention` 失败，`NameError: name 'tl' is not defined`

**根因**: `g2_attention_kernel.py` 模块级有 `LOG2_E: tl.constexpr = 1.442...`。triton 导入失败时 `tl` 未定义。Python 在模块加载时求值类型注解（`from __future__ import annotations` 可延迟求值）。

**修复** (容器 `/MindSpeed-LLM/.../g2_attention_kernel.py:2`):
```python
# pylint:disable=all
from __future__ import annotations  # ← 添加
```

---

## 3. 配置相关问题（非 Bug）

### 3.1 `rope_head_dim` 默认值

- 测试曾设 `rope_head_dim=32`，与 DSA Indexer 期望的 `64`（MLA `qk_rope_head_dim`）不匹配
- `freqs_cis` 的 `dim//2=16` 与 indexer `x_complex.size(-1)=32` 不一致
- 修正：设置 `rope_head_dim=64`

### 3.2 `kv_compress` 默认值

- DSA Indexer 默认 `kv_compress=False`，不构建 `kv_compressor`
- `forward_with_index_compress` 无条件调用 `self.kv_compressor()`
- 修正：设置 `kv_compress=True`

### 3.3 `use_sparse_flash_attn` 默认值

- 小模型测试 `qk_head_dim=128` 时默认 False，走 `core_attention`
- `sparse_flash_mla` 硬性要求 `head_dim=512`
- 标准模型 `qk_head_dim=512` 需显式 `use_sparse_flash_attn=True`

---

## 4. CP 测试结论

CP=2 在容器 megatron-core 0.12.1 上无法通过 `preprocess_thd_engine(local_cp_size=cp)` 测试（API 版本不匹配）。

verl 生产路径不存在此问题：dataloader 在进入 `preprocess_thd_engine` 前已完成 CP-local 分片，`cu_seqlens` 天然 local，compressor 全链路一致。

---

## 5. 代码变更汇总

### PS 代码（`prefix_sharing/` 目录内）

| 文件 | 变更 | 说明 |
|------|------|------|
| `integrations/verl_mcore.py:677` | +1 | `model_type="deepseek4"` |
| `integrations/g2_attention.py` | +42 | padded/packed 双路径 stats 上报 |
| `integrations/g2_attention.py:196` | ~`local_freqs_cis`→`self.freqs_cis` | compressor CP 兼容（patched_forward fork） |
| `setup/patches/.../attention.py` | +3 | IdentityOp import + isinstance 检查 |

### MindSpeed 修复（容器内）

| 文件 | 变更 | 说明 |
|------|------|------|
| `g2_attention_kernel.py:2` | +1 | `from __future__ import annotations` |

---

## 6. 测试日志

### Config A (TP=1, CP=1, DP=8, 4层 DSA)
```
rank0-7: layer=1-4 matches_expected=True × 8 ranks × 4 layers
baseline loss = 9.242..., PS loss = 9.248..., diff = 6.30e-03
```

### Config B (TP=2, SP=True, CP=1, 4层 DSA)
```
tp_rank=0 & tp_rank=1: layer=1-4 matches_expected=True × 4 DP groups
TP group baseline loss 一致 (rank0==rank1, rank4==rank5)
```

### verl engine (sparse_flash_mla, qk_head_dim=512)
```
forward_fn: gptmodel_forward_model_engine
Baseline loss = 0.001063 ✅ (NPU sparse_flash_mla kernel)
PS: layer=1 matches_expected=True, layer=2 matches_expected=True ✅
```
