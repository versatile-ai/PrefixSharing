# 功能组 B：Attention 注入 — ratio=128 静态路径 任务分解

> **差异回顾**：`self.sparse_attention()` 不可替换，需 fork forward 做 pre/post hook。RoPE 需在扩展前执行（位置对称）。`kv_compress`/`compress_topk_idxs`/`cu_seqlens` 需同步扩展。
>
> **目标**：ratio≤1 和 ratio=128 层的 prefix sharing 精度与 baseline 一致。

---

## 总览

```
B1 ────────→ B2 ────────→ B3 ────────→ B4
工具函数     存储/扩展     KV扩展       Fork Forward
(纯Mac)     (纯Mac)     +topk         +PatchSpec
                         (Mac+NPU)     (NPU only)
```

| Checkpoint | 内容 | 可测性 | 预计时间 | 新增/修改文件数 |
|------------|------|:--:|------|:--:|
| B1 | 工具函数扩展 | Mac ✅ | 半天 | 1 改 + 1 测 |
| B2 | 存储/扩展辅助函数 | Mac ✅ | 1 天 | 1 新 + 1 测 |
| B3 | KV 扩展 + topk 调整 | Mac + NPU | 1 天 | 2 改 + 2 测 |
| B4 | Fork Forward + PatchSpec | Mac + NPU | 1 天 | 3 新 + 1 改 + 2 测 |

---

## B1：工具函数扩展

**目标**：在 `g2_attention_utils.py` 中实现三个纯张量工具函数，所有逻辑在 Mac 上完成并验证。

### B1-1：`_split_by_cu_seqlens`

**文件**：`prefix-sharing/prefix_sharing/backends/g2_attention_utils.py`（扩展）

**功能**：将 packed tensor 按 `padded_lengths` 拆分为 per-sequence 列表。

```python
def _split_by_cu_seqlens(tensor, padded_lengths):
    """将 packed tensor 按 padded_lengths 拆分。

    Args:
        tensor: [total_padded, ...]  packed 张量
        padded_lengths: list[int]  每条序列的 padded 长度

    Returns:
        list[Tensor]  每条序列的完整 padded 行

    Raises:
        ValueError: 当 sum(padded_lengths) != tensor.shape[0]
    """
```

**注意**：`torch_ref.py` 和 `flash_atten_npu.py` 中各有一个私有 `_split_packed`，实现略有不同。B1 创建的 `_split_by_cu_seqlens` 是**独立实现**，基于设计文档的命名约定。后续可考虑统一——但不在本 checkpoint 范围。

### B1-2：`_compute_cmp_lengths`

**功能**：计算 per-sequence 压缩 KV 长度。

```python
def _compute_cmp_lengths(layout, compress_ratio, kv_compress_shape_0):
    """计算压缩 KV 的 per-sequence padded lengths。

    kv_compress 的 dim=0 长度是 sum(valid // ratio)，不能用 Q 路径的 padded_lengths。
    从 valid_lengths 计算，并通过运行时断言检测 compressor 内部对齐导致的长度偏差。

    Args:
        layout: PackedBatchLayout
        compress_ratio: int  (如 128)
        kv_compress_shape_0: int  kv_compress 的实际 dim=0 长度

    Returns:
        list[int]  per-sequence 压缩长度
    """
```

### B1-3：`_adjust_cu_seqlens_for_batch`

**功能**：调整 packed_seq_params 中 reuser 序列的 cu_seqlens_kv。

```python
def _adjust_cu_seqlens_for_batch(packed_seq_params, plan, compress_ratio):
    """调整 packed_seq_params 中所有 reuser 序列的 cu_seqlens_kv。

    使用 dataclasses.replace() 创建新实例（不原地修改）。
    同时处理 cu_seqlens_cmp_kv（如果存在）。

    Args:
        packed_seq_params: Megatron packed_seq_params dataclass
        plan: PrefixSharingPlan
        compress_ratio: int

    Returns:
        新的 packed_seq_params 实例（或 None 当入参为 None）
    """
```

**实现注意 — cu_seqlens 属性优先级**：

Megatron 不同版本的 `packed_seq_params` 中 cu_seqlens_kv 属性名可能不同。按以下优先级查找：

```python
kv_attr = (
    "cu_seqlens_kv_padded"
    if hasattr(packed_seq_params, "cu_seqlens_kv_padded")
    else "cu_seqlens_kv"
)
```

`cu_seqlens_kv_padded`（含 TP padding）优先于 `cu_seqlens_kv`。这是因为 MindSpeed 某些版本在 dataclass 中同时定义了两个字段，优先使用 padded 版本能保证长度与 packed tensor dim=0 一致。

### B1 验证

**文件**：`prefix-sharing/tests/unit_test/test_g2_attention_utils.py`（新建）

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 1 | `test_split_by_cu_seqlens` | 2 序列 packed tensor，验证 per-sequence shapes、valid 截断行为 | shapes 正确，sum 匹配 |
| 2 | `test_compute_cmp_lengths` | compress_ratio=128 时计算，边界情况（非整除）触发断言 | 长度正确，非整除时 assert 命中 |
| 3 | `test_adjust_cu_seqlens_for_batch` | 模拟 packed_seq_params dataclass，设置 3 序列 (provider/ reuser/ provider)，验证 reuser 偏移后的 cu_seqlens | cu_seqlens_kv 偏移正确，cu_seqlens_cmp_kv 也正确 |

**验证命令**：
```bash
cd prefix-sharing
python -m pytest tests/unit_test/test_g2_attention_utils.py -q
```

**准出标准**：3/3 通过，Mac 本地。

---

## B2：存储/扩展辅助函数

**目标**：实现 per-sequence store 和 expand 辅助函数。依赖 B1 + A。

### B2-1：`_g2_store_with_kwargs`

**文件**：`prefix-sharing/prefix_sharing/integrations/g2_attention.py`（新建）

```python
def _g2_store_with_kwargs(store, slot_id, data):
    """将 StoredG2Activation 显式传参存入 store。

    G2AttentionStore.store() 接受 keyword-only arguments，
    StoredG2Activation 是 frozen dataclass，显式传参避免 **dict 解包的类型安全问题。

    不接受 dict——调用方必须先构造 StoredG2Activation 实例，
    保证类型安全和字段完整性。

    Args:
        store: G2AttentionStore
        slot_id: PrefixActivationSlotId
        data: StoredG2Activation
    """
    store.store(slot_id,
        kv=data.kv, kv_compress=data.kv_compress, attn_o=data.attn_o,
        residual_prefix=data.residual_prefix,
        post_prefix=data.post_prefix, comb_prefix=data.comb_prefix,
        indexer_score=data.indexer_score,
        stored_len=data.stored_len, overwrite=True)
```

### B2-2：`_g2_store_per_sequence`

```python
def _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank, field, tensor):
    """逐序列存储 provider 数据到 G2AttentionStore。

    将 packed tensor 按 padded_lengths 拆分，对每个 provider row
    增量存入 store。同一 slot 支持多次调用（先存 kv，再存 kv_compress，再存 attn_o）。

    Args:
        ctx: PrefixSharingRuntimeContext
        layout: PackedBatchLayout
        plan: PrefixSharingPlan
        layer_id: int
        tp_rank: int
        field: str  字段名 ("kv", "kv_compress", "attn_o", "indexer_score")
        tensor: Tensor   packed 张量（可能为 None，直接返回）
    """
```

**核心流程**：
1. `_split_by_cu_seqlens(tensor, padded_lengths)` → per-sequence 列表
2. 遍历 batch_idx → `plan.is_provider(batch_idx)` 过滤
3. `valid_row = row[:valid_len]` 截取有效部分
4. `ctx.store.load(slot_id)` 取已有数据 → `_merge_g2_fields` 合并 → `_g2_store_with_kwargs` 存入

### B2-3：`_g2_expand_attn_output`

```python
def _g2_expand_attn_output(ctx, layout, plan, layer_id, tp_rank, o):
    """逐序列扩展 reuser 的 attention output。

    Provider: 保留原样（截取 valid_len）。
    Reuser: 从 G2AttentionStore 取 provider prefix，拼接在 suffix 前面。
    扩展后回存 attn_o（支持 transitive reuse）。

    Args:
        ctx: PrefixSharingRuntimeContext
        layout: PackedBatchLayout
        plan: PrefixSharingPlan
        layer_id: int
        tp_rank: int
        o: Tensor  [total_padded, b, n_local, 512]  suffix-only attention output

    Returns:
        Tensor  [total_padded + sum(reuser_prefix_lens), ...]  扩展后的 attn_o
    """
```

### B2 验证

**文件**：`prefix-sharing/tests/unit_test/test_g2_attention.py`（新建）

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 1 | `test_g2_store_with_kwargs` | StoredG2Activation → store → load 往返 | 往返后数据一致 |
| 2 | `test_g2_store_per_sequence` | 构造 provider(seq0, len=6) + non-provider(seq1, len=4) 的 packed kv，调用 store | provider kv 正确存入，seq1 不存 |
| 3 | `test_g2_expand_attn_output` | Provider seq0 prefix_len=3, Reuser seq1 suffix_len=3 | 扩展后 shape[0]=9, prefix 来自 provider |
| 4 | `test_transitive_reuse_store` | Reuser A 扩展后回存 attn_o，Reuser B 从 A 读取 | B 的扩展结果与直接从 provider 扩展一致 |

**验证命令**：
```bash
cd prefix-sharing
python -m pytest tests/unit_test/test_g2_attention.py -q
```

**准出标准**：4/4 通过，Mac 本地。

---

## B3：KV 扩展 + topk 调整

**目标**：实现核心 KV/CMP KV 扩展逻辑和 topk indices 调整。Mac 上验证 shape 逻辑，NPU 上验证等价性。

### B3 Mac Mock 策略

`_adjust_topk_indices_for_batch` 依赖 `attention_module.get_compress_topk_idxs()`——这是 MindSpeed 实例方法，Mac 上不可用。Mac 测试采用以下 mock 规范：

**Mock 行为**：`get_compress_topk_idxs(compress_ratio, bsz, expanded_seqlen, start_pos=..., offset=0, cp_shard=...)` 返回一个合成 tensor。

**Mock 返回值规范**：

| 参数 | 合成值 | 说明 |
|------|--------|------|
| shape | `[bsz, expanded_seqlen, expanded_seqlen // compress_ratio]` | ratio=128 静态路径的标准 shape |
| dtype | `torch.int64` | 与 MindSpeed 实现一致 |
| 值域 | `[0, expanded_seqlen // compress_ratio)` 内的随机整数，含 `-1`（mask） | 模拟实际 topk indices 的值域 |
| `-1` 比例 | ~20% | 模拟部分位置被 mask 的情况 |

**Mock 注入方式**：在测试中构造一个简单的 `MockAttentionModule` 对象，仅实现 `get_compress_topk_idxs` 方法。不 mock 整个 `DeepSeek4SelfAttention`。

```python
class MockAttentionModule:
    def get_compress_topk_idxs(self, compress_ratio, bsz, expanded_seqlen,
                                start_pos=0, offset=0, cp_shard=False):
        topk_len = expanded_seqlen // compress_ratio
        idxs = torch.randint(0, topk_len, (bsz, expanded_seqlen, topk_len))
        # 20% mask
        mask = torch.rand(bsz, expanded_seqlen, topk_len) < 0.2
        idxs[mask] = -1
        return idxs
```

**Mac 测试覆盖范围**：Mac 测试验证 shape 正确性和值域约束——确认调整后的 indices shape 正确、所有有效值在合法范围内。**语义等价性**（indices 实际指向正确的 prefix cmp KV 位置）留给 NPU 测试。

### B3-1：`_adjust_topk_indices_for_batch`

**文件**：`prefix-sharing/prefix_sharing/backends/g2_attention_utils.py`（扩展）

```python
def _adjust_topk_indices_for_batch(compress_topk_idxs, plan, layout, compress_ratio, *,
                                    attention_module, start_pos, kv_allgather,
                                    sequence_parallel):
    """调整 packed 中所有 reuser 序列的 compress_topk_idxs。

    ratio≤1: 无压缩，返回原值。
    ratio>1 (无 indexer): 用 expanded seqlen 重新生成。
    ratio>1 (有 indexer): Phase 1 跳过——此函数不会被调用。

    原地修改 compress_topk_idxs 并返回同一引用。
    """
```

**核心逻辑**（per-reuser）：
1. `q_len_local = layout.valid_lengths[batch_idx]`
2. `expanded_seqlen = prefix_len + q_len_global`
3. `new_idxs = attention_module.get_compress_topk_idxs(compress_ratio, bsz, expanded_seqlen, ...)`
4. 原地覆盖：`compress_topk_idxs[batch_idx, :q_len_local, :] = new_idxs[batch_idx, -q_len_local:, :]`

### B3-2：`_g2_expand_kv_and_adjust`

**文件**：`prefix-sharing/prefix_sharing/integrations/g2_attention.py`（扩展）

```python
def _g2_expand_kv_and_adjust(ctx, layout, plan, layer_id, tp_rank,
                              kv, kv_compress, compress_topk_idxs,
                              packed_seq_params, compress_ratio, *,
                              attention_module, start_pos, kv_allgather,
                              sequence_parallel):
    """逐序列扩展 reuser 的 KV/CMP KV + 调整 indices/cu_seqlens。

    Returns:
        (new_kv, new_cmp, new_indices, new_packed)
    """
```

**核心流程**（per-reuser）：
1. `provider_data = ctx.store.load(provider_slot_id)`
2. KV 扩展：`cat([provider_data.kv[:prefix_len], suffix_kv])`
3. CMP KV 扩展：`cat([provider_data.kv_compress[:prefix_len//ratio], suffix_cmp])`（ratio>1 时）
4. 回存 expanded KV（transitive reuse）：`store(own_slot_id, StoredG2Activation(kv=..., kv_compress=..., stored_len=P+S))`
5. 调用 `_adjust_topk_indices_for_batch` + `_adjust_cu_seqlens_for_batch`

### B3 验证

**文件**：
- `prefix-sharing/tests/unit_test/test_g2_kv_expansion.py`（新建，Mac）
- `prefix-sharing/tests/integrated_test/optional/test_g2_attention_npu.py`（新建，NPU）

**Mac 测试**：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 1 | `test_kv_expand_shape` | 构造 provider+reuser packed KV，调用 expand | shape[0] 增加 prefix_len，cat 顺序正确 |
| 2 | `test_topk_adjust_shape` | mock `get_compress_topk_idxs`，验证 indices 取值边界 | 所有有效值在 [0, P+S//ratio) 范围内 |
| 3 | `test_expand_kv_and_adjust_return` | 完整调用返回四元组的 shape 一致性 | new_kv, new_cmp dim=0 匹配，new_indices shape 保持 |

**NPU 测试**（需要 1 卡 NPU + MindSpeed 运行时，单层 Attention，seqlen=128）：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 4 | `test_kv_expansion_equivalence` | Provider 完整计算 vs Reuser KV 拼接 | attention 输出一致（误差 < 1e-6） |
| 5 | `test_cmp_kv_expansion` | CMP KV 扩展后长度正确 | kv_compress cat 后 dim=0 = sum((P+S)//ratio) |
| 6 | `test_topk_adjust_static` | ratio=128 静态路径 expanded seqlen 重生成 indices | 调整后 indices 指向正确 expanded cmp KV 位置 |
| 7 | `test_topk_adjust_start_pos` | 续训场景 start_pos > 0 | indices 调整正确（start_pos 透传） |
| 8 | `test_second_rope_symmetry` | 2nd RoPE 在扩展前执行，encode/decode 位置对称 | 扩展后 o 与 baseline full-sequence o 一致 |

**验证命令**：
```bash
# Mac
cd prefix-sharing
python -m pytest tests/unit_test/test_g2_kv_expansion.py -q

# NPU
python -m pytest tests/integrated_test/optional/test_g2_attention_npu.py -q
```

**准出标准**：
- Mac：3/3 通过
- NPU：5/5 通过（1 卡 NPU，单层 Attention）

---

## B4：Fork Forward + Patch Spec

**目标**：实现 `DeepSeek4SelfAttention.forward` 的 monkey-patch，集成所有 B1-B3 的函数。

### B4-1：PatchSpec 定义

**文件**：`prefix-sharing/prefix_sharing/setup/patches/mindspeed_deepseek4/__init__.py`（新建）

```python
from prefix_sharing.setup.registry import PatchSpec

PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention",
        target_getter=lambda mod: (getattr(mod, "DeepSeek4SelfAttention"), "forward"),
        patch_factory=patch_g2_attention,
        description="DeepSeek4SelfAttention.forward → fork forward + 4 Hook",
    ),
]
```

### B4-2：Fork Forward

**文件**：`prefix-sharing/prefix_sharing/setup/patches/mindspeed_deepseek4/attention.py`（新建）

**前置条件**：实现前需读取目标 MindSpeed 版本的 `g2_attention.py`，确认 forward 的精确签名。

```python
def patch_g2_attention(original_forward):
    """创建 DeepSeek4SelfAttention.forward 的 patch wrapper。

    快速退出（不走 fork）：
    1. context 未激活 → original_forward
    2. ctx.store 不是 G2AttentionStore → original_forward
    3. compress_ratio==4 and indexer is not None → original_forward (skip 层)

    Fork 编排（4 个 Hook 点）：
    - Phase 1: Q/KV 投影 + RoPE → Hook A: Provider 存 KV
    - Phase 2: compress_topk_idxs 生成 → 保留完整 ratio=4 路径代码
    - Phase 3: Compressed KV → Hook B: Provider 存 CMP KV
    - Hook C: Reuser 扩展 KV/CMP KV + 调整 indices/cu_seqlens
    - Phase 4: Sparse Attention（调原 self.sparse_attention()）
    - Phase 5a: 第二次 RoPE（在扩展前执行，freqs 覆盖 suffix-only 的 o）
    - Hook D: Provider 存 post-RoPE attn_o / Reuser 扩展 attn_o
    - Phase 5b: Rearrange + Output Projection
    """
```

**注意**：Fork 代码中**保留完整 ratio=4 路径**（indexer 调用 + DSA loss），当前通过 `_is_skip_layer()` 跳过，为功能组 C 预留。

### B4-3：Compat Matrix

**文件**：`prefix-sharing/prefix_sharing/setup/compat_matrix.py`（修改）

新增条目：
```python
CompatEntry(
    verl=None,
    megatron_core="0.16.1",
    mindspeed="2.2.0",
    patch_set_id="mindspeed_deepseek4",
    notes="DeepSeek V4 standalone pretrain (non-verl path)",
)
```

### B4-4：Compat Matrix 配置验证（Mac）

**文件**：`prefix-sharing/tests/unit_test/test_g2_compat.py`（新建，Mac）

**目标**：在 Mac 上直接验证 CompatEntry 配置的正确性，不依赖 MindSpeed 运行时。

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 1 | `test_compat_entry_parse` | 构造 CompatEntry，验证各字段值 | 字段值与定义一致 |
| 2 | `test_patch_set_id_unique` | 遍历所有已知 CompatEntry，`mindspeed_deepseek4` 不与现有条目共享同一 (verl, megatron_core, mindspeed) 三元组 | 无冲突 |
| 3 | `test_patch_set_exists` | `mindspeed_deepseek4` patch_set 对应的 PATCH_SET 模块可 import | import 成功，PATCH_SET 是 list[PatchSpec] |
| 4 | `test_patch_spec_target` | PatchSpec 的 module_name 和 target 名称格式正确 | module_name 非空，target 是 (class, method) 元组 |

### B4 验证

**文件**：
- `prefix-sharing/tests/unit_test/test_g2_compat.py`（新建，Mac）
- `prefix-sharing/tests/integrated_test/optional/test_g2_patch_npu.py`（新建，NPU）

**Mac 测试**：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 1 | `test_compat_entry_parse` | CompatEntry 构造和字段值 | 见 B4-4 |
| 2 | `test_patch_set_id_unique` | patch_set_id 三元组不冲突 | 见 B4-4 |
| 3 | `test_patch_set_exists` | PATCH_SET 可 import | 见 B4-4 |
| 4 | `test_patch_spec_target` | PatchSpec 格式正确 | 见 B4-4 |

**NPU 测试**（需要 1 卡 NPU + MindSpeed 运行时，单层 Attention，seqlen=128）：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 5 | `test_skip_layer_attention` | ratio=4 + indexer 层走 original_forward | 输出与 baseline 一致，fork 逻辑不执行 |
| 6 | `test_ratio128_full_flow` | ratio=128 层 provider+reuser 完整 forward | attention 输出一致性，Hook A/B/C/D 全部命中 |
| 7 | `test_transitive_reuse` | A reuse B, B reuse C | 深度 reuser 输出与直接从 root provider 扩展一致 |

**验证命令**：
```bash
# Mac
cd prefix-sharing
python -m pytest tests/unit_test/test_g2_compat.py -q

# NPU（1 卡，单层 DeepSeek4SelfAttention）
python -m pytest tests/integrated_test/optional/test_g2_patch_npu.py -q
```

**准出标准**：
- Mac：4/4 通过
- NPU：3/3 通过

---

## 依赖关系图

```
Group A (已完成)
    │
    ▼
B1: g2_attention_utils 工具函数
    │ _split_by_cu_seqlens
    │ _compute_cmp_lengths
    │ _adjust_cu_seqlens_for_batch
    │
    ▼
B2: g2_attention 存储/扩展
    │ _g2_store_with_kwargs
    │ _g2_store_per_sequence
    │ _g2_expand_attn_output
    │
    ▼
B3: KV 扩展 + topk
    │ _adjust_topk_indices_for_batch
    │ _g2_expand_kv_and_adjust
    │
    ▼
B4: Fork Forward + PatchSpec
    │ attention.py (fork forward)
    │ __init__.py (PATCH_SET)
    │ compat_matrix.py (条目)
```

## 文件变更汇总

| Checkpoint | 新增文件 | 修改文件 |
|------------|----------|----------|
| B1 | `tests/unit_test/test_g2_attention_utils.py` | `backends/g2_attention_utils.py` |
| B2 | `integrations/g2_attention.py`<br>`tests/unit_test/test_g2_attention.py` | — |
| B3 | `tests/unit_test/test_g2_kv_expansion.py`<br>`tests/integrated_test/optional/test_g2_attention_npu.py` | `backends/g2_attention_utils.py`<br>`integrations/g2_attention.py` |
| B4 | `setup/patches/mindspeed_deepseek4/__init__.py`<br>`setup/patches/mindspeed_deepseek4/attention.py`<br>`tests/unit_test/test_g2_compat.py`<br>`tests/integrated_test/optional/test_g2_patch_npu.py` | `setup/compat_matrix.py` |

## 测试统计

| 环境 | B1 | B2 | B3 | B4 | 合计 |
|------|:--:|:--:|:--:|:--:|:--:|
| Mac 单元测试 | 3 | 4 | 3 | 4 | **14** |
| NPU 集成测试 | — | — | 5 | 3 | **8** |
| **合计** | **3** | **4** | **8** | **7** | **22** |
