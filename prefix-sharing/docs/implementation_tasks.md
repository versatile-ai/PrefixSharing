# DeepSeek V4 PrefixSharing 实现任务分解

> **日期**：2026-07-23
> **原则**：每个 Task 独立开发、独立验证。必须全部测试通过后再进入下一个 Task。

## 依赖关系

```
Task 1 (Store) ──→ Task 2 (Core) ──→ Task 3 (Topk) ──→ Task 4 (Patch) ──→ Task 5 (集成)
   0.5天             1天               0.5天              1天                 1天
```

## Task 1：修正 StoredG2Activation

**目标**：将原 8 字段裁剪为 4 字段，对齐新设计。

**文件**：

| 文件 | 操作 |
|------|------|
| `core/prefix_store.py` | `StoredG2Activation`：删 attn_o/residual/post/comb/indexer_score，增 indexer_k；`G2AttentionStore.store()` 参数同步 |
| `backends/g2_attention_utils.py` | `_merge_g2_fields`：删 5 分支，增 indexer_k 分支；`_merge_g2_transformer_fields`：整函数删除 |
| `integrations/g2_attention.py` | `_g2_store_per_sequence`：删 attn_o/indexer_score 的 field 处理 |
| `tests/unit_test/test_g2_store.py` | 重写 |

**验证**：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_store_lifecycle` | store → load → 字段一致性（4 fields），close 后抛异常，type guard |
| 2 | `test_store_overwrite_reject` | 无 overwrite 时重复 store 抛 KeyError |
| 3 | `test_store_overwrite_merge` | overwrite=True 时增量 store（kv → kv_compress），已有字段保留 |
| 4 | `test_store_indexer_k` | ratio=4 层 store indexer_k，load 后 shape/dtype 正确 |
| 5 | `test_store_indexer_k_none` | ratio=128 层不存 indexer_k（为 None），不影响其他字段 |
| 6 | `test_merge_g2_fields` | 逐字段更新：kv → kv_compress → indexer_k，旧字段保留，stored_len 取 max |
| 7 | `test_merge_transformer_deleted` | `_merge_g2_transformer_fields` 已被删除，import 报 ImportError |

```bash
cd prefix-sharing && python -m pytest tests/unit_test/test_g2_store.py -q
```

## Task 2：_g2_kv_store_or_expand 核心逻辑

**目标**：实现插入点的核心函数——拆分 packed tensor，按 batch 身份分 provider/reuser 路径。

**文件**：

| 文件 | 操作 |
|------|------|
| `integrations/g2_attention.py` | 新增 `_g2_kv_store_or_expand`；修复 `_g2_store_per_sequence`（删 attn_o/indexer_score）；确认 `_g2_store_with_kwargs` 无 dict 分支；删除 `_g2_expand_attn_output` |
| `tests/unit_test/test_g2_attention.py` | 重写 |

**函数签名**：

```python
def _g2_kv_store_or_expand(
    ctx, kv, kv_compress, indexer_k,
    compress_topk_idxs, packed_seq_params,
    compress_ratio, attention_module,
    start_pos, kv_allgather, sequence_parallel
) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
```

**核心逻辑**（per-batch-index 分支）：

```
Provider:
  _split_by_cu_seqlens → 截 [:valid_len] → store(kv, kv_compress, indexer_k)
  → 返回原始值（不变）

Reuser:
  _split_by_cu_seqlens → load(provider_slot)
  → cat(provider[:P], row[:valid])  # 扩 kv / kv_compress / indexer_k
  → 调 _adjust_cu_seqlens_for_batch   # cu_seqlens 偏移
  → 回存 own_slot                     # transitive reuse
  → 返回扩展后的值
```

**注意**：此 Task 暂不实现 topk 重算（compress_topk_idxs 和 packed_seq_params 保持原值返回）。topk 在 Task 3 补充。

**验证**：

| # | 测试 | 内容 |
|---|------|------|
| 1 | `test_provider_store_kv` | Provider batch_idx=0 的 kv 正确存入，shape=[valid, ...] |
| 2 | `test_provider_store_kv_compress` | Provider 的 kv_compress 正确存入（shape=[valid//r, ...]） |
| 3 | `test_provider_store_indexer_k` | Provider 的 indexer_k 正确存入 |
| 4 | `test_non_provider_skipped` | 非 provider 不触发 store |
| 5 | `test_reuser_expand_kv` | Reuser 扩 kv：shape[0] = P + S，cat 顺序 prefix 在前 suffix 在后 |
| 6 | `test_reuser_expand_kv_compress` | Reuser 扩 kv_compress：shape[0] = P//r + S//r |
| 7 | `test_reuser_expand_indexer_k` | Reuser 扩 indexer_k：shape[0] = P//4 + S//4 |
| 8 | `test_reuser_preserves_own_data` | 扩展后 reuser suffix 部分（后 S 行）与原始输入一致 |
| 9 | `test_return_values_unchanged_for_provider` | Provider 路径返回的 kv/cmp/indexer_k 与入参一致 |
| 10 | `test_transitive_reuse` | Reuser A 扩展后回存，Reuser B 从 A load，结果与从 root provider 一致 |
| 11 | `test_compress_ratio_le1_no_cmp` | compress_ratio≤1 时 kv_compress=None，不报错 |
| 12 | `test_indexer_k_none_for_ratio128` | indexer_k=None 时 reuser 扩展跳过 indexer_k |

```bash
cd prefix-sharing && python -m pytest tests/unit_test/test_g2_attention.py -q
```

## Task 3：Topk 重算 + cu_seqlens 调整

**目标**：在 `_g2_kv_store_or_expand` 的 reuser 分支中补全 topk 重算逻辑。

**文件**：

| 文件 | 操作 |
|------|------|
| `integrations/g2_attention.py` | 在 `_g2_kv_store_or_expand` reuser 分支增加 `_recompute_topk_for_reuser` 调用 |
| `backends/g2_attention_utils.py` | 新增 `_adjust_topk_indices_for_reuser`（ratio=128: 纯位置重算；ratio=4: 用 expanded indexer_k 重打分） |
| `tests/unit_test/test_g2_topk.py` | 新建 |

**ratio=128 逻辑**：

```python
expanded_seqlen = prefix_len + q_len_global
new_idxs = attention_module.get_compress_topk_idxs(
    compress_ratio, bsz, expanded_seqlen,
    start_pos=start_pos, offset=0, cp_shard=kv_allgather)
compress_topk_idxs[batch_idx, :q_len_local, :] = new_idxs[batch_idx, -q_len_local:, :]
```

**ratio=4 逻辑**：

```python
# q_r 和 w_r 从 Phase 2 传入（前一步已产生）
expanded_k = cat(provider.indexer_k[:P//4], k_r[:S//4])
compress_topk_idxs[batch_idx] = attention_module.indexer.forward_with_scores_compress(
    x=dsa_hidden, q=q_r, k=expanded_k, w=w_r,
    mask=attention_mask, packed_seq_params=packed, start_pos=start_pos,
    index_topk=attention_module.indexer.index_topk, offset=offset,
    compress_ratio=compress_ratio)[0]
```

**验证**：

| # | 测试 | 环境 | 内容 |
|---|------|:--:|------|
| 1 | `test_topk_ratio128_shape` | Mac | ratio=128 重算后 shape 保持 [b, S, S//128]（mock get_compress_topk_idxs） |
| 2 | `test_topk_ratio128_value_range` | Mac | 重算后所有有效值在 [0, expanded_seqlen//128) 范围 |
| 3 | `test_topk_ratio4_shape` | Mac | ratio=4 重算后 shape 保持 [b, S, 512]（mock forward_with_scores_compress） |
| 4 | `test_topk_ratio4_expanded_k` | Mac | 传入 forward_with_scores_compress 的 k shape = [(P+S)//4, ...] |
| 5 | `test_cu_seqlens_offset` | Mac | cu_seqlens_kv 偏移正确（已有 B1 测试覆盖） |
| 6 | `test_topk_ratio128_start_pos` | Mac | start_pos>0 时重算正确（透传 start_pos） |
| 7 | `test_topk_ratio128_equivalence` | NPU | ratio=128 全量 forward vs 扩展路径输出一致（compress_ratios=[128]） |
| 8 | `test_topk_ratio4_equivalence` | NPU | ratio=4 全量 forward vs 扩展路径输出一致 |

```bash
# Mac
cd prefix-sharing && python -m pytest tests/unit_test/test_g2_topk.py -q

# NPU (1 卡, compress_ratios=[0] 先跑通，[128]/[4] 等 cann_ops_transformer 就绪后跑)
cd prefix-sharing && python -m pytest tests/integrated_test/optional/test_g2_topk_npu.py -q
```

## Task 4：Patch 注入 + Compat Matrix

**目标**：将 `_g2_kv_store_or_expand` 接入真实的 `DeepSeek4SelfAttention.forward`。

**文件**：

| 文件 | 操作 |
|------|------|
| `setup/patches/mindspeed_deepseek4/attention.py` | 新建：`patch_g2_attention(original_forward)` |
| `setup/patches/mindspeed_deepseek4/__init__.py` | 新建：`PATCH_SET` |
| `setup/compat_matrix.py` | 修改：新增 MindSpeed DeepSeek4 条目 |
| `tests/unit_test/test_g2_compat.py` | 新建（Mac） |
| `tests/integrated_test/optional/test_g2_patch_npu.py` | 新建（NPU） |

**Patch 代码**（最小化 fork，~40 行编排代码 + 5 行插入）：

```python
def patched_forward(self, hidden_states, attention_mask, rotary_pos_emb, ...):
    # Phase 1-3: 原始编排代码照抄
    ...
    # ═══ 插入点 ═══
    ctx = current_prefix_sharing_context()
    if ctx is not None and isinstance(ctx.store, G2AttentionStore):
        indexer_k = key_index if self.compress_ratio == 4 else None
        kv, kv_compress, indexer_k, compress_topk_idxs, packed_seq_params = \
            _g2_kv_store_or_expand(ctx, kv, kv_compress, indexer_k,
                compress_topk_idxs, packed_seq_params,
                self.compress_ratio, self, start_pos,
                self.kv_allgather, self.config.sequence_parallel)
    # ═══════════════
    # Phase 4-5: 原始编排代码照抄
    ...
```

**验证**：

| # | 测试 | 环境 | 内容 |
|---|------|:--:|------|
| 1 | `test_compat_entry_parse` | Mac | CompatEntry 字段值正确 |
| 2 | `test_patch_set_id_unique` | Mac | mindspeed_deepseek4 三元组不与现有条目冲突 |
| 3 | `test_patch_spec_importable` | Mac | PATCH_SET 模块可 import，PatchSpec 格式正确 |
| 4 | `test_patch_no_context_noop` | Mac | context 未激活时 patch 直接调用 original_forward |
| 5 | `test_patch_wrong_store_type_noop` | Mac | store 不是 G2AttentionStore 时走 original_forward |
| 6 | `test_patch_compress_ratio_zero` | NPU | compress_ratios=[0] 时完整 forward 不报错，输出 shape 正确 |
| 7 | `test_patch_compress_ratio_128` | NPU | compress_ratios=[128] 时 provider+reuser forward 等价性 |
| 8 | `test_patch_transitive_reuse` | NPU | 三层 reuser chain 输出一致性 |

```bash
# Mac
cd prefix-sharing && python -m pytest tests/unit_test/test_g2_compat.py -q

# NPU (1 卡)
cd prefix-sharing && python -m pytest tests/integrated_test/optional/test_g2_patch_npu.py -q
```

## Task 5：训练流程集成

**目标**：实现 `wrap_forward_step()`，将 prefix sharing 接入 MindSpeed pretrain 入口。

**文件**：

| 文件 | 操作 |
|------|------|
| `integrations/g2_batch.py` | 新建：`wrap_forward_step()` |
| `tests/integrated_test/optional/test_g2_e2e_npu.py` | 新建（NPU） |

**`wrap_forward_step` 流程**：

```python
def wrap_forward_step(original_forward_step, ps_config):
    def wrapped(data_iterator, model):
        batch = get_batch(data_iterator)
        input_ids = extract_input_ids(batch)
        
        plan = planner.plan(input_ids)
        if not plan.has_sharing:
            return original_forward_step(data_iterator, model)
        
        trimmed = trim_batch(batch, plan)
        state = PrefixSharingRuntimeState(
            plan=plan, layout=build_layout(trimmed),
            model_type="deepseek4", ...)
        
        with prefix_sharing_runtime_context(state) as ctx:
            output = original_forward_step(data_iterator, model)
        
        output = restore_reuser_prefix_columns_2d(output, plan, layout)
        return output
    return wrapped
```

**验证**：

| # | 测试 | 内容 | 预期 |
|---|------|------|------|
| 1 | `test_wrap_no_sharing` | 无共享前缀时返回原始结果 | 与 baseline 一致 |
| 2 | `test_wrap_with_sharing` | 有共享前缀时 forward 不报错，输出 shape 正确 | 无异常 |
| 3 | `test_prefix_last_restore` | Reuser prefix-last logprob 正确恢复 | 与独立计算一致 |

```bash
# NPU (1 卡，小模型)
cd prefix-sharing && python -m pytest tests/integrated_test/optional/test_g2_e2e_npu.py -q
```

## 测试汇总

| Task | Mac 测试 | NPU 测试 | 合计 |
|------|:--:|:--:|:--:|
| Task 1 | 7 | — | 7 |
| Task 2 | 12 | — | 12 |
| Task 3 | 6 | 2 | 8 |
| Task 4 | 5 | 3 | 8 |
| Task 5 | — | 3 | 3 |
| **合计** | **30** | **8** | **38** |

## 注意事项

1. **每个 Task 完成后必须跑全部已有测试**，确认无回归
2. **每个 Task 独立提交**，commit message 格式 `[feat] <中文说明>`
3. **NPU 测试**：先确保 `compress_ratios=[0]` 路径可跑（不需要 sparse_flash_mla），`cann_ops_transformer` 就绪后再跑 ratio=128/4
4. **Compressed Boundary**：Phase 1 断言 `P % compress_ratio == 0`，非对齐场景直接报错。Task 3 的 `_g2_kv_store_or_expand` reuser 分支第一行加入此断言
