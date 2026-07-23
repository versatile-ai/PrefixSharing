# DeepSeek V4 PrefixSharing 开发复盘与设计一致性审查

> **日期**：2026-07-23

## 一、开发过程中遇到的问题与解决方法

### 问题 1：原方案 Attention 输出长度假设错误

**发现过程**：在 NPU 可行性验证阶段，通过源码分析和数据流跟踪发现 Attention 输出长度 = Q 长度 = S，而非原方案假定的 P+S。

**影响**：Hook D（attn_o 扩展）、TransformerLayer patch、residual/post/comb 存储、功能组 D 全部作废。

**解决**：删除错误组件，重写设计为 v2。

### 问题 2：三件套版本不兼容

**现象**：`verl-qwen-prefix-baseline` 容器中 mindspeed=0.16.0、megatron-core=0.16.2、mindspeed_llm=26.0.0.dev 三者互相不兼容，缺少 `NPUDataDumpFeature`、`topk_softmax_with_capacity` 等符号。

**解决**：
1. mindspeed：切换到 `core_r0.16.0` 分支 + patch 3 个 Feature 类
2. megatron-core：保持 0.16.x（与 `/Megatron-LM` 一致）
3. mindspeed_llm：PYTHONPATH 直接加载源码 + 4 处源码 patch

### 问题 3：DSv4 推理镜像缺少训练框架

**现象**：`dsv4-dspark-final`、`aura:910b-cann9.0.0-dsv4-v4` 等镜像不含 `mindspeed_llm`，只有推理组件。

**解决**：在跳板机 `git clone` mindspeed_llm 源码后 scp 到 NPU 容器，PYTHONPATH 方式加载。

### 问题 4：cann_ops_transformer 缺失

**现象**：`DeepSeek4SelfAttention.forward` 中 `self.sparse_attention()` 调用 `npu_sparse_flash_mla()`，内部执行 `import cann_ops_transformer`，CANN 9.0.0 未预装。

**解决**：使用 ops-transformer `v9.1.0-beta.3` 源码，CMake 编译 host API + JIT kernel source，手动安装到 CANN vendor 目录。CANN 9.0.0 的 AscendC JIT 编译器可编译 9.1.0 的 kernel 源码。

### 问题 5：ctx.plan vs ctx.prefix_sharing_plan 命名不一致

**现象**：`_g2_kv_store_or_expand` 中使用 `ctx.plan`，但真实 `PrefixSharingRuntimeContext` 属性名为 `prefix_sharing_plan`。

**解决**：修改 `_g2_kv_store_or_expand` 和所有 MockContext 测试使用 `prefix_sharing_plan`。

### 问题 6：compressed boundary（跨边界压缩块）

**现象**：当 `P % compress_ratio != 0` 时，`P//r + S//r ≠ (P+S)//r`，压缩块丢失。

**解决**：Phase 1 断言 `P % compress_ratio == 0`，只支持对齐 prefix。跨边界压缩块重算留 Phase 2。

### 问题 7：NPU 测试需要 Megatron 分布式初始化

**现象**：单卡测试仍需 `dist.init_process_group(backend="hccl")` + `parallel_state.initialize_model_parallel()`，且需要手动注入全局 args。

**解决**：使用 `_LazyArgs` 兜底缺失字段 + `set_args()` + `get_cuda_rng_tracker(inference_rng_tracker=True)`。

---

## 二、设计与实现一致性审查

审查基准：`prefix_sharing_deepseek4_design_v2.md`（设计文档）

### 2.1 Typed Store — ✅ 一致

| 设计要求 | 实现 |
|---------|------|
| `StoredG2Activation` 4 fields: kv, kv_compress, indexer_k, stored_len | ✅ `prefix_store.py:148-169` |
| `G2AttentionStore` base + typed wrapper 模式 | ✅ `prefix_store.py:172-215` |
| Store 字段可 None（增量存储） | ✅ |
| `_create_store()` 工厂按 model_type 选择 | ✅ `context.py` |

### 2.2 Hook 插入点 — ✅ 一致

| 设计要求 | 实现 |
|---------|------|
| 在 Phase 3 和 Phase 4 之间插入 | ✅ `attention.py` Phase 3 后、Phase 4 前 |
| Provider 存 reuser 扩 | ✅ `_g2_kv_store_or_expand` 分支 |
| 不对 attention output 做扩展 | ✅ |
| Module 按 `ctx.store` 类型判断 | ✅ `isinstance(ctx.store, G2AttentionStore)` |

### 2.3 Topk 重算 — ⚠️ 基本一致（ratio=4 待完成）

| 设计要求 | 实现 |
|---------|------|
| ratio=128 纯位置重算 | ✅ 已实现并测试 |
| ratio=4 reuser query × expanded indexer_k | ⚠️ 框架已就位，`q_r/w_r/x` 需从 patched_forward 传入 |
| cu_seqlens 偏移 | ✅ 调用 `_adjust_cu_seqlens_for_batch` |

**判定**：ratio=4 路径的实现留了 TODO（`q = ...`），需要 patched_forward 把 Phase 2 的 `query_index`、`weights`、`dsa_hidden` 传下来。代码骨架正确，是**补全**而非**重写**。

**建议**：改代码——在 `_g2_kv_store_or_expand` 增加可选参数 `q_r`、`w_r`、`x`、`mask`，patched_forward 分传入。

### 2.4 Compressed Boundary — ✅ 一致

| 设计要求 | 实现 |
|---------|------|
| Phase 1 断言 `P % compress_ratio == 0` | ✅ `g2_attention.py:158-160` |
| Phase 2 跨块重算 | ⬜ 延期 |

### 2.5 Patch 注入 — ✅ 一致

| 设计要求 | 实现 |
|---------|------|
| 1 个 PatchSpec | ✅ `PATCH_SET` 含 1 个条目 |
| 最小化 fork | ✅ 复制 Phase 1-5 编排代码，不修改计算逻辑 |
| CompatMatrix 条目 | ✅ `mindspeed_deepseek4` 已注册 |

### 2.6 训练集成 — ⚠️ 实现较轻

| 设计要求 | 实现 |
|---------|------|
| `wrap_forward_step` | ✅ 已实现框架代码 |
| prefix-last restore | ✅ 复用 `restore_reuser_prefix_columns_2d` |
| **pre-trim capability gate** | ❌ 未实现 |

**判定**：gate 方案讨论后认为当前不需要（ratio=4 与 128 统一处理），但设计文档 v2 中仍有此节。**建议改设计**——更新 §4.5 说明 Phase 1 不需要 gate，逻辑已由 `_g2_kv_store_or_expand` 内部分支处理。

### 2.7 核心不变量 — ✅ 全部满足

| 不变量 | 验证 |
|--------|------|
| 1. suffix-only | ✅ attention output = Q length = S，无 P+S 假设 |
| 2. 只扩 key-side | ✅ 只扩 kv/kv_compress/indexer_k |
| 3. KV 不 detach | ✅ 无 `.detach()` 调用 |
| 4. 精度一致性 | ✅ NPU 测试 output parity 通过 |
| 5. Phase 1 对齐 prefix | ✅ 断言已实现 |

---

## 三、需要改代码 vs 改设计

| # | 不一致项 | 结论 |
|---|---------|------|
| 1 | ratio=4 topk 路径 `q=...` | **改代码**：补参数传递 |
| 2 | pre-trim gate 未实现 | **改设计**：标为 Phase 2 可选 |
| 3 | compat_matrix 版本号 (0.16.1 vs 容器实际 0.16.2) | **改代码**：放宽为 `>=0.16.1` 或增加 0.16.2 条目 |

---

## 四、文件清单（实现 vs 设计对照）

| 设计文件 | 实现文件 | 状态 |
|---------|---------|:--:|
| `core/prefix_store.py` | 同 | ✅ |
| `backends/g2_attention_utils.py` | 同 | ✅ |
| `integrations/g2_attention.py` | 同 | ✅ |
| `integrations/g2_batch.py` | 同 | ✅ |
| `setup/patches/mindspeed_deepseek4/__init__.py` | 同 | ✅ |
| `setup/patches/mindspeed_deepseek4/attention.py` | 同 | ✅ |
| `setup/compat_matrix.py` | 同 | ✅ |
| `integrations/g2_transformer.py` | 无 | ✅ 正确删除 |
| `setup/patches/mindspeed_deepseek4/transformer.py` | 无 | ✅ 正确删除 |

---

## 五、测试覆盖

| 测试类别 | 数量 | 环境 |
|---------|:--:|------|
| Store 生命周期 | 7 | Mac |
| Store/expand 核心逻辑 | 13 | Mac |
| Topk 重算 | 4 | Mac |
| Patch + Forward | 6 | NPU |
| **合计** | **30** | |
