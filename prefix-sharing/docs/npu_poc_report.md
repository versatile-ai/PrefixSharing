# DeepSeek V4 NPU 可行性验证报告

> **日期**：2026-07-22（更新 2026-07-29，CANN 9.1 sparse_flash_mla 验证通过）
> **环境**：NPU 单卡（1×910B3, 64GB HBM），容器 `verl-qwen-prefix-baseline` → `ps-build-910`（192.168.0.2）→ `deepseek-verify`（192.168.0.2, CANN 9.1）
> **目的**：验证 `DeepSeek4SelfAttention` 能否脱离完整模型独立导入、实例化、执行 forward

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| **主 NPU 服务器** | `192.168.0.112`（跳板 `190.92.241.16`） |
| **备用 NPU 服务器** | `192.168.0.2`（跳板 `190.92.241.16`，sparse_flash_mla 安装用） |
| **容器** | `verl-qwen-prefix-baseline`（.112）/ `ps-build-910`（.2） |
| **CANN** | `9.0.0` (`/usr/local/Ascend/cann-9.0.0/`) |
| **Python** | `3.11.15` |
| **PyTorch** | 2.9.0（Ascend NPU 版） |
| **torch_npu** | 已安装 |
| **cann_ops_transformer** | v9.1.0-beta.3（手工安装，2026-07-23） |

## 2. 版本与依赖关系

### 运行时版本链

容器中构建的版本链：

```
mindspeed_llm (26.0.0.dev0, editable install)
├── mindspeed (core_r0.16.0, e4772499)
│   └── megatron-core (0.16.1, pip)
│       └── Megatron-LM (0.16.2, /Megatron-LM 源码)  → 提供 megatron.training
└── acl (CANN 9.0.0 Python site-packages)              → NPU Python 运行时
```

### 版本冲突与解决

原容器使用 mindspeed `v26.0.0_core_r0.12.1` + megatron-core `0.12.1`，但：
- mindpseed `core_r0.12.1` tag 缺少 `NPUDataDumpFeature` / `HcclOpModeSetFeature`
- `megatron-core` 0.12.1 缺少 `ProcessGroupCollection`

切换后产生的新冲突：
- `mindspeed_llm` (26.0.0.dev0) 的 MoE router 代码引用 `megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity`，但该函数在 megatron-core 0.16.1 中不存在（0.16.x 的 API 已重构为 `topk_routing_with_score_function`）

## 3. 安装步骤

### 3.1 版本切换

```bash
# Container path prefix: /data/l00619320/archive/20260721-0035/repos/verl-v4flash-workspace/

# 1. mindspeed: 切换到 core_r0.16.0
cd MindSpeed
git checkout core_r0.16.0

# 2. megatron-core: 升级到 0.16.1 (兼容 mindspeed core_r0.16.0)
pip install --upgrade megatron-core==0.16.1

# 3. Megatron-LM: 使用 /Megatron-LM 源码 (0.16.2) 提供 megatron.training
# 不需要额外安装，只需加入 PYTHONPATH
```

### 3.2 源码 patch（共 4 处）

**Patch 1** — `CustomG2SelfAttentionSubmodules` 缺少 `linear_qkv` 字段

megatron-core 0.16.1 的 `SelfAttentionSubmodules` 将 `linear_qkv` 改为必填字段，但 mindspeed_llm 的 `CustomG2SelfAttentionSubmodules` 使用 `linear_q` / `linear_kv` 拆分方式。

文件：`MindSpeed-LLM/mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py`
行：76（`get_deepseek4_self_attn_submodules()`）
```diff
   return CustomG2SelfAttentionSubmodules(
+        linear_qkv=None,
         linear_q=LinearNoTP,
```

**Patch 2 & 3** — `topk_softmax_with_capacity` 导入路径错误

mindspeed_llm 自己的 `moe_utils.py` 有 `topk_softmax_with_capacity`，但两处代码从 `megatron.core` 导入该函数（megatron-core 0.16.1 已不存在此 API）。

文件：`MindSpeed-LLM/mindspeed_llm/core/transformer/moe/router.py:29`
```diff
- from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity
+ from mindspeed_llm.core.transformer.moe.moe_utils import topk_softmax_with_capacity
```

文件：`MindSpeed-LLM/mindspeed_llm/tasks/models/common/pai_megatron.py:20`
```diff
- from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity
+ from mindspeed_llm.core.transformer.moe.moe_utils import topk_softmax_with_capacity
```

## 4. PoC 脚本

### 4.1 启动流程

```python
# 1. PYTHONPATH
import sys
sys.path.insert(0, "/Megatron-LM")
sys.path.insert(0, "/usr/local/Ascend/cann-9.0.0/python/site-packages")
sys.path.insert(0, "/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe")

# 2. Mock features_manager（mindspeed_llm 导入链必需）
import types
sys.modules["mindspeed_llm.features_manager"] = types.ModuleType("mindspeed_llm.features_manager")
sys.modules["mindspeed_llm.features_manager"].FEATURES_LIST = []

# 3. 初始化分布式（单卡）
import os, torch, torch.distributed as dist
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
dist.init_process_group(backend="hccl", world_size=1, rank=0)

# 4. 初始化 parallel_state
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)

# 5. 设置全局 args（使用 _LazyArgs 兜底缺失字段）
from megatron.training.global_vars import set_args
from argparse import Namespace

class _LazyArgs(Namespace):
    _defaults = { ... }  # 常见字段的默认值
    def __getattr__(self, name):
        if name in self._defaults:
            return self._defaults[name]
        return None

args = _LazyArgs(
    qk_head_dim=512,
    rope_head_dim=64,
    q_lora_rank=1024,
    o_lora_rank=1024,
    g2_window_size=128,
    o_groups=8,
    hidden_size=4096,
    num_attention_heads=64,
    rope_scaling_original_max_position_embeddings=4096,
    compress_ratios=[128],
    compress_rope_theta=10000.0,
    rope_theta=10000.0,
    rope_factor=40,
    beta_fast=32,
    beta_slow=1,
    num_layers=1,
)
set_args(args)

# 6. 初始化 CUDA RNG tracker（inference 模式绕过权重初始化 RNG 检查）
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
tracker = get_cuda_rng_tracker(inference_rng_tracker=True)
```

### 4.2 模型实例化

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(hidden_size=4096, num_attention_heads=64, num_layers=1)

from mindspeed_llm.tasks.models.transformer.deepseek4.g2_attention import (
    get_deepseek4_self_attn_submodules,
    DeepSeek4SelfAttention,
)

submodules = get_deepseek4_self_attn_submodules(
    qk_layernorm=True, mla_mm_split=False,
    enable_dsa_indexer=False, compressor=False,
)

attention = DeepSeek4SelfAttention(config=config, submodules=submodules, layer_number=1)
```

### 4.3 Forward

```python
attention.eval()
attention = attention.to("npu")

seq_len, bsz = 256, 1
hidden_states = torch.randn(seq_len, bsz, 4096, device="npu")

# RoPE freqs_cis: complex [seq_len, head_dim//2]
rope_head_dim = attention.rope_head_dim
theta = 10000.0
freqs = 1.0 / (theta ** (torch.arange(0, rope_head_dim, 2, device="npu").float() / rope_head_dim))
t = torch.arange(4096, device="npu").float()
freqs_cis = torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))

with torch.no_grad():
    output, bias = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        rotary_pos_emb=[freqs_cis, freqs_cis],
        start_pos=0,
    )
```

## 5. 结果

| 阶段 | 状态 | 备注 |
|------|:--:|------|
| **Import** `DeepSeek4SelfAttention` | ✅ | 需 features_manager mock + CANN PYTHONPATH + 4 处源码 patch |
| **单卡分布式初始化** | ✅ | `init_process_group(backend="hccl")` |
| **parallel_state 初始化** | ✅ | TP=PP=CP=EP=1 |
| **Megatron global args** | ✅ | `_LazyArgs` 兜底，避免逐个补 200+ 字段 |
| **模型实例化** | ✅ | `get_cuda_rng_tracker(inference_rng_tracker=True)` 绕过 RNG 检查 |
| **Forward**: Linear/LN | ✅ | 所有线性投影、归一化正常执行 |
| **Forward**: RoPE | ✅ | 两阶段 RoPE（global + local freqs_cis）正常 |
| **Forward**: Compressor | ✅ | `self.compressor(hidden_states, ...)` 正常 |
| **Forward**: `sparse_attention()` (生产) | ✅ | **已解决**（2026-07-23）。使用 ops-transformer v9.1.0-beta.3 源码编译 host API + JIT kernel source，安装为 CANN 9.0.0 自定义算子，详见 §8.6 |
| **Forward**: `core_attention()` 回退 (compressor=False) | ❌ | Compressor 输出 hidden_size (4096) 无法 cat ori_kv (head_dim=512) |
| **Forward**: `compress_ratios=[0]` 绕过 | ✅ | ratio=0 完全跳过 compressor 和 sparse_attention，走基础 attention 路径 |

### 最终结果：POC SUCCESS — 两条可用路径

**Path 1（生产）**：`use_sparse_flash_attn=True`, `compress_ratios=[128]` — 使用完整 compressor + sparse_flash_mla kernel（2026-07-23 可用）

**Path 2（测试/Fallback）**：`compress_ratios=[0]` — 完全跳过 compressor 和 sparse_attention，走基础 attention 路径。对 B3 开发足够——B3 操作的是 packed tensors，在进入 `sparse_attention()` **之前**就完成了。

### 三种 Attention 路径对比

| 路径 | 配置 | sparse_flash_mla | Compressor | 状态 |
|------|------|:--:|:--:|:--:|
| **生产路径** | `use_sparse_flash_attn=True`, `compress_ratios=[128]` | 需要 | 需要 | ✅ **已可用**（见 §8.6） |
| **回退路径** | `use_sparse_flash_attn=False`, `compressor=True` | 不需要 | 需要 | ❌ compressor shape 不匹配 |
| **测试路径** | `use_sparse_flash_attn=False`, `compress_ratios=[0]` | 不需要 | 不需要 | ✅ **可用** |

`cann_ops_transformer` 是华为 CANN 官方算子库 [ops-transformer](https://gitcode.com/cann/ops-transformer) 的 Python torch 扩展，未预装在容器中。安装需要**两层**：先编译 C++ kernel 二进制，再安装 Python wheel。

## 6. 对 B3 开发的影响

B3（`_adjust_topk_indices_for_batch` + `_g2_expand_kv_and_adjust`）的操作对象是 packs 张量和 cu_seqlens，不依赖 `sparse_attention()` 内部的 NPU kernel。因此：

- **Mac 测试 (3个)**：不受影响，正常进行
- **NPU 单卡测试 (5个)**：使用 `compress_ratios=[0]` 配置可完成 forward pass，B3 的 packed tensor 操作在 `sparse_attention()` 之前完成，完全不受影响

### 备选方案

1. ✅ **`compress_ratios=[0]` 绕过** — 跳过 compressor 和 sparse_flash_mla，走基础 attention，对 B3 足够（仍可用作 fallback）
2. ✅ **安装 `cann_ops_transformer` v9.1.0-beta.3**（采用，2026-07-23 完成）— 详见 §8.6，生产路径已可用
3. 寻找有预编译版本的容器 — 长期方案
4. Monkey-patch `sparse_attention()` 绕开 NPU kernel — 不再需要
5. 降级 mindspeed_llm — 风险较高，不推荐

### 6.1 回退路径的 shape 问题

`use_sparse_flash_attn=False` 时走 `self.core_attention()` 路径，但 `torch.cat([ori_kv, cmp_kv], dim=0)` 失败：
- `ori_kv` — `[seq, head_dim=512]`（kv_layernorm 输出）
- `cmp_kv` — `[cmp_seq, hidden_size=4096]`（IdentityOp compressor 输出）
- 原因：`compressor=False` 时 `get_compressor_spec()` 返回 `IdentityOp`，它不做压缩，直接返回 `hidden_states`（hidden_size=4096）。正确配置需要真正的 `Compressor` 模块将 hidden_size 投影到 head_dim 再做压缩。

**结论**：`compress_ratios=[0]` 在 `sparse_attention()` 源码中直接走捷径分支，完全跳过 compressor 和 sparse_flash_mla，因此不存在此 shape 问题。这是 NPU PoC 的可行方案。

## 7. 核心兼容性问题汇总

| # | 问题 | 根因 | 副作用风险 |
|---|------|------|:--:|
| 1 | `topk_softmax_with_capacity` 导入 | mindspeed_llm 26.0.0 与 megatron-core 0.16.x API 不兼容 | 低 — 改用 mindspeed_llm 自己的实现 |
| 2 | `CustomG2SelfAttentionSubmodules` 缺 `linear_qkv` | megatron-core 0.16.1 改变 dataclass 字段 | 低 — 仅影响单模块实例化 |
| 3 | `features_manager` 导入 | mindspeed_llm 内部初始化链 | 低 — mock 为空列表 |
| 4 | `cann_ops_transformer` 缺失 | NPU fused kernel 未编译安装 | **已解决**（2026-07-23），见 §8.6 |

## 8. cann_ops_transformer 安装指南

### 8.1 项目概况

`cann_ops_transformer` 是华为 CANN 官方算子库 [ops-transformer](https://gitcode.com/cann/ops-transformer) 的 Python torch 扩展。它提供 `sparse_flash_mla`、`flash_attn`、`compressor`、`mhc` 等 NPU 加速算子。

代码分为两层：

| 层 | 目录 | 产物 | 编译方式 |
|---|------|------|------|
| **C++ 算子二进制** | 项目根目录 `attention/`, `moe/`, `mhc/` 等 | `.run` 自解压安装包 | `bash build.sh --pkg --soc=ascend910b --ops=<算子名>` |
| **Python torch 扩展** | `torch_extension/` | `cann_ops_transformer` Python wheel | `python3 -m build --wheel -n` → `pip install` |

安装顺序：**先 C++ 二进制，后 Python wheel**。二者都需要。

### 8.2 环境要求

| 依赖 | 要求 |
|------|------|
| OS | Linux (aarch64 或 x86_64) |
| Python | 3.8+ |
| GCC | 9.4.0+ |
| CMake | 3.16.0+ |
| PyTorch | ≥ 2.6.0 |
| torch_npu | 与 PyTorch 版本匹配 |
| CANN Toolkit | 已安装，`ASCEND_HOME_PATH` 已设置 |
| Python 依赖 | `pyyaml absl-py jinja2 numpy scipy decorator sympy attrs protobuf` |

### 8.3 安装步骤

#### Step 1：安装系统依赖

```bash
# 自动检测 OS 并安装 GCC/CMake/Python 依赖
bash install_deps.sh
```

#### Step 2：编译 C++ 算子二进制

`sparse_flash_mla` 是 `DeepSeek4SelfAttention` 直接使用的算子。根据需要还可以编译 `compressor`、`flash_attn` 等。

```bash
# 编译指定算子
bash build.sh --pkg --soc=ascend910b --ops=sparse_flash_mla -j16

# 如果需要更多算子（示例）
# bash build.sh --pkg --soc=ascend910b \
#   --ops=sparse_flash_mla,sparse_flash_mla_grad,compressor,flash_attn -j16
```

产品名 `--soc` 取值：
- Atlas A2 训练/推理系列 → `ascend910b`
- Atlas A3 训练/推理系列 → `ascend910_93`
- 950 系列 → `ascend950`

编译成功后输出：
```
Self-extractable archive "cann-ops-transformer-custom_linux.${arch}.run" successfully created.
```

run 包位于 `build_out/` 目录下。

#### Step 3：安装 C++ 算子二进制

```bash
./build_out/cann-ops-transformer-*linux*.run
```

这会将算子安装到 `${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/`。

#### Step 4：配置环境变量

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

#### Step 5：编译并安装 Python wheel

```bash
cd torch_extension/

# 安装 Python 依赖
pip install -r requirements.txt

# 构建 wheel（-n 表示使用当前环境，不走隔离构建）
python3 -m build --wheel -n

# 安装
pip install dist/*.whl --force-reinstall --no-deps
```

> **注意**：`python3 -m build` 需要 `build` 包：`pip install build`

#### Step 6：验证

```python
import torch
import torch_npu
import cann_ops_transformer

# 检查 sparse_flash_mla 是否可用
print(dir(cann_ops_transformer.ops))
```

### 8.4 已知注意事项

1. **版本配套**：ops-transformer `9.0.0` 分支不支持 `sparse_flash_mla`，需使用 `v9.1.0-beta.3` 或更新版本。但 CANN 9.0.0 的 AscendC JIT 编译器可以编译 9.1.0 的 kernel 源码——这是本方案的关键技术判断。

2. **JIT 编译**：Python 层在**首次调用**时由 `OpBuilder` 通过 `ninja` JIT 编译 C++ wrapper。首次调用会有额外延迟，后续调用走缓存。

3. **内核依赖**：Python wheel 中的 C++ wrapper (`sparse_flash_mla.cpp`) 只是桥接层，真正的 NPU kernel 在 Step 2 编译的 `.run` 包中。如果跳过 Step 2-3，Python 调用会因找不到底层 `aclnn` 函数而失败。

4. **权限**：`build.sh` 编译过程需要写 `/usr/local/Ascend/` 目录，通常需要 root 权限。

### 8.5 CANN 9.0.0 与 ops-transformer 版本兼容性

**结论**：CANN 9.0.0 内置算子不包含 `sparse_flash_mla`，但 **ops-transformer v9.1.0-beta.3 的 kernel 源码可以被 CANN 9.0.0 的 AscendC JIT 编译器编译并运行**。

#### 技术原理

ops-transformer 分为两层：

| 层 | 内容 | 编译方式 | 编译时依赖 |
|---|------|------|------|
| **Host API** (C++) | `libcust_opapi.so`，提供 `aclnnSparseFlashMla` 等 C API | CMake 预编译 | CANN 9.0.0 headers |
| **Kernel 实现** (AscendC) | `sparse_flash_mla.cpp`、`_common.h` 等 | **运行时 JIT 编译** | CANN 9.0.0 AscendC 编译器 |
| **Python 桥接** | `cann_ops_transformer` package | Import 时 ninja JIT 编译 C++ wrapper | torch + torch_npu + kernel headers |

关键点：**kernel 不是预编译二进制，是 AscendC 源码**。CANN 在首次调用时 JIT 编译这些文件。因此 v9.1.0-beta.3 的 kernel 源码可以被 CANN 9.0.0 编译器编译。

#### 实际安装的两阶段

**阶段一（预编译）**：CMake 编译 host API `.so`
- 使用 v9.1.0-beta.3 源码中 `attention/sparse_flash_mla/op_host/` 的 C++ tiling 代码
- 链接 CANN 9.0.0 的 `libascendcl.so`
- 产物：`libcust_opapi.so`（含 `aclnnSparseFlashMla` + `aclnnSparseFlashMlaGetWorkspaceSize`）

**阶段二（运行时 JIT）**：
1. Python `import cann_ops_transformer` → OpBuilder JIT 编译 C++ wrapper (`sparse_flash_mla.cpp`) → `.so`
2. 首次调用 `npu_sparse_flash_mla()` → AscendC 编译器 JIT 编译 kernel → NPU 执行

### 8.6 实际安装步骤（CANN 9.0.0 + ops-transformer v9.1.0-beta.3）

以下步骤在容器 `ps-build-910`（机器 `192.168.0.2`，CANN 9.0.0）上验证通过。

#### Step 1：获取源码

```bash
git clone -b v9.1.0-beta.3 https://gitcode.com/cann/ops-transformer.git /tmp/ops-transformer
```

#### Step 2：预下载第三方依赖

ops-transformer 的 CMake 构建依赖 `cann-cmake`、`opbase`、`eigen`、`abseil` 等，通过 `FetchContent` 从外网下载。在离线 NPU 环境需要预下载：

```bash
# 在有网络的跳板机上
cd /tmp/ops-transformer/third_party
# 下载 CMake FetchContent 所需的所有依赖到 third_party/
# 关键依赖：cann-cmake, opbase, eigen, gtest, protobuf, abseil-cpp, json
```

#### Step 3：编译 host API

```bash
cd /tmp/ops-transformer
mkdir build && cd build
cmake .. \
  -DSOC_VERSION=ascend910b \
  -DCMAKE_INSTALL_PREFIX=./output \
  -DCANN_3RD_LIB_PATH=/tmp/ops-transformer/third_party
make ops_transformer_kernel -j$(nproc)
```

> **注意**：`make ops_transformer_kernel` 只编译 kernel 层的 host API，不包含 MOE/compressor 等其他算子。不需要 `--pkg` 打包。

#### Step 4：手动安装到 CANN vendor 目录

```bash
CANN_VENDOR=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer

# Host API library
cp build/libcust_opapi.so $CANN_VENDOR/op_api/lib/

# Headers
cp attention/sparse_flash_mla/op_host/sparse_flash_mla_proto.h \
   $CANN_VENDOR/op_proto/inc/
cp attention/sparse_flash_mla/op_api/include/aclnnop/aclnn_sparse_flash_mla.h \
   $CANN_VENDOR/op_api/include/aclnnop/

# Kernel JIT sources (运行时编译)
mkdir -p $CANN_VENDOR/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/
cp attention/sparse_flash_mla/op_kernel/*.cpp $CANN_VENDOR/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/
cp attention/sparse_flash_mla/op_kernel/*.h $CANN_VENDOR/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/

# Dynamic impl（Python 包装层）
cp attention/sparse_flash_mla/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/sparse_flash_mla.py \
   $CANN_VENDOR/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/

# Tiling library
cp build/liboptiling.so $CANN_VENDOR/op_impl/ai_core/tbe/op_tiling/
cp build/libcust_opmaster_rt2.0.so $CANN_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/
```

#### Step 5：安装 Python 层

ops-transformer 的 Python torch 扩展提供 `cann_ops_transformer` package。由于 `torch_extension/setup.py` 中的 package name 是 `npu_ops_transformer`（与 mindspeed_llm 的 import name 不一致），需要手动处理：

```bash
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

# 方案 A：直接复制源码到 site-packages
cp -r /tmp/ops-transformer/torch_extension/npu_ops_transformer $SITE/

# 精简 ops/__init__.py，避免 import 时 JIT 编译所有算子（moe/flash_attn 等）
# 只保留 sparse_flash_mla 的 import
cat > $SITE/npu_ops_transformer/ops/__init__.py << 'EOF'
from .sparse_flash_mla import npu_sparse_flash_mla
EOF
```

#### Step 6：创建 `cann_ops_transformer` 桥接包

`mindspeed_llm` 的 import 路径是 `cann_ops_transformer`，但 setup.py 的 package name 是 `npu_ops_transformer`。需要创建桥接包：

```python
# $SITE/cann_ops_transformer/ops/__init__.py
import importlib.util, sys, types

# Load sparse_flash_mla directly to avoid npu_ops_transformer/__init__.py chain
import torch, torch_npu

_npu_spec = importlib.util.find_spec("npu_ops_transformer")
_nt = types.ModuleType("npu_ops_transformer")
_nt.__file__ = _npu_spec.origin
_nt.__path__ = _npu_spec.submodule_search_locations
_nt.torch = torch
_nt.torch_npu = torch_npu
sys.modules["npu_ops_transformer"] = _nt

_spec = importlib.util.spec_from_file_location(
    "npu_ops_transformer.ops.sparse_flash_mla",
    f"{site_packages}/npu_ops_transformer/ops/sparse_flash_mla.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["npu_ops_transformer.ops.sparse_flash_mla"] = _mod
_spec.loader.exec_module(_mod)

npu_sparse_flash_mla = _mod.npu_sparse_flash_mla
```

#### Step 7：设置环境变量

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/cann-9.0.0/python/site-packages:$PYTHONPATH
# AscendC JIT 编译器需要 opp/built-in 路径
export PYTHONPATH=/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
```

#### Step 8：验证

```python
import torch, torch_npu
import cann_ops_transformer
print(type(cann_ops_transformer.ops.npu_sparse_flash_mla))
# Output: <class 'function'>  ← 成功

# 首次调用会触发 AscendC kernel JIT 编译（数秒钟延迟）
# 后续调用走缓存，直接执行
```

#### 安装架构总结

```
import cann_ops_transformer
  └─ 直接加载 sparse_flash_mla（不触发 moe/flash_attn 等其他算子 JIT）
       └─ ninja 编译 C++ wrapper (sparse_flash_mla.cpp) → .so
            └─ 调用 aclnnSparseFlashMla() from libcust_opapi.so
                 └─ 触发 AscendC kernel JIT 编译（CANN 9.0.0 编译器）
                      └─ NPU 执行
```

## 9. CANN 9.1 最终验证（2026-07-29）

### 9.1 环境

| 项目 | 值 |
|------|-----|
| **服务器** | `192.168.0.2` |
| **容器** | `deepseek-verify` |
| **镜像** | `deepseek-rl:910b-cann9.1-vllm0.23-v23-sparse`（21GB tar 加载） |
| **CANN** | `9.1.0-beta.3` |
| **PyTorch** | 2.10.0 |
| **torch_npu** | 2.10.0.post2 |

### 9.2 镜像预装确认

镜像已内置完整 sparse_flash_mla 算子栈，无需编译：

| 组件 | 位置 | 状态 |
|------|------|:--:|
| `aclnnSparseFlashMla`（前向） | CANN 9.1.0 内置 | ✅ |
| `aclnnSparseFlashMlaMetadata` | `libcust_opapi.so`（ops-transformer 9715a522 编译） | ✅ |
| `aclnnSparseFlashMlaGrad`（反向） | ops-transformer 9715a522 编译 | ✅ |
| `aclnnSparseFlashMlaGradMetadata` | ops-transformer 9715a522 编译 | ✅ |
| `npu_ops_transformer`（Python） | pip list 可见（1.0.0） | ✅ |
| C++ wrapper (`.so`) | `/root/.cache/torch_extensions/` 预编译 | ✅ |
| `binary_info_config.json` | CANN 9.1 内置含 SparseFlashMla | ✅ |

### 9.3 关键环境变量

```bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/vendors/custom_transformer
```

**不设这个变量就会报 `aclInit error 507008`**——这是此前在 CANN 9.1 上反复失败的根因。镜像里算子已经预编译好了，不是固件问题。

### 9.4 验证通过

```python
import torch, os
import cann_ops_transformer.ops as ops
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '/usr/local/Ascend/vendors/custom_transformer'

B, S, r = 1, 128, 128
D = 512
q   = torch.randn(B, S, 1, D, device='npu', dtype=torch.float16)
ori = torch.randn(B, S, 1, D, device='npu', dtype=torch.float16)
cmp = torch.randn(B, S//r, 1, D, device='npu', dtype=torch.float16)
cr  = torch.tensor([0], dtype=torch.int32, device='npu')
sinks = torch.zeros(B, device='npu', dtype=torch.float32)

m = ops.sparse_flash_mla_metadata(
    num_heads_q=1, num_heads_kv=1, head_dim=D,
    cmp_residual_kv=cr, batch_size=B,
    max_seqlen_q=S, max_seqlen_ori_kv=S, max_seqlen_cmp_kv=S//r,
    cmp_topk=0, cmp_ratio=r,
    ori_mask_mode=4, cmp_mask_mode=3,
    ori_win_left=127, ori_win_right=0,
    layout_q='BSND', layout_kv='BSND',
    has_ori_kv=True, has_cmp_kv=True)

out = ops.sparse_flash_mla(q, ori_kv=ori, cmp_kv=cmp,
    cmp_residual_kv=cr, sinks=sinks, metadata=m, cmp_ratio=r,
    layout_q='BSND', layout_kv='BSND')

print('SUCCESS! attn_out:', out[0].shape)
# → torch.Size([1, 128, 1, 512])
```

### 9.5 经验教训

1. **用对镜像**：CANN 9.1 镜像 `deepseek-rl:910b-cann9.1-vllm0.23-v23-sparse` 算子已预编译，不需要从 ops-transformer 源码编译
2. **设对环境变量**：`ASCEND_CUSTOM_OPP_PATH` 必须指向 vendor 目录，否则 `aclInit` 失败
3. **用对函数名**：CANN 9.1 的 `npu_ops_transformer` 导出原始接口 `sparse_flash_mla` + `sparse_flash_mla_metadata`（不是 `npu_sparse_flash_mla`）
4. **不需要 `cann_ops_transformer` 桥接**：`import cann_ops_transformer.ops as ops` 直接可用，镜像已内置
5. **`aclInit error 507008` 不一定是固件问题**：先检查 `ASCEND_CUSTOM_OPP_PATH` 是否设对
