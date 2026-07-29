# sparse_flash_mla CANN 9.0.0 环境问题分析

> **日期**：2026-07-27
> **环境**：`ps-build-910` 容器（192.168.0.2），CANN 9.0.0，Atlas 910B
> **目的**：记录 sparse_flash_mla 从安装到调用遇到的全部问题、分析和尝试

## 1. 目标

在 CANN 9.0.0 环境上使 `npu_sparse_flash_mla`（DeepSeek V4 的 NPU 融合 MLA attention kernel）可用，以支持 ratio=128/4 精度验证测试。

## 2. 已完成的安装步骤

使用 ops-transformer **v9.1.0-beta.3** 源码（CANN 9.0.0 分支不含 sparse_flash_mla）。

### 2.1 Host API 编译（✅ 成功）

```bash
cd /tmp/ops-transformer && mkdir build && cd build
cmake .. -DSOC_VERSION=ascend910b -DCANN_3RD_LIB_PATH=/tmp/ops-transformer/third_party
make ops_transformer_kernel -j4
```

产物：
- `libcust_opapi.so` — 含 `aclnnSparseFlashMla` + `aclnnSparseFlashMlaGetWorkspaceSize`
- `libes_transformer_cust.so` — 含 `EsSparseFlashMla`（Engine 层包装）
- `libproto_transformer_cust.so` — SparseFlashMla 算子原型
- `libop_host_aclnn.so` / `libop_host_aclnnExc.so` / `libop_host_aclnnInner.so` — host 侧 ACL NN 桥接
- `liboptiling.so` / `libcust_opmaster_rt2.0.so` — tiling 运行时

### 2.2 文件安装（✅ 成功）

安装到 `/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/`：

```
op_api/lib/
  libcust_opapi.so
  libes_transformer_cust.so
  libproto_transformer_cust.so
  libop_host_aclnn.so / libop_host_aclnnExc.so / libop_host_aclnnInner.so

op_api/include/aclnnop/
  aclnn_sparse_flash_mla.h

op_proto/inc/
  sparse_flash_mla_proto.h

op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/
  sparse_flash_mla.cpp          (AscendC 内核源码)
  sparse_flash_mla_common.h
  sparse_flash_mla_metadata.h
  sparse_flash_mla_template_tiling_key.h

op_impl/ai_core/tbe/custom_transformer_impl/dynamic/
  sparse_flash_mla.py            (TBE 动态编译入口)

op_impl/ai_core/tbe/op_tiling/
  liboptiling.so
  lib/linux/aarch64/libcust_opmaster_rt2.0.so
```

### 2.3 Python 桥接包（✅ 成功）

创建了 `cann_ops_transformer` bridge package，直接加载 sparse_flash_mla 模块（绕过 npu_ops_transformer 的完整 import 链以避免触发 MOE/flash_attn 等全部算子的 JIT 编译）。

```python
import cann_ops_transformer
fn = cann_ops_transformer.ops.npu_sparse_flash_mla
# → <class 'function'>  ✅ 可导入
```

C++ wrapper（`sparse_flash_mla.cpp`）通过 ninja JIT 编译为 `.so`，编译和链接均成功。

## 3. 第一阶段错误：Import 正常但首次调用失败

### 3.1 错误信息

```
RuntimeError: call aclnnSparseFlashMla failed, detail:
  Get regInfo failed, The binary_info_config.json of socVersion [ascend910b]
  does not support opType [SparseFlashMla].
  Check nnopExecutor != nullptr failed
```

### 3.2 调用路径分析

```
Python: npu_sparse_flash_mla(q, kv, ...)
  → op_module.npu_sparse_flash_mla (C++ wrapper, ninja JIT 编译)
    → aclnnSparseFlashMla() from libcust_opapi.so
      → aclnnSparseFlashMlaGetWorkspaceSize() ✅ (tensor 创建成功)
      → nnopExecutor 创建 ❌ (kernel 未注册)
```

关键观察：`aclnnSparseFlashMlaGetWorkspaceSize` **成功执行**了（输出了 atten_out 和 softmax_lse tensor 的构造日志），说明 host API 库加载和符号解析都正常。失败在最后一步——`nnopExecutor` 的创建需要 kernel 在平台配置中注册。

### 3.3 根因定位

`aclnn`（AscendCL Neural Network）runtime 通过 `binary_info_config.json` 查找算子实现。CANN 9.0.0 内置的 `ops_transformer/binary_info_config.json` 包含 `SparseFlashAttention`、`KvQuantSparseFlashAttention` 等，但 **不含 `SparseFlashMla`**（该算子为 9.1.0 新增）。

```
/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_transformer/binary_info_config.json
  ├── SparseFlashAttention    ✅  有预编译 .o 和 .json
  ├── KvQuantSparseFlashAttention ✅
  └── SparseFlashMla          ❌  不存在
```

## 4. 尝试的解决方案及结果

### 4.1 尝试 1：修改 built-in `binary_info_config.json` 添加 SparseFlashMla 条目

将 SparseFlashMla 的算子元数据（inputs/outputs/attrs）添加到内置 config 中，`binaryList` 设为空数组（表示 JIT 编译）。

**结果**：❌ 无效。`aclnn` runtime 仍然报同样的错误。

### 4.2 尝试 2：创建 vendor 级 `binary_info_config.json`

在 vendor 路径创建：
`/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_impl/ai_core/tbe/config/ascend910b/ops_transformer/binary_info_config.json`

**结果**：❌ 无效。同样错误。

### 4.3 尝试 3：创建独立算子 JSON 配置文件

参考 `mla_prolog.json` 的格式，创建：
`/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_impl/ai_core/tbe/config/ascend910b/ops_transformer/sparse_flash_mla.json`

包含完整的 `binList` 结构定义。

**结果**：❌ 无效。`aclnn` runtime 读的是 `binary_info_config.json`，不是独立 JSON。

### 4.4 尝试 4：使用 `ccec`（AscendC 编译器）手动编译内核

目的：生成预编译的 `.o` 和 `.json` 文件，填入 `binary_info_config.json` 的 `binaryList`。

```bash
ccec -O2 -std=c++17 \
  -I$ASCEND_HOME/include \
  -I$ASCEND_HOME/include/ascendc/basic_api \
  -I$ASCEND_HOME/include/ascendc/highlevel_api \
  ... \
  sparse_flash_mla.cpp -o sparse_flash_mla.o
```

**结果**：❌ 编译失败。

```
fatal error: 'kernel_tpipe.h' file not found
#include "kernel_tpipe.h"
         ^~~~~~~~~~~~~~~~
```

`kernel_tpipe.h` 是 AscendC Highlevel API 的头文件，**在 CANN 9.0.0 中不存在**。该头文件是 CANN 更新版本（推测 9.1.0+）引入的，sparse_flash_mla 内核源码依赖它。

### 4.5 尝试 5：ops-transformer `make package` 完整构建

```bash
cd /tmp/ops-transformer
bash build.sh --pkg --soc=ascend910b --ops=sparse_flash_mla -j4
```

**结果**：❌ 失败。`gmake: *** [Makefile:156: all] Error 2`，因三个独立错误：
1. `ascend_protobuf_build_transformer-patch` — protobuf patch 失败
2. `generate_es_math_whl` — 尝试从 pypi.org 下载 setuptools（无外网）
3. `generate_ops_info_ascend910b` — 输入 `.ini` 文件不存在

### 4.6 尝试 6：绕开 es_math 和 protobuf，单独跑内核相关步骤

绕过方法：
- 创建假的 `whl_generated.flag` 文件（跳过 math wheel）
- 创建假的 protobuf stamp 文件（跳过 protobuf 构建）

**结果**：`make all` 推进到 56%，只剩 `generate_ops_info_ascend910b` 失败。

### 4.7 尝试 7：用 `--jit` 模式构建（运行时 JIT，不预编译内核）

```bash
bash build.sh --jit --soc=ascend910b --ops=sparse_flash_mla -j4
```

**结果**：❌ 与 `--pkg` 模式同样的 `all` 失败。JIT 模式不解决 `binary_info_config.json` 生成问题。

### 4.8 尝试 8：手动生成 `aic-ascend910b-ops-info.json`

从已有的 `aic-ascend950-ops-info.ini`（含 SparseFlashMla）手动生成：

```bash
cp autogen/aic-ascend950-ops-info.ini autogen/aic-ascend910b-ops-info.ini
python3 parse_ini_to_json.py ... autogen/aic-ascend910b-ops-info.json
# → Compile op info cfg successfully ✅
```

产物 `aic-ascend910b-ops-info.json` 包含 `{"SparseFlashMla": {...}}`。

**关键发现**：该文件是**算子能力声明**（告诉 CANN "SparseFlashMla 在 ascend910b 上存在"），但**不是** `binary_info_config.json`。aclnn runtime 执行时需要的是后者（包含内核实现路径）。安装此文件后，错误不变——仍报 `binary_info_config.json does not support opType [SparseFlashMla]`。

**结论**：`aic-*-ops-info.json`（算子能力声明）和 `binary_info_config.json`（内核实现注册）是两层独立配置。`--jit` 模式能生成前者但无法生成后者。

## 5. 结论与需求

### 5.1 技术瓶颈

| 尝试 | 结果 | 阻塞原因 |
|------|:--:|------|
| Import Python bridge | ✅ | — |
| C++ wrapper JIT 编译 | ✅ | — |
| aclnn host API 调用 | ❌ | `binary_info_config.json` 不含 `SparseFlashMla` |
| 修改内置 config | ❌ | aclnn 不识别 JIT-only 条目（需要真实 .o+.json） |
| 创建 vendor config | ❌ | 同上 |
| ccec 手动编译内核 | ❌ | `kernel_tpipe.h` 在 CANN 9.0.0 中不存在 |
| ops-transformer make package | ❌ | protobuf patch 失败 + 无外网下载 math 依赖 |

**核心矛盾**：`aclnn` runtime 要求算子有预编译的 `.o` 和 `.json` 文件注册在 `binary_info_config.json` 中。但 CANN 9.0.0 的 AscendC 编译器缺少编译 sparse_flash_mla 所需的 Highlevel API 头文件（`kernel_tpipe.h`）。

### 5.2 潜在解决路径

#### 路径 A：换 CANN 9.1.0+ 容器

CANN 9.1.0+ 原生支持 `SparseFlashMla`，内置 `binary_info_config.json` 包含该算子，AscendC 编译器包含所需的 highlevel API 头文件。无需任何手动安装。

- 优点：最省时间，零风险
- 缺点：需要新的 NPU 环境和容器
- 关键字：查找 `cann-9.1.0` 或 `ascend-toolkit-9.1.0` Docker 镜像

#### 路径 B：绕过 aclnn，走 TBE JIT 路径

CANN 的 TBE（Tensor Boost Engine）路径支持运行时 JIT 编译 AscendC 内核，不经过 `aclnn` 的 `binary_info_config.json` 检查。`dynamic/sparse_flash_mla.py` 已经注册了 `@tbe_register.register_operator("SparseFlashMla")`。

需要修改 Python 调用链：
- 当前：`OpBuilder.load()` → C++ wrapper `.so` → `aclnnSparseFlashMla()` → ❌
- 改为：TBE `compile_op()` → 直接编译 AscendC 源码 → 加载执行

关键文件：
- CANN 内置：`/usr/local/Ascend/cann-9.0.0/python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py`
- 动态实现：`op_impl/ai_core/tbe/custom_transformer_impl/dynamic/sparse_flash_mla.py`
- 内核源码：`op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/sparse_flash_mla.cpp`

- 优点：不换环境，利用已有的 JIT 能力
- 缺点：需要修改算子调用路径，不确定 TBE 路径是否有 CANN 版本兼容问题

#### 路径 C：修复 ops-transformer `make package` 构建

1. 修复 protobuf patch 问题（正确的 third_party 配置）
2. 跳过 es_math wheel 构建（设置环境变量或修改 build.sh）
3. 仅编译 sparse_flash_mla 内核（`--ops=sparse_flash_mla`）
4. 安装生成的 `.o` 和 `.json` 到 vendor kernel 目录

- 优点：走标准路径，后续其他算子也可复用
- 缺点：需要调试 ops-transformer 构建系统，可能需要外网或预下载更多依赖

#### 路径 D：确认 kernel_tpipe.h 的来源

`kernel_tpipe.h` 可能在 CANN 的某些未安装的包中。需要确认：
- CANN 9.0.0 是否有单独的 AscendC Highlevel 开发包？
- `Ascend-cann-toolkit-dev` 或 `Ascend-cann-ascendc-dev` RPM/DEB 包？
- 该头文件是否可以从 CANN 9.1.0 复制到 9.0.0 使用？（可能不兼容）

### 5.3 当前进展总结

```
cann_ops_transformer 可 import    ✅
C++ wrapper 编译和链接            ✅
libcust_opapi.so 包含 API 符号   ✅
kernel 源码已安装到 vendor 路径  ✅
TBE 动态编译入口已安装           ✅
Engine 层包装已安装             ✅
aclnn 算子注册（binary_info）   ❌  阻塞
内核二进制编译（ccec）          ❌  kernel_tpipe.h 缺失
ops-transformer 完整构建        ❌  依赖和外网问题
```

### 5.4 对验证计划的影响

- Task 1（Mac 单测）：不受影响 ✅
- Task 0（conftest.py）：已完成 ✅
- Task 2（单卡精度，ratio=128/4）：**阻塞** — 需要 sparse_flash_mla 真实 kernel
- Task 3-5（TP/CP/E2E）：**暂不阻塞** — 但 Task 2 是前置条件

短期内可以用 `compress_ratios=[0]`（绕过 sparse_flash_mla）验证 PS 框架的 store/expand/topk/cu_seqlens 正确性，但无法验证生产路径的 sparse attention kernel 精度。
