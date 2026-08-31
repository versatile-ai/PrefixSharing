# setup/aux-patches — 非宿主参考补丁

本目录包含非 PS 宿主仓的参考补丁（diff 格式），仅作形态记录和追溯用，**不参与 host-patches apply 流程**。

| 文件 | 行 | 覆盖域 | 来源仓 | 来源 commit | 说明 |
|------|-----|--------|--------|-------------|------|
| golden_mbridge.diff | +175/-74 | mbridge (DSV4 bridge weight loading) | [mbridge](https://github.com/tsinghua/mbridge) | `0cd4ae2` | `bridge.py` DSV4 expert FC 名称适配 `linear_fc`→`local_experts` |
| golden_megatron.diff | +54/-8 | Megatron-LM (5 文件) | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | `a845aa7e1` | distrib_optimizer / p2p_comm / JIT / config 修补 |

**非宿主含义**：上述文件所在的源码树（mbridge、Megatron-LM）不属于 PS 宿主仓
（`prefix-sharing-repo`）的管理范围，补丁以 diff 形式归档在此以保持可追溯性，
实际 apply 需在各自仓库上手工执行。

**生成命令**（等效力容器内回放）：
```bash
diff -ruN mbridge/ mbridge-ps/ > golden_mbridge.diff
diff -ruN Megatron-LM/ Megatron-LM-ps/ > golden_megatron.diff
```

**宿主补丁**见 `../host-patches/`。

PS 仓终态(B 系列 2026-08-30):改动1 + fix17b + g2_attention + setup 落位,见 git log --oneline -5
记录日期: 2026-08-30