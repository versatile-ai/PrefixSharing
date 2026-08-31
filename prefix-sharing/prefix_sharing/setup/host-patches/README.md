# setup/host-patches — PS 宿主补丁包

本目录为 dsv4-mini prefix-sharing 生产补丁包的**宿主三件**（B 系列落位，2026-08-30；M3/M4 调用点归位 2026-08-31）。
apply 工作流:只 apply `ps-core.patch` + `verl-env.patch`（SKILL.md 已同步），`ps-extra.patch`
为 B/C 类归属说明参考件，不参与 apply。

| 件 | 文件数 | 形态 | 用途 | apply |
|----|--------|------|------|-------|
| ps-core.patch | 15（MS 6 + verl 9） | A 类 19 + B 类 3 + C 类 1（dsa_indexer 同 hunk 交错）+ M3/M4 调用点修复 | 生产必需 | `git apply ps-core.patch` |
| verl-env.patch | 10（verl） | D 类 E1-E10（环境/部署） | 集群部署必需 | `git apply verl-env.patch` |
| ps-extra.patch | 0 | 无独立 hunk，仅 B/C 归属说明 | 参考件 | 不 apply |

**基树**:`verl 809f2d8f` + `mindspeed_llm 99f7fc1d`（部署树核实，2026-08-30）。
**验证**:25 文件 scratch repo `git apply --check` + apply → 生产逐字节 25/25 OK；+473/-89
（M12 新文件 +308 另计；M3/M4 g2_attention.py +8/-2 另计）。详细分类见 `experiments/dsv4-mini/patch-bundle/host-patches-spec.md`。

**非宿主参考件**（mbridge / Megatron-LM）见 `../aux-patches/`（README 注明来源 commit）。

PS 仓终态(2026-08-31):改动1 + fix17b + g2_attention 生产形态 + host-patches 三件落位,见 git log --oneline -6
记录日期: 2026-08-31
