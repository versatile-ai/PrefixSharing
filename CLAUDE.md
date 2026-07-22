# CLAUDE.md — PrefixSharing 项目

## 项目背景

PrefixSharing 是 RL 训练中 prefix KV 复用的插件化框架，通过 monkey-patch 注入 verl + Megatron / MindSpeed 训练流程。

## 开发规范

遵守 `AGENTS.md`（仓库级规范）和 `prefix-sharing/AGENTS.md`（模块级规范）。关键红线：
- **精度一致性优先**：logprob / loss / 梯度必须与 baseline 完全一致
- **KV 不 detach**：缓存 prefix KV 保留完整 autograd 计算图
- **TDD 优先**：新增功能前先写测试
- **提交前必须等用户审核**

## 当前开发：DeepSeek V4 适配

**分支**：`feature/deepseek4-prefix-sharing`

**目标**：在 DeepSeek V4 预训练场景中支持 micro-batch 内 prefix KV 复用。

**架构分层**（四层，严格遵守边界）：
```
prefix-sharing/prefix_sharing/
├── core/          # 框架无关语义（config, detector, planner, store）
├── backends/      # 硬件执行（packed_layout, g2_attention_utils）
├── integrations/  # 框架适配（g2_attention, g2_transformer, g2_batch）
└── setup/patches/ # Monkey-patch（mindspeed_deepseek4/）
```

**开发顺序**（五个功能组，按依赖逐个开发并验证）：
1. **A — 数据存储层**：`G2AttentionStore` + `StoredG2Activation`（无外部依赖，~2 天）
2. **B — Attention 注入 ratio=128**：Fork forward + 4 Hook（依赖 A，~3 天）
3. **C — Attention 注入 ratio=4**：DSA Indexer 渐进三层（依赖 B，~3 天）
4. **D — Transformer 注入**：Residual 扩展（依赖 A+B+C，~2 天）
5. **E — 训练流程集成**：`wrap_forward_step()`（依赖 A-D，~2 天）

**关键设计文档**（均在仓库中）：
- `prefix_sharing_deepseek4_design.md` — 完整设计方案（差异→影响→设计→验证）
- `dsv4_analysis.md` — MindSpeed 实际代码分析
- Memory: `deepseek4-design-overview`、`deepseek4-dev-plan`、`deepseek4-module-mapping`

**MindSpeed 参考路径**（只读）：
- `mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py:360-582`
- `mindspeed_llm/core/transformer/transformer_layer.py:155-249`

## 工作区结构

```
PrefixSharing/
├── prefix-sharing/          # 正式开发目录
│   ├── prefix_sharing/      # 源代码（core/ backends/ integrations/ setup/）
│   └── tests/               # 测试（unit_test/ integrated_test/ system_test/）
├── docs/                    # 架构文档
├── dependency/              # 框架快照（verl, Megatron-LM-core, MindSpeed）
├── survey/                  # 调研 PoC（只读参考）
├── prefix_sharing_deepseek4_design.md  # DeepSeek V4 设计文档
├── dsv4_analysis.md                   # DeepSeek V4 代码分析
├── CLAUDE.md               # 本文件
└── AGENTS.md               # 仓库开发规范
```
