#!/usr/bin/env bash
# verify-host-patches.sh — host-patches 三件装 L0/L1 一键验证
#
# 用法:
#   verify-host-patches.sh [--apply] [--golden <dir>] [MS_ROOT] [VERL_ROOT]
#
#   MS_ROOT   : mindspeed_llm 仓 checkout 根(含 mindspeed_llm/),默认 ./MindSpeed-LLM
#   VERL_ROOT : verl 仓 checkout 根(含 verl/),默认 ./verl
#   --apply   : check 通过后实际应用。ps-extra.patch 为参考件(0 文件),不参与 apply。
#               check 失败但树已带冒烟锚点 → 视为已应用,跳过 apply 继续验证。
#   --golden  : 期望树(联合布局 <dir>/mindspeed_llm + <dir>/verl,如 A4 的 tmp/golden)。
#               传 "--golden skip" 跳过 L1(仅 L0)。
#
# 出口码:0 = 通过(允许 WARN);1 = 有 FAIL。对应用收案集 acceptance-cases.md v2 的 L0/L1 层。
#
# 注:MS 锚点只在 MS 树(及 golden)检查,verl 锚点只在 verl 树(及 golden)检查——
#     两个仓的树根互不包含对方的包目录。
set -u

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
MS_BASE=99f7fc1d
VERL_BASE=809f2d8f
PASS=0; FAIL=0; WARN=0

MS_ROOT="./MindSpeed-LLM"
VERL_ROOT="./verl"
APPLY=0
GOLDEN="TREE"

POS_ARGS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --golden) GOLDEN="$2"; shift 2 ;;
    --golden=*) GOLDEN="${1#--golden=}"; shift ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) POS_ARGS=$((POS_ARGS+1)); if [ "$POS_ARGS" = "1" ]; then MS_ROOT="$1"; else VERL_ROOT="$1"; fi; shift ;;
  esac
done

say()  { echo "== $*"; }
ok()   { echo "   [OK]   $*"; PASS=$((PASS+1)); }
bad()  { echo "   [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "   [WARN] $*"; WARN=$((WARN+1)); }

# ---------- L1 锚点表(仓内路径 | 模式 | 说明)----------
# 严格锚点 = 真实修复特征(缺失必查:厂商已修→upgrade-guide §4 退役判定 / 重制漏项→必须修)。
# 探针锚点 = PS_DEBUG 门控探针(剥除允许,缺失=WARN;A5 验收矩阵依赖则须保留)。
STRICT_MS=(
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|(_ci_fix19 + 1) // compress_ratio|M7a fix19 floor+1(四站窗口 N=(P+1)//ratio)"
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|_seg_len_c|M7d fencepost clamp"
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|topk_idxs - _seg_start_c|M7f DOMAIN-CONV 段相对"
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|cu_seqlens_cmp_kv_i|M9/M11 per-chunk cu"
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|cu_seqlens_q_i|M9/M11 per-chunk cu(q)"
  "mindspeed_llm/ops/npu_sparse_flash_mla.py|get_cmp_cu_seqlens(cu_seqlens_kv|fix4(cmp cu 从 KV 侧推导)"
  "mindspeed_llm/ops/npu_sparse_flash_mla.py|pad_mask|BSND-pad-fix(forward/backward pad 行 zero-fill)"
  "mindspeed_llm/tasks/models/transformer/deepseek4/compressor.py|freqs_cis.shape[0] != kv.shape[0]|M2(SP freqs 守卫)"
  "mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention_kernel.py|import triton.language as tl|M5(缺 import)"
  "mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py|k_is_global=packed_seq_params is not None|M3(all_gather k_is_global 传参)"
  "mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py|packed_seq_params is None and (self.config.sequence_parallel or self.kv_allgather)|M4(kv_compress gather 分支条件)"
  "mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py|pad_mask = attention_mask|BSND-pad-fix(pad 行标记提取)"
)
STRICT_VERL=(
  "verl/utils/torch_functional.py|offsets.clone()|fix5a(offsets 断共享)"
  "verl/workers/engine/megatron/transformer_impl.py|input_ids.values().clone()|fix5b(label 断共享)"
  "verl/models/mcore/util.py|align_value = 2048|align-2048(CP seqlen 上取整)"
  "verl/trainer/config/engine/mindspeed.yaml|llm_kwargs: {}|V6"
  "verl/workers/engine/mindspeed/utils.py|all_config = {**engine_config}|V5"
  "verl/workers/engine/mindspeed/transformer_impl.py|memory_efficient=True|V4"
  "verl/trainer/ppo/ray_trainer.py|asyncio.run|E4"
  "verl/workers/rollout/vllm_rollout/utils.py|VERL_VLLM_STRIP_HCCL_PORT_RANGE|E10"
  "verl/utils/device.py|torch.npu.is_available|E5"
  "verl/checkpoint_engine/hccl_checkpoint_engine.py|\"hccl\"|E1"
  "verl/trainer/constants_ppo.py|GLOO_SOCKET_IFNAME|E3(值按 upgrade-guide §3 核对)"
  "verl/utils/vllm/patch.py|deepseek_v4|V2"
  "verl/utils/vllm/npu_vllm_patch.py|version.parse|E8"
)
PROBES_MS=(
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|IDXER-BRANCH|IDXER-BRANCH 探针"
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|FUSED-OUT|FUSED-OUT 探针"
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|MLA-CP-IDX|MLA-CP-IDX 探针"
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|Z1-PROBE|Z1-PROBE(退役验证)"
)
PROBES_VERL=(
  "verl/models/mcore/model_forward.py|PS_DEBUG|DSA-DBG 探针门控"
  "verl/workers/actor/megatron_actor.py|PS_DEBUG|FWD-TEMP/LOGITS/LOGPROBS 探针门控"
)
# 幂等判定冒烟锚点(树已带 = 补丁已应用)
SMOKE=(
  "mindspeed_llm/tasks/models/transformer/dsa_indexer.py|(_ci_fix19 + 1) // compress_ratio"
  "verl/workers/engine/mindspeed/utils.py|all_config = {**engine_config}"
)

# 实验件标记全集(生产树应为 0;含跨文件全剥除目标)
EXP_MARKERS='GRAD_DUAL_PASS|DUAL_PASS|/tmp/gr_dual_pass|GRAD-DUAL|_ps_skip_sync_wrapper|PS_SKIP_WEIGHT_SYNC|WDUMP|update_weights_from_ipc SKIPPED|PS_ROLLOUT_MODEL_PATH|VLLM_LAUNCH_ERROR'

anchor_check() {  # $1=树根 $2=表名 $3=mode(ok|warn|smoke) $4=label $5=smoke 指针输出
  local tree="$1" table="$2" mode="$3" label="$4"
  eval 'local arr=("${'"$table"'[@]}")'
  local n=${#arr[@]} miss=0 line f pat
  for line in "${arr[@]}"; do
    f="${line%%|*}"; pat="$(echo "${line#*|}" | cut -d'|' -f1)"
    if [ -f "$tree/$f" ] && grep -qF "$pat" "$tree/$f" 2>/dev/null; then
      continue
    fi
    miss=$((miss+1))
    case "$mode" in
      ok)    echo "   [MISS] $label :: $f :: $pat (是否厂商已修? upgrade-guide §4)";;
      warn)  echo "   [WARN] $label :: $f :: $pat";;
      smoke) : ;;
    esac
  done
  case "$mode" in
    smoke) return $([ "$miss" = "0" ] && echo 0 || echo 1) ;;
    warn)
      if [ "$miss" = "0" ]; then ok "$label: $n 锚点全在"; else warn "$label: $miss/$n 探针缺失(剥除允许;A5 矩阵依赖则须保留)"; fi ;;
    *)
      if [ "$miss" = "0" ]; then ok "$label: $n 锚点全在"; else bad "$label: $miss/$n 缺失(upgrade-guide §4 判退役 vs 漏重制)"; fi ;;
  esac
}

smoke_hit() {  # $1=树根;任一冒烟锚点命中=0
  local tree="$1" line f pat
  for line in "${SMOKE[@]}"; do
    f="${line%%|*}"; pat="${line#*|}"
    if [ -f "$tree/$f" ] && grep -qF "$pat" "$tree/$f" 2>/dev/null; then return 0; fi
  done
  return 1
}

scan_exp_markers() {  # $1=树根 $2=补丁文件清单标题
  local root="$1"
  local hit=0 f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if [ -f "$root/$f" ] && grep -qE "$EXP_MARKERS" "$root/$f" 2>/dev/null; then
      echo "   [FAIL] $root/$f 残留实验件标记"
      hit=1
    fi
  done < "$PATCH_FILE_LIST"
  [ "$hit" = "0" ]
}

# ---------- L0 ----------
say "L0: 树与基座"
for r in "$MS_ROOT" "$VERL_ROOT"; do
  [ -d "$r" ] && ok "树存在: $r" || bad "树不存在: $r"
done
ms_head=$(git -C "$MS_ROOT" rev-parse --short HEAD 2>/dev/null || echo "?")
verl_head=$(git -C "$VERL_ROOT" rev-parse --short HEAD 2>/dev/null || echo "?")
if [ "$ms_head" = "?" ]; then warn "MS 树非 git checkout;L0 apply/基座校验跳过该步"
elif [ "$ms_head" = "$MS_BASE" ]; then ok "MS 基座 $MS_BASE"
else warn "MS HEAD=$ms_head(期望 $MS_BASE)=升级后场景,先走 upgrade-guide 再验"; fi
if [ "$verl_head" = "?" ]; then warn "verl 树非 git checkout;L0 apply/基座校验跳过该步"
elif [ "$verl_head" = "$VERL_BASE" ]; then ok "verl 基座 $VERL_BASE"
else warn "verl HEAD=$verl_head(期望 $VERL_BASE)=升级后场景,先走 upgrade-guide 再验"; fi

say "L0: 补丁形态(ps-core 15 / verl-env 10 / ps-extra 0=参考件)"
pc_n=$(grep -c '^diff --git' "$BUNDLE_DIR/ps-core.patch")
ve_n=$(grep -c '^diff --git' "$BUNDLE_DIR/verl-env.patch")
pe_n=$(grep -c '^diff --git' "$BUNDLE_DIR/ps-extra.patch")
[ "$pc_n" = "15" ] && ok "ps-core 15 文件($pc_n)" || bad "ps-core 文件数 $pc_n ≠ 15(spec §0)"
[ "$ve_n" = "10" ] && ok "verl-env 10 文件($ve_n)" || bad "verl-env 文件数 $ve_n ≠ 10"
[ "$pe_n" = "0" ] && ok "ps-extra 0 文件(参考件不参与 apply)" || bad "ps-extra 文件数 $pe_n ≠ 0"

# 补丁集合文件清单(实验件扫描/compile 的域)
PATCH_FILE_LIST=$(mktemp)
(grep -h '^diff --git' "$BUNDLE_DIR/ps-core.patch" "$BUNDLE_DIR/verl-env.patch" | sed -n 's/.* b\/\(.*\)$/\1/p') > "$PATCH_FILE_LIST"

say "L0: apply --check / apply(先 check 后 apply;ps-core 按仓库域切片)"
SLICE_MS=/tmp/vhp-ps-ms.patch; SLICE_VERL=/tmp/vhp-ps-verl.patch
slice_patch() {  # $1=补丁全路径;产出 $SLICE_MS/$SLICE_VERL(可能为空文件)
  local patch="$1" rc
  python3 - "$patch" "$SLICE_MS" "$SLICE_VERL" <<'PYEOF'
import re, sys
src = open(sys.argv[1]).read()
# 以行首 "diff --git a/..." 为文件块边界;容错:注释里的 diff --git 文本不算块头
blocks = re.split(r"^diff --git ", src, flags=re.M)
ms, v = "", ""
for b in blocks:
    if not b.strip():
        continue
    head = b.split(" ", 1)[0]
    body = "diff --git " + b
    if head.startswith("a/mindspeed_llm/"):
        ms += body
    elif head.startswith("a/verl/"):
        v += body
    elif "--- " not in b:
        continue  # 文件头注释块(preamble),非文件部分
    else:
        print("未识别域: " + head, file=sys.stderr)
        sys.exit(1)
open(sys.argv[2], "w").write(ms)
open(sys.argv[3], "w").write(v)
PYEOF
  return $?
}
apply_one() {  # $1=树根 $2=补丁 $3=标签
  local root="$1" patch="$2" label="$3"
  if [ -n "$patch" ] && [ ! -s "$patch" ]; then ok "$label: 切片为空(域内无文件,跳过)"; return 0; fi
  if ! git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then warn "$label: 非 git checkout,跳过 apply 校验"; return 0; fi
  if git -C "$root" apply --check "$patch" 2>/tmp/vhp_err; then
    ok "$label: apply --check 通过"
    if [ "$APPLY" = "1" ]; then
      if git -C "$root" apply "$patch"; then ok "$label: applied"; else bad "$label: apply 失败"; fi
    fi
  else
    if smoke_hit "$root" 0 || smoke_hit "$root" 1; then
      warn "$label: check 失败但树已带冒烟锚点 → 视为已应用,跳过($(head -1 /tmp/vhp_err))"
    else
      bad "$label: apply --check 失败($(head -1 /tmp/vhp_err));非幂等 → 核查基座/按 upgrade-guide §0"
    fi
  fi
}
# ps-core 复合补丁:同文件含 mindspeed_llm/*(6)与 verl/*(9),单侧 checkout 承受不了
# (缺域目录即报错;verl/mindspeed_llm 软链只服务 PYTHONPATH)。按域切片后分别 apply。
if slice_patch "$BUNDLE_DIR/ps-core.patch"; then
  apply_one "$MS_ROOT" "$SLICE_MS" "MS ps-core"
  apply_one "$VERL_ROOT" "$SLICE_VERL" "verl ps-core"
else
  bad "ps-core 切片失败"
fi
rm -f "$SLICE_MS" "$SLICE_VERL"
# verl-env 纯 verl 域(E1-E10),直接 apply
apply_one "$VERL_ROOT" "$BUNDLE_DIR/verl-env.patch" "verl verl-env"

say "L0: 实验件标记(补丁集合文件应为 0 命中)"
exp_all=0
scan_exp_markers "$MS_ROOT" || exp_all=1
scan_exp_markers "$VERL_ROOT" || exp_all=1
[ "$exp_all" = "0" ] && ok "实验件标记 0 命中" || bad "实验件标记残留(spec §4 剥除清单 / 重跑 rebuild_patches.py)"

say "L0: 生产 .py compile(补丁集合内 .py,语法级)"
compile_err=0
while IFS= read -r f; do
  case "$f" in *.py) : ;; *) continue ;; esac
  [ -n "$f" ] || continue
  for root in "$MS_ROOT" "$VERL_ROOT"; do
    [ -f "$root/$f" ] || continue
    if ! python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$root/$f" 2>/dev/null; then
      echo "   [FAIL] 语法错误: $root/$f"; compile_err=1
    fi
  done
done < "$PATCH_FILE_LIST"
rm -f "$PATCH_FILE_LIST"
[ "$compile_err" = "0" ] && ok "compile 全过" || bad "compile 有错(见上)"

say "L0: 运行时依赖存在性(补丁集新增的库内模块 import 目标文件在树中)"
dep_err=0
python3 - "$MS_ROOT" "$VERL_ROOT" "$BUNDLE_DIR" <<'PYEOF' || dep_err=1
import os, re, sys
ms_root, verl_root, bundle = sys.argv[1], sys.argv[2], sys.argv[3]
patches = [os.path.join(bundle, n) for n in ("ps-core.patch", "verl-env.patch")]
added = {}
cur = None
for patch in patches:
    for line in open(patch):
        if line.startswith('diff --git'):
            cur = './' + line.split()[2].lstrip('a/')
        elif line.startswith('--- '):
            p = line.split()[1]
            cur = None if p == '/dev/null' else './' + p.lstrip('a/')
        elif line.startswith('+++'):
            continue
        elif cur and line.startswith('+') and not line.startswith('+++'):
            s = line[1:].strip()
            if re.match(r'^(from|import)\s', s):
                added.setdefault(cur, set()).add(s)
def check(tree, rel):
    cands = [os.path.join(tree, rel + '.py'), os.path.join(tree, rel, '__init__.py')]
    return any(os.path.isfile(c) for c in cands)
bad = []
for f, stmts in sorted(added.items()):
    for s in sorted(stmts):
        m = re.match(r'^from\s+([\w.]+)\s+import', s)
        if not m:
            continue
        mod = m.group(1)
        in_ms = mod.startswith('mindspeed_llm')
        in_verl = mod.startswith('verl')
        if not (in_ms or in_verl):
            continue
        tree = ms_root if in_ms else verl_root
        parts = mod.split('.')
        if len(parts) < 2:
            continue  # from 包 import 名(属性/子模块),无法静态判定
        rel = '/'.join(parts)
        if not check(tree, rel):
            bad.append((f, s))
if bad:
    print("   [FAIL] 补丁集新增 import 目标模块在树中不存在:")
    for f, s in bad:
        print(f"       {s}  <-  {f}")
    sys.exit(1)
print("   [OK]   新增库内模块 import 全部可解析")
PYEOF
[ "$dep_err" = "0" ] && ok "运行时依赖存在性通过" || bad "运行时依赖缺口(见上)"

# ---------- L1 ----------
say "L1: 严格锚点(真实修复特征;缺失按 §4 判退役 vs 漏重制)"
case "$GOLDEN" in
  skip) : ;;
  TREE)
    anchor_check "$MS_ROOT" STRICT_MS ok "L1 MS($MS_ROOT)"
    anchor_check "$VERL_ROOT" STRICT_VERL ok "L1 verl($VERL_ROOT)"
    ;;
  *)
    anchor_check "$GOLDEN" STRICT_MS ok "L1 期望树-ms($GOLDEN)"
    anchor_check "$GOLDEN" STRICT_VERL ok "L1 期望树-verl($GOLDEN)"
    ;;
esac

say "L1: 探针锚点(PS_DEBUG 门控;缺失=WARN)"
case "$GOLDEN" in
  skip) : ;;
  TREE)
    anchor_check "$MS_ROOT" PROBES_MS warn "L1 MS 探针($MS_ROOT)"
    anchor_check "$VERL_ROOT" PROBES_VERL warn "L1 verl 探针($VERL_ROOT)"
    ;;
  *)
    anchor_check "$GOLDEN" PROBES_MS warn "L1 期望树-ms 探针($GOLDEN)"
    anchor_check "$GOLDEN" PROBES_VERL warn "L1 期望树-verl 探针($GOLDEN)"
    ;;
esac

echo
echo "=========================================="
echo "PASS=$PASS FAIL=$FAIL WARN=$WARN"
if [ "$FAIL" = "0" ]; then
  echo "L0/L1: 通过(升级后仍需 L2/L3 手动,见 acceptance-cases.md v2)"
else
  echo "L0/L1: 失败"
fi
echo "=========================================="
[ "$FAIL" = "0" ]
