"""Patch 注册与调度：PatchSpec 注册 + LoggedPatchManager 安装 + import hook。

设计要点：
- 模块已加载且目标存在 → 立即 patch
- 模块已加载但目标不存在（模块正在 import 中）→ 加入 pending，稍后重试
- 模块未加载 → import hook 拦截，加载完成后 patch
- import hook 完成后立即恢复原始 __import__
"""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from typing import Callable

from prefix_sharing.setup.logged_patch import LoggedPatchManager, PatchHandle, PatchRecord


@dataclass
class PatchSpec:
    """一个待安装的 patch 规格。"""

    module_name: str          # 目标模块全限定名
    target_getter: Callable   # (module) → (target_obj, attr_name)
    patch_factory: Callable   # (original) → patched
    description: str = ""     # 人类可读描述


class PatchRegistry:
    """全局 patch 注册表，install_all() 时一次性应用所有已注册的 patch。"""

    _specs: list[PatchSpec] = []

    @classmethod
    def register(cls, spec: PatchSpec) -> None:
        cls._specs.append(spec)

    @classmethod
    def install_all(cls) -> PatchHandle:
        """应用所有已注册的 patch。

        三种情况：
        1. 模块已加载且目标可解析 → 立即 patch
        2. 模块已加载但目标不可解析（模块正在 import 中）→ 加入 pending
        3. 模块未加载 → 加入 pending，由 import hook 在加载时 patch

        所有 pending 最终统一由 import hook 处理。
        import hook 在模块加载完成后才尝试解析目标，确保类定义已完成。
        """
        shared_records: list[PatchRecord] = []
        mgr = LoggedPatchManager(shared_records)
        pending: list[PatchSpec] = []

        for spec in cls._specs:
            module = sys.modules.get(spec.module_name)
            if module is not None:
                try:
                    target_obj, attr_name = spec.target_getter(module)
                    original = getattr(target_obj, attr_name)
                    patched = spec.patch_factory(original)
                    mgr.patch_attr(target_obj, attr_name, patched)
                    print(
                        f"[PS] Immediately patched {spec.description} (module already loaded)"
                    )
                except (AttributeError, KeyError):
                    # 模块已加载但目标不存在——
                    # 可能是模块正在 import 中，类定义尚未完成。
                    # 加入 pending，等模块完全加载后再 patch。
                    pending.append(spec)
                    print(
                        f"[PS] Target not yet defined in {spec.module_name}, "
                        f"deferring patch: {spec.description}"
                    )
            else:
                pending.append(spec)

        handle = PatchHandle(shared_records, specs=list(cls._specs))

        if pending:
            _activate_import_hook(pending, shared_records)

        return handle


_original_import = None


def _activate_import_hook(
    pending_specs: list[PatchSpec],
    shared_records: list[PatchRecord],
) -> None:
    """对未加载或目标尚未定义的模块，临时拦截 __import__。

    模块加载完成后，尝试解析目标并 patch。如果目标仍然不存在
    （极端情况：模块被 import 但类在延迟定义），记录 warning 并跳过。

    所有 pending 模块处理完毕后立即恢复原始 __import__。
    """
    global _original_import

    if _original_import is not None:
        print("[PS] Import hook already active, skipping re-activation")
        return

    # 同一模块可能有多个待安装的 patch（如 transformer_impl 上的
    # forward_step 和 vocab_parallel_log_probs_from_logits），
    # 必须用 multimap 保留全部 spec，dict 会按 module_name 去重丢 spec。
    lookup: dict[str, list[PatchSpec]] = {}
    for spec in pending_specs:
        lookup.setdefault(spec.module_name, []).append(spec)
    _original_import = builtins.__import__

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _original_import
        module = _original_import(name, globals, locals, fromlist, level)

        # [fix] 原实现只匹配 __import__ 的顶层 name:
        # `from ..megatron import X` 这类相对/父包 from-import 时,
        # __import__ 收到的是父包名, 目标子模块由 importlib 内部加载,
        # hook 永远等不到完整子模块名 → 补丁永久 pending (实测:
        # verl.workers.engine.megatron.transformer_impl 的 forward_step 补丁
        # 因此从未应用, PS context 未建立, 整条 PS 链路静默旁路)。
        # 改为每次 import 后扫描全部 pending: 模块已加载且目标可解析即 patch;
        # 解析失败(模块仍在 import 中)保留在 pending, 待后续 import 重试, 不跳过。
        for mod_name in list(lookup.keys()):
            # __import__ 在 fromlist 为空时返回顶层包而非子模块，
            # 必须从 sys.modules 取实际加载的模块对象。
            actual_module = sys.modules.get(mod_name)
            if actual_module is None:
                continue
            specs = lookup[mod_name]
            for spec in list(specs):
                try:
                    target_obj, attr_name = spec.target_getter(actual_module)
                    original = getattr(target_obj, attr_name)
                    patched = spec.patch_factory(original)
                    setattr(target_obj, attr_name, patched)
                    shared_records.append(
                        PatchRecord(
                            target=target_obj,
                            attr_name=attr_name,
                            original=original,
                            replacement=patched,
                        )
                    )
                    print(
                        f"[PS] Auto-patched {spec.description} on import scan of {mod_name}"
                    )
                    specs.remove(spec)
                except (AttributeError, KeyError):
                    # 目标尚未定义(模块仍在 import 中)——保留待重试
                    pass
            if not specs:
                lookup.pop(mod_name)

        if not lookup:
            builtins.__import__ = _original_import
            _original_import = None
            print("[PS] All import hooks resolved, __import__ restored")

        return module

    builtins.__import__ = hooked_import
    print(
        f"[PS] Import hook activated for {len(lookup)} modules: {list(lookup.keys())}"
    )