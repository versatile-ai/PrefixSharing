"""DeepSeek4 G2 attention store/expand helpers.

Per-sequence functions that split packed tensors, iterate over batch
indices, and call into :class:`G2AttentionStore` for provider storage
and reuser expansion.  Framework-agnostic — no MindSpeed dependency.
"""

from __future__ import annotations

import os as _os

import torch

from prefix_sharing.backends.g2_attention_utils import (
    _adjust_cu_seqlens_for_batch,
    _compute_cmp_lengths,
    _merge_g2_fields,
    _split_by_cu_seqlens,
)
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_G2_ATTENTION,
    PrefixActivationSlotId,
    StoredG2Activation,
)



def _g2_store_with_kwargs(store, slot_id, data):
    """Store *data* (a :class:`StoredG2Activation`) into *store*.

    Fields are passed explicitly as keyword arguments to
    :meth:`G2AttentionStore.store`, avoiding the type-safety issues of
    ``**dict`` unpacking with a frozen dataclass.

    Args:
        store: :class:`G2AttentionStore`.
        slot_id: :class:`PrefixActivationSlotId`.
        data: :class:`StoredG2Activation` — all fields are forwarded.
    """
    store.store(
        slot_id,
        kv=data.kv,
        kv_compress=data.kv_compress,
        indexer_k=data.indexer_k,
        stored_len=data.stored_len,
        overwrite=True,
    )


def _g2_store_per_sequence(ctx, layout, plan, layer_id, tp_rank, field, tensor):
    """Split packed *tensor* and store provider rows into the G2 store.

    Each provider's ``valid_row[:valid_len]`` is merged with any existing
    entry for the same slot (incremental store: kv → kv_compress → indexer_k).

    Args:
        ctx: Runtime context with ``.store`` (G2AttentionStore).
        layout: :class:`PackedBatchLayout`.
        plan: :class:`PrefixSharingPlan`.
        layer_id: int.
        tp_rank: int.
        field: One of ``"kv"``, ``"kv_compress"``, ``"indexer_k"``.
        tensor: Packed tensor ``[total_padded, ...]``, or ``None``
            (no-op).
    """
    if tensor is None:
        return

    rows = _split_by_cu_seqlens(tensor, layout.padded_lengths)
    for batch_idx, row in enumerate(rows):
        if not plan.is_provider[batch_idx]:
            continue
        valid_len = layout.valid_lengths[batch_idx]
        valid_row = row[:valid_len]
        slot_id = PrefixActivationSlotId(
            plan.forward_id,
            plan.micro_batch_id,
            layer_id,
            batch_idx,
            PREFIX_STATE_TYPE_G2_ATTENTION,
            tp_rank,
        )
        existing = ctx.store.load(slot_id) if ctx.store.contains(slot_id) else None
        merged = _merge_g2_fields(existing, field, valid_row)
        _g2_store_with_kwargs(ctx.store, slot_id, merged)


def _g2_padded_store_or_replace(
    ctx, kv, kv_compress, indexer_k,
    compress_topk_idxs, packed_seq_params, compress_topk_score,
    compress_ratio, attention_module, layout, plan, layer_id, tp_rank,
):
    """Store/replace for padded batch [S, B, D] format.

    Unlike the packed path which concatenates variable-length sequences,
    padded batch keeps all sequences at the same length S. Each sequence
    is in kv[:, batch_idx, :].

    Provider: store kv[:valid_len, provider_idx, :] into G2AttentionStore.
    Reuser:   replace kv[:prefix_len, reuser_idx, :] with stored provider KV.
    Tensors are returned with the same shape — no length change.
    """
    seq_len = kv.shape[0]

    for batch_idx in range(layout.batch_size):
        valid_len = layout.valid_lengths[batch_idx]

        if plan.is_provider[batch_idx]:
            provider_kv = kv[:valid_len, batch_idx, :].clone()
            provider_cmp = None
            if kv_compress is not None and compress_ratio > 1:
                cmp_len = valid_len // compress_ratio
                provider_cmp = kv_compress[:cmp_len, batch_idx, :].clone() \
                    if kv_compress.ndim == 3 else kv_compress[:cmp_len].clone()
            provider_idxk = None
            if indexer_k is not None and compress_ratio > 1:
                idxk_len = valid_len // compress_ratio
                if indexer_k.ndim >= 3 and indexer_k.shape[1] > 1:
                    provider_idxk = indexer_k[:idxk_len, batch_idx].clone()
                else:
                    provider_idxk = indexer_k[:idxk_len].clone()

            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            _g2_store_with_kwargs(ctx.store, slot_id, StoredG2Activation(
                kv=provider_kv, kv_compress=provider_cmp,
                indexer_k=provider_idxk, stored_len=valid_len))

        elif plan.is_reuser(batch_idx):
            prefix_len = plan.prefix_lens[batch_idx]
            provider_idx = plan.provider_index[batch_idx]
            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                provider_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            if __import__("os").environ.get("PS_CACHE_OFF") == "1":
                # [PS-dualpass] 方案 5 缓存短路:跳过拼接(padded 防御路径,训练不走)
                continue
            provider = ctx.store.load(slot_id)

            kv[:prefix_len, batch_idx, :] = provider.kv[:prefix_len]

            if kv_compress is not None and compress_ratio > 1 and provider.kv_compress is not None:
                cmp_p = prefix_len // compress_ratio
                if kv_compress.ndim == 3 and kv_compress.shape[1] > 1:
                    kv_compress[:cmp_p, batch_idx, :] = provider.kv_compress[:cmp_p]
                else:
                    kv_compress[:cmp_p] = provider.kv_compress[:cmp_p]

    # ── Stats reporting ──────────────────────────────────────────
    if ctx.stats is not None:
        _store_count = sum(1 for i in range(layout.batch_size) if plan.is_provider[i])
        _reuse_count = sum(1 for i in range(layout.batch_size) if plan.is_reuser(i))
        _stored_tokens = sum(
            layout.valid_lengths[i] for i in range(layout.batch_size) if plan.is_provider[i])
        _reused_prefix_tokens = sum(
            plan.prefix_lens[i] for i in range(layout.batch_size) if plan.is_reuser(i))
        _expanded_kv_tokens = sum(layout.valid_lengths)  # padded: total valid lengths unchanged
        ctx.stats.record_attention_kv_build(
            layer_id=layer_id, store_count=_store_count,
            reuse_count=_reuse_count, reuse_hit_count=_reuse_count,
            reuse_miss_count=0,
            stored_tokens=_stored_tokens,
            reused_prefix_tokens=_reused_prefix_tokens,
            expanded_kv_tokens=_expanded_kv_tokens,
            valid_q_tokens=sum(layout.valid_lengths),
            padded_q_tokens=sum(layout.padded_lengths),
        )

    return (kv, kv_compress, indexer_k,
            compress_topk_idxs, packed_seq_params, compress_topk_score)


def _ps_cp_local_map_verl080(row_start, n_rows, q_total, cp_size, cp_rank):
    """[PS-fix17] 全局 1D packed 行区间 → CP 本地行区间(双块交错布局)。

    kvallgather_context_parallel.get_seq_chunk_ids_on_for_sharding:
    rank r 持块 {r, 2cp-1-r},本地顺序 [块r, 块2cp-1-r],块宽 q_total/(2cp)。
    区间须整块对齐(跨块 raise 暴露)。返回 (local_start, local_end, is_owner)。
    """
    if cp_size <= 1:
        return row_start, row_start + n_rows, (0 <= row_start and row_start + n_rows <= q_total)
    w = q_total // (2 * cp_size)
    g0 = row_start // w
    g1 = (row_start + n_rows - 1) // w
    if g0 != g1:
        raise RuntimeError(
            f"[PS-fix17] 行区间 [{row_start},{row_start + n_rows}) 跨块 {g0}..{g1},"
            f"不支持(块宽 {w}, 2cp={2 * cp_size})")
    g = g0
    if g < cp_size:
        owner = g
        local = row_start - g * w
    else:
        owner = 2 * cp_size - 1 - g
        local = w + (row_start - g * w)
    return local, local + n_rows, owner == cp_rank


def _g2_kv_store_or_expand(
    ctx,
    kv: torch.Tensor,
    kv_compress: torch.Tensor | None,
    indexer_k: torch.Tensor | None,
    compress_topk_idxs,
    packed_seq_params,
    compress_ratio: int,
    attention_module,
    start_pos: int,
    kv_allgather: bool,
    sequence_parallel: bool,
    *,
    query_index=None,            # ratio=4: q_r from Phase 2
    indexer_weights=None,        # ratio=4: w_r from Phase 2
    dsa_hidden=None,             # ratio=4: dsa_hidden from Phase 2
    attention_mask=None,         # ratio=4: forward_with_scores_compress mask
    compress_topk_score=None,    # ratio=4: updated by re-scoring
):
    """Provider store / Reuser expand for all key-side data.

    Called in the patched forward between Phase 3 and Phase 4.
    Splits packed tensors, iterates batch indices, and branches by
    provider/reuser identity.

    Returns expanded (kv, kv_compress, indexer_k, compress_topk_idxs,
    packed_seq_params).  For providers fields are unchanged.
    Topk recomputation and cu_seqlens adjustment deferred to Task 3.
    """
    layout = ctx.packed_batch_layout
    plan = ctx.prefix_sharing_plan
    tp_rank = ctx.parallel_info.tp_rank
    layer_id = attention_module.layer_number if attention_module is not None else 0
    # [PS-fix13-rc] recompute 分叉检测:按 (forward_id, micro_batch_id) 键控计数。
    # megatron full-recompute 下同一层同一 micro-batch 的 forward 跑两次,
    # 第二次调用 = 重算 pass。GAS>1 时纯奇偶计数会漂移,故按键区分。
    # 计数载体 = attention_module;无层载体形态(None,单测 mock/纯 ctx 调用)
    # → 无 module 状态可挂,rc=False(与上方 layer_number 的 None 容错同风格)。
    _ps_rc_key = (plan.forward_id, plan.micro_batch_id)
    _ps_is_recompute = False
    if attention_module is not None:
        _ps_rc_counts = getattr(attention_module, "_ps_rc_counts", {})
        _ps_rc_n = _ps_rc_counts.get(_ps_rc_key, 0) + 1
        _ps_rc_counts[_ps_rc_key] = _ps_rc_n
        setattr(attention_module, "_ps_rc_counts", _ps_rc_counts)
        _ps_is_recompute = _ps_rc_n > 1
    # Empty batch (all sequences trimmed away) → return empty tensors unchanged.
    if layout.batch_size == 0:
        return (kv, kv_compress, indexer_k,
                compress_topk_idxs, packed_seq_params, compress_topk_score)

    # Padded batch (BSND): kv is [S, B, D] with B > 1.
    # Each sequence already has full KV for all positions (no trim).
    # Store provider prefix KV, then replace reuser prefix KV in-place.
    # Tensor shape stays [S, B, D] — no length change.
    _is_padded = kv.ndim == 3 and kv.shape[1] > 1
    if _is_padded:
        return _g2_padded_store_or_replace(
            ctx, kv, kv_compress, indexer_k,
            compress_topk_idxs, packed_seq_params, compress_topk_score,
            compress_ratio, attention_module, layout, plan, layer_id, tp_rank)

    # Split raw KV by padded lengths (matching Q path)
    kv_rows = _split_by_cu_seqlens(kv, layout.padded_lengths)

    # Split compressed fields by cmp lengths (valid//ratio)
    has_cmp = kv_compress is not None and compress_ratio > 1
    has_idxk = indexer_k is not None and compress_ratio > 1

    cmp_rows = None
    idxk_rows = None
    # [PS-fix7] gather_from_sp_cp 会把每个 TP rank 的全表拷贝拼接成 tp_size 份重复。
    # 只需按 Q 侧真实结构切第一份拷贝;多余的行是 TP 重复,绝不能塞进最后一个序列
    # (旧的 remainder 吸收逻辑会把 reuser 的 cmp 行切到垃圾区段 → 注意力 NaN)。
    if has_cmp:
        _cmp_true_lengths = [vl // compress_ratio for vl in layout.valid_lengths]
        _cmp_true_total = sum(_cmp_true_lengths)
        cmp_rows = _split_by_cu_seqlens(kv_compress[:_cmp_true_total], _cmp_true_lengths)
    if has_idxk:
        _idxk_true_lengths = [vl // compress_ratio for vl in layout.valid_lengths]
        _idxk_true_total = sum(_idxk_true_lengths)
        idxk_rows = _split_by_cu_seqlens(indexer_k[:_idxk_true_total], _idxk_true_lengths)

    new_kv: list[torch.Tensor] = []
    new_cmp: list[torch.Tensor] = []
    new_idxk: list[torch.Tensor] = []

    for batch_idx in range(layout.batch_size):
        valid_len = layout.valid_lengths[batch_idx]

        if plan.is_provider[batch_idx]:
            # ── Provider: store ──
            valid_kv = kv_rows[batch_idx][:valid_len]
            valid_cmp = cmp_rows[batch_idx][:valid_len // compress_ratio] if cmp_rows else None
            valid_idxk = idxk_rows[batch_idx][:valid_len // compress_ratio] if idxk_rows else None

            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            _g2_store_with_kwargs(ctx.store, slot_id, StoredG2Activation(
                kv=valid_kv, kv_compress=valid_cmp, indexer_k=valid_idxk,
                stored_len=valid_len))

            new_kv.append(kv_rows[batch_idx])
            if cmp_rows:
                new_cmp.append(cmp_rows[batch_idx])
            if idxk_rows:
                new_idxk.append(idxk_rows[batch_idx])

        elif plan.is_reuser(batch_idx):
            # ── Reuser: expand ──
            prefix_len = plan.prefix_lens[batch_idx]
            if compress_ratio > 1:
                assert prefix_len % compress_ratio == 0, (
                    f"Phase 1 requires aligned prefix: "
                    f"prefix_len={prefix_len}, compress_ratio={compress_ratio}")

            provider_idx = plan.provider_index[batch_idx]
            slot_id = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                provider_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            if __import__("os").environ.get("PS_CACHE_OFF") == "1":
                # [PS-dualpass v2] 缓存短路修正(语义同裁决):不能"跳过拼接"——plan
                # 的 cu 布局期望 expanded_kv,行数不足 → fused kernel 越界 NaN(实证
                # rank1 ratio=4 nan=3031040)。正确语义:reuser 用本次计算的 provider
                # kv 行(同 forward,kv_rows[provider_idx] 截 valid_len),非 store
                # 历史值 → 布局保持 expand,唯一差异 = store 读写路径(bf16 保真)。
                _pv_kv = kv_rows[provider_idx][:layout.valid_lengths[provider_idx]]
                _pv_cmp = (cmp_rows[provider_idx][:layout.valid_lengths[provider_idx] // compress_ratio]
                           if cmp_rows else None)
                _pv_idxk = (idxk_rows[provider_idx][:layout.valid_lengths[provider_idx] // compress_ratio]
                            if idxk_rows else None)
            else:
                print(f"[VAL-DBG] rank={torch.distributed.get_rank()} layer={layer_id} enter-load PS_CACHE_OFF={__import__('os').environ.get('PS_CACHE_OFF')} GRAD_DUAL_PASS={__import__('os').environ.get('GRAD_DUAL_PASS')}", flush=True)
                _provider_ent = ctx.store.load(slot_id)
                _pv_kv = _provider_ent.kv
                _pv_cmp = _provider_ent.kv_compress
                _pv_idxk = _provider_ent.indexer_k
                # [PS-VAL v2] 值级探针:同趟比较缓存返回值 vs provider 本次计算值。
                # v1 门控 GRAD_DUAL_PASS 在 forward 进程恒 None(跨进程 env 不共享,
                # VAL-DBG 3d 实证)→ 去掉。计数改 per-(rank,layer),每对前 12 次采样。
                _ps_val_src = kv_rows[provider_idx][:layout.valid_lengths[provider_idx]]
                _ps_val_ok = _pv_kv.shape == _ps_val_src.shape and (_pv_kv.data_ptr() == _ps_val_src.data_ptr())
                _ps_val_md = ((_pv_kv.detach().float() - _ps_val_src.detach().float()).abs().max().item()
                              if _pv_kv.shape == _ps_val_src.shape else float("inf"))
                _pv_key = (layer_id, torch.distributed.get_rank())
                _pv_ns = __import__("prefix_sharing.integrations.g2_attention", fromlist=["_PS_VAL_N"]).__dict__.setdefault("_PS_VAL_N", {})
                _pv_n = _pv_ns.get(_pv_key, 0)
                if _ps_val_md > 0 or _pv_n < 12:
                    # v1.1:异常行(maxdiff>0)带 slot 标识,定位键错配/对象被改
                    _ps_val_extra = f" slot={slot_id}" if _ps_val_md > 0 else ""
                    print(f"[PS-VAL] rank={torch.distributed.get_rank()} layer={layer_id} maxdiff={_ps_val_md:.3e} same_ptr={_ps_val_ok} shape={tuple(_pv_kv.shape)} n={_pv_n}{_ps_val_extra}", flush=True)
                _pv_ns[_pv_key] = _pv_n + 1
            # Expand kv
            _swap_layout = __import__("os").path.exists("/tmp/ps_swap_layout")
            if _swap_layout:
                # [PS-fix11i 实验] [suffix + prefix] 布局:kernel 按 cu 推导的批内
                # 查询位置与真实位置对齐(ori ±127 窗口正确),前缀放批尾。
                expanded_kv = torch.cat([
                    kv_rows[batch_idx][:valid_len],
                    _pv_kv[:prefix_len]], dim=0)
            else:
                expanded_kv = torch.cat([
                    _pv_kv[:prefix_len],
                    kv_rows[batch_idx][:valid_len]], dim=0)
            new_kv.append(expanded_kv)

            # Expand kv_compress
            expanded_cmp = None
            if cmp_rows is not None:
                cmp_p = prefix_len // compress_ratio
                cmp_s = valid_len // compress_ratio
                if _swap_layout:
                    expanded_cmp = torch.cat([
                        cmp_rows[batch_idx][:cmp_s],
                        _pv_cmp[:cmp_p]], dim=0)
                else:
                    expanded_cmp = torch.cat([
                        _pv_cmp[:cmp_p],
                        cmp_rows[batch_idx][:cmp_s]], dim=0)
                new_cmp.append(expanded_cmp)

            # Expand indexer_k
            expanded_idxk = None
            if idxk_rows is not None:
                idxk_p = prefix_len // compress_ratio
                idxk_s = valid_len // compress_ratio
                if _swap_layout:
                    expanded_idxk = torch.cat([
                        idxk_rows[batch_idx][:idxk_s],
                        _pv_idxk[:idxk_p]], dim=0)
                else:
                    expanded_idxk = torch.cat([
                        _pv_idxk[:idxk_p],
                        idxk_rows[batch_idx][:idxk_s]], dim=0)
                new_idxk.append(expanded_idxk)

            # THD format (bsz=1): map batch_idx to tensor batch dim 0
            _topk_batch_idx = 0 if compress_topk_idxs is not None and compress_topk_idxs.shape[0] == 1 else batch_idx

            import os as _os_dbg4
            if _os_dbg4.path.exists("/tmp/ps_dbg_topk"):
                with open("/tmp/ps_dbg_topk.log", "a") as _f4:
                    _f4.write(f"ENTER layer={layer_id} batch={batch_idx} ratio={compress_ratio} cpi={tuple(compress_topk_idxs.shape) if compress_topk_idxs is not None else None} valid={valid_len} rc={_ps_is_recompute} qi={tuple(query_index.shape) if query_index is not None else None} vl={layout.valid_lengths}\n")
            # Recompute topk for expanded key space
            if compress_topk_idxs is not None and compress_ratio > 1 and                     __import__("os").environ.get("PS_DISABLE_TOPK_RESCORE") != "1":
                if hasattr(attention_module, 'indexer') and attention_module.indexer is not None:
                    # ratio=4: re-score with expanded indexer_k
                    if expanded_idxk is not None and query_index is not None:
                        # [PS-leak-mask 2026-08-28] 泄漏防护核查(LEAK-PROBE 基线实证):
                    # fix17 重评只处理 reuser 行,expanded 表 = provider 前缀 + 自己
                    # 后缀,表内全合法(下界 s_k=0,无非法块);exactwin 原始输出侧由
                    # dsa_indexer 的 _seg_start_c 下界屏蔽,provider 私有后缀块 0 命中。
                    # 压缩候选通道泄漏计数 = 0。若未来批次引入无关样本(一般样本)形态,
                    # 在此叠加 '< 段起点' 下界屏蔽(score -> -inf before topk)。
                    # [PS-fix17] CP 双块布局重评分重写(替代 fix11b)。
                        # fix11b 的 CP1×TP4 假设(每 rank 672 行、reuser 落最后 TP
                        # rank)在 CP2 下失效:query_index 336 行 = CP 本地 1344 的
                        # TP 分片,旧局部映射全负 → 跳过 → torch.empty 垃圾写回。
                        # 重写:TP all_gather 拼 CP 本地 1344 行 → 双块映射得
                        # reuser 本地 [960,1344) → 仅 owner CP rank 重评分写回;
                        # 窗口同 fix12b 用 ceil((prefix_len + j + ratio) // ratio)。
                        _row_start = sum(layout.valid_lengths[:batch_idx])
                        _q_total17 = sum(layout.valid_lengths)
                        import torch.distributed as _ps_dist17
                        from megatron.core import parallel_state as _ps_mpu17
                        from mindspeed_llm.tasks.models.transformer.dsa_indexer import (
                            bf16_index as _ps_bf16_index17)
                        _tp_size17 = _ps_mpu17.get_tensor_model_parallel_world_size()
                        _cp_size17 = _ps_mpu17.get_context_parallel_world_size()
                        _cp_rank17 = _ps_mpu17.get_context_parallel_rank()
                        _local_n = query_index.shape[0]
                        _l0, _l1, _is_owner17 = _ps_cp_local_map_verl080(
                            _row_start, valid_len, _q_total17, _cp_size17, _cp_rank17)
                        _topk_w = min(int(attention_module.indexer.index_topk),
                                      int(expanded_idxk.shape[0]))
                        _dev17 = query_index.device
                        _new_topk_all = torch.empty(
                            (valid_len, _topk_w), dtype=torch.int32, device=_dev17)
                        _new_score_all = torch.empty(
                            (valid_len, _topk_w), dtype=torch.float32, device=_dev17)
                        _tp_group17 = _ps_mpu17.get_tensor_model_parallel_group()
                        if _is_owner17:
                            # TP all_gather:336 行碎片 → CP 本地 1344 行(TP rank 序)
                            _qi_g17 = [torch.empty_like(query_index)
                                       for _ in range(_tp_size17)]
                            _ps_dist17.all_gather(_qi_g17, query_index, group=_tp_group17)
                            _wi_g17 = [torch.empty_like(indexer_weights)
                                       for _ in range(_tp_size17)]
                            _ps_dist17.all_gather(_wi_g17, indexer_weights, group=_tp_group17)
                            _qi_full17 = torch.cat(_qi_g17, dim=0)
                            _wi_full17 = torch.cat(_wi_g17, dim=0)
                            _q_local17 = _qi_full17[_l0:_l1].contiguous()
                            _w_local17 = _wi_full17[_l0:_l1]
                            _k_global17 = expanded_idxk.contiguous()
                            _scores17 = _ps_bf16_index17(
                                _q_local17, _w_local17.unsqueeze(-1), _k_global17)
                            # floor+1 窗口(= kernel s2IdLimit = N，与 indexer F1 同式):
                            _win17 = ((prefix_len
                                       + torch.arange(valid_len, device=_dev17)
                                       + 1)
                                      // compress_ratio).to(torch.int32)
                            # [L1-audit] LEAK-PROBE 2c 20步验证 foreign=0，下界屏蔽(s_k)暂无需新增(2026-08-29)
                            _mask17 = (torch.arange(_k_global17.shape[0], device=_dev17)
                                       .unsqueeze(0).to(torch.int32) >= _win17.unsqueeze(1))
                            _scores17 = _scores17 + torch.where(
                                _mask17, torch.finfo(_scores17.dtype).min, 0)
                            _ts_l17, _ti_l17 = _scores17.topk(_topk_w, dim=-1)
                            _ti_l17 = _ti_l17[0].int()
                            _ts_l17 = _ts_l17[0]
                            _mask18 = _ti_l17 >= _win17.unsqueeze(1)
                            _ti_l17 = torch.where(_mask18, -1, _ti_l17)
                            _new_topk_all.copy_(_ti_l17)
                            _new_score_all.copy_(_ts_l17)
                            # TP 组内广播,保证 4 个 TP rank 的 cpi 副本一致
                            _grp_r17 = torch.distributed.get_process_group_ranks(
                                _tp_group17)
                            _ps_dist17.broadcast(_new_topk_all, src=int(_grp_r17[0]),
                                                 group=_tp_group17)
                            _ps_dist17.broadcast(_new_score_all, src=int(_grp_r17[0]),
                                                 group=_tp_group17)
                        q_len_local = valid_len
                        topk_len = min(_topk_w, compress_topk_idxs.shape[-1])
                        # 广播后的行 [0:valid_len) 即 reuser 的查询;写回 packed 表的
                        # reuser 区段,并把本地索引偏移到展开 cmp 表的 reuser 区段起点。
                        _new_topk_vals = _new_topk_all[:q_len_local, :topk_len].clone()

                        # [PS-fix13-rc] 两次 pass 的 reuser topk/score 位级比对(实现)。
                        # 首次 pass(call#==1,rc=False)存快照;重放趟(call#==2,rc=True)
                        # 比对。topk 差异 = 重算 pass 与首次 pass 选择不同的压缩块
                        # (recompute 非位级确定);score 按 uint8 字节比对。
                        # 仅 owner CP rank 参与(非 owner 的 _new_topk_vals 是空张量)。
                        if _is_owner17:
                            _rc_key = (_ps_rc_key[0], _ps_rc_key[1], layer_id, batch_idx)
                            _snaps = getattr(ctx, "_ps_rc_snapshots", None)
                            if _snaps is None:
                                _snaps = {}
                                ctx._ps_rc_snapshots = _snaps
                            if _ps_is_recompute:
                                _ref_rc = _snaps.pop(_rc_key, None)
                                if _ref_rc is not None:
                                    _d_t = int((_new_topk_vals != _ref_rc[0]).sum().item())
                                    _d_s = int(
                                        (_new_score_all.view(torch.uint8)
                                         != _ref_rc[1].view(torch.uint8)).sum().item())
                                    print(
                                        f"[PS-fix13-rc] L{layer_id} b{batch_idx} "
                                        f"REPLAY topk_diff={_d_t}/{_new_topk_vals.numel()} "
                                        f"score_byte_diff={_d_s}", flush=True)
                            else:
                                _snaps[_rc_key] = (
                                    _new_topk_vals.clone(),
                                    _new_score_all.clone())
                        # [PS-fix11d 实验] 不加全局 cmp 偏移,验证 sparse MLA kernel 的
                        # topk 索引是否为 per-batch 相对解释(相对本 batch 的 cmp 区段起点)。
                        # provider 区段起点是 0,两种解释等价,所以 provider 一直正常;
                        # reuser 区段起点是 576,+576 在相对解释下反而全部越界 → NaN。
                        # 若相对解释成立:[0..575] 相对 [576..1152) 区段正好正确;
                        # 若绝对解释成立:reuser 会读到 provider 区段(数值有限但语义错),
                        # NaN 同样应消失——两种解释都应恢复有限,语义由精度验证裁决。
                        # [PS-fix11f 实验] 二分:把 reuser 的 cmp topk 全部置 -1
                        # (仅保留 ori 窗口注意力)。若 layer3 的 13 个种子 NaN 消失
                        # → cmp 路径(kernel 因果窗口屏蔽前缀块)是元凶;
                        # 若种子依旧 → ori 路径/展开几何问题。
                        _os_h = __import__("os")
                        if (_os_h.environ.get("PS_REUSER_CMP_OFF") == "1"
                                or _os_h.path.exists("/tmp/ps_reuser_cmp_off")):
                            _new_topk_vals[:] = -1
                        import os as _os_dbg3
                        if _os_dbg3.path.exists("/tmp/ps_dbg_topk"):
                            with open("/tmp/ps_dbg_topk.log", "a") as _f3:
                                _f3.write(f"WRITE layer={layer_id} batch={batch_idx} valid={valid_len} tb={_topk_batch_idx} rs={_row_start} cpi={tuple(compress_topk_idxs.shape)} nv={tuple(_new_topk_vals.shape)} tw={_topk_w} klen={topk_len} rc={_ps_is_recompute} qi={tuple(query_index.shape)} idxk={tuple(expanded_idxk.shape) if expanded_idxk is not None else None} vl={layout.valid_lengths} tp={tp_rank}\n")
                        import os as _os_lp
                        if _os_lp.path.exists("/tmp/ps_leak_probe") and _is_owner17:
                            _lp_raw = compress_topk_idxs[_topk_batch_idx, _l0:_l1, :topk_len].clone()
                            _lp_p = prefix_len // compress_ratio
                            _lp_neg = int((_lp_raw < 0).sum().item())
                            _lp_pfx = int(((_lp_raw >= 0) & (_lp_raw < _lp_p)).sum().item())
                            _lp_nw = _new_topk_vals.clone()
                            _lp_nw_neg = int((_lp_nw < 0).sum().item())
                            _lp_nw_pfx = int(((_lp_nw >= 0) & (_lp_nw < _lp_p)).sum().item())
                            _lp_nw_oob = int((_lp_nw >= expanded_idxk.shape[0]).sum().item())
                            with open("/tmp/ps_leak_probe.log", "a") as _flp:
                                _flp.write(f"LP L{layer_id} b{batch_idx} tp{tp_rank} valid={valid_len} rows={_l1 - _l0} pfx={_lp_p} raw_neg={_lp_neg} raw_pfx={_lp_pfx} new_neg={_lp_nw_neg} new_pfx={_lp_nw_pfx} new_oob={_lp_nw_oob}\n")
                        # [PS-fix17] 双块布局下仅 owner CP rank 写回本地
                        # [l0, l1);非 owner 的 cpi 无 reuser 行,跳过。
                        if _is_owner17:
                            compress_topk_idxs[_topk_batch_idx,
                                               _l0:_l1,
                                               :topk_len] = _new_topk_vals
                        if compress_topk_score is not None:
                            # [PS-fix11g] 窗口 < topk 宽度时 topk 会选入被 mask 的条目
                            # (idx=-1, score=finfo.min)。finfo.min(-3.4e38) 写回
                            # bf16 溢出成 -inf,原版模型从不给 kernel/损失喂 -inf
                            # score——这是 hook 独有输入,置 0.0 消除。
                            _new_score_vals = _new_score_all[:q_len_local, :topk_len].clone()
                            _new_score_vals[_new_topk_vals == -1] = 0.0
                            if _is_owner17:
                                compress_topk_score[_topk_batch_idx,
                                                   _l0:_l1,
                                                   :topk_len] = \
                                    _new_score_vals.to(compress_topk_score.dtype)
                else:
                    # ratio=128: recompute by position with expanded seqlen
                    tp_size = 1
                    cp_size = 1
                    try:
                        from megatron.core import parallel_state
                        tp_size = parallel_state.get_tensor_model_parallel_world_size()
                        cp_size = parallel_state.get_context_parallel_world_size()
                    except (ImportError, RuntimeError, AssertionError):
                        pass
                    q_len_local = valid_len
                    q_len = q_len_local * tp_size if sequence_parallel else q_len_local
                    q_len_global = q_len * cp_size if cp_size > 1 else q_len
                    expanded_seqlen = prefix_len + q_len_global
                    bsz = compress_topk_idxs.shape[0]
                    # [PS-fix19] 弃用 get_compress_topk_idxs:内部 floor bug
                    # (mask = arange(1,seqlen+1)//ratio → 每窗口第 1 位置整行全遮、
                    # 其余行缺自己窗口);且 [-q_len_local:] 取 expanded 表末尾 +
                    # offset=0(cmp_rows=None)→ 写回 idx 达 35 > cpi 槽 21
                    # (实测 MLA-CP-IDX oob=True)→ kernel OOB 读 → NaN。
                    # 改为与 fix11b 同款坐标:每行窗口 = ceil((prefix_len + j)/ratio)
                    # (per-batch 相对,与 fix16 写回同坐标系),槽 = [0, win) 递增,
                    # 超出置 -1。
                    # [PS-fix21] fix19A 重写删掉了 topk_len 定义(旧定义在
                    # get_compress_topk_idxs 调用的 min() 内)。此处按 cpi 槽数定:
                    # ratio=128 → cpi 槽 21;窗口上限 19 ≤ 21,_new_idxs 已掩码截断,
                    # 超出槽数的槽位写 -1,与 fix16 坐标系一致。
                    topk_len = compress_topk_idxs.shape[-1]
                    _dev16b = compress_topk_idxs.device
                    _win16b = ((prefix_len
                                + torch.arange(q_len_local, device=_dev16b)
                                + 1)
                               // compress_ratio).to(torch.int32)
                    _new_idxs = torch.arange(
                        topk_len, device=_dev16b).to(torch.int32) \
                        .unsqueeze(0).expand(q_len_local, topk_len).clone()
                    _new_idxs[_new_idxs >= _win16b.unsqueeze(1)] = -1
                    # [PS-fix11] 与 indexer 分支相同的行定位修复:reuser 的行位于
                    # packed 表的 [_row_start, _row_start+q_len_local),
                    # 写 [0:q_len_local) 会覆盖 provider 的 topk。
                    _row_start_p = sum(layout.valid_lengths[:batch_idx])
                    # [PS-fix17] ratio=128 分支:new_idxs 为 CP 本地表(负索引在
                    # owner rank 恰取 reuser 行),写回同走双块映射,仅 owner 写回。
                    try:
                        from megatron.core import parallel_state
                        _cp_rank16b = parallel_state.get_context_parallel_rank()
                        _cp_size16b = parallel_state.get_context_parallel_world_size()
                    except (ImportError, RuntimeError, AssertionError):
                        _cp_rank16b, _cp_size16b = 0, 1
                    _l0b, _l1b, _is_owner16b = _ps_cp_local_map_verl080(
                        _row_start_p, q_len_local, sum(layout.valid_lengths),
                        _cp_size16b, _cp_rank16b)
                    if _is_owner16b:
                        compress_topk_idxs[_topk_batch_idx,
                                           _l0b:_l1b,
                                           :topk_len] = _new_idxs

            # Store back for transitive reuse
            own_slot = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, layer_id,
                batch_idx, PREFIX_STATE_TYPE_G2_ATTENTION, tp_rank)
            _g2_store_with_kwargs(ctx.store, own_slot, StoredG2Activation(
                kv=expanded_kv, kv_compress=expanded_cmp,
                indexer_k=expanded_idxk, stored_len=prefix_len + valid_len))

        else:
            # Non-provider, non-reuser — pass through unchanged
            new_kv.append(kv_rows[batch_idx][:valid_len])
            if cmp_rows:
                new_cmp.append(cmp_rows[batch_idx][:valid_len // compress_ratio])
            if idxk_rows:
                new_idxk.append(idxk_rows[batch_idx][:valid_len // compress_ratio])

    # ── Stats reporting ──────────────────────────────────────────
    if ctx.stats is not None:
        _store_count = sum(1 for i in range(layout.batch_size) if plan.is_provider[i])
        _reuse_count = sum(1 for i in range(layout.batch_size) if plan.is_reuser(i))
        _stored_tokens = sum(
            layout.valid_lengths[i] for i in range(layout.batch_size) if plan.is_provider[i])
        _reused_prefix_tokens = sum(
            plan.prefix_lens[i] for i in range(layout.batch_size) if plan.is_reuser(i))
        _expanded_kv_tokens = sum(
            (plan.prefix_lens[i] + layout.valid_lengths[i]) if plan.is_reuser(i)
            else layout.valid_lengths[i] for i in range(layout.batch_size))
        ctx.stats.record_attention_kv_build(
            layer_id=layer_id, store_count=_store_count,
            reuse_count=_reuse_count, reuse_hit_count=_reuse_count,
            reuse_miss_count=0,
            stored_tokens=_stored_tokens,
            reused_prefix_tokens=_reused_prefix_tokens,
            expanded_kv_tokens=_expanded_kv_tokens,
            valid_q_tokens=sum(layout.valid_lengths),
            padded_q_tokens=sum(layout.padded_lengths),
        )

    # Adjust cu_seqlens for reusers (offsets all subsequent entries)
    if packed_seq_params is not None:
        packed_seq_params = _adjust_cu_seqlens_for_batch(
            packed_seq_params, plan, compress_ratio)
    result_kv = torch.cat(new_kv, dim=0)
    result_cmp = torch.cat(new_cmp, dim=0) if new_cmp else (kv_compress if has_cmp else None)
    result_idxk = torch.cat(new_idxk, dim=0) if new_idxk else (indexer_k if has_idxk else None)

    # [PS-fix10 收编] 越界 topk 索引钳制为表尾(兜底防御):rescore(fix12)后值域
    # 已段内自洽,正常不触发;触发时打印告警(不再静默掩盖),语义级重映射是后续工作。
    if compress_topk_idxs is not None and result_cmp is not None:
        _cmp_n = result_cmp.shape[0]
        _n_clamped = int((compress_topk_idxs > _cmp_n - 1).sum())
        if _n_clamped > 0:
            print(f"[PS] WARNING: clamped {_n_clamped} out-of-range topk indices "
                  f"(max={int(compress_topk_idxs.max())}, cmp_n={_cmp_n})", flush=True)
        compress_topk_idxs = compress_topk_idxs.clamp(max=_cmp_n - 1)

    # [PS-fix22-sentinel] 毒判别实验(peer 机制假设:kernel 消费槽数 deal >
    # host 有效槽数 win → 消费进 −1 区 → vector 跳 −1 早退 → merge 欠填 →
    # cube 按 deal 读陈旧尾 → NaN)。把 −1 槽替换为合法 idx(win−1 重复填充):
    # 毒消失 → 实锤 −1 跳读欠填;毒不变 → 机制在 dense 侧。
    _diag22 = str(__import__("os").environ.get("PS_SENTINEL_FILL"))+ "/file:" + ("Y" if __import__("os").path.exists("/tmp/ps_sentinel_fill") else "N")
    print(f"[PS-sentinel-diag] env={_diag22!r} cpi={None if compress_topk_idxs is None else tuple(compress_topk_idxs.shape)} neg={int((compress_topk_idxs == -1).sum()) if compress_topk_idxs is not None else -1} cmin={int(compress_topk_idxs.min()) if compress_topk_idxs is not None else -1} cmax={int(compress_topk_idxs.max()) if compress_topk_idxs is not None else -1} padded={_is_padded} rc={_ps_is_recompute} kv={tuple(kv.shape)} layer={layer_id}", flush=True)
    # PS_SENTINEL_FILL=1:全填;=K(K>1):只填前 K 个 −1 槽(二分找最小毒消失 K
    # = kernel 真实 deal,得精确 cmpMaskRight)。
    import os as _os_s22
    # [PS-fix22-file] ray 常驻集群(8-20 起)worker env 不继承训练脚本 export
    # (诊断实证 env=None);改文件开关,与 ps_swap_layout 同模式。内容=fill 值。
    _ps_sentinel = None
    if _os_s22.path.exists("/tmp/ps_sentinel_fill"):
        try:
            with open("/tmp/ps_sentinel_fill") as _f22:
                _ps_sentinel = _f22.read().strip() or "1"
        except Exception:
            _ps_sentinel = "1"
    if _ps_sentinel is not None and compress_topk_idxs is not None:
        _sf22 = int(_ps_sentinel)
        _neg22 = compress_topk_idxs == -1
        if bool(_neg22.any()):
            # 行内第一个 −1 的列位置 = win(探针实证:valid=[0,win)、−1=[win,512),
            # neg_n=Σ(512−win) 逐项吻合)。填充值 = win−1(行内最后合法槽)。
            _wins22 = _neg22.int().argmax(dim=-1, keepdim=True)
            if _sf22 == 0:
                # fill=0: −1 槽全填 0(0 < 一切 s2IdLimit → 全部真实 gather)
                _fill22 = torch.zeros_like(_wins22).to(compress_topk_idxs.dtype)
            else:
                _fill22 = torch.clamp(_wins22 - 1, min=0).to(compress_topk_idxs.dtype)
            if _sf22 > 1:
                _cols22 = torch.arange(
                    compress_topk_idxs.shape[-1],
                    device=compress_topk_idxs.device,
                    dtype=torch.int64).unsqueeze(0).unsqueeze(0)
                _mask22 = _neg22 & ((_cols22 - _wins22) < _sf22)
            else:
                _mask22 = _neg22
            compress_topk_idxs = torch.where(
                _mask22, _fill22.expand_as(_neg22), compress_topk_idxs)
            print(f"[PS-sentinel] fill={_sf22} replaced={int(_mask22.sum())} "
                  f"neg_total={int(_neg22.sum())} "
                  f"win=[{int(_wins22.min())},{int(_wins22.max())}]", flush=True)

    return (result_kv, result_cmp, result_idxk,
            compress_topk_idxs, packed_seq_params, compress_topk_score)
