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
    _ps_rc_key = (plan.forward_id, plan.micro_batch_id)
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
            provider = ctx.store.load(slot_id)
            # Expand kv
            _swap_layout = __import__("os").path.exists("/tmp/ps_swap_layout")
            if _swap_layout:
                # [PS-fix11i 实验] [suffix + prefix] 布局:kernel 按 cu 推导的批内
                # 查询位置与真实位置对齐(ori ±127 窗口正确),前缀放批尾。
                expanded_kv = torch.cat([
                    kv_rows[batch_idx][:valid_len],
                    provider.kv[:prefix_len]], dim=0)
            else:
                expanded_kv = torch.cat([
                    provider.kv[:prefix_len],
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
                        provider.kv_compress[:cmp_p]], dim=0)
                else:
                    expanded_cmp = torch.cat([
                        provider.kv_compress[:cmp_p],
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
                        provider.indexer_k[:idxk_p]], dim=0)
                else:
                    expanded_idxk = torch.cat([
                        provider.indexer_k[:idxk_p],
                        idxk_rows[batch_idx][:idxk_s]], dim=0)
                new_idxk.append(expanded_idxk)

            # THD format (bsz=1): map batch_idx to tensor batch dim 0
            _topk_batch_idx = 0 if compress_topk_idxs is not None and compress_topk_idxs.shape[0] == 1 else batch_idx

            # Recompute topk for expanded key space
            if compress_topk_idxs is not None and compress_ratio > 1 and                     __import__("os").environ.get("PS_DISABLE_TOPK_RESCORE") != "1":
                if hasattr(attention_module, 'indexer') and attention_module.indexer is not None:
                    # ratio=4: re-score with expanded indexer_k
                    if expanded_idxk is not None and query_index is not None:
                        # [PS-fix11b] SP 局部空间自洽的直接重评分。
                        # 关键事实(实测 + 代码确认):
                        #  - query_index/indexer_weights/dsa_hidden 是 SP 局部张量,
                        #    每 rank 672 行(全局 2688 的 1/4 分片);
                        #  - expanded_idxk 是全局压缩表(576 行 = 480 前缀 + 96 后缀),
                        #    fix7 已按 Q 侧真实长度截断,无 TP 重复;
                        #  - 全局 reuser 行 [_row_start, _row_start+valid_len) 只落在
                        #    最后一个 TP rank 的局部区间。
                        # 因此不再复用 forward_with_scores_compress(它的 cu/窗口
                        # 推导与局部切片语义不匹配),直接按 non-fused indexer 的
                        # 等价逻辑评分:bf16 内积 + 显式因果窗口 topk,然后 TP 组
                        # 广播,保证所有 rank 的 compress_topk_idxs 一致。
                        _row_start = sum(layout.valid_lengths[:batch_idx])
                        _cmp_offset_local = sum(
                            vl // compress_ratio for vl in layout.valid_lengths[:batch_idx])
                        import torch.distributed as _ps_dist11
                        from megatron.core import parallel_state as _ps_mpu11
                        from mindspeed_llm.tasks.models.transformer.dsa_indexer import (
                            bf16_index as _ps_bf16_index)
                        _tp_size11 = _ps_mpu11.get_tensor_model_parallel_world_size()
                        _tp_rank11 = _ps_mpu11.get_tensor_model_parallel_rank()
                        _local_n = query_index.shape[0]
                        _shard_start = _local_n * _tp_rank11
                        _l0 = max(0, _row_start - _shard_start)
                        _l1 = min(_local_n, _row_start + valid_len - _shard_start)
                        n_local_reuser = _l1 - _l0
                        _topk_w = min(int(attention_module.indexer.index_topk),
                                      int(expanded_idxk.shape[0]))
                        _dev11 = query_index.device
                        _new_topk_all = torch.empty(
                            (valid_len, _topk_w), dtype=torch.int32, device=_dev11)
                        _new_score_all = torch.empty(
                            (valid_len, _topk_w), dtype=torch.float32, device=_dev11)
                        if n_local_reuser > 0:
                            _q_local = query_index[_l0:_l1].contiguous()
                            _w_local = indexer_weights[_l0:_l1]
                            _k_global = expanded_idxk.contiguous()
                            _scores = _ps_bf16_index(
                                _q_local, _w_local.unsqueeze(-1), _k_global)
                            # [PS-fix12] 窗口上限对齐 kernel 的 ceil 约定:
                            # kernel s2IdLimit = (cmpMaskRight + s1EndIdx + 1)/cmpRatio
                            # = (1920+j+1)/4; hook 原 floor(480+j//4), j%4==3 时差 1 行
                            # -> kernel 多读 1 列未初始化 -> NaN
                            _win = ((prefix_len + torch.arange(n_local_reuser, device=_dev11) + 1)
                                    // compress_ratio).to(torch.int32)
                            _mask11 = (torch.arange(_k_global.shape[0], device=_dev11)
                                       .unsqueeze(0).to(torch.int32) >= _win.unsqueeze(1))
                            _scores = _scores + torch.where(
                                _mask11, torch.finfo(_scores.dtype).min, 0)
                            _topk_score_l, _topk_idxs_l = _scores.topk(_topk_w, dim=-1)
                            # [PS-fix11b3] bf16_index 返回 (b, s_q, s_k) 三维(b=1),
                            # topk 后为 (1, n_local, K),压掉 batch 维再写入二维缓冲。
                            _topk_idxs_l = _topk_idxs_l[0].int()
                            _topk_score_l = _topk_score_l[0]
                            _mask12 = _topk_idxs_l >= _win.unsqueeze(1)
                            _topk_idxs_l = torch.where(_mask12, -1, _topk_idxs_l)
                            _new_topk_all.copy_(_topk_idxs_l)
                            _new_score_all.copy_(_topk_score_l)
                        _src_tp_rank11 = min(_row_start // _local_n, _tp_size11 - 1)
                        _tp_group11 = _ps_mpu11.get_tensor_model_parallel_group()
                        # megatron-core 无 get_tensor_model_parallel_global_ranks,
                        # 用 torch 标准 API 从进程组取全局 rank 列表。
                        _grp_ranks11 = torch.distributed.get_process_group_ranks(
                            _tp_group11)
                        _src_global11 = int(_grp_ranks11[_src_tp_rank11])
                        _ps_dist11.broadcast(_new_topk_all, src=_src_global11, group=_tp_group11)
                        _ps_dist11.broadcast(_new_score_all, src=_src_global11, group=_tp_group11)
                        # [PS-fix13-rc] 两次 pass 的 reuser topk/score 位级比对。
                        # call#==1 存快照(按 key,forward_id 变更时清理旧 key);
                        # call#==2 比对。topk 差异 = 重算 pass 与首次 pass 选择不同的
                        # 压缩块(recompute 非位级确定);score 按 uint8 字节比对。
                        q_len_local = valid_len
                        topk_len = min(_topk_w, compress_topk_idxs.shape[-1])
                        # 广播后的行 [0:valid_len) 即 reuser 的查询;写回 packed 表的
                        # reuser 区段,并把本地索引偏移到展开 cmp 表的 reuser 区段起点。
                        _new_topk_vals = _new_topk_all[:q_len_local, :topk_len].clone()
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
                        compress_topk_idxs[_topk_batch_idx,
                                           _row_start:_row_start + q_len_local,
                                           :topk_len] = _new_topk_vals
                        if compress_topk_score is not None:
                            # [PS-fix11g] 窗口 < topk 宽度时 topk 会选入被 mask 的条目
                            # (idx=-1, score=finfo.min)。finfo.min(-3.4e38) 写回
                            # bf16 溢出成 -inf,原版模型从不给 kernel/损失喂 -inf
                            # score——这是 hook 独有输入,置 0.0 消除。
                            _new_score_vals = _new_score_all[:q_len_local, :topk_len].clone()
                            _new_score_vals[_new_topk_vals == -1] = 0.0
                            compress_topk_score[_topk_batch_idx,
                                               _row_start:_row_start + q_len_local,
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
                    new_idxs = attention_module.get_compress_topk_idxs(
                        compress_ratio, bsz, expanded_seqlen,
                        start_pos=start_pos, offset=0, cp_shard=kv_allgather)
                    topk_len = min(new_idxs.shape[-1],
                                   compress_topk_idxs.shape[-1])
                    # get_compress_topk_idxs returns LOCAL indices [0, seqlen//r).
                    # Add provider's CMP block offset to convert to global packed indices.
                    # Provider CMP blocks before this reuser = global offset
                    # [PS-fix7b] cmp_lengths 已随 fix7 移除:用 Q 侧真实 cmp 长度求和
                    _cmp_offset = (
                        sum(vl // compress_ratio for vl in layout.valid_lengths[:batch_idx])
                        if cmp_rows else 0
                    )
                    _new_idxs = new_idxs[_topk_batch_idx, -q_len_local:, :topk_len].clone()
                    _new_idxs[_new_idxs >= 0] += _cmp_offset
                    # [PS-fix11] 与 indexer 分支相同的行定位修复:reuser 的行位于
                    # packed 表的 [_row_start, _row_start+q_len_local),
                    # 写 [0:q_len_local) 会覆盖 provider 的 topk。
                    _row_start_p = sum(layout.valid_lengths[:batch_idx])
                    compress_topk_idxs[_topk_batch_idx,
                                       _row_start_p:_row_start_p + q_len_local,
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

    # [PS-fix10 实验性] indexer 的 topk 索引空间与展开 cmp 表不匹配(实测 max=2028 > 1151)。
    # 先把越界索引钳制为 -1(哨兵,"无块"),验证越界是 NaN 的唯一原因;
    # 语义级重映射是后续 PS 仓工作。
    if compress_topk_idxs is not None and result_cmp is not None:
        _cmp_n = result_cmp.shape[0]
        _n_clamped = int((compress_topk_idxs > _cmp_n - 1).sum())
        compress_topk_idxs = compress_topk_idxs.clamp(max=_cmp_n - 1)

    return (result_kv, result_cmp, result_idxk,
            compress_topk_idxs, packed_seq_params, compress_topk_score)
