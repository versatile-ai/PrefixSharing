"""DeepSeek4SelfAttention.forward → prefix-sharing KV store/expand hook.

Minimal fork: copies Phase 1-3 orchestration code (~40 lines) from
the original forward to gain variable access, inserts one hook call
between Phase 3 and Phase 4.  All computation methods are still
called on ``self`` — only the orchestration is forked.
"""

from __future__ import annotations

from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.core.prefix_store import G2AttentionStore
from prefix_sharing.integrations.g2_attention import _g2_kv_store_or_expand


def patch_g2_attention(original_forward):
    """Patch DeepSeek4SelfAttention.forward with a single insertion point.

    Context not active or wrong store type → original_forward (no-op).
    """

    def patched_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb=None,
        start_pos: int = 0,
        attention_bias=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        sequence_len_offset=None,
    ):
        ctx = current_prefix_sharing_context()
        if ctx is None or not isinstance(ctx.store, G2AttentionStore):
            return original_forward(
                self, hidden_states, attention_mask, rotary_pos_emb,
                start_pos=start_pos, packed_seq_params=packed_seq_params,
                attention_bias=attention_bias, inference_context=inference_context,
                rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
                sequence_len_offset=sequence_len_offset)

        # ── Phase 1-3: copied orchestration (no logic changes) ──
        import torch
        import torch_npu
        from einops import rearrange
        from contextlib import nullcontext
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
        from megatron.training import get_args
        from mindspeed_llm.core.context_parallel.kvallgather_context_parallel import (
            gather_from_sp_cp, permute_cp_shard)
        from mindspeed_llm.tasks.models.transformer.deepseek4.deepseek_utils import (
            apply_rotary_emb)
        from mindspeed_llm.ops.npu_sparse_flash_mla import npu_sparse_flash_mla
        from mindspeed_llm.tasks.models.transformer.dsa_indexer import (
            DSAIndexerLossAutoScaler, compute_dsa_indexer_loss_dsv4, get_attn_scores,
            DSAIndexerLossLoggingHelper)

        args = get_args()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        cp_size = parallel_state.get_context_parallel_world_size()

        q_len_local, bsz, _ = hidden_states.shape
        q_len = q_len_local * tp_size if self.config.sequence_parallel else q_len_local
        q_len_global = q_len * cp_size if cp_size > 1 else q_len

        self.freqs_cis = rotary_pos_emb[0] if self.compress_ratio > 1 else rotary_pos_emb[1]
        self.freqs_cis = self.freqs_cis[start_pos: start_pos + q_len_global]
        if self.kv_allgather:
            self.freqs_cis = permute_cp_shard(self.freqs_cis, reorder=False)

        q_compressed = self.linear_q(hidden_states)
        kv_compressed = self.linear_kv(hidden_states)

        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(q_len, bsz, self.n_local_heads, -1)

        if args.use_fused_rmsnorm:
            nD = q.shape[-1]
            norm_gamma = torch.ones(nD, device=q.device, dtype=torch.float32)
            q = torch_npu.npu_rms_norm(q, gamma=norm_gamma, epsilon=self.config.layernorm_epsilon)[0]
        else:
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.config.layernorm_epsilon)

        q = q.transpose(0, 1)
        global_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=True)
        local_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=False)
        q[..., -self.rope_head_dim:] = apply_rotary_emb(q[..., -self.rope_head_dim:], global_freqs_cis)
        q = q.transpose(0, 1)

        kv = self.kv_layernorm(kv_compressed)
        kv = kv.transpose(0, 1)
        kv[..., -self.rope_head_dim:] = apply_rotary_emb(kv[..., -self.rope_head_dim:], local_freqs_cis)
        kv = kv.transpose(0, 1)
        if self.config.sequence_parallel or self.kv_allgather:
            kv = gather_from_sp_cp(kv)

        # Phase 2: compress_topk_idxs
        compress_topk_idxs = None
        key_index = None
        query_index = None
        weights = None
        dsa_hidden_states = None

        if self.compress_ratio > 1:
            offset = 0 if self.use_sparse_flash_attn else kv.size(0)
            if self.indexer is not None:
                query_index, key_index, weights, dsa_hidden_states = (
                    self.indexer.forward_with_index_compress(
                        hidden_states.detach(), q_compressed.detach(),
                        start_pos, local_freqs_cis, packed_seq_params))
                query_index, key_index, weights = (
                    self.indexer.all_gather_qk_weight_kvallgather(
                        query_index, key_index, weights))
                dsa_indexer_context = (
                    torch.no_grad() if args.use_fused_lightning_indexer_loss
                    else nullcontext())
                with dsa_indexer_context:
                    compress_topk_idxs, compress_topk_score = (
                        self.indexer.forward_with_scores_compress(
                            dsa_hidden_states, query_index, key_index, weights,
                            attention_mask, packed_seq_params, start_pos,
                            self.indexer.index_topk, offset,
                            self.indexer.compress_ratio))
                    compress_topk_idxs, compress_topk_score = (
                        self.indexer.post_process_index(
                            compress_topk_idxs, compress_topk_score))
            else:
                compress_topk_idxs = self.get_compress_topk_idxs(
                    self.compress_ratio, bsz, q_len_global, start_pos, offset,
                    self.kv_allgather)

        # Phase 3: Compressed KV
        kv_compress = None
        if self.compress_ratio > 1:
            kv_compress = self.compressor(
                hidden_states, start_pos, local_freqs_cis, packed_seq_params)
            if kv_compress is not None:
                if self.config.sequence_parallel or self.kv_allgather:
                    kv_compress = gather_from_sp_cp(kv_compress)

        # ═══════════ Hook: Store / Expand ═══════════
        indexer_k = key_index if self.indexer is not None else None
        kv, kv_compress, indexer_k, compress_topk_idxs, packed_seq_params = (
            _g2_kv_store_or_expand(
                ctx, kv, kv_compress, indexer_k,
                compress_topk_idxs, packed_seq_params,
                self.compress_ratio, self, start_pos,
                self.kv_allgather, self.config.sequence_parallel,
                query_index=query_index,
                indexer_weights=weights,
                dsa_hidden=dsa_hidden_states,
                attention_mask=attention_mask))
        # ═════════════════════════════════════════════

        # ── Phase 4-5: copied orchestration ──
        self.attn_sink = self.attn_sink.to(hidden_states.device)

        use_smla_with_slig = (
            self.indexer is not None
            and args.use_g2_indexer_loss
            and torch.is_grad_enabled()
            and args.use_fused_lightning_indexer_loss)
        if use_smla_with_slig:
            o = self.sparse_attention_with_indexer_loss(
                q, kv, kv_compress, compress_topk_idxs,
                self.attn_sink, self.softmax_scale, self.compress_ratio,
                q_len_global, query_index, key_index, weights, packed_seq_params)
        else:
            o = self.sparse_attention(
                q, kv, kv_compress, compress_topk_idxs,
                self.attn_sink, self.softmax_scale, self.compress_ratio,
                q_len_global, packed_seq_params)
            if (args.use_g2_indexer_loss and self.compress_ratio > 1
                    and self.indexer is not None and torch.is_grad_enabled()):
                compress_topk_idxs_adj = (
                    torch.where(compress_topk_idxs == -1, compress_topk_idxs,
                                compress_topk_idxs - offset)
                    if offset != 0 else compress_topk_idxs)
                if tp_size > 1:
                    total_query = gather_from_tensor_model_parallel_region(
                        q.view(*q.shape[:2], -1))
                    total_query = total_query.view(*q.shape[:2], -1, q.shape[-1])
                else:
                    total_query = q
                if len(kv_compress.shape) == 3:
                    kv_compress_exp = kv_compress.unsqueeze(2)
                else:
                    kv_compress_exp = kv_compress
                main_attn_dist = get_attn_scores(
                    total_query.detach(), kv_compress_exp.detach(),
                    attention_mask, self.n_local_heads * tp_size,
                    self.softmax_scale, allgather_q=True)
                loss = compute_dsa_indexer_loss_dsv4(
                    main_attn_dist, compress_topk_score,
                    compress_topk_idxs_adj, args.indexer_loss_coeff,
                    cmp_ratio=self.compress_ratio)
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss, self.layer_number, self.config.num_layers,
                    avg_group=parallel_state.get_tensor_and_context_parallel_group())
                o = DSAIndexerLossAutoScaler.apply(o, loss)

        o = o.transpose(0, 1)
        o_rotated = o.clone()
        o_rotated[..., -self.rope_head_dim:] = apply_rotary_emb(
            o[..., -self.rope_head_dim:], global_freqs_cis, True)
        o = o_rotated.transpose(0, 1)

        o = rearrange(o, 's b (g h) d -> s b g (h d)',
                      s=q_len, b=bsz, g=self.n_groups // self.world_size,
                      h=self.n_heads // self.n_groups, d=self.head_dim)
        weight_woa = rearrange(
            self.linear_o_down_proj.weight,
            '(g l) (d h)->g l (d h)',
            d=self.head_dim // self.n_groups,
            l=self.o_lora_rank, h=self.n_heads, g=self.n_local_groups)
        o = torch.einsum("sbgd,gld->sbgl", o, weight_woa)
        core_attn_out, bias = self.linear_o_up_proj(o.flatten(2))
        return core_attn_out, bias

    return patched_forward
