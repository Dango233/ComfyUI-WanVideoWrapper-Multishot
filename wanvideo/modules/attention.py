import torch
from typing import List, Optional, Sequence

from einops import rearrange

from ...utils import log
from .shot_utils import normalize_smooth_windows

from comfy.ldm.modules.attention import optimized_attention

def attention_func_error(*args, **kwargs):
    raise ImportError("Selected attention mode not available. Please ensure required packages are installed correctly.")

from .attention_flash import flash_attention

_VARLEN_FALLBACK_WARNED = False

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception:
    FLASH_ATTN_2_AVAILABLE = False

# Sage Attention imports
# using custom ops to avoid graph breaks with torch.compile
try:
    from sageattention import sageattn

    @torch.library.custom_op("wanvideo::sageattn", mutates_args=())
    def sageattn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, tensor_layout: str = "HND"
    ) -> torch.Tensor:
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
        elif q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)

    @sageattn_func.register_fake
    def _(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, tensor_layout="HND"):
        # Return tensor with same shape as q
        return q.clone()

    sageattn_func = torch.ops.wanvideo.sageattn

    def sageattn_func_compiled(q, k, v, attn_mask=None, dropout_p=0, is_causal=False, tensor_layout="HND"):
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
        elif q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
except Exception as e:
    log.warning(f"Warning: Could not load sageattention: {str(e)}")
    if isinstance(e, ModuleNotFoundError):
        log.warning("sageattention package is not installed, sageattention will not be available")
    elif isinstance(e, ImportError) and "DLL" in str(e):
        log.warning("sageattention DLL loading error, sageattention will not be available")
    sageattn_func = attention_func_error

try:
    from sageattention import sageattn_varlen
    from typing import List

    @torch.library.custom_op("wanvideo::sageattn_varlen", mutates_args=())
    def sageattn_varlen_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_lens: List[int], k_lens: List[int], max_seqlen_q: int, max_seqlen_k: int, dropout_p: float = 0.0, is_causal: bool = False) -> torch.Tensor:
        cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), dim=0)), device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(k_lens), dim=0)), device=q.device, dtype=torch.int32)
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn_varlen(q, k.to(q.dtype), v.to(q.dtype), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)
        elif q.dtype == torch.float32:
            return sageattn_varlen(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal).to(torch.float32)
        else:
            return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)

    @sageattn_varlen_func.register_fake
    def _(q, k, v, q_lens, k_lens, max_seqlen_q, max_seqlen_k, dropout_p=0.0, is_causal=False):
        # Return tensor with same shape as q
        return q.clone()
    sageattn_varlen_func = torch.ops.wanvideo.sageattn_varlen
except:
    sageattn_varlen_func = attention_func_error

# sage3
try:
    from sageattn3 import sageattn3_blackwell as sageattn_blackwell
except:
    try:
        from sageattn import sageattn_blackwell
    except:
        sageattn_blackwell = attention_func_error

try:
    from ...ultravico.sageattn.core import sage_attention as sageattn_ultravico
    @torch.library.custom_op("wanvideo::sageattn_ultravico", mutates_args=())
    def sageattn_func_ultravico(qkv: List[torch.Tensor], attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, multi_factor: float = 0.9, frame_tokens: int = 1536
    ) -> torch.Tensor:
        return sageattn_ultravico(qkv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, multi_factor=multi_factor, frame_tokens=frame_tokens)

    @sageattn_func_ultravico.register_fake
    def _(qkv, attn_mask=None, dropout_p=0.0, is_causal=False, multi_factor=0.9):
        return torch.empty_like(qkv[0]).contiguous()
    sageattn_func_ultravico = torch.ops.wanvideo.sageattn_ultravico
except:
    sageattn_func_ultravico = attention_func_error


def attention(q, k, v, q_lens=None, k_lens=None, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.,
    softmax_scale=None, q_scale=None, causal=False,  window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
    attention_mode='sdpa', attn_mask=None, transformer_options={}, frame_tokens=1536, heads=128):
    if "flash" in attention_mode:
        return flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic, dtype=dtype, version=2 if attention_mode == 'flash_attn_2' else 3,
        )
    elif attention_mode == 'sageattn_3':
        return sageattn_blackwell(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), per_block_mean=False).transpose(1,2).contiguous()
    elif attention_mode == 'sageattn_varlen':
        return sageattn_varlen_func(q,k,v, q_lens=q_lens, k_lens=k_lens, max_seqlen_k=max_seqlen_k, max_seqlen_q=max_seqlen_q)
    elif attention_mode == 'sageattn_compiled': # for sage versions that allow torch.compile, may be redundant now as other sageattn ops are wrapper in custom ops
        return sageattn_func_compiled(q, k, v, tensor_layout="NHD").contiguous()
    elif attention_mode == 'sageattn':
        return sageattn_func(q, k, v, tensor_layout="NHD").contiguous()
    elif attention_mode == 'sageattn_ultravico':
        return sageattn_func_ultravico([q, k, v], multi_factor=transformer_options.get("ultravico_alpha", 0.9), frame_tokens=frame_tokens).contiguous()
    elif attention_mode == 'comfy':
        return optimized_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), heads=heads, skip_reshape=True)
    else: # sdpa
        if attn_mask is not None and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)
        if not (q.dtype == k.dtype == v.dtype):
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype), attn_mask=attn_mask).transpose(1, 2).contiguous()
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask).transpose(1, 2).contiguous()


def _get_flash_attn_varlen(required: bool = True):
    if FLASH_ATTN_3_AVAILABLE:
        return flash_attn_interface.flash_attn_varlen_func
    if FLASH_ATTN_2_AVAILABLE:
        return flash_attn.flash_attn_varlen_func
    if required:
        raise RuntimeError("flash-attn varlen kernel is required for sparse shot attention")
    return None


def _build_global_reps(
    locals_k: List[torch.Tensor],
    locals_v: List[torch.Tensor],
    g_per: int,
    mode: str,
):
    if g_per <= 0 or len(locals_k) == 0:
        empty_k = torch.empty(0, *locals_k[0].shape[1:], device=locals_k[0].device, dtype=locals_k[0].dtype)
        empty_v = torch.empty_like(empty_k)
        return empty_k, empty_v

    reps_k, reps_v = [], []
    for k_shot, v_shot in zip(locals_k, locals_v):
        if k_shot.size(0) == 0:
            continue
        if mode == "mean":
            if g_per >= k_shot.size(0):
                reps_k.append(k_shot)
                reps_v.append(v_shot)
            else:
                segment_edges = torch.linspace(0, k_shot.size(0), steps=g_per + 1, device=k_shot.device, dtype=torch.float32)
                seg_k: List[torch.Tensor] = []
                seg_v: List[torch.Tensor] = []
                for seg_idx in range(g_per):
                    start = int(segment_edges[seg_idx].floor().item())
                    end = int(segment_edges[seg_idx + 1].ceil().item())
                    end = max(end, start + 1)
                    end = min(end, k_shot.size(0))
                    slice_k = k_shot[start:end]
                    slice_v = v_shot[start:end]
                    if slice_k.numel() == 0:
                        slice_k = k_shot[end - 1:end]
                        slice_v = v_shot[end - 1:end]
                    seg_k.append(slice_k.mean(dim=0, keepdim=True))
                    seg_v.append(slice_v.mean(dim=0, keepdim=True))
                reps_k.append(torch.cat(seg_k, dim=0))
                reps_v.append(torch.cat(seg_v, dim=0))
        elif mode == "linspace":
            idx = torch.linspace(0, k_shot.size(0) - 1, steps=g_per, device=k_shot.device).long()
            reps_k.append(k_shot.index_select(0, idx))
            reps_v.append(v_shot.index_select(0, idx))
        else:  # firstk
            take = min(g_per, k_shot.size(0))
            reps_k.append(k_shot[:take])
            reps_v.append(v_shot[:take])

    if len(reps_k) == 0:
        empty_k = torch.empty(0, *locals_k[0].shape[1:], device=locals_k[0].device, dtype=locals_k[0].dtype)
        empty_v = torch.empty_like(empty_k)
        return empty_k, empty_v

    return torch.cat(reps_k, dim=0), torch.cat(reps_v, dim=0)


def sparse_shot_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shot_latent_indices: Sequence[Sequence[int]],
    num_heads: int,
    per_g: int = 64,
    mode: str = "firstk",
    causal: bool = False,
    backend: str = "auto",
    attention_mode: str = "sdpa",
    prefix_tokens: int = 0,
    smooth_windows: Optional[Sequence[int]] = None,
):
    """Shot-aware attention with optional varlen flash kernels or dense fallback."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share the same shape")

    smooth_values = normalize_smooth_windows(smooth_windows)

    backend = (backend or "sparse_fallback").lower()
    backend = {
        "flash": "sparse_flash_attn",
        "dense": "sparse_fallback",
    }.get(backend, backend)
    attn_mode_effective = (attention_mode or "sdpa")

    if backend not in {"sparse_flash_attn", "sparse_fallback"}:
        raise ValueError(f"Unsupported sparse shot attention backend '{backend}'.")

    varlen_attn = None
    if backend == "sparse_flash_attn":
        varlen_attn = _get_flash_attn_varlen(required=False)
        if varlen_attn is None:
            global _VARLEN_FALLBACK_WARNED
            if not _VARLEN_FALLBACK_WARNED:
                log.warning(
                    "Shot attention requested sparse_flash_attn backend but flash varlen kernel not available; using sparse_fallback instead.",
                    attn_mode_effective,
                )
                _VARLEN_FALLBACK_WARNED = True
            backend = "sparse_fallback"

    if backend != "sparse_flash_attn":
        return _sparse_shot_attention_fallback(
            q,
            k,
            v,
            shot_latent_indices=shot_latent_indices,
            per_g=per_g,
            mode=mode,
            causal=causal,
            attention_mode=attn_mode_effective,
            prefix_tokens=prefix_tokens,
            smooth_windows=smooth_values,
        )

    if varlen_attn is None:
        raise RuntimeError("flash-attn varlen kernel is required for sparse_flash_attn backend")

    batch, seqlen, heads, head_dim = q.shape
    q = rearrange(q, "b s h d -> b h s d").contiguous()
    k = rearrange(k, "b s h d -> b h s d").contiguous()
    v = rearrange(v, "b s h d -> b h s d").contiguous()

    outputs = []

    for b_idx in range(batch):
        cuts = list(shot_latent_indices[b_idx])
        if not cuts or cuts[0] != 0 or cuts[-1] != seqlen:
            raise ValueError("shot_latent_indices must start at 0 and end at sequence length")

        q_shots = [q[b_idx, :, cuts[i]:cuts[i + 1], :] for i in range(len(cuts) - 1)]
        k_shots = [k[b_idx, :, cuts[i]:cuts[i + 1], :] for i in range(len(cuts) - 1)]
        v_shots = [v[b_idx, :, cuts[i]:cuts[i + 1], :] for i in range(len(cuts) - 1)]

        q_locals = [rearrange(qi, "h s d -> s h d") for qi in q_shots]
        k_locals = [rearrange(ki, "h s d -> s h d") for ki in k_shots]
        v_locals = [rearrange(vi, "h s d -> s h d") for vi in v_shots]

        prefix_tokens = max(int(prefix_tokens or 0), 0)
        prefix_k = prefix_v = None
        if prefix_tokens > 0 and len(k_locals) > 0:
            take = min(prefix_tokens, k_locals[0].size(0))
            if take > 0:
                prefix_k = k_locals[0][:take]
                prefix_v = v_locals[0][:take]

        k_global, v_global = _build_global_reps(k_locals, v_locals, per_g, mode)

        kv_lengths = []
        k_concat, v_concat = [], []
        num_shots = len(k_locals)
        overlap_prev = [0] * num_shots
        overlap_next = [0] * num_shots
        if smooth_values:
            limit = min(len(smooth_values), num_shots - 1)
            for sid in range(limit):
                window = smooth_values[sid]
                if window <= 0 or sid + 1 >= num_shots:
                    continue
                cur_len = k_locals[sid].size(0)
                next_len = k_locals[sid + 1].size(0)
                share_len = min(int(window), cur_len, next_len)
                if share_len <= 0:
                    continue
                overlap_next[sid] = share_len
                overlap_prev[sid + 1] = share_len

        for shot_idx, (k_local, v_local) in enumerate(zip(k_locals, v_locals)):
            parts_k = [k_local]
            parts_v = [v_local]
            if k_global.numel() > 0:
                parts_k.append(k_global)
                parts_v.append(v_global)
            if prefix_k is not None and shot_idx != 0:
                parts_k.append(prefix_k)
                parts_v.append(prefix_v)
            prev_share = overlap_prev[shot_idx] if shot_idx < len(overlap_prev) else 0
            if prev_share > 0 and shot_idx > 0:
                prev_local = k_locals[shot_idx - 1]
                parts_k.append(prev_local[-prev_share:])
                parts_v.append(v_locals[shot_idx - 1][-prev_share:])
            next_share = overlap_next[shot_idx] if shot_idx < len(overlap_next) else 0
            if next_share > 0 and shot_idx + 1 < num_shots:
                next_local = k_locals[shot_idx + 1]
                parts_k.append(next_local[:next_share])
                parts_v.append(v_locals[shot_idx + 1][:next_share])
            k_cat = torch.cat(parts_k, dim=0)
            v_cat = torch.cat(parts_v, dim=0)
            k_concat.append(k_cat)
            v_concat.append(v_cat)
            kv_lengths.append(k_cat.size(0))

        q_packed = torch.cat(q_locals, dim=0)
        k_packed = torch.cat(k_concat, dim=0)
        v_packed = torch.cat(v_concat, dim=0)

        shot_lengths = [end - start for start, end in zip(cuts[:-1], cuts[1:])]
        q_cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(shot_lengths, device=q.device), dim=0)), device=q.device, dtype=torch.int32)
        kv_cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(kv_lengths, device=q.device), dim=0)), device=q.device, dtype=torch.int32)

        max_q = max(shot_lengths) if shot_lengths else 0
        max_kv = max(kv_lengths) if kv_lengths else 0

        out_packed = varlen_attn(
            q_packed,
            k_packed,
            v_packed,
            q_seqlens=q_cu,
            k_seqlens=kv_cu,
            max_seqlen_q=max_q,
            max_seqlen_k=max_kv,
            causal=causal,
        )

        out_chunks = []
        for i in range(len(shot_lengths)):
            start = q_cu[i].item()
            end = q_cu[i + 1].item()
            out_chunks.append(out_packed[start:end])
        out_local = torch.cat(out_chunks, dim=0)
        outputs.append(rearrange(out_local, "s h d -> h s d"))

    stacked = torch.stack(outputs, dim=0)
    return rearrange(stacked, "b h s d -> b s h d")


def _sparse_shot_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shot_latent_indices: Sequence[Sequence[int]],
    per_g: int,
    mode: str,
    causal: bool,
    attention_mode: str,
    prefix_tokens: int = 0,
    smooth_windows: Optional[Sequence[int]] = None,
):
    batch, seqlen, heads, head_dim = q.shape
    q_bhg = rearrange(q, "b s h d -> b h s d").contiguous()
    k_bhg = rearrange(k, "b s h d -> b h s d").contiguous()
    v_bhg = rearrange(v, "b s h d -> b h s d").contiguous()

    outputs = []

    for b_idx in range(batch):
        cuts = list(shot_latent_indices[b_idx])
        if not cuts or cuts[0] != 0 or cuts[-1] != seqlen:
            raise ValueError("shot_latent_indices must start at 0 and end at sequence length")

        q_shots = [q_bhg[b_idx, :, start:end, :] for start, end in zip(cuts[:-1], cuts[1:])]
        k_shots = [k_bhg[b_idx, :, start:end, :] for start, end in zip(cuts[:-1], cuts[1:])]
        v_shots = [v_bhg[b_idx, :, start:end, :] for start, end in zip(cuts[:-1], cuts[1:])]

        q_locals = [rearrange(qi, "h s d -> s h d") for qi in q_shots]
        k_locals = [rearrange(ki, "h s d -> s h d") for ki in k_shots]
        v_locals = [rearrange(vi, "h s d -> s h d") for vi in v_shots]

        prefix_tokens = max(int(prefix_tokens or 0), 0)
        prefix_k = prefix_v = None
        if prefix_tokens > 0 and len(k_locals) > 0:
            take = min(prefix_tokens, k_locals[0].size(0))
            if take > 0:
                prefix_k = k_locals[0][:take]
                prefix_v = v_locals[0][:take]

        k_global, v_global = _build_global_reps(k_locals, v_locals, per_g, mode)

        out_locals = []
        num_shots = len(k_locals)
        overlap_prev = [0] * num_shots
        overlap_next = [0] * num_shots
        if smooth_windows:
            limit = min(len(smooth_windows), num_shots - 1)
            for sid in range(limit):
                window = smooth_windows[sid]
                if window <= 0 or sid + 1 >= num_shots:
                    continue
                cur_len = k_locals[sid].size(0)
                next_len = k_locals[sid + 1].size(0)
                share_len = min(int(window), cur_len, next_len)
                if share_len <= 0:
                    continue
                overlap_next[sid] = share_len
                overlap_prev[sid + 1] = share_len

        for shot_idx, (k_local, v_local, q_local) in enumerate(zip(k_locals, v_locals, q_locals)):
            parts_k = [k_local]
            parts_v = [v_local]
            if k_global.numel() > 0:
                parts_k.append(k_global)
                parts_v.append(v_global)
            if prefix_k is not None and shot_idx != 0:
                parts_k.append(prefix_k)
                parts_v.append(prefix_v)
            prev_share = overlap_prev[shot_idx] if shot_idx < len(overlap_prev) else 0
            if prev_share > 0 and shot_idx > 0:
                prev_local = k_locals[shot_idx - 1]
                parts_k.append(prev_local[-prev_share:])
                parts_v.append(v_locals[shot_idx - 1][-prev_share:])
            next_share = overlap_next[shot_idx] if shot_idx < len(overlap_next) else 0
            if next_share > 0 and shot_idx + 1 < num_shots:
                next_local = k_locals[shot_idx + 1]
                parts_k.append(next_local[:next_share])
                parts_v.append(v_locals[shot_idx + 1][:next_share])

            k_cat = torch.cat(parts_k, dim=0)
            v_cat = torch.cat(parts_v, dim=0)

            q_chunk = rearrange(q_local, "s h d -> 1 s h d")
            k_chunk = rearrange(k_cat, "s h d -> 1 s h d")
            v_chunk = rearrange(v_cat, "s h d -> 1 s h d")

            attn_out = attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attention_mode=attention_mode,
                attn_mask=None,
                causal=causal,
            )

            out_locals.append(attn_out.squeeze(0))

        out_cat = torch.cat(out_locals, dim=0)
        outputs.append(rearrange(out_cat, "s h d -> h s d"))

    stacked = torch.stack(outputs, dim=0)
    return rearrange(stacked, "b h s d -> b s h d")
