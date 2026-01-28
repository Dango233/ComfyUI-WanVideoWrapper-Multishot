import importlib.metadata
import torch
import logging
import math
from tqdm import tqdm
from pathlib import Path
import gc
import os
import weakref
import types, collections
from comfy.utils import ProgressBar, copy_to_param, set_attr_param
from comfy.model_patcher import get_key_weight, string_to_seed
from comfy.lora import calculate_weight

from comfy.float import stochastic_rounding
from .custom_linear import remove_lora_from_module, remove_shot_lora_from_module
import folder_paths
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

import comfy.model_management as mm
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

try:
    from .gguf.gguf import GGUFParameter
except:
    pass

COLOR_CODES = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

def color_text(text, color):
    try:
        return f"{COLOR_CODES.get(color, COLOR_CODES['reset'])}{text}{COLOR_CODES['reset']}"
    except Exception:
        return text

class MetaParameter(torch.nn.Parameter):
    def __new__(cls, dtype, quant_type=None):
        data = torch.empty(0, dtype=dtype)
        self = torch.nn.Parameter(data, requires_grad=False)
        self.quant_type = quant_type
        return self

def _unwrap_module(module):
    current = module
    while True:
        next_mod = None
        for attr in ("_orig_mod", "module"):
            candidate = getattr(current, attr, None)
            if isinstance(candidate, torch.nn.Module):
                next_mod = candidate
                break
        if next_mod is None or next_mod is current:
            return current
        current = next_mod


def offload_transformer(transformer, remove_lora=True):
    target = _unwrap_module(transformer)
    for mod in {transformer, target}:
        for attr in ("teacache_state", "magcache_state", "easycache_state"):
            state = getattr(mod, attr, None)
            if state is not None:
                state.clear_all()

    # Always clear per-shot LoRA buffers (runtime-only) before offloading.
    try:
        remove_shot_lora_from_module(target)
    except Exception as exc:
        log.warning(f"Failed to clear per-shot LoRA before offload: {exc}")

    if getattr(target, "patched_linear", False):
        for name, param in target.named_parameters():
            if "loras" in name or "controlnet" in name:
                continue
            module = target
            subnames = name.split('.')
            for subname in subnames[:-1]:
                module = getattr(module, subname)
            attr_name = subnames[-1]
            if param.data.is_floating_point():
                meta_param = torch.nn.Parameter(torch.empty_like(param.data, device='meta'), requires_grad=False)
                setattr(module, attr_name, meta_param)
            elif isinstance(param.data, GGUFParameter):
                quant_type = getattr(param, 'quant_type', None)
                setattr(module, attr_name, MetaParameter(param.data.dtype, quant_type))
            else:
                pass
        if remove_lora:
            remove_lora_from_module(target)
    else:
        target.to(offload_device)

    for block in target.blocks:
        block.kv_cache = None
        if target.audio_model is not None and hasattr(block, 'audio_block'):
            block.audio_block = None

    mm.soft_empty_cache()
    gc.collect()


def hard_offload_transformer(transformer, remove_lora=True):
    """Force-release GPU memory even if the model stays referenced in node cache."""
    target = _unwrap_module(transformer)
    for mod in {transformer, target}:
        for attr in ("teacache_state", "magcache_state", "easycache_state"):
            state = getattr(mod, attr, None)
            if state is not None:
                state.clear_all()

    # Always clear per-shot LoRA buffers (runtime-only) before offloading.
    try:
        remove_shot_lora_from_module(target)
    except Exception as exc:
        log.warning(f"Failed to clear per-shot LoRA before hard offload: {exc}")

    for name, param in target.named_parameters():
        if "loras" in name or "controlnet" in name:
            continue
        module = target
        subnames = name.split('.')
        for subname in subnames[:-1]:
            module = getattr(module, subname)
        attr_name = subnames[-1]
        if param.data.is_floating_point():
            meta_param = torch.nn.Parameter(torch.empty_like(param.data, device='meta'), requires_grad=False)
            setattr(module, attr_name, meta_param)
        elif isinstance(param.data, GGUFParameter):
            quant_type = getattr(param, 'quant_type', None)
            setattr(module, attr_name, MetaParameter(param.data.dtype, quant_type))
        else:
            pass

    if remove_lora:
        remove_lora_from_module(target)

    for block in target.blocks:
        block.kv_cache = None
        if target.audio_model is not None and hasattr(block, 'audio_block'):
            block.audio_block = None

    mm.soft_empty_cache()
    gc.collect()


def init_blockswap(transformer, block_swap_args, model):
    if not transformer.patched_linear:
        if block_swap_args is not None:
            for name, param in transformer.named_parameters():
                if "block" not in name or "control_adapter" in name or "face" in name:
                    param.data = param.data.to(device)
                elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(offload_device)
                elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(offload_device)

            transformer.block_swap(
                block_swap_args["blocks_to_swap"] - 1 ,
                block_swap_args["offload_txt_emb"],
                block_swap_args["offload_img_emb"],
                vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
            )
        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
            for block in transformer.blocks:
                block.modulation = torch.nn.Parameter(block.modulation.to(device))
            transformer.head.modulation = torch.nn.Parameter(transformer.head.modulation.to(device))
        else:
            transformer.to(device)

def check_device_same(first_device, second_device):
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        second_device = torch.device("cuda", index=0)

    return first_device == second_device

# simplified version of the accelerate function https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py
def set_module_tensor_to_device(module, tensor_name, device, value=None, dtype=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    if value is not None:
        if dtype is None:
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    device_quantization = None
    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
            if dtype is not None and device in ["meta", torch.device("meta")]:
                if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype)

                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            new_value = param_cls(new_value, requires_grad=False).to(device)
            module._parameters[tensor_name] = new_value

    #if device != "cpu":
    #    mm.soft_empty_cache()

def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.31.0'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")

def print_memory(device, process="Sampling"):
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    log.info(f"[{process}] Max allocated memory: {max_memory=:.3f} GB")
    log.info(f"[{process}] Max reserved memory: {max_reserved=:.3f} GB")
    #memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    #log.info(f"Memory Summary:\n{memory_summary}")


def _cuda_tensor_bytes(obj, seen, depth, max_depth):
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if depth > max_depth:
        return 0
    try:
        if torch.is_tensor(obj):
            if obj.is_cuda:
                return obj.nelement() * obj.element_size()
            return 0
    except Exception:
        return 0

    if isinstance(obj, dict):
        return sum(_cuda_tensor_bytes(v, seen, depth + 1, max_depth) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_cuda_tensor_bytes(v, seen, depth + 1, max_depth) for v in obj)

    if hasattr(obj, "__dict__"):
        return _cuda_tensor_bytes(vars(obj), seen, depth + 1, max_depth)
    return 0


def _collect_cache_entries(cache_obj):
    entries = []
    if cache_obj is None:
        return entries

    # BasicCache / HierarchicalCache / LRUCache-like objects
    if hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, dict):
        def _walk(cache, prefix=""):
            for key, value in cache.cache.items():
                entries.append((f"{prefix}{repr(key)}", value))
            subcaches = getattr(cache, "subcaches", None)
            if isinstance(subcaches, dict):
                for subkey, subcache in subcaches.items():
                    _walk(subcache, prefix=f"{prefix}{repr(subkey)}/")
        _walk(cache_obj)
        return entries

    if isinstance(cache_obj, dict):
        entries.extend([(repr(k), v) for k, v in cache_obj.items()])
        return entries

    for attr in ("cache", "node_cache", "NODE_CACHE"):
        if hasattr(cache_obj, attr):
            val = getattr(cache_obj, attr)
            if isinstance(val, dict):
                entries.extend([(repr(k), v) for k, v in val.items()])
                return entries
    return entries


def _collect_cache_entry_handles(cache_obj):
    """Return list of (container_dict, key, display_key) for cache entries."""
    entries = []
    if cache_obj is None:
        return entries

    if hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, dict):
        def _walk(cache, prefix=""):
            for key in list(cache.cache.keys()):
                entries.append((cache.cache, key, f"{prefix}{repr(key)}"))
            subcaches = getattr(cache, "subcaches", None)
            if isinstance(subcaches, dict):
                for subkey, subcache in list(subcaches.items()):
                    _walk(subcache, prefix=f"{prefix}{repr(subkey)}/")
        _walk(cache_obj)
        return entries

    if isinstance(cache_obj, dict):
        for key in list(cache_obj.keys()):
            entries.append((cache_obj, key, repr(key)))
        return entries

    for attr in ("cache", "node_cache", "NODE_CACHE"):
        if hasattr(cache_obj, attr):
            val = getattr(cache_obj, attr)
            if isinstance(val, dict):
                for key in list(val.keys()):
                    entries.append((val, key, repr(key)))
                return entries
    return entries


def _ensure_execution_hook(execution):
    if getattr(execution, "_wanvideo_executor_hooked", False):
        return
    PromptExecutor = getattr(execution, "PromptExecutor", None)
    if PromptExecutor is None:
        return
    original_async = getattr(PromptExecutor, "execute_async", None)
    if original_async is None:
        return

    async def _wrapped_execute_async(self, *args, **kwargs):
        execution._wanvideo_executor_ref = weakref.ref(self)
        return await original_async(self, *args, **kwargs)

    PromptExecutor.execute_async = _wrapped_execute_async
    execution._wanvideo_executor_hooked = True


def _snapshot_from_cache_sources(cache_sources, max_depth):
    snapshot = {}
    for source_name, cache_obj in cache_sources:
        entries = _collect_cache_entries(cache_obj)
        entry_bytes = {}
        total_bytes = 0
        for key, value in entries:
            bytes_now = _cuda_tensor_bytes(value, set(), 0, max_depth)
            if bytes_now > 0:
                entry_bytes[key] = bytes_now
                total_bytes += bytes_now
        snapshot[source_name] = {
            "total": total_bytes,
            "entries": entry_bytes,
        }
    return snapshot


def _snapshot_node_cache_for_executor(executor, max_depth=6):
    if os.getenv("WANVIDEO_DEBUG_NODECACHE", "").lower() not in ("1", "true", "yes"):
        return None
    if executor is None or not hasattr(executor, "caches"):
        return None
    cache_sources = []
    for cache_name in ("outputs", "ui", "objects"):
        cache_obj = getattr(executor.caches, cache_name, None)
        if cache_obj is not None:
            cache_sources.append((f"PromptExecutor.caches.{cache_name}", cache_obj))
    if not cache_sources:
        return None
    return _snapshot_from_cache_sources(cache_sources, max_depth)


def _soft_empty_cache_raw():
    original = getattr(mm, "_wanvideo_soft_cache_original", None)
    if original is None:
        original = getattr(mm, "soft_empty_cache", None)
    if original is None:
        return
    try:
        original()
    except Exception:
        pass


def _evict_node_cache_entries(executor, tag="", min_delta_mb=64):
    """Evict cache entries one-by-one and report allocated-memory deltas."""
    if executor is None or not hasattr(executor, "caches"):
        return None
    if not torch.cuda.is_available():
        return None
    min_delta_mb = _get_min_delta_mb(default=min_delta_mb)
    report_sizes = _env_enabled("WANVIDEO_DEBUG_NODECACHE_EVICT_REPORT_SIZE")
    delta_only = _env_enabled("WANVIDEO_DEBUG_NODECACHE_EVICT_DELTA")
    topn_enabled = _env_enabled("WANVIDEO_DEBUG_NODECACHE_EVICT_TOPN")
    decode_enabled = _env_enabled("WANVIDEO_DEBUG_NODECACHE_EVICT_DECODE")
    probe_enabled = _env_enabled("WANVIDEO_DEBUG_NODECACHE_EVICT_PROBE")
    decode_depth = _get_env_int("WANVIDEO_DEBUG_NODECACHE_EVICT_DECODE_DEPTH", 6)
    decode_len = _get_env_int("WANVIDEO_DEBUG_NODECACHE_EVICT_DECODE_LEN", 240)
    logfile = _get_env_str("WANVIDEO_DEBUG_NODECACHE_EVICT_LOGFILE")
    logfile_len = _get_env_int("WANVIDEO_DEBUG_NODECACHE_EVICT_LOGFILE_LEN", 4096)
    logfile_depth = _get_env_int("WANVIDEO_DEBUG_NODECACHE_EVICT_LOGFILE_DEPTH", decode_depth)
    topn_min_mb = _get_env_float("WANVIDEO_DEBUG_NODECACHE_EVICT_TOPN_MIN_MB", 1.0)
    topn_limit = _get_max_items(default=10) if topn_enabled else 0
    cache_sources = []
    for cache_name in ("outputs", "ui", "objects"):
        cache_obj = getattr(executor.caches, cache_name, None)
        if cache_obj is not None:
            cache_sources.append((f"PromptExecutor.caches.{cache_name}", cache_obj))

    if not cache_sources:
        return None

    total_entries = 0
    evict_deltas = []
    before_all = _cuda_mem_snapshot()
    for source_name, cache_obj in cache_sources:
        entries = _collect_cache_entry_handles(cache_obj)
        if not entries:
            continue
        _cleanup_log(tag, f"{source_name} entries={len(entries)} start")
        total_entries += len(entries)
        for container, key, display_key in entries:
            key_text = display_key
            if len(key_text) > 160:
                key_text = key_text[:157] + "..."
            if decode_enabled:
                key_text = _format_cache_key(key, max_len=decode_len, max_depth=decode_depth)
            file_key_text = None
            if logfile:
                file_key_text = _format_cache_key(key, max_len=logfile_len, max_depth=logfile_depth)
            value = None
            try:
                if hasattr(container, "get"):
                    value = container.get(key, None)
                else:
                    value = container[key]
            except Exception:
                value = None
            probe_text = None
            if probe_enabled:
                try:
                    probe_text = _describe_cache_value(value)
                except Exception:
                    probe_text = "probe_error"
            if report_sizes:
                try:
                    bytes_now = _cuda_tensor_bytes(value, set(), 0, 6)
                except Exception:
                    bytes_now = 0
                if bytes_now > 0:
                    log.info(
                        f"[NodeCache] {tag} {source_name} {key_text} "
                        f"cuda_bytes {bytes_now / (1024 ** 2):.2f} MB"
                    )
            before = _cuda_mem_snapshot()
            try:
                del container[key]
            except Exception:
                try:
                    container.pop(key, None)
                except Exception:
                    pass
            value = None
            try:
                gc.collect()
            except Exception:
                pass
            after = _cuda_mem_snapshot()

            if before is None or after is None:
                continue
            alloc_delta = after["allocated"] - before["allocated"]
            reserv_delta = after["reserved"] - before["reserved"]
            freed_mb = (before["allocated"] - after["allocated"]) / (1024 ** 2)
            if freed_mb > 0:
                evict_deltas.append((freed_mb, alloc_delta, reserv_delta, source_name, key, key_text, probe_text))
            if delta_only:
                if (
                    abs(alloc_delta) < min_delta_mb * 1024 * 1024
                    and abs(reserv_delta) < min_delta_mb * 1024 * 1024
                ):
                    continue
            else:
                freed = before["allocated"] - after["allocated"]
                if freed < min_delta_mb * 1024 * 1024:
                    continue
            line = (
                f"[NodeCache] {tag} {source_name} {key_text} "
                f"allocated {before['allocated'] / (1024 ** 3):.3f} GB -> {after['allocated'] / (1024 ** 3):.3f} GB "
                f"(delta {alloc_delta / (1024 ** 3):.3f} GB) "
                f"reserved {before['reserved'] / (1024 ** 3):.3f} GB -> {after['reserved'] / (1024 ** 3):.3f} GB "
                f"(delta {reserv_delta / (1024 ** 3):.3f} GB)"
            )
            if probe_text:
                line = f"{line} probe={probe_text}"
            log.info(line)
            if logfile and file_key_text:
                file_line = (
                    f"[NodeCache] {tag} {source_name} {file_key_text} "
                    f"allocated {before['allocated'] / (1024 ** 3):.3f} GB -> {after['allocated'] / (1024 ** 3):.3f} GB "
                    f"(delta {alloc_delta / (1024 ** 3):.3f} GB) "
                    f"reserved {before['reserved'] / (1024 ** 3):.3f} GB -> {after['reserved'] / (1024 ** 3):.3f} GB "
                    f"(delta {reserv_delta / (1024 ** 3):.3f} GB)"
                )
                if probe_text:
                    file_line = f"{file_line} probe={probe_text}"
                _append_debug_file(logfile, file_line)
    if topn_limit and evict_deltas:
        evict_deltas.sort(key=lambda item: item[0], reverse=True)
        shown = 0
        for freed_mb, alloc_delta, reserv_delta, source_name, key_obj, key_text, saved_probe in evict_deltas:
            if freed_mb < topn_min_mb:
                continue
            if not decode_enabled:
                key_text = _format_cache_key(key_obj, max_len=decode_len, max_depth=decode_depth)
            line = (
                f"[NodeCache] {tag}_top {source_name} {key_text} "
                f"freed {freed_mb:.2f} MB "
                f"alloc_delta {alloc_delta / (1024 ** 3):.3f} GB "
                f"reserv_delta {reserv_delta / (1024 ** 3):.3f} GB"
            )
            if probe_enabled and saved_probe:
                line = f"{line} probe={saved_probe}"
            log.info(line)
            if logfile:
                file_key_text = _format_cache_key(key_obj, max_len=logfile_len, max_depth=logfile_depth)
                file_line = (
                    f"[NodeCache] {tag}_top {source_name} {file_key_text} "
                    f"freed {freed_mb:.2f} MB "
                    f"alloc_delta {alloc_delta / (1024 ** 3):.3f} GB "
                    f"reserv_delta {reserv_delta / (1024 ** 3):.3f} GB"
                )
                if probe_enabled and saved_probe:
                    file_line = f"{file_line} probe={saved_probe}"
                _append_debug_file(logfile, file_line)
            shown += 1
            if shown >= topn_limit:
                break
    after_entries = _cuda_mem_snapshot()
    if before_all is not None and after_entries is not None:
        report_cuda_mem_delta(before_all, after_entries, tag=f"{tag}_entries")

    before_flush = _cuda_mem_snapshot()
    _soft_empty_cache_raw()
    after_flush = _cuda_mem_snapshot()
    if before_flush is not None and after_flush is not None:
        report_cuda_mem_delta(before_flush, after_flush, tag=f"{tag}_allocator_flush")

    after_all = after_flush or after_entries
    if before_all is not None and after_all is not None:
        report_cuda_mem_delta(before_all, after_all, tag=f"{tag}_summary")
    log.info(f"[NodeCache] {tag} done entries={total_entries}")
    return True


def _ensure_prompt_executor_reset_hook(execution):
    if getattr(execution, "_wanvideo_reset_hooked", False):
        return
    PromptExecutor = getattr(execution, "PromptExecutor", None)
    if PromptExecutor is None:
        return
    original_reset = getattr(PromptExecutor, "reset", None)
    if original_reset is None:
        return

    def _wrapped_reset(self, *args, **kwargs):
        node_debug = _env_enabled("WANVIDEO_DEBUG_NODECACHE")
        mem_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_MEM")
        census_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_CENSUS")
        stats_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_STATS")
        snapshot_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_SNAPSHOT")
        evict_debug = _env_enabled("WANVIDEO_DEBUG_NODECACHE_EVICT")

        _cleanup_log("prompt_executor_reset", "start")
        if evict_debug:
            _evict_node_cache_entries(self, tag="prompt_executor_reset_evict")

        before_cache = _snapshot_node_cache_for_executor(self) if node_debug else None
        before_mem = _cuda_mem_snapshot() if mem_debug else None
        before_census = snapshot_cuda_tensor_census() if census_debug else None
        before_stats = snapshot_cuda_stats() if stats_debug else None
        before_snapshot = snapshot_cuda_snapshot_summary() if snapshot_debug else None

        result = original_reset(self, *args, **kwargs)
        _cleanup_log("prompt_executor_reset", "original reset done")

        after_cache = _snapshot_node_cache_for_executor(self) if node_debug else None
        after_mem = _cuda_mem_snapshot() if mem_debug else None
        after_census = snapshot_cuda_tensor_census() if census_debug else None
        after_stats = snapshot_cuda_stats() if stats_debug else None
        after_snapshot = snapshot_cuda_snapshot_summary() if snapshot_debug else None

        if node_debug:
            report_node_cache_cuda_delta(before_cache, after_cache, tag="prompt_executor_reset")
        if mem_debug:
            report_cuda_mem_delta(before_mem, after_mem, tag="prompt_executor_reset")
        if census_debug:
            report_cuda_tensor_census_delta(before_census, after_census, tag="prompt_executor_reset")
        if stats_debug:
            report_cuda_stats_delta(before_stats, after_stats, tag="prompt_executor_reset")
        if snapshot_debug:
            report_cuda_snapshot_delta(before_snapshot, after_snapshot, tag="prompt_executor_reset")
        _cleanup_log("prompt_executor_reset", "done")
        return result

    PromptExecutor.reset = _wrapped_reset
    execution._wanvideo_reset_hooked = True


def _find_prompt_executor_instances(execution):
    PromptExecutor = getattr(execution, "PromptExecutor", None)
    if PromptExecutor is None:
        return []
    instances = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, PromptExecutor):
                instances.append(obj)
        except Exception:
            continue
    return instances


def _get_execution_module():
    execution = None
    try:
        import execution as _execution
        execution = _execution
    except Exception:
        try:
            import comfy.execution as _execution
            execution = _execution
        except Exception:
            execution = None
    return execution


def _collect_node_cache_sources(execution):
    cache_sources = []
    _ensure_execution_hook(execution)
    _ensure_prompt_executor_reset_hook(execution)

    executor_ref = getattr(execution, "_wanvideo_executor_ref", None)
    if executor_ref is not None:
        try:
            executor = executor_ref()
        except Exception:
            executor = None
        if executor is not None and hasattr(executor, "caches"):
            for cache_name in ("outputs", "ui", "objects"):
                cache_obj = getattr(executor.caches, cache_name, None)
                if cache_obj is not None:
                    cache_sources.append((f"PromptExecutor.caches.{cache_name}", cache_obj))

    if not cache_sources:
        instances = _find_prompt_executor_instances(execution)
        if instances:
            execution._wanvideo_executor_ref = weakref.ref(instances[0])
            for cache_name in ("outputs", "ui", "objects"):
                cache_obj = getattr(instances[0].caches, cache_name, None)
                if cache_obj is not None:
                    cache_sources.append((f"PromptExecutor.caches.{cache_name}", cache_obj))

    if not cache_sources:
        for name in dir(execution):
            if "cache" not in name.lower():
                continue
            try:
                obj = getattr(execution, name)
            except Exception:
                continue
            cache_sources.append((f"execution.{name}", obj))

    return cache_sources


def snapshot_node_cache_cuda_usage(max_depth=6):
    if os.getenv("WANVIDEO_DEBUG_NODECACHE", "").lower() not in ("1", "true", "yes"):
        return None

    execution = _get_execution_module()
    if execution is None:
        log.warning("[NodeCache] execution module not available")
        return None

    cache_sources = _collect_node_cache_sources(execution)
    if not cache_sources:
        log.info("[NodeCache] no cache sources found")
        return {}

    return _snapshot_from_cache_sources(cache_sources, max_depth)


def report_node_cache_cuda_delta(before, after, tag="", max_items=10):
    if before is None or after is None:
        return None

    sources = set(before.keys()) | set(after.keys())
    for source in sorted(sources):
        before_info = before.get(source, {})
        after_info = after.get(source, {})
        before_total = before_info.get("total", 0)
        after_total = after_info.get("total", 0)
        if before_total == after_total:
            continue

        log.info(
            f"[NodeCache] {tag} {source} "
            f"{before_total / (1024 ** 3):.3f} GB -> {after_total / (1024 ** 3):.3f} GB "
            f"(delta {(after_total - before_total) / (1024 ** 3):.3f} GB)"
        )

        before_entries = before_info.get("entries", {})
        after_entries = after_info.get("entries", {})
        changes = []
        for key in set(before_entries.keys()) | set(after_entries.keys()):
            b = before_entries.get(key, 0)
            a = after_entries.get(key, 0)
            if a != b:
                changes.append((abs(a - b), a - b, key, b, a))

        changes.sort(key=lambda x: x[0], reverse=True)
        for _, delta, key, b, a in changes[:max_items]:
            key_text = key
            if len(key_text) > 160:
                key_text = key_text[:157] + "..."
            log.info(
                f"[NodeCache] {tag} {source} {key_text} "
                f"{b / (1024 ** 2):.2f} MB -> {a / (1024 ** 2):.2f} MB "
                f"(delta {delta / (1024 ** 2):.2f} MB)"
            )
    return True


def report_node_cache_cuda_usage(tag="", max_items=10, max_depth=6):
    """Best-effort scan of ComfyUI node cache for CUDA tensor usage."""
    if os.getenv("WANVIDEO_DEBUG_NODECACHE", "").lower() not in ("1", "true", "yes"):
        return None
    execution = _get_execution_module()
    if execution is None:
        log.warning(f"[NodeCache] {tag} execution module not available")
        return None

    cache_sources = _collect_node_cache_sources(execution)

    if not cache_sources:
        log.info(f"[NodeCache] {tag} no cache sources found")
        return None

    for source_name, cache_dict in cache_sources:
        entries = []
        total_bytes = 0
        cache_entries = _collect_cache_entries(cache_dict)
        for key, value in cache_entries:
            bytes_now = _cuda_tensor_bytes(value, set(), 0, max_depth)
            total_bytes += bytes_now
            if bytes_now > 0:
                entries.append((bytes_now, key, type(value).__name__))

        if total_bytes == 0:
            log.info(f"[NodeCache] {tag} {source_name} has no CUDA tensors")
            continue

        entries.sort(key=lambda x: x[0], reverse=True)
        log.info(
            f"[NodeCache] {tag} {source_name} entries={len(cache_entries)} "
            f"cuda={total_bytes / (1024 ** 3):.3f} GB"
        )
        for bytes_now, key, type_name in entries[:max_items]:
            key_text = repr(key)
            if len(key_text) > 160:
                key_text = key_text[:157] + "..."
            log.info(
                f"[NodeCache] {tag} {source_name} {type_name} {key_text} "
                f"{bytes_now / (1024 ** 2):.2f} MB"
            )
    return True


def _cuda_mem_snapshot(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        try:
            device = mm.get_torch_device()
        except Exception:
            device = torch.device("cuda")
    try:
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    except Exception:
        return None
    return {"device": str(device), "allocated": allocated, "reserved": reserved}


def report_cuda_mem_delta(before, after, tag="", min_delta_mb=32):
    if before is None or after is None:
        return None
    min_delta_mb = _get_min_delta_mb(default=min_delta_mb)
    alloc_delta = after["allocated"] - before["allocated"]
    reserv_delta = after["reserved"] - before["reserved"]
    if abs(alloc_delta) < min_delta_mb * 1024 * 1024 and abs(reserv_delta) < min_delta_mb * 1024 * 1024:
        return None
    log.info(
        f"[CUDA] {tag} {before['device']} "
        f"allocated {before['allocated'] / (1024 ** 3):.3f} GB -> {after['allocated'] / (1024 ** 3):.3f} GB "
        f"(delta {alloc_delta / (1024 ** 3):.3f} GB) "
        f"reserved {before['reserved'] / (1024 ** 3):.3f} GB -> {after['reserved'] / (1024 ** 3):.3f} GB "
        f"(delta {reserv_delta / (1024 ** 3):.3f} GB)"
    )
    return True


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def _cleanup_verbose_enabled() -> bool:
    return _env_enabled("WANVIDEO_DEBUG_CLEANUP_VERBOSE")


def _cleanup_log(tag: str, message: str):
    if not _cleanup_verbose_enabled():
        return
    log.info(f"[Cleanup] {tag} {message}")


def _get_min_delta_mb(default=64):
    raw = os.getenv("WANVIDEO_DEBUG_MIN_DELTA_MB")
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _get_max_items(default=20):
    raw = os.getenv("WANVIDEO_DEBUG_MAX_ITEMS")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _get_env_float(name: str, default: float):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _get_env_int(name: str, default: int):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _get_env_str(name: str, default: str | None = None):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


def _append_debug_file(path: str, line: str):
    try:
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write(line)
            f.write("\n")
    except Exception:
        pass


def _decode_cache_key(obj, depth=0, max_depth=6):
    if depth >= max_depth:
        return "<max_depth>"
    if isinstance(obj, frozenset):
        items = list(obj)
        if items and all(isinstance(it, tuple) and len(it) == 2 for it in items):
            try:
                items = sorted(items, key=lambda it: it[0])
            except Exception:
                pass
            decoded = []
            for k, v in items:
                decoded.append((_decode_cache_key(k, depth + 1, max_depth), _decode_cache_key(v, depth + 1, max_depth)))
            return decoded
        return [_decode_cache_key(it, depth + 1, max_depth) for it in items]
    if isinstance(obj, tuple):
        return tuple(_decode_cache_key(it, depth + 1, max_depth) for it in obj)
    if isinstance(obj, list):
        return [_decode_cache_key(it, depth + 1, max_depth) for it in obj]
    if isinstance(obj, dict):
        decoded = []
        for k, v in obj.items():
            decoded.append((_decode_cache_key(k, depth + 1, max_depth), _decode_cache_key(v, depth + 1, max_depth)))
        return decoded
    return obj


def _format_cache_key(obj, max_len=240, max_depth=6):
    try:
        decoded = _decode_cache_key(obj, 0, max_depth)
        text = repr(decoded)
    except Exception:
        text = repr(obj)
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _describe_cache_value(value, max_items=6, depth=0, max_depth=2, max_len=240):
    if value is None:
        return "None"
    if depth == 0:
        max_items = _get_env_int("WANVIDEO_DEBUG_NODECACHE_EVICT_PROBE_MAX_ITEMS", max_items)
        max_depth = _get_env_int("WANVIDEO_DEBUG_NODECACHE_EVICT_PROBE_DEPTH", max_depth)
        max_len = _get_env_int("WANVIDEO_DEBUG_NODECACHE_EVICT_PROBE_LEN", max_len)
    try:
        vtype = type(value)
        name = f"{vtype.__module__}.{vtype.__name__}"
    except Exception:
        name = "unknown"
    details = [name]
    try:
        if torch.is_tensor(value):
            details.append(f"tensor {tuple(value.shape)} {str(value.dtype)} {str(value.device)}")
    except Exception:
        pass
    try:
        if isinstance(value, (list, tuple)):
            details.append(f"{type(value).__name__}[{len(value)}]")
            if depth < max_depth:
                item_descs = []
                for idx, item in enumerate(list(value)[:max_items]):
                    desc = _describe_cache_value(item, max_items=max_items, depth=depth + 1, max_depth=max_depth, max_len=max_len)
                    item_descs.append(f"{idx}:{desc}")
                if item_descs:
                    details.append("items={" + "; ".join(item_descs) + "}")
    except Exception:
        pass
    try:
        if isinstance(value, dict):
            details.append(f"dict[{len(value)}]")
    except Exception:
        pass
    try:
        if hasattr(value, "model"):
            details.append("has_model")
    except Exception:
        pass
    try:
        if hasattr(value, "model_patcher"):
            details.append("has_model_patcher")
    except Exception:
        pass
    try:
        if hasattr(value, "model") and torch.is_tensor(value.model):
            details.append("model_tensor")
    except Exception:
        pass
    try:
        if hasattr(value, "model") and hasattr(value.model, "parameters"):
            params = 0
            on_cuda = 0
            for i, p in enumerate(value.model.parameters()):
                if i >= max_items:
                    break
                params += 1
                if getattr(p, "is_cuda", False):
                    on_cuda += 1
            details.append(f"model_params_cuda_sample={on_cuda}/{params}")
    except Exception:
        pass
    text = " | ".join(details)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def snapshot_cuda_tensor_census():
    if not _env_enabled("WANVIDEO_DEBUG_CUDA_CENSUS"):
        return None
    if not torch.cuda.is_available():
        return None

    census = {}
    total = 0
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj):
                continue
        except Exception:
            continue
        try:
            if not obj.is_cuda:
                continue
        except Exception:
            continue

        try:
            numel = obj.nelement()
            elem = obj.element_size()
            bytes_now = numel * elem
        except Exception:
            continue

        dtype = str(obj.dtype)
        device = str(obj.device)
        try:
            shape = tuple(obj.shape)
        except Exception:
            shape = ("?",)
        key = f"{type(obj).__name__} {device} {dtype} {shape}"
        census[key] = census.get(key, 0) + bytes_now
        total += bytes_now

    census["_total_bytes"] = total
    return census


def report_cuda_tensor_census_delta(before, after, tag="", min_delta_mb=64, max_items=20):
    if before is None or after is None:
        return None
    min_delta_mb = _get_min_delta_mb(default=min_delta_mb)
    max_items = _get_max_items(default=max_items)

    before_total = before.get("_total_bytes", 0)
    after_total = after.get("_total_bytes", 0)
    total_delta = after_total - before_total
    if abs(total_delta) < min_delta_mb * 1024 * 1024:
        return None

    log.info(
        f"[CUDA] {tag} tensor_census total "
        f"{before_total / (1024 ** 3):.3f} GB -> {after_total / (1024 ** 3):.3f} GB "
        f"(delta {total_delta / (1024 ** 3):.3f} GB)"
    )

    changes = []
    keys = set(before.keys()) | set(after.keys())
    keys.discard("_total_bytes")
    for key in keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        delta = a - b
        if abs(delta) >= min_delta_mb * 1024 * 1024:
            changes.append((abs(delta), delta, key, b, a))

    changes.sort(key=lambda x: x[0], reverse=True)
    for _, delta, key, b, a in changes[:max_items]:
        log.info(
            f"[CUDA] {tag} tensor_census {key} "
            f"{b / (1024 ** 2):.2f} MB -> {a / (1024 ** 2):.2f} MB "
            f"(delta {delta / (1024 ** 2):.2f} MB)"
        )
    return True


def snapshot_cuda_stats():
    if not _env_enabled("WANVIDEO_DEBUG_CUDA_STATS"):
        return None
    if not torch.cuda.is_available():
        return None
    try:
        device = mm.get_torch_device()
    except Exception:
        device = torch.device("cuda")
    try:
        stats = torch.cuda.memory_stats(device)
    except Exception:
        return None
    filtered = {}
    for k, v in stats.items():
        if not isinstance(v, (int, float)):
            continue
        if "bytes" not in k:
            continue
        if ".current" not in k:
            continue
        filtered[k] = int(v)
    filtered["_device"] = str(device)
    return filtered


def report_cuda_stats_delta(before, after, tag="", min_delta_mb=64, max_items=20):
    if before is None or after is None:
        return None
    min_delta_mb = _get_min_delta_mb(default=min_delta_mb)
    max_items = _get_max_items(default=max_items)

    device = before.get("_device") or after.get("_device") or "cuda"
    keys = set(before.keys()) | set(after.keys())
    keys.discard("_device")
    changes = []
    for key in keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        delta = a - b
        if abs(delta) >= min_delta_mb * 1024 * 1024:
            changes.append((abs(delta), delta, key, b, a))

    if not changes:
        return None

    changes.sort(key=lambda x: x[0], reverse=True)
    log.info(f"[CUDA] {tag} stats {device} changes={len(changes)}")
    for _, delta, key, b, a in changes[:max_items]:
        log.info(
            f"[CUDA] {tag} stats {key} "
            f"{b / (1024 ** 2):.2f} MB -> {a / (1024 ** 2):.2f} MB "
            f"(delta {delta / (1024 ** 2):.2f} MB)"
        )
    return True


def snapshot_cuda_snapshot_summary():
    if not _env_enabled("WANVIDEO_DEBUG_CUDA_SNAPSHOT"):
        return None
    if not torch.cuda.is_available():
        return None
    try:
        snapshot = torch.cuda.memory_snapshot()
    except Exception:
        return None

    summary = {
        "_total_segments": 0,
        "_total_bytes": 0,
        "_active_bytes": 0,
        "_inactive_bytes": 0,
    }

    for segment in snapshot:
        try:
            total_size = int(segment.get("total_size", 0))
        except Exception:
            total_size = 0
        summary["_total_segments"] += 1
        summary["_total_bytes"] += total_size

        blocks = segment.get("blocks", [])
        for block in blocks:
            try:
                size = int(block.get("size", 0))
            except Exception:
                size = 0
            state = block.get("state", "")
            if state == "active":
                summary["_active_bytes"] += size
            else:
                summary["_inactive_bytes"] += size

    return summary


def report_cuda_snapshot_delta(before, after, tag="", min_delta_mb=64):
    if before is None or after is None:
        return None
    min_delta_mb = _get_min_delta_mb(default=min_delta_mb)
    keys = ["_total_bytes", "_active_bytes", "_inactive_bytes", "_total_segments"]
    changes = []
    for key in keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        delta = a - b
        if key == "_total_segments":
            if delta != 0:
                changes.append((abs(delta), delta, key, b, a))
        else:
            if abs(delta) >= min_delta_mb * 1024 * 1024:
                changes.append((abs(delta), delta, key, b, a))

    if not changes:
        return None

    log.info(f"[CUDA] {tag} snapshot summary")
    for _, delta, key, b, a in changes:
        if key == "_total_segments":
            log.info(f"[CUDA] {tag} snapshot {key} {b} -> {a} (delta {delta})")
        else:
            log.info(
                f"[CUDA] {tag} snapshot {key} "
                f"{b / (1024 ** 2):.2f} MB -> {a / (1024 ** 2):.2f} MB "
                f"(delta {delta / (1024 ** 2):.2f} MB)"
            )
    return True


def _ensure_soft_empty_cache_hook():
    if getattr(mm, "_wanvideo_soft_cache_hooked", False):
        return
    original = getattr(mm, "soft_empty_cache", None)
    if original is None:
        return

    def _wrapped_soft_empty_cache(*args, **kwargs):
        mem_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_MEM")
        stats_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_STATS")
        snapshot_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_SNAPSHOT")

        before_mem = _cuda_mem_snapshot() if mem_debug else None
        before_stats = snapshot_cuda_stats() if stats_debug else None
        before_snapshot = snapshot_cuda_snapshot_summary() if snapshot_debug else None
        if _cleanup_verbose_enabled():
            snap = _cuda_mem_snapshot()
            if snap is not None:
                _cleanup_log("soft_empty_cache", f"before allocated={snap['allocated'] / (1024 ** 3):.3f} GB reserved={snap['reserved'] / (1024 ** 3):.3f} GB")

        result = original(*args, **kwargs)

        after_mem = _cuda_mem_snapshot() if mem_debug else None
        after_stats = snapshot_cuda_stats() if stats_debug else None
        after_snapshot = snapshot_cuda_snapshot_summary() if snapshot_debug else None
        if _cleanup_verbose_enabled():
            snap = _cuda_mem_snapshot()
            if snap is not None:
                _cleanup_log("soft_empty_cache", f"after allocated={snap['allocated'] / (1024 ** 3):.3f} GB reserved={snap['reserved'] / (1024 ** 3):.3f} GB")

        if mem_debug:
            report_cuda_mem_delta(before_mem, after_mem, tag="soft_empty_cache")
        if stats_debug:
            report_cuda_stats_delta(before_stats, after_stats, tag="soft_empty_cache")
        if snapshot_debug:
            report_cuda_snapshot_delta(before_snapshot, after_snapshot, tag="soft_empty_cache")
        return result

    mm._wanvideo_soft_cache_original = original
    mm.soft_empty_cache = _wrapped_soft_empty_cache
    mm._wanvideo_soft_cache_hooked = True


def cleanup_cuda_cache(tag=""):
    """Run gc + soft_empty_cache with optional tagged debug output."""
    mem_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_MEM")
    stats_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_STATS")
    snapshot_debug = _env_enabled("WANVIDEO_DEBUG_CUDA_SNAPSHOT")

    before_mem = _cuda_mem_snapshot() if mem_debug else None
    before_stats = snapshot_cuda_stats() if stats_debug else None
    before_snapshot = snapshot_cuda_snapshot_summary() if snapshot_debug else None

    _cleanup_log(tag or "sampler_cleanup", "gc.collect start")
    try:
        gc.collect()
    except Exception as exc:
        _cleanup_log(tag or "sampler_cleanup", f"gc.collect error: {exc}")
    _cleanup_log(tag or "sampler_cleanup", "gc.collect done")

    _cleanup_log(tag or "sampler_cleanup", "soft_empty_cache start")
    mm.soft_empty_cache()
    _cleanup_log(tag or "sampler_cleanup", "soft_empty_cache done")

    after_mem = _cuda_mem_snapshot() if mem_debug else None
    after_stats = snapshot_cuda_stats() if stats_debug else None
    after_snapshot = snapshot_cuda_snapshot_summary() if snapshot_debug else None

    if not tag:
        tag = "sampler_cleanup"
    if mem_debug:
        report_cuda_mem_delta(before_mem, after_mem, tag=tag)
    if stats_debug:
        report_cuda_stats_delta(before_stats, after_stats, tag=tag)
    if snapshot_debug:
        report_cuda_snapshot_delta(before_snapshot, after_snapshot, tag=tag)
    return True


if (
    _env_enabled("WANVIDEO_DEBUG_NODECACHE")
    or _env_enabled("WANVIDEO_DEBUG_CUDA_MEM")
    or _env_enabled("WANVIDEO_DEBUG_CUDA_CENSUS")
    or _env_enabled("WANVIDEO_DEBUG_CUDA_STATS")
    or _env_enabled("WANVIDEO_DEBUG_CUDA_SNAPSHOT")
):
    _exec_mod = _get_execution_module()
    if _exec_mod is not None:
        _ensure_execution_hook(_exec_mod)
        _ensure_prompt_executor_reset_hook(_exec_mod)
    _ensure_soft_empty_cache_hook()

def get_module_memory_mb(module):
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            memory += param.nelement() * param.element_size()
    return memory / (1024 * 1024)  # Convert to MB

def get_module_memory_mb_per_device(module):
    memory_per_device = {}
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            device = str(param.device)
            memory += param.nelement() * param.element_size()
            memory_per_device[device] = memory_per_device.get(device, 0) + memory

    memory_per_device = {dev: mem / (1024 * 1024) for dev, mem in memory_per_device.items()}
    return memory_per_device

def get_tensor_memory(tensor):
    memory_bytes = tensor.element_size() * tensor.nelement()
    return f"{memory_bytes / (1024 * 1024):.2f} MB"

def patch_weight_to_device(self, key, device_to=None, inplace_update=False, backup_keys=False, scale_weight=None):
    if key not in self.patches:
        return

    weight, set_func, convert_func = get_key_weight(self.model, key)
    inplace_update = self.weight_inplace_update or inplace_update

    if backup_keys and key not in self.backup:
        self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

    if device_to is not None:
        temp_weight = mm.cast_to_device(weight, device_to, torch.float32, copy=True)
    else:
        temp_weight = weight.to(torch.float32, copy=True)
    if convert_func is not None:
        temp_weight = convert_func(temp_weight, inplace=True)

    if scale_weight is not None:
        temp_weight = temp_weight * scale_weight.to(temp_weight.device, temp_weight.dtype)

    out_weight = calculate_weight(self.patches[key], temp_weight, key)

    if set_func is None:
        out_weight = stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
        if inplace_update:
            copy_to_param(self.model, key, out_weight)
        else:
            set_attr_param(self.model, key, out_weight)
    else:
        set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

def apply_lora(model, device_to, transformer_load_device, params_to_keep=None, dtype=None, 
               base_dtype=None, state_dict=None, low_mem_load=False, control_lora=False, scale_weights={}):
        model.patch_weight_to_device = types.MethodType(patch_weight_to_device, model)
        to_load = []
        for n, m in model.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                to_load.append((n, m, params))

        to_load.sort(reverse=True)
        cnt = 0
        pbar = ProgressBar(len(to_load))
        for x in tqdm(to_load, desc="Loading model and applying LoRA weights:", leave=True):
            name = x[0]
            m = x[1]
            params = x[2]
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights == True:
                    continue
            for param in params:
                name = name.replace("._orig_mod.", ".") # torch compiled modules have this prefix
                if low_mem_load:
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    key = f"{name.replace('diffusion_model.', '')}.{param}"
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[key])
                    except:
                        continue
                key = f"{name}.{param}"
                if scale_weights is not None:
                    scale_key = key.replace("weight", "scale_weight").replace("diffusion_model.", "") if "weight" in key else None
                if low_mem_load:
                    model.patch_weight_to_device(f"{name}.{param}", device_to=device_to, inplace_update=True, backup_keys=control_lora, scale_weight=scale_weights.get(scale_key, None))
                else:
                    model.patch_weight_to_device(f"{name}.{param}", device_to=device_to, backup_keys=control_lora, scale_weight=scale_weights.get(scale_key, None))
                    if device_to != transformer_load_device:
                        set_module_tensor_to_device(m, param, device=transformer_load_device)
                if low_mem_load:
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=model.model.diffusion_model.state_dict()[key])
                    except:
                        continue
            m.comfy_patched_weights = True
            cnt += 1
            if cnt % 100 == 0:
                pbar.update(100)


        # After LoRA patching, scale weights that have scale_weight but are NOT LoRA patched
        if len(scale_weights) > 0 and not getattr(model, "scale_weights_applied", False):
            for name, param in model.model.diffusion_model.named_parameters():
                scale_key = name.replace("weight", "scale_weight").replace("diffusion_model.", "") if "weight" in name else None
                full_param_name = f"diffusion_model.{name}"
                if scale_key and scale_key in scale_weights and full_param_name not in model.patches:
                    scale = scale_weights[scale_key]
                    param_fp32 = param.to(torch.float32)
                    param_fp32.mul_(scale.to(param.device, torch.float32))
                    param.copy_(param_fp32.to(param.dtype))
            model.scale_weights_applied = True

        model.current_weight_patches_uuid = model.patches_uuid
        if low_mem_load:
            for name, param in model.model.diffusion_model.named_parameters():
                if param.device != transformer_load_device:
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, name, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[name])
                    except:
                        continue
        return model


# from https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/9d076a3df0d2763cef5510ec5ab807f6632c39f5/utils.py#L181
def split_tiles(embeds, num_split):
    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)
        out.append(x_split)

    x_split = torch.stack(out, dim=0)

    return x_split

def merge_hiddenstates(x, tiles):
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1]-1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1)
                            for i in range(grid_size)], dim=0)

        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]

        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
        flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])

        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)

    out = torch.cat(out, dim=0)

    return out

from comfy.clip_vision import clip_preprocess, ClipVisionModel

def clip_encode_image_tiled(clip_vision, image, tiles=1, ratio=1.0):
    embeds = encode_image_(clip_vision, image)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = {}
        for i in image_split:
            encoded = encode_image_(clip_vision, i)
            if not hasattr(embeds_split, "last_hidden_state"):
                embeds_split["last_hidden_state"] = encoded
            else:
                embeds_split["last_hidden_state"] = torch.cat(embeds_split["last_hidden_state"], encoded, dim=0)

        embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'], tiles)

        if embeds.shape[0] > 1: # if we have more than one image we need to average the embeddings for consistency
            embeds = embeds * ratio + embeds_split['last_hidden_state']*(1-ratio)
        else: # otherwise we can concatenate them, they can be averaged later
            embeds = torch.cat([embeds * ratio, embeds_split['last_hidden_state']])

    return embeds

def encode_image_(clip_vision, image):
    if isinstance(clip_vision, ClipVisionModel):
        out = clip_vision.encode_image(image).last_hidden_state
    else:
        pixel_values = clip_preprocess(image, size=224, crop=True).float()
        out = clip_vision.visual(pixel_values)

    return out

# Code based on https://github.com/WikiChao/FreSca (MIT License)
import torch
import torch.fft as fft

def fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    """
    Apply frequency-dependent scaling to an image tensor using Fourier transforms.

    Parameters:
        x:           Input tensor of shape (B, C, H, W)
        scale_low:   Scaling factor for low-frequency components (default: 1.0)
        scale_high:  Scaling factor for high-frequency components (default: 1.5)
        freq_cutoff: Number of frequency indices around center to consider as low-frequency (default: 20)

    Returns:
        x_filtered: Filtered version of x in spatial domain with frequency-specific scaling applied.
    """
    # Preserve input dtype and device
    dtype, device = x.dtype, x.device

    # Convert to float32 for FFT computations
    x = x.to(torch.float32)

    # 1) Apply FFT and shift low frequencies to center
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # 2) Create a mask to scale frequencies differently
    C, B, H, W = x_freq.shape
    crow, ccol = H // 2, W // 2

    # Initialize mask with high-frequency scaling factor
    mask = torch.ones((C, B, H, W), device=device) * scale_high

    # Apply low-frequency scaling factor to center region
    mask[
        ...,
        crow - freq_cutoff : crow + freq_cutoff,
        ccol - freq_cutoff : ccol + freq_cutoff,
    ] = scale_low

    # 3) Apply frequency-specific scaling
    x_freq = x_freq * mask

    # 4) Convert back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    # 5) Restore original dtype
    x_filtered = x_filtered.to(dtype)

    return x_filtered

def is_image_black(image, threshold=1e-3):
    if image.min() < 0:
        image = (image + 1) / 2
    return torch.all(image < threshold).item()

def add_noise_to_reference_video(image, ratio=None):
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image

def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star

def find_closest_valid_dim(fixed_dim, var_dim, block_size):
    for delta in range(1, 17):
        for sign in [-1, 1]:
            candidate = var_dim + sign * delta
            if candidate > 0 and ((fixed_dim * candidate) // 4) % block_size == 0:
                return candidate
    return var_dim

 # Radial attention setup
def setup_radial_attention(transformer, transformer_options, latent, seq_len, latent_video_length, context_options=None):
    if context_options is not None:
        context_frames =  (context_options["context_frames"] - 1) // 4 + 1

    dense_timesteps = transformer_options.get("dense_timesteps", 1)
    dense_blocks = transformer_options.get("dense_blocks", 1)
    dense_vace_blocks = transformer_options.get("dense_vace_blocks", 1)
    decay_factor = transformer_options.get("decay_factor", 0.2)
    dense_attention_mode = transformer_options.get("dense_attention_mode", "sageattn")
    block_size = transformer_options.get("block_size", 128)

    # Calculate closest valid latent sizes
    if latent.shape[2] % (block_size/8) != 0 or latent.shape[3] % (block_size/8) != 0:
        block_div = int(block_size // 8)
        closest_h = round(latent.shape[2] / block_div) * block_div
        closest_w = round(latent.shape[3] / block_div) * block_div
        raise Exception(
            f"Radial attention mode only supports image size divisible by block size. "
            f"Got {latent.shape[3] * 8}x{latent.shape[2] * 8} with block size {block_size}.\n"
            f"Closest valid sizes: {closest_w * 8}x{closest_h * 8} (width x height in pixels)."
        )
    tokens_per_frame = (latent.shape[2] * latent.shape[3]) // 4
    if tokens_per_frame % block_size != 0:
        closest_latent_h = find_closest_valid_dim(latent.shape[3], latent.shape[2], block_size)
        closest_latent_w = find_closest_valid_dim(latent.shape[2], latent.shape[3], block_size)
        raise Exception(
            f"Radial attention mode requires tokens per frame ((latent_h * latent_w) // 4) to be divisible by block size ({block_size}).\n"
            f"Current size in latent space:{latent.shape[3]}x{latent.shape[2]}, pixel space: {latent.shape[3]*8}x{latent.shape[2]*8} tokens_per_frame={tokens_per_frame}.\n"
            f"Try adjusting to one of these latent sizes (in pixels):\n"
            f"  Height: {latent.shape[2]*8} -> {closest_latent_h * 8}\n"
            f"  Width: {latent.shape[3]*8} -> {closest_latent_w * 8}\n"
            f"Or choose another resolution so that (latent_h * latent_w) // 4 is divisible by {block_size}."
        )

    from .wanvideo.radial_attention.attn_mask import MaskMap
    for i, block in enumerate(transformer.blocks):
        block.self_attn.mask_map = block.dense_attention_mode = block.dense_timesteps = block.self_attn.decay_factor = None
        if isinstance(dense_blocks, list):
            block.dense_block = i in dense_blocks
        else:
            block.dense_block = i < dense_blocks
        block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=latent_video_length if context_options is None else context_frames, block_size=block_size)
        block.dense_attention_mode = dense_attention_mode
        block.dense_timesteps = dense_timesteps
        block.self_attn.decay_factor = decay_factor
    if transformer.vace_layers is not None:
        for i, block in enumerate(transformer.vace_blocks):
            block.self_attn.mask_map = block.dense_attention_mode = block.dense_timesteps = block.self_attn.decay_factor = None
            if isinstance(dense_vace_blocks, list):
                block.dense_block = i in dense_vace_blocks
            else:
                block.dense_block = i < dense_vace_blocks
            block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=latent_video_length if context_options is None else context_frames, block_size=block_size)
            block.dense_attention_mode = dense_attention_mode
            block.dense_timesteps = dense_timesteps
            block.self_attn.decay_factor = decay_factor

    log.info(f"Radial attention mode enabled.")
    log.info(f"dense_attention_mode: {dense_attention_mode}, dense_timesteps: {dense_timesteps}, decay_factor: {decay_factor}")
    log.info(f"dense_blocks: {[i for i, block in enumerate(transformer.blocks) if getattr(block, 'dense_block', False)]})")



def list_to_device(tensor_list, device, dtype=None):
    """
    Move all tensors in a list to the specified device and optionally cast to dtype.
    """
    return [t.to(device, dtype=dtype) if dtype is not None else t.to(device) for t in tensor_list]

def dict_to_device(tensor_dict, device, dtype=None):
    """
    Move all tensors (and tensor lists) in a dict to the specified device and optionally cast to dtype.
    Supports values that are tensors or lists of tensors.
    """
    result = {}
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, dtype=dtype) if dtype is not None else v.to(device)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            result[k] = list_to_device(v, device, dtype)
        else:
            result[k] = v
    return result

def compile_model(transformer, compile_args=None):
    if compile_args is None:
        return transformer
    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
        torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
        torch._dynamo.config.force_parameter_static_shapes = compile_args["force_parameter_static_shapes"]
        try:
            if hasattr(torch._dynamo.config, 'allow_unspec_int_on_nn_module'):
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
        except Exception as e:
            log.warning(f"Could not set allow_unspec_int_on_nn_module: {e}")
        try:
            torch._dynamo.config.recompile_limit = compile_args["dynamo_recompile_limit"]
        except Exception as e:
            log.warning(f"Could not set recompile_limit: {e}")

    if compile_args["compile_transformer_blocks_only"]:
        for i, block in enumerate(transformer.blocks):
            if hasattr(block, "_orig_mod"):
                block = block._orig_mod
            transformer.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
        if transformer.vace_layers is not None:
            for i, block in enumerate(transformer.vace_blocks):
                if hasattr(block, "_orig_mod"):
                    block = block._orig_mod
                transformer.vace_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
    else:
        transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
    return transformer

#https://5410tiffany.github.io/tcfg.github.io/
def tangential_projection(pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> torch.Tensor:
    cond_dtype = pred_cond.dtype
    preds = torch.stack([pred_cond, pred_uncond], dim=1).float()
    orig_shape = preds.shape[2:]
    preds_flat = preds.flatten(2)
    U, S, Vh = torch.linalg.svd(preds_flat, full_matrices=False)
    Vh_modified = Vh.clone()
    Vh_modified[:, 1] = 0
    recon = U @ torch.diag_embed(S) @ Vh_modified
    return recon[:, 1].view(pred_uncond.shape).to(cond_dtype)

#https://arxiv.org/abs/2508.03442
def get_raag_guidance(noise_pred_cond, noise_pred_uncond, w_max, alpha=1.0, eps=1e-8):
    delta = noise_pred_cond - noise_pred_uncond
    norm_delta = torch.norm(delta.flatten(1), dim=1, keepdim=True)
    norm_uncond = torch.norm(noise_pred_uncond.flatten(1), dim=1, keepdim=True)
    ratio = norm_delta / (norm_uncond + eps)
    ratio_mean = ratio.mean().item()
    adaptive_w = 1.0 + (w_max - 1.0) * math.exp(-alpha * ratio_mean)
    return adaptive_w

def tensor_pingpong_pad(video, target_len):
    """
    Pads a video tensor along the frame dimension (dim=2) in a ping-pong fashion.
    video: torch.Tensor of shape [B, C, F, H, W]
    target_len: desired number of frames
    Returns: padded tensor of shape [B, C, target_len, H, W]
    """
    in_dims = len(video.shape)
    if in_dims == 4:
        video = video.unsqueeze(0)
    B, C, F, H, W = video.shape
    idx = 0
    flip = False
    indices = []
    while len(indices) < target_len:
        indices.append(idx)
        if flip:
            idx -= 1
        else:
            idx += 1
        if idx == 0 or idx == F - 1:
            flip = not flip
    indices = indices[:target_len]
    padded_video = video[:, :, indices, :, :]
    if in_dims == 4:
        padded_video = padded_video.squeeze(0)
    return padded_video


def check_duplicate_nodes():
    """Check ComfyUI custom_nodes directory for duplicate installations"""
    custom_nodes_dir = Path(folder_paths.folder_names_and_paths["custom_nodes"][0][0])
    current_path = Path(__file__).parent

    wanvideo_dirs = []

    # Check all directories in custom_nodes
    for path in custom_nodes_dir.iterdir():
        if (path.is_dir() and 
            path != current_path and
            'wanvideo' in path.name.lower() and
            'wrapper' in path.name.lower()):
            wanvideo_dirs.append(str(path))

    return wanvideo_dirs

#https://github.com/temporalscorerescaling/TSR/
def temporal_score_rescaling(model_output, sample, timestep, k=1.0, tsr_sigma=0.1):
    t = (timestep / 1000)
    if t == 0.0:
        ratio = k
    else:
        snr_t = (1 - t)**2 / t**2
        ratio = (snr_t * tsr_sigma**2 + 1) / (snr_t * tsr_sigma**2 / k + 1)

    if not t == 1.0:
        model_output = (ratio * ((1-t) * model_output + sample) - sample) / (1 - t)
    return model_output
