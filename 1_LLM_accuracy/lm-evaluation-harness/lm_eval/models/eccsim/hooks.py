import os
from collections.abc import Mapping

import torch

from .logger import GPUChangeLogger
from .patterns import sample_bits_dist
from .schemes import get_scheme
from .utils import (
    BFPER_BLOCK,
    as_uint8_view,
    bf16_u16_views,
    blocks_32B,
    changed_slots_from_bit_indices,
    xor_inplace,
)


class ECCSimState:
    def __init__(
        self,
        prob_10neg: int,
        ecc_enable: bool,
        ecc_scheme: str,
        device,
        *,
        max_weight_logs: int = 1000,
        max_activation_logs: int = 1000,
        max_blocks_per_pass: int | None = None,
        max_pass_megabytes: int | None = 2048,
    ):
        self.prob = int(prob_10neg)
        self.ecc_enable = bool(ecc_enable)
        logger_enabled = os.getenv("ECCSIM_LOGGER", "0") == "1"
        if logger_enabled:
            self.logger = GPUChangeLogger(
                device,
                max_weight=max_weight_logs,
                max_activation=max_activation_logs,
            )
        else:
            self.logger = None

        self.weight_done = False
        self.device = device

        env_blocks = os.getenv("ECCSIM_MAX_BLOCKS_PER_PASS")
        env_mb = os.getenv("ECCSIM_MAX_PASS_MB")

        if max_blocks_per_pass is not None:
            self.max_blocks_per_pass = int(max_blocks_per_pass)
        elif env_blocks is not None:
            self.max_blocks_per_pass = max(1, int(env_blocks))
        else:
            mb = max_pass_megabytes if max_pass_megabytes is not None else 16
            if env_mb is not None:
                try:
                    mb = int(env_mb)
                except ValueError:
                    pass
            self.max_blocks_per_pass = max(1, int((mb * (1 << 20)) // 32))

        self.scheme = get_scheme(ecc_scheme, device)


def _module_path(model, module):
    if not hasattr(model, "_eccsim_named"):
        model._eccsim_named = {m: n for n, m in model.named_modules()}
    return model._eccsim_named.get(module, "")


def _param_path(model, param):
    if not hasattr(model, "_eccsim_named_params"):
        model._eccsim_named_params = {p: n for n, p in model.named_parameters()}
    return model._eccsim_named_params.get(param, "")


def _sample_bits_dist_compat(num_blocks: int, state: ECCSimState, device):
    try:
        return sample_bits_dist(
            num_blocks,
            state.prob,
            device=device,
            allow_multi=True,
            max_events_per_block=None,
            no_overlap_in_block=False,
        )
    except TypeError:
        try:
            return sample_bits_dist(
                num_blocks,
                state.prob,
                device=device,
                allow_multi=True,
            )
        except TypeError:
            # Backward-compat path when allow_multi is unsupported.
            return sample_bits_dist(num_blocks, state.prob, device=device)


def _process_blocks_in_chunks(blocks_u8: torch.Tensor, state: ECCSimState, where: str):
    num_blocks = blocks_u8.shape[0]
    if num_blocks == 0:
        return

    step = max(1, state.max_blocks_per_pass)
    scheme = state.scheme

    for start in range(0, num_blocks, step):
        end = min(num_blocks, start + step)
        blk = blocks_u8[start:end]

        by_pat = _sample_bits_dist_compat(blk.shape[0], state, blk.device)
        if not by_pat:
            continue

        blk_counts = by_pat.pop("__blk_event_counts__", None)

        u16_pre = bf16_u16_views(blk)
        flat_pre = u16_pre.reshape(-1)

        per = {}
        total_bit_indices = []

        for pat, bit_indices in by_pat.items():
            if bit_indices.numel() == 0:
                continue
            if not scheme.wants_injection(pat):
                continue

            local_idx = changed_slots_from_bit_indices(bit_indices)
            if local_idx.numel() == 0:
                continue
            local_idx = torch.unique(local_idx, sorted=True)
            if local_idx.numel() == 0:
                continue

            per[pat] = {
                "bit_indices": bit_indices,
                "local_idx": local_idx,
                "orig_vals_i32": flat_pre.index_select(0, local_idx).to(torch.int32),
            }
            total_bit_indices.append(bit_indices)

        if not per:
            continue

        all_indices = torch.cat(total_bit_indices, dim=0)
        xor_inplace(blk, all_indices)

        au16 = bf16_u16_views(blk)
        flat_post = au16.reshape(-1)

        for pat, rec in per.items():
            l_idx = rec["local_idx"]
            o_vals = rec["orig_vals_i32"]

            if blk_counts is not None:
                block_ids = l_idx // BFPER_BLOCK
                counts_per_slot = blk_counts.index_select(0, block_ids)
                mask_multi = counts_per_slot > 1
                mask_single = ~mask_multi
            else:
                mask_single = torch.ones_like(l_idx, dtype=torch.bool)
                mask_multi = torch.zeros_like(l_idx, dtype=torch.bool)

            def _dispatch(mask: torch.Tensor, tag_prefix: str):
                if not mask.any():
                    return

                sub_idx = l_idx[mask]
                sub_orig = o_vals[mask]
                current_where = f"{where}|{tag_prefix}={pat}"

                if state.ecc_enable:
                    scheme(
                        pattern=pat,
                        after_u16=au16,
                        slot_idx=sub_idx,
                        logger=state.logger,
                        where=current_where,
                        orig_vals_i32=sub_orig,
                        blk_event_counts=blk_counts,
                    )
                    return

                if state.logger is not None:
                    after_vals = flat_post.index_select(0, sub_idx).to(torch.int32)
                    state.logger.log(current_where, sub_idx, sub_orig, after_vals)

            _dispatch(mask_single, "pat")
            _dispatch(mask_multi, "combo")


def _process_parameter_tensor(t: torch.Tensor, state: ECCSimState, where: str = ""):
    if not (t.is_floating_point() or t.dtype in (torch.bfloat16, torch.float16)):
        return

    target = t if t.is_contiguous() else t.contiguous()
    u8 = as_uint8_view(target.data)
    blocks, _ = blocks_32B(u8)

    if blocks.shape[0] == 0:
        if target is not t:
            t.copy_(target)
        return

    _process_blocks_in_chunks(blocks, state, where=where)
    if target is not t:
        t.copy_(target)


def attach_eccsim_hooks(
    model: torch.nn.Module,
    *,
    error_prob: int,
    ecc_enable: bool,
    ecc_scheme: str,
):
    device = next(model.parameters()).device
    state = ECCSimState(error_prob, ecc_enable, ecc_scheme, device=device)

    def pre_hook(mod, __):
        if state.weight_done:
            return None

        for p in mod.parameters(recurse=True):
            pname = _param_path(model, p)
            where = (
                f"weight:{pname}"
                if pname
                else f"weight:{_module_path(model, mod) or '(root)'}"
            )
            _process_parameter_tensor(p, state, where=where)
        return None

    def post_hook_finalize(_, __, ___):
        if not state.weight_done:
            state.weight_done = True
        return None

    def act_hook(mod, __, output):
        layer_path = _module_path(model, mod)

        def _handle_out(value):
            if torch.is_tensor(value):
                _process_parameter_tensor(value, state, where=f"activation:{layer_path}")

        if torch.is_tensor(output):
            _handle_out(output)
        elif isinstance(output, (list, tuple)):
            for value in output:
                _handle_out(value)
        elif isinstance(output, Mapping):
            for value in output.values():
                _handle_out(value)
        elif hasattr(output, "to_tuple"):
            for value in output.to_tuple():
                _handle_out(value)
        return None

    model._eccsim_state = state
    handles = []
    handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=False))
    handles.append(model.register_forward_hook(post_hook_finalize, with_kwargs=False))
    for mod in model.modules():
        handles.append(mod.register_forward_hook(act_hook, with_kwargs=False))
    model._eccsim_handles = handles
    return state


def detach_eccsim_hooks(model: torch.nn.Module):
    handles = getattr(model, "_eccsim_handles", [])
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass

    if hasattr(model, "_eccsim_state"):
        delattr(model, "_eccsim_state")
    if hasattr(model, "_eccsim_handles"):
        delattr(model, "_eccsim_handles")
