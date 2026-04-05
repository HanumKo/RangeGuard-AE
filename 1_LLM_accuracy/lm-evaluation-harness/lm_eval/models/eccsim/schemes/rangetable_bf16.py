# rangetable_bf16.py
from __future__ import annotations

import json
import os
from typing import Any

import torch

# Default behavior matches the newly uploaded exponent JSON.
# You can override these with environment variables if needed:
#   export RANGEGUARD_JSON_FILENAME=rangeguard_mapping.json
#   export RANGEGUARD_TARGET_SIGMA=4.0
TARGET_SIGMA = float(os.environ.get("RANGEGUARD_TARGET_SIGMA", "4.0"))
JSON_FILENAME = os.environ.get("RANGEGUARD_JSON_FILENAME", "rangeguard_mapping.json")

_CACHE: dict[tuple[int, float, str], dict[str, torch.Tensor | None]] = {}


def _normalize_sigma(value: Any) -> float:
    return float(value)


def _load_json(json_path: str) -> dict[str, Any]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_target_config(root: dict[str, Any], n_bins: int, target_sigma: float) -> dict[str, Any]:
    # New exponent JSON format
    if "results" in root:
        target = next(
            (
                entry
                for entry in root["results"]
                if int(entry["N"]) == int(n_bins)
                and _normalize_sigma(entry["Sigma"]) == _normalize_sigma(target_sigma)
            ),
            None,
        )
        if target is None:
            available = sorted(
                {(int(entry["N"]), _normalize_sigma(entry["Sigma"])) for entry in root["results"]}
            )
            raise ValueError(
                f"No exponent config for N={n_bins}, Sigma={target_sigma} in {JSON_FILENAME}. "
                f"Available (N, Sigma): {available}"
            )
        return target

    # Backward-compatible old format support if needed later.
    if "configs" in root:
        target = next(
            (
                entry
                for entry in root["configs"]
                if int(entry["N"]) == int(n_bins)
                and _normalize_sigma(entry["Sigma"]) == _normalize_sigma(target_sigma)
            ),
            None,
        )
        if target is None:
            raise ValueError(f"No config for N={n_bins}, Sigma={target_sigma}")
        return target

    raise ValueError(
        f"Unsupported JSON schema in {JSON_FILENAME}. Expected either 'results' or 'configs'."
    )


def _build_tables_exponent(boundaries: list[int], reps_list: list[int], n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(boundaries) != n_bins - 1:
        raise ValueError(
            f"Boundary length mismatch: expected {n_bins - 1}, got {len(boundaries)}"
        )
    if len(reps_list) != n_bins:
        raise ValueError(
            f"Representative length mismatch: expected {n_bins}, got {len(reps_list)}"
        )

    # LUT over all BF16 bit patterns.
    u16_all = torch.arange(0, 1 << 16, dtype=torch.int32)

    # BF16 exponent field: bits [14:7]
    exponents = (u16_all >> 7) & 0xFF

    # The JSON boundaries are the start indices of later bins.
    # Example: [126, 127, 128] => [0..125], [126], [127], [128..255]
    bounds_t = torch.tensor(boundaries, dtype=torch.int32)
    lut = torch.bucketize(exponents, bounds_t, right=True).to(torch.int32)

    # Representative exponent -> representative BF16 magnitude.
    # This follows the same proxy used in the exponent DP generator: 2^(E-127).
    reps_exp = torch.tensor(reps_list, dtype=torch.float32)
    rep_vals = torch.pow(2.0, reps_exp - 127.0)
    reps_tensor = rep_vals.to(torch.bfloat16).view(torch.int16).cpu()

    return lut.cpu(), reps_tensor


def _load_and_build_tables(n_bins: int, target_sigma: float = TARGET_SIGMA):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, JSON_FILENAME)
    root = _load_json(json_path)
    target = _find_target_config(root, n_bins, target_sigma)
    boundaries = [int(x) for x in target["Boundaries"]]
    reps_list = [int(x) for x in target["Representatives"]]
    return _build_tables_exponent(boundaries, reps_list, n_bins)


def _get_global_tables(n_bins: int, target_sigma: float = TARGET_SIGMA):
    cache_key = (int(n_bins), float(target_sigma), JSON_FILENAME)
    if cache_key not in _CACHE:
        _CACHE[cache_key] = {"lut": None, "reps": None}

    if _CACHE[cache_key]["lut"] is None:
        lut, reps = _load_and_build_tables(n_bins, target_sigma=target_sigma)
        _CACHE[cache_key]["lut"] = lut
        _CACHE[cache_key]["reps"] = reps

    return _CACHE[cache_key]["lut"], _CACHE[cache_key]["reps"]


class BCAMapper:
    def __init__(self, device, num_bins, target_sigma: float = TARGET_SIGMA):
        self.device = device
        self.num_bins = int(num_bins)
        self.target_sigma = float(target_sigma)

        lut_cpu, reps_cpu = _get_global_tables(self.num_bins, self.target_sigma)
        self.lut = lut_cpu.to(device)
        # keep int16 BF16 bit-patterns as-is
        self.reps = reps_cpu.to(device)

    def value_to_rid(self, u16_data: torch.Tensor) -> torch.Tensor:
        if u16_data.dtype == torch.bfloat16:
            u16_data = u16_data.view(torch.int16)
        idx = u16_data.to(torch.int32) & 0xFFFF
        return self.lut.index_select(0, idx)

    def rid_to_value(self, rids: torch.Tensor) -> torch.Tensor:
        # Returns BF16 bit-patterns stored as int16.
        return self.reps.index_select(0, rids.long())

    def rid_to_value_with_sign(self, rids: torch.Tensor, sign_source_u16: torch.Tensor) -> torch.Tensor:
        """
        Optional helper if you later decide to preserve the sign bit separately.
        sign_source_u16: tensor containing the sign bit source (original or corrupted BF16 bit patterns).
        Returns int16 BF16 bit-patterns.
        """
        mag = self.rid_to_value(rids).to(torch.int32) & 0x7FFF
        sign = sign_source_u16.to(torch.int32) & 0x8000
        return (mag | sign).to(torch.int16)
