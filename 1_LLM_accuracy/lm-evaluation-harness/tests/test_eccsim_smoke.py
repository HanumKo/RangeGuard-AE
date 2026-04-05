import torch

from lm_eval.models.eccsim.logger import GPUChangeLogger
from lm_eval.models.eccsim.patterns import sample_bits_dist
from lm_eval.models.eccsim.schemes import get_scheme
from lm_eval.models.eccsim.schemes.none import ECCNone
from lm_eval.models.eccsim.schemes.rangeguard_common import (
    log_corrected_uncorrectable,
    merge_rep_with_corrupted_sign,
)


def test_patterns_allow_multi_emits_block_counts():
    torch.manual_seed(0)
    out = sample_bits_dist(
        num_blocks=128,
        per_bit_prob_10neg=2,
        device="cpu",
        allow_multi=True,
        dist={"SE": 1.0},
    )
    assert "__blk_event_counts__" in out
    counts = out["__blk_event_counts__"]
    assert counts.shape[0] == 128


def test_scheme_registry_known_and_fallback():
    known = get_scheme("RANGEGUARD_BF16_SSC_FAST", device="cpu")
    assert known is not None

    unknown = get_scheme("SOME_UNKNOWN_SCHEME", device="cpu")
    assert isinstance(unknown, ECCNone)


def test_merge_rep_with_corrupted_sign_keeps_sign():
    rep = torch.tensor([0x1234, 0x7FFF], dtype=torch.int16)
    cor = torch.tensor([0x9234, 0x8001], dtype=torch.int32).to(torch.int16)

    merged = merge_rep_with_corrupted_sign(rep, cor).to(torch.int32) & 0xFFFF
    assert int(merged[0]) == 0x9234
    assert int(merged[1]) == 0xFFFF


def test_logger_flush_collects_records():
    logger = GPUChangeLogger("cpu", max_weight=10, max_activation=10)
    slot_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    before = torch.tensor([1, 2, 3], dtype=torch.int32)
    after = torch.tensor([4, 5, 6], dtype=torch.int32)

    logger.log("weight:test", slot_idx, before, after)
    out = logger.flush()
    assert len(out["where"]) == 3
    assert out["slot_idx"].numel() == 3


def test_log_corrected_uncorrectable_helper_runs():
    logger = GPUChangeLogger("cpu", max_weight=10, max_activation=10)
    after_u16 = torch.zeros((2, 16), dtype=torch.int16)
    slot_idx = torch.tensor([0, 1, 16, 17], dtype=torch.long)
    block_indices = torch.tensor([0, 1], dtype=torch.long)
    corrected_mask = torch.tensor([True, False], dtype=torch.bool)
    uncorrectable_mask = torch.tensor([False, True], dtype=torch.bool)
    orig = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

    log_corrected_uncorrectable(
        logger=logger,
        where="weight:test",
        slot_idx=slot_idx,
        block_indices=block_indices,
        corrected_mask=corrected_mask,
        uncorrectable_mask=uncorrectable_mask,
        orig_vals_i32=orig,
        after_u16=after_u16,
    )
    out = logger.flush()
    assert len(out["where"]) == 4
