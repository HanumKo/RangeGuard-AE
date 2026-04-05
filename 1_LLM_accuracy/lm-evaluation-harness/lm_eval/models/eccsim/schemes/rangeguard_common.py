import torch

SLOTS_PER_BLOCK = 16


def merge_rep_with_corrupted_sign(
    rep_vals: torch.Tensor,
    corrupted_vals: torch.Tensor,
) -> torch.Tensor:
    """Keep sign from corrupted values while replacing magnitude with representative values."""
    rep_i32 = rep_vals.to(torch.int32)
    cor_i32 = corrupted_vals.to(torch.int32)
    sign_bits = cor_i32 & 0x8000
    mag_bits = rep_i32 & 0x7FFF
    return (sign_bits | mag_bits).to(torch.int16)


def _log_for_blocks(
    logger,
    where: str,
    suffix: str,
    slot_idx: torch.Tensor,
    block_ids: torch.Tensor,
    orig_vals_i32: torch.Tensor,
    after_u16: torch.Tensor,
):
    if block_ids.numel() == 0:
        return

    mask = torch.isin(slot_idx // SLOTS_PER_BLOCK, block_ids)
    if not mask.any():
        return

    idx = slot_idx[mask]
    val_orig = orig_vals_i32[mask]
    val_final = after_u16.view(-1).index_select(0, idx).to(torch.int32)
    logger.log(f"{where}|{suffix}", idx, val_orig, val_final)


def log_corrected_uncorrectable(
    *,
    logger,
    where: str,
    slot_idx: torch.Tensor,
    block_indices: torch.Tensor,
    corrected_mask: torch.Tensor | None,
    uncorrectable_mask: torch.Tensor | None,
    orig_vals_i32: torch.Tensor,
    after_u16: torch.Tensor,
):
    """Log corrected/uncorrectable slot values by block masks over block_indices."""
    if logger is None:
        return

    if corrected_mask is not None and corrected_mask.any():
        _log_for_blocks(
            logger=logger,
            where=where,
            suffix="corrected",
            slot_idx=slot_idx,
            block_ids=block_indices[corrected_mask],
            orig_vals_i32=orig_vals_i32,
            after_u16=after_u16,
        )

    if uncorrectable_mask is not None and uncorrectable_mask.any():
        _log_for_blocks(
            logger=logger,
            where=where,
            suffix="uncorrectable",
            slot_idx=slot_idx,
            block_ids=block_indices[uncorrectable_mask],
            orig_vals_i32=orig_vals_i32,
            after_u16=after_u16,
        )
