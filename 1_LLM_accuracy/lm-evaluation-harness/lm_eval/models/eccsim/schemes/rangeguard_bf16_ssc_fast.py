import torch
from .base import ECCBase
from .rangeguard_common import log_corrected_uncorrectable, merge_rep_with_corrupted_sign
from .rangetable_bf16 import BCAMapper

class ECCRangeGuardSSCFast(ECCBase):
    """
    RangeGuard BF16 SSC (Fast Oracle Mode).
    - 4-bit RID (16 Bins)
    - RS(10, 8) Simulation: Corrects up to 1 symbol error.
    - Fast execution using ground-truth comparison.
    """
    def __init__(self, device):
        super().__init__(device)
        self.mapper = BCAMapper(device, num_bins=16)
        self.device = device
        self.ecc_t = 1  # Max correctable symbols (RS 10,8)

    def wants_injection(self, pattern: str) -> bool: return True
    def max_events_per_block(self) -> int: return 9999

    def _pack_symbols(self, rids):
        # 2 RIDs (4-bit) -> 1 Symbol (8-bit)
        pairs = rids.view(rids.shape[0], 8, 2)
        return (pairs[:, :, 0] << 4) | pairs[:, :, 1]

    def __call__(self, *, pattern: str, after_u16: torch.Tensor, slot_idx: torch.Tensor, 
                 logger=None, where: str="", orig_vals_i32: torch.Tensor | None = None, 
                 blk_event_counts: torch.Tensor | None = None):
        
        if slot_idx.numel() == 0 or orig_vals_i32 is None: return after_u16

        # 1. 데이터 준비
        total_blocks = after_u16.numel() // 16
        flat_data = after_u16.view(-1).view(total_blocks, 16)
        block_indices = torch.unique(slot_idx // 16)
        
        corrupted_vals = flat_data.index_select(0, block_indices).clone()
        orig_vals_block = corrupted_vals.clone()
        
        local_blk_idx = torch.bucketize(slot_idx // 16, block_indices)
        local_slot_off = slot_idx % 16
        
        orig_flat = orig_vals_block.view(-1)
        target_idx = local_blk_idx * 16 + local_slot_off
        orig_flat.index_copy_(0, target_idx, orig_vals_i32.to(torch.int16))

        # 2. Value -> RID
        rids_corr = self.mapper.value_to_rid(corrupted_vals.view(-1)).view(-1, 16)
        rids_orig = self.mapper.value_to_rid(orig_vals_block.view(-1)).view(-1, 16)
        
        # 3. Symbol Packing
        sym_corr = self._pack_symbols(rids_corr)
        sym_orig = self._pack_symbols(rids_orig)

        # 4. Oracle Decoding
        sym_diff_cnt = (sym_corr != sym_orig).sum(dim=1)
        
        is_correctable = (sym_diff_cnt <= self.ecc_t)
        is_uncorrectable = (sym_diff_cnt > self.ecc_t)

        # 5. RangeGuard Apply

        # (1) 모든 원본 RID에 대해 대표값 미리 계산
        target_rids = rids_orig.long()

        flat_rids = target_rids.view(-1)
        flat_reps = self.mapper.rid_to_value(flat_rids)
        rep_vals = flat_reps.view(target_rids.shape).to(torch.int16)

        rep_vals_signed = merge_rep_with_corrupted_sign(
            rep_vals=rep_vals,
            corrupted_vals=corrupted_vals
        )

        correction_mask = (rids_corr != rids_orig) & is_correctable.unsqueeze(1)
        corrupted_vals = torch.where(correction_mask, rep_vals_signed, corrupted_vals)
        
        # 6. 메모리 반영
        flat_data.index_copy_(0, block_indices, corrupted_vals)

        # 7. Logging
        log_corrected_uncorrectable(
            logger=logger,
            where=where,
            slot_idx=slot_idx,
            block_indices=block_indices,
            corrected_mask=is_correctable,
            uncorrectable_mask=is_uncorrectable,
            orig_vals_i32=orig_vals_i32,
            after_u16=after_u16,
        )

        return after_u16
