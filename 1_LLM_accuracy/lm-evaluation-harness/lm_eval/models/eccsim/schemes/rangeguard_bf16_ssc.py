import torch
from .base import ECCBase
from .rangeguard_common import log_corrected_uncorrectable, merge_rep_with_corrupted_sign
from .rangetable_bf16 import BCAMapper

class ECCRangeGuardSSC(ECCBase):
    """
    RangeGuard BF16 SSC (Single Symbol Correction).
    - 4-bit RID (16 Bins)
    - RS(10, 8) over GF(2^8)
    - Real GF Arithmetic Implementation
    """
    def __init__(self, device):
        super().__init__(device)
        self.mapper = BCAMapper(device, num_bins=16) # Use 16 Bins
        self.device = device
        
        # GF(2^8) Tables (Primitive: 0x11D)
        self.gf_exp = torch.zeros(512, dtype=torch.long, device=device)
        self.gf_log = torch.full((256,), -1, dtype=torch.long, device=device)
        
        x = 1
        prim = 0x11D
        for i in range(255):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & 0x100: x ^= prim
        self.gf_exp[255] = 1
        for i in range(256, 512): self.gf_exp[i] = self.gf_exp[i-255]

    def wants_injection(self, pattern: str) -> bool: return True
    def max_events_per_block(self) -> int: return 9999

    def gf_mul_vec(self, a, b_scalar):
        # [FIX] Type match for indexing
        if b_scalar == 0: return torch.zeros_like(a)
        mask = (a != 0)
        idx = (self.gf_log[a] + self.gf_log[b_scalar].item())
        res = torch.zeros_like(a)
        res[mask] = self.gf_exp[idx[mask]].to(res.dtype)
        return res

    def gf_div_scalar(self, a, b):
        if a == 0: return 0
        idx = self.gf_log[a].item() - self.gf_log[b].item()
        if idx < 0: idx += 255
        return self.gf_exp[idx].item()

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
        
        # [FIX] int32 변환 제거 -> int16 그대로 사용
        flat_data = after_u16.view(-1).view(total_blocks, 16)
        
        block_indices = torch.unique(slot_idx // 16)
        
        corrupted_vals = flat_data.index_select(0, block_indices).clone()
        orig_vals_block = corrupted_vals.clone()
        
        local_blk_idx = torch.bucketize(slot_idx // 16, block_indices)
        local_slot_off = slot_idx % 16
        
        orig_flat = orig_vals_block.view(-1)
        target_idx = local_blk_idx * 16 + local_slot_off
        
        # [FIX] orig_vals(int32) -> int16 캐스팅
        orig_flat.index_copy_(0, target_idx, orig_vals_i32.to(torch.int16))

        # 2. Value -> RID
        rids_corr = self.mapper.value_to_rid(corrupted_vals.view(-1)).view(-1, 16)
        rids_orig = self.mapper.value_to_rid(orig_vals_block.view(-1)).view(-1, 16)
        
        sym_corr = self._pack_symbols(rids_corr)
        sym_orig = self._pack_symbols(rids_orig)

        # 3. Syndrome Calculation
        sym_diff = sym_corr ^ sym_orig
        S0 = torch.zeros(block_indices.numel(), dtype=torch.int64, device=self.device)
        S1 = torch.zeros(block_indices.numel(), dtype=torch.int64, device=self.device)
        
        for i in range(8):
            d = sym_diff[:, i]
            S0 ^= d
            coeff = self.gf_exp[i].item()
            term = self.gf_mul_vec(d, coeff)
            S1 ^= term

        # 4. Decoding (SSC)
        has_error = (S0 != 0) | (S1 != 0)
        error_blks_local = torch.where(has_error)[0]
        corrected_mask_local = torch.zeros_like(has_error, dtype=torch.bool)
        final_syms = sym_corr.clone()

        if error_blks_local.numel() > 0:
            s0_vec = S0[error_blks_local]
            s1_vec = S1[error_blks_local]
            
            log_s0 = self.gf_log[s0_vec]
            log_s1 = self.gf_log[s1_vec]
            log_ratio = log_s1 - log_s0
            log_ratio[log_ratio < 0] += 255
            
            J = log_ratio
            valid_pos = (J >= 0) & (J < 8) & (s0_vec != 0)
            
            if valid_pos.any():
                valid_idx = torch.where(valid_pos)[0]
                target_blks = error_blks_local[valid_idx]
                target_cols = J[valid_idx]
                err_vals = s0_vec[valid_idx]
                
                # Apply XOR
                # indexing trick for 2D assignment with 1D indices
                # final_syms[target_blks, target_cols] does NOT work as expected for picking points
                # We need to index linearly or use advanced indexing properly
                # Pytorch: tensor[row_indices, col_indices] works if both are long tensors
                
                final_syms[target_blks, target_cols] ^= err_vals
                corrected_mask_local[target_blks] = True

        # 5. Reconstruction
        rec_r0 = (final_syms >> 4) & 0xF
        rec_r1 = final_syms & 0xF
        rec_rids = torch.stack([rec_r0, rec_r1], dim=2).view(-1, 16)

        # 6. Apply RangeGuard
        ecc_action_mask = (rids_corr != rec_rids)
        if ecc_action_mask.any():
            target_rids = rec_rids[ecc_action_mask]
            rep_vals = self.mapper.rid_to_value(target_rids).to(torch.int16)

            flat_cv = corrupted_vals.view(-1)
            flat_mask = ecc_action_mask.view(-1)

            merged_vals = merge_rep_with_corrupted_sign(
                rep_vals=rep_vals,
                corrupted_vals=flat_cv[flat_mask]
            )

            flat_cv[flat_mask] = merged_vals

        flat_data.index_copy_(0, block_indices, corrupted_vals)

        # 7. Logging
        uncorr_mask = has_error & (~corrected_mask_local)
        log_corrected_uncorrectable(
            logger=logger,
            where=where,
            slot_idx=slot_idx,
            block_indices=block_indices,
            corrected_mask=corrected_mask_local,
            uncorrectable_mask=uncorr_mask,
            orig_vals_i32=orig_vals_i32,
            after_u16=after_u16,
        )

        return after_u16
