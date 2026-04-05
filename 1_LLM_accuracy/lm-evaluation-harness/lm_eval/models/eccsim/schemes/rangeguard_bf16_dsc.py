import torch
from .base import ECCBase
from .rangeguard_common import log_corrected_uncorrectable, merge_rep_with_corrupted_sign
from .rangetable_bf16 import BCAMapper

class ECCRangeGuardDSC(ECCBase):
    """
    RangeGuard BF16 DSC (Double Symbol Correction).
    - 2-bit RID (4 Bins)
    - RS(12, 8) over GF(2^4)
    - Real GF Arithmetic Implementation
    """
    def __init__(self, device):
        super().__init__(device)
        self.mapper = BCAMapper(device, num_bins=4) # Use 4 Bins
        self.device = device
        
        # GF(2^4) Setup - Table values fit in int32/int64
        self.gf_exp = torch.zeros(32, dtype=torch.long, device=device)
        self.gf_log = torch.full((16,), -1, dtype=torch.long, device=device)
        x = 1; prim = 0x13
        for i in range(15):
            self.gf_exp[i] = x; self.gf_log[x] = i
            x <<= 1; 
            if x & 0x10: x ^= prim
        self.gf_exp[15] = 1
        for i in range(16, 32): self.gf_exp[i] = self.gf_exp[i-15]
        self.X_ = self.gf_exp[:12].clone()

    def wants_injection(self, pattern: str) -> bool: return True
    def max_events_per_block(self) -> int: return 9999

    def gf_mul(self, a, b):
        if a == 0 or b == 0: return 0
        return self.gf_exp[(self.gf_log[a] + self.gf_log[b]) % 15].item()
    
    def _gf_div_cpu(self, a, b): 
        if a == 0: return 0
        if not hasattr(self, 'gf_log_cpu'): 
            self.gf_log_cpu = self.gf_log.cpu().numpy()
            self.gf_exp_cpu = self.gf_exp.cpu().numpy()
        return self.gf_exp_cpu[(self.gf_log_cpu[a] - self.gf_log_cpu[b]) % 15]
        
    def _gf_mul_cpu(self, a, b):
        if a == 0 or b == 0: return 0
        if not hasattr(self, 'gf_log_cpu'):
            self.gf_log_cpu = self.gf_log.cpu().numpy()
            self.gf_exp_cpu = self.gf_exp.cpu().numpy()
        return self.gf_exp_cpu[(self.gf_log_cpu[a] + self.gf_log_cpu[b]) % 15]

    def gf_pow(self, base, power):
        if power == 0: return 1
        if base == 0: return 0
        idx = (self.gf_log[base] * power) % 15
        return self.gf_exp[idx].item()

    def G(self, row, col): return self.gf_pow(self.X_[col].item(), row)

    def gf_mul_vec(self, a, scalar):
        # [FIX] Type mismatch fix
        if scalar == 0: return torch.zeros_like(a)
        mask = (a != 0)
        idx = (self.gf_log[a] + self.gf_log[scalar].item()) % 15
        res = torch.zeros_like(a)
        # self.gf_exp is Long(int64), but 'a' (and 'res') is likely Int(int32).
        # Explicit cast required.
        res[mask] = self.gf_exp[idx[mask]].to(res.dtype)
        return res

    def _pack_symbols(self, rids):
        # 2 RIDs (2-bit) -> 1 Symbol (4-bit)
        pairs = rids.view(rids.shape[0], 8, 2)
        return (pairs[:, :, 0] << 2) | pairs[:, :, 1]

    def __call__(self, *, pattern: str, after_u16: torch.Tensor, slot_idx: torch.Tensor, 
                 logger=None, where: str="", orig_vals_i32: torch.Tensor | None = None, 
                 blk_event_counts: torch.Tensor | None = None):
        
        if slot_idx.numel() == 0 or orig_vals_i32 is None: return after_u16

        # 1. 데이터 준비
        total_blocks = after_u16.numel() // 16
        # int16 유지
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
        
        sym_corr = self._pack_symbols(rids_corr)
        sym_orig = self._pack_symbols(rids_orig)

        # 3. Syndrome Calculation
        sym_diff = sym_corr ^ sym_orig
        S = torch.zeros((block_indices.numel(), 4), dtype=torch.int64, device=self.device)
        
        for j in range(4):
            acc = torch.zeros(block_indices.numel(), dtype=torch.int64, device=self.device)
            for i in range(8):
                coeff = self.G(j, i)
                term = self.gf_mul_vec(sym_diff[:, i], coeff)
                acc ^= term
            S[:, j] = acc

        # 4. Decoding (DSC)
        has_error = (S != 0).any(dim=1)
        error_blks_local = torch.where(has_error)[0]
        corrected_mask_local = torch.zeros_like(has_error, dtype=torch.bool)
        final_syms = sym_corr.clone()

        if len(error_blks_local) > 0:
            S_cpu = S[error_blks_local].cpu().numpy()
            sym_cpu = final_syms[error_blks_local].cpu().numpy()
            fixed_indices = []

            for k in range(len(error_blks_local)):
                s_vec = S_cpu[k]
                row_sym = sym_cpu[k]
                fixed = False
                
                # Try 1-symbol
                for p in range(12):
                    e_val, have, ok = 0, False, True
                    for j in range(4):
                        g = self.G(j, p)
                        if g == 0:
                            if s_vec[j] != 0: ok = False; break
                        else:
                            ej = self._gf_div_cpu(s_vec[j], g)
                            if not have: e_val = ej; have = True
                            elif ej != e_val: ok = False; break
                    if ok and have and e_val != 0:
                        if p < 8: row_sym[p] ^= e_val
                        fixed = True
                        break
                
                # Try 2-symbol
                if not fixed:
                    for a in range(12):
                        for b in range(a+1, 12):
                            ga0, gb0 = self.G(0, a), self.G(0, b)
                            ga1, gb1 = self.G(1, a), self.G(1, b)
                            det = self._gf_mul_cpu(ga0, gb1) ^ self._gf_mul_cpu(gb0, ga1)
                            if det == 0: continue
                            inv = self._gf_div_cpu(1, det)
                            t1 = self._gf_mul_cpu(s_vec[0], gb1); t2 = self._gf_mul_cpu(s_vec[1], gb0)
                            e0 = self._gf_mul_cpu(inv, t1 ^ t2)
                            t3 = self._gf_mul_cpu(s_vec[1], ga0); t4 = self._gf_mul_cpu(s_vec[0], ga1)
                            e1 = self._gf_mul_cpu(inv, t3 ^ t4)
                            if e0==0 and e1==0: continue
                            
                            lhs2 = self._gf_mul_cpu(e0, self.G(2,a)) ^ self._gf_mul_cpu(e1, self.G(2,b))
                            if lhs2 != s_vec[2]: continue
                            lhs3 = self._gf_mul_cpu(e0, self.G(3,a)) ^ self._gf_mul_cpu(e1, self.G(3,b))
                            if lhs3 != s_vec[3]: continue
                            
                            if a < 8: row_sym[a] ^= e0
                            if b < 8: row_sym[b] ^= e1
                            fixed = True
                            break
                        if fixed: break
                
                if fixed: fixed_indices.append(k)

            if fixed_indices:
                final_syms[error_blks_local] = torch.tensor(sym_cpu, device=self.device)
                corrected_mask_local[error_blks_local[fixed_indices]] = True

        # 5. Reconstruction
        rec_r0 = (final_syms >> 2) & 0x3
        rec_r1 = final_syms & 0x3
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
