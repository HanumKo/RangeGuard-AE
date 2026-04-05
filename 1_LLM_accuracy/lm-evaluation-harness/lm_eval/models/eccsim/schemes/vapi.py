import torch
import random
from .base import ECCBase

class ECCVAPI_DEC(ECCBase):
    """
    VAPI-DEC (78, 64) Implementation.
    - Generates a collision-free H-Matrix dynamically on initialization.
    - Guarantees 100% correction for 1-bit and 2-bit errors.
    """
    def __init__(self, device, seed=12345):
        super().__init__(device)
        self.seed = seed
        self._build_tables()

    def wants_injection(self, pattern: str) -> bool:
        return True

    def max_events_per_block(self) -> int:
        return 2

    def _build_tables(self):
        rng = random.Random(self.seed)
        m = 14 # Parity bits
        n_data = 64
        
        cols = []
        forbidden = set()
        
        candidates = list(range(1, 1 << m))
        rng.shuffle(candidates)
        
        cand_idx = 0
        while len(cols) < n_data:
            if cand_idx >= len(candidates):
                raise RuntimeError("VAPI Init Failed: Cannot find 64 collision-free columns. Change seed.")
            
            c = candidates[cand_idx]
            cand_idx += 1
            
            if c in forbidden:
                continue
                
            is_valid = True
            new_pairs = []
            
            for old in cols:
                pair_syn = c ^ old
                if pair_syn in forbidden:
                    is_valid = False
                    break
                new_pairs.append(pair_syn)
            
            if not is_valid:
                continue
            
            cols.append(c)
            forbidden.add(c)
            for p in new_pairs:
                forbidden.add(p) 

        self.A_cols = torch.tensor(cols, dtype=torch.int16, device=self.device)
        self.syndrome_map = torch.zeros(1 << 14, dtype=torch.int64, device=self.device)
        
        def get_mask_tensor(val):
            if val >= (1 << 63):
                return val - (1 << 64)
            return val

        for i in range(64):
            s = int(self.A_cols[i])
            mask_val = (1 << i)
            self.syndrome_map[s] = get_mask_tensor(mask_val)
            
        for i in range(64):
            col_i = int(self.A_cols[i])
            for j in range(i + 1, 64):
                col_j = int(self.A_cols[j])
                s = col_i ^ col_j
                mask_val = (1 << i) | (1 << j)
                self.syndrome_map[s] = get_mask_tensor(mask_val)

    def __call__(self, 
                 *, 
                 pattern: str, 
                 after_u16: torch.Tensor, 
                 slot_idx: torch.Tensor, 
                 logger=None, 
                 where: str="", 
                 orig_vals_i32: torch.Tensor | None = None, 
                 blk_event_counts: torch.Tensor | None = None):
        
        if slot_idx.numel() == 0 or orig_vals_i32 is None:
            return after_u16

        # 1. 64-bit Chunk 준비 (4 * 16bit)
        chunk_idx = torch.unique(slot_idx // 4)
        
        # [수정] Shape 보정: view(-1)을 먼저 수행하여 1차원으로 만든 후 int64로 변환
        # 이렇게 하면 입력 텐서의 차원 모양과 상관없이 안전하게 변환됩니다.
        flat_data_64 = after_u16.view(-1).view(torch.int64)
        
        bad_chunks = flat_data_64.index_select(0, chunk_idx).clone()
        
        # [중요] 수정 전 상태 백업 (Corrupted)
        corrupted_chunks_backup = bad_chunks.clone()

        # 원본 복원
        orig_chunks = bad_chunks.clone()
        orig_chunks_16 = orig_chunks.view(torch.int16).view(-1, 4)
        chunk_map = torch.bucketize(slot_idx // 4, chunk_idx)
        offsets = slot_idx % 4
        
        vals_i16 = orig_vals_i32.to(torch.int16)
        orig_chunks_16[chunk_map, offsets] = vals_i16
        orig_chunks = orig_chunks_16.view(torch.int64).view(-1)

        # 2. 신드롬 계산
        diff = bad_chunks ^ orig_chunks
        syndromes = torch.zeros(diff.shape, dtype=torch.int16, device=self.device)
        
        for j in range(64):
            mask = (diff >> j) & 1
            col_val = self.A_cols[j]
            syndromes ^= (mask.to(torch.int16) * col_val)

        # 3. Lookup & Correction
        syn_idx = syndromes.to(torch.long)
        correction_masks = self.syndrome_map.index_select(0, syn_idx)
        
        is_error = (syn_idx != 0)
        has_valid_mask = (correction_masks != 0)
        
        correctable = is_error & has_valid_mask
        uncorrectable = is_error & (~has_valid_mask)
        
        if correctable.any():
            corr_idx = torch.where(correctable)[0]
            masks = correction_masks[corr_idx]
            # [수정] bad_chunks가 Recovered 상태로 변함
            bad_chunks[corr_idx] ^= masks

        # 4. 메모리 반영
        flat_data_64.index_copy_(0, chunk_idx, bad_chunks)
        
        # 5. 로깅 (Original -> Corrupted -> Recovered)
        if logger is not None:
            # (A) Corrected
            if correctable.any():
                corr_local = torch.where(correctable)[0]
                corr_global = chunk_idx[corr_local]
                mask_in_corr = torch.isin(slot_idx // 4, corr_global)
                
                if mask_in_corr.any():
                    idx = slot_idx[mask_in_corr]
                    
                    # 1) Original
                    val_orig = orig_vals_i32[mask_in_corr]
                    
                    # 2) Corrupted (백업해둔 corrupted_chunks_backup에서 추출)
                    offsets = idx % 4
                    chunk_pos = torch.bucketize(idx // 4, chunk_idx)
                    
                    # 64비트 청크를 다시 16비트 뷰로 변환하여 추출
                    c_chunks_i16 = corrupted_chunks_backup.view(torch.int16).view(-1, 4)
                    val_corrupted = c_chunks_i16[chunk_pos, offsets].to(torch.int32)
                    
                    # 3) Recovered (이미 수정된 after_u16에서 추출)
                    # (after_u16이 이미 flat_data_64 업데이트를 통해 갱신되었음)
                    # 여기서는 뷰 변환을 다시 해서 안전하게 가져옵니다.
                    val_rec = after_u16.view(-1).index_select(0, idx).to(torch.int32)
                    
                    # [수정] 2개 인자만 전달
                    logger.log(f"{where}|corrected", idx, val_orig, val_rec)

            # (B) Uncorrectable
            if uncorrectable.any():
                due_chunk_local = torch.where(uncorrectable)[0]
                due_chunk_global = chunk_idx[due_chunk_local]
                mask_in_due = torch.isin(slot_idx // 4, due_chunk_global)
                
                if mask_in_due.any():
                    idx = slot_idx[mask_in_due]
                    
                    # 1) Original
                    val_orig = orig_vals_i32[mask_in_due]
                    
                    # 2) Recovered (수정 실패 -> Corrupted와 동일)
                    val_rec = after_u16.view(-1).index_select(0, idx).to(torch.int32)
                    
                    # [수정] 2개 인자만 전달
                    logger.log(f"{where}|uncorrectable", idx, val_orig, val_rec)

        return after_u16