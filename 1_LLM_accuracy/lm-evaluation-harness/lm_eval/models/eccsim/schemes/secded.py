import torch
from .base import ECCBase
from ..utils import BLOCK_BYTES

class ECCSECDED(ECCBase):
    """
    SEC-DED (272, 256) implementation based on Extended Hamming Code.
    Logic ported from C++ SecDedHamming256.
    """
    def __init__(self, device):
        super().__init__(device)
        self._build_tables()

    def wants_injection(self, pattern: str) -> bool:
        return True

    def max_events_per_block(self) -> int:
        return 1

    def _build_tables(self):
        """
        C++ build_all() 로직을 파이썬으로 구현.
        """
        # 1. Patterns 생성
        patterns = []
        used = set()
        
        for r in range(15):
            v = (1 << r)
            patterns.append(v)
            used.add(v)
            
        v = 1
        while len(patterns) < 271 and v < 32768:
            if v not in used:
                patterns.append(v)
                used.add(v)
            v += 1
            
        self.patterns = torch.tensor(patterns, dtype=torch.int32, device=self.device)
        
        # 2. Column Split
        parity_cols_list = list(range(15)) + [271]
        is_parity = [False] * 272
        for c in parity_cols_list:
            is_parity[c] = True
            
        data_cols_list = [c for c in range(272) if not is_parity[c]]
        
        # 3. Parity Masks 생성
        masks = torch.zeros((16, 8), dtype=torch.int32, device=self.device)
        
        for r in range(15):
            for i, col_idx in enumerate(data_cols_list):
                if 0 <= col_idx <= 270:
                    pat = patterns[col_idx]
                    if (pat >> r) & 1:
                        word_idx = i // 32
                        bit_idx = i % 32
                        masks[r, word_idx] |= (1 << bit_idx)
                        
        masks[15, :] = -1  # Overall parity
        
        final_masks = masks.clone()
        xor_sum = torch.zeros(8, dtype=torch.int32, device=self.device)
        for r in range(16):
            xor_sum ^= masks[r]
        final_masks[15] = xor_sum
        
        self.parity_masks = final_masks
        
        # Syndrome -> Data Index Map
        self.syndrome_to_data_idx = torch.full((32768,), -1, dtype=torch.int64, device=self.device)
        for i, col in enumerate(data_cols_list):
            if col <= 270:
                pat = patterns[col]
                self.syndrome_to_data_idx[pat] = i

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        return None

    def decode_and_correct(self, corrupted_data, parity, **kwargs):
        pass

    def __call__(self, 
                 *, 
                 pattern: str, 
                 after_u16: torch.Tensor, 
                 slot_idx: torch.Tensor, 
                 logger=None, 
                 where: str="", 
                 orig_vals_i32: torch.Tensor | None = None, 
                 blk_event_counts: torch.Tensor | None = None):
        
        if slot_idx.numel() == 0:
            return after_u16

        if not after_u16.is_contiguous():
            after_u16 = after_u16.contiguous()

        # -----------------------------------------------------------
        # 1. 데이터 준비
        # -----------------------------------------------------------
        total_blocks = after_u16.numel() // 16
        flat_data = after_u16.view(-1).view(torch.int32).view(total_blocks, 8)
        
        # 영향받은 블록 인덱스
        block_idx = torch.unique(slot_idx // 16)
        
        # Corrupted 상태의 블록 가져오기
        bad_blocks = flat_data.index_select(0, block_idx).clone() 

        # 원본(Original) 복원 (신드롬 계산용)
        orig_blocks = bad_blocks.clone()
        if orig_vals_i32 is not None:
            block_map = torch.bucketize(slot_idx // 16, block_idx)
            offsets = slot_idx % 16
            
            orig_blocks_i16 = orig_blocks.view(torch.int16).view(-1, 16)
            vals_i16 = orig_vals_i32.to(torch.int16)
            orig_blocks_i16[block_map, offsets] = vals_i16
            orig_blocks = orig_blocks_i16.view(torch.int32).view(-1, 8)

        # -----------------------------------------------------------
        # [DEBUG] 실제 비트 차이 확인 (DAE 패턴일 때만)
        # -----------------------------------------------------------
        # if "DAE" in pattern:
        #     # XOR Diff 구하기
        #     debug_diff = bad_blocks ^ orig_blocks
            
        #     # Popcount (비트 개수 세기)
        #     if hasattr(torch, "bitwise_count"):
        #         bit_diff_cnt = torch.bitwise_count(debug_diff).sum(dim=1) # [Block_Count]
        #     else:
        #         bit_diff_cnt = torch.zeros(debug_diff.shape[0], dtype=torch.int32, device=self.device)
        #         t = debug_diff.clone()
        #         for _ in range(32):
        #             bit_diff_cnt += (t & 1).sum(dim=1).to(torch.int32)
        #             t >>= 1
            
        #     # 통계 출력
        #     # 1비트 차이인 블록 수
        #     cnt_1bit = (bit_diff_cnt == 1).sum().item()
        #     # 2비트 차이인 블록 수
        #     cnt_2bit = (bit_diff_cnt == 2).sum().item()
        #     # 그 외
        #     cnt_other = (bit_diff_cnt > 2).sum().item()
            
        #     if cnt_1bit > 0 or cnt_2bit > 0:
        #         print(f"\n[DEBUG SECDED] Pattern={pattern} | Total Blocks: {block_idx.numel()}")
        #         print(f"  - 1-bit errors (Impossible for DAE): {cnt_1bit}")
        #         print(f"  - 2-bit errors (Correct for DAE):    {cnt_2bit}")
        #         print(f"  - Other errors:                      {cnt_other}")
                
        #         # 만약 1비트 에러가 있다면 샘플 하나 출력해보기
        #         if cnt_1bit > 0:
        #             print("  [WARNING] DAE pattern but 1-bit error detected! This causes 'Corrected'.")

        # -----------------------------------------------------------
        # 2. 신드롬 계산 & 정정
        # -----------------------------------------------------------
        diff = bad_blocks ^ orig_blocks
        syndromes = torch.zeros((block_idx.numel(), 16), dtype=torch.int32, device=self.device)
        
        for r in range(16):
            mask_r = self.parity_masks[r]
            masked = diff & mask_r
            if hasattr(torch, "bitwise_count"):
                cnt = torch.bitwise_count(masked).sum(dim=1)
            else:
                cnt = torch.zeros(masked.shape[0], dtype=torch.int32, device=self.device)
                t = masked.clone()
                for _ in range(32):
                    cnt += (t & 1).sum(dim=1).to(torch.int32)
                    t >>= 1
            syndromes[:, r] = cnt % 2

        s_raw = syndromes
        sop = torch.zeros(block_idx.numel(), dtype=torch.int32, device=self.device)
        for r in range(16):
            sop ^= s_raw[:, r]
            
        shi = torch.zeros(block_idx.numel(), dtype=torch.int64, device=self.device)
        for r in range(15):
            bit = s_raw[:, r].to(torch.int64)
            shi |= (bit << r)
            
        corr_mask = (sop == 1)
        due_mask = (sop == 0) & (shi != 0)
        
        # 실제 비트 플립(Correction) 수행
        data_err_mask = corr_mask & (shi != 0)
        if data_err_mask.any():
            target_shi = shi[data_err_mask]
            bit_idx = self.syndrome_to_data_idx[target_shi] 
            valid_correction = (bit_idx != -1)
            
            # 유효한 1비트 에러라고 판단된 경우만 수정
            if valid_correction.any():
                final_blk_indices = torch.where(data_err_mask)[0][valid_correction]
                final_bit_indices = bit_idx[valid_correction]
                
                wd = final_bit_indices // 32
                off = final_bit_indices % 32
                mask_flip = (1 << off).to(torch.int32)
                
                bad_blocks[final_blk_indices, wd] ^= mask_flip

        # -----------------------------------------------------------
        # 3. 결과 반영 (메모리 업데이트)
        # -----------------------------------------------------------
        # [중요] 로깅 전에 메모리에 먼저 반영해야 'recovered_val'을 정확히 뽑을 수 있음
        flat_data.index_copy_(0, block_idx, bad_blocks)

        # -----------------------------------------------------------
        # 4. 로깅 (중복 제거됨)
        # -----------------------------------------------------------
        if logger is not None:
            # A. 수정 수행 (Corrected) - 1비트 에러로 판단하고 고친 경우
            # (DAE가 여기 찍힌다면 SDC(오정정) 상황입니다)
            if corr_mask.any():
                c_blk_indices = torch.where(corr_mask)[0]
                c_blk_global = block_idx[c_blk_indices]
                
                # 전체 slot_idx 중 해당 블록에 속하는 것들만 필터링
                mask_c = torch.isin(slot_idx // 16, c_blk_global)
                
                if mask_c.any():
                    idx = slot_idx[mask_c]
                    val_orig = orig_vals_i32[mask_c]
                    # 이미 반영된 after_u16에서 값 추출
                    val_rec = after_u16.view(-1).index_select(0, idx).to(torch.int32)
                    
                    logger.log(f"{where}|corrected", idx, val_orig, val_rec)

            # B. 수정 불가 (Uncorrectable) - 2비트 이상 에러로 판단한 경우
            if due_mask.any():
                d_blk_indices = torch.where(due_mask)[0]
                d_blk_global = block_idx[d_blk_indices]
                
                mask_d = torch.isin(slot_idx // 16, d_blk_global)
                
                if mask_d.any():
                    idx = slot_idx[mask_d]
                    val_orig = orig_vals_i32[mask_d]
                    val_rec = after_u16.view(-1).index_select(0, idx).to(torch.int32)
                    
                    logger.log(f"{where}|uncorrectable", idx, val_orig, val_rec)

        return after_u16