# lm_eval/models/eccsim/schemes/nulling.py
import torch
from .base import ECCBase

class ECCNullingBF16(ECCBase):
    """
    Weight Nulling: 16-bit 청크 단위로 패리티를 검사하고,
    에러가 감지되면 해당 청크를 0으로 만듦.
    """
    def wants_injection(self, pattern: str) -> bool:
        return True

    def max_events_per_block(self) -> int:
        # Nulling은 이론상 모든 16비트 청크가 다 깨져도 감지(홀수 비트 에러) 및 Nulling 가능
        return 16 

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        # 시뮬레이션에서는 원본을 참조하므로 encode 생략 가능하지만,
        # 원리상으로는 각 16비트 청크의 패리티를 계산해야 함.
        return None

    def decode_and_correct(self, 
                           corrupted_data: torch.Tensor, 
                           parity: torch.Tensor, # 여기선 원본 데이터
                           **kwargs) -> torch.Tensor:
        """
        패리티 불일치 시 0으로 만듦 (Nulling).
        """
        # 데이터가 [B, 16] (int16) 형태로 들어옴
        # 각 int16 원소가 하나의 청크임.
        
        # 1. 원본과 현재 데이터 비교 (XOR) -> 0이 아니면 에러
        #    Nulling은 패리티(홀수 개 비트 에러)만 감지하지만,
        #    여기서는 시뮬레이션 편의상 '값이 다르면' 에러로 간주하고 Nulling 할 수도 있음.
        #    하지만 C++ 코드 로직(Parity Check)을 정확히 따르려면:
        #    Diff의 Popcount가 홀수여야만 감지됨. (짝수 비트 에러는 감지 불가)
        
        # 원본 데이터 복원 (필요 시)
        # hooks.py 구조상 orig_vals_i32는 부분적으로만 옴.
        # 하지만 Nulling 로직은 "저장된 패리티"와 "현재 데이터의 패리티"를 비교함.
        # 시뮬레이션에서는 "원본 데이터의 패리티"를 정답으로 사용.
        
        # 여기서는 단순화를 위해 "값이 다르면(에러 있으면) 무조건 Nulling"이 아니라,
        # "홀수 비트 에러일 때만 Nulling" 하는 정확한 로직을 구현해야 함.
        
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
        
        if orig_vals_i32 is None:
            return after_u16

        # ---------------------------------------------------------
        # 1. 값 준비 (Original vs Corrupted)
        # ---------------------------------------------------------
        # Corrupted Value (에러 주입된 현재 상태)
        bad_vals = after_u16.view(-1).index_select(0, slot_idx).to(torch.int32)
        # Original Value (정답)
        good_vals = orig_vals_i32
        
        diff = bad_vals ^ good_vals
        
        # ---------------------------------------------------------
        # 2. 패리티 검사 (홀수 비트 에러 감지)
        # ---------------------------------------------------------
        if hasattr(torch, "bitwise_count"):
            cnt = torch.bitwise_count(diff)
        else:
            cnt = torch.zeros_like(diff)
            t = diff.clone()
            for _ in range(16):
                cnt += (t & 1)
                t >>= 1
                
        detected_mask = (cnt % 2 != 0)
        
        # ---------------------------------------------------------
        # 3. Nulling 수행 및 로깅
        # ---------------------------------------------------------
        if detected_mask.any():
            null_idx = slot_idx[detected_mask]
            
            # A. Original
            val_orig = good_vals[detected_mask]
            # B. Corrupted
            val_corrupted = bad_vals[detected_mask]
            # C. Recovered (Nulling -> 0)
            val_recovered = torch.zeros_like(val_orig) # 0으로 초기화
            
            # 실제 메모리 Nulling 수행
            zeros = torch.zeros_like(null_idx, dtype=torch.int16)
            after_u16.view(-1).index_copy_(0, null_idx, zeros)
            
            if logger is not None:
                logger.log(f"{where}|nulling", null_idx, val_orig, val_recovered)

        undetected_mask = ~detected_mask
        if undetected_mask.any() and logger is not None:
            undet_idx = slot_idx[undetected_mask]
            val_orig = good_vals[undetected_mask]
            val_curr = bad_vals[undetected_mask]
            logger.log(f"{where}|undetected", undet_idx, val_orig, val_curr)

        return after_u16