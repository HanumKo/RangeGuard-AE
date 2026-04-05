import torch
from .base import ECCBase

class ECCNone(ECCBase):
    def wants_injection(self, pattern: str) -> bool:
        return True

    def can_recover(self, error_mask) -> bool:
        return False

    def max_events_per_block(self) -> int:
        return 999999

    def __call__(self, *, pattern: str, after_u16: torch.Tensor, slot_idx: torch.Tensor, 
                 logger=None, where: str="", orig_vals_i32: torch.Tensor | None = None, 
                 blk_event_counts: torch.Tensor | None = None):
        
        if slot_idx.numel() == 0:
            return after_u16

        if logger is not None and orig_vals_i32 is not None:
            flat_data = after_u16.view(-1)
            val_orig = orig_vals_i32
            # None 스킴은 수정하지 않으므로 현재 값(corrupted)이 곧 최종 값입니다.
            val_final = flat_data.index_select(0, slot_idx).to(torch.int32)
            
            # [수정] 인자를 2개(Orig, Final)만 전달
            logger.log(f"{where}|uncorrectable", slot_idx, val_orig, val_final)

        return after_u16