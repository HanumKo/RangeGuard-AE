import torch

class ECCBase:
    """
    모든 ECC 스킴의 기본 클래스.
    - wants_injection: 어떤 패턴을 주입받을지 결정
    - max_events_per_block: 한 블록 내 최대 허용 에러 수
    - encode: 데이터 인코딩 (패리티 생성) - Real ECC 구현 시 사용
    - decode_and_correct: 정정 로직 - Real ECC 구현 시 사용
    """
    def __init__(self, device):
        self.device = device

    def wants_injection(self, pattern: str) -> bool:
        return True

    def max_events_per_block(self) -> int:
        return 1

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        [Real ECC] 데이터를 받아 패리티(Parity)를 계산하여 반환.
        """
        return None

    def decode_and_correct(self, 
                           corrupted_data: torch.Tensor, 
                           parity: torch.Tensor, 
                           **kwargs) -> torch.Tensor:
        """
        [Real ECC] 손상된 데이터와 패리티를 받아 정정된 데이터를 반환.
        """
        raise NotImplementedError

    def __call__(self, 
                 *, 
                 pattern: str, 
                 after_u16: torch.Tensor, 
                 slot_idx: torch.Tensor, 
                 logger=None, 
                 where: str="", 
                 orig_vals_i32: torch.Tensor | None = None, 
                 blk_event_counts: torch.Tensor | None = None):
        """
        hooks.py에서 호출하는 진입점.
        """
        pass