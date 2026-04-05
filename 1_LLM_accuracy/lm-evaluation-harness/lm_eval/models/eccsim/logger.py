import torch


class GPUChangeLogger:
    """
    Minimal GPU logger that stores only changes:
    a list of records with fields:
      block_idx [M], slot_idx [M], before_u16 [M], after_u16 [M]
    """
    def __init__(self, device, max_weight: int = 1000, max_activation: int = 1000):
        self.device = torch.device(device)
        self._where = []
        self._slot = []
        self._before = []
        self._after = []
        self._max = {"weight": int(max_weight), "activation": int(max_activation)}
        self._cnt = {"weight": 0, "activation": 0}

    def _bucket_and_room(self, where: str):
        if where.startswith("weight:"):
            cat = "weight"
        elif where.startswith("activation:"):
            cat = "activation"
        else:
            cat = "other"

        limit = self._max.get(cat, 0)
        used = self._cnt.get(cat, 0)
        room = max(0, limit - used)
        return cat, room

    def log(self, where: str, slot_idx: torch.Tensor, before_u16: torch.Tensor, after_u16: torch.Tensor):
        # slot_idx/before_u16/after_u16: 1D [K], 모두 같은 길이
        k = int(slot_idx.numel())
        if k == 0:
            return
        cat, room = self._bucket_and_room(where)
        if room <= 0:
            return
        if k > room:
            # 남은 몫만 취한다
            slot_idx = slot_idx[:room]
            before_u16 = before_u16[:room]
            after_u16 = after_u16[:room]
            k = room

        self._where.extend([where] * k)
        # GPU 상에서만 유지
        self._slot.append(slot_idx.detach())
        self._before.append(before_u16.detach())
        self._after.append(after_u16.detach())
        if cat in self._cnt:
            self._cnt[cat] += k

    def _u16_to_bf16_float(self, t_int32: torch.Tensor) -> torch.Tensor:
        # t_int32: 0..65535 범위의 정수 텐서 (dtype=int32, device=self.device)
        # bf16은 fp32 상위 16비트이므로, 상위로 16비트 시프트 후 float32로 비트캐스트
        u32 = (t_int32 & 0xFFFF).to(torch.int32)
        u32 = torch.bitwise_left_shift(u32, 16)
        # 같은 storage에서 dtype만 바꿔 비트캐스트
        f32 = u32.view(torch.float32)
        return f32

    def flush(self):
        # Empty-buffer fast path
        if not self._slot:
            return {
                "where": [],
                "slot_idx": torch.empty(0, dtype=torch.int16, device=self.device),
                "before": [],
                "after": [],
            }
        slot = torch.cat(self._slot).to(torch.int16)
        bef = torch.cat(self._before).to(torch.int32)
        aft = torch.cat(self._after).to(torch.int32)
        bef_f32 = self._u16_to_bf16_float(bef)
        aft_f32 = self._u16_to_bf16_float(aft)
        out = {
            "where": self._where,
            "slot_idx": slot,
            "before": bef_f32,
            "after": aft_f32,
        }
        self._where, self._slot, self._before, self._after = [], [], [], []
        return out
