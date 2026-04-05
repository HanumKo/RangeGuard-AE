import torch

# --- Constants ---
BLOCK_BYTES = 32               # 32B block
BF16_BYTES = 2                 # bf16 = 2 bytes
BFPER_BLOCK = BLOCK_BYTES // BF16_BYTES  # 16 values

def changed_slots_from_bit_indices(bit_indices: torch.Tensor) -> torch.Tensor:
    """
    bit_indices: [M, 2] (blk_idx, bit_idx: 0..255)
    반환: 평탄화 슬롯 인덱스 [K], 전체 [num_blocks * 16] 기준
    """
    if bit_indices.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=bit_indices.device)

    blk = bit_indices[:, 0].long()              # [M]
    bit = bit_indices[:, 1].long()              # [M]

    slot_in_block = bit // 16                   # [M] 0..15
    flat_slot = blk * 16 + slot_in_block        # [M]
    flat_slot = torch.unique(flat_slot)         # 중복 제거

    return flat_slot.to(torch.int64)

def changed_slots_from_flips(flips_u8: torch.Tensor) -> torch.Tensor:
    """
    flips_u8: [B,32] uint8 XOR 마스크
    같은 bf16 슬롯(2바이트) 중 하나라도 뒤집혔으면 그 슬롯을 'changed'로 간주
    반환: 1D int64 평탄화 슬롯 인덱스 (길이 K), 전체 [B*16] 기준
    """
    if flips_u8.numel() == 0:
        return flips_u8.new_empty((0,), dtype=torch.int64)
    B = flips_u8.shape[0]
    f16 = flips_u8.view(B, 16, 2)          # [B,16,2 bytes]
    changed = f16.any(dim=2)               # [B,16] bool
    b, s = torch.where(changed)
    return (b.to(torch.int64) * 16 + s.to(torch.int64))

def bf16_get_exponent_sparse(u16_view_Bx16: torch.Tensor, slot_idx: torch.Tensor) -> torch.Tensor:
    if slot_idx.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8, device=u16_view_Bx16.device)
    flat = u16_view_Bx16.reshape(-1).to(torch.int32)     # int16 -> int32 (부호 확장 방지용 마스크 필요)
    sel  = flat.index_select(0, slot_idx)                # [K] int32 (하위 16b만 유효)
    uns  = sel & 0xFFFF                                   # 16비트 무부호로 정규화
    exp  = (uns >> 7) & 0xFF                              # [14:7]
    return exp.to(torch.uint8)

def set_bf16_exponent_sparse(u16_view_Bx16: torch.Tensor, slot_idx: torch.Tensor, new_exp_u8: torch.Tensor):
    if slot_idx.numel() == 0:
        return
    flat = u16_view_Bx16.reshape(-1).to(torch.int32)     # 작업은 32비트에서
    sel  = flat.index_select(0, slot_idx)                # [K] int32
    uns  = sel & 0xFFFF                                   # 16비트 영역만
    cleared = uns & ~(0xFF << 7)                          # 지수 필드 [14:7]을 0으로
    new_uns = cleared | ((new_exp_u8.to(torch.int32) & 0xFF) << 7)
    new_i16 = (new_uns & 0xFFFF).to(torch.int16)          # 16비트로 되돌림

    # write-back
    flat_i16 = u16_view_Bx16.reshape(-1)                  # int16 뷰
    flat_i16.index_copy_(0, slot_idx, new_i16)

def as_uint8_view(t: torch.Tensor) -> torch.Tensor:
    """
    Return a writeable uint8 view of the underlying tensor memory on the same device.
    Only supports contiguous tensors.
    """
    if not t.is_contiguous():
        raise RuntimeError("as_uint8_view: tensor must be contiguous; use caller-side copy-back path")
    return t.view(torch.uint8)

def blocks_32B(u8: torch.Tensor):
    """
    Reshape a flat uint8 tensor into [num_blocks, 32].
    If the length is not multiple of 32, we ignore the tail (no partial block processing).
    """
    n = (u8.numel() // BLOCK_BYTES) * BLOCK_BYTES
    if n == 0:
        return u8.new_zeros((0, BLOCK_BYTES)), 0
    v = u8[:n].view(-1, BLOCK_BYTES)
    return v, n

def bf16_u16_views(blocks: torch.Tensor):
    """
    Given blocks [N,32] uint8, return an int16 little-endian view [N,16] of bf16 slots.
    """
    assert blocks.dtype == torch.uint8
    # little-endian: low byte first; torch.view on cuda respects native endianness (LE on x86/NVIDIA)
    return blocks.view(blocks.shape[0], -1, 2).view(torch.int16)[..., :]

def bf16_get_exponent(u16: torch.Tensor) -> torch.Tensor:
    # bf16(top 16 of fp32): 1 sign, 8 exp, 7 mantissa
    # u16은 int16 뷰 → 연산은 int32로 올려서 sign 확장 영향 제거 후 마스크
    u32 = u16.to(torch.int32) & 0xFFFF
    return (u32 >> 7) & 0xFF

def bf16_set_exponent(u16: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
    u32 = u16.to(torch.int32) & 0xFFFF
    e32 = exp.to(torch.int32) & 0xFF
    out = (u32 & ~(0xFF << 7)) | (e32 << 7)
    return out.to(torch.int16)

def xor_inplace(blocks: torch.Tensor, bit_indices: torch.Tensor) -> None:
    """
    Apply bit flips to blocks using XOR.
    Handles multiple bit flips in the same byte correctly using index_add_ reduction.
    """
    if bit_indices.numel() == 0:
        return

    blk = bit_indices[:, 0].long()
    bit = bit_indices[:, 1].long()

    # 1. 바이트 인덱스 및 마스크 계산
    flat_byte_idx = blk * BLOCK_BYTES + (bit >> 3)
    bit_off = (bit & 7).long()
    
    lut = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8, device=blocks.device)
    raw_masks = lut.index_select(0, bit_off)

    # 2. 충돌 처리 (같은 바이트에 여러 비트 에러가 있는 경우 합치기)
    # 유니크한 바이트 위치를 찾습니다.
    unique_bytes, inverse_indices = torch.unique(flat_byte_idx, return_inverse=True)
    
    # 해당 유니크 바이트에 적용할 마스크들을 누적(Sum)합니다.
    # DAE는 서로 다른 비트(예: 1, 2)를 건드리므로 Sum(1+2=3)은 OR/XOR(1^2=3)와 결과가 같습니다.
    # int32로 변환하여 안전하게 더합니다.
    temp_masks = torch.zeros_like(unique_bytes, dtype=torch.int32)
    
    # index_add_: 중복된 인덱스(inverse_indices)에 해당하는 값들을 모두 더해줍니다. (Race Condition 해결)
    temp_masks.index_add_(0, inverse_indices, raw_masks.to(torch.int32))
    
    # 다시 uint8로 변환 (비트 마스크 완성)
    final_masks = temp_masks.to(torch.uint8)
    
    # 3. 원본 데이터에 XOR 적용 (유니크한 위치에 대해 한 번씩만 수행)
    flat = blocks.view(-1)
    
    # (1) 읽기
    target_vals = flat.index_select(0, unique_bytes)
    # (2) 수정 (XOR)
    modified_vals = target_vals ^ final_masks
    # (3) 쓰기
    flat.index_copy_(0, unique_bytes, modified_vals)

def mask_to_pairs(mask: torch.Tensor):
    """
    Convert a boolean mask [N,16] indicating which bf16 slots changed
    to a list of (block_idx, slot_idx) pairs.
    """
    idx = mask.nonzero(as_tuple=False)
    return idx

def changed_slots(before_u16: torch.Tensor, after_u16: torch.Tensor) -> torch.Tensor:
    return (before_u16 != after_u16)
