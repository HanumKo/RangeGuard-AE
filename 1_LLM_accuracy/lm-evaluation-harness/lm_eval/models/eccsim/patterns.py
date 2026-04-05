import os
import torch
from .utils import BLOCK_BYTES

# 단일 기본 분포(패턴 가중치).
DEFAULT_DIST = {
    "SE":  0.1002,
    "TSV": 0.00,
    "DAE": 0.1245,
    "SWL": 0.2374,
    "SWD": 0.5380,
}

# 패턴 1회 발생 시 '할당'되는 비트 수
BITS_PER_EVENT = {
    "SE": 1,
    "TSV": 2,
    "DAE": 2,
    "SWL": 16,
    "SWD": 32,
    "SE+SE": 2,
    "SE2": 2,
}

_AR16 = {}
_AR32 = {}

def _arange16(device):
    t = _AR16.get(device)
    if t is None:
        t = torch.arange(16, device=device, dtype=torch.long)
        _AR16[device] = t
    return t

def _arange32(device):
    t = _AR32.get(device)
    if t is None:
        t = torch.arange(32, device=device, dtype=torch.long)
        _AR32[device] = t
    return t

# =========================================================================
# [핵심 최적화] Sparse Sampling Helper
# 전체 블록을 스캔하지 않고, 에러 개수만큼만 인덱스를 생성합니다.
# =========================================================================
def _sample_sparse(num_blocks: int, prob: float, device):
    """
    Returns: (indices, count)
        - indices: 에러가 발생한 블록 인덱스들 [k]
        - count: 에러 개수 k (int)
    """
    if prob <= 0.0 or num_blocks <= 0:
        return None, 0

    # 1. 예상 에러 개수 계산 (Poisson 분포 근사)
    # lambda = n * p
    # 에러율이 매우 낮으므로 Poisson 분포를 따름
    expected_errors = num_blocks * prob
    
    # 에러가 너무 많으면(전체의 1% 이상) 기존 방식(Dense)이 나을 수도 있으나,
    # ECC 시뮬레이션에서는 보통 에러율이 낮으므로 항상 Sparse 방식이 유리함.
    
    # GPU에서 Poisson 샘플링하여 총 에러 개수 k 결정
    # (단일 스칼라 텐서)
    k_tensor = torch.poisson(torch.tensor(expected_errors, device=device))
    k = int(k_tensor.item()) # [Sync Point] 어쩔 수 없지만 1회만 발생

    if k == 0:
        return None, 0

    # 2. k개의 랜덤 위치 생성 (중복 허용 - 충돌은 자연스러운 현상)
    # randint는 매우 빠름
    indices = torch.randint(0, num_blocks, (k,), device=device, dtype=torch.long)
    
    return indices, k

# =========================================================================
# Pattern Generators (Optimized)
# =========================================================================

def _gen_SE(num_blocks: int, prob: float, device):
    # Single Error: 임의의 블록, 임의의 1비트
    blk_idx, k = _sample_sparse(num_blocks, prob, device)
    if k == 0: return torch.empty((0, 2), dtype=torch.long, device=device)

    # 1개 블록당 1비트 (0..255)
    # BLOCK_BYTES(32) * 8 = 256
    bit_idx = torch.randint(0, 256, (k,), device=device, dtype=torch.long)
    
    return torch.stack([blk_idx, bit_idx], dim=1)

def _gen_DAE(num_blocks: int, prob: float, device):
    # Double Adjacent Error: 임의의 블록, 임의의 비트 i와 i+1
    blk_idx, k = _sample_sparse(num_blocks, prob, device)
    if k == 0: return torch.empty((0, 2), dtype=torch.long, device=device)

    # 시작 비트 (0..254) - 마지막 비트는 i+1이 범위 밖일 수 있으므로 제외하거나 처리 필요
    # 여기서는 간단히 0..254로 제한 (맨 끝 비트 에러 빈도는 무시 가능)
    start_bit = torch.randint(0, 255, (k,), device=device, dtype=torch.long)
    
    # [k, 2] 형태: (blk, start), (blk, start+1)
    # 펼쳐서 [2k, 2]로 만듦
    
    # blk_idx: [A, B, ...] -> [A, A, B, B, ...]
    blk_expanded = blk_idx.repeat_interleave(2)
    
    # bit_idx: [s1, s2, ...] -> [s1, s1+1, s2, s2+1, ...]
    # stack([s, s+1]) -> [2, k] -> transpose -> [k, 2] -> flatten
    pair_bits = torch.stack([start_bit, start_bit + 1], dim=1).reshape(-1)
    
    # Modulo 256 (Wrap around) or just clip?
    # DAE usually stays in word. Let's wrap for simplicity or assume valid.
    pair_bits %= 256
    
    return torch.stack([blk_expanded, pair_bits], dim=1)

def _gen_SWL(num_blocks: int, prob: float, device):
    # Single Word Line: 한 블록 내의 특정 16비트(Word)가 랜덤하게 깨짐 (50% 확률 등)
    # 여기서는 정의상 "Word Line Fail"을 시뮬레이션.
    # 보통 32B 블록 내에 여러 비트 에러 발생.
    # 구현: 해당 블록의 256비트 중 랜덤하게 N개 비트 플립 or 특정 패턴
    # 기존 로직: "flatten().repeat_interleave(16)" -> 16비트 단위 에러?
    # 여기서는 "임의의 16비트 청크(Word) 전체가 랜덤 에러"라고 가정하거나
    # 기존 구현을 따라가겠습니다. (기존: start + span16)
    
    blk_idx, k = _sample_sparse(num_blocks, prob, device)
    if k == 0: return torch.empty((0, 2), dtype=torch.long, device=device)

    # 블록 내 어떤 Word(16bit)가 깨질지 선택 (32B = 16 words of 16bits)
    word_sel = torch.randint(0, 16, (k,), device=device, dtype=torch.long)
    start_bit = word_sel * 16 # 0, 16, 32 ...
    
    # 각 에러마다 16개의 비트 인덱스 생성
    # [k, 16]
    span = _arange16(device).unsqueeze(0) + start_bit.unsqueeze(1) # [k, 16]
    span = span.reshape(-1) # [16k]
    
    blk_expanded = blk_idx.repeat_interleave(16)
    
    # 마스킹 (보통 하드웨어 에러는 모든 비트가 1이 되진 않고 랜덤하게 0/1이 됨)
    # 50% 확률로 비트 반전
    mask = (torch.rand(blk_expanded.shape, device=device) < 0.5)
    
    return torch.stack([blk_expanded[mask], span[mask]], dim=1)

def _gen_SWD(num_blocks: int, prob: float, device):
    # Single Word Line (Entire Block Fail equivalent for 32B?)
    # 기존 코드: span32 (32 bits?) -> 기존 코드 로직 참조
    # 기존: (s * 32).long() + span32. -> 32비트 단위 에러
    
    blk_idx, k = _sample_sparse(num_blocks, prob, device)
    if k == 0: return torch.empty((0, 2), dtype=torch.long, device=device)

    # 32B 블록 = 256비트. 32비트 청크 8개.
    chunk_sel = torch.randint(0, 8, (k,), device=device, dtype=torch.long)
    start_bit = chunk_sel * 32
    
    span = _arange32(device).unsqueeze(0) + start_bit.unsqueeze(1) # [k, 32]
    span = span.reshape(-1)
    
    blk_expanded = blk_idx.repeat_interleave(32)
    
    mask = (torch.rand(blk_expanded.shape, device=device) < 0.5)
    
    return torch.stack([blk_expanded[mask], span[mask]], dim=1)


# =========================================================================
# Main Entry Point
# =========================================================================

def sample_bits_dist(num_blocks: int, per_bit_prob_10neg: int, device, dist=None, allow_multi=False):
    """
    Generate bit error indices based on distribution.
    Optimized for sparse errors (low probability).
    """
    if dist is None:
        dist = DEFAULT_DIST

    # 1. 기본 비트 에러율 (p_bit)
    # prob_10neg가 6이면 10^-6
    p_bit = 10.0 ** (-per_bit_prob_10neg)
    
    # 2. 각 패턴별 발생 확률(p_event) 계산
    # E_bits = sum( weight_i * bits_i )
    # P_event_total * E_bits = P_bit * Total_Bits
    # => P_event_total = (P_bit * 256) / E_bits  (Block size 32B=256bit 기준)
    
    total_weight = sum(dist.values())
    if total_weight <= 0: return {}
    
    avg_bits_per_event = 0.0
    for k, w in dist.items():
        # [수정] 할당 크기(Allocated Size) 가져오기
        b = float(BITS_PER_EVENT.get(k, 1))
        # [수정] SWD, SWL은 50% 마스킹되므로 기대 비트 수를 절반으로 줄임
        if k in ["SWD", "SWL"]:
            b *= 0.5
        avg_bits_per_event += w * b
        
    avg_bits_per_event /= total_weight
    
    # 블록 하나당 이벤트가 발생할 확률
    # (비트 에러율 * 블록 크기) / (이벤트당 평균 비트 수)
    p_block_event = (p_bit * 256.0) / avg_bits_per_event
    
    result = {}
    
    # 각 패턴별로 샘플링
    for pat_name, weight in dist.items():
        if weight <= 0: continue
        
        # 해당 패턴의 발생 확률
        # p_i = P_total * (weight_i / total_weight)
        p_pat = p_block_event * (weight / total_weight)
        
        # 패턴 생성 (Sparse)
        if pat_name == "SE":
            indices = _gen_SE(num_blocks, p_pat, device)
        elif pat_name == "DAE":
            indices = _gen_DAE(num_blocks, p_pat, device)
        elif pat_name == "SWL":
            indices = _gen_SWL(num_blocks, p_pat, device)
        elif pat_name == "SWD":
            indices = _gen_SWD(num_blocks, p_pat, device)
        else:
            # Fallback for unknown simple types (treat as SE)
            indices = _gen_SE(num_blocks, p_pat, device)
            
        if indices.numel() > 0:
            result[pat_name] = indices

    # allow_multi=True일 때 블록별 이벤트 카운트 계산
    if allow_multi and result:
        # 모든 에러 인덱스 모으기
        # 필요한 경우에만 계산 (오버헤드 방지)
        all_indices = []
        for v in result.values():
            all_indices.append(v[:, 0]) # 블록 인덱스만
        
        if all_indices:
            cat_blk = torch.cat(all_indices)
            if num_blocks > 100_000_000: # 1억개 이상이면 맵 생성 포기
                pass 
            else:
                counts = torch.bincount(cat_blk, minlength=num_blocks)
                result["__blk_event_counts__"] = counts.to(torch.uint8) # 메모리 절약

    return result

def sample_bits(kind_or_dist, num_blocks: int, per_bit_prob_10neg: int, device):
    if isinstance(kind_or_dist, dict):
        return sample_bits_dist(num_blocks, per_bit_prob_10neg, device=device, dist=kind_or_dist)
    return sample_bits_dist(num_blocks, per_bit_prob_10neg, device=device, dist={kind_or_dist: 1.0})