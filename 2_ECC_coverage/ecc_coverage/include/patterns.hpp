#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <array>

constexpr int kPayloadBits = 256;
constexpr int kWordBits    = 16;
constexpr int kDwordBits   = 32;

struct PatternShape {
    int total_bits = kPayloadBits;
    int word_bits = kWordBits;
    int dword_bits = kDwordBits;

    inline int word_spans() const { return total_bits / word_bits; }
    inline int dword_spans() const { return total_bits / dword_bits; }
};

struct PatternRNG {
    std::mt19937_64 gen;
    PatternRNG(uint64_t seed): gen(seed) {}

    inline int rand_bit(int total_bits) {
        return std::uniform_int_distribution<int>(0, total_bits - 1)(gen);
    }

    inline int rand_word(int word_spans) {
        return std::uniform_int_distribution<int>(0, word_spans - 1)(gen);
    }
};

// span 내부 각 비트를 p=0.5로 뒤집되, 최소 min_flips개는 뒤집히도록
// (현재 호출부에서는 min_flips=1로 사용)
inline std::vector<int> bernoulli_span_min(const int start, const int span_bits,
                                           PatternRNG& rng, const int min_flips = 1){
    // 선택 플래그 (최대 64비트까지만 필요: span_bits ∈ {16,32})
    uint64_t sel = 0;
    int cnt = 0;
    // p=0.5 → 하드웨어 난수의 LSB 사용
    for(int i = 0; i < span_bits; i++){
        if (rng.gen() & 1ULL){
            sel |= (1ULL << i);
            cnt++;
        }
    }
    // 최소 개수 보장: 부족하면 추가 뽑기(중복 회피)
    if (cnt < min_flips){
        std::uniform_int_distribution<int> pick(0, span_bits - 1);
        while (cnt < min_flips){
            int b = pick(rng.gen);
            if (((sel >> b) & 1ULL) == 0ULL){
                sel |= (1ULL << b);
                cnt++;
            }
        }
    }
    // 결과 벡터(정렬 상태)
    std::vector<int> v;
    v.reserve(cnt);
    for(int i = 0; i < span_bits; i++){
        if ((sel >> i) & 1ULL) v.push_back(start + i);
    }
    return v;
}

// ========== base patterns ==========

// SE: single random bit (항상 정확히 1비트)
inline std::vector<int> sample_SE(PatternRNG& rng, const PatternShape& shape){
    return { rng.rand_bit(shape.total_bits) };
}

// DAE: adjacent 2-bit region을 고른 뒤 각 비트를 p=0.5로 flip.
// 단, 00이 나오면 01 또는 10으로 강제하여 최소 1비트는 뒤집히게 함.
inline std::vector<int> sample_DAE(PatternRNG& rng, const PatternShape& shape){
    int w = rng.rand_word(shape.word_spans());
    int b = std::uniform_int_distribution<int>(0, shape.word_bits - 2)(rng.gen);
    int a = w * shape.word_bits + b;
    int c = a + 1;

    bool flip_a = (rng.gen() & 1ULL) != 0;
    bool flip_c = (rng.gen() & 1ULL) != 0;
    if (!flip_a && !flip_c) {
        if (rng.gen() & 1ULL) flip_a = true;
        else                  flip_c = true;
    }

    std::vector<int> out;
    out.reserve(2);
    if (flip_a) out.push_back(a);
    if (flip_c) out.push_back(c);
    return out;
}

// [PDE 관련 패턴 제거됨]
// - sample_PDE
// - sample_PDE_unique
// - sample_SE_plus_PDE_disjoint

// SWL16_aligned: 16비트 정렬 span에서 각 비트를 p=0.5로 뒤집되, 최소 1비트 보장
inline std::vector<int> sample_SWL16_aligned(PatternRNG& rng, const PatternShape& shape){
    int s = std::uniform_int_distribution<int>(0, shape.word_spans() - 1)(rng.gen);
    int start = s * shape.word_bits;
    return bernoulli_span_min(start, shape.word_bits, rng, /*min_flips=*/1);
}

// SWD32_aligned: 32비트 정렬 span에서 각 비트를 p=0.5로 뒤집되, 최소 1비트 보장
inline std::vector<int> sample_SWD32_aligned(PatternRNG& rng, const PatternShape& shape){
    int s = std::uniform_int_distribution<int>(0, shape.dword_spans() - 1)(rng.gen);
    int start = s * shape.dword_bits;
    return bernoulli_span_min(start, shape.dword_bits, rng, /*min_flips=*/1);
}

// shorthand
inline std::vector<int> sample_SWL16(PatternRNG& rng, const PatternShape& shape){
    return sample_SWL16_aligned(rng, shape);
}
inline std::vector<int> sample_SWD32(PatternRNG& rng, const PatternShape& shape){
    return sample_SWD32_aligned(rng, shape);
}

// codeword 전체에서 각 비트를 p=0.5로 뒤집되, 최소 1비트는 반드시 뒤집힘
inline std::vector<int> sample_ALL256_half(PatternRNG& rng, const PatternShape& shape) {
    const int span_bits = shape.total_bits;
    std::vector<uint64_t> sel((span_bits + 63) / 64, 0);
    int cnt = 0;

    // p=0.5로 flip 결정 (rng.gen의 LSB 사용)
    for (int i = 0; i < span_bits; ++i) {
        if (rng.gen() & 1ULL) {
            sel[i / 64] |= (1ULL << (i % 64));
            cnt++;
        }
    }

    // 최소 1비트 보장
    if (cnt < 1) {
        std::uniform_int_distribution<int> pick(0, span_bits - 1);
        int b = pick(rng.gen);
        sel[b / 64] |= (1ULL << (b % 64));
        cnt = 1;
    }

    // 결과 벡터 (정렬 상태)
    std::vector<int> out;
    out.reserve(cnt);
    for (int i = 0; i < span_bits; ++i) {
        if (sel[i / 64] & (1ULL << (i % 64))) out.push_back(i);
    }
    return out;
}

// ========== disjoint helpers ==========

inline void mark_used(const std::vector<int>& xs, std::vector<uint8_t>& used){
    for (int b : xs) used[b] = 1u;
}

inline std::vector<int> union_sorted(std::vector<int> a, std::vector<int> b, const PatternShape& shape){
    std::vector<uint8_t> pres(shape.total_bits, 0); // presence bitset
    for (int x : a) pres[x] = 1u;
    for (int x : b) pres[x] = 1u;
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    for (int i = 0; i < shape.total_bits; ++i) if (pres[i]) out.push_back(i);
    return out;
}

// ========== unique variants (non-overlapping spans) ==========

inline std::vector<int> sample_SE_unique(PatternRNG& rng, const PatternShape& shape, const std::vector<uint8_t>& used){
    for (int t = 0; t < 32; ++t){
        int b = rng.rand_bit(shape.total_bits);
        if (!used[b]) return { b };
    }
    for (int i = 0; i < shape.total_bits; ++i) if (!used[i]) return { i };
    return {};
}

// DAE_unique: 두 adjacent bit 모두 unused인 span을 찾은 뒤,
// 각 비트를 p=0.5로 flip하되 00이면 01/10으로 강제.
inline std::vector<int> sample_DAE_unique(PatternRNG& rng, const PatternShape& shape, const std::vector<uint8_t>& used){
    auto sample_pair = [&](int a, int c) {
        bool flip_a = (rng.gen() & 1ULL) != 0;
        bool flip_c = (rng.gen() & 1ULL) != 0;
        if (!flip_a && !flip_c) {
            if (rng.gen() & 1ULL) flip_a = true;
            else                  flip_c = true;
        }

        std::vector<int> out;
        out.reserve(2);
        if (flip_a) out.push_back(a);
        if (flip_c) out.push_back(c);
        return out;
    };

    for (int t = 0; t < 64; ++t){
        int w = rng.rand_word(shape.word_spans());
        int b = std::uniform_int_distribution<int>(0, shape.word_bits - 2)(rng.gen);
        int a = w * shape.word_bits + b;
        int c = a + 1;
        if (!used[a] && !used[c]){
            return sample_pair(a, c);
        }
    }
    for (int w = 0; w < shape.word_spans(); ++w){
        for (int b = 0; b < shape.word_bits - 1; ++b){
            int a = w * shape.word_bits + b;
            int c = a + 1;
            if (!used[a] && !used[c]){
                return sample_pair(a, c);
            }
        }
    }
    return {};
}

// [PDE unique 제거]

// SWL16_aligned_unique: span이 완전히 비어 있어야 함. 그 내부에서 p=0.5, 최소 1비트 보장
inline std::vector<int> sample_SWL16_aligned_unique(PatternRNG& rng, const PatternShape& shape, const std::vector<uint8_t>& used){
    auto ok_span = [&](int start){
        for (int i = 0; i < shape.word_bits; ++i) if (used[start + i]) return false;
        return true;
    };
    for (int t = 0; t < 64; ++t){
        int s = std::uniform_int_distribution<int>(0, shape.word_spans() - 1)(rng.gen);
        int start = s * shape.word_bits;
        if (ok_span(start)){
            return bernoulli_span_min(start, shape.word_bits, rng, /*min_flips=*/1);
        }
    }
    for (int s = 0; s < shape.word_spans(); ++s){
        int start = s * shape.word_bits;
        if (ok_span(start)){
            return bernoulli_span_min(start, shape.word_bits, rng, /*min_flips=*/1);
        }
    }
    return {};
}

// SWD32_aligned_unique: span이 완전히 비어 있어야 함. 그 내부에서 p=0.5, 최소 1비트 보장
inline std::vector<int> sample_SWD32_aligned_unique(PatternRNG& rng, const PatternShape& shape, const std::vector<uint8_t>& used){
    auto ok_span = [&](int start){
        for (int i = 0; i < shape.dword_bits; ++i) if (used[start + i]) return false;
        return true;
    };
    for (int t = 0; t < 64; ++t){
        int s = std::uniform_int_distribution<int>(0, shape.dword_spans() - 1)(rng.gen);
        int start = s * shape.dword_bits;
        if (ok_span(start)){
            return bernoulli_span_min(start, shape.dword_bits, rng, /*min_flips=*/1);
        }
    }
    for (int s = 0; s < shape.dword_spans(); ++s){
        int start = s * shape.dword_bits;
        if (ok_span(start)){
            return bernoulli_span_min(start, shape.dword_bits, rng, /*min_flips=*/1);
        }
    }
    return {};
}

// ========== composite patterns (disjoint by construction) ==========

inline std::vector<int> sample_SE_plus_SE_disjoint(PatternRNG& rng, const PatternShape& shape){
    std::vector<uint8_t> used(shape.total_bits, 0);
    auto a = sample_SE(rng, shape); mark_used(a, used);                 // SE: 항상 1비트
    auto b = sample_SE_unique(rng, shape, used);                        // 또 다른 1비트
    return union_sorted(std::move(a), b, shape);
}

inline std::vector<int> sample_SE_plus_DAE_disjoint(PatternRNG& rng, const PatternShape& shape){
    std::vector<uint8_t> used(shape.total_bits, 0);
    auto a = sample_SE(rng, shape); mark_used(a, used);                 // SE: 1비트
    auto b = sample_DAE_unique(rng, shape, used);                       // DAE: 항상 인접 2비트
    return union_sorted(std::move(a), b, shape);
}

// [PDE를 포함하는 복합 패턴 제거됨]
// - sample_SE_plus_PDE_disjoint

inline std::vector<int> sample_SE_plus_SWL_disjoint(PatternRNG& rng, const PatternShape& shape){
    std::vector<uint8_t> used(shape.total_bits, 0);
    auto a = sample_SE(rng, shape); mark_used(a, used);                 // SE: 1비트
    auto b = sample_SWL16_aligned_unique(rng, shape, used);             // SWL: span 내에서 p=0.5, 최소 1비트
    return union_sorted(std::move(a), b, shape);
}

inline std::vector<int> sample_SE_plus_SWD_disjoint(PatternRNG& rng, const PatternShape& shape){
    std::vector<uint8_t> used(shape.total_bits, 0);
    auto a = sample_SE(rng, shape); mark_used(a, used);                 // SE: 1비트
    auto b = sample_SWD32_aligned_unique(rng, shape, used);             // SWD: span 내에서 p=0.5, 최소 1비트
    return union_sorted(std::move(a), b, shape);
}

inline std::vector<int> sample_SWL_plus_SWL_disjoint(PatternRNG& rng, const PatternShape& shape){
    std::vector<uint8_t> used(shape.total_bits, 0);
    auto a = sample_SWL16_aligned_unique(rng, shape, used); mark_used(a, used);
    auto b = sample_SWL16_aligned_unique(rng, shape, used);
    return union_sorted(std::move(a), b, shape);
}

inline std::vector<int> sample_SWD_plus_SWD_disjoint(PatternRNG& rng, const PatternShape& shape){
    std::vector<uint8_t> used(shape.total_bits, 0);
    auto a = sample_SWD32_aligned_unique(rng, shape, used); mark_used(a, used);
    auto b = sample_SWD32_aligned_unique(rng, shape, used);
    return union_sorted(std::move(a), b, shape);
}
