#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <array>

constexpr int kBits      = 256;  // block bits (32B)
constexpr int kWordBits  = 16;   // 16-bit span
constexpr int kDwordBits = 32;   // 32-bit span

struct PatternRNG {
    std::mt19937_64 gen;
    std::uniform_int_distribution<int> bit_dist{0, kBits - 1};
    std::uniform_int_distribution<int> word_dist{0, kBits / kWordBits - 1};     // 0..15
    std::uniform_int_distribution<int> bit_in_word_dist{0, kWordBits - 1};      // 0..15
    PatternRNG(uint64_t seed): gen(seed) {}
    inline int rand_bit()        { return bit_dist(gen); }
    inline int rand_word()       { return word_dist(gen); }
    inline int rand_bit_in_word(){ return bit_in_word_dist(gen); }
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
inline std::vector<int> sample_SE(PatternRNG& rng){
    return { rng.rand_bit() };
}

// DAE: 항상 인접한 2비트를 함께 뒤집음
inline std::vector<int> sample_DAE(PatternRNG& rng){
    int w = rng.rand_word();
    int b = std::uniform_int_distribution<int>(0, kWordBits - 2)(rng.gen); // 0..14
    int a = w * kWordBits + b;
    int c = a + 1;
    return { a, c };
}

// [PDE 관련 패턴 제거됨]
// - sample_PDE
// - sample_PDE_unique
// - sample_SE_plus_PDE_disjoint

// SWL16_aligned: 16비트 정렬 span에서 각 비트를 p=0.5로 뒤집되, 최소 1비트 보장
inline std::vector<int> sample_SWL16_aligned(PatternRNG& rng){
    int s = std::uniform_int_distribution<int>(0, kBits / kWordBits - 1)(rng.gen); // 0..15
    int start = s * kWordBits; // 16*i
    return bernoulli_span_min(start, kWordBits, rng, /*min_flips=*/1);
}

// SWD32_aligned: 32비트 정렬 span에서 각 비트를 p=0.5로 뒤집되, 최소 1비트 보장
inline std::vector<int> sample_SWD32_aligned(PatternRNG& rng){
    int s = std::uniform_int_distribution<int>(0, kBits / kDwordBits - 1)(rng.gen); // 0..7
    int start = s * kDwordBits; // 32*i
    return bernoulli_span_min(start, kDwordBits, rng, /*min_flips=*/1);
}

// shorthand
inline std::vector<int> sample_SWL16(PatternRNG& rng){
    return sample_SWL16_aligned(rng);
}
inline std::vector<int> sample_SWD32(PatternRNG& rng){
    return sample_SWD32_aligned(rng);
}

// 256비트 전체에서 각 비트를 p=0.5로 뒤집되, 최소 1비트는 반드시 뒤집힘
inline std::vector<int> sample_ALL256_half(PatternRNG& rng) {
    constexpr int span_bits = kBits; // 256
    std::array<uint64_t, 4> sel{};   // 4*64 = 256 bits
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

inline void mark_used(const std::vector<int>& xs, std::array<uint8_t, kBits>& used){
    for (int b : xs) used[b & (kBits - 1)] = 1u;
}

inline std::vector<int> union_sorted(std::vector<int> a, std::vector<int> b){
    std::array<uint8_t, kBits> pres{}; // presence bitset
    for (int x : a) pres[x & (kBits - 1)] = 1u;
    for (int x : b) pres[x & (kBits - 1)] = 1u;
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    for (int i = 0; i < kBits; ++i) if (pres[i]) out.push_back(i);
    return out;
}

// ========== unique variants (non-overlapping spans) ==========

inline std::vector<int> sample_SE_unique(PatternRNG& rng, const std::array<uint8_t, kBits>& used){
    for (int t = 0; t < 32; ++t){
        int b = rng.rand_bit();
        if (!used[b & (kBits - 1)]) return { b };
    }
    for (int i = 0; i < kBits; ++i) if (!used[i]) return { i };
    return {};
}

// DAE_unique: 두 adjacent bit 모두 unused인 span을 찾아,
// 항상 그 두 비트를 함께 뒤집음
inline std::vector<int> sample_DAE_unique(PatternRNG& rng, const std::array<uint8_t, kBits>& used){
    for (int t = 0; t < 64; ++t){
        int w = rng.rand_word();
        int b = std::uniform_int_distribution<int>(0, kWordBits - 2)(rng.gen);
        int a = w * kWordBits + b;
        int c = a + 1;
        if (!used[a] && !used[c]){
            return { a, c };
        }
    }
    for (int w = 0; w < kBits / kWordBits; ++w){
        for (int b = 0; b < kWordBits - 1; ++b){
            int a = w * kWordBits + b;
            int c = a + 1;
            if (!used[a] && !used[c]){
                return { a, c };
            }
        }
    }
    return {};
}

// [PDE unique 제거]

// SWL16_aligned_unique: span이 완전히 비어 있어야 함. 그 내부에서 p=0.5, 최소 1비트 보장
inline std::vector<int> sample_SWL16_aligned_unique(PatternRNG& rng, const std::array<uint8_t, kBits>& used){
    auto ok_span = [&](int start){
        for (int i = 0; i < kWordBits; ++i) if (used[start + i]) return false;
        return true;
    };
    for (int t = 0; t < 64; ++t){
        int s = std::uniform_int_distribution<int>(0, kBits / kWordBits - 1)(rng.gen);
        int start = s * kWordBits;
        if (ok_span(start)){
            return bernoulli_span_min(start, kWordBits, rng, /*min_flips=*/1);
        }
    }
    for (int s = 0; s < kBits / kWordBits; ++s){
        int start = s * kWordBits;
        if (ok_span(start)){
            return bernoulli_span_min(start, kWordBits, rng, /*min_flips=*/1);
        }
    }
    return {};
}

// SWD32_aligned_unique: span이 완전히 비어 있어야 함. 그 내부에서 p=0.5, 최소 1비트 보장
inline std::vector<int> sample_SWD32_aligned_unique(PatternRNG& rng, const std::array<uint8_t, kBits>& used){
    auto ok_span = [&](int start){
        for (int i = 0; i < kDwordBits; ++i) if (used[start + i]) return false;
        return true;
    };
    for (int t = 0; t < 64; ++t){
        int s = std::uniform_int_distribution<int>(0, kBits / kDwordBits - 1)(rng.gen);
        int start = s * kDwordBits;
        if (ok_span(start)){
            return bernoulli_span_min(start, kDwordBits, rng, /*min_flips=*/1);
        }
    }
    for (int s = 0; s < kBits / kDwordBits; ++s){
        int start = s * kDwordBits;
        if (ok_span(start)){
            return bernoulli_span_min(start, kDwordBits, rng, /*min_flips=*/1);
        }
    }
    return {};
}

// ========== composite patterns (disjoint by construction) ==========

inline std::vector<int> sample_SE_plus_SE_disjoint(PatternRNG& rng){
    std::array<uint8_t, kBits> used{};
    auto a = sample_SE(rng); mark_used(a, used);                 // SE: 항상 1비트
    auto b = sample_SE_unique(rng, used);                        // 또 다른 1비트
    return union_sorted(std::move(a), b);
}

inline std::vector<int> sample_SE_plus_DAE_disjoint(PatternRNG& rng){
    std::array<uint8_t, kBits> used{};
    auto a = sample_SE(rng); mark_used(a, used);                 // SE: 1비트
    auto b = sample_DAE_unique(rng, used);                       // DAE: 항상 인접 2비트
    return union_sorted(std::move(a), b);
}

// [PDE를 포함하는 복합 패턴 제거됨]
// - sample_SE_plus_PDE_disjoint

inline std::vector<int> sample_SE_plus_SWL_disjoint(PatternRNG& rng){
    std::array<uint8_t, kBits> used{};
    auto a = sample_SE(rng); mark_used(a, used);                 // SE: 1비트
    auto b = sample_SWL16_aligned_unique(rng, used);             // SWL: span 내에서 p=0.5, 최소 1비트
    return union_sorted(std::move(a), b);
}

inline std::vector<int> sample_SE_plus_SWD_disjoint(PatternRNG& rng){
    std::array<uint8_t, kBits> used{};
    auto a = sample_SE(rng); mark_used(a, used);                 // SE: 1비트
    auto b = sample_SWD32_aligned_unique(rng, used);             // SWD: span 내에서 p=0.5, 최소 1비트
    return union_sorted(std::move(a), b);
}

inline std::vector<int> sample_SWL_plus_SWL_disjoint(PatternRNG& rng){
    std::array<uint8_t, kBits> used{};
    auto a = sample_SWL16_aligned_unique(rng, used); mark_used(a, used);
    auto b = sample_SWL16_aligned_unique(rng, used);
    return union_sorted(std::move(a), b);
}

inline std::vector<int> sample_SWD_plus_SWD_disjoint(PatternRNG& rng){
    std::array<uint8_t, kBits> used{};
    auto a = sample_SWD32_aligned_unique(rng, used); mark_used(a, used);
    auto b = sample_SWD32_aligned_unique(rng, used);
    return union_sorted(std::move(a), b);
}
