
#pragma once
#include <cstdint>
#include <array>

// 256-bit bitset as 4x uint64_t (little-endian: word[0] holds bits 0..63)
struct BitBlock256 {
    std::array<uint64_t,4> w{};

    inline void flip(int idx) noexcept {
        int wi = idx >> 6;           // /64
        int bi = idx & 63;           // %64
        w[wi] ^= (uint64_t(1) << bi);
    }

    inline int parity_and(const BitBlock256& mask) const noexcept {
        // return (popcount(w0&mask0)+...+w3)&1
        uint64_t a = w[0] & mask.w[0];
        uint64_t b = w[1] & mask.w[1];
        uint64_t c = w[2] & mask.w[2];
        uint64_t d = w[3] & mask.w[3];
        unsigned pc = __builtin_popcountll(a) + __builtin_popcountll(b)
                    + __builtin_popcountll(c) + __builtin_popcountll(d);
        return pc & 1u;
    }

    inline int parity() const noexcept {
        unsigned pc = __builtin_popcountll(w[0]) + __builtin_popcountll(w[1])
                    + __builtin_popcountll(w[2]) + __builtin_popcountll(w[3]);
        return pc & 1u;
    }
};
