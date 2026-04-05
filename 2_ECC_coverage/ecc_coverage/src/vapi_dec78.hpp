#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <random>

#include "ecc/base.hpp"
#include "util/bitblock256.hpp"

class VapiDEC64x4 final : public ECCScheme {
public:
    VapiDEC64x4(uint64_t seed = 12345);
    const char* name() const override { return "VAPI"; }
    std::vector<bool> encode(const BitBlock256& d) const override;
    ECCResult decode(const BitBlock256& noisy, const std::vector<bool>& parity) const override;

    struct Code78 { uint64_t data; uint16_t par; };
    struct DEC78 {
        std::vector<uint16_t> Hcols, Acols;
        std::unordered_map<uint16_t,int> single;
        std::unordered_map<uint16_t,uint16_t> pairs;
        static std::vector<uint16_t> build_H_unique_pairs(int n=78, int m=14, uint64_t seed=1);
        static void to_systematic(std::vector<uint16_t>& Hcols);
        static inline uint16_t syndrome(const std::vector<uint16_t>& Hcols, uint64_t lo, uint16_t hi);
        Code78 encode(uint64_t data) const;
        std::tuple<Code78, int> decode(const Code78& r) const; // 0/1/2/-1
        static inline void flip_bit(Code78& w, int i){
            if (i < 64) w.data ^= (1ull << i);
            else        w.par  ^= (uint16_t)(1u << (i-64));
        }
    };

private:
    DEC78 dec_;

    static inline int  get_bit(const BitBlock256& b, int i){
        return (int)((b.w[i>>6] >> (i&63)) & 1ull);
    }
    static inline void flip_bit(BitBlock256& b, int i){
        b.w[i>>6] ^= (1ull << (i&63));
    }
};
