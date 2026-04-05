#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "ecc/base.hpp"
#include "util/bitblock256.hpp"

// RS(12,8) over GF(2^4), symbols = upper 4 bits of each 32b lane (8 lanes) + 4 parity
// Full 2-symbol correction (d=5 MDS with GRS form).
// name(): "BCA-RS4U(12,8)" to match main.cpp classification.
class BcaRS4Upper : public ECCScheme {
public:
    BcaRS4Upper();
    const char* name() const override { return "RangeGuard DSC"; }

    // parity (4 symbols = 16 bits) as vector<bool> size 16
    std::vector<bool> encode(const BitBlock256& data) const override;

    ECCResult decode(const BitBlock256& data_err,
                     const std::vector<bool>& parity) const override;

private:
    // GF(2^4) tables for primitive 0x13 (x^4 + x + 1)
    uint8_t gf_exp_[32];
    int8_t  gf_log_[16];

    uint8_t X_[12]; // evaluation points (nonzero distinct): 1, α, α^2, ...

    static inline uint32_t get_lane_u32(const BitBlock256& b, int lane);
    static inline uint8_t  get_lane_upper4(const BitBlock256& b, int lane);
    static inline void     set_lane_upper4(BitBlock256& b, int lane, uint8_t up4);

    static inline uint8_t  gf_add(uint8_t a, uint8_t b){ return uint8_t(a ^ b); }
    inline uint8_t  gf_mul(uint8_t a, uint8_t b) const;
    inline uint8_t  gf_div(uint8_t a, uint8_t b) const;

    uint8_t  pow_field(uint8_t base, int j) const; // base^j in GF(16)
    inline uint8_t  G(int row, int col) const { return pow_field(X_[col], row); }

    void gf_init();
    void init_X();

    // 4x4 Gauss-Jordan in GF(16): A p = b
    static void solve4(uint8_t A[4][4], uint8_t b[4], uint8_t out[4], const BcaRS4Upper* self);

    static inline void pack16(std::vector<bool>& out, const uint8_t p[4]);
    static inline void unpack16(const std::vector<bool>& in, uint8_t p[4]);

    // syndromes S0..S3 from 12 symbols
    inline void syndromes(const uint8_t sym[12], uint8_t S[4]) const;

    // try 1-symbol and 2-symbol fixes on the 12-symbol vector
    bool try_fix_1(uint8_t sym[12], const uint8_t S[4], int& pos) const;
    bool try_fix_2(uint8_t sym[12], const uint8_t S[4], int& p0, int& p1) const;

    static bool solve2(const BcaRS4Upper* self,
                       uint8_t a,uint8_t b,uint8_t c,uint8_t d,
                       uint8_t u,uint8_t v, uint8_t& x,uint8_t& y);
};
