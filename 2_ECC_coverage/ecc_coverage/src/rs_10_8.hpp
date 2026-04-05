#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "ecc/base.hpp"
#include "util/bitblock256.hpp"

// RS(10,8) over GF(2^8), symbols = upper 8 bits of each 32b lane (8 lanes)
// Parity: 2 symbols (16 bits total). SSC(1-symbol) decoder.
// name(): "BCA-RS8U(10,8)" to match main.cpp classification.
class BcaRS8Upper : public ECCScheme {
public:
    BcaRS8Upper();
    const char* name() const override { return "RangeGuard SSC"; }

    // parity bitstream (16 bits) returned as vector<bool> of size 16
    std::vector<bool> encode(const BitBlock256& data) const override;

    // decode uses only upper-8 of each 32b lane; lower bits are passed-through
    ECCResult decode(const BitBlock256& data_err,
                     const std::vector<bool>& parity) const override;

private:
    // GF(2^8) tables for primitive 0x11D
    uint8_t gf_exp_[512];
    int16_t gf_log_[256];

    static inline uint32_t get_lane_u32(const BitBlock256& b, int lane);
    static inline void     set_lane_upper8(BitBlock256& b, int lane, uint8_t up8);

    static inline uint8_t  gf_add(uint8_t a, uint8_t b){ return uint8_t(a ^ b); }
    inline uint8_t  gf_mul(uint8_t a, uint8_t b) const;
    inline uint8_t  gf_div(uint8_t a, uint8_t b) const;

    void gf_init();

    // RS(10,8) helpers
    static inline void pack16(std::vector<bool>& out, uint8_t s8, uint8_t s9);
    static inline void unpack16(const std::vector<bool>& in, uint8_t& s8, uint8_t& s9);

    // compute syndromes from received upper8 symbols and stored parity symbols
    inline void syndromes(const uint8_t sym[10], uint8_t& S0, uint8_t& S1) const;
};
