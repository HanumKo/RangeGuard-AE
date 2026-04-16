#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "ecc/base.hpp"

// HBM3 read-path style ECC:
// - CRC16/CCITT over the 256-bit payload
// - On-die SSC over 17 data symbols (16 payload + 1 CRC) with 2 parity symbols
// Returned parity layout:
//   [0..15]   : CRC16 symbol (MSB-first)
//   [16..31]  : SSC parity p0 (MSB-first)
//   [32..47]  : SSC parity p1 (MSB-first)
class Hbm3Crc16Ssc final : public ECCScheme {
public:
    struct LegacyClassification {
        ECCStatus oecc_status = ECCStatus::Clean;
        ECCStatus secc_status = ECCStatus::Clean;
        ECCStatus final_status = ECCStatus::Clean;
        BitBlock256 corrected{};
    };

    Hbm3Crc16Ssc();

    const char* name() const override { return "HBM3-CRC16+SSC"; }
    std::vector<bool> encode(const BitBlock256& data) const override;
    ECCResult decode(const BitBlock256& data_err,
                     const std::vector<bool>& parity) const override;
    LegacyClassification classify_legacy(const BitBlock256& original_data,
                                         const BitBlock256& data_err,
                                         const std::vector<bool>& parity_err) const;

private:
    static constexpr int kPayloadBits = 256;
    static constexpr int kSymbolBits = 16;
    static constexpr int kDataSymbols = 17;
    static constexpr int kCodeSymbols = 19;

    std::array<uint16_t, 2 * 65535> gf_exp_{};
    std::array<uint16_t, 65536> gf_log_{};

    void gf_init();
    inline uint16_t gf_mul(uint16_t a, uint16_t b) const;
    inline uint16_t gf_div(uint16_t a, uint16_t b) const;
    inline uint16_t gf_pow_alpha(int i) const;

    static inline int get_bit(const BitBlock256& data, int idx);
    static inline void set_bit(BitBlock256& data, int idx, int value);
    static uint16_t load_payload_symbol(const BitBlock256& data, int sym_idx);
    static void store_payload_symbol(BitBlock256& data, int sym_idx, uint16_t value);
    static uint16_t bits_to_u16_msb(const std::vector<bool>& bits, int offset);
    static void u16_to_bits_msb(uint16_t value, std::vector<bool>& out, int offset);
    static uint16_t crc16_ccitt_0init(const BitBlock256& data);
};
