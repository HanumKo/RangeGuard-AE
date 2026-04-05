#include "hbm3_crc_ssc.hpp"

Hbm3Crc16Ssc::Hbm3Crc16Ssc() {
    gf_init();
}

void Hbm3Crc16Ssc::gf_init() {
    uint32_t x = 1;
    for (int i = 0; i < 65535; ++i) {
        const uint16_t a = static_cast<uint16_t>(x & 0xFFFFu);
        gf_exp_[i] = a;
        gf_log_[a] = static_cast<uint16_t>(i);
        x <<= 1;
        if (x & 0x10000u) x ^= 0x1100Bu;
        x &= 0xFFFFu;
    }
    for (int i = 65535; i < 2 * 65535; ++i) gf_exp_[i] = gf_exp_[i - 65535];
    gf_log_[0] = 0xFFFFu;
}

uint16_t Hbm3Crc16Ssc::gf_mul(uint16_t a, uint16_t b) const {
    if (a == 0 || b == 0) return 0;
    return gf_exp_[gf_log_[a] + gf_log_[b]];
}

uint16_t Hbm3Crc16Ssc::gf_div(uint16_t a, uint16_t b) const {
    if (a == 0 || b == 0) return 0;
    int idx = static_cast<int>(gf_log_[a]) - static_cast<int>(gf_log_[b]);
    idx %= 65535;
    if (idx < 0) idx += 65535;
    return gf_exp_[idx];
}

uint16_t Hbm3Crc16Ssc::gf_pow_alpha(int i) const {
    i %= 65535;
    if (i < 0) i += 65535;
    return gf_exp_[i];
}

int Hbm3Crc16Ssc::get_bit(const BitBlock256& data, int idx) {
    const int wi = idx >> 6;
    const int bi = idx & 63;
    return static_cast<int>((data.w[wi] >> bi) & 1ull);
}

void Hbm3Crc16Ssc::set_bit(BitBlock256& data, int idx, int value) {
    const int wi = idx >> 6;
    const int bi = idx & 63;
    const uint64_t mask = (1ull << bi);
    if (value) data.w[wi] |= mask;
    else       data.w[wi] &= ~mask;
}

uint16_t Hbm3Crc16Ssc::load_payload_symbol(const BitBlock256& data, int sym_idx) {
    const int base = sym_idx * kSymbolBits;
    uint16_t value = 0;
    for (int b = 0; b < kSymbolBits; ++b) {
        value |= static_cast<uint16_t>(get_bit(data, base + b) << (kSymbolBits - 1 - b));
    }
    return value;
}

void Hbm3Crc16Ssc::store_payload_symbol(BitBlock256& data, int sym_idx, uint16_t value) {
    const int base = sym_idx * kSymbolBits;
    for (int b = 0; b < kSymbolBits; ++b) {
        const int bit = (value >> (kSymbolBits - 1 - b)) & 1u;
        set_bit(data, base + b, bit);
    }
}

uint16_t Hbm3Crc16Ssc::bits_to_u16_msb(const std::vector<bool>& bits, int offset) {
    uint16_t value = 0;
    for (int i = 0; i < kSymbolBits; ++i) {
        const int idx = offset + i;
        if (idx < static_cast<int>(bits.size()) && bits[idx]) {
            value |= static_cast<uint16_t>(1u << (kSymbolBits - 1 - i));
        }
    }
    return value;
}

void Hbm3Crc16Ssc::u16_to_bits_msb(uint16_t value, std::vector<bool>& out, int offset) {
    for (int i = 0; i < kSymbolBits; ++i) {
        out[offset + i] = ((value >> (kSymbolBits - 1 - i)) & 1u) != 0;
    }
}

uint16_t Hbm3Crc16Ssc::crc16_ccitt_0init(const BitBlock256& data) {
    uint16_t crc = 0x0000u;
    for (int i = 0; i < kPayloadBits / 8; ++i) {
        uint8_t byte = 0;
        for (int b = 0; b < 8; ++b) {
            byte |= static_cast<uint8_t>(get_bit(data, i * 8 + b) << (7 - b));
        }
        crc ^= static_cast<uint16_t>(byte << 8);
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 0x8000u) crc = static_cast<uint16_t>((crc << 1) ^ 0x1021u);
            else               crc = static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

std::vector<bool> Hbm3Crc16Ssc::encode(const BitBlock256& data) const {
    uint16_t symbols[kCodeSymbols] = {};
    for (int i = 0; i < 16; ++i) symbols[i] = load_payload_symbol(data, i);
    symbols[16] = crc16_ccitt_0init(data);

    uint16_t d0 = 0;
    uint16_t d1 = 0;
    for (int i = 0; i < kDataSymbols; ++i) {
        d0 ^= symbols[i];
        d1 ^= gf_mul(symbols[i], gf_pow_alpha(i));
    }

    const uint16_t a17 = gf_pow_alpha(17);
    const uint16_t a18 = gf_pow_alpha(18);
    const uint16_t denom = static_cast<uint16_t>(a17 ^ a18);
    const uint16_t p0 = gf_div(static_cast<uint16_t>(d1 ^ gf_mul(a18, d0)), denom);
    const uint16_t p1 = static_cast<uint16_t>(d0 ^ p0);

    std::vector<bool> out(48, false);
    u16_to_bits_msb(symbols[16], out, 0);
    u16_to_bits_msb(p0, out, 16);
    u16_to_bits_msb(p1, out, 32);
    return out;
}

ECCResult Hbm3Crc16Ssc::decode(const BitBlock256& data_err,
                               const std::vector<bool>& parity) const {
    uint16_t symbols[kCodeSymbols] = {};
    for (int i = 0; i < 16; ++i) symbols[i] = load_payload_symbol(data_err, i);
    symbols[16] = bits_to_u16_msb(parity, 0);
    symbols[17] = bits_to_u16_msb(parity, 16);
    symbols[18] = bits_to_u16_msb(parity, 32);

    uint16_t S0 = 0;
    uint16_t S1 = 0;
    for (int i = 0; i < kCodeSymbols; ++i) {
        const uint16_t ri = symbols[i];
        if (ri == 0) continue;
        const uint16_t ei = gf_log_[ri];
        if (ei == 0xFFFFu) continue;
        S0 ^= gf_exp_[ei];
        S1 ^= gf_exp_[(ei + i) % 65535];
    }

    bool corrected_any = false;
    if (!(S0 == 0 && S1 == 0)) {
        if (S0 == 0 || S1 == 0) {
            return {ECCStatus::DetectedUncorrectable, data_err};
        }

        int err_pos = -1;
        for (int j = 0; j < kCodeSymbols; ++j) {
            if (gf_mul(S0, gf_exp_[j]) == S1) {
                err_pos = j;
                break;
            }
        }
        if (err_pos < 0) {
            const uint16_t p = gf_log_[S0];
            const uint16_t q = gf_log_[S1];
            if (p == 0xFFFFu || q == 0xFFFFu) {
                return {ECCStatus::DetectedUncorrectable, data_err};
            }
            int pos = static_cast<int>(q) - static_cast<int>(p);
            pos %= 65535;
            if (pos < 0) pos += 65535;
            if (0 <= pos && pos < kCodeSymbols) err_pos = pos;
        }

        if (!(0 <= err_pos && err_pos < kCodeSymbols)) {
            return {ECCStatus::DetectedUncorrectable, data_err};
        }

        symbols[err_pos] ^= S0;
        corrected_any = true;
    }

    BitBlock256 corrected = data_err;
    for (int i = 0; i < 16; ++i) store_payload_symbol(corrected, i, symbols[i]);

    const uint16_t crc_expected = crc16_ccitt_0init(corrected);
    if (crc_expected != symbols[16]) {
        return {ECCStatus::DetectedUncorrectable, corrected};
    }

    return {corrected_any ? ECCStatus::Corrected : ECCStatus::Clean, corrected};
}
