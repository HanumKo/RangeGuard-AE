#include "weight_nulling.hpp"

static inline uint8_t parity16_of_chunk(const BitBlock256& b, int chunk_idx) {
    int start = chunk_idx * 16;
    uint8_t acc = 0;
    for (int j = 0; j < 16; ++j) {
        int bi = start + j;           // 0..255
        int wi = bi >> 6;             // /64
        int bj = bi & 63;             // %64
        acc ^= (uint8_t)((b.w[wi] >> bj) & 1ull);
    }
    return (uint8_t)(acc & 1u);
}

static inline void zero_chunk16(BitBlock256& b, int chunk_idx) {
    int start = chunk_idx * 16;
    for (int j = 0; j < 16; ++j) {
        int bi = start + j;
        int wi = bi >> 6;
        int bj = bi & 63;
        b.w[wi] &= ~(1ull << bj);
    }
}

std::vector<bool> WeightNulling16::encode(const BitBlock256& d) const {
    std::vector<bool> par(16);
    for (int c = 0; c < 16; ++c) par[c] = (parity16_of_chunk(d, c) != 0);
    return par;
}

ECCResult WeightNulling16::decode(const BitBlock256& e, const std::vector<bool>& parity) const {
    ECCResult r;
    r.corrected = e;

    if ((int)parity.size() != 16) {
        r.status = ECCStatus::DetectedUncorrectable;
        return r;
    }

    bool any_mismatch = false;
    for (int c = 0; c < 16; ++c) {
        bool now = (parity16_of_chunk(r.corrected, c) != 0);
        if (now != parity[c]) {
            any_mismatch = true;
            zero_chunk16(r.corrected, c);  // 감지된 청크 → 0으로 nulling
        }
    }

    // 감지된 청크가 하나라도 있으면 '교정(=nulling 수행)'으로 표기, 없으면 미감지
    r.status = any_mismatch ? ECCStatus::Corrected
                            : ECCStatus::UndetectedError;
    return r;
}
