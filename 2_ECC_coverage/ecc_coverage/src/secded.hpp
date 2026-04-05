#pragma once
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>
#include "ecc/base.hpp"            // ECCScheme, ECCStatus, ECCResult, BitBlock256

// SEC-DED(272,256): Extended Hamming (r=15 + overall parity = 16 parity bits)
class SecDedHamming256 : public ECCScheme {
public:
    SecDedHamming256();
    const char* name() const override { return "SEC-DED"; }

    // parity 길이 16 반환: [0..14] = 해밍 코어, [15] = 전체 패리티
    std::vector<bool> encode(const BitBlock256& data) const override;

    // parity.size() >= 16 필요
    ECCResult decode(const BitBlock256& data_err,
                     const std::vector<bool>& parity) const override;

private:
    // 상위 15행 패턴들(비영 15-bit) 271개, lookup: pattern -> column(0..270)
    std::vector<uint16_t> patterns_;                 // size 271
    std::unordered_map<uint16_t,int> pat2col_;       // 15bit -> column

    // 열 분할: parity 열 16개(0..14=코어, 271=전체) / data 열 256개
    std::vector<int> parity_cols_;                   // size 16
    std::vector<int> data_cols_;                     // size 256

    // 인코딩용: 각 parity bit가 XOR할 데이터 비트 집합을 256-bit 마스크로 보관
    std::array<BitBlock256,16> parity_masks_{};

    // 내부 유틸
    static BitBlock256 mask_from_indices(const std::vector<int>& idxs);
    void build_all();   // patterns_, pat2col_, parity_cols_, data_cols_, parity_masks_ 구성
};
