#include "secded.hpp"
#include <stdexcept>
#include <algorithm>

// BitBlock256 마스크 생성: 데이터 인덱스(0..255) 목록 -> 256비트 마스크
BitBlock256 SecDedHamming256::mask_from_indices(const std::vector<int>& idxs){
    BitBlock256 m{};
    for(int i: idxs){
        int wi = i >> 6, bi = i & 63;
        m.w[wi] |= (uint64_t(1) << bi);
    }
    return m;
}

SecDedHamming256::SecDedHamming256(){
    build_all();
}

void SecDedHamming256::build_all(){
    // 1) 상위 15행 패턴: 첫 15개를 1<<r 로 두어 상단 15x15가 항등행렬이 되게 함
    patterns_.clear(); patterns_.reserve(271);
    std::vector<uint8_t> used(32768, 0);
    for(int r=0;r<15;r++){
        uint16_t v = uint16_t(1u<<r);
        patterns_.push_back(v);
        used[v]=1;
    }
    for(uint32_t v=1; v<32768 && (int)patterns_.size()<271; ++v){
        if(!used[v]) { patterns_.push_back(uint16_t(v)); used[v]=1; }
    }
    pat2col_.clear(); pat2col_.reserve(271);
    for(int c=0;c<271;c++) pat2col_[patterns_[c]] = c;

    // 2) parity/data 열 분할
    parity_cols_.clear();
    for(int r=0;r<15;r++) parity_cols_.push_back(r); // 0..14
    parity_cols_.push_back(271);                     // overall parity
    std::vector<int> is_par(272,0);
    for(int c:parity_cols_) is_par[c]=1;
    data_cols_.clear(); data_cols_.reserve(256);
    for(int j=0;j<272;j++) if(!is_par[j]) data_cols_.push_back(j);
    if((int)data_cols_.size()!=256) throw std::runtime_error("SECDED(272,256): bad split");

    // 3) HK로부터 parity 마스크 계산
    // HK(16x256) 정의:
    //  - 상위 15행 r: HK[r, i] = bit r of patterns_[ data_cols_[i] ] (단, data_cols_[i]∈[0..270])
    //  - 마지막 행(15): HK[15, i] = 1 (모든 데이터열)
    std::array<std::vector<int>,16> idxs_per_row;
    for(auto& v: idxs_per_row) v.clear();

    for(int i=0;i<256;i++){
        int col = data_cols_[i];
        if(0 <= col && col <= 270){
            uint16_t pat = patterns_[col];
            for(int r=0;r<15;r++){
                if((pat>>r)&1u) idxs_per_row[r].push_back(i);
            }
        }
        // overall parity row
        idxs_per_row[15].push_back(i);
    }

    // HR^{-1} 닫힌형:
    // HR =
    //   [ I_15 | 0 ]
    //   [ 1...1| 1 ]
    // ⇒ p[0..14] = s[0..14] ^ s[15],  p[15] = s[15]
    for (int r = 0; r < 15; ++r) {
        parity_masks_[r] = mask_from_indices(idxs_per_row[r]);
    }

    // p[15] = (xor of s[0..14]) XOR s[15]
    std::vector<int> merged;
    for (int r = 0; r < 15; ++r) {
        merged.insert(merged.end(), idxs_per_row[r].begin(), idxs_per_row[r].end());
    }
    merged.insert(merged.end(), idxs_per_row[15].begin(), idxs_per_row[15].end());

    // 대칭차(짝수번 등장 제거)
    std::sort(merged.begin(), merged.end());
    std::vector<int> sym; sym.reserve(merged.size());
    for (size_t t=0; t<merged.size(); ) {
        size_t u=t+1;
        while (u<merged.size() && merged[u]==merged[t]) ++u;
        if (((u-t)&1u)==1u) sym.push_back(merged[t]);
        t=u;
    }
    parity_masks_[15] = mask_from_indices(sym);
}

std::vector<bool> SecDedHamming256::encode(const BitBlock256& data) const{
    std::vector<bool> p(16,false);
    for(int r=0;r<16;r++){
        p[r] = (data.parity_and(parity_masks_[r]) != 0);
    }
    return p;
}

ECCResult SecDedHamming256::decode(const BitBlock256& data_err,
                                   const std::vector<bool>& parity) const{
    if((int)parity.size() < 16) throw std::runtime_error("SECDED decode: need 16 parity bits");

    // 예상 패리티
    int pexp[16];
    for(int r=0;r<16;r++) pexp[r] = data_err.parity_and(parity_masks_[r]) & 1;

    // 신드롬 s = p_stored XOR p_expected
    int s[16];
    for(int r=0;r<16;r++) s[r] = ((parity[r]?1:0) ^ pexp[r]);

    int sop = 0;
    for (int r=0; r<16; ++r) sop ^= s[r];       // s_true[15] = XOR of all s[*]

    uint16_t shi = 0;
    for (int r=0; r<15; ++r)                     // s_true[0..14] = s[0..14]
        if (s[r]) shi |= uint16_t(1u << r);

    // 판정
    if(sop==0 && shi==0){
        return {ECCStatus::Clean, data_err};
    }
    if(sop==1 && shi==0){
        // overall parity bit only
        return {ECCStatus::Corrected, data_err};
    }
    if(sop==1 && shi!=0){
        auto it = pat2col_.find(shi);
        if(it != pat2col_.end()){
            int col = it->second;       // 0..270
            if(0 <= col && col <= 14){
                // top-15 parity bit error
                return {ECCStatus::Corrected, data_err};
            }
            // 데이터 열 → data_cols_에서 위치 찾기
            auto it2 = std::lower_bound(data_cols_.begin(), data_cols_.end(), col);
            if(it2 != data_cols_.end() && *it2 == col){
                int di = int(it2 - data_cols_.begin()); // 0..255
                BitBlock256 corrected = data_err;
                corrected.flip(di);
                return {ECCStatus::Corrected, corrected};
            }
            // 이론상 거의 없음(열 분할 불일치)
            return {ECCStatus::Corrected, data_err};
        }
        // 단축으로 불가능한 패턴 → 다중 비트일 확률 높음
        return {ECCStatus::DetectedUncorrectable, data_err};
    }
    if(sop==0 && shi!=0){
        return {ECCStatus::DetectedUncorrectable, data_err};
    }
    return {ECCStatus::UndetectedError, data_err};
}
