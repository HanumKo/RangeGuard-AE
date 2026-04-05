#include "rs_10_8.hpp"
#include <cassert>

BcaRS8Upper::BcaRS8Upper(){ gf_init(); }

void BcaRS8Upper::gf_init(){
    const uint16_t prim = 0x11D;
    uint16_t x = 1;
    for (int i=0;i<255;++i){ gf_exp_[i]=uint8_t(x); gf_log_[x]=i; x<<=1; if (x&0x100) x^=prim; }
    gf_exp_[255]=1;
    for (int i=256;i<512;++i) gf_exp_[i]=gf_exp_[i-255];
    gf_log_[0] = -1;
}
uint8_t BcaRS8Upper::gf_mul(uint8_t a, uint8_t b) const{
    if (!a || !b) return 0;
    return gf_exp_[ gf_log_[a] + gf_log_[b] ];
}
uint8_t BcaRS8Upper::gf_div(uint8_t a, uint8_t b) const{
    if (!a) return 0;
    if (!b) return 0;
    int idx = gf_log_[a] - gf_log_[b];
    if (idx < 0) idx += 255;
    return gf_exp_[idx];
}

uint32_t BcaRS8Upper::get_lane_u32(const BitBlock256& b, int lane){
    const int bit0 = lane * 32;
    const int wi   = bit0 >> 6;   // 0..3
    const int off  = bit0 & 63;   // 0 또는 32
    if (off == 0){
        return uint32_t(b.w[wi] & 0xFFFFFFFFull);
    } else {
        const uint64_t lo = b.w[wi] >> off;
        const uint64_t hi = (wi+1 < 4) ? (b.w[wi+1] << (64 - off)) : 0ull;
        return uint32_t((lo | hi) & 0xFFFFFFFFull);
    }
}

void BcaRS8Upper::set_lane_upper8(BitBlock256& b, int lane, uint8_t up8){
    // write bits [lane*32+24 .. lane*32+31] with up8
    const int base = lane*32 + 24;
    for (int k=0;k<8;++k){
        int bi = base + k;
        int wi = bi >> 6, bo = bi & 63;
        uint64_t mask = (1ull<<bo);
        if ( (up8 >> k) & 1 ) b.w[wi] |= mask; else b.w[wi] &= ~mask;
    }
}

void BcaRS8Upper::pack16(std::vector<bool>& out, uint8_t s8, uint8_t s9){
    out.assign(16,false);
    for (int i=0;i<8;++i){ out[i]   = ((s8>>i)&1)!=0; }
    for (int i=0;i<8;++i){ out[8+i] = ((s9>>i)&1)!=0; }
}
void BcaRS8Upper::unpack16(const std::vector<bool>& in, uint8_t& s8, uint8_t& s9){
    s8=0; s9=0; const int n=(int)in.size();
    for (int i=0;i<8 && i<n; ++i)    if (in[i])     s8 |= (1u<<i);
    for (int i=0;i<8 && (8+i)<n; ++i)if (in[8+i])   s9 |= (1u<<i);
}

void BcaRS8Upper::syndromes(const uint8_t sym[10], uint8_t& S0, uint8_t& S1) const{
    // parity symbols are sym[8], sym[9] (systematic)
    // parity check: S0 = sum s0..s8,  S1 = sum a^i*s_i (i=0..7) XOR s9
    S0 = 0; for (int i=0;i<=8;++i) S0 ^= sym[i];
    S1 = sym[9];
    for (int i=0;i<8;++i) S1 ^= gf_mul(gf_exp_[i], sym[i]);
}

std::vector<bool> BcaRS8Upper::encode(const BitBlock256& data) const{
    // extract 8 symbols (upper-8 of each lane)
    uint8_t s[10]={0};
    for (int i=0;i<8;++i){
        uint32_t lane = get_lane_u32(data, i);
        s[i] = uint8_t((lane >> 24) & 0xFF);
    }
    // systematic parity
    uint8_t p8=0; for (int i=0;i<8;++i) p8 ^= s[i];
    uint8_t p9=0; for (int i=0;i<8;++i) p9 ^= gf_mul(gf_exp_[i], s[i]);
    s[8]=p8; s[9]=p9;

    std::vector<bool> out;
    pack16(out, p8, p9);
    return out;
}

ECCResult BcaRS8Upper::decode(const BitBlock256& data_err,
                              const std::vector<bool>& parity) const{
    // get received upper8 symbols, append provided parity
    uint8_t sym[10]={0};
    for (int i=0;i<8;++i){
        uint32_t lane = get_lane_u32(data_err, i);
        sym[i] = uint8_t((lane >> 24) & 0xFF);
    }
    uint8_t p8=0,p9=0;
    unpack16(parity, p8, p9);
    sym[8]=p8; sym[9]=p9;

    // compute syndromes
    uint8_t S0,S1; syndromes(sym, S0, S1);
    if (S0==0 && S1==0){
        return {ECCStatus::Clean, data_err};
    }

    // SSC: try parity-only errors first
    BitBlock256 corrected = data_err;
    if (S0!=0 && S1==0){
        // error at s8 by value S0
        // nothing to change in data bits (parity-only); treat as Corrected
        return {ECCStatus::Corrected, corrected};
    }
    if (S0==0 && S1!=0){
        // error at s9 by value S0
        return {ECCStatus::Corrected, corrected};
    }
    if (S0!=0){
        uint8_t ratio = gf_div(S1, S0); // should be a^j
        if (ratio!=0){
            int j = gf_log_[ratio];
            if (0 <= j && j < 8){
                // flip upper8 at lane j by amount S0 (GF add = XOR)
                uint8_t cur = sym[j];
                uint8_t nxt = uint8_t(cur ^ S0);
                set_lane_upper8(corrected, j, nxt);
                // verify
                uint8_t tS0,tS1;
                uint8_t temp_sym[10];
                for (int i=0;i<10;++i) temp_sym[i]=sym[i];
                temp_sym[j]=nxt;
                syndromes(temp_sym, tS0, tS1);
                if (tS0==0 && tS1==0){
                    return {ECCStatus::Corrected, corrected};
                } else {
                    // miscorrection attempt
                    return {ECCStatus::UndetectedError, corrected};
                }
            }
        }
    }
    return {ECCStatus::DetectedUncorrectable, data_err};
}
