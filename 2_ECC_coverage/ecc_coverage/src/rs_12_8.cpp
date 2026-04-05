#include "rs_12_8.hpp"
#include <cassert>
#include <algorithm>

BcaRS4Upper::BcaRS4Upper(){ gf_init(); init_X(); }

void BcaRS4Upper::gf_init(){
    const uint8_t prim = 0x13; // x^4 + x + 1
    uint8_t x=1;
    for (int i=0;i<15;++i){ gf_exp_[i]=x; gf_log_[x]=i; x<<=1; if (x&0x10) x^=prim; }
    gf_exp_[15]=1;
    for (int i=16;i<32;++i) gf_exp_[i]=gf_exp_[i-15];
    gf_log_[0]=-1;
}
void BcaRS4Upper::init_X(){
    for (int i=0;i<12;++i) X_[i]=gf_exp_[i]; // 1, α, α^2, ...
}
uint8_t BcaRS4Upper::gf_mul(uint8_t a, uint8_t b) const{
    if(!a||!b) return 0;
    return gf_exp_[ (gf_log_[a] + gf_log_[b]) % 15 ];
}
uint8_t BcaRS4Upper::gf_div(uint8_t a, uint8_t b) const{
    if(!a) return 0;
    if(!b) return 0;
    int idx = gf_log_[a] - gf_log_[b]; if (idx<0) idx+=15; return gf_exp_[idx];
}
uint8_t BcaRS4Upper::pow_field(uint8_t base, int j) const{
    if (j==0) return 1;
    return gf_exp_[ (gf_log_[base]*j) % 15 ];
}

uint32_t BcaRS4Upper::get_lane_u32(const BitBlock256& b, int lane){
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

uint8_t BcaRS4Upper::get_lane_upper4(const BitBlock256& b, int lane){
    uint32_t v = get_lane_u32(b, lane);
    return uint8_t( (v >> 28) & 0xF );
}
void BcaRS4Upper::set_lane_upper4(BitBlock256& b, int lane, uint8_t up4){
    // write bits [lane*32+28 .. lane*32+31]
    const int base = lane*32 + 28;
    for (int k=0;k<4;++k){
        int bi = base + k;
        int wi = bi >> 6, bo = bi & 63;
        uint64_t mask = (1ull<<bo);
        if ( (up4 >> k) & 1 ) b.w[wi] |= mask; else b.w[wi] &= ~mask;
    }
}

void BcaRS4Upper::pack16(std::vector<bool>& out, const uint8_t p[4]){
    out.assign(16,false);
    for (int j=0;j<4;++j)
        for (int i=0;i<4;++i)
            out[j*4 + i] = ((p[j]>>i)&1)!=0;
}
void BcaRS4Upper::unpack16(const std::vector<bool>& in, uint8_t p[4]){
    for (int j=0;j<4;++j){ p[j]=0; for (int i=0;i<4;++i){
        int k=j*4+i; if (k<(int)in.size() && in[k]) p[j]|=(1u<<i);
    } }
}

void BcaRS4Upper::syndromes(const uint8_t sym[12], uint8_t S[4]) const{
    for (int j=0;j<4;++j){
        uint8_t Sj=0;
        for (int i=0;i<12;++i) Sj ^= gf_mul(G(j,i), sym[i]);
        S[j]=Sj;
    }
}

bool BcaRS4Upper::solve2(const BcaRS4Upper* self,
                         uint8_t a,uint8_t b,uint8_t c,uint8_t d,
                         uint8_t u,uint8_t v, uint8_t& x,uint8_t& y){
    uint8_t ad  = self->gf_mul(a,d);
    uint8_t bc  = self->gf_mul(b,c);
    uint8_t det = ad ^ bc;                 // ad - bc (char-2)
    if (det == 0) return false;
    uint8_t inv = self->gf_div(1, det);
    x = self->gf_mul(inv, uint8_t(self->gf_mul(u,d) ^ self->gf_mul(b,v)));
    y = self->gf_mul(inv, uint8_t(self->gf_mul(a,v) ^ self->gf_mul(u,c)));
    return true;
}

bool BcaRS4Upper::try_fix_1(uint8_t sym[12], const uint8_t S[4], int& pos) const{
    pos=-1;
    for (int p=0;p<12;++p){
        uint8_t e=0; bool have=false, ok=true;
        for (int j=0;j<4;++j){
            uint8_t g = G(j,p);
            if (g==0){ if (S[j]!=0){ ok=false; break; } }
            else{
                uint8_t ej = gf_div(S[j], g);
                if (!have){ e=ej; have=true; }
                else if (ej!=e){ ok=false; break; }
            }
        }
        if (ok && have && e!=0){
            sym[p] ^= e; pos=p; return true;
        }
    }
    return false;
}

bool BcaRS4Upper::try_fix_2(uint8_t sym[12], const uint8_t S[4], int& p0, int& p1) const{
    uint8_t col[12][4];
    for (int p=0;p<12;++p) for (int j=0;j<4;++j) col[p][j]=G(j,p);

    for (int a=0;a<12;++a) for (int b=a+1;b<12;++b){
        for (int j0=0;j0<4;++j0) for (int j1=j0+1;j1<4;++j1){
            uint8_t e0,e1;
            if (!BcaRS4Upper::solve2(this,
                    col[a][j0],col[b][j0], col[a][j1],col[b][j1],
                    S[j0],S[j1], e0,e1)) continue;
            if (e0==0 && e1==0) continue;
            bool ok=true;
            for (int j=0;j<4 && ok;++j){
                uint8_t lhs = (uint8_t)( gf_mul(col[a][j],e0) ^ gf_mul(col[b][j],e1) );
                if (lhs != S[j]) ok=false;
            }
            if (!ok) continue;
            sym[a]^=e0; sym[b]^=e1; p0=a; p1=b; return true;
        }
    }
    return false;
}

std::vector<bool> BcaRS4Upper::encode(const BitBlock256& data) const{
    // 8 data symbols (upper4 of 8 lanes) + 4 parity = 12
    uint8_t sym[12]={0};
    for (int i=0;i<8;++i) sym[i] = get_lane_upper4(data,i);

    // build b[j] = sum_{i<8} X[i]^j * s_i
    uint8_t b[4]={0,0,0,0};
    for (int j=0;j<4;++j){
        uint8_t bj=0;
        for (int i=0;i<8;++i) bj ^= gf_mul(G(j,i), sym[i]);
        b[j]=bj;
    }
    // A[j,k] = X[8+k]^j
    uint8_t A[4][4];
    for (int j=0;j<4;++j) for (int k=0;k<4;++k) A[j][k]=G(j,8+k);

    uint8_t p[4];
    solve4(A,b,p,this);
    for (int k=0;k<4;++k) sym[8+k]=p[k];

    std::vector<bool> out; pack16(out, p);
    return out;
}

void BcaRS4Upper::solve4(uint8_t A[4][4], uint8_t b[4], uint8_t out[4], const BcaRS4Upper* self){
    uint8_t M[4][5];
    for (int r=0;r<4;++r){ for (int c=0;c<4;++c) M[r][c]=A[r][c]; M[r][4]=b[r]; }
    for (int c=0;c<4;++c){
        int piv=c; for (int r=c;r<4;++r) if (M[r][c]!=0){ piv=r; break; }
        if (M[piv][c]==0) continue;
        if (piv!=c) for (int k=c;k<=4;++k) std::swap(M[c][k], M[piv][k]);
        uint8_t inv = self->gf_div(1, M[c][c]);
        for (int k=c;k<=4;++k) M[c][k] = self->gf_mul(M[c][k], inv);
        for (int r=0;r<4;++r) if (r!=c && M[r][c]!=0){
            uint8_t f=M[r][c];
            for (int k=c;k<=4;++k) M[r][k] = self->gf_add(M[r][k], self->gf_mul(f, M[c][k]));
        }
    }
    for (int i=0;i<4;++i) out[i]=M[i][4];
}

ECCResult BcaRS4Upper::decode(const BitBlock256& data_err,
                              const std::vector<bool>& parity) const{
    // assemble 12 symbols: 8 from lanes, 4 from stored parity
    uint8_t sym[12]={0};
    for (int i=0;i<8;++i) sym[i] = get_lane_upper4(data_err,i);
    uint8_t p[4]; unpack16(parity,p);
    for (int k=0;k<4;++k) sym[8+k]=p[k];

    uint8_t S[4]; syndromes(sym,S);
    bool zero=true; for (int j=0;j<4;++j) if (S[j]){ zero=false; break; }
    if (zero) return {ECCStatus::Clean, data_err};

    // try 1-symbol
    {
        uint8_t tmp[12]; std::copy(sym,sym+12,tmp);
        int pos=-1;
        if (try_fix_1(tmp,S,pos)){
            uint8_t T[4]; syndromes(tmp,T);
            bool ok=true; for (int j=0;j<4;++j) if (T[j]) ok=false;
            if (ok){
                BitBlock256 corrected = data_err;
                if (pos<8) set_lane_upper4(corrected,pos,tmp[pos]);
                return {ECCStatus::Corrected, corrected};
            }
        }
    }
    // try 2-symbol
    {
        uint8_t tmp[12]; std::copy(sym,sym+12,tmp);
        int a=-1,b=-1;
        if (try_fix_2(tmp,S,a,b)){
            uint8_t T[4]; syndromes(tmp,T);
            bool ok=true; for (int j=0;j<4;++j) if (T[j]) ok=false;
            if (ok){
                BitBlock256 corrected = data_err;
                if (a<8) set_lane_upper4(corrected,a,tmp[a]);
                if (b<8) set_lane_upper4(corrected,b,tmp[b]);
                return {ECCStatus::Corrected, corrected};
            }
        }
    }
    return {ECCStatus::DetectedUncorrectable, data_err};
}
