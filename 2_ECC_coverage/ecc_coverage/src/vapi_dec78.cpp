#include "vapi_dec78.hpp"
#include <cassert>

// --------- small GF(2) helpers ----------
static inline int popcount16(uint16_t x){ return __builtin_popcount((unsigned)x); }

static int gf2_rank(std::vector<uint16_t> A){
    int r=0;
    for (int bit=13; bit>=0; --bit){
        int piv=-1;
        for (int i=r;i<(int)A.size();++i) if ((A[i]>>bit)&1u){piv=i;break;}
        if (piv<0) continue;
        std::swap(A[r],A[piv]);
        for (int i=0;i<(int)A.size();++i) if (i!=r && ((A[i]>>bit)&1u)) A[i]^=A[r];
        if (++r==14) break;
    }
    return r;
}
static std::array<uint16_t,14> invert14(const std::array<uint16_t,14>& M){
    std::array<uint16_t,14> L=M, R{};
    for (int i=0;i<14;++i) R[i]=(1u<<i);
    for (int col=0; col<14; ++col){
        int piv=-1;
        for (int r=col;r<14;++r) if ((L[r]>>col)&1u){piv=r;break;}
        if (piv<0) throw std::runtime_error("invert14: singular");
        if (piv!=col){ std::swap(L[piv],L[col]); std::swap(R[piv],R[col]); }
        for (int r=0;r<14;++r) if (r!=col && ((L[r]>>col)&1u)){ L[r]^=L[col]; R[r]^=R[col]; }
    }
    return R;
}

// --------- DEC78 impl ----------
std::vector<uint16_t>
VapiDEC64x4::DEC78::build_H_unique_pairs(int n, int m, uint64_t seed){
    std::mt19937_64 rng(seed);
    auto rand_col=[&](){ uint16_t c=0; do{ c=(uint16_t)(rng() & ((1u<<m)-1)); }while(!c); return c; };
    std::vector<uint16_t> cols; cols.reserve(n);
    std::unordered_map<uint16_t,char> single, pairs;
    size_t trials=0;
    while ((int)cols.size()<n){
        if (++trials>200000) throw std::runtime_error("build_H_unique_pairs: try another seed");
        uint16_t c=rand_col();
        if (single.count(c) || pairs.count(c)) continue;
        bool ok=true;
        for (auto s: cols){
            uint16_t x=(uint16_t)(c^s);
            if (pairs.count(x) || single.count(x)){ ok=false; break; }
        }
        if (!ok) continue;
        for (auto s: cols) pairs[(uint16_t)(c^s)] = 1;
        cols.push_back(c); single[c]=1;
    }
    return cols;
}

void VapiDEC64x4::DEC78::to_systematic(std::vector<uint16_t>& Hcols){
    const int n=78, m=14;
    // parity block 14 independent columns selection
    std::vector<uint16_t> basis; std::vector<int> pidx;
    for (int idx=n-1; idx>=0; --idx){
        auto tmp=basis; tmp.push_back(Hcols[idx]);
        if (gf2_rank(tmp)>(int)basis.size()){
            basis.push_back(Hcols[idx]); pidx.push_back(idx);
            if ((int)basis.size()==m) break;
        }
    }
    if ((int)pidx.size()!=m){
        basis.clear(); pidx.clear();
        for (int idx=0; idx<n; ++idx){
            auto tmp=basis; tmp.push_back(Hcols[idx]);
            if (gf2_rank(tmp)>(int)basis.size()){
                basis.push_back(Hcols[idx]); pidx.push_back(idx);
                if ((int)basis.size()==m) break;
            }
        }
    }
    if ((int)pidx.size()!=m) throw std::runtime_error("to_systematic: cannot find 14 independent columns");
    std::sort(pidx.begin(), pidx.end());
    std::vector<int> didx; didx.reserve(n-m);
    for (int i=0;i<n;++i) if (!std::binary_search(pidx.begin(),pidx.end(),i)) didx.push_back(i);
    std::vector<int> perm = didx; perm.insert(perm.end(), pidx.begin(), pidx.end());
    std::vector<uint16_t> Hperm(n);
    for (int i=0;i<n;++i) Hperm[i]=Hcols[perm[i]];

    // make last 14 columns identity by left-multiplying T
    std::array<uint16_t,14> M{};
    for (int r=0;r<14;++r){
        uint16_t row=0;
        for (int j=0;j<14;++j) if ((Hperm[n-14+j]>>r)&1u) row|=(1u<<j);
        M[r]=row;
    }
    auto T = invert14(M);
    std::vector<uint16_t> Hprime(n);
    for (int c=0;c<n;++c){
        uint16_t col=0;
        for (int r=0;r<14;++r){
            if (popcount16((uint16_t)(T[r] & Hperm[c])) & 1) col |= (1u<<r);
        }
        Hprime[c]=col;
    }
    for (int j=0;j<14;++j) if (Hprime[n-14+j] != (uint16_t)(1u<<j))
        throw std::runtime_error("to_systematic: parity block not identity");
    Hcols.swap(Hprime);
}

static inline uint16_t H_times_vec(const std::vector<uint16_t>& Hcols, uint64_t lo, uint16_t hi){
    uint16_t s=0;
    for (int i=0;i<78;++i){
        bool bit = (i<64) ? ((lo>>i)&1ull) : ((hi>>(i-64))&1u);
        if (bit) s ^= Hcols[i];
    }
    return s;
}
uint16_t VapiDEC64x4::DEC78::syndrome(const std::vector<uint16_t>& Hcols, uint64_t lo, uint16_t hi){
    return H_times_vec(Hcols, lo, hi);
}
VapiDEC64x4::Code78
VapiDEC64x4::DEC78::encode(uint64_t data) const {
    // parity = A * data
    uint16_t p=0; uint64_t x=data; int i=0;
    while (x){ if (x & 1ull) p ^= Acols[i]; x >>= 1ull; ++i; }
    return {data, p};
}
std::tuple<VapiDEC64x4::Code78, int>
VapiDEC64x4::DEC78::decode(const Code78& r) const {
    uint16_t s = syndrome(Hcols, r.data, r.par);
    if (!s) return {r, 0};
    auto it1 = single.find(s);
    if (it1 != single.end()){
        Code78 c=r; flip_bit(c, it1->second);
        return {c, 1};
    }
    auto it2 = pairs.find(s);
    if (it2 != pairs.end()){
        Code78 c=r;
        int i=(it2->second>>8)&0xFF, j=(it2->second)&0xFF;
        flip_bit(c,i); flip_bit(c,j);
        return {c, 2};
    }
    return {r, -1}; // not expected for <=2-bit
}

// --------- VapiDEC64x4 public ----------
VapiDEC64x4::VapiDEC64x4(uint64_t seed){
    // H'=[A|I] 구성
    dec_.Hcols = DEC78::build_H_unique_pairs(78,14,seed);
    DEC78::to_systematic(dec_.Hcols);
    dec_.Acols.assign(dec_.Hcols.begin(), dec_.Hcols.begin()+64);
    // syndrome maps
    for (int i=0;i<78;++i) dec_.single[dec_.Hcols[i]] = i;
    for (int i=0;i<78;++i){
        uint16_t ci = dec_.Hcols[i];
        for (int j=i+1;j<78;++j){
            uint16_t s = (uint16_t)(ci ^ dec_.Hcols[j]);
            if (dec_.single.count(s) || dec_.pairs.count(s))
                throw std::runtime_error("VAPI-DEC78: syndrome collision; change seed");
            dec_.pairs[s] = (uint16_t)((i<<8)|j);
        }
    }
}

std::vector<bool> VapiDEC64x4::encode(const BitBlock256& d) const {
    // 4개 64비트 조각 → parity 14b씩 = 56비트 반환
    std::vector<bool> out; out.reserve(56);
    for (int seg=0; seg<4; ++seg){
        uint64_t data = d.w[seg];
        auto cw = dec_.encode(data);
        // parity 14비트 LSB-first로 push
        for (int b=0;b<14;++b) out.push_back( (cw.par>>b) & 1u );
    }
    return out;
}

ECCResult VapiDEC64x4::decode(const BitBlock256& noisy, const std::vector<bool>& parity) const {
    ECCResult r; r.corrected = noisy; r.status = ECCStatus::UndetectedError;

    if ((int)parity.size() != 56){
        r.status = ECCStatus::DetectedUncorrectable;
        return r;
    }

    bool any_detected = false;
    bool any_corrected = false;
    bool any_due = false;

    for (int seg=0; seg<4; ++seg){
        // parity 14비트 추출
        uint16_t p=0;
        for (int b=0;b<14;++b){
            if (parity[seg*14 + b]) p |= (uint16_t)(1u<<b);
        }
        VapiDEC64x4::Code78 recv{ noisy.w[seg], p };
        auto [corr, st] = dec_.decode(recv);

        if (st != 0) any_detected = true;
        if (st > 0)  any_corrected = true;
        if (st < 0)  any_due = true;

        // 데이터 비트 수정 반영
        if (corr.data != noisy.w[seg]){
            // 어떤 위치가 바뀌었는지 모를 필요 없이 통째로 교체
            r.corrected.w[seg] = corr.data;
        }
        // (패리티 비트는 외부 저장이므로, 여기서 corr.par는 기록용일 뿐)
        if (st < 0) r.status = ECCStatus::DetectedUncorrectable;
    }

    if (any_due) {
        r.status = ECCStatus::DetectedUncorrectable;
    } else if (any_corrected) {
        r.status = ECCStatus::Corrected;
    } else if (any_detected && r.status != ECCStatus::DetectedUncorrectable) {
        r.status = ECCStatus::Corrected;
    }
    return r;
}
