#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <getopt.h>
#include <memory>
#include <fstream>
#include <random>
#include <algorithm>

#include "ecc/base.hpp"
#include "patterns.hpp"
#include "secded.hpp"
#include "hbm3_crc_ssc.hpp"
#include "weight_nulling.hpp"
#include "vapi_dec78.hpp"
#include "rs_10_8.hpp"
#include "rs_12_8.hpp"
#include "util/bitblock256.hpp"

struct Counters {
    uint64_t trials = 0;
    uint64_t bits   = 0;
    uint64_t ce     = 0;
    uint64_t due    = 0;
    uint64_t sdc    = 0;

    inline void add_CE(int k){ trials++; bits += (uint64_t)k; ce++; }
    inline void add_DUE(int k){ trials++; bits += (uint64_t)k; due++; }
    inline void add_SDC(int k){ trials++; bits += (uint64_t)k; sdc++; }
};

static BitBlock256 random_block(std::mt19937_64& gen){
    BitBlock256 b{};
    std::uniform_int_distribution<uint64_t> d;
    b.w[0] = d(gen); b.w[1] = d(gen); b.w[2] = d(gen); b.w[3] = d(gen);
    return b;
}

// ---------- CSV helpers ----------
static inline std::string csv_quote(const std::string& s){
    bool need = s.find_first_of(",\n\" ") != std::string::npos;
    if (!need) return s;
    std::string t = "\"";
    for (char c: s) t += (c=='"' ? "\"\"" : std::string(1,c));
    t += "\"";
    return t;
}
static inline void csv_write_header(std::ofstream& ofs){
    ofs << "ECC,Pattern,CE,DUE,SDC,Trials\n";
}
static inline void csv_write_row(std::ofstream& ofs,
                                 const std::string& ecc,
                                 const std::string& pat,
                                 uint64_t ce, uint64_t due, uint64_t sdc, uint64_t trials){
    ofs << csv_quote(ecc) << ','
        << csv_quote(pat) << ','
        << ce << ',' << due << ',' << sdc << ',' << trials << '\n';
}

// ---------- helpers for RS CE policy (upper-n bits only) ----------
static inline uint32_t get_lane_u32(const BitBlock256& b, int lane){
    // lane: 0..7, 각 32b 슬라이스
    const int bit0 = lane * 32;
    const int wi  = bit0 >> 6;   // 0..3
    const int off = bit0 & 63;   // 0 또는 32
    if (off == 0){
        const uint64_t lo = b.w[wi];
        return uint32_t(lo & 0xFFFFFFFFull);
    } else {
        const uint64_t lo = b.w[wi] >> off;
        const uint64_t hi = (wi+1 < 4) ? (b.w[wi+1] << (64 - off)) : 0ull;
        return uint32_t((lo | hi) & 0xFFFFFFFFull);
    }
}

static inline bool upper_n_equal(const BitBlock256& a, const BitBlock256& b, int n){
    // n ∈ {4,8}. 각 32b lane의 상위 n비트만 비교
    const uint32_t mask = (n == 32) ? 0xFFFFFFFFu
                        : ((n == 0) ? 0u : ((~0u) << (32 - n)));
    for (int lane = 0; lane < 8; ++lane){
        uint32_t ua = get_lane_u32(a, lane) & mask;
        uint32_t ub = get_lane_u32(b, lane) & mask;
        if (ua != ub) return false;
    }
    return true;
}

static void print_usage(const char* argv0){
    fprintf(stderr,
        "Usage: %s [--trials N] [--seed S] [--patterns NAME|all] [--csv PATH]\n"
        "  patterns: all | SE | DAE | 16E | 32E | FC | SE+SE | SE+DAE | SE+16E | SE+32E | 16E+16E | 32E+32E\n"
        "            aliases also accepted: SWL16 SWD32 FE ALL256 SE+SWL SE+SWD SWL+SWL SWD+SWD\n",
        argv0);
}

int main(int argc, char** argv){
    uint64_t trials = 1000000;
    uint64_t seed = 123;
    std::string which_patterns = "all";
    std::string csv_path; // empty -> no csv

    static struct option long_opts[] = {
        {"trials",   required_argument, 0, 'n'},
        {"seed",     required_argument, 0, 's'},
        {"patterns", required_argument, 0, 'p'}, // all|SE|DAE|SWL16|SWD32|...
        {"csv",      required_argument, 0, 'c'},
        {0,0,0,0}
    };
    int c;
    while ((c=getopt_long(argc, argv, "n:s:p:c:", long_opts, nullptr)) != -1){
        if (c=='n') trials = std::strtoull(optarg, nullptr, 10);
        else if (c=='s') seed = std::strtoull(optarg, nullptr, 10);
        else if (c=='p') which_patterns = optarg;
        else if (c=='c') csv_path = optarg;
        else { print_usage(argv[0]); return 1; }
    }

    // ECC registry
    std::vector<std::unique_ptr<ECCScheme>> eccs;
    eccs.emplace_back(new SecDedHamming256());
    eccs.emplace_back(new Hbm3Crc16Ssc());
    eccs.emplace_back(new WeightNulling16());
    eccs.emplace_back(new VapiDEC64x4());
    eccs.emplace_back(new BcaRS8Upper()); // RS(10,8) upper-8 policy
    eccs.emplace_back(new BcaRS4Upper()); // RS(12,8) upper-4 policy

    // Pattern registry
    std::vector<std::string> patterns = {
        "SE","DAE","16E","32E","FC",
        "SE+SE","SE+DAE","SE+16E","SE+32E","16E+16E","32E+32E"
    };

    if (which_patterns != "all"){
        patterns = { which_patterns };
    }

    std::mt19937_64 gen(seed);
    PatternRNG prng(seed);

    // CSV open (optional)
    std::ofstream csv_ofs;
    if (!csv_path.empty()){
        csv_ofs.open(csv_path, std::ios::out | std::ios::trunc);
        if (!csv_ofs){
            fprintf(stderr, "Failed to open CSV path: %s\n", csv_path.c_str());
            return 3;
        }
        csv_write_header(csv_ofs);
    }

    printf("\n=== ECC Coverage (trials per pattern = %llu) ===\n", (unsigned long long)trials);
    printf("%-16s %-12s %10s %10s %10s %10s\n", "ECC", "Pattern", "CE", "DUE", "SDC", "Trials");
    printf("%s\n", std::string(16+1+12+1+10*4+4, '-').c_str());

    for (auto& ecc_ptr : eccs){
        const char* ecc_name = ecc_ptr->name();
        for (const auto& pat : patterns){
            Counters ctr;
            for (uint64_t t=0; t<trials; ++t){
                BitBlock256 d = random_block(gen);
                auto parity = ecc_ptr->encode(d);

                std::vector<int> flips;
                if      (pat=="SE")         flips = sample_SE(prng);
                else if (pat=="DAE")        flips = sample_DAE(prng);
                else if (pat=="16E" || pat=="SWL16")      flips = sample_SWL16(prng);
                else if (pat=="32E" || pat=="SWD32")      flips = sample_SWD32(prng);
                else if (pat=="FC" || pat=="FE" || pat=="ALL256")      flips = sample_ALL256_half(prng);
                // disjoint composites
                else if (pat=="SE+SE")      flips = sample_SE_plus_SE_disjoint(prng);
                else if (pat=="SE+DAE")     flips = sample_SE_plus_DAE_disjoint(prng);
                else if (pat=="SE+16E" || pat=="SE+SWL")  flips = sample_SE_plus_SWL_disjoint(prng);
                else if (pat=="SE+32E" || pat=="SE+SWD")  flips = sample_SE_plus_SWD_disjoint(prng);
                else if (pat=="16E+16E" || pat=="SWL+SWL") flips = sample_SWL_plus_SWL_disjoint(prng);
                else if (pat=="32E+32E" || pat=="SWD+SWD") flips = sample_SWD_plus_SWD_disjoint(prng);
                else { fprintf(stderr,"Unknown pattern %s\n", pat.c_str()); return 2; }

                // 주입
                BitBlock256 e = d;
                for (int ix : flips) e.flip(ix);
                BitBlock256 v1 = e;

                // 디코드
                ECCResult r = ecc_ptr->decode(e, parity);

                auto is_weight_nulling =
                    (std::string(ecc_ptr->name()).rfind("WeightNulling", 0) == 0);

                const int kflips = (int)flips.size();

                if (is_weight_nulling) {
                    auto chunk_is_zero = [](const BitBlock256& b, int c)->bool {
                        int start = c * 16;
                        for (int j = 0; j < 16; ++j) {
                            int bi = start + j;
                            int wi = bi >> 6;
                            int bj = bi & 63;
                            if ((b.w[wi] >> bj) & 1ull) return false;
                        }
                        return true;
                    };

                    bool affected[16] = {false};
                    for (int ix : flips) affected[ix / 16] = true;

                    const bool detected =
                        (r.status == ECCStatus::Corrected) ||
                        (r.status == ECCStatus::DetectedUncorrectable);

                    bool any_affected = false;
                    bool all_policy_zero = true;
                    for (int c = 0; c < 16; ++c) {
                        if (!affected[c]) continue;
                        any_affected = true;
                        const bool before_zero = chunk_is_zero(v1, c);
                        const bool after_zero  = chunk_is_zero(r.corrected, c);
                        const bool policy_zero = after_zero;
                        if (!policy_zero) { all_policy_zero = false; break; }
                    }

                    if (detected && any_affected && all_policy_zero) ctr.add_DUE(kflips);
                    else                                             ctr.add_SDC(kflips);

                } else {
                    const std::string nm = ecc_ptr->name();
                    const bool detected =
                        (r.status == ECCStatus::Corrected) ||
                        (r.status == ECCStatus::DetectedUncorrectable);

                    if (nm.rfind("RangeGuard DSC", 0) == 0) {
                        const bool same_upper = upper_n_equal(r.corrected, d, 4);
                        if (same_upper)        ctr.add_CE(kflips);
                        else if (detected)     ctr.add_DUE(kflips);
                        else                   ctr.add_SDC(kflips);

                    } else if (nm.rfind("RangeGuard SSC", 0) == 0) {
                        const bool same_upper = upper_n_equal(r.corrected, d, 8);
                        if (same_upper)        ctr.add_CE(kflips);
                        else if (detected)     ctr.add_DUE(kflips);
                        else                   ctr.add_SDC(kflips);

                    } else {
                        const bool same =
                            (r.corrected.w[0]==d.w[0])&&(r.corrected.w[1]==d.w[1])&&
                            (r.corrected.w[2]==d.w[2])&&(r.corrected.w[3]==d.w[3]);

                        if (same)              ctr.add_CE(kflips);
                        else if (detected)     ctr.add_DUE(kflips);
                        else                   ctr.add_SDC(kflips);
                    }
                }
            }

            printf("%-16s %-12s %10llu %10llu %10llu %10llu\n",
                   ecc_name, pat.c_str(),
                   (unsigned long long)ctr.ce,
                   (unsigned long long)ctr.due,
                   (unsigned long long)ctr.sdc,
                   (unsigned long long)ctr.trials);

            if (csv_ofs.is_open()){
                csv_write_row(csv_ofs, ecc_name, pat, ctr.ce, ctr.due, ctr.sdc, ctr.trials);
            }
        }
    }
    if (csv_ofs.is_open()) csv_ofs.close();

    printf("\n");
    return 0;
}
