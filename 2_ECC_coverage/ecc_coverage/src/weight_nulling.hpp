#pragma once
#include <vector>
#include <cstdint>
#include "ecc/base.hpp"
#include "util/bitblock256.hpp"

struct WeightNulling16 : public ECCScheme {
    const char* name() const noexcept override { return "WeightNulling"; }

    std::vector<bool> encode(const BitBlock256& d) const override;
    ECCResult decode(const BitBlock256& e, const std::vector<bool>& parity) const override;
};
