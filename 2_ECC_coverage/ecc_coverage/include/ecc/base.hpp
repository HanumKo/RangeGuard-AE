
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "util/bitblock256.hpp"

enum class ECCStatus {
    Clean,
    Corrected,                 // CE
    DetectedUncorrectable,     // DUE
    UndetectedError            // SDC
};

struct ECCResult {
    ECCStatus status;
    BitBlock256 corrected; // may equal input if no data correction
};

class ECCScheme {
public:
    virtual ~ECCScheme() = default;
    virtual const char* name() const = 0;
    virtual std::vector<bool> encode(const BitBlock256& data) const = 0;
    virtual ECCResult decode(const BitBlock256& data_err, const std::vector<bool>& parity) const = 0;
};
