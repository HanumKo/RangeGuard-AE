from .none import ECCNone
from .secded import ECCSECDED
from .nulling import ECCNullingBF16
from .vapi import ECCVAPI_DEC

# Real Versions
from .rangeguard_bf16_ssc import ECCRangeGuardSSC
from .rangeguard_bf16_dsc import ECCRangeGuardDSC

# Fast Versions
from .rangeguard_bf16_ssc_fast import ECCRangeGuardSSCFast
from .rangeguard_bf16_dsc_fast import ECCRangeGuardDSCFast

_SCHEME_REGISTRY = {
    "SECDED": ECCSECDED,
    "NULLING_BF16": ECCNullingBF16,
    "VAPI": ECCVAPI_DEC,
    "RANGEGUARD_BF16_SSC": ECCRangeGuardSSC,
    "RANGEGUARD_BF16_DSC": ECCRangeGuardDSC,
    "RANGEGUARD_BF16_SSC_FAST": ECCRangeGuardSSCFast,
    "RANGEGUARD_BF16_DSC_FAST": ECCRangeGuardDSCFast,
}


def get_scheme(name: str, device):
    scheme_cls = _SCHEME_REGISTRY.get(name.upper(), ECCNone)
    return scheme_cls(device)
