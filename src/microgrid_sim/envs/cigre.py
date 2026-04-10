"""Legacy CIGRE wrapper mapped onto the network-first environment."""

from __future__ import annotations

from dataclasses import replace

from ..cases import CIGREEuropeanLVConfig
from .network_microgrid import NetworkMicrogridEnv


class CIGREMicrogridEnv(NetworkMicrogridEnv):
    """Compatibility wrapper for the new CIGRE European LV network environment."""

    def __init__(self, config: CIGREEuropeanLVConfig | None = None, battery_model: str = "thevenin", **kwargs):
        del kwargs
        cfg = config or CIGREEuropeanLVConfig()
        cfg = replace(cfg, battery_model=battery_model)
        super().__init__(cfg)
