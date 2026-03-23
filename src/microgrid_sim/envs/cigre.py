"""MG-CIGRE environment wrapper."""

from __future__ import annotations

from ..cases import CIGREConfig
from .microgrid import MicrogridEnv


class CIGREMicrogridEnv(MicrogridEnv):
    def __init__(self, config: CIGREConfig | None = None, battery_model: str = "thevenin", **kwargs):
        cfg = config or CIGREConfig()
        cfg.battery_model = battery_model
        super().__init__(cfg, **kwargs)
