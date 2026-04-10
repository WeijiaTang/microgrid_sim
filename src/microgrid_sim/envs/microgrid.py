"""Legacy environment names mapped onto the new network-first environment."""

from __future__ import annotations

from dataclasses import replace

from ..cases import CIGREEuropeanLVConfig, NetworkCaseConfig
from .network_microgrid import NetworkMicrogridEnv, NetworkMicrogridEnvSimple, NetworkMicrogridEnvThevenin


class MicrogridEnv(NetworkMicrogridEnv):
    """Compatibility alias for the default network-first environment."""

    def __init__(self, config: NetworkCaseConfig | None = None, render_mode: str | None = None):
        del render_mode
        super().__init__(config or CIGREEuropeanLVConfig())


class MicrogridEnvThevenin(NetworkMicrogridEnvThevenin):
    """Compatibility alias for the Thevenin-based network environment."""

    def __init__(self, config: NetworkCaseConfig | None = None, render_mode: str | None = None):
        del render_mode
        cfg = replace(config, battery_model="thevenin") if config is not None else CIGREEuropeanLVConfig(battery_model="thevenin")
        super().__init__(cfg)


class MicrogridEnvSimple(NetworkMicrogridEnvSimple):
    """Compatibility alias for the simple-battery network environment."""

    def __init__(self, config: NetworkCaseConfig | None = None, render_mode: str | None = None):
        del render_mode
        cfg = replace(config, battery_model="simple") if config is not None else CIGREEuropeanLVConfig(battery_model="simple")
        super().__init__(cfg)
