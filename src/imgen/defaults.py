"""Default parameter values + resource thresholds + mflux pin."""
from __future__ import annotations

__all__ = [
    "DEFAULTS",
    "HISTORY_SCHEMA_VERSION",
    "MFLUX_PIN",
    "MIN_BATTERY_PCT",
    "MIN_DISK_GB",
    "PREVIEW_OVERRIDES",
    "RAM_REQUIRED_GB",
]

DEFAULTS = {
    "style": "pixar",
    "backend": "flux",   # flux | qwen
    "quantize": 8,       # 3 4 5 6 8
    "steps": 20,
    "guidance": 3.5,
    "strength": 0.55,
    "mlx_cache_gb": 12,
    "battery_stop": 20,  # %
}

# --preview overrides (only applied if user didn't explicitly set the flag)
PREVIEW_OVERRIDES = {
    "quantize": 4,
    "steps": 8,
}

# Peak RAM (GB) required during inference: UNet weights + text encoders
# + activations + MLX cache headroom. Conservative estimates.
RAM_REQUIRED_GB = {
    ("flux", 3): 8,
    ("flux", 4): 9,
    ("flux", 5): 12,
    ("flux", 6): 14,
    ("flux", 8): 18,
    ("qwen", 3): 10,
    ("qwen", 4): 12,
    ("qwen", 5): 16,
    ("qwen", 6): 18,
    ("qwen", 8): 25,
}

MIN_DISK_GB = 5             # minimum free disk to attempt
MIN_BATTERY_PCT = 30        # below this on battery → warn (not block)

# Pin to known-working mflux version. Bump after manual verification.
MFLUX_PIN = "mflux==0.17.5"

# Bump when an entry field changes meaning. Old entries without "v" key
# are treated as v=0 and still replay (best-effort .get throughout). An
# entry with "v" > this constant is refused with a "run imgen upgrade" hint.
HISTORY_SCHEMA_VERSION = 1
