"""mflux backend registry.

In v0.2 this is still a simple dict — task #3 will refactor to a dataclass
registry so adding a new backend collapses to one row and the three
`if backend == "flux"` branches in cli.cmd_generate disappear.
"""
from __future__ import annotations

BACKENDS = {
    "flux": "mflux-generate-kontext",
    "qwen": "mflux-generate-qwen-edit",
}
