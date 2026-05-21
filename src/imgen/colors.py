"""Terminal colors + leveled print helpers.

Colors are auto-disabled when stdout is not a tty (e.g. piped to a file).
"""
from __future__ import annotations

import sys

_USE_COLOR = sys.stdout.isatty()


class C:
    OK = "\033[92m" if _USE_COLOR else ""
    WARN = "\033[93m" if _USE_COLOR else ""
    ERR = "\033[91m" if _USE_COLOR else ""
    INFO = "\033[94m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""
    END = "\033[0m" if _USE_COLOR else ""


def ok(msg: str) -> None:
    print(f"{C.OK}✅{C.END} {msg}")


def warn(msg: str) -> None:
    print(f"{C.WARN}⚠️ {C.END} {msg}")


def err(msg: str) -> None:
    print(f"{C.ERR}❌{C.END} {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"{C.INFO}🔍{C.END} {msg}")


def step(msg: str) -> None:
    print(f"{C.BOLD}{C.INFO}🚀 {msg}{C.END}")


def dim(msg: str) -> None:
    print(f"{C.DIM}{msg}{C.END}")


def die(msg: str, code: int = 1, hint: str | None = None) -> None:
    err(msg)
    if hint:
        print(f"   {C.DIM}{hint}{C.END}", file=sys.stderr)
    sys.exit(code)
