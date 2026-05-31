"""Session management for the Pie Kernel (V3.0).

This package provides identity bootstrap from SEED files and
persistent session storage so that Ivy can maintain state across
multiple interactions.
"""

from .bootstrap import load_seed, build_identity_snapshot
from .manager import SessionManager, SessionContext

__all__ = [
    "load_seed",
    "build_identity_snapshot",
    "SessionManager",
    "SessionContext",
]
