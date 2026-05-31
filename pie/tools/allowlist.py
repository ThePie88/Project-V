"""Network allowlist and filesystem capability enforcement (V2.4).

Both classes implement a deny-by-default policy: anything not
explicitly allowed is denied.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set
from urllib.parse import urlparse


class NetworkAllowlist:
    """Deny-by-default network allowlist.

    Only requests whose host is in the allowlist are permitted.
    """

    def __init__(self, allowed_hosts: List[str]) -> None:
        self.allowed: Set[str] = {h.lower() for h in allowed_hosts}

    def check(self, url: str) -> bool:
        """Return True if *url*'s host is in the allowlist."""
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            return False
        return host.lower() in self.allowed


class FSCapability:
    """Filesystem capability-first access control.

    All paths must be inside ``sandbox_root`` and the requested
    operation must be in the capabilities list.
    """

    def __init__(self, sandbox_root: Path, capabilities: List[str]) -> None:
        self.sandbox_root = sandbox_root.resolve()
        self.capabilities: Set[str] = set(capabilities)

    def check(self, operation: str, path: Path) -> bool:
        """Return True if *operation* on *path* is allowed."""
        if operation not in self.capabilities:
            return False
        try:
            resolved = path.resolve()
        except Exception:
            return False
        # Path must be inside sandbox
        try:
            resolved.relative_to(self.sandbox_root)
        except ValueError:
            return False
        return True
