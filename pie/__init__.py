"""Top‑level package for the Pie Kernel runtime.

This package exposes high level functions to run the kernel and validate
artifacts.  The actual command line interface is defined in
``pie.cli``.

The M0 release of the kernel focuses on deterministic, reproducible
behaviour.  All event, state and speech plan structures include a
``schema_version`` field so that later versions can validate and
migrate data as the specification evolves.
"""

__all__ = ["contracts", "llm", "runtime"]