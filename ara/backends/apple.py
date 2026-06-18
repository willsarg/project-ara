"""Apple-Silicon backend adapter — the ONLY module that imports wmx-suite.

Lazy by design: this module isn't imported unless detect picks ``"apple"``, and
the heavy ``wmx_suite`` import happens inside the call that needs it — so even on
a Mac, nothing MLX-shaped loads until ARA actually runs the engine.
"""
from __future__ import annotations


def characterize():
    """Find this machine's safe memory ceiling by delegating to wmx-suite.

    Stub for now — wired onto wmx-suite's real API when ARA's
    detect -> recommend -> run pipeline is built.
    """
    import wmx_suite  # noqa: F401  — engine pulled in only when we run

    raise NotImplementedError("characterize() lands with ARA v1's pipeline")
