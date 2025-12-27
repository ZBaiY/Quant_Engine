from __future__ import annotations

def test_execution_modules_importable():
    import quant_engine.execution.engine as _  # noqa: F401
    import quant_engine.execution.matching.simulated as _  # noqa: F401
    import quant_engine.execution.router.simple as _  # noqa: F401
    import quant_engine.execution.slippage.linear as _  # noqa: F401
