"""
Internal Model Control (IMC) based PID tuning.

IMC gives a single tuning parameter lambda (λ) — the desired closed-loop
time constant.  Larger λ = slower, more robust response.
A common guideline: lambda >= max(0.1*tau, 0.8*L).
"""

from __future__ import annotations

from .base import ProcessParams, TuningResult


def imc_tune(
    params: ProcessParams,
    lambda_: float | None = None,
    robustness: str = "moderate",
) -> TuningResult:
    """
    IMC-based PID tuning for FOPDT processes.

    Parameters
    ----------
    params:     FOPDT process identification results.
    lambda_:    Desired closed-loop time constant [s].
                If None, auto-selected based on `robustness`.
    robustness: "aggressive", "moderate", or "conservative".
                Only used when lambda_ is None.

    Notes
    -----
    The IMC filter time constant λ is the sole design parameter.
    This maps to standard PID gains via:
        Kp = (2*tau + L) / (2*K*(lambda_ + L))  [approximation for small L]
    """
    k, tau, dead_time = params.K, params.tau, params.L

    if dead_time == 0:
        dead_time = 1e-6

    # Auto-select lambda from robustness level
    if lambda_ is None:
        factors = {"aggressive": 0.2, "moderate": 0.5, "conservative": 1.0}
        factor = factors.get(robustness, 0.5)
        lambda_ = max(factor * tau, 0.8 * dead_time)

    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")

    # IMC → PID conversion (Skogestad's simplified IMC)
    kp = (2 * tau + dead_time) / (2 * k * (lambda_ + dead_time / 2))
    ti = tau + dead_time / 2
    td = tau * dead_time / (2 * tau + dead_time)

    ki = kp / ti
    kd = kp * td

    return TuningResult(
        method="IMC (Internal Model Control)",
        kp=kp,
        ki=ki,
        kd=kd,
        notes=(
            f"lambda={lambda_:.3f}s ({robustness}). "
            "IMC is robust and model-based. Tune lambda to trade speed vs. robustness."
        ),
    )
