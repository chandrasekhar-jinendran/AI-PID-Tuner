"""
Cohen-Coon tuning rules.

Improves on Ziegler-Nichols for processes with large dead-time-to-time-constant
ratios (L/tau > 0.1).  Generally gives less overshoot than ZN.
"""

from __future__ import annotations

from .base import ProcessParams, TuningResult


def cohen_coon(
    params: ProcessParams,
    controller_type: str = "PID",
) -> TuningResult:
    """
    Cohen-Coon tuning for FOPDT processes.

    Better than ZN when dead time is significant (L/tau in 0.1..1.0).
    Produces ~5-10% overshoot for PID.

    Parameters
    ----------
    params:          FOPDT process identification results.
    controller_type: "P", "PI", or "PID".
    """
    K, tau, L = params.K, params.tau, params.L

    if L == 0:
        L = 1e-6

    r = L / tau  # dead-time ratio (dimensionless)
    ct = controller_type.upper()

    if ct == "P":
        kp = (1 / K) * (1 / r + 1 / 3)
        ki, kd = 0.0, 0.0
        note = "P: Cohen-Coon."
    elif ct == "PI":
        kp = (1 / K) * (0.9 / r + 1 / 12)
        Ti = L * (30 + 3 * r) / (9 + 20 * r)
        ki = kp / Ti
        kd = 0.0
        note = "PI: Cohen-Coon. Good disturbance rejection."
    elif ct == "PID":
        kp = (1 / K) * (4 / 3 / r + 1 / 4)
        Ti = L * (32 + 6 * r) / (13 + 8 * r)
        Td = L * 4 / (11 + 2 * r)
        ki = kp / Ti
        kd = kp * Td
        note = "PID: Cohen-Coon. Balanced speed and overshoot."
    else:
        raise ValueError(f"controller_type must be P/PI/PID, got '{controller_type}'")

    return TuningResult(
        method=f"Cohen-Coon ({ct})",
        kp=kp,
        ki=ki,
        kd=kd,
        notes=note,
    )
