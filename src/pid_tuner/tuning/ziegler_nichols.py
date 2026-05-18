"""
Ziegler-Nichols tuning rules.

Two variants:
  1. Open-loop (step test) — uses FOPDT parameters (K, tau, L)
  2. Closed-loop (ultimate gain) — uses Ku (ultimate gain) and Pu (period)
"""

from __future__ import annotations

from .base import ProcessParams, TuningResult


def ziegler_nichols_open_loop(
    params: ProcessParams,
    controller_type: str = "PID",
) -> TuningResult:
    """
    Ziegler-Nichols open-loop (reaction curve) tuning.

    Designed for FOPDT processes.  Produces aggressive (fast) tuning —
    typically 20-30% overshoot.  Good starting point, often needs
    detuning for smoother response.

    Parameters
    ----------
    params:          FOPDT process identification results.
    controller_type: "P", "PI", or "PID".
    """
    K, tau, L = params.K, params.tau, params.L

    if L == 0:
        L = 1e-6  # prevent division by zero for delay-free plants

    ct = controller_type.upper()
    if ct == "P":
        kp = tau / (K * L)
        ki, kd = 0.0, 0.0
        note = "Pure P: ZN open-loop. No steady-state elimination."
    elif ct == "PI":
        kp = 0.9 * tau / (K * L)
        Ti = L / 0.3
        ki = kp / Ti
        kd = 0.0
        note = "PI: ZN open-loop. Eliminates offset, moderate overshoot."
    elif ct == "PID":
        kp = 1.2 * tau / (K * L)
        Ti = 2.0 * L
        Td = 0.5 * L
        ki = kp / Ti
        kd = kp * Td
        note = "PID: ZN open-loop. Fast response, ~25% overshoot expected."
    else:
        raise ValueError(f"controller_type must be P/PI/PID, got '{controller_type}'")

    return TuningResult(
        method=f"Ziegler-Nichols open-loop ({ct})",
        kp=kp,
        ki=ki,
        kd=kd,
        notes=note,
    )


def ziegler_nichols_closed_loop(
    Ku: float,
    Pu: float,
    controller_type: str = "PID",
) -> TuningResult:
    """
    Ziegler-Nichols closed-loop (ultimate gain) tuning.

    Determine Ku and Pu experimentally:
      1. Use P-only control, increase Kp until sustained oscillation.
      2. Ku = that gain, Pu = oscillation period [s].

    Parameters
    ----------
    Ku:              Ultimate gain (onset of sustained oscillation).
    Pu:              Ultimate period [s].
    controller_type: "P", "PI", or "PID".
    """
    if Ku <= 0:
        raise ValueError("Ku must be positive")
    if Pu <= 0:
        raise ValueError("Pu must be positive")

    ct = controller_type.upper()
    if ct == "P":
        kp, ki, kd = 0.5 * Ku, 0.0, 0.0
        note = "P: ZN closed-loop."
    elif ct == "PI":
        kp = 0.45 * Ku
        Ti = Pu / 1.2
        ki = kp / Ti
        kd = 0.0
        note = "PI: ZN closed-loop."
    elif ct == "PID":
        kp = 0.6 * Ku
        Ti = 0.5 * Pu
        Td = 0.125 * Pu
        ki = kp / Ti
        kd = kp * Td
        note = "PID: ZN closed-loop."
    else:
        raise ValueError(f"controller_type must be P/PI/PID, got '{controller_type}'")

    return TuningResult(
        method=f"Ziegler-Nichols closed-loop ({ct})",
        kp=kp,
        ki=ki,
        kd=kd,
        notes=note,
    )
