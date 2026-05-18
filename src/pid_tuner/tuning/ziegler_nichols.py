"""
Ziegler-Nichols tuning rules.

Two variants:
  1. Open-loop (step test) — uses FOPDT parameters (K, tau, L)
  2. Closed-loop (ultimate gain) — uses ku (ultimate gain) and pu (period)
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
    k, tau, dead_time = params.K, params.tau, params.L

    if dead_time == 0:
        dead_time = 1e-6  # prevent division by zero for delay-free plants

    ct = controller_type.upper()
    if ct == "P":
        kp = tau / (k * dead_time)
        ki, kd = 0.0, 0.0
        note = "Pure P: ZN open-loop. No steady-state elimination."
    elif ct == "PI":
        kp = 0.9 * tau / (k * dead_time)
        ti = dead_time / 0.3
        ki = kp / ti
        kd = 0.0
        note = "PI: ZN open-loop. Eliminates offset, moderate overshoot."
    elif ct == "PID":
        kp = 1.2 * tau / (k * dead_time)
        ti = 2.0 * dead_time
        td = 0.5 * dead_time
        ki = kp / ti
        kd = kp * td
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
    ku: float,
    pu: float,
    controller_type: str = "PID",
) -> TuningResult:
    """
    Ziegler-Nichols closed-loop (ultimate gain) tuning.

    Determine ku and pu experimentally:
      1. Use P-only control, increase Kp until sustained oscillation.
      2. ku = that gain, pu = oscillation period [s].

    Parameters
    ----------
    ku:              Ultimate gain (onset of sustained oscillation).
    pu:              Ultimate period [s].
    controller_type: "P", "PI", or "PID".
    """
    if ku <= 0:
        raise ValueError("ku must be positive")
    if pu <= 0:
        raise ValueError("pu must be positive")

    ct = controller_type.upper()
    if ct == "P":
        kp, ki, kd = 0.5 * ku, 0.0, 0.0
        note = "P: ZN closed-loop."
    elif ct == "PI":
        kp = 0.45 * ku
        ti = pu / 1.2
        ki = kp / ti
        kd = 0.0
        note = "PI: ZN closed-loop."
    elif ct == "PID":
        kp = 0.6 * ku
        ti = 0.5 * pu
        td = 0.125 * pu
        ki = kp / ti
        kd = kp * td
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
