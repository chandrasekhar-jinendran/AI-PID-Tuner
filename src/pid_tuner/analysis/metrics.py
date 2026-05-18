"""
Step-response performance metrics.

Computes the standard set used in control engineering:
  - Rise time, Settling time, Peak time
  - Overshoot / Undershoot
  - Steady-state error
  - ISE, IAE, ITAE (integral error criteria)
  - Control effort (total variation of u)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pid_tuner.simulation.simulator import SimulationResult


@dataclass
class ResponseMetrics:
    """All performance metrics for one step response."""

    # Time-domain metrics
    rise_time_10_90: float       # [s] 10% → 90% of setpoint
    settling_time_2pct: float    # [s] to stay within ±2% of setpoint
    settling_time_5pct: float    # [s] to stay within ±5% of setpoint
    peak_time: float             # [s] time of first peak
    overshoot_pct: float         # [%] (peak - setpoint) / setpoint * 100
    undershoot_pct: float        # [%] minimum dip below setpoint

    # Steady-state
    steady_state_error: float    # final error value
    steady_state_output: float   # final plant output

    # Integral error criteria (lower = better)
    ise: float    # Integral of Squared Error   ∫ e² dt
    iae: float    # Integral of Absolute Error  ∫ |e| dt
    itae: float   # Integral of Time*|e|        ∫ t·|e| dt

    # Control effort
    total_variation: float       # ∑|Δu| — total actuator movement

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 6) for k, v in asdict(self).items()}

    def summary(self) -> str:
        """Human-readable one-liner for quick inspection."""
        return (
            f"RT={self.rise_time_10_90:.2f}s  "
            f"ST(2%)={self.settling_time_2pct:.2f}s  "
            f"OS={self.overshoot_pct:.1f}%  "
            f"SSE={self.steady_state_error:.4f}  "
            f"IAE={self.iae:.4f}"
        )


def compute_metrics(result: SimulationResult) -> ResponseMetrics:
    """
    Derive all performance metrics from a SimulationResult.

    Parameters
    ----------
    result: Output of `run_simulation`.

    Returns
    -------
    ResponseMetrics dataclass.
    """
    t = result.time
    y = result.output
    u = result.control
    sp = result.setpoint[0]   # step amplitude (constant)

    # --- Rise time (10% → 90%) ---
    y10 = 0.10 * sp
    y90 = 0.90 * sp
    idx10 = np.argmax(y >= y10) if np.any(y >= y10) else len(t) - 1
    idx90 = np.argmax(y >= y90) if np.any(y >= y90) else len(t) - 1
    rise_time = float(t[idx90] - t[idx10])

    # --- Overshoot / undershoot ---
    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    overshoot = max(0.0, (peak_val - sp) / abs(sp) * 100) if sp != 0 else 0.0
    min_val = float(np.min(y))
    undershoot = max(0.0, (sp - min_val) / abs(sp) * 100) if sp != 0 else 0.0
    peak_time = float(t[peak_idx])

    # --- Settling time ---
    def _settling(band_pct: float) -> float:
        band = band_pct / 100 * abs(sp)
        outside = np.abs(y - sp) > band
        # Find last index that is outside the band
        out_indices = np.where(outside)[0]
        if len(out_indices) == 0:
            return 0.0
        last_out = out_indices[-1]
        return float(t[min(last_out + 1, len(t) - 1)])

    settling_2 = _settling(2.0)
    settling_5 = _settling(5.0)

    # --- Steady-state ---
    tail = max(1, int(0.05 * len(t)))   # last 5% of simulation
    ss_output = float(np.mean(y[-tail:]))
    ss_error = float(sp - ss_output)

    # --- Integral criteria ---
    e = result.error
    ise = float(np.trapezoid(e**2, t))
    iae = float(np.trapezoid(np.abs(e), t))
    itae = float(np.trapezoid(t * np.abs(e), t))

    # --- Control effort (total variation) ---
    tv = float(np.sum(np.abs(np.diff(u))))

    return ResponseMetrics(
        rise_time_10_90=rise_time,
        settling_time_2pct=settling_2,
        settling_time_5pct=settling_5,
        peak_time=peak_time,
        overshoot_pct=overshoot,
        undershoot_pct=undershoot,
        steady_state_error=ss_error,
        steady_state_output=ss_output,
        ise=ise,
        iae=iae,
        itae=itae,
        total_variation=tv,
    )
