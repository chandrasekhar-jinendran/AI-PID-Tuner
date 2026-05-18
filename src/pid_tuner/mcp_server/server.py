"""
PID Tuner MCP Server
====================

Exposes the full PID tuner engine as MCP tools that Claude Code can call
directly.  Start the server with:

    pid-tuner-mcp          (after pip install -e .)
    python -m pid_tuner.mcp_server.server

Then configure Claude Code to connect (see CLAUDE.md).

Available tools
---------------
  simulate_pid          — Run a closed-loop step-response simulation
  tune_pid              — Apply a classical tuning algorithm
  analyze_response      — Compute performance metrics from simulation data
  compare_tunings       — Simulate + analyse multiple PID configs side-by-side
  list_plants           — List available plant models with descriptions
  identify_fopdt        — Estimate FOPDT params from open-loop step data
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from pid_tuner.analysis.metrics import compute_metrics
from pid_tuner.controller.pid import PIDGains
from pid_tuner.simulation.plant_models import get_plant
from pid_tuner.simulation.simulator import SimulationResult, run_simulation
from pid_tuner.tuning.base import ProcessParams
from pid_tuner.tuning.cohen_coon import cohen_coon
from pid_tuner.tuning.imc import imc_tune
from pid_tuner.tuning.ziegler_nichols import ziegler_nichols_closed_loop, ziegler_nichols_open_loop

mcp = FastMCP(
    name="pid-tuner",
    instructions=(
        "You are a control-systems expert using this PID Tuner MCP server. "
        "Use `list_plants` first to see available models, then `tune_pid` to get "
        "initial gains, then `simulate_pid` + `analyze_response` to evaluate. "
        "Iterate: adjust gains, re-simulate, compare metrics until specs are met. "
        "Use `compare_tunings` to benchmark multiple algorithms at once."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: list_plants
# ---------------------------------------------------------------------------

@mcp.tool()
def list_plants() -> str:
    """
    List all available plant (process) models and their parameters.

    Returns a JSON object mapping model name → description and required kwargs.
    """
    catalogue = {
        "first_order_plus_dead_time": {
            "description": "FOPDT: G(s) = K*exp(-L*s)/(tau*s+1). Most common industrial model.",
            "kwargs": {
                "K": "process gain (float, >0)",
                "tau": "time constant [s] (float, >0)",
                "L": "dead time [s] (float, >=0, default 0)",
            },
        },
        "second_order": {
            "description": "2nd-order: G(s)=K*wn²/(s²+2ζwn·s+wn²). Oscillatory/mechanical systems.",
            "kwargs": {
                "K": "DC gain (float)",
                "wn": "natural frequency [rad/s] (float, >0)",
                "zeta": "damping ratio (float, typical 0.3-2.0)",
            },
        },
        "integrating": {
            "description": "Pure integrator: G(s) = K/s. Level tanks, velocity control.",
            "kwargs": {"K": "integrator gain (float)"},
        },
    }
    return json.dumps(catalogue, indent=2)


# ---------------------------------------------------------------------------
# Tool 2: tune_pid
# ---------------------------------------------------------------------------

@mcp.tool()
def tune_pid(
    method: str,
    k: float,
    tau: float,
    dead_time: float = 0.0,
    controller_type: str = "PID",
    lambda_: float | None = None,
    robustness: str = "moderate",
    ku: float | None = None,
    pu: float | None = None,
) -> str:
    """
    Compute PID gains using a classical tuning algorithm.

    Parameters
    ----------
    method:          Tuning method. One of:
                       "zn_open_loop"   — Ziegler-Nichols step-test
                       "zn_closed_loop" — Ziegler-Nichols ultimate-gain
                       "cohen_coon"     — Cohen-Coon (better for large L/tau)
                       "imc"            — IMC / Skogestad (most robust)
    k:               Process gain (ΔY/ΔU from open-loop step test).
    tau:             Dominant time constant [s].
    dead_time:       Dead time [s] (default 0).
    controller_type: "P", "PI", or "PID" (default "PID").
    lambda_:         IMC only — closed-loop time constant [s]. Auto-set if None.
    robustness:      IMC only — "aggressive", "moderate", "conservative".
    ku:              ZN closed-loop only — ultimate gain.
    pu:              ZN closed-loop only — ultimate period [s].

    Returns
    -------
    JSON with keys: method, kp, ki, kd, notes.
    """
    params = ProcessParams(K=k, tau=tau, L=dead_time)
    m = method.lower().strip()

    if m == "zn_open_loop":
        result = ziegler_nichols_open_loop(params, controller_type)
    elif m == "zn_closed_loop":
        if ku is None or pu is None:
            return json.dumps({"error": "zn_closed_loop requires ku and pu arguments."})
        result = ziegler_nichols_closed_loop(ku, pu, controller_type)
    elif m == "cohen_coon":
        result = cohen_coon(params, controller_type)
    elif m == "imc":
        result = imc_tune(params, lambda_=lambda_, robustness=robustness)
    else:
        return json.dumps({
            "error": f"Unknown method '{method}'.",
            "available": ["zn_open_loop", "zn_closed_loop", "cohen_coon", "imc"],
        })

    return json.dumps(result.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Tool 3: simulate_pid
# ---------------------------------------------------------------------------

@mcp.tool()
def simulate_pid(
    kp: float,
    ki: float,
    kd: float,
    plant_model: str,
    plant_k: float = 1.0,
    plant_tau: float = 1.0,
    plant_l: float = 0.0,
    plant_wn: float = 1.0,
    plant_zeta: float = 0.7,
    duration: float = 30.0,
    dt: float = 0.01,
    setpoint: float = 1.0,
    output_min: float = -1e6,
    output_max: float = 1e6,
    disturbance_time: float | None = None,
    disturbance_magnitude: float = 0.2,
    derivative_filter_coeff: float = 10.0,
) -> str:
    """
    Run a closed-loop step-response simulation.

    Parameters
    ----------
    kp, ki, kd:            PID gains to simulate.
    plant_model:           Plant model name (use list_plants to see options).
    plant_k, plant_tau, plant_l:  FOPDT/integrating plant params.
    plant_wn, plant_zeta:  Second-order plant params.
    duration:              Simulation duration [s] (default 30).
    dt:                    Integration step [s] (default 0.01).
    setpoint:              Step target value (default 1.0).
    output_min/max:        Controller output clamps.
    disturbance_time:      Time [s] to inject a load disturbance (None = skip).
    disturbance_magnitude: Magnitude of the disturbance (default 0.2).
    derivative_filter_coeff: Derivative filter N (default 10).

    Returns
    -------
    JSON with time, setpoint, output, control, error arrays plus gains/plant info.
    Data is downsampled to ≤500 points to keep response compact.
    """
    gains = PIDGains(
        kp=kp, ki=ki, kd=kd,
        output_min=output_min,
        output_max=output_max,
        derivative_filter_coeff=derivative_filter_coeff,
    )

    # Build plant kwargs based on model type — keys match constructor arg names (K, L)
    plant_kwargs: dict[str, Any] = {"K": plant_k}
    if plant_model == "first_order_plus_dead_time":
        plant_kwargs.update({"tau": plant_tau, "L": plant_l})

    elif plant_model == "second_order":
        plant_kwargs.update({"wn": plant_wn, "zeta": plant_zeta})
    # integrating only needs K

    try:
        plant = get_plant(plant_model, **plant_kwargs)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    result = run_simulation(
        gains=gains,
        plant=plant,
        duration=duration,
        dt=dt,
        setpoint=setpoint,
        disturbance_time=disturbance_time,
        disturbance_magnitude=disturbance_magnitude,
    )

    # Downsample to max 500 points so JSON stays readable
    data = result.to_dict()
    n = len(data["time"])
    if n > 500:
        idx = np.round(np.linspace(0, n - 1, 500)).astype(int)
        for key in ("time", "setpoint", "output", "control", "error"):
            data[key] = [data[key][i] for i in idx]

    return json.dumps(data)


# ---------------------------------------------------------------------------
# Tool 4: analyze_response
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_response(
    time: list[float],
    output: list[float],
    control: list[float],
    setpoint: float = 1.0,
) -> str:
    """
    Compute performance metrics from raw simulation time-series data.

    Parameters
    ----------
    time:     Time vector [s].
    output:   Plant output y(t) vector.
    control:  Controller output u(t) vector.
    setpoint: Step reference value (scalar, default 1.0).

    Returns
    -------
    JSON with keys: rise_time_10_90, settling_time_2pct, settling_time_5pct,
    peak_time, overshoot_pct, undershoot_pct, steady_state_error,
    steady_state_output, ise, iae, itae, total_variation.
    """
    t_arr = np.array(time)
    y_arr = np.array(output)
    u_arr = np.array(control)
    e_arr = setpoint - y_arr
    sp_arr = np.full_like(t_arr, setpoint)

    dummy = SimulationResult(
        time=t_arr,
        setpoint=sp_arr,
        output=y_arr,
        control=u_arr,
        error=e_arr,
        gains={},
        plant_info={},
    )
    metrics = compute_metrics(dummy)
    return json.dumps(metrics.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Tool 5: compare_tunings
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_tunings(
    k: float,
    tau: float,
    dead_time: float = 0.0,
    plant_model: str = "first_order_plus_dead_time",
    duration: float = 30.0,
    setpoint: float = 1.0,
) -> str:
    """
    Run all three tuning algorithms (ZN, Cohen-Coon, IMC) and return a
    side-by-side comparison of their gains and key performance metrics.

    Parameters
    ----------
    k:           Process gain.
    tau:         Time constant [s].
    dead_time:   Dead time [s] (default 0).
    plant_model: Plant model to simulate on.
    duration:    Simulation duration [s].
    setpoint:    Step target.

    Returns
    -------
    JSON list of {method, kp, ki, kd, metrics} for each algorithm.
    """
    params = ProcessParams(K=k, tau=tau, L=dead_time)
    algorithms = [
        ziegler_nichols_open_loop(params, "PID"),
        cohen_coon(params, "PID"),
        imc_tune(params, robustness="moderate"),
    ]

    plant_kwargs: dict[str, Any] = {"K": k}
    if plant_model == "first_order_plus_dead_time":
        plant_kwargs.update({"tau": tau, "L": dead_time})

    comparison = []
    for tuning in algorithms:
        gains = PIDGains(
                kp=tuning.kp, ki=tuning.ki, kd=tuning.kd,
                output_min=-1e6, output_max=1e6, derivative_filter_coeff=10.0,
            )
        try:
            plant = get_plant(plant_model, **plant_kwargs)
            sim = run_simulation(gains=gains, plant=plant, duration=duration, setpoint=setpoint)
            metrics = compute_metrics(sim)
            entry = {
                **tuning.to_dict(),
                "metrics": metrics.to_dict(),
            }
        except Exception as exc:
            entry = {**tuning.to_dict(), "error": str(exc)}
        comparison.append(entry)

    return json.dumps(comparison, indent=2)


# ---------------------------------------------------------------------------
# Tool 6: identify_fopdt
# ---------------------------------------------------------------------------

@mcp.tool()
def identify_fopdt(
    time: list[float],
    output: list[float],
    step_magnitude: float = 1.0,
    initial_output: float = 0.0,
) -> str:
    """
    Estimate FOPDT parameters (K, tau, L) from an open-loop step-test response.

    Uses the 28.3% / 63.2% tangent-line method (process reaction curve).

    Parameters
    ----------
    time:            Time vector [s].
    output:          Measured plant output y(t) (open-loop, no feedback).
    step_magnitude:  Size of the step applied to the input.
    initial_output:  Plant output before the step (baseline).

    Returns
    -------
    JSON with K, tau, L, and a confidence note.
    """
    t = np.array(time)
    y = np.array(output) - initial_output   # zero-based
    final_val = float(np.mean(y[-max(1, len(y)//20):]))

    if abs(final_val) < 1e-9:
        return json.dumps({"error": "Output did not change — no identifiable step response."})

    k = final_val / step_magnitude

    # 28.3% and 63.2% crossing times
    y283 = 0.283 * final_val
    y632 = 0.632 * final_val

    idx283 = int(np.argmax(y >= y283)) if np.any(y >= y283) else -1
    idx632 = int(np.argmax(y >= y632)) if np.any(y >= y632) else -1

    if idx283 < 0 or idx632 < 0:
        return json.dumps(
            {"error": "Output did not reach 63.2% of final value — extend simulation."}
        )

    t283 = float(t[idx283])
    t632 = float(t[idx632])

    tau = 1.5 * (t632 - t283)
    dead_time = max(0.0, t632 - tau)

    return json.dumps(
        {
            "K": round(k, 6),
            "tau": round(tau, 6),
            "L": round(dead_time, 6),
            "note": (
                "Estimated via 28.3%/63.2% reaction-curve method. "
                "Accuracy depends on data quality and step size. "
                f"L/tau ratio = {dead_time/tau:.3f} "
                "(ZN reliable for <0.5, CC better for 0.1-1.0)."
            ),
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
