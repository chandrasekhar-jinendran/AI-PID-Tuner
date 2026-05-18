"""Closed-loop step-response simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pid_tuner.controller.pid import PIDController, PIDGains
from pid_tuner.simulation.plant_models import PlantModel


@dataclass
class SimulationResult:
    """Everything produced by one simulation run."""

    time: NDArray[np.float64]
    setpoint: NDArray[np.float64]
    output: NDArray[np.float64]       # plant output y(t)
    control: NDArray[np.float64]      # controller output u(t)
    error: NDArray[np.float64]        # e(t) = setpoint - output
    gains: dict[str, float]
    plant_info: dict

    def to_dict(self) -> dict:
        return {
            "time": self.time.tolist(),
            "setpoint": self.setpoint.tolist(),
            "output": self.output.tolist(),
            "control": self.control.tolist(),
            "error": self.error.tolist(),
            "gains": self.gains,
            "plant": self.plant_info,
        }


def run_simulation(
    gains: PIDGains,
    plant: PlantModel,
    duration: float = 20.0,
    dt: float = 0.01,
    setpoint: float = 1.0,
    disturbance_time: float | None = None,
    disturbance_magnitude: float = 0.2,
) -> SimulationResult:
    """
    Run a closed-loop step-response simulation.

    Parameters
    ----------
    gains:                 PID gains to use.
    plant:                 Plant model instance (will be reset).
    duration:              Simulation duration [s].
    dt:                    Integration time step [s].
    setpoint:              Step amplitude (applied at t=0).
    disturbance_time:      Time [s] at which a load disturbance is injected.
    disturbance_magnitude: Amplitude of the load disturbance.

    Returns
    -------
    SimulationResult with full time-series data.
    """
    plant.reset()
    controller = PIDController(gains=gains, dt=dt)
    controller.reset()

    n_steps = int(duration / dt)
    time_arr = np.linspace(0, duration, n_steps)
    sp_arr = np.full(n_steps, setpoint)
    out_arr = np.zeros(n_steps)
    ctrl_arr = np.zeros(n_steps)
    err_arr = np.zeros(n_steps)

    measurement = 0.0
    disturbance_step = (
        int(disturbance_time / dt) if disturbance_time is not None else None
    )

    for i in range(n_steps):
        u = controller.compute(setpoint, measurement)

        # Inject additive disturbance on the plant input
        if disturbance_step is not None and i == disturbance_step:
            u_actual = u + disturbance_magnitude
        else:
            u_actual = u

        measurement = plant.step(u_actual, dt)

        out_arr[i] = measurement
        ctrl_arr[i] = u
        err_arr[i] = setpoint - measurement

    return SimulationResult(
        time=time_arr,
        setpoint=sp_arr,
        output=out_arr,
        control=ctrl_arr,
        error=err_arr,
        gains=gains.to_dict(),
        plant_info=plant.to_dict(),
    )
