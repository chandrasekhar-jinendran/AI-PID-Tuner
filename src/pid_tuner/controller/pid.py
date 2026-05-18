"""Discrete PID controller with anti-windup and bumpless transfer."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, model_validator


class ControllerMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class PIDGains(BaseModel):
    """Validated PID gain set."""

    kp: float = Field(..., description="Proportional gain", ge=0)
    ki: float = Field(0.0, description="Integral gain", ge=0)
    kd: float = Field(0.0, description="Derivative gain", ge=0)
    output_min: float = Field(-1e6, description="Lower output clamp")
    output_max: float = Field(1e6, description="Upper output clamp")
    derivative_filter_coeff: float = Field(
        10.0,
        description="N in filtered derivative: s/(s/N+1). Higher = less filtering.",
        gt=0,
    )

    @model_validator(mode="after")
    def _clamp_order(self) -> "PIDGains":
        if self.output_min >= self.output_max:
            raise ValueError("output_min must be strictly less than output_max")
        return self

    def to_dict(self) -> dict[str, float]:
        return {"kp": self.kp, "ki": self.ki, "kd": self.kd}


@dataclass
class PIDController:
    """
    Discrete-time PID controller.

    Implements the ISA standard form:
        u(t) = Kp * [e(t) + (1/Ti) * ∫e dt + Td * de/dt]

    where Ti = Kp/Ki and Td = Kd/Kp (when Ki, Kd > 0).

    Features
    --------
    - Back-calculation anti-windup
    - Filtered derivative (avoids derivative kick on setpoint changes)
    - Bumpless manual-to-auto transfer
    - Output clamping
    """

    gains: PIDGains
    dt: float = 0.01
    mode: ControllerMode = ControllerMode.AUTO

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_measurement: float = field(default=0.0, init=False, repr=False)
    _prev_derivative: float = field(default=0.0, init=False, repr=False)
    _prev_output: float = field(default=0.0, init=False, repr=False)

    def reset(self, measurement: float = 0.0) -> None:
        """Reset internal state. Call before a new simulation run."""
        self._integral = 0.0
        self._prev_measurement = measurement
        self._prev_derivative = 0.0
        self._prev_output = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """
        Compute one control step.

        Parameters
        ----------
        setpoint:    Desired process value.
        measurement: Current process measurement (PV).

        Returns
        -------
        Control output u, clamped to [output_min, output_max].
        """
        if self.mode is ControllerMode.MANUAL:
            return self._prev_output

        g = self.gains
        error = setpoint - measurement

        # Proportional term
        p_term = g.kp * error

        # Filtered derivative on measurement (avoids setpoint-kick)
        alpha = g.derivative_filter_coeff * self.dt
        raw_derivative = -(measurement - self._prev_measurement) / self.dt
        d_term = g.kd * (alpha * raw_derivative + (1 - alpha) * self._prev_derivative)
        self._prev_derivative = alpha * raw_derivative + (1 - alpha) * self._prev_derivative

        # Unclamped output (for anti-windup)
        unclamped = p_term + g.ki * self._integral + d_term

        # Output clamping
        output = float(np.clip(unclamped, g.output_min, g.output_max))

        # Back-calculation anti-windup: freeze integrator when saturated
        if g.ki > 0:
            saturation_error = output - unclamped
            self._integral += self.dt * (error + saturation_error / g.ki)
        else:
            self._integral += self.dt * error

        self._prev_measurement = measurement
        self._prev_output = output
        return output

    @property
    def state(self) -> dict[str, float]:
        return {
            "integral": self._integral,
            "prev_measurement": self._prev_measurement,
            "prev_derivative": self._prev_derivative,
            "prev_output": self._prev_output,
        }
