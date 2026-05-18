"""
Plant (process) models for closed-loop simulation.

Each plant exposes a `step(u, dt)` method that advances the internal state
by one time-step and returns the new output y.  All models are implemented
as continuous ODEs integrated with 4th-order Runge-Kutta so the simulator
can use any step size without numerical instability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PlantModel(ABC):
    """Abstract base for all plant models."""

    @abstractmethod
    def step(self, u: float, dt: float) -> float:
        """Advance the model by dt and return the new output y."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state to initial conditions."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class FirstOrderPlus(PlantModel):
    """
    First-Order Plus Dead Time (FOPDT).

    Transfer function: G(s) = K * exp(-L*s) / (tau*s + 1)

    Parameters
    ----------
    K:   Process gain (steady-state output/input ratio)
    tau: Time constant [s]
    L:   Dead time / transport delay [s]
    """

    def __init__(self, K: float = 1.0, tau: float = 1.0, L: float = 0.0) -> None:  # noqa: N803
        if tau <= 0:
            raise ValueError("tau must be positive")
        if L < 0:
            raise ValueError("dead time L must be >= 0")
        self.K = K
        self.tau = tau
        self.L = L
        self._y: float = 0.0
        self._delay_buffer: list[float] = []
        self._delay_steps: int = 0

    def reset(self) -> None:
        self._y = 0.0
        self._delay_buffer = []

    def _ode(self, y: float, u_delayed: float) -> float:
        """dy/dt = (K*u - y) / tau"""
        return (self.K * u_delayed - y) / self.tau

    def step(self, u: float, dt: float) -> float:
        # Initialise delay buffer on first step (adaptive to dt)
        if not self._delay_buffer:
            self._delay_steps = max(0, round(self.L / dt))
            self._delay_buffer = [0.0] * (self._delay_steps + 1)

        self._delay_buffer.append(u)
        u_delayed = self._delay_buffer.pop(0)

        # RK4
        k1 = self._ode(self._y, u_delayed)
        k2 = self._ode(self._y + 0.5 * dt * k1, u_delayed)
        k3 = self._ode(self._y + 0.5 * dt * k2, u_delayed)
        k4 = self._ode(self._y + dt * k3, u_delayed)
        self._y += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self._y

    @property
    def name(self) -> str:
        return "first_order_plus_dead_time"

    @property
    def description(self) -> str:
        return f"FOPDT: K={self.K}, tau={self.tau}s, L={self.L}s"

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.name, "K": self.K, "tau": self.tau, "L": self.L}


class SecondOrder(PlantModel):
    """
    Second-order system.

    Transfer function: G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)

    Parameters
    ----------
    K:    DC gain
    wn:   Natural frequency [rad/s]
    zeta: Damping ratio (0=undamped, 1=critically damped, >1=overdamped)
    """

    def __init__(self, K: float = 1.0, wn: float = 1.0, zeta: float = 0.7) -> None:  # noqa: N803
        if wn <= 0:
            raise ValueError("wn must be positive")
        self.K = K
        self.wn = wn
        self.zeta = zeta
        self._x1: float = 0.0  # position
        self._x2: float = 0.0  # velocity

    def reset(self) -> None:
        self._x1 = 0.0
        self._x2 = 0.0

    def _ode(self, x1: float, x2: float, u: float) -> tuple[float, float]:
        dx1 = x2
        dx2 = self.K * self.wn**2 * u - 2 * self.zeta * self.wn * x2 - self.wn**2 * x1
        return dx1, dx2

    def step(self, u: float, dt: float) -> float:
        # RK4 on 2-state system
        k1x1, k1x2 = self._ode(self._x1, self._x2, u)
        k2x1, k2x2 = self._ode(self._x1 + 0.5*dt*k1x1, self._x2 + 0.5*dt*k1x2, u)
        k3x1, k3x2 = self._ode(self._x1 + 0.5*dt*k2x1, self._x2 + 0.5*dt*k2x2, u)
        k4x1, k4x2 = self._ode(self._x1 + dt*k3x1, self._x2 + dt*k3x2, u)
        self._x1 += (dt / 6.0) * (k1x1 + 2*k2x1 + 2*k3x1 + k4x1)
        self._x2 += (dt / 6.0) * (k1x2 + 2*k2x2 + 2*k3x2 + k4x2)
        return self._x1

    @property
    def name(self) -> str:
        return "second_order"

    @property
    def description(self) -> str:
        return f"2nd order: K={self.K}, wn={self.wn} rad/s, zeta={self.zeta}"

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.name, "K": self.K, "wn": self.wn, "zeta": self.zeta}


class IntegratingPlant(PlantModel):
    """
    Integrating process (pure integrator).

    Transfer function: G(s) = K / s

    Common in level-control and velocity systems.
    """

    def __init__(self, K: float = 1.0) -> None:  # noqa: N803
        self.K = K
        self._y: float = 0.0

    def reset(self) -> None:
        self._y = 0.0

    def step(self, u: float, dt: float) -> float:
        self._y += self.K * u * dt
        return self._y

    @property
    def name(self) -> str:
        return "integrating"

    @property
    def description(self) -> str:
        return f"Integrating: K={self.K}"

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.name, "K": self.K}


_REGISTRY: dict[str, type[PlantModel]] = {
    "first_order_plus_dead_time": FirstOrderPlus,
    "second_order": SecondOrder,
    "integrating": IntegratingPlant,
}

AVAILABLE_PLANTS = list(_REGISTRY.keys())


def get_plant(model: str, **kwargs: Any) -> PlantModel:
    """Factory: instantiate a plant by name."""
    if model not in _REGISTRY:
        raise ValueError(f"Unknown plant '{model}'. Available: {AVAILABLE_PLANTS}")
    return _REGISTRY[model](**kwargs)
