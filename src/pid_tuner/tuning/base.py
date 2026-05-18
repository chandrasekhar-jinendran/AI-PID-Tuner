"""Shared data structures for tuning algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class ProcessParams(BaseModel):
    """
    Identified FOPDT process parameters.

    These three numbers fully describe a First-Order Plus Dead Time plant,
    which is the input required by most classical tuning methods.

    K   — Process gain:    steady-state ΔY / ΔU
    tau — Time constant:   time [s] to reach 63.2% of final value
    L   — Dead time [s]:   pure transport delay
    """

    K: float = Field(..., description="Process gain (ΔY/ΔU)", gt=0)
    tau: float = Field(..., description="Dominant time constant [s]", gt=0)
    L: float = Field(0.0, description="Dead time / transport delay [s]", ge=0)


@dataclass(frozen=True)
class TuningResult:
    """Output of any tuning algorithm."""

    method: str
    kp: float
    ki: float
    kd: float
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "kp": round(self.kp, 6),
            "ki": round(self.ki, 6),
            "kd": round(self.kd, 6),
            "notes": self.notes,
        }
