"""Tests for plant simulation models."""

import pytest
import numpy as np
from pid_tuner.simulation.plant_models import (
    FirstOrderPlus, SecondOrder, IntegratingPlant, get_plant, AVAILABLE_PLANTS
)


class TestFirstOrderPlus:
    def test_step_to_steady_state(self):
        plant = FirstOrderPlus(K=2.0, tau=1.0, L=0.0)
        plant.reset()
        dt = 0.001
        y = 0.0
        for _ in range(10_000):
            y = plant.step(1.0, dt)
        assert y == pytest.approx(2.0, rel=1e-2)

    def test_dead_time_delays_response(self):
        plant_no_delay = FirstOrderPlus(K=1.0, tau=1.0, L=0.0)
        plant_delayed = FirstOrderPlus(K=1.0, tau=1.0, L=0.5)
        dt = 0.01
        plant_no_delay.reset()
        plant_delayed.reset()

        # At t=0.1s, delayed plant should still be near zero
        for _ in range(10):
            y_nd = plant_no_delay.step(1.0, dt)
            y_d = plant_delayed.step(1.0, dt)
        assert y_nd > y_d

    def test_invalid_tau_rejected(self):
        with pytest.raises(ValueError):
            FirstOrderPlus(tau=-1.0)

    def test_reset_clears_state(self):
        plant = FirstOrderPlus(K=1.0, tau=1.0)
        for _ in range(100):
            plant.step(1.0, 0.01)
        plant.reset()
        y = plant.step(0.0, 0.01)
        assert y == pytest.approx(0.0, abs=1e-3)


class TestSecondOrder:
    def test_dc_gain(self):
        plant = SecondOrder(K=3.0, wn=2.0, zeta=1.0)
        plant.reset()
        dt = 0.001
        y = 0.0
        for _ in range(20_000):
            y = plant.step(1.0, dt)
        assert y == pytest.approx(3.0, rel=1e-2)

    def test_overdamped_no_overshoot(self):
        plant = SecondOrder(K=1.0, wn=1.0, zeta=2.0)
        plant.reset()
        dt = 0.01
        peak = 0.0
        for _ in range(2000):
            y = plant.step(1.0, dt)
            peak = max(peak, y)
        assert peak <= 1.01  # negligible overshoot


class TestIntegratingPlant:
    def test_ramp_output(self):
        plant = IntegratingPlant(K=1.0)
        plant.reset()
        dt = 0.1
        for _ in range(10):
            y = plant.step(1.0, dt)
        assert y == pytest.approx(1.0, rel=1e-6)


class TestFactory:
    def test_get_plant_valid(self):
        plant = get_plant("first_order_plus_dead_time", K=1.0, tau=2.0, L=0.1)
        assert isinstance(plant, FirstOrderPlus)

    def test_get_plant_unknown_raises(self):
        with pytest.raises(ValueError):
            get_plant("nonexistent_model")

    def test_available_plants_complete(self):
        assert "first_order_plus_dead_time" in AVAILABLE_PLANTS
        assert "second_order" in AVAILABLE_PLANTS
        assert "integrating" in AVAILABLE_PLANTS
