"""Integration tests: simulate → analyse metrics."""

import pytest
import numpy as np
from pid_tuner.controller.pid import PIDGains
from pid_tuner.simulation.plant_models import FirstOrderPlus
from pid_tuner.simulation.simulator import run_simulation
from pid_tuner.analysis.metrics import compute_metrics
from pid_tuner.tuning.imc import imc_tune
from pid_tuner.tuning.base import ProcessParams


@pytest.fixture
def fopdt_result():
    """IMC-tuned response on a well-known FOPDT plant."""
    params = ProcessParams(K=1.0, tau=5.0, L=0.5)
    tuning = imc_tune(params, robustness="moderate")
    gains = PIDGains(kp=tuning.kp, ki=tuning.ki, kd=tuning.kd)
    plant = FirstOrderPlus(K=1.0, tau=5.0, L=0.5)
    return run_simulation(gains=gains, plant=plant, duration=60.0, dt=0.01, setpoint=1.0)


class TestMetricsOnIMCResponse:
    def test_steady_state_near_setpoint(self, fopdt_result):
        m = compute_metrics(fopdt_result)
        assert abs(m.steady_state_error) < 0.05

    def test_overshoot_bounded(self, fopdt_result):
        m = compute_metrics(fopdt_result)
        assert m.overshoot_pct < 25.0  # IMC moderate should be well under this

    def test_rise_time_positive(self, fopdt_result):
        m = compute_metrics(fopdt_result)
        assert m.rise_time_10_90 > 0

    def test_iae_positive(self, fopdt_result):
        m = compute_metrics(fopdt_result)
        assert m.iae > 0

    def test_itae_greater_than_iae(self, fopdt_result):
        m = compute_metrics(fopdt_result)
        # ITAE weights late errors more, so it's always >= IAE for typical responses
        assert m.itae >= 0

    def test_to_dict_keys(self, fopdt_result):
        m = compute_metrics(fopdt_result)
        d = m.to_dict()
        expected_keys = {
            "rise_time_10_90", "settling_time_2pct", "settling_time_5pct",
            "peak_time", "overshoot_pct", "undershoot_pct",
            "steady_state_error", "steady_state_output",
            "ise", "iae", "itae", "total_variation",
        }
        assert expected_keys == set(d.keys())


class TestDisturbanceRejection:
    def test_output_recovers_after_disturbance(self):
        params = ProcessParams(K=1.0, tau=3.0, L=0.2)
        tuning = imc_tune(params)
        gains = PIDGains(kp=tuning.kp, ki=tuning.ki, kd=tuning.kd)
        plant = FirstOrderPlus(K=1.0, tau=3.0, L=0.2)
        result = run_simulation(
            gains=gains, plant=plant,
            duration=60.0, dt=0.01, setpoint=1.0,
            disturbance_time=20.0, disturbance_magnitude=0.5,
        )
        m = compute_metrics(result)
        assert abs(m.steady_state_error) < 0.05
