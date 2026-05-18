"""Tests for the core PID controller."""

import pytest

from pid_tuner.controller.pid import ControllerMode, PIDController, PIDGains


def make_controller(kp=1.0, ki=0.1, kd=0.0, dt=0.01) -> PIDController:
    gains = PIDGains(kp=kp, ki=ki, kd=kd)
    ctrl = PIDController(gains=gains, dt=dt)
    ctrl.reset()
    return ctrl


class TestPIDGains:
    def test_valid_gains(self):
        g = PIDGains(kp=1.0, ki=0.5, kd=0.1)
        assert g.kp == 1.0

    def test_negative_kp_rejected(self):
        with pytest.raises(Exception):
            PIDGains(kp=-1.0)

    def test_output_clamp_order_enforced(self):
        with pytest.raises(Exception):
            PIDGains(kp=1.0, output_min=5.0, output_max=1.0)

    def test_to_dict(self):
        g = PIDGains(kp=2.0, ki=0.3, kd=0.05)
        d = g.to_dict()
        assert set(d.keys()) == {"kp", "ki", "kd"}


class TestPIDController:
    def test_zero_error_zero_output(self):
        ctrl = make_controller(kp=1.0, ki=0.0, kd=0.0)
        out = ctrl.compute(setpoint=0.0, measurement=0.0)
        assert out == pytest.approx(0.0)

    def test_proportional_only(self):
        ctrl = make_controller(kp=2.0, ki=0.0, kd=0.0)
        out = ctrl.compute(setpoint=1.0, measurement=0.0)
        assert out == pytest.approx(2.0)

    def test_output_clamp_upper(self):
        gains = PIDGains(kp=1000.0, ki=0.0, kd=0.0, output_max=5.0)
        ctrl = PIDController(gains=gains, dt=0.01)
        ctrl.reset()
        out = ctrl.compute(setpoint=10.0, measurement=0.0)
        assert out <= 5.0

    def test_output_clamp_lower(self):
        gains = PIDGains(kp=1000.0, ki=0.0, kd=0.0, output_min=-5.0)
        ctrl = PIDController(gains=gains, dt=0.01)
        ctrl.reset()
        out = ctrl.compute(setpoint=-10.0, measurement=0.0)
        assert out >= -5.0

    def test_manual_mode_holds_output(self):
        ctrl = make_controller()
        ctrl.compute(setpoint=1.0, measurement=0.0)   # auto step
        ctrl.mode = ControllerMode.MANUAL
        out1 = ctrl.compute(setpoint=5.0, measurement=0.0)
        out2 = ctrl.compute(setpoint=5.0, measurement=0.0)
        assert out1 == out2  # no change in manual mode

    def test_integral_accumulates(self):
        ctrl = make_controller(kp=0.0, ki=1.0, kd=0.0, dt=0.1)
        # With kp=0, output is purely integral
        ctrl.compute(setpoint=1.0, measurement=0.0)
        out2 = ctrl.compute(setpoint=1.0, measurement=0.0)
        assert out2 > 0.0

    def test_reset_clears_state(self):
        ctrl = make_controller()
        for _ in range(100):
            ctrl.compute(setpoint=1.0, measurement=0.0)
        ctrl.reset()
        assert ctrl.state["integral"] == pytest.approx(0.0)
