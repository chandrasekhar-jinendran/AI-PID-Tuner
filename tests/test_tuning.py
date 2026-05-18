"""Tests for tuning algorithms."""

import pytest
from pid_tuner.tuning.base import ProcessParams
from pid_tuner.tuning.ziegler_nichols import ziegler_nichols_open_loop, ziegler_nichols_closed_loop
from pid_tuner.tuning.cohen_coon import cohen_coon
from pid_tuner.tuning.imc import imc_tune


PARAMS = ProcessParams(K=2.0, tau=5.0, L=1.0)


class TestZieglerNicholsOpenLoop:
    def test_pid_gains_positive(self):
        r = ziegler_nichols_open_loop(PARAMS)
        assert r.kp > 0
        assert r.ki > 0
        assert r.kd > 0

    def test_p_only_no_integral(self):
        r = ziegler_nichols_open_loop(PARAMS, "P")
        assert r.ki == pytest.approx(0.0)
        assert r.kd == pytest.approx(0.0)

    def test_pi_no_derivative(self):
        r = ziegler_nichols_open_loop(PARAMS, "PI")
        assert r.kd == pytest.approx(0.0)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ziegler_nichols_open_loop(PARAMS, "PDI")


class TestZieglerNicholsClosedLoop:
    def test_pid_gains_positive(self):
        r = ziegler_nichols_closed_loop(Ku=10.0, Pu=2.0)
        assert r.kp > 0 and r.ki > 0 and r.kd > 0

    def test_bad_ku_raises(self):
        with pytest.raises(ValueError):
            ziegler_nichols_closed_loop(Ku=-1.0, Pu=2.0)


class TestCohenCoon:
    def test_pid_gains_positive(self):
        r = cohen_coon(PARAMS)
        assert r.kp > 0 and r.ki > 0 and r.kd > 0

    def test_gains_differ_from_zn(self):
        r_zn = ziegler_nichols_open_loop(PARAMS)
        r_cc = cohen_coon(PARAMS)
        assert r_zn.kp != pytest.approx(r_cc.kp)


class TestIMC:
    def test_pid_gains_positive(self):
        r = imc_tune(PARAMS)
        assert r.kp > 0 and r.ki > 0 and r.kd > 0

    def test_larger_lambda_smaller_kp(self):
        r_fast = imc_tune(PARAMS, lambda_=0.5)
        r_slow = imc_tune(PARAMS, lambda_=5.0)
        assert r_fast.kp > r_slow.kp

    def test_robustness_levels(self):
        r_agg = imc_tune(PARAMS, robustness="aggressive")
        r_con = imc_tune(PARAMS, robustness="conservative")
        assert r_agg.kp > r_con.kp

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            imc_tune(PARAMS, lambda_=-1.0)
