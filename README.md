# PID Tuner — AI-Assisted PID Controller Tuning via MCP

[![CI](https://github.com/YOUR_USERNAME/pid-tuner/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/pid-tuner/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-quality PID controller tuner written in Python, exposing its full engine to **Claude Code via the Model Context Protocol (MCP)**. Claude can call tuning algorithms, run closed-loop simulations, and analyse performance metrics — all without leaving the chat.

> **Learning goal:** This project is a primer for MCP integration with MATLAB/Simulink. Everything here maps directly to how you'd expose a MATLAB engine as MCP tools.

---

## Features

| Layer | What's inside |
|-------|--------------|
| **Controller** | Discrete PID with back-calculation anti-windup, filtered derivative, bumpless manual/auto transfer |
| **Plant models** | FOPDT (1st-order + dead time), 2nd-order, integrating — all with RK4 integration |
| **Tuning algorithms** | Ziegler-Nichols (open & closed loop), Cohen-Coon, IMC / Skogestad |
| **Metrics** | Rise time, settling time, overshoot, ISE, IAE, ITAE, total control variation |
| **MCP server** | 6 Claude-callable tools via `FastMCP` / stdio transport |
| **Testing** | pytest suite with 30+ tests, coverage report |
| **CI/CD** | GitHub Actions: lint → type-check → test → build |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/pid-tuner.git
cd pid-tuner
pip install -e ".[dev]"
```

### 2. Run the MCP server

```bash
pid-tuner-mcp
```

### 3. Connect Claude Code

Add to `~/.claude/mcp.json` (or `.claude/mcp.json` in this project):

```json
{
  "mcpServers": {
    "pid-tuner": {
      "command": "pid-tuner-mcp",
      "env": {}
    }
  }
}
```

Restart Claude Code. Now ask it:

> *"Tune a PID for a plant with K=2, tau=5s, L=1s and show me the step response metrics."*

Claude will call `tune_pid` → `simulate_pid` → `analyze_response` automatically.

---

## MCP Tools Reference

### `list_plants`
Returns all available plant models with their parameter descriptions.

### `tune_pid(method, K, tau, L, controller_type, ...)`
Computes PID gains using a classical algorithm.

| `method` | Algorithm | Best for |
|----------|-----------|---------|
| `zn_open_loop` | Ziegler-Nichols (step test) | Fast aggressive tuning |
| `zn_closed_loop` | Ziegler-Nichols (ultimate gain) | When you can oscillate the loop |
| `cohen_coon` | Cohen-Coon | Large dead-time processes |
| `imc` | IMC / Skogestad | Robust, single-knob tuning |

### `simulate_pid(kp, ki, kd, plant_model, ...)`
Runs a closed-loop step-response simulation and returns time-series data.

### `analyze_response(time, output, control, setpoint)`
Computes all standard performance metrics from raw time-series data.

### `compare_tunings(K, tau, L)`
Benchmarks ZN, Cohen-Coon, and IMC side-by-side in one call.

### `identify_fopdt(time, output, step_magnitude)`
Estimates K, tau, L from open-loop step-test data (reaction curve method).

---

## Project Structure

```
pid-tuner/
├── src/pid_tuner/
│   ├── controller/          # PID math
│   │   └── pid.py           # PIDController, PIDGains
│   ├── simulation/          # Plant models + simulator
│   │   ├── plant_models.py  # FOPDT, SecondOrder, Integrating
│   │   └── simulator.py     # run_simulation → SimulationResult
│   ├── tuning/              # Classical tuning algorithms
│   │   ├── ziegler_nichols.py
│   │   ├── cohen_coon.py
│   │   └── imc.py
│   ├── analysis/            # Performance metrics
│   │   └── metrics.py       # compute_metrics → ResponseMetrics
│   └── mcp_server/          # MCP integration layer
│       └── server.py        # FastMCP + 6 tools
├── tests/                   # pytest suite
├── .github/workflows/ci.yml # CI pipeline
├── CLAUDE.md                # Claude Code integration guide
└── pyproject.toml
```

---

## Engineering Decisions

**Why discrete PID with RK4 plants?**
Continuous-time PID (Laplace domain) is elegant for theory but you can't run it on a computer without discretising. This project uses the ISA standard discrete form, which is how real PLCs implement PID. The plants use 4th-order Runge-Kutta so the simulation is numerically stable at any step size.

**Why back-calculation anti-windup?**
When the actuator saturates (hits its output limit), a naive PID keeps integrating the error — then when it unsaturates, there's a huge delayed response. Back-calculation feeds the saturation error back into the integrator to "undo" the wind-up instantly.

**Why filtered derivative on measurement (not error)?**
When the setpoint steps, the derivative of the error has a huge instantaneous spike (derivative kick). By differentiating the measured output instead, we avoid this — the derivative term only reacts to actual process changes.

**Why IMC as the recommended starting point?**
IMC gives one tuning parameter (λ = desired closed-loop time constant) with a clear physical meaning. You can explain it to a process engineer in one sentence: *"λ is how fast you want the loop to respond."*

---

## Transferring to MATLAB / Simulink

The MCP pattern maps directly:

| This project (Python) | MATLAB equivalent |
|----------------------|-------------------|
| `PlantModel.step(u, dt)` | Simulink block or `lsim()` |
| `run_simulation()` | `sim()` or `lsim()` with feedback loop |
| `compute_metrics()` | `stepinfo()` + custom ITAE/ISE |
| `mcp_server/server.py` | MATLAB Engine API + FastMCP wrapper |
| `PIDGains` (Pydantic) | `struct` or `pidtune()` output |

To expose MATLAB as MCP: wrap `matlab.engine` calls in FastMCP tools the same way `simulate_pid` wraps `run_simulation`.

---

## Development

```bash
pytest                  # full test suite
ruff check src tests    # linting
mypy src                # type checking
python -m build         # build wheel
```

---

## License

MIT — see [LICENSE](LICENSE).
