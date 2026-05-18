# PID Tuner — Claude Code Guide

This project is an AI-assisted PID controller tuner with a live MCP server.
Claude Code can interact with the tuner engine directly via MCP tools.

## Project Layout

```
src/pid_tuner/
├── controller/     PID math (discrete, anti-windup, filtered derivative)
├── simulation/     Plant models (FOPDT, 2nd-order, integrating) + RK4 simulator
├── tuning/         Classical algorithms: ZN open/closed, Cohen-Coon, IMC
├── analysis/       Performance metrics: RT, ST, OS, ISE, IAE, ITAE
└── mcp_server/     FastMCP server — this is what Claude Code calls
```

## Starting the MCP Server

```bash
# Install (first time only)
pip install -e ".[dev]"

# Run the MCP server
pid-tuner-mcp
# or: python -m pid_tuner.mcp_server.server
```

The server uses stdio transport. Claude Code connects via the config below.

## Connecting Claude Code to This MCP Server

Add this to your Claude Code MCP config (`~/.claude/mcp.json` or project `.claude/mcp.json`):

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

Restart Claude Code after adding the config.

## Available MCP Tools

| Tool | What it does |
|------|-------------|
| `list_plants` | Show available plant models |
| `tune_pid(method, K, tau, L)` | Get gains from ZN/Cohen-Coon/IMC |
| `simulate_pid(kp, ki, kd, plant_model, ...)` | Run closed-loop simulation |
| `analyze_response(time, output, control, setpoint)` | Compute metrics |
| `compare_tunings(K, tau, L)` | Benchmark all 3 algorithms at once |
| `identify_fopdt(time, output, step_magnitude)` | Estimate process params |

## Typical Workflow for Claude Code

1. **Identify the plant**: Run an open-loop step test or use known params.
   ```
   Call identify_fopdt with open-loop step data to get K, tau, L
   ```

2. **Get initial gains**: Pick a tuning method.
   ```
   Call tune_pid(method="imc", K=..., tau=..., L=...) for robust starting point
   ```

3. **Simulate and measure**: Run the closed loop.
   ```
   Call simulate_pid(kp=..., ki=..., kd=..., plant_model="first_order_plus_dead_time", ...)
   Call analyze_response(time=[...], output=[...], control=[...])
   ```

4. **Iterate**: Adjust gains based on metrics and re-simulate.
   - Overshoot too high → reduce Kp or increase Ki
   - Too slow → increase Kp
   - Oscillating → reduce Kd or increase derivative filter

5. **Compare**: Benchmark algorithms.
   ```
   Call compare_tunings(K=..., tau=..., L=...) to see ZN vs CC vs IMC side-by-side
   ```

## Performance Specifications (default targets)

| Metric | Target |
|--------|--------|
| Overshoot | < 10% |
| Settling time (2%) | < 3 × tau |
| Steady-state error | < 1% |
| IAE | minimise |

## Development Commands

```bash
pytest                    # run test suite
ruff check src tests      # lint
mypy src                  # type check
```

## Key Design Decisions

- **Discrete PID with RK4 plant**: keeps simulation numerically stable at any dt
- **Back-calculation anti-windup**: prevents integrator wind-up on saturated outputs
- **Filtered derivative on measurement** (not error): avoids derivative kick on setpoint steps
- **FastMCP / stdio**: simplest transport, works everywhere without network config
- **Pydantic models for gains**: validation at the boundary, not buried in math
