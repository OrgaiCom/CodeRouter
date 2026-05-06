"""Long-run reliability guards (v1.9-E).

CodeRouter's third pillar (``docs/inside/future.md`` §1: P3 Long-run
Reliability) lives here. Each module addresses one of the systematic
failure modes that a continuously-running local-LLM agent loop tends
to hit:

  * :mod:`coderouter.guards.tool_loop`      — L3 stuck-tool detection
  * :mod:`coderouter.guards.memory_pressure` — L2 backend OOM awareness
  * :mod:`coderouter.guards.backend_health`  — L5 health state machine +
                                                 chain reorder
  * :mod:`coderouter.guards.self_healing`    — v2.0-J auto-exclude +
                                                 restart + recovery probe
  * :mod:`coderouter.guards.continuous_probe` — v2.0-I background probing

Each guard is a pure-functional / single-class module that the engine
consults at the appropriate dispatch point. Guards never block the
fast path — they observe and either log, mutate, or short-circuit
based on operator-configured policy.
"""
