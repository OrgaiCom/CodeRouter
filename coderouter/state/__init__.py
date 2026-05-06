"""Persistent state layer (v2.0-K).

Four modules:

* :mod:`coderouter.state.store`       — sqlite3 KV store for operational
                                         metadata (budget totals, health
                                         state, self-healing exclusions).
* :mod:`coderouter.state.audit_log`   — JSONL structured event log with
                                         rotation and CLI reader.
* :mod:`coderouter.state.request_log` — JSONL request metadata journal
                                         (per-request token counts, cost,
                                         provider — no request body).
* :mod:`coderouter.state.replay`      — Statistical A/B analysis engine
                                         over request journal entries.
"""
