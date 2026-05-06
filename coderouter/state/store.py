"""Persistent KV store backed by sqlite3 (v2.0-K).

Provides durable cross-restart storage for operational metadata:
budget totals, backend health state, self-healing exclusions, and
metrics counters.

Design choices
==============

- **sqlite3 stdlib** — zero new dependencies, matching the 5-deps
  invariant.  WAL mode for concurrent readers + single writer.
- **Namespace-scoped keys** — each subsystem (budget, health, metrics,
  self_healing) gets its own namespace, so ``store.get("budget",
  "totals")`` won't collide with ``store.get("health", "totals")``.
- **JSON values** — all values are serialized as JSON strings. This
  keeps the schema trivial (one table) while letting subsystems store
  arbitrary structured data.
- **Thread-safe** — sqlite3 in WAL mode + ``check_same_thread=False``
  is safe for the multi-thread (asyncio + thread pool) pattern
  CodeRouter uses.
- **Graceful degradation** — if the database can't be opened or
  written, errors are logged but never raised to callers. State
  persistence is a best-effort enhancement, not a correctness
  requirement.

Schema::

    CREATE TABLE IF NOT EXISTS kv (
        namespace TEXT NOT NULL,
        key       TEXT NOT NULL,
        value     TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (namespace, key)
    );
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from coderouter.logging import get_logger

logger = get_logger(__name__)


class StateStore:
    """Persistent KV store for operational metadata.

    Public API:

    - :meth:`get(namespace, key)` — retrieve a JSON-decoded value,
      or None if not found.
    - :meth:`put(namespace, key, value)` — upsert a JSON-encoded value.
    - :meth:`delete(namespace, key)` — remove a key.
    - :meth:`get_all(namespace)` — retrieve all key-value pairs in a
      namespace as a dict.
    - :meth:`clear(namespace)` — remove all keys in a namespace.
    - :meth:`close()` — close the database connection.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Create the database and table if they don't exist."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=5.0,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    namespace TEXT NOT NULL,
                    key       TEXT NOT NULL,
                    value     TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            conn.commit()
            self._conn = conn
        except (OSError, sqlite3.Error) as exc:
            logger.warning(
                "state-store-init-failed",
                extra={"error": str(exc), "db_path": str(self._db_path)},
            )
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, namespace: str, key: str) -> object | None:
        """Retrieve a value by namespace+key, or None if not found."""
        if self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT value FROM kv WHERE namespace = ? AND key = ?",
                    (namespace, key),
                ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            logger.warning(
                "state-store-get-failed",
                extra={"namespace": namespace, "key": key, "error": str(exc)},
            )
            return None

    def put(self, namespace: str, key: str, value: object) -> None:
        """Upsert a value (JSON-serialized)."""
        if self._conn is None:
            return
        try:
            now = datetime.now(UTC).isoformat()
            value_json = json.dumps(value, default=str)
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO kv (namespace, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(namespace, key)
                    DO UPDATE SET value = excluded.value,
                                  updated_at = excluded.updated_at
                    """,
                    (namespace, key, value_json, now),
                )
                self._conn.commit()
        except (sqlite3.Error, TypeError) as exc:
            logger.warning(
                "state-store-put-failed",
                extra={"namespace": namespace, "key": key, "error": str(exc)},
            )

    def delete(self, namespace: str, key: str) -> None:
        """Remove a key from the store."""
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM kv WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning(
                "state-store-delete-failed",
                extra={"namespace": namespace, "key": key, "error": str(exc)},
            )

    def get_all(self, namespace: str) -> dict[str, object]:
        """Return all key-value pairs in a namespace."""
        if self._conn is None:
            return {}
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT key, value FROM kv WHERE namespace = ?",
                    (namespace,),
                ).fetchall()
            return {row[0]: json.loads(row[1]) for row in rows}
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            logger.warning(
                "state-store-get-all-failed",
                extra={"namespace": namespace, "error": str(exc)},
            )
            return {}

    def clear(self, namespace: str) -> None:
        """Remove all keys in a namespace."""
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM kv WHERE namespace = ?",
                    (namespace,),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning(
                "state-store-clear-failed",
                extra={"namespace": namespace, "error": str(exc)},
            )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None


__all__ = ["StateStore"]
