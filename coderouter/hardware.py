"""Shared hardware detection + memory accounting (low-memory track, L0).

Background
==========

Low-memory machines (8-16 GB unified / discrete VRAM) can only run
small GGUF models, and CodeRouter's existing memory handling is purely
*reactive*: :mod:`coderouter.guards.memory_pressure` only fires *after*
a backend has already tripped an OOM. To prevent OOM *before* dispatch
we need to know how much memory the host actually has.

The detection primitive already existed inside
``coderouter.ingress.launcher_routes._detect_hardware`` but was only
wired to the launcher UI. This module promotes it to a shared,
cached, dependency-free utility so the guard path can consume it too.

5-deps invariant
================

Detection is **best-effort and uses only the standard library**
(``os.sysconf`` / ``subprocess`` calling ``sysctl`` / ``nvidia-smi``).
No ``psutil`` / ``pynvml``. Every probe is wrapped so a missing tool or
permission error degrades gracefully to ``0.0`` rather than raising.

Caching
=======

Detection performs blocking I/O (subprocess). Results are cached in
process with a short TTL (:data:`_CACHE_TTL_S`) so the hot dispatch
path pays the cost at most once per minute. ``detect_hardware`` is
safe to call from async code via ``asyncio.to_thread``.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess  # controlled: fixed argv, no shell
import threading
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BYTES_PER_GB: int = 1024**3

#: Detection cache TTL. Hardware doesn't change mid-session, but we keep
#: a TTL so a hot-plugged eGPU or driver restart is eventually noticed.
_CACHE_TTL_S: float = 60.0

#: Default headroom reserved for the OS and other processes, in GB.
#: On unified-memory (Metal) systems the OS + UI already consume a few
#: GB, so a conservative floor avoids starving the desktop.
DEFAULT_HEADROOM_GB: float = 1.5

#: Default headroom as a fraction of usable memory. The effective
#: headroom is ``max(DEFAULT_HEADROOM_GB, usable * DEFAULT_HEADROOM_RATIO)``.
DEFAULT_HEADROOM_RATIO: float = 0.15


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Best-effort snapshot of the host's compute resources.

    All memory values are in GiB. ``0.0`` means "could not detect"
    (caller should treat detection as unavailable, not as "zero RAM").
    """

    #: System RAM in GiB (0.0 if undetectable).
    ram_gb: float
    #: GPU VRAM in GiB. For Metal/unified memory this mirrors ``ram_gb``;
    #: for CPU-only it is 0.0.
    vram_gb: float
    #: One of ``"cuda"`` / ``"metal"`` / ``"cpu"``.
    gpu: str
    #: Logical CPU count (best-effort, defaults to 4).
    cpu_count: int

    @property
    def detected(self) -> bool:
        """True iff at least RAM was detected (a usable budget exists)."""
        return self.ram_gb > 0.0

    @property
    def unified_memory(self) -> bool:
        """True for Apple-silicon Metal, where VRAM and RAM are shared."""
        return self.gpu == "metal"


# ---------------------------------------------------------------------------
# Detection (cached)
# ---------------------------------------------------------------------------

_cache_lock = threading.RLock()
_cache_value: HardwareInfo | None = None
_cache_ts: float = 0.0


def _detect_ram_gb() -> float:
    """Detect system RAM in GiB via stdlib, then ``sysctl`` fallback."""
    ram_gb = 0.0
    with contextlib.suppress(ValueError, OSError, AttributeError):
        ram_gb = (
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        ) / _BYTES_PER_GB
    if ram_gb <= 0:
        with contextlib.suppress(ValueError, OSError, subprocess.SubprocessError):
            out = subprocess.run(  # fixed argv, no shell
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            ram_gb = int(out.stdout.strip()) / _BYTES_PER_GB
    return ram_gb


def _detect_gpu(ram_gb: float) -> tuple[str, float]:
    """Detect (gpu_kind, vram_gb).

    Apple silicon → unified memory (VRAM == RAM). NVIDIA → query
    ``nvidia-smi``. Otherwise CPU with 0 VRAM.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal", ram_gb  # unified memory
    if shutil.which("nvidia-smi"):
        with contextlib.suppress(ValueError, OSError, subprocess.SubprocessError):
            out = subprocess.run(  # fixed argv, no shell
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            mb = max(
                (int(x) for x in out.stdout.split() if x.strip().isdigit()),
                default=0,
            )
            if mb > 0:
                return "cuda", mb / 1024
    return "cpu", 0.0


def _detect_uncached() -> HardwareInfo:
    """Run the full best-effort detection (no caching)."""
    cpu = os.cpu_count() or 4
    ram_gb = _detect_ram_gb()
    gpu, vram_gb = _detect_gpu(ram_gb)
    return HardwareInfo(
        ram_gb=round(ram_gb, 1),
        vram_gb=round(vram_gb, 1),
        gpu=gpu,
        cpu_count=cpu,
    )


def detect_hardware(*, force_refresh: bool = False) -> HardwareInfo:
    """Return a cached :class:`HardwareInfo` snapshot.

    Blocking (subprocess). Call via ``asyncio.to_thread`` from async
    code. The result is cached for :data:`_CACHE_TTL_S` seconds.

    Parameters
    ----------
    force_refresh
        Bypass the cache and re-probe immediately (e.g. after a
        backend restart).
    """
    global _cache_value, _cache_ts
    now = time.monotonic()
    with _cache_lock:
        if (
            not force_refresh
            and _cache_value is not None
            and (now - _cache_ts) < _CACHE_TTL_S
        ):
            return _cache_value
        info = _detect_uncached()
        _cache_value = info
        _cache_ts = now
        return info


def reset_cache() -> None:
    """Drop the detection cache. Mainly for tests."""
    global _cache_value, _cache_ts
    with _cache_lock:
        _cache_value = None
        _cache_ts = 0.0


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


def usable_memory_gb(hw: HardwareInfo) -> float:
    """Memory available for model weights + KV cache, in GiB.

    CUDA → dedicated VRAM. Metal/CPU → system RAM (unified or host).
    Returns 0.0 when nothing was detected (caller should no-op rather
    than make a wrong decision).
    """
    if not hw.detected:
        return 0.0
    if hw.gpu == "cuda":
        return hw.vram_gb
    return hw.ram_gb


def headroom_gb(
    usable_gb: float,
    *,
    floor_gb: float = DEFAULT_HEADROOM_GB,
    ratio: float = DEFAULT_HEADROOM_RATIO,
) -> float:
    """Memory to *reserve* for the OS / other processes, in GiB.

    ``max(floor_gb, usable_gb * ratio)`` — a fixed floor protects tiny
    machines, the ratio scales the reserve on larger ones.
    """
    return max(floor_gb, usable_gb * ratio)


def available_budget_gb(
    hw: HardwareInfo,
    *,
    floor_gb: float = DEFAULT_HEADROOM_GB,
    ratio: float = DEFAULT_HEADROOM_RATIO,
) -> float:
    """Net memory usable for weights + KV after subtracting headroom.

    Never negative. Returns 0.0 when hardware is undetected.
    """
    usable = usable_memory_gb(hw)
    if usable <= 0.0:
        return 0.0
    return max(0.0, usable - headroom_gb(usable, floor_gb=floor_gb, ratio=ratio))


__all__ = [
    "DEFAULT_HEADROOM_GB",
    "DEFAULT_HEADROOM_RATIO",
    "HardwareInfo",
    "available_budget_gb",
    "detect_hardware",
    "headroom_gb",
    "reset_cache",
    "usable_memory_gb",
]
