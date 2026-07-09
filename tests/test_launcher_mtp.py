"""Tests for MTP / speculative-decoding support in the llama.cpp launcher.

Covers the shared decision logic in
:mod:`coderouter.launcher_speculative` (``resolve_speculative`` /
``find_draft_companion``) plus the launcher-route integration:

* auto detection of nextn-embedded main ggufs → ``--spec-type draft-mtp``,
* same-folder companion discovery (size ratio, arch-mismatch rejection,
  mtp- vs draft-named ``--spec-type`` choice),
* the ``mtp_mode='off'`` / explicit-path / defer / non-llama.cpp guards,
* the ``--split-mode tensor`` crash warning,
* ``_MODEL_FLAGS`` rejection of ``-md`` and the end-to-end ``api_start``
  behaviour (400 on override, ``speculative`` key in the start response).
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import CodeRouterConfig, FallbackChain, ProviderConfig
from coderouter.ingress import launcher_routes
from coderouter.ingress.app import create_app
from coderouter.ingress.launcher_routes import _build_cmd
from coderouter.launcher_speculative import find_draft_companion, resolve_speculative
from coderouter.metrics import uninstall_collector

# ---------------------------------------------------------------------------
# Synthetic GGUF builders (mirrors tests/test_gguf_introspect.py helpers)
# ---------------------------------------------------------------------------

_MAGIC = b"GGUF"
_T_UINT32 = 4
_T_STRING = 8


def _gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _T_STRING) + _gguf_string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _gguf_bytes(arch: str, *, nextn: int | None = None) -> bytes:
    kvs = [_kv_string("general.architecture", arch)]
    if nextn is not None:
        kvs.append(_kv_u32(f"{arch}.nextn_predict_layers", nextn))
    header = _MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 0)
    header += struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)


def _make_gguf(
    path: Path, arch: str, *, nextn: int | None = None, size: int = 0
) -> Path:
    """Write a parseable GGUF header, padded with zero bytes to ``size``.

    Trailing padding is never read by the header parser, so it is a cheap way
    to give the file a controllable on-disk size for the ratio checks.
    """
    data = _gguf_bytes(arch, nextn=nextn)
    if size > len(data):
        data = data + b"\x00" * (size - len(data))
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# resolve_speculative — mtp_mode / backend guards
# ---------------------------------------------------------------------------


def test_off_returns_no_flags() -> None:
    tokens, _notes = resolve_speculative("llama.cpp", "/m.gguf", None, "off", [])
    assert tokens == []


def test_off_with_draft_path_raises() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_speculative("llama.cpp", "/m.gguf", "/d.gguf", "off", [])


def test_non_llamacpp_with_draft_raises() -> None:
    with pytest.raises(ValueError, match=r"only supported for llama\.cpp"):
        resolve_speculative("vllm", "/m.safetensors", "/d.gguf", "auto", [])


def test_non_llamacpp_off_raises() -> None:
    with pytest.raises(ValueError, match=r"only supported for llama\.cpp"):
        resolve_speculative("mlx", "/m", None, "off", [])


def test_non_llamacpp_auto_is_noop() -> None:
    tokens, notes = resolve_speculative("vllm", "/m.safetensors", None, "auto", [])
    assert tokens == [] and notes == []


def test_user_spec_type_defers(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "m.gguf", "glm4moe", nextn=1, size=4000)
    tokens, notes = resolve_speculative(
        "llama.cpp", str(main), None, "auto", ["--spec-type", "draft-eagle3"]
    )
    assert tokens == []
    assert any("skipped" in n for n in notes)


def test_user_spec_type_equals_form_defers(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "m.gguf", "glm4moe", nextn=1, size=4000)
    tokens, _ = resolve_speculative(
        "llama.cpp", str(main), None, "auto", ["--spec-type=draft-mtp"]
    )
    assert tokens == []


# ---------------------------------------------------------------------------
# resolve_speculative — explicit draft path
# ---------------------------------------------------------------------------


def test_explicit_missing_path_raises() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        resolve_speculative(
            "llama.cpp", "/m.gguf", "/nope/missing.gguf", "auto", []
        )


def test_explicit_draft_simple(tmp_path: Path) -> None:
    draft = _make_gguf(tmp_path / "small-draft.gguf", "llama", size=1000)
    tokens, _notes = resolve_speculative(
        "llama.cpp", "/m.gguf", str(draft), "auto", []
    )
    assert tokens == ["--spec-type", "draft-simple", "--model-draft", str(draft)]


def test_explicit_draft_mtp_by_name(tmp_path: Path) -> None:
    draft = _make_gguf(tmp_path / "model-mtp.gguf", "llama", size=1000)
    tokens, _ = resolve_speculative(
        "llama.cpp", "/m.gguf", str(draft), "auto", []
    )
    assert tokens[:2] == ["--spec-type", "draft-mtp"]
    assert tokens[2:] == ["--model-draft", str(draft)]


# ---------------------------------------------------------------------------
# resolve_speculative — auto detection
# ---------------------------------------------------------------------------


def test_auto_nextn_main_gguf(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "glm.gguf", "glm4moe", nextn=2, size=4000)
    tokens, notes = resolve_speculative("llama.cpp", str(main), None, "auto", [])
    assert tokens == ["--spec-type", "draft-mtp"]
    assert any("nextn layers (2)" in n for n in notes)


def test_auto_no_gguf_skips_silently() -> None:
    tokens, notes = resolve_speculative(
        "llama.cpp", "/models/model.safetensors", None, "auto", []
    )
    assert tokens == []
    assert any("not an existing .gguf" in n for n in notes)


def test_auto_nothing_found(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "solo.gguf", "llama", size=4000)
    tokens, notes = resolve_speculative("llama.cpp", str(main), None, "auto", [])
    assert tokens == []
    assert any("not found next to" in n for n in notes)


def test_auto_companion_mtp_named(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "qwen-7b.gguf", "qwen2", size=8000)
    _make_gguf(tmp_path / "qwen-7b-mtp.gguf", "qwen2", size=1000)
    tokens, _notes = resolve_speculative("llama.cpp", str(main), None, "auto", [])
    assert tokens[:2] == ["--spec-type", "draft-mtp"]
    assert tokens[2] == "--model-draft"
    assert tokens[3].endswith("qwen-7b-mtp.gguf")


def test_auto_companion_draft_named(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "qwen-7b.gguf", "qwen2", size=8000)
    _make_gguf(tmp_path / "qwen-7b-draft.gguf", "qwen2", size=1000)
    tokens, _ = resolve_speculative("llama.cpp", str(main), None, "auto", [])
    assert tokens[:2] == ["--spec-type", "draft-simple"]
    assert tokens[3].endswith("qwen-7b-draft.gguf")


# ---------------------------------------------------------------------------
# find_draft_companion — ratio & arch-mismatch
# ---------------------------------------------------------------------------


def test_companion_too_large_is_ignored(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "base.gguf", "llama", size=4000)
    # 3000 >= 50% of 4000 → not a plausible draft.
    _make_gguf(tmp_path / "base-draft.gguf", "llama", size=3000)
    assert find_draft_companion(main) is None


def test_companion_arch_mismatch_dropped(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "base.gguf", "llama", size=8000)
    # draft-named but wrong architecture → vocab mismatch → rejected.
    _make_gguf(tmp_path / "base-draft.gguf", "qwen2", size=1000)
    assert find_draft_companion(main) is None


def test_companion_arch_match_selected(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "base.gguf", "llama", size=8000)
    good = _make_gguf(tmp_path / "base-draft.gguf", "llama", size=1000)
    assert find_draft_companion(main) == good


def test_companion_mtp_outranks_draft(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "base.gguf", "llama", size=8000)
    _make_gguf(tmp_path / "base-draft.gguf", "llama", size=1000)
    mtp = _make_gguf(tmp_path / "base-mtp.gguf", "llama", size=1200)
    assert find_draft_companion(main) == mtp


# ---------------------------------------------------------------------------
# split-mode tensor warning
# ---------------------------------------------------------------------------


def test_split_mode_tensor_warns(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "glm.gguf", "glm4moe", nextn=1, size=4000)
    tokens, notes = resolve_speculative(
        "llama.cpp", str(main), None, "auto", ["--split-mode", "tensor"]
    )
    assert tokens == ["--spec-type", "draft-mtp"]
    assert any("#24309" in n for n in notes)


def test_split_mode_tensor_equals_form_warns(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "glm.gguf", "glm4moe", nextn=1, size=4000)
    _tokens, notes = resolve_speculative(
        "llama.cpp", str(main), None, "auto", ["--split-mode=tensor"]
    )
    assert any("#24309" in n for n in notes)


def test_no_warning_without_spec_flags(tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "solo.gguf", "llama", size=4000)
    tokens, notes = resolve_speculative(
        "llama.cpp", str(main), None, "auto", ["--split-mode", "tensor"]
    )
    assert tokens == []
    assert not any("#24309" in n for n in notes)


# ---------------------------------------------------------------------------
# _build_cmd — spec token placement + -md rejection
# ---------------------------------------------------------------------------


def test_build_cmd_places_spec_tokens_after_port() -> None:
    cmd = _build_cmd(
        "llama.cpp",
        "/models/good.gguf",
        8080,
        {"--threads": 8},
        "-ngl 99",
        spec_tokens=["--spec-type", "draft-mtp"],
    )
    port_idx = cmd.index("--port")
    st_idx = cmd.index("--spec-type")
    threads_idx = cmd.index("--threads")
    ngl_idx = cmd.index("-ngl")
    # spec tokens land right after the port args, before profile/extra args.
    assert st_idx == port_idx + 2
    assert st_idx < threads_idx < ngl_idx
    assert cmd.count("--spec-type") == 1


@pytest.mark.parametrize("flag", ["-md", "--model-draft", "--spec-draft-model"])
def test_build_cmd_rejects_draft_flag_in_extra_args(flag: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _build_cmd(
            "llama.cpp",
            "/models/good.gguf",
            8080,
            {},
            f"{flag} /models/draft.gguf",
        )


def test_build_cmd_allows_spec_type_in_extra_args() -> None:
    """The remaining spec knobs stay free-form in extra_args."""
    cmd = _build_cmd(
        "llama.cpp",
        "/models/good.gguf",
        8080,
        {},
        "--spec-type draft-mtp --spec-draft-n-max 3",
    )
    assert "--spec-type" in cmd and "--spec-draft-n-max" in cmd


# ---------------------------------------------------------------------------
# api_start integration (TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:8080/v1",
                model="qwen-coder",
                paid=False,
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
    )


@pytest.fixture
def client(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


class _FakeProc:
    """Minimal stand-in for an asyncio subprocess used by _tail_logs."""

    def __init__(self) -> None:
        self.pid = 4321
        self.stdout = None
        self.stderr = None
        self.returncode = 0

    async def wait(self) -> int:
        return 0

    def terminate(self) -> None:  # pragma: no cover - not exercised
        pass

    def kill(self) -> None:  # pragma: no cover - not exercised
        pass


@pytest.fixture
def _patch_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_exec(*_args: object, **_kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(
        launcher_routes.asyncio, "create_subprocess_exec", _fake_exec
    )


def test_api_start_rejects_draft_flag_in_extra_args(client: TestClient) -> None:
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "llama.cpp",
            "model_path": "/models/good.gguf",
            "port": 8080,
            "extra_args": "-md /models/draft.gguf",
        },
    )
    assert resp.status_code == 400, resp.text
    assert "not allowed" in resp.json()["detail"]


def test_api_start_rejects_bad_mtp_mode(client: TestClient) -> None:
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "llama.cpp",
            "model_path": "/models/good.gguf",
            "port": 8080,
            "mtp_mode": "always",
        },
    )
    assert resp.status_code == 400, resp.text
    assert "mtp_mode" in resp.json()["detail"]


def test_api_start_off_with_draft_is_400(client: TestClient) -> None:
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "llama.cpp",
            "model_path": "/models/good.gguf",
            "port": 8080,
            "mtp_mode": "off",
            "draft_model_path": "/models/draft.gguf",
        },
    )
    assert resp.status_code == 400, resp.text


def test_api_start_response_has_speculative(
    client: TestClient, tmp_path: Path, _patch_subprocess: None
) -> None:
    main = _make_gguf(tmp_path / "glm.gguf", "glm4moe", nextn=1, size=4000)
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "glm",
            "backend": "llama.cpp",
            "model_path": str(main),
            "port": 8080,
            "mtp_mode": "auto",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["speculative"] == ["--spec-type", "draft-mtp"]
    # The resolved flags are present in the command exactly once.
    assert body["command"].count("--spec-type") == 1
