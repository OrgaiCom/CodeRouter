"""Unit tests for coderouter.gguf_introspect (self-written GGUF parser)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from coderouter.gguf_introspect import (
    GGUFParseError,
    read_gguf_metadata,
    try_read_gguf_metadata,
)

_MAGIC = b"GGUF"
_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _T_STRING) + _gguf_string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _kv_array_u32(key: str, values: list[int]) -> bytes:
    body = (
        _gguf_string(key)
        + struct.pack("<I", _T_ARRAY)
        + struct.pack("<I", _T_UINT32)
        + struct.pack("<Q", len(values))
    )
    for v in values:
        body += struct.pack("<I", v)
    return body


def _build_gguf(kvs: list[bytes], *, version: int = 3, tensor_count: int = 0) -> bytes:
    header = _MAGIC + struct.pack("<I", version)
    header += struct.pack("<Q", tensor_count)
    header += struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)


def _write(tmp_path: Path, data: bytes, name: str = "model.gguf") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_parses_full_metadata(tmp_path: Path) -> None:
    kvs = [
        _kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.embedding_length", 4096),
        _kv_u32("llama.attention.head_count", 32),
        _kv_u32("llama.attention.head_count_kv", 8),
        _kv_u32("general.file_type", 15),  # Q4_K_M
    ]
    p = _write(tmp_path, _build_gguf(kvs))
    info = read_gguf_metadata(p)
    assert info.architecture == "llama"
    assert info.n_layers == 32
    assert info.n_embd == 4096
    assert info.n_heads == 32
    assert info.n_kv_heads == 8
    assert info.file_type == 15
    assert info.quant_name == "Q4_K_M"
    assert info.file_size_bytes == len(_build_gguf(kvs))
    assert info.weights_bytes == info.file_size_bytes


def test_skips_unknown_and_array_values(tmp_path: Path) -> None:
    kvs = [
        _kv_string("general.architecture", "qwen2"),
        _kv_array_u32("tokenizer.ggml.tokens_dummy", [1, 2, 3, 4, 5]),
        _kv_u32("qwen2.block_count", 28),
        _kv_u32("qwen2.embedding_length", 3584),
    ]
    p = _write(tmp_path, _build_gguf(kvs))
    info = read_gguf_metadata(p)
    assert info.architecture == "qwen2"
    assert info.n_layers == 28
    assert info.n_embd == 3584
    # Missing head counts → None (parser doesn't invent values).
    assert info.n_heads is None
    assert info.n_kv_heads is None


def test_bad_magic_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, b"NOPE" + b"\x00" * 64)
    with pytest.raises(GGUFParseError):
        read_gguf_metadata(p)


def test_unsupported_version_raises(tmp_path: Path) -> None:
    data = _MAGIC + struct.pack("<I", 99) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    p = _write(tmp_path, data)
    with pytest.raises(GGUFParseError):
        read_gguf_metadata(p)


def test_truncated_header_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, _MAGIC + b"\x03")  # version cut short
    with pytest.raises(GGUFParseError):
        read_gguf_metadata(p)


def test_oversized_string_length_is_clamped(tmp_path: Path) -> None:
    # Declare a key string of 2 GiB → must be rejected, not allocated.
    data = (
        _MAGIC
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 1 << 31)  # absurd key length
    )
    p = _write(tmp_path, data)
    with pytest.raises(GGUFParseError):
        read_gguf_metadata(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GGUFParseError):
        read_gguf_metadata(tmp_path / "does_not_exist.gguf")


def test_try_variant_returns_none_on_bad_file(tmp_path: Path) -> None:
    p = _write(tmp_path, b"junk")
    assert try_read_gguf_metadata(p) is None


def test_nextn_predict_layers_detected(tmp_path: Path) -> None:
    """A ``{arch}.nextn_predict_layers`` KV surfaces as n_nextn / supports_mtp."""
    kvs = [
        _kv_string("general.architecture", "glm4moe"),
        _kv_u32("glm4moe.block_count", 40),
        _kv_u32("glm4moe.nextn_predict_layers", 1),
    ]
    p = _write(tmp_path, _build_gguf(kvs))
    info = read_gguf_metadata(p)
    assert info.n_nextn == 1
    assert info.supports_mtp is True


def test_nextn_absent_is_none_and_not_mtp(tmp_path: Path) -> None:
    """Without the nextn KV, n_nextn is None and supports_mtp is False."""
    kvs = [
        _kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 32),
    ]
    p = _write(tmp_path, _build_gguf(kvs))
    info = read_gguf_metadata(p)
    assert info.n_nextn is None
    assert info.supports_mtp is False
