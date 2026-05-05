#!/usr/bin/env python3
"""
gguf_dl.py - Hugging Face から GGUF (など) のファイルをまとめてダウンロードする補助ツール

特徴
----
- huggingface.co の URL を貼るだけで repo_id とファイル名を自動解析
    例: https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf
        https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
- 複数ファイル/ワイルドカード一括ダウンロード (split GGUF の *.gguf-00001-of-00003 などにも対応)
- 進捗バー表示 (huggingface_hub 標準)
- 中断したダウンロードの再開 (HEAD ETag 一致なら resume)
- CLI 引数モード & 引数なしの場合は対話プロンプト
- 保存先の優先順位: --dest > $GGUF_DL_DIR > ./models/

使い方
------
事前に依存ライブラリをインストール (新 CLI `hf` も同梱されます):

    pip install --upgrade "huggingface_hub[hf_transfer]"

プライベートリポジトリは `hf auth login` で先にログインするか、環境変数
`HF_TOKEN` を設定してください (旧 `huggingface-cli login` は `hf auth login` に変わりました)。

# 単一ファイル (URL指定)
    python gguf_dl.py https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf

# repo_id + ファイル名
    python gguf_dl.py TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf

# 保存先指定
    python gguf_dl.py <URL> -d ~/models/gguf

# パターンで複数ファイルまとめて (split gguf 等)
    python gguf_dl.py TheBloke/SomeBigModel-GGUF -p "*Q4_K_M*.gguf"

# レポジトリ全 GGUF を表示するだけ
    python gguf_dl.py TheBloke/Llama-2-7B-GGUF --list

# 何もつけずに対話モード
    python gguf_dl.py

# (参考) 同等のことを新 hf CLI で直接やる場合
#   hf download <repo_id> <file>             … 単一ファイル
#   hf download <repo_id> --include "*.gguf" … パターン一括
#   hf download <repo_id> --local-dir <dir>  … 保存先指定
# 本スクリプトは URL を貼るだけで repo_id/ファイル名/ブランチを解析する点と、
# 対話モード・既定保存先のフォールバックを足したラッパーです。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse


def _ensure_hf_hub() -> None:
    """huggingface_hub が無ければ案内して終了。"""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "huggingface_hub がインストールされていません。\n"
            "  pip install --upgrade 'huggingface_hub[hf_transfer]'\n"
            "を実行してから再試行してください。\n"
            "(新 CLI `hf` も同パッケージで提供されます。\n"
            " プライベートレポは `hf auth login` でログインしてください。)\n"
        )
        sys.exit(2)


# -- URL/入力 解析 --------------------------------------------------------

# https://huggingface.co/<owner>/<repo>/(blob|resolve|tree|raw)/<rev>/<path...>
_HF_URL_RE = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"(?:/(?:blob|resolve|tree|raw)/(?P<rev>[^/]+)(?:/(?P<path>.+))?)?/?$"
)

# repo_id 単体形式 (owner/repo)
_REPO_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


def parse_hf_target(s: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    入力文字列を (repo_id, revision, filename) に分解する。

    対応形式:
        - https://huggingface.co/<owner>/<repo>/blob/<rev>/<path>
        - https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>
        - https://huggingface.co/<owner>/<repo>/tree/<rev>
        - https://huggingface.co/<owner>/<repo>
        - <owner>/<repo>
    """
    s = s.strip().strip('"').strip("'")
    if not s:
        raise ValueError("空の入力です。")

    m = _HF_URL_RE.match(s)
    if m:
        owner = m.group("owner")
        repo = m.group("repo")
        rev = m.group("rev") or None
        path = m.group("path") or None
        if path:
            path = unquote(path)
        return f"{owner}/{repo}", rev, path

    if _REPO_ID_RE.match(s):
        return s, None, None

    # URL っぽければ最終手段でパス分解を試す
    if s.startswith(("http://", "https://")):
        u = urlparse(s)
        if u.netloc.endswith("huggingface.co"):
            parts = [p for p in u.path.split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}", None, None

    raise ValueError(
        f"Hugging Face の URL もしくは <owner>/<repo> 形式で指定してください: {s!r}"
    )


# -- 保存先決定 -----------------------------------------------------------

def resolve_dest(cli_dest: Optional[str]) -> Path:
    """保存先の優先順: CLI > 環境変数 GGUF_DL_DIR > ./models"""
    if cli_dest:
        dest = Path(cli_dest).expanduser()
    elif os.environ.get("GGUF_DL_DIR"):
        dest = Path(os.environ["GGUF_DL_DIR"]).expanduser()
    else:
        dest = Path.cwd() / "models"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


# -- HF API 呼び出し ------------------------------------------------------

def list_repo_ggufs(repo_id: str, revision: Optional[str], token: Optional[str]) -> List[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, revision=revision, token=token, files_metadata=False)
    files = [s.rfilename for s in (info.siblings or [])]
    return files


def filter_files(files: Iterable[str], patterns: Iterable[str]) -> List[str]:
    import fnmatch

    pats = list(patterns)
    if not pats:
        return [f for f in files if f.lower().endswith(".gguf")]
    out: List[str] = []
    for f in files:
        if any(fnmatch.fnmatch(f, p) or fnmatch.fnmatch(os.path.basename(f), p) for p in pats):
            out.append(f)
    # 重複除去 + 順序維持
    seen = set()
    uniq = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def download_one(
    repo_id: str,
    filename: str,
    dest: Path,
    revision: Optional[str],
    token: Optional[str],
    flat: bool,
) -> Path:
    """1 ファイルを `dest` 配下にダウンロード。

    flat=True なら dest 直下に実体を配置 (新しい hf 系 CLI と同じ挙動)。
    flat=False なら hf キャッシュ形式 (snapshots) で保存。
    再開はライブラリが ETag/サイズで自動判定するので明示フラグ不要。
    """
    from huggingface_hub import hf_hub_download

    common = dict(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
    if flat:
        # 直接 dest にダウンロード。local_dir 指定時は自動的に
        # シンボリックリンクではなく実体配置になり、再開も自動。
        path = hf_hub_download(local_dir=str(dest), **common)
    else:
        path = hf_hub_download(cache_dir=str(dest / ".hf_cache"), **common)
    return Path(path)


# -- 対話モード -----------------------------------------------------------

def prompt(msg: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            ans = input(f"{msg}{suffix}: ").strip()
        except EOFError:
            ans = ""
        if ans:
            return ans
        if default is not None:
            return default
        print("値を入力してください。")


def interactive_mode() -> argparse.Namespace:
    print("=== gguf_dl 対話モード ===")
    target = prompt("Hugging Face の URL もしくは <owner>/<repo>")
    repo_id, rev_from_url, filename = parse_hf_target(target)
    revision = rev_from_url or prompt("revision (ブランチ/タグ)", "main")

    use_pattern = False
    pattern: Optional[str] = None
    if not filename:
        ans = prompt("ファイル指定方法 [single/pattern/all-gguf]", "all-gguf").lower()
        if ans.startswith("p"):
            pattern = prompt("パターン (例: *Q4_K_M*.gguf)")
            use_pattern = True
        elif ans.startswith("s"):
            filename = prompt("ファイル名 (リポジトリルートからの相対パス)")
        else:
            use_pattern = True
            pattern = "*.gguf"

    dest = prompt("保存先フォルダ", str(Path.cwd() / "models"))

    args = argparse.Namespace(
        target=repo_id,
        filename=filename,
        revision=revision,
        dest=dest,
        pattern=[pattern] if use_pattern and pattern else [],
        list=False,
        nested=False,
        token=None,
        yes=False,
    )
    return args


# -- メイン ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gguf_dl",
        description="Hugging Face から GGUF を一括ダウンロードする補助ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Hugging Face URL もしくは <owner>/<repo> (省略すると対話モード)",
    )
    p.add_argument(
        "filename",
        nargs="?",
        help="ダウンロードするファイル名 (URL に含まれていれば省略可)",
    )
    p.add_argument("-d", "--dest", help="保存先フォルダ ($GGUF_DL_DIR か ./models が既定)")
    p.add_argument("-r", "--revision", help="リビジョン (既定: URL から or main)")
    p.add_argument(
        "-p",
        "--pattern",
        action="append",
        default=[],
        help="ワイルドカードパターンで複数指定可 (例: -p '*Q4_K_M*.gguf')。複数指定可。",
    )
    p.add_argument("--list", action="store_true", help="リポジトリ内のファイル一覧を表示するだけ")
    p.add_argument(
        "--nested",
        action="store_true",
        help="HF キャッシュ形式で保存 (デフォルトは dest 直下にフラット配置)",
    )
    p.add_argument("--token", help="プライベートリポジトリ用 HF トークン (環境変数 HF_TOKEN も可)")
    p.add_argument("-y", "--yes", action="store_true", help="確認プロンプトをスキップ")
    return p


def confirm(msg: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        ans = input(f"{msg} [Y/n]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("", "y", "yes")


def human_size(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_hf_hub()

    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        # 対話モード
        try:
            args = interactive_mode()
        except (KeyboardInterrupt, EOFError):
            print("\n中断しました。")
            return 130
    else:
        args = parser.parse_args(argv)

    if not args.target:
        parser.print_help()
        return 1

    # 入力を解析
    try:
        repo_id, rev_from_url, fn_from_url = parse_hf_target(args.target)
    except ValueError as e:
        sys.stderr.write(f"エラー: {e}\n")
        return 1

    revision = args.revision or rev_from_url
    filename = args.filename or fn_from_url

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dest = resolve_dest(args.dest)

    # --list / パターン展開のためにファイル一覧取得が必要かどうか
    need_listing = args.list or (not filename) or args.pattern

    files_to_get: List[str] = []

    if filename and not args.pattern and not args.list:
        files_to_get = [filename]
    else:
        try:
            all_files = list_repo_ggufs(repo_id, revision, token)
        except Exception as e:
            sys.stderr.write(f"リポジトリ情報の取得に失敗: {e}\n")
            return 1

        if args.list:
            ggufs = [f for f in all_files if f.lower().endswith(".gguf")]
            others = [f for f in all_files if not f.lower().endswith(".gguf")]
            print(f"# {repo_id} (rev={revision or 'default'})")
            print(f"## GGUF ({len(ggufs)})")
            for f in ggufs:
                print(f"  {f}")
            if others:
                print(f"## その他 ({len(others)})")
                for f in others:
                    print(f"  {f}")
            return 0

        files_to_get = filter_files(all_files, args.pattern or ([] if filename else []))
        if filename:
            # filename もリストに含まれていればそのまま使う
            if filename not in files_to_get:
                files_to_get.insert(0, filename)

    if not files_to_get:
        sys.stderr.write("対象ファイルが見つかりませんでした (パターンを見直してください)。\n")
        return 1

    # 確認
    print(f"レポジトリ : {repo_id}")
    print(f"リビジョン : {revision or 'default'}")
    print(f"保存先     : {dest}")
    print(f"ファイル数 : {len(files_to_get)}")
    for f in files_to_get:
        print(f"  - {f}")

    if not confirm("ダウンロードを開始しますか？", args.yes):
        print("キャンセルしました。")
        return 0

    # 高速転送 (ある場合)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    failed: List[Tuple[str, str]] = []
    for i, f in enumerate(files_to_get, 1):
        print(f"\n[{i}/{len(files_to_get)}] {f}")
        try:
            path = download_one(
                repo_id=repo_id,
                filename=f,
                dest=dest,
                revision=revision,
                token=token,
                flat=not args.nested,
            )
            print(f"  -> {path}")
        except KeyboardInterrupt:
            print("\n中断されました (途中までのデータは保持されます。再実行で続きから再開します)。")
            return 130
        except Exception as e:
            print(f"  失敗: {e}")
            failed.append((f, str(e)))

    if failed:
        print("\n失敗したファイル:")
        for f, e in failed:
            print(f"  - {f}: {e}")
        return 1

    print("\n完了しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
