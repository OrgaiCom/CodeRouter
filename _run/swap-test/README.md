# swap-test — launcher.swap (Phase 1) 実機テストキット

修正版 CodeRouter の「モデル自動スワップ」(オンデマンド起動 / ロード保留 / TTLアンロード)を、Mac のターミナルで 1 本実行するだけで検証するキット。設計書: `docs/designs/launcher-model-swap.md`。

## 1. インストール(修正版の反映)

修正版は作業ツリーに反映済みなので、リポジトリ直下で editable インストールを更新するだけ:

```bash
cd ~/works/project/CodeRouter
uv sync                      # または: uv pip install -e .
uv run coderouter --version  # 動作確認
uv run python -c "import coderouter.launcher_swap; print('swap OK')"
```

`uv run pytest tests/ -q` が 1725 passed になることは確認済み(2026-07-12)。

## 2. 前提

1. **llama-server**(llama.cpp)がビルド済みで実行できること。PATH に無い場合は `LLAMA_SERVER=/path/to/llama-server` で指定。
2. **GGUF が 1 つ以上**手元にあること。テストは小さいモデルほど速い(1〜4GB 級推奨。例: Qwen3 系 4B/8B の Q4_K_M)。`~/models/*.gguf` か リポジトリの `models/*.gguf` にあれば自動で拾う(最小サイズを選択)。別の場所なら `GGUF_PATH=` で指定。
   - まだ無い場合の入手例: `uv run python gguf_dl.py`(リポジトリ同梱のダウンローダ)や huggingface から任意の GGUF を `~/models/` に置く。
3. ポート 8288(CodeRouter)と 18081(テストモデル)が空いていること。既存の serve(8088 等)とは共存可。
4. Ollama は**不要**(このテストは llama.cpp 直のみ使う)。

## 3. 実行

```bash
cd ~/works/project/CodeRouter/_run/swap-test
bash run_swap_test.sh
# 例: bash run_swap_test.sh だけでOK。カスタムする場合:
#   GGUF_PATH=~/models/Qwen3-4B-Q4_K_M.gguf LLAMA_SERVER=~/llama.cpp/build/bin/llama-server bash run_swap_test.sh
```

所要はモデルロード時間×3+TTL待ち(既定25s)で、小型モデルなら **2〜4 分**。終了時に `results-<ts>/report.md` が出る。FAIL があればフォルダごと Claude に共有(判定します)。

## 4. 何を検証しているか

| Phase | 検証内容 | 対応する実装 |
|---|---|---|
| 0 | 修正版の導入確認(`coderouter.launcher_swap` の存在)、llama-server / GGUF / ポート | - |
| A | **コールドスタート**: 未起動モデル名へのリクエスト → 自動 spawn → ロード完了まで保留 → 応答。`coderouter_provider=launcher-swap-*` と `swap:<name>` ルール発火をログで確認 | オンデマンド spawn + readiness 保留 + auto_router ルール自動注入 |
| B | **ウォーム**: 2回目は再 spawn せず(LISTENプロセス1個のまま)高速応答 | thundering-herd 対策 / fast-path |
| C | **カタログ外 model 名**: フォールスルー先(swapプロファイル)で応答 | M-3 修正(カタログ非一致 model のリース保護経路) |
| D | **TTLアンロード**: アイドル `TTL` 秒+sweep で `swap-unload` 発火、ポート解放 | TTL sweeper + deregister_provider + 意図的停止経路(auto-restart 不発) |
| E | **再spawn**: アンロード後のリクエストで再び自動起動して応答 | 状態リセット(poison 化しないこと) |

## 5. テスト用設定について

`providers.tpl.yaml` から `results-<ts>/providers.generated.yaml` を生成して使う(手元の providers.yaml には触れない)。テストを速くするため `ttl_seconds` は既定 25 秒、`sweep_interval_s: 5` にしてある。実運用サンプルは `examples/providers.swap.yaml` を参照。

## 6. トラブルシューティング早見

| 症状 | 対処 |
|---|---|
| preflight で launcher_swap が import できない | 修正版が入っていない。`uv sync` 後に再実行。リポジトリ直下の作業ツリーで実行しているか確認 |
| PhaseA がタイムアウト(330s) | モデルが大きすぎるかロードが遅い。小さい GGUF を `GGUF_PATH=` で指定。`results-*/serve.log` の readiness ログを確認 |
| PhaseA で port は開くが応答が空 | llama-server は起動したが生成に失敗。`serve.log` と `-ngl 99` が手元の VRAM に合うかを確認(必要なら providers.tpl.yaml の extra_args を調整) |
| PhaseD で unload されない | リクエストが残っている(ストリーミング中は unload されない仕様)か、TTL/sweep 設定。`serve.log` の swap-unload 有無を確認 |
| serve 起動直後に 18081 が LISTEN | swap ではなく別プロセスが同ポートを使用。MODEL_PORT=別番号 で再実行 |
