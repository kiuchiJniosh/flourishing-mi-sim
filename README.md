# flourishing-mi-sim

`mi_sim` は、動機づけ面接（MI）ベースのカウンセラーとクライアントによる self-play シミュレーションを実行するための Python package です。

現状の公開範囲は self-play 実行コアです。人間カウンセラー向け CLI、学習用スクリプト、研究用補助コードは含みません。

## Contents

- `src/mi_sim/`
- `src/mi_sim/config/`
- `pyproject.toml`
- `.env.example`

## Requirements

- Python 3.11+
- OpenAI API key

## Installation

```bash
python -m pip install -e .
```

## Setup

`.env.example` を参考に `.env` を作成し、少なくとも `OPENAI_API_KEY` を設定してください。

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your_api_key_here
```

必要に応じて次の環境変数も上書きできます。

- `CLIENT_CODE`
- `CLIENT_STYLE`
- `PHASE_SLOT_QUALITY_MIN_THRESHOLD`

## Quick Start

単発の self-play:

```bash
mi-sim self-play --max-turns 5
```

または:

```bash
python -m mi_sim self-play --max-turns 5
```

15ケース一括実行:

```bash
mi-sim self-play --all-cases --max-turns 8
```

## Outputs

既定では実行結果をカレントディレクトリ配下の `logs/mi_sim/` に保存します。

- 単発実行: CSV と `client_eval.json`
- 一括実行: `logs/mi_sim/self_play_batch/` 配下にケースごとの成果物

出力先は `--logs-dir` で変更できます。

## CLI

主なオプション:

- `--max-turns`
- `--max-turns-completion {hard_stop,phase_to_closing}`
- `--max-total-turns`
- `--client-style {auto,cooperative,ambivalent,resistant}`
- `--client-code <CLIENT_CODE>`
- `--all-cases`
- `--logs-dir <DIR>`
- `--artifact-id <ID>`
- `--first-client-utterance <TEXT>`
- `--no-print-full-log`

詳細は次で確認できます。

```bash
mi-sim self-play --help
```

## Scope

この公開 repo には次を含めません。

- API キーや `.env`
- 個人情報や非公開データ
- `app/`, `utility/`, `vendor/`, `results/`, `.github/` など monorepo 固有ファイル
- 学習用の DSPy コードや archive

maintainer 向けの subtree 運用手順は `MAINTAINERS.md` を参照してください。
