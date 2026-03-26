# flourishing-mi-sim

`mi_sim` is a Python package for running self-play simulations between an MI-based counselor and a client.

The public release currently includes only the self-play execution core. It does not include the CLI for human counselors, training scripts, or research support code.

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

Create a `.env` file based on `.env.example` and set at least `OPENAI_API_KEY`.

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your_api_key_here
```

You can override the following environment variables if needed:

- `CLIENT_CODE`
- `CLIENT_STYLE`
- `PHASE_SLOT_QUALITY_MIN_THRESHOLD`

## Quick Start

Run a single self-play session:

```bash
mi-sim self-play --max-turns 5
```

Or:

```bash
python -m mi_sim self-play --max-turns 5
```

Run all 15 cases in a batch:

```bash
mi-sim self-play --all-cases --max-turns 8
```

## Outputs

By default, output is written to `logs/mi_sim/` under the current working directory.

- Single run: CSV and `client_eval.json`
- Batch run: per-case artifacts under `logs/mi_sim/self_play_batch/`

You can change the output directory with `--logs-dir`.

## CLI

Common options:

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

See the full help output with:

```bash
mi-sim self-play --help
```

## Scope

This public repository does not include:

- API keys and `.env`
- Personal data or private datasets
- Monorepo-specific files such as `app/`, `utility/`, `vendor/`, `results/`, and `.github/`
- Training-oriented DSPy code and archives

For subtree maintenance instructions, see `MAINTAINERS.md`.
