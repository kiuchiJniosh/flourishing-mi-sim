from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import yaml


BATCH_PERMA_PREFIXES = ("LANG", "SOCIAL", "UNSOCIAL")
BATCH_CASE_SUFFIXES = ("MGR", "LOWINC", "ISO", "STABLE", "MOB")
DEFAULT_BATCH_ARTIFACT_ID = "result"


@dataclass(frozen=True)
class SelfPlayBatchCase:
    order: int
    client_code: str
    log_prefix: str
    artifact_id: str
    csv_path: Path
    client_eval_path: Path

    @property
    def is_complete(self) -> bool:
        return self.csv_path.is_file() and self.client_eval_path.is_file()


def load_available_client_codes(profiles_path: Path) -> list[str]:
    with profiles_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError("client_profiles.yaml must contain a mapping at the root.")

    clients = data.get("clients")
    if not isinstance(clients, dict):
        raise KeyError("client_profiles.yaml must contain a 'clients' mapping.")

    codes = [str(code).strip() for code in clients.keys() if str(code).strip()]
    if not codes:
        raise ValueError("No client codes were found in client_profiles.yaml.")
    return codes


def build_batch_client_codes(
    available_codes: Iterable[str],
    *,
    perma_prefixes: Sequence[str] = BATCH_PERMA_PREFIXES,
    case_suffixes: Sequence[str] = BATCH_CASE_SUFFIXES,
) -> list[str]:
    normalized = {str(code).strip().upper() for code in available_codes if str(code).strip()}
    ordered_codes = [
        f"{perma_prefix}_{case_suffix}"
        for case_suffix in case_suffixes
        for perma_prefix in perma_prefixes
    ]
    missing = [code for code in ordered_codes if code not in normalized]
    if missing:
        raise ValueError(
            "Missing client codes required for batch self-play: " + ", ".join(missing)
        )
    return ordered_codes


def resolve_effective_max_total_turns(
    *,
    max_turns: int,
    max_turns_completion: str,
    max_total_turns: Optional[int],
) -> Optional[int]:
    completion_mode = str(max_turns_completion or "hard_stop").strip().lower()
    if completion_mode != "phase_to_closing":
        return None
    if max_total_turns is None:
        return max(0, int(max_turns)) + 7
    return max(max(0, int(max_turns)), int(max_total_turns))


def build_batch_run_dir(
    base_dir: Path,
    *,
    max_turns: int,
    max_turns_completion: str,
    max_total_turns: Optional[int],
    client_style: str,
) -> Path:
    completion_mode = str(max_turns_completion or "hard_stop").strip().lower() or "hard_stop"
    parts = [f"max_turns_{int(max_turns)}", completion_mode]
    effective_max_total = resolve_effective_max_total_turns(
        max_turns=max_turns,
        max_turns_completion=completion_mode,
        max_total_turns=max_total_turns,
    )
    if effective_max_total is not None:
        parts.append(f"max_total_turns_{effective_max_total}")
    parts.append(f"style_{(client_style or 'auto').strip().lower() or 'auto'}")
    return base_dir / "__".join(parts)


def build_batch_case_plan(
    batch_dir: Path,
    client_codes: Sequence[str],
    *,
    artifact_id: str = DEFAULT_BATCH_ARTIFACT_ID,
) -> list[SelfPlayBatchCase]:
    resolved_artifact_id = str(artifact_id).strip() or DEFAULT_BATCH_ARTIFACT_ID
    cases: list[SelfPlayBatchCase] = []
    for order, client_code in enumerate(client_codes, start=1):
        normalized_code = str(client_code).strip().upper()
        log_prefix = f"session_simulation_{order:02d}_{normalized_code}"
        csv_path = batch_dir / f"{log_prefix}_{resolved_artifact_id}.csv"
        client_eval_path = batch_dir / f"{log_prefix}_{resolved_artifact_id}_client_eval.json"
        cases.append(
            SelfPlayBatchCase(
                order=order,
                client_code=normalized_code,
                log_prefix=log_prefix,
                artifact_id=resolved_artifact_id,
                csv_path=csv_path,
                client_eval_path=client_eval_path,
            )
        )
    return cases
