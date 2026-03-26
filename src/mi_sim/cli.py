from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

from .client_llm_loader import build_client_llms
from .conversation_environment import ConversationEnvironment, ConversationTurn
from .counselor_llm_loader import build_counselor_stack
from .env_utils import load_openai_api_key
from .paths import resolve_config_path, resolve_env_path
from .perma_client_agent import (
    DEFAULT_CLIENT_CODE,
    DEFAULT_FIRST_CLIENT_UTTERANCE,
    SimpleClientLLM,
)
from .self_play_batch import (
    build_batch_case_plan,
    build_batch_client_codes,
    build_batch_run_dir,
    load_available_client_codes,
)
from .session_log_tools import finalize_session


def _load_simulation_settings() -> Dict[str, Any]:
    path = resolve_config_path("simulation_settings.yaml")
    if not path.exists():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_self_play_client_style(client_style: Optional[str]) -> str:
    sim_cfg = _load_simulation_settings()
    style_cfg = ((sim_cfg.get("client") or {}).get("style") if isinstance(sim_cfg, dict) else None) or None
    return (client_style or os.getenv("CLIENT_STYLE") or style_cfg or "auto")


def _print_simulation_log(log: List[ConversationTurn]) -> None:
    for idx, turn in enumerate(log):
        prefix = "C" if turn.speaker == "client" else "T"
        print(f"[{idx:02d}] {prefix}: {turn.text}")
        if turn.meta and turn.speaker == "counselor":
            phase = turn.meta.get("phase")
            action = turn.meta.get("main_action")
            print(f"      meta: phase={phase}, action={action}")


def run_self_play(
    *,
    script_name: str = "mi_sim.self_play",
    first_client_utterance: str = DEFAULT_FIRST_CLIENT_UTTERANCE,
    max_turns: int = 15,
    max_turns_completion: str = "phase_to_closing",
    max_total_turns: Optional[int] = None,
    client_style: Optional[str] = None,
    client_code: Optional[str] = None,
    logs_dir: Optional[Path] = None,
    log_prefix: str = "session_simulation",
    artifact_id: Optional[str] = None,
    print_full_log: bool = True,
) -> Dict[str, str]:
    api_key = load_openai_api_key()
    counselor_mode = os.getenv("COUNSELOR_MODE", "counselor_llm")
    counselor_stack = build_counselor_stack(api_key=api_key)
    counselor = counselor_stack["counselor"]
    counselor_llm = counselor_stack["llm"]
    counselor_cfg = counselor_stack["counselor_cfg"]
    counselor_slot_fill_cfg = counselor_stack.get("slot_fill_cfg", {})
    counselor_action_cfg = counselor_stack["action_cfg"]
    risk_detector_cfg = counselor_stack["risk_cfg"]
    mi_evaluator_cfg = counselor_stack["mi_eval_cfg"]

    client_llms = build_client_llms(api_key=api_key)
    client_state_cfg = client_llms["client_state_cfg"]
    client_reply_cfg = client_llms["client_reply_cfg"]
    client_llm_state = client_llms["client_llm_state"]
    client_llm_reply = client_llms["client_llm_reply"]

    client_code_effective = (client_code or os.getenv("CLIENT_CODE") or DEFAULT_CLIENT_CODE).strip() or DEFAULT_CLIENT_CODE
    client_style_effective = _resolve_self_play_client_style(client_style)
    client, client_bundle = SimpleClientLLM.from_profile(
        client_code=client_code_effective,
        llm=client_llm_reply,
        llm_state=client_llm_state,
        llm_reply=client_llm_reply,
        env_style=client_style_effective,
        first_client_utterance_env=os.getenv("FIRST_CLIENT_UTTERANCE"),
        max_state_step_env=os.getenv("CLIENT_MAX_STATE_STEP"),
        default_first_utterance=first_client_utterance,
    )

    session_meta = {
        "session_mode": "self_play",
        "openai_model": counselor_cfg.get("model", ""),
        "reasoning_effort": counselor_cfg.get("reasoning_effort", ""),
        "script_name": script_name,
        "client_style": client_bundle.style,
        "client_code": client_code_effective,
        "client_pattern": client_bundle.derived_meta.get("pattern_code", ""),
        "client_pattern_label": client_bundle.derived_meta.get("pattern_label", ""),
        "client_primary_focus": client_bundle.derived_meta.get("primary_focus_code", ""),
        "client_primary_focus_label": client_bundle.derived_meta.get("primary_focus_label", ""),
        "client_interpersonal_style": client_bundle.derived_meta.get("interpersonal_style_code", ""),
        "client_age_range": client_bundle.derived_meta.get("age_range", ""),
        "client_sex": client_bundle.derived_meta.get("sex", ""),
        "client_marital_status": client_bundle.derived_meta.get("marital_status", ""),
        "client_profiles_path": str(client_bundle.profiles_path),
        "client_model_state": client_state_cfg.get("model", ""),
        "client_model_reply": client_reply_cfg.get("model", ""),
        "phase_slot_filler_model": counselor_slot_fill_cfg.get("model", "") if counselor_slot_fill_cfg.get("enabled") else "",
        "action_ranker_model": counselor_action_cfg.get("model", "") if counselor_action_cfg.get("enabled") else "",
        "risk_detector_model": risk_detector_cfg.get("model", "") if risk_detector_cfg.get("enabled") else "",
        "mi_evaluator_model": mi_evaluator_cfg.get("model", "") if mi_evaluator_cfg.get("enabled") else "",
        "counselor_mode": counselor_mode,
        "planner_config": asdict(counselor.cfg),
    }

    env = ConversationEnvironment(counselor=counselor, client=client, session_meta=session_meta)
    env.reset()
    try:
        init_state = client.get_internal_state()  # type: ignore[attr-defined]
        if isinstance(init_state, dict):
            env.session_meta["client_initial_state"] = dict(init_state)
    except Exception:
        pass

    print("自己対話シミュレーションを開始します...")
    env.simulate(
        first_client_utterance=client_bundle.first_utterance,
        max_turns=max_turns,
        progress=True,
        max_turns_completion=max_turns_completion,
        max_total_turns=max_total_turns,
    )

    if print_full_log:
        print("シミュレーション完了。ログを出力します。")
        print("\n==== SIMULATION LOG ====")
        _print_simulation_log(env.log)
    else:
        print("シミュレーション完了。成果物を保存します。")

    return finalize_session(
        env,
        counselor_llm,
        log_prefix=log_prefix,
        logs_dir=logs_dir,
        artifact_id=artifact_id,
    )


def run_self_play_batch(
    *,
    script_name: str = "mi_sim.self_play_batch",
    first_client_utterance: str = DEFAULT_FIRST_CLIENT_UTTERANCE,
    max_turns: int = 15,
    max_turns_completion: str = "phase_to_closing",
    max_total_turns: Optional[int] = None,
    client_style: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> None:
    profiles_path = SimpleClientLLM._find_client_profiles_path()
    available_codes = load_available_client_codes(profiles_path)
    client_codes = build_batch_client_codes(available_codes)
    batch_root_dir = logs_dir or (Path.cwd() / "logs" / "mi_sim" / "self_play_batch")
    batch_dir = build_batch_run_dir(
        batch_root_dir,
        max_turns=max_turns,
        max_turns_completion=max_turns_completion,
        max_total_turns=max_total_turns,
        client_style=_resolve_self_play_client_style(client_style),
    )
    cases = build_batch_case_plan(batch_dir, client_codes)
    pending_cases = [case for case in cases if not case.is_complete]

    print("自己対話シミュレーションの15ケース一括実行を開始します。")
    print(f"出力先: {batch_dir}")
    print(f"対象={len(cases)} / 既完了={len(cases) - len(pending_cases)} / 今回実行={len(pending_cases)}")

    if not pending_cases:
        print("CSV と client_eval.json が揃っているため、再実行はありません。")
        return

    batch_dir.mkdir(parents=True, exist_ok=True)
    total = len(cases)
    for case in cases:
        if case.is_complete:
            print(f"[SKIP {case.order:02d}/{total}] {case.client_code}")
            continue

        print(f"[RUN  {case.order:02d}/{total}] {case.client_code}")
        artifacts = run_self_play(
            script_name=script_name,
            first_client_utterance=first_client_utterance,
            max_turns=max_turns,
            max_turns_completion=max_turns_completion,
            max_total_turns=max_total_turns,
            client_style=client_style,
            client_code=case.client_code,
            logs_dir=batch_dir,
            log_prefix=case.log_prefix,
            artifact_id=case.artifact_id,
            print_full_log=False,
        )
        print(
            f"[DONE {case.order:02d}/{total}] {case.client_code} "
            f"csv={artifacts.get('csv', '')} json={artifacts.get('client_eval', '')}"
        )

    print("15ケース一括実行が完了しました。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mi-sim",
        description="MI self-play simulation package CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    self_play = sub.add_parser("self-play", help="LLM 同士の自己対話シミュレーション")
    self_play.add_argument("--max-turns", type=int, default=5, help="カウンセラー→クライアントのターン数")
    self_play.add_argument(
        "--max-turns-completion",
        choices=["hard_stop", "phase_to_closing"],
        default="phase_to_closing",
        help="max-turns 到達後の終了方式",
    )
    self_play.add_argument(
        "--max-total-turns",
        type=int,
        default=None,
        help="phase_to_closing 時の安全上限（未指定なら max-turns + 7）",
    )
    self_play.add_argument(
        "--client-style",
        choices=["auto", "cooperative", "ambivalent", "resistant"],
        default=None,
        help="クライアントの対話スタイル",
    )
    self_play.add_argument("--client-code", default=None, help="単発実行する CLIENT_CODE。'all' で15ケース一括実行。")
    self_play.add_argument("--all-cases", action="store_true", help="15ケース一括実行")
    self_play.add_argument("--logs-dir", default=None, help="ログ出力先ディレクトリ")
    self_play.add_argument("--artifact-id", default=None, help="成果物ファイル名に付与する識別子")
    self_play.add_argument(
        "--first-client-utterance",
        default=DEFAULT_FIRST_CLIENT_UTTERANCE,
        help="初回クライアント発話のデフォルト値",
    )
    self_play.add_argument(
        "--no-print-full-log",
        action="store_true",
        help="単発実行時に対話ログ全文を表示しない",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv(dotenv_path=resolve_env_path())
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command != "self-play":
        parser.error("unknown command")

    logs_dir = Path(args.logs_dir).expanduser() if args.logs_dir else None
    run_all_cases = bool(args.all_cases) or str(args.client_code or "").strip().lower() == "all"
    if run_all_cases:
        run_self_play_batch(
            first_client_utterance=args.first_client_utterance,
            max_turns=args.max_turns,
            max_turns_completion=args.max_turns_completion,
            max_total_turns=args.max_total_turns,
            client_style=args.client_style,
            logs_dir=logs_dir,
        )
    else:
        run_self_play(
            first_client_utterance=args.first_client_utterance,
            max_turns=args.max_turns,
            max_turns_completion=args.max_turns_completion,
            max_total_turns=args.max_total_turns,
            client_style=args.client_style,
            client_code=args.client_code,
            logs_dir=logs_dir,
            artifact_id=args.artifact_id,
            print_full_log=not bool(args.no_print_full_log),
        )
    return 0
