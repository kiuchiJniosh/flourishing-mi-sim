from __future__ import annotations

import csv
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
from .paths import resolve_config_path, resolve_project_path

if TYPE_CHECKING:
    # 実行時にはインポートしない（循環依存を避けるため）
    from .conversation_environment import ConversationTurn
    from .mi_counselor_agent import LLMClient


# ==============================
# Protocol for session log handling
# ==============================

class SupportsConversationTurn(Protocol):
    speaker: str
    text: str
    meta: Optional[Dict[str, Any]]


class SupportsSessionLog(Protocol):
    """
    session_log_tools が受け付ける最小限のインタフェース。
    - log: ConversationTurn 互換の配列（speaker/text/meta があればOK）
    - session_meta: セッション共通メタデータ（任意のMapping）
    - to_json_serializable(): JSON保存用のレコード配列を返す
    """

    log: Sequence[SupportsConversationTurn]
    session_meta: Mapping[str, Any]

    def to_json_serializable(self) -> List[Dict[str, Any]]:
        ...


CSV_SCHEMA_VERSION = "2.11"
_CSV_CELL_MAX_CHARS = 32000
_MAX_SLOT_QUALITY_TARGET_EXAMPLES_FOR_CSV = 3
_SESSION_EVAL_PROMPT_CONFIG_PATH = resolve_config_path("layer3_layer4_action_prompts.yaml")
_SESSION_EVAL_MI_KNOWLEDGE_PATH = resolve_config_path("mi_knowledge.md")
_REFLECT_ACTION_NAMES = {
    "REFLECT",
    "REFLECT_SIMPLE",
    "REFLECT_COMPLEX",
    "REFLECT_DOUBLE",
}
_REFLECT_ENDING_FAMILY_DETECTOR: Any = None
_REFLECT_ENDING_FAMILY_DETECTOR_LOADED = False
_DEFAULT_CLIENT_FEEDBACK_PROMPT_TEMPLATE = """あなたは、以下の CLIENT_CODE で設定されたクライアント本人です。
セッション直後に自分の体験を振り返り、自然な日本語で1段落だけ書いてください。
ratings は今の自己評価、session log は実際のやり取りです。両方を踏まえてください。
約400字で、目安は320〜480字です。
必ず含めること:
- このセッションがどんな体験だったか
- 良かった点
- 物足りなかった点、まだ残る不安や迷い
- このあと自分がどうなりそうか、何を試しそうか
箇条書き・見出しは禁止です。日本語の本文だけを JSON に入れてください。

出力は必ず次の JSON 形式だけ:
{{
  "client_feedback": "約400字の日本語本文"
}}

=== CLIENT_CONTEXT ===
{client_context_json}

=== RATINGS ===
{ratings_json}

=== FINAL_CLIENT_INTERNAL_STATE ===
{final_client_internal_state_json}

=== SESSION_LOG ===
{session_log_text}
"""
_DEFAULT_SUPERVISOR_FEEDBACK_PROMPT_TEMPLATE = """あなたは動機づけ面接（MI）の熟練スーパーバイザーです。
以下のセッションをレビューし、カウンセラーへのフィードバックを自然な日本語の1段落で書いてください。
約400字で、目安は320〜480字です。
必ず含めること:
- カウンセラーの良い点
- 課題点
- 次回に向けた改善の示唆
MIの観点から、反映・是認・焦点化・自律性支援・喚起の質を具体的に見てください。
箇条書き・見出しは禁止です。日本語の本文だけを JSON に入れてください。

出力は必ず次の JSON 形式だけ:
{{
  "supervisor_feedback": "約400字の日本語本文"
}}

以下の MI 知識全文を文脈知識として参照し、判断の土台にしてください。セッションの実際の文脈と矛盾する場合は、セッション本文を優先してください。

=== MI_KNOWLEDGE ===
{mi_knowledge_text}

=== RATINGS ===
{ratings_json}

=== FINAL_CLIENT_INTERNAL_STATE ===
{final_client_internal_state_json}

=== SESSION_LOG ===
{session_log_text}
"""


def _to_json_records(env: SupportsSessionLog) -> List[Dict[str, Any]]:
    """to_json_serializable() の実行と簡易検証を一元化。"""
    records = env.to_json_serializable()
    if not isinstance(records, list):
        raise TypeError("to_json_serializable() は list を返す必要があります。")
    return records


@lru_cache(maxsize=1)
def _load_session_eval_prompt_config() -> Dict[str, Any]:
    try:
        if not _SESSION_EVAL_PROMPT_CONFIG_PATH.exists():
            return {}
        loaded = yaml.safe_load(_SESSION_EVAL_PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            return dict(loaded)
    except Exception:
        return {}
    return {}


def _load_session_eval_mi_knowledge_text() -> str:
    raw = (os.getenv("MI_KNOWLEDGE_MD_PATH") or "").strip()
    if raw:
        knowledge_path = resolve_project_path(raw)
    else:
        knowledge_path = _SESSION_EVAL_MI_KNOWLEDGE_PATH

    try:
        if not knowledge_path.exists() or not knowledge_path.is_file():
            return ""
        return knowledge_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _get_session_eval_prompt_template(prompt_key: str, default: str) -> str:
    root = _load_session_eval_prompt_config().get("session_evaluation_prompts")
    if isinstance(root, Mapping):
        value = root.get(prompt_key)
        if isinstance(value, str) and value.strip():
            return value.strip("\n")
    return default


def _format_session_eval_prompt(template: str, **kwargs: Any) -> str:
    try:
        return str(template).format(**kwargs)
    except Exception:
        return str(template)


def _normalize_csv_cell(value: Any) -> Any:
    """
    CSV をスプレッドシート/行指向ツールで扱いやすくするため、
    セル内改行を可視化文字列 "\\n" に正規化し、
    長すぎるセルは安全な長さまで切り詰める。
    """
    if not isinstance(value, str):
        return value
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", "\\n")
    if len(normalized) <= _CSV_CELL_MAX_CHARS:
        return normalized
    omitted = len(normalized) - _CSV_CELL_MAX_CHARS
    marker = f"...[TRUNCATED:{omitted}]"
    head_len = max(0, _CSV_CELL_MAX_CHARS - len(marker))
    return normalized[:head_len] + marker


def _normalize_csv_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _normalize_csv_cell(value) for key, value in row.items()}


def _split_csv_cell_into_two(value: str) -> Tuple[str, str]:
    if len(value) <= _CSV_CELL_MAX_CHARS:
        return value, ""
    return value[:_CSV_CELL_MAX_CHARS], value[_CSV_CELL_MAX_CHARS:]


def _extract_last_fallback_event(fallback_events: Any) -> Dict[str, Any]:
    if isinstance(fallback_events, Mapping):
        for nested_key in ("events", "attempts", "history"):
            nested = fallback_events.get(nested_key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                for event in reversed(nested):
                    if isinstance(event, Mapping):
                        return dict(event)
        return dict(fallback_events)
    if isinstance(fallback_events, Sequence) and not isinstance(fallback_events, (str, bytes)):
        for event in reversed(fallback_events):
            if isinstance(event, Mapping):
                return dict(event)
    return {}


def _is_reflect_action_name(action: Any) -> bool:
    return str(action or "").strip() in _REFLECT_ACTION_NAMES


def _get_reflect_ending_family_detector() -> Any:
    global _REFLECT_ENDING_FAMILY_DETECTOR_LOADED, _REFLECT_ENDING_FAMILY_DETECTOR
    if not _REFLECT_ENDING_FAMILY_DETECTOR_LOADED:
        try:
            from .mi_counselor_agent import _detect_reflect_ending_family
        except Exception:
            _REFLECT_ENDING_FAMILY_DETECTOR = None
        else:
            _REFLECT_ENDING_FAMILY_DETECTOR = _detect_reflect_ending_family
        _REFLECT_ENDING_FAMILY_DETECTOR_LOADED = True
    return _REFLECT_ENDING_FAMILY_DETECTOR


def _detect_reflect_ending_family_for_csv(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.strip()
    if not normalized:
        return ""
    detector = _get_reflect_ending_family_detector()
    if detector is None:
        return ""
    try:
        detected = detector(normalized)
    except Exception:
        return ""
    return str(detected or "").strip()


def _flatten_reflect_ending_family_columns(
    *,
    speaker: Any,
    action: Any,
    debug: Mapping[str, Any],
    draft_response_text: str,
    final_response_text: str,
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "layer3_draft_ending_family": "",
        "layer4_final_ending_family": "",
        "layer3_preferred_ending_family": "",
        "layer3_avoid_recent_ending_families_json": "",
        "layer3_recent_reflect_families_json": "",
        "layer3_recent_family_counts_json": "",
        "layer3_recent_same_family_count": "",
        "layer3_ending_family_lookback_assistant_turns": "",
        "layer3_ending_family_repetition_threshold": "",
        "layer3_ending_family_bias_warning": "",
        "layer4_changed_ending_family": "",
    }
    if str(speaker or "") != "counselor" or not _is_reflect_action_name(action):
        return defaults

    response_brief = debug.get("response_brief")
    writer_plan: Mapping[str, Any] = {}
    if isinstance(response_brief, Mapping):
        raw_writer_plan = response_brief.get("writer_plan")
        if isinstance(raw_writer_plan, Mapping):
            writer_plan = raw_writer_plan

    preferred_ending_family = str(writer_plan.get("preferred_ending_family", "") or "")
    avoid_recent_ending_families_json = ""
    raw_avoid_recent = writer_plan.get("avoid_recent_ending_families")
    if isinstance(raw_avoid_recent, Sequence) and not isinstance(raw_avoid_recent, (str, bytes)):
        avoid_recent = [str(item).strip() for item in raw_avoid_recent if str(item).strip()]
        if avoid_recent:
            avoid_recent_ending_families_json = json.dumps(avoid_recent, ensure_ascii=False)

    layer3_validation = debug.get("layer3_draft_validation")
    ending_family_guidance: Mapping[str, Any] = {}
    if isinstance(layer3_validation, Mapping):
        raw_guidance = layer3_validation.get("ending_family_guidance")
        if isinstance(raw_guidance, Mapping):
            ending_family_guidance = raw_guidance

    layer3_family = str(ending_family_guidance.get("current_family", "") or "")
    if not layer3_family:
        layer3_family = _detect_reflect_ending_family_for_csv(draft_response_text)

    raw_recent_families = ending_family_guidance.get("recent_families")
    recent_families: List[str] = []
    if isinstance(raw_recent_families, Sequence) and not isinstance(raw_recent_families, (str, bytes)):
        recent_families = [str(item).strip() for item in raw_recent_families if str(item).strip()]

    raw_recent_family_counts = ending_family_guidance.get("recent_family_counts")
    recent_family_counts: Dict[str, int] = {}
    if isinstance(raw_recent_family_counts, Mapping):
        for key, value in raw_recent_family_counts.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            try:
                recent_family_counts[key_text] = int(value)
            except (TypeError, ValueError):
                continue

    recent_same_family_count: Any = ""
    if layer3_family:
        if layer3_family in recent_family_counts:
            recent_same_family_count = recent_family_counts.get(layer3_family, "")
        elif recent_families:
            recent_same_family_count = sum(1 for family in recent_families if family == layer3_family)

    warning = False
    if isinstance(layer3_validation, Mapping):
        soft_warnings = layer3_validation.get("soft_warnings")
        if isinstance(soft_warnings, Sequence) and not isinstance(soft_warnings, (str, bytes)):
            warning = "reflect_ending_family_bias" in soft_warnings
    if not warning:
        warning = bool(ending_family_guidance.get("warning"))

    layer4_family = _detect_reflect_ending_family_for_csv(final_response_text)
    if not layer4_family and draft_response_text == final_response_text:
        layer4_family = layer3_family

    layer4_changed_ending_family: Any = ""
    if layer3_family and layer4_family:
        layer4_changed_ending_family = layer3_family != layer4_family

    return {
        "layer3_draft_ending_family": layer3_family,
        "layer4_final_ending_family": layer4_family,
        "layer3_preferred_ending_family": preferred_ending_family,
        "layer3_avoid_recent_ending_families_json": avoid_recent_ending_families_json,
        "layer3_recent_reflect_families_json": (
            json.dumps(recent_families, ensure_ascii=False) if recent_families else ""
        ),
        "layer3_recent_family_counts_json": (
            json.dumps(recent_family_counts, ensure_ascii=False)
            if recent_family_counts
            else ""
        ),
        "layer3_recent_same_family_count": recent_same_family_count,
        "layer3_ending_family_lookback_assistant_turns": ending_family_guidance.get(
            "lookback_assistant_turns", ""
        ),
        "layer3_ending_family_repetition_threshold": ending_family_guidance.get(
            "repetition_threshold", ""
        ),
        "layer3_ending_family_bias_warning": warning,
        "layer4_changed_ending_family": layer4_changed_ending_family,
    }


def _flatten_assistant_fallback_columns(
    *,
    debug: Mapping[str, Any],
    default_action: str,
) -> Dict[str, Any]:
    fallback_events = debug.get("assistant_output_fallback")
    fallback_events_json = (
        json.dumps(fallback_events, ensure_ascii=False)
        if fallback_events is not None
        else ""
    )
    fallback_last = _extract_last_fallback_event(fallback_events)

    fallback_used = False
    if isinstance(fallback_events, Mapping) and "used" in fallback_events:
        fallback_used = bool(fallback_events.get("used"))
    elif fallback_last:
        if "used" in fallback_last:
            fallback_used = bool(fallback_last.get("used"))
        else:
            fallback_used = True

    fallback_stage = str(fallback_last.get("stage", "") or "")
    fallback_reason = str(fallback_last.get("reason", "") or "")
    fallback_action = str(fallback_last.get("action", "") or "")
    if fallback_used and not fallback_action:
        fallback_action = default_action

    return {
        "assistant_fallback_used": fallback_used,
        "assistant_fallback_stage": fallback_stage,
        "assistant_fallback_reason": fallback_reason,
        "assistant_fallback_action": fallback_action,
        "assistant_fallback_json": fallback_events_json,
    }


def _flatten_layer4_writer_columns(*, debug: Mapping[str, Any]) -> Dict[str, Any]:
    layer4_writer = debug.get("layer4_writer")
    if not isinstance(layer4_writer, Mapping):
        return {
            "layer4_writer_fallback_used": "",
            "layer4_writer_fallback_stage": "",
            "layer4_writer_timeout": "",
            "layer4_writer_error_type": "",
            "layer4_writer_error_reason": "",
        }
    return {
        "layer4_writer_fallback_used": bool(layer4_writer.get("fallback_used"))
        if "fallback_used" in layer4_writer
        else "",
        "layer4_writer_fallback_stage": str(layer4_writer.get("fallback_stage", "") or ""),
        "layer4_writer_timeout": bool(layer4_writer.get("timeout")) if "timeout" in layer4_writer else "",
        "layer4_writer_error_type": str(layer4_writer.get("error_type", "") or ""),
        "layer4_writer_error_reason": str(layer4_writer.get("error_reason", "") or ""),
    }


def _flatten_layer4_edit_audit_columns(*, debug: Mapping[str, Any]) -> Dict[str, Any]:
    audit_payload = debug.get("layer4_edit_audit")
    if not isinstance(audit_payload, Mapping):
        return {
            "layer4_edit_audit_checked": "",
            "layer4_edit_audit_issue_count": "",
            "layer4_edit_audit_subject_reversal": "",
            "layer4_edit_audit_viewpoint_reversal": "",
            "layer4_edit_audit_concrete_anchor_dropped": "",
            "layer4_edit_audit_over_abstracted": "",
            "layer4_edit_audit_json": "",
        }
    return {
        "layer4_edit_audit_checked": bool(audit_payload.get("checked")) if "checked" in audit_payload else "",
        "layer4_edit_audit_issue_count": audit_payload.get("issue_count", ""),
        "layer4_edit_audit_subject_reversal": (
            bool(audit_payload.get("subject_reversal"))
            if "subject_reversal" in audit_payload
            else ""
        ),
        "layer4_edit_audit_viewpoint_reversal": (
            bool(audit_payload.get("viewpoint_reversal"))
            if "viewpoint_reversal" in audit_payload
            else ""
        ),
        "layer4_edit_audit_concrete_anchor_dropped": (
            bool(audit_payload.get("concrete_anchor_dropped"))
            if "concrete_anchor_dropped" in audit_payload
            else ""
        ),
        "layer4_edit_audit_over_abstracted": (
            bool(audit_payload.get("over_abstracted"))
            if "over_abstracted" in audit_payload
            else ""
        ),
        "layer4_edit_audit_json": json.dumps(audit_payload, ensure_ascii=False),
    }


def _extract_assistant_validation_attempts(validation_payload: Any) -> List[Dict[str, Any]]:
    attempts_payload: Any = None
    if isinstance(validation_payload, Mapping):
        attempts_payload = validation_payload.get("attempts")
    elif isinstance(validation_payload, Sequence) and not isinstance(validation_payload, (str, bytes)):
        attempts_payload = validation_payload

    if not isinstance(attempts_payload, Sequence) or isinstance(attempts_payload, (str, bytes)):
        return []

    attempts: List[Dict[str, Any]] = []
    for attempt in attempts_payload:
        if isinstance(attempt, Mapping):
            attempts.append(dict(attempt))
    return attempts


def _extract_assistant_validation_attempt_by_stage(
    attempts: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> Dict[str, Any]:
    for attempt in reversed(attempts):
        if str(attempt.get("stage", "") or "") == stage:
            return dict(attempt)
    return {}


def _flatten_assistant_output_validation_columns(*, debug: Mapping[str, Any]) -> Dict[str, Any]:
    validation_payload = debug.get("assistant_output_validation")
    validation_json = (
        json.dumps(validation_payload, ensure_ascii=False)
        if validation_payload is not None
        else ""
    )

    attempts = _extract_assistant_validation_attempts(validation_payload)
    attempts_json = json.dumps(attempts, ensure_ascii=False) if attempts else ""
    initial_attempt = _extract_assistant_validation_attempt_by_stage(attempts, stage="initial")
    retry_attempt = _extract_assistant_validation_attempt_by_stage(attempts, stage="validation_retry")

    final_ok: Any = ""
    final_reason = ""
    if isinstance(validation_payload, Mapping):
        if "ok" in validation_payload:
            final_ok = validation_payload.get("ok", "")
        final_reason = str(validation_payload.get("reason", "") or "")

    return {
        "assistant_output_validation_ok": final_ok,
        "assistant_output_validation_reason": final_reason,
        "assistant_output_validation_initial_ok": initial_attempt.get("ok", ""),
        "assistant_output_validation_initial_reason": str(initial_attempt.get("reason", "") or ""),
        "assistant_output_validation_retry_ok": retry_attempt.get("ok", ""),
        "assistant_output_validation_retry_reason": str(retry_attempt.get("reason", "") or ""),
        "assistant_output_validation_attempts_json": attempts_json,
        "assistant_output_validation_json": validation_json,
    }


def _flatten_low_confidence_fallback_columns(
    *,
    debug: Mapping[str, Any],
    default_action: str,
) -> Dict[str, Any]:
    action_source = str(debug.get("action_source", "") or "")
    payload = debug.get("low_confidence_fallback")

    from_action = ""
    to_action = ""
    threshold: Any = ""
    confidence: Any = ""

    if isinstance(payload, Mapping):
        from_action = str(payload.get("from_action", "") or "")
        to_action = str(payload.get("to_action", "") or "")
        threshold = payload.get("threshold", "")
        confidence = payload.get("confidence", "")

    if action_source == "ranker_low_confidence_fallback":
        if not from_action:
            from_candidate = (
                debug.get("ranker_directive_main_action")
                or debug.get("ranker_proposed_main_action")
            )
            from_action = str(from_candidate or "")
        if not to_action:
            to_action = default_action
        if confidence in (None, ""):
            confidence = debug.get("ranker_confidence", "")
        if threshold in (None, ""):
            threshold = 0.45

    return {
        "low_confidence_from_action": from_action,
        "low_confidence_to_action": to_action,
        "low_confidence_threshold": threshold,
        "low_confidence_confidence": confidence,
    }


def _extract_slot_quality_target_examples(debug: Mapping[str, Any]) -> Any:
    phase_debug = debug.get("phase_debug")
    if not isinstance(phase_debug, Mapping):
        phase_debug = {}
    slot_review = phase_debug.get("slot_review")
    if not isinstance(slot_review, Mapping):
        slot_review = {}
    response_brief = debug.get("response_brief")
    if not isinstance(response_brief, Mapping):
        response_brief = {}
    integrator_debug = debug.get("response_integrator_debug")
    if not isinstance(integrator_debug, Mapping):
        integrator_debug = {}

    for container in (debug, phase_debug):
        if isinstance(container, Mapping) and "slot_quality_target_examples" in container:
            return container.get("slot_quality_target_examples")
        if isinstance(container, Mapping) and "slot_repair_hints" in container:
            return container.get("slot_repair_hints")

    candidates = (
        slot_review.get("slot_quality_target_examples"),
        response_brief.get("slot_quality_target_examples"),
        integrator_debug.get("slot_quality_target_examples"),
        slot_review.get("slot_repair_hints"),
        response_brief.get("slot_repair_hints"),
        integrator_debug.get("slot_repair_hints"),
    )
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 0:
                continue
            return value
        if isinstance(value, Mapping):
            if len(value) == 0:
                continue
            return value
        text = str(value).strip()
        if text:
            return value
    for value in candidates:
        if value is not None:
            return value
    return None


def _format_slot_quality_target_examples_for_csv(
    *,
    slot_quality_target_examples: Any,
    current_phase: str,
) -> str:
    if slot_quality_target_examples is None:
        return ""

    normalized_current_phase = str(current_phase or "").strip()
    if isinstance(slot_quality_target_examples, Mapping):
        hint_items: List[Any] = [slot_quality_target_examples]
    elif isinstance(slot_quality_target_examples, Sequence) and not isinstance(slot_quality_target_examples, (str, bytes)):
        hint_items = list(slot_quality_target_examples)
    else:
        raw = str(slot_quality_target_examples).strip()
        return f"・slot_unknown: {raw}" if raw else ""

    lines: List[str] = []
    seen = set()
    for item in hint_items:
        if isinstance(item, Mapping):
            hint_phase = str(item.get("phase", "") or "").strip()
            if normalized_current_phase and hint_phase and hint_phase != normalized_current_phase:
                continue
            slot_key = str(item.get("slot_key", "") or "").strip()
            slot_label = str(item.get("slot_label", "") or "").strip()
            slot_name = slot_label or slot_key or "slot_unknown"
            hint_text = str(item.get("detail", "") or item.get("target_information", "")).strip()
            if not hint_text:
                hint_text = str(
                    item.get("preferred_probe_style", "")
                    or item.get("issue_type", "")
                    or ""
                ).strip()
            if not slot_name:
                slot_name = "slot_unknown"
            if hint_text:
                line = f"・{slot_name}: {hint_text}"
            else:
                line = f"・{slot_name}"
        else:
            raw = str(item).strip()
            if not raw:
                continue
            line = f"・slot_unknown: {raw}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= _MAX_SLOT_QUALITY_TARGET_EXAMPLES_FOR_CSV:
            break
    return "\n".join(lines)


def _flatten_layer2_slot_review_columns(*, debug: Mapping[str, Any]) -> Dict[str, str]:
    phase_debug = debug.get("phase_debug")
    if not isinstance(phase_debug, Mapping):
        phase_debug = {}

    slot_review_payload: Mapping[str, Any] = {}
    for candidate in (debug.get("slot_review"), phase_debug.get("slot_review")):
        if isinstance(candidate, Mapping):
            slot_review_payload = candidate
            break

    slot_review_debug: Mapping[str, Any] = {}
    for candidate in (debug.get("slot_review_debug"), phase_debug.get("slot_review_debug")):
        if isinstance(candidate, Mapping):
            slot_review_debug = candidate
            break

    reviewed_updates = slot_review_payload.get("reviewed_updates")
    if not (
        isinstance(reviewed_updates, Sequence)
        and not isinstance(reviewed_updates, (str, bytes))
    ):
        reviewed_updates = phase_debug.get("reviewed_phase_slots")
    if not (
        isinstance(reviewed_updates, Sequence)
        and not isinstance(reviewed_updates, (str, bytes))
        and len(reviewed_updates) > 0
    ):
        reviewed_updates_json = ""
    else:
        reviewed_updates_json = json.dumps(reviewed_updates, ensure_ascii=False)

    slot_quality = slot_review_payload.get("slot_quality")
    if not (isinstance(slot_quality, Mapping) and len(slot_quality) > 0):
        slot_quality_json = ""
    else:
        slot_quality_json = json.dumps(slot_quality, ensure_ascii=False)

    raw_slot_reviewer_ran = phase_debug.get("parallel_with_slot_reviewer")
    if isinstance(raw_slot_reviewer_ran, bool):
        slot_reviewer_ran = str(raw_slot_reviewer_ran)
    else:
        method_for_inference = str(slot_review_debug.get("method", "") or "")
        if method_for_inference == "slot_reviewer_skipped_no_review_targets":
            slot_reviewer_ran = "False"
        elif method_for_inference:
            slot_reviewer_ran = "True"
        else:
            slot_reviewer_ran = ""

    return {
        "layer2_reviewed_updates_json": reviewed_updates_json,
        "layer2_slot_quality_json": slot_quality_json,
        "layer2_slot_reviewer_ran": slot_reviewer_ran,
        "layer2_slot_review_method": str(slot_review_debug.get("method", "") or ""),
        "layer2_slot_review_fallback_from": str(slot_review_debug.get("fallback_from", "") or ""),
        "layer2_slot_review_schema_issues_json": (
            json.dumps(slot_review_debug.get("schema_issues"), ensure_ascii=False)
            if slot_review_debug.get("schema_issues") is not None
            else ""
        ),
        "layer2_slot_review_contract_issues_json": (
            json.dumps(slot_review_debug.get("contract_issues"), ensure_ascii=False)
            if slot_review_debug.get("contract_issues") is not None
            else ""
        ),
        "layer2_slot_review_raw_output": (
            str(slot_review_debug.get("raw_output", "") or "")
            if slot_review_debug.get("raw_output") is not None
            else ""
        ),
    }


def _flatten_layer2_phase_snapshot_columns(*, debug: Mapping[str, Any]) -> Dict[str, str]:
    phase_debug = debug.get("phase_debug")
    if not isinstance(phase_debug, Mapping):
        phase_debug = {}

    slot_gate_payload: Mapping[str, Any] = {}
    for candidate in (debug.get("slot_gate"), phase_debug.get("slot_gate")):
        if isinstance(candidate, Mapping):
            slot_gate_payload = candidate
            break

    slot_gate_phase = str(slot_gate_payload.get("current_phase", "") or "")

    slot_values = slot_gate_payload.get("slot_values")
    if not (isinstance(slot_values, Mapping) and len(slot_values) > 0):
        phase_slots_after_review_json = ""
    else:
        phase_slots_after_review_json = json.dumps(slot_values, ensure_ascii=False)

    slot_quality = slot_gate_payload.get("slot_quality")
    if not (isinstance(slot_quality, Mapping) and len(slot_quality) > 0):
        phase_slot_quality_after_review_json = ""
    else:
        phase_slot_quality_after_review_json = json.dumps(slot_quality, ensure_ascii=False)

    return {
        "layer2_slot_gate_phase": slot_gate_phase,
        "layer2_phase_slots_after_review_json": phase_slots_after_review_json,
        "layer2_phase_slot_quality_after_review_json": phase_slot_quality_after_review_json,
    }


# ==============================
# 保存系：JSONL / CSV
# ==============================

def save_log_jsonl(env: SupportsSessionLog, path: str) -> None:
    """
    SupportsSessionLog 互換のセッションログを JSONL 形式で保存します。
    - 1行1発話（ConversationTurn）となるように json を書き出します。
    - 文字コードは UTF-8（BOM なし）です。
    """
    records = _to_json_records(env)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def save_log_csv(env: SupportsSessionLog, path: str) -> None:
    """
    SupportsSessionLog 互換のセッションログを CSV 形式で保存します。
    - 文字コードは BOM 付き UTF-8（Excel などで文字化けしにくくするため）。
    - 各行は 1発話（ConversationTurn）です。
    - counselor 発話のときだけ phase / main_action / add_affirm / debug 情報が入ります。
    - session_meta（session_id / mode / model / planner_config など）があれば各行に付与します。
    """
    records = _to_json_records(env)

    fieldnames = [
        "index",
        "speaker",
        "text",
        "draft_response_text",
        "layer3_draft_ending_family",
        "layer4_final_ending_family",
        "layer3_preferred_ending_family",
        "layer3_avoid_recent_ending_families_json",
        "layer3_recent_reflect_families_json",
        "layer3_recent_family_counts_json",
        "layer3_recent_same_family_count",
        "layer3_ending_family_lookback_assistant_turns",
        "layer3_ending_family_repetition_threshold",
        "layer3_ending_family_bias_warning",
        "layer4_changed_ending_family",
        "change_talk_inference",
        "slot_quality_target_examples",
        "utterance_target",
        "phase_predicted",
        "current_phase_slots_json",
        "phase_transition_note",
        "main_action",
        "add_affirm",
        "latency_ms",
        "latency_start_unix_ms",
        "latency_end_unix_ms",
        "session_id",
        "session_mode",
        "openai_model",
        "reasoning_effort",
        "layer4_enabled",
        "client_style",
        "client_initial_state_json",
        "client_code",
        "client_pattern",
        "client_pattern_label",
        "client_primary_focus",
        "client_primary_focus_label",
        "client_interpersonal_style",
        # ラベル列は削除。代わりに属性（年代/性別/婚姻）を出力
        "client_age_range",
        "client_sex",
        "client_marital_status",
        "client_profiles_path",
        "planner_config_json",
        "phase_confidence",
        "phase_enforce_decision",
        "phase_transition_code",
        "layer1_slot_fill_debug_json",
        "layer1_slot_fill_debug_json_part2",
        "layer1_need_previous_phase_slot_updates",
        "layer1_need_previous_phase_slot_updates_raw",
        "layer1_previous_phase_updates_applied_json",
        "reflect_streak_before",
        "r_since_q_before",
        "features_json",
        "feature_method",
        "feature_user_is_question",
        "feature_user_requests_info",
        "feature_has_permission",
        "feature_resistance",
        "feature_discord",
        "feature_change_talk",
        "feature_novelty",
        "feature_is_short_reply",
        "feature_topic_shift",
        "feature_need_summary",
        "feature_allow_reflect_override",
        "layer1_feature_extractor_request_label",
        "layer1_phase_slot_filler_request_label",
        "layer2_change_talk_inferer_request_label",
        "layer2_action_ranker_request_label",
        "layer2_affirmation_decider_request_label",
        "change_talk_debug_json",
        "slot_quality_target_examples_json",
        "action_source",
        "ranker_proposal_applied",
        "low_confidence_from_action",
        "low_confidence_to_action",
        "low_confidence_threshold",
        "low_confidence_confidence",
        "assistant_fallback_used",
        "assistant_fallback_stage",
        "assistant_fallback_reason",
        "assistant_fallback_action",
        "assistant_fallback_json",
        "assistant_raw_output",
        "assistant_output_validation_ok",
        "assistant_output_validation_reason",
        "assistant_output_validation_initial_ok",
        "assistant_output_validation_initial_reason",
        "assistant_output_validation_retry_ok",
        "assistant_output_validation_retry_reason",
        "assistant_output_validation_attempts_json",
        "assistant_output_validation_json",
        "layer2_reviewed_updates_json",
        "layer2_slot_quality_json",
        "layer2_slot_reviewer_ran",
        "layer2_slot_review_method",
        "layer2_slot_reviewer_request_label",
        "layer2_non_current_slot_reviewer_request_label",
        "layer2_slot_review_fallback_from",
        "layer2_slot_review_schema_issues_json",
        "layer2_slot_review_contract_issues_json",
        "layer2_slot_review_raw_output",
        "layer2_slot_gate_phase",
        "layer2_phase_slots_after_review_json",
        "layer2_phase_slot_quality_after_review_json",
        "layer3_response_integrator_request_label",
        "layer4_response_writer_request_label",
        "layer4_writer_fallback_used",
        "layer4_writer_fallback_stage",
        "layer4_writer_timeout",
        "layer4_writer_error_type",
        "layer4_writer_error_reason",
        "layer4_edit_audit_checked",
        "layer4_edit_audit_issue_count",
        "layer4_edit_audit_subject_reversal",
        "layer4_edit_audit_viewpoint_reversal",
        "layer4_edit_audit_concrete_anchor_dropped",
        "layer4_edit_audit_over_abstracted",
        "layer4_edit_audit_json",
        "phase_slots_json",
        "client_internal_state_json",
        "client_internal_state_reason_json",
        "client_meta_json",
        "client_raw",
        "client_parse_status",
        "schema_version",
    ]

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            writer.writeheader()

            for idx, rec in enumerate(records):
                meta = rec.get("meta") or {}
                debug = meta.get("debug") or {}
                features = debug.get("features") or None
                feature_debug = debug.get("feature_debug") or {}
                layer4_writer = debug.get("layer4_writer")
                layer4_enabled: Any = ""
                if isinstance(layer4_writer, Mapping) and "enabled" in layer4_writer:
                    layer4_enabled = bool(layer4_writer.get("enabled"))
                phase_debug = debug.get("phase_debug") or {}
                if not isinstance(phase_debug, dict):
                    phase_debug = {}
                enforce_debug = phase_debug.get("enforce") or {}
                if not isinstance(enforce_debug, dict):
                    enforce_debug = {}
                phase_predicted = (
                    phase_debug.get("predicted_phase")
                    or phase_debug.get("parsed_phase")
                    or enforce_debug.get("predicted")
                    or phase_debug.get("fallback_phase")
                    or ""
                )
                phase_confidence = phase_debug.get("confidence", "")
                phase_enforce_decision = enforce_debug.get("decision", "")
                phase_transition_code = phase_debug.get("phase_transition_code", "")
                phase_transition_note = phase_debug.get("phase_transition_note", "")
                layer1_slot_fill_debug = phase_debug.get("layer1_slot_fill")
                if layer1_slot_fill_debug is None:
                    layer1_slot_fill_debug = debug.get("layer1_slot_fill_debug")
                layer1_slot_fill_debug_json = (
                    json.dumps(layer1_slot_fill_debug, ensure_ascii=False)
                    if layer1_slot_fill_debug is not None
                    else ""
                )
                (
                    layer1_slot_fill_debug_json_part1,
                    layer1_slot_fill_debug_json_part2,
                ) = _split_csv_cell_into_two(layer1_slot_fill_debug_json)
                slot_quality_target_examples = _extract_slot_quality_target_examples(debug)
                slot_quality_target_examples_json = (
                    json.dumps(slot_quality_target_examples, ensure_ascii=False)
                    if slot_quality_target_examples is not None
                    else ""
                )
                current_phase = str(meta.get("phase", "") or phase_predicted or "")
                slot_quality_target_examples_text = _format_slot_quality_target_examples_for_csv(
                    slot_quality_target_examples=slot_quality_target_examples,
                    current_phase=current_phase,
                )
                layer2_slot_review_columns = _flatten_layer2_slot_review_columns(debug=debug)
                layer2_phase_snapshot_columns = _flatten_layer2_phase_snapshot_columns(debug=debug)
                response_brief_payload = debug.get("response_brief")
                raw_utterance_target: Any = ""
                if isinstance(response_brief_payload, Mapping):
                    raw_utterance_target = response_brief_payload.get("utterance_target", "")
                if raw_utterance_target in (None, ""):
                    raw_utterance_target = debug.get("utterance_target", "")
                if raw_utterance_target is None:
                    utterance_target = ""
                elif isinstance(raw_utterance_target, str):
                    utterance_target = raw_utterance_target
                else:
                    try:
                        utterance_target = json.dumps(raw_utterance_target, ensure_ascii=False)
                    except (TypeError, ValueError):
                        utterance_target = str(raw_utterance_target)
                raw_draft_response_text: Any = ""
                if isinstance(response_brief_payload, Mapping):
                    raw_draft_response_text = response_brief_payload.get("draft_response_text", "")
                if raw_draft_response_text in (None, ""):
                    raw_draft_response_text = debug.get("draft_response_text", "")
                if raw_draft_response_text is None:
                    draft_response_text = ""
                elif isinstance(raw_draft_response_text, str):
                    draft_response_text = raw_draft_response_text
                else:
                    try:
                        draft_response_text = json.dumps(raw_draft_response_text, ensure_ascii=False)
                    except (TypeError, ValueError):
                        draft_response_text = str(raw_draft_response_text)
                if (rec.get("speaker") or "") == "client" and not draft_response_text:
                    draft_response_text = str(rec.get("text", "") or "")
                reflect_ending_family_columns = _flatten_reflect_ending_family_columns(
                    speaker=rec.get("speaker"),
                    action=meta.get("main_action", ""),
                    debug=debug if isinstance(debug, Mapping) else {},
                    draft_response_text=draft_response_text,
                    final_response_text=str(rec.get("text", "") or ""),
                )
                layer1_slot_fill_payload: Dict[str, Any] = {}
                if isinstance(layer1_slot_fill_debug, dict):
                    nested_slot_fill_debug = layer1_slot_fill_debug.get("slot_fill_debug")
                    if isinstance(nested_slot_fill_debug, dict):
                        layer1_slot_fill_payload = nested_slot_fill_debug
                    else:
                        layer1_slot_fill_payload = layer1_slot_fill_debug
                layer1_need_previous_phase_slot_updates = layer1_slot_fill_payload.get(
                    "need_previous_phase_slot_updates", ""
                )
                layer1_need_previous_phase_slot_updates_raw = layer1_slot_fill_payload.get(
                    "need_previous_phase_slot_updates_raw", ""
                )
                layer1_previous_phase_updates_applied = layer1_slot_fill_payload.get("previous_phase_updates_applied")
                layer1_feature_extractor_request_label = ""
                if isinstance(feature_debug, Mapping):
                    layer1_feature_extractor_request_label = str(
                        feature_debug.get("request_label", "") or ""
                    )
                layer1_phase_slot_filler_request_label = str(
                    layer1_slot_fill_payload.get("request_label", "") or ""
                )
                layer1_previous_phase_updates_applied_json = (
                    json.dumps(layer1_previous_phase_updates_applied, ensure_ascii=False)
                    if layer1_previous_phase_updates_applied is not None
                    else ""
                )
                features_json = json.dumps(features, ensure_ascii=False) if features is not None else ""
                if not isinstance(features, dict):
                    features = {}
                change_talk_inference = debug.get("change_talk_inference", "")
                change_talk_debug = debug.get("change_talk_debug")
                layer2_change_talk_inferer_request_label = ""
                if isinstance(change_talk_debug, Mapping):
                    layer2_change_talk_inferer_request_label = str(
                        change_talk_debug.get("request_label", "") or ""
                    )
                change_talk_debug_json = (
                    json.dumps(change_talk_debug, ensure_ascii=False)
                    if change_talk_debug is not None
                    else ""
                )
                current_slot_review_debug = phase_debug.get("slot_review_debug")
                layer2_slot_reviewer_request_label = ""
                if isinstance(current_slot_review_debug, Mapping):
                    layer2_slot_reviewer_request_label = str(
                        current_slot_review_debug.get("request_label", "") or ""
                    )
                non_current_slot_review_debug = phase_debug.get("non_current_slot_review_debug")
                layer2_non_current_slot_reviewer_request_label = ""
                if isinstance(non_current_slot_review_debug, Mapping):
                    layer2_non_current_slot_reviewer_request_label = str(
                        non_current_slot_review_debug.get("request_label", "") or ""
                    )
                ranker_debug = debug.get("ranker_debug")
                layer2_action_ranker_request_label = ""
                if isinstance(ranker_debug, Mapping):
                    layer2_action_ranker_request_label = str(ranker_debug.get("request_label", "") or "")
                affirm_debug = debug.get("affirm_debug")
                layer2_affirmation_decider_request_label = ""
                if isinstance(affirm_debug, Mapping):
                    layer2_affirmation_decider_request_label = str(
                        affirm_debug.get("request_label", "") or ""
                    )
                response_integrator_debug = debug.get("response_integrator_debug")
                layer3_response_integrator_request_label = ""
                if isinstance(response_integrator_debug, Mapping):
                    layer3_response_integrator_request_label = str(
                        response_integrator_debug.get("request_label", "") or ""
                    )
                layer4_response_writer_request_label = ""
                if isinstance(layer4_writer, Mapping):
                    layer4_response_writer_request_label = str(
                        layer4_writer.get("request_label", "") or ""
                    )
                fallback_columns = _flatten_assistant_fallback_columns(
                    debug=debug,
                    default_action=str(meta.get("main_action", "") or ""),
                )
                layer4_writer_columns = _flatten_layer4_writer_columns(debug=debug)
                layer4_edit_audit_columns = _flatten_layer4_edit_audit_columns(debug=debug)
                output_validation_columns = _flatten_assistant_output_validation_columns(debug=debug)
                low_confidence_columns = _flatten_low_confidence_fallback_columns(
                    debug=debug,
                    default_action=str(meta.get("main_action", "") or ""),
                )
                assistant_raw_output_value = debug.get("assistant_raw_output")
                if assistant_raw_output_value is None:
                    assistant_raw_output_value = meta.get("assistant_raw_output", "")
                if assistant_raw_output_value is None:
                    assistant_raw_output = ""
                elif isinstance(assistant_raw_output_value, str):
                    assistant_raw_output = assistant_raw_output_value
                else:
                    try:
                        assistant_raw_output = json.dumps(assistant_raw_output_value, ensure_ascii=False)
                    except (TypeError, ValueError):
                        assistant_raw_output = str(assistant_raw_output_value)
                client_state = meta.get("client_internal_state") or None
                client_state_json = json.dumps(client_state, ensure_ascii=False) if client_state is not None else ""
                client_state_reason = meta.get("client_internal_state_reason") or None
                client_state_reason_json = (
                    json.dumps(client_state_reason, ensure_ascii=False) if client_state_reason is not None else ""
                )
                phase_slots = debug.get("phase_slots")
                phase_slots_json = json.dumps(phase_slots, ensure_ascii=False) if phase_slots is not None else ""
                current_phase_slots = debug.get("current_phase_slots")
                current_phase_slots_json = (
                    json.dumps(current_phase_slots, ensure_ascii=False) if current_phase_slots is not None else ""
                )
                client_meta_extra = meta.get("client_meta") or None
                client_meta_json = (
                    json.dumps(client_meta_extra, ensure_ascii=False) if client_meta_extra is not None else ""
                )
                client_raw = meta.get("client_raw", "")
                client_parse_status = meta.get("parse_status", "")
                session_meta = rec.get("session_meta") or {}
                planner_cfg = session_meta.get("planner_config")
                planner_json = json.dumps(planner_cfg, ensure_ascii=False) if planner_cfg is not None else ""
                client_profiles_path = session_meta.get("client_profiles_path") or session_meta.get("clients_yaml_path", "")

                row = {
                    "schema_version": CSV_SCHEMA_VERSION,
                    "index": idx,
                    "speaker": rec.get("speaker"),
                    "text": rec.get("text", ""),
                    "draft_response_text": draft_response_text,
                    "layer3_draft_ending_family": reflect_ending_family_columns[
                        "layer3_draft_ending_family"
                    ],
                    "layer4_final_ending_family": reflect_ending_family_columns[
                        "layer4_final_ending_family"
                    ],
                    "layer3_preferred_ending_family": reflect_ending_family_columns[
                        "layer3_preferred_ending_family"
                    ],
                    "layer3_avoid_recent_ending_families_json": reflect_ending_family_columns[
                        "layer3_avoid_recent_ending_families_json"
                    ],
                    "layer3_recent_reflect_families_json": reflect_ending_family_columns[
                        "layer3_recent_reflect_families_json"
                    ],
                    "layer3_recent_family_counts_json": reflect_ending_family_columns[
                        "layer3_recent_family_counts_json"
                    ],
                    "layer3_recent_same_family_count": reflect_ending_family_columns[
                        "layer3_recent_same_family_count"
                    ],
                    "layer3_ending_family_lookback_assistant_turns": (
                        reflect_ending_family_columns[
                            "layer3_ending_family_lookback_assistant_turns"
                        ]
                    ),
                    "layer3_ending_family_repetition_threshold": reflect_ending_family_columns[
                        "layer3_ending_family_repetition_threshold"
                    ],
                    "layer3_ending_family_bias_warning": reflect_ending_family_columns[
                        "layer3_ending_family_bias_warning"
                    ],
                    "layer4_changed_ending_family": reflect_ending_family_columns[
                        "layer4_changed_ending_family"
                    ],
                    "phase_predicted": phase_predicted,
                    "main_action": meta.get("main_action", ""),
                    "add_affirm": meta.get("add_affirm", ""),
                    "latency_ms": meta.get("latency_ms", ""),
                    "latency_start_unix_ms": meta.get("latency_start_unix_ms", ""),
                    "latency_end_unix_ms": meta.get("latency_end_unix_ms", ""),
                    "current_phase_slots_json": current_phase_slots_json,
                    "session_id": session_meta.get("session_id", ""),
                    "session_mode": session_meta.get("session_mode", ""),
                    "openai_model": session_meta.get("openai_model", ""),
                    "reasoning_effort": session_meta.get("reasoning_effort", ""),
                    "layer4_enabled": layer4_enabled,
                    "client_style": session_meta.get("client_style", ""),
                    "client_initial_state_json": json.dumps(session_meta.get("client_initial_state", {}), ensure_ascii=False) if session_meta.get("client_initial_state") else "",
                    "client_code": session_meta.get("client_code", ""),
                    "client_pattern": session_meta.get("client_pattern", ""),
                    "client_pattern_label": session_meta.get("client_pattern_label", ""),
                    "client_primary_focus": session_meta.get("client_primary_focus", ""),
                    "client_primary_focus_label": session_meta.get("client_primary_focus_label", ""),
                    "client_interpersonal_style": session_meta.get("client_interpersonal_style", ""),
                    "client_age_range": session_meta.get("client_age_range", ""),
                    "client_sex": session_meta.get("client_sex", ""),
                    "client_marital_status": session_meta.get("client_marital_status", ""),
                    "client_profiles_path": client_profiles_path,
                    "planner_config_json": planner_json,
                    "phase_confidence": phase_confidence,
                    "phase_enforce_decision": phase_enforce_decision,
                    "phase_transition_code": phase_transition_code,
                    "phase_transition_note": phase_transition_note,
                    "layer1_slot_fill_debug_json": layer1_slot_fill_debug_json_part1,
                    "layer1_slot_fill_debug_json_part2": layer1_slot_fill_debug_json_part2,
                    "slot_quality_target_examples": slot_quality_target_examples_text,
                    "utterance_target": utterance_target,
                    "layer1_need_previous_phase_slot_updates": layer1_need_previous_phase_slot_updates,
                    "layer1_need_previous_phase_slot_updates_raw": layer1_need_previous_phase_slot_updates_raw,
                    "layer1_previous_phase_updates_applied_json": layer1_previous_phase_updates_applied_json,
                    "reflect_streak_before": debug.get("reflect_streak"),
                    "r_since_q_before": debug.get("r_since_q"),
                    "features_json": features_json,
                    "feature_method": feature_debug.get("method", ""),
                    "feature_user_is_question": features.get("user_is_question", ""),
                    "feature_user_requests_info": features.get("user_requests_info", ""),
                    "feature_has_permission": features.get("has_permission", ""),
                    "feature_resistance": features.get("resistance", ""),
                    "feature_discord": features.get("discord", ""),
                    "feature_change_talk": features.get("change_talk", ""),
                    "feature_novelty": features.get("novelty", ""),
                    "feature_is_short_reply": features.get("is_short_reply", ""),
                    "feature_topic_shift": features.get("topic_shift", ""),
                    "feature_need_summary": features.get("need_summary", ""),
                    "feature_allow_reflect_override": features.get("allow_reflect_override", ""),
                    "layer1_feature_extractor_request_label": layer1_feature_extractor_request_label,
                    "layer1_phase_slot_filler_request_label": layer1_phase_slot_filler_request_label,
                    "layer2_change_talk_inferer_request_label": layer2_change_talk_inferer_request_label,
                    "layer2_action_ranker_request_label": layer2_action_ranker_request_label,
                    "layer2_affirmation_decider_request_label": layer2_affirmation_decider_request_label,
                    "change_talk_inference": change_talk_inference,
                    "change_talk_debug_json": change_talk_debug_json,
                    "slot_quality_target_examples_json": slot_quality_target_examples_json,
                    "action_source": debug.get("action_source", ""),
                    "ranker_proposal_applied": debug.get("ranker_proposal_applied", ""),
                    "layer2_reviewed_updates_json": layer2_slot_review_columns[
                        "layer2_reviewed_updates_json"
                    ],
                    "layer2_slot_quality_json": layer2_slot_review_columns["layer2_slot_quality_json"],
                    "layer2_slot_reviewer_ran": layer2_slot_review_columns["layer2_slot_reviewer_ran"],
                    "layer2_slot_review_method": layer2_slot_review_columns["layer2_slot_review_method"],
                    "layer2_slot_reviewer_request_label": layer2_slot_reviewer_request_label,
                    "layer2_non_current_slot_reviewer_request_label": (
                        layer2_non_current_slot_reviewer_request_label
                    ),
                    "layer2_slot_review_fallback_from": layer2_slot_review_columns[
                        "layer2_slot_review_fallback_from"
                    ],
                    "layer2_slot_review_schema_issues_json": layer2_slot_review_columns[
                        "layer2_slot_review_schema_issues_json"
                    ],
                    "layer2_slot_review_contract_issues_json": layer2_slot_review_columns[
                        "layer2_slot_review_contract_issues_json"
                    ],
                    "layer2_slot_review_raw_output": layer2_slot_review_columns[
                        "layer2_slot_review_raw_output"
                    ],
                    "layer2_slot_gate_phase": layer2_phase_snapshot_columns["layer2_slot_gate_phase"],
                    "layer2_phase_slots_after_review_json": layer2_phase_snapshot_columns[
                        "layer2_phase_slots_after_review_json"
                    ],
                    "layer2_phase_slot_quality_after_review_json": layer2_phase_snapshot_columns[
                        "layer2_phase_slot_quality_after_review_json"
                    ],
                    "layer3_response_integrator_request_label": (
                        layer3_response_integrator_request_label
                    ),
                    "layer4_response_writer_request_label": layer4_response_writer_request_label,
                    "layer4_writer_fallback_used": layer4_writer_columns["layer4_writer_fallback_used"],
                    "layer4_writer_fallback_stage": layer4_writer_columns["layer4_writer_fallback_stage"],
                    "layer4_writer_timeout": layer4_writer_columns["layer4_writer_timeout"],
                    "layer4_writer_error_type": layer4_writer_columns["layer4_writer_error_type"],
                    "layer4_writer_error_reason": layer4_writer_columns["layer4_writer_error_reason"],
                    "layer4_edit_audit_checked": layer4_edit_audit_columns["layer4_edit_audit_checked"],
                    "layer4_edit_audit_issue_count": layer4_edit_audit_columns["layer4_edit_audit_issue_count"],
                    "layer4_edit_audit_subject_reversal": layer4_edit_audit_columns[
                        "layer4_edit_audit_subject_reversal"
                    ],
                    "layer4_edit_audit_viewpoint_reversal": layer4_edit_audit_columns[
                        "layer4_edit_audit_viewpoint_reversal"
                    ],
                    "layer4_edit_audit_concrete_anchor_dropped": layer4_edit_audit_columns[
                        "layer4_edit_audit_concrete_anchor_dropped"
                    ],
                    "layer4_edit_audit_over_abstracted": layer4_edit_audit_columns[
                        "layer4_edit_audit_over_abstracted"
                    ],
                    "layer4_edit_audit_json": layer4_edit_audit_columns["layer4_edit_audit_json"],
                    "low_confidence_from_action": low_confidence_columns["low_confidence_from_action"],
                    "low_confidence_to_action": low_confidence_columns["low_confidence_to_action"],
                    "low_confidence_threshold": low_confidence_columns["low_confidence_threshold"],
                    "low_confidence_confidence": low_confidence_columns["low_confidence_confidence"],
                    "assistant_fallback_used": fallback_columns["assistant_fallback_used"],
                    "assistant_fallback_stage": fallback_columns["assistant_fallback_stage"],
                    "assistant_fallback_reason": fallback_columns["assistant_fallback_reason"],
                    "assistant_fallback_action": fallback_columns["assistant_fallback_action"],
                    "assistant_fallback_json": fallback_columns["assistant_fallback_json"],
                    "assistant_raw_output": assistant_raw_output,
                    "assistant_output_validation_ok": output_validation_columns["assistant_output_validation_ok"],
                    "assistant_output_validation_reason": output_validation_columns["assistant_output_validation_reason"],
                    "assistant_output_validation_initial_ok": output_validation_columns[
                        "assistant_output_validation_initial_ok"
                    ],
                    "assistant_output_validation_initial_reason": output_validation_columns[
                        "assistant_output_validation_initial_reason"
                    ],
                    "assistant_output_validation_retry_ok": output_validation_columns[
                        "assistant_output_validation_retry_ok"
                    ],
                    "assistant_output_validation_retry_reason": output_validation_columns[
                        "assistant_output_validation_retry_reason"
                    ],
                    "assistant_output_validation_attempts_json": output_validation_columns[
                        "assistant_output_validation_attempts_json"
                    ],
                    "assistant_output_validation_json": output_validation_columns["assistant_output_validation_json"],
                    "phase_slots_json": phase_slots_json,
                    "client_internal_state_json": client_state_json,
                    "client_internal_state_reason_json": client_state_reason_json,
                    "client_meta_json": client_meta_json,
                    "client_raw": client_raw,
                    "client_parse_status": client_parse_status,
                }
                writer.writerow(_normalize_csv_row(row))
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def finalize_session(
    env: SupportsSessionLog,
    llm: "LLMClient",
    *,
    log_prefix: str,
    logs_dir: Optional[Path] = None,
    artifact_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    CSV ログとクライアント視点評価 JSON をまとめて保存する共通関数。
    戻り値: {"csv": <path>, "client_eval": <path>}
    """
    if not getattr(env, "log", None):
        return {}

    base_dir = logs_dir or (Path.cwd() / "logs" / "mi_sim")
    base_dir.mkdir(parents=True, exist_ok=True)

    resolved_artifact_id = (
        str(artifact_id).strip()
        if artifact_id is not None and str(artifact_id).strip()
        else f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{os.getpid()}"
    )
    csv_path = base_dir / f"{log_prefix}_{resolved_artifact_id}.csv"
    save_log_csv(env, str(csv_path))
    print(f"ログを保存しました: {csv_path}")

    client_eval = evaluate_session_from_client_pov(env, llm)
    eval_path = base_dir / f"{log_prefix}_{resolved_artifact_id}_client_eval.json"
    save_client_evaluation_json(client_eval, str(eval_path))
    print(f"クライアント評価を保存しました: {eval_path}")

    return {"csv": str(csv_path), "client_eval": str(eval_path)}


# ==============================
# 解析系：フェーズ・アクション・ストリーク
# ==============================

@dataclass
class PhaseAnalysis:
    phase_counts: Dict[str, int]
    transitions: Dict[str, int]  # "A->B" : count


@dataclass
class ActionAnalysis:
    action_counts: Dict[str, int]
    reflect_count: int
    question_count: int
    total_counselor_turns: int
    reflect_ratio: Optional[float]
    question_ratio: Optional[float]


@dataclass
class ReflectStreakAnalysis:
    streaks: List[int]                  # 各ストリークの長さ（1,2,3,...）
    distribution: Dict[int, int]        # 長さ -> 出現回数
    max_streak: int
    mean_streak: float


@dataclass
class ConditionalActionAnalysis:
    """
    「ある条件を満たすターンに対して、どの main_action が選ばれているか」
    を集計した結果（チェンジトークあり/抵抗あり のとき用）。
    """
    label: str                  # "change_talk" など
    threshold: float            # この値以上を「あり」とみなした閾値
    total_turns: int            # 条件を満たした counselor ターン数
    action_counts: Dict[str, int]
    action_ratios: Dict[str, float]


def analyze_phases(env: SupportsSessionLog) -> PhaseAnalysis:
    """
    ログからフェーズ推移を解析します。
    - 各フェーズの出現回数
    - フェーズ遷移 (A -> B) の回数
    """
    phases: List[str] = []
    for turn in env.log:
        if turn.speaker != "counselor" or not turn.meta:
            continue
        ph = turn.meta.get("phase")
        if ph:
            phases.append(ph)

    phase_counts = Counter(phases)

    transitions: Counter[Tuple[str, str]] = Counter()
    for i in range(len(phases) - 1):
        a = phases[i]
        b = phases[i + 1]
        transitions[(a, b)] += 1

    transitions_str: Dict[str, int] = {
        f"{a}->{b}": c for (a, b), c in transitions.items()
    }

    return PhaseAnalysis(
        phase_counts=dict(phase_counts),
        transitions=transitions_str,
    )


def analyze_actions(env: SupportsSessionLog) -> ActionAnalysis:
    """
    ログから main_action（REFLECT_* / QUESTION / SUMMARY / ASK_PERMISSION_TO_SHARE_INFO /
    PROVIDE_INFO など）の頻度と、
    反射 vs 質問 の比率を解析します。
    """
    actions: List[str] = []
    for turn in env.log:
        if turn.speaker != "counselor" or not turn.meta:
            continue
        act = turn.meta.get("main_action")
        if act:
            actions.append(act)

    counts = Counter(actions)
    reflect_count = (
        counts.get("REFLECT", 0)
        + counts.get("REFLECT_SIMPLE", 0)
        + counts.get("REFLECT_COMPLEX", 0)
        + counts.get("REFLECT_DOUBLE", 0)
    )
    question_count = counts.get("QUESTION", 0)
    total_counselor_turns = len(actions)

    rq_total = reflect_count + question_count
    if rq_total > 0:
        reflect_ratio = reflect_count / rq_total
        question_ratio = question_count / rq_total
    else:
        reflect_ratio = None
        question_ratio = None

    return ActionAnalysis(
        action_counts=dict(counts),
        reflect_count=reflect_count,
        question_count=question_count,
        total_counselor_turns=total_counselor_turns,
        reflect_ratio=reflect_ratio,
        question_ratio=question_ratio,
    )


def analyze_reflect_streaks(env: SupportsSessionLog) -> ReflectStreakAnalysis:
    """
    ログから「REFLECT が何回連続するか」のストリーク分布を解析します。
    例：
      REFLECT, REFLECT, QUESTION, REFLECT, SUMMARY
      -> ストリーク長 [2, 1]
    """
    streaks: List[int] = []
    current = 0

    for turn in env.log:
        if turn.speaker != "counselor" or not turn.meta:
            # counselor 以外 or meta なしのときはストリークを切る
            if current > 0:
                streaks.append(current)
                current = 0
            continue

        act = turn.meta.get("main_action")
        if act in ("REFLECT", "REFLECT_SIMPLE", "REFLECT_COMPLEX", "REFLECT_DOUBLE"):
            current += 1
        else:
            if current > 0:
                streaks.append(current)
                current = 0

    if current > 0:
        streaks.append(current)

    dist_counter: Counter[int] = Counter(streaks)
    distribution = dict(sorted(dist_counter.items(), key=lambda kv: kv[0]))
    max_streak = max(streaks) if streaks else 0
    mean_streak = (sum(streaks) / len(streaks)) if streaks else 0.0

    return ReflectStreakAnalysis(
        streaks=streaks,
        distribution=distribution,
        max_streak=max_streak,
        mean_streak=mean_streak,
    )


# ==============================
# クライアント内部状態ログ → 時系列データ
# ==============================

def collect_client_internal_trajectory(env: SupportsSessionLog) -> List[Dict[str, Any]]:
    """
    セッションログから「クライアント発話ごとの内部状態」の時系列を取り出します。

    戻り値の各要素は
      {
        "index": 発話インデックス（env.log 内の位置）,
        "text": そのときのクライアント発話,
        "internal_state": { ... } または None
      }
    の形になります。
    """
    trajectory: List[Dict[str, Any]] = []

    for idx, turn in enumerate(env.log):
        if turn.speaker != "client":
            continue
        state = None
        if turn.meta:
            state = turn.meta.get("client_internal_state")
        trajectory.append(
            {
                "index": idx,
                "text": turn.text,
                "internal_state": state,
            }
        )

    return trajectory


def _summarize_client_internal_state_series(
    state_series: Sequence[Mapping[str, float]],
) -> Dict[str, Any]:
    """
    クライアント内部状態の時系列から、開始値・終了値・差分・平均を計算する。
    """
    if not state_series:
        return {}

    keys = set().union(*[state.keys() for state in state_series])
    summary: Dict[str, Any] = {"n_client_turns": len(state_series), "by_metric": {}}
    for key in sorted(keys):
        vals = [float(state[key]) for state in state_series if key in state]
        if not vals:
            continue
        summary["by_metric"][key] = {
            "start": round(vals[0], 2),
            "end": round(vals[-1], 2),
            "delta": round(vals[-1] - vals[0], 2),
            "mean": round(sum(vals) / len(vals), 2),
        }
    return summary


def _project_client_internal_state_snapshot(
    summary: Mapping[str, Any],
    field: str,
) -> Dict[str, float]:
    by_metric = summary.get("by_metric")
    if not isinstance(by_metric, Mapping):
        return {}

    projected: Dict[str, float] = {}
    for key, metric_summary in by_metric.items():
        if not isinstance(key, str) or not isinstance(metric_summary, Mapping):
            continue
        value = metric_summary.get(field)
        try:
            projected[key] = round(float(value), 2)
        except (TypeError, ValueError):
            continue
    return projected


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_session_eval_llm_bundle(
    fallback_llm: "LLMClient",
    *,
    default_temperature: float,
) -> Dict[str, Dict[str, Any]]:
    """
    セッション評価用の LLM 設定を model_settings.yaml から解決する。
    失敗時は既存の llm 引数へフォールバックする。
    """
    fallback_spec = {
        "ratings": {
            "llm": fallback_llm,
            "temperature": float(default_temperature),
            "mode": "fallback",
            "model": "",
        },
        "client_feedback": {
            "llm": fallback_llm,
            "temperature": float(default_temperature),
            "mode": "fallback",
            "model": "",
        },
        "supervisor_feedback": {
            "llm": fallback_llm,
            "temperature": float(default_temperature),
            "mode": "fallback",
            "model": "",
        },
    }

    try:
        from .env_utils import build_llm_from_config, get_model_config, load_openai_api_key

        api_key = load_openai_api_key()
        mode_specs = {
            "ratings": "counselor_session_eval_ratings",
            "client_feedback": "counselor_session_eval_client_feedback",
            "supervisor_feedback": "counselor_session_eval_supervisor_feedback",
        }
        resolved: Dict[str, Dict[str, Any]] = {}
        for key, mode in mode_specs.items():
            cfg = get_model_config(
                mode,
                role="counselor",
                fallback_modes=("counselor_response_writer", "counselor_llm"),
            )
            if cfg.get("enabled") is False:
                return fallback_spec
            resolved[key] = {
                "llm": build_llm_from_config(cfg, api_key),
                "temperature": _safe_float(cfg.get("temperature"), float(default_temperature)),
                "mode": mode,
                "model": str(cfg.get("model", "") or ""),
            }
        return resolved
    except Exception:
        return fallback_spec


def _build_session_eval_context(
    env: SupportsSessionLog,
) -> Tuple[List[str], List[Dict[str, float]], Dict[str, Any]]:
    lines: List[str] = []
    state_series: List[Dict[str, float]] = []
    for idx, turn in enumerate(env.log):
        if turn.speaker == "client":
            state = (turn.meta or {}).get("client_internal_state")
            state_str = json.dumps(state, ensure_ascii=False) if state is not None else ""
            lines.append(f"[{idx:02d}] client    : {turn.text}")
            if state_str:
                lines.append(f"       internal_state: {state_str}")
                try:
                    rec: Dict[str, float] = {}
                    for key, value in (state or {}).items():
                        try:
                            rec[str(key)] = float(value)
                        except (TypeError, ValueError):
                            continue
                    if rec:
                        state_series.append(rec)
                except Exception:
                    pass
        else:
            lines.append(f"[{idx:02d}] counselor: {turn.text}")

    client_context_keys = (
        "client_code",
        "client_style",
        "client_pattern",
        "client_pattern_label",
        "client_primary_focus",
        "client_primary_focus_label",
        "client_interpersonal_style",
        "client_age_range",
        "client_sex",
        "client_marital_status",
    )
    client_context: Dict[str, Any] = {}
    for key in client_context_keys:
        value = env.session_meta.get(key, "")
        if value not in (None, ""):
            client_context[key] = value

    return lines, state_series, client_context


def _generate_session_eval_json(
    llm: "LLMClient",
    *,
    prompt: str,
    temperature: float,
    request_label: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    raw = llm.generate(messages, temperature=float(temperature), request_label=request_label)
    text = str(raw).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = dict(fallback)
        data["raw_output"] = text
    return data if isinstance(data, dict) else dict(fallback)


# ==============================
# 解析系：チェンジトーク／抵抗 → 次アクション
# ==============================

def _collect_conditional_actions(
    env: SupportsSessionLog,
    feature_key: str,
    threshold: float,
    label: str,
) -> ConditionalActionAnalysis:
    """
    feature_key（'change_talk' / 'resistance'）の値が threshold 以上の
    counselor ターンについて、main_action の分布を集計します。

    - feature は MIRhythmBot の Decision.meta["debug"]["features"] に
      格納されている前提です。
    """
    actions: List[str] = []

    for turn in env.log:
        if turn.speaker != "counselor" or not turn.meta:
            continue
        meta = turn.meta or {}
        debug = meta.get("debug") or {}
        features = debug.get("features") or {}
        value = features.get(feature_key)
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue

        if v >= threshold:
            act = meta.get("main_action")
            if act:
                actions.append(act)

    counts = Counter(actions)
    total = sum(counts.values())
    if total > 0:
        ratios = {k: c / total for k, c in counts.items()}
    else:
        ratios = {}

    return ConditionalActionAnalysis(
        label=label,
        threshold=threshold,
        total_turns=total,
        action_counts=dict(counts),
        action_ratios=ratios,
    )


def analyze_change_talk_responses(
    env: SupportsSessionLog,
    threshold: float = 0.6,
) -> ConditionalActionAnalysis:
    """
    チェンジトーク（features['change_talk']）が threshold 以上のターンで、
    カウンセラーがどの main_action を選んでいるかを集計します。
    """
    return _collect_conditional_actions(
        env=env,
        feature_key="change_talk",
        threshold=threshold,
        label="change_talk",
    )


def analyze_resistance_responses(
    env: SupportsSessionLog,
    threshold: float = 0.6,
) -> ConditionalActionAnalysis:
    """
    抵抗（features['resistance']）が threshold 以上のターンで、
    カウンセラーがどの main_action を選んでいるかを集計します。
    """
    return _collect_conditional_actions(
        env=env,
        feature_key="resistance",
        threshold=threshold,
        label="resistance",
    )


# ==============================
# 結果をざっくり表示するユーティリティ
# ==============================

def print_basic_analysis(env: SupportsSessionLog) -> None:
    """
    フェーズ・アクション・反射ストリークに加えて、
    「チェンジトーク／抵抗が強いターンで次に何をしているか」
    も含めてざっくり統計をコンソールに出力します。
    """
    pa = analyze_phases(env)
    aa = analyze_actions(env)
    ra = analyze_reflect_streaks(env)

    ct_analysis = analyze_change_talk_responses(env, threshold=0.6)
    rs_analysis = analyze_resistance_responses(env, threshold=0.6)

    print("=== Phase Counts ===")
    for ph, c in pa.phase_counts.items():
        print(f"  {ph}: {c}")

    print("\n=== Phase Transitions ===")
    for k, v in pa.transitions.items():
        print(f"  {k}: {v}")

    print("\n=== Action Counts (overall) ===")
    for act, c in aa.action_counts.items():
        print(f"  {act}: {c}")
    print(f"  total counselor turns: {aa.total_counselor_turns}")
    if aa.reflect_ratio is not None:
        print(
            f"  REFLECT vs QUESTION = "
            f"{aa.reflect_count} vs {aa.question_count} "
            f"(ratio: {aa.reflect_ratio:.2f} / {aa.question_ratio:.2f})"
        )
    else:
        print("  REFLECT / QUESTION がほとんどありません。")

    print("\n=== Reflect Streaks ===")
    print(f"  streaks (per run): {ra.streaks}")
    print(f"  distribution (length -> count): {ra.distribution}")
    print(f"  max_streak: {ra.max_streak}")
    print(f"  mean_streak: {ra.mean_streak:.2f}")

    # ---- チェンジトークに対する応答 ----
    print("\n=== Responses to Change Talk ===")
    print(
        f"  condition: change_talk >= {ct_analysis.threshold:.2f}, "
        f"turns: {ct_analysis.total_turns}"
    )
    if ct_analysis.total_turns == 0:
        print("  該当ターンがありません。")
    else:
        for act, count in ct_analysis.action_counts.items():
            r = ct_analysis.action_ratios.get(act, 0.0)
            print(f"  {act}: {count} (ratio: {r:.2f})")

    # ---- 抵抗に対する応答 ----
    print("\n=== Responses to Resistance ===")
    print(
        f"  condition: resistance >= {rs_analysis.threshold:.2f}, "
        f"turns: {rs_analysis.total_turns}"
    )
    if rs_analysis.total_turns == 0:
        print("  該当ターンがありません。")
    else:
        for act, count in rs_analysis.action_counts.items():
            r = rs_analysis.action_ratios.get(act, 0.0)
            print(f"  {act}: {count} (ratio: {r:.2f})")


# ==============================
# クライアント視点のセッション評価（LLM 使用）
# ==============================

def evaluate_session_from_client_pov(
    env: SupportsSessionLog,
    llm: "LLMClient",
    *,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    クライアントの内部状態ログとセッション全体のやり取りを LLM に渡して、
    「クライアント本人の主観的評価（数値＋フィードバック）」を JSON で生成してもらう関数。

    - ratings: 0〜10 の評価値（必要に応じて拡張可）
    - client_feedback: クライアント本人の振り返り（約400字）
    - supervisor_feedback: MI専門家からのレビュー（約400字）
    """
    session_id = env.session_meta.get("session_id", "")
    lines, state_series, client_context = _build_session_eval_context(env)
    internal_state_summary = _summarize_client_internal_state_series(state_series)
    if internal_state_summary:
        lines.insert(0, json.dumps(internal_state_summary, ensure_ascii=False))
        lines.insert(0, "=== INTERNAL_STATE_SUMMARY ===")

    llm_bundle = _build_session_eval_llm_bundle(llm, default_temperature=float(temperature))

    ratings_prompt = (
        "あなたはカウンセリング研究者です。\n"
        "以下は、クライアントとカウンセラーの1セッション分の対話ログです。\n"
        "クライアント発話には、そのときの内部状態を表す 0〜10 のスコアが付与されている場合があります。"
        "上部の INTERNAL_STATE_SUMMARY（開始/終了/平均/差分）を主たる根拠として、数値傾向と整合的な評価を返してください。\n\n"
        "出力は必ず次の JSON 形式「だけ」で返してください:\n"
        "{\n"
        '  "ratings": {\n'
        '    "overall_satisfaction": 0〜10の数値,\n'
        '    "goal_importance": 0〜10の数値,\n'
        '    "goal_confidence": 0〜10の数値,\n'
        '    "alliance_bond": 0〜10の数値,\n'
        '    "positive_affect": 0〜10の数値,\n'
        '    "negative_affect": 0〜10の数値\n'
        "  }\n"
        "}\n\n"
        "=== セッションログ ===\n"
        + "\n".join(lines)
    )

    ratings_data = _generate_session_eval_json(
        llm_bundle["ratings"]["llm"],
        prompt=ratings_prompt,
        temperature=float(llm_bundle["ratings"]["temperature"]),
        request_label="session_eval:ratings",
        fallback={"ratings": {}},
    )

    ratings = ratings_data.get("ratings")
    if not isinstance(ratings, Mapping):
        ratings = {}

    initial_state = _project_client_internal_state_snapshot(internal_state_summary, "start")
    final_state = _project_client_internal_state_snapshot(internal_state_summary, "end")
    delta_state = _project_client_internal_state_snapshot(internal_state_summary, "delta")
    client_context_json = json.dumps(client_context, ensure_ascii=False) if client_context else "{}"
    ratings_json = json.dumps(ratings, ensure_ascii=False)
    final_client_internal_state_json = json.dumps(final_state, ensure_ascii=False)
    session_log_text = "\n".join(lines)
    mi_knowledge_text = _load_session_eval_mi_knowledge_text() or "MI knowledge is unavailable."

    client_feedback_prompt = _format_session_eval_prompt(
        _get_session_eval_prompt_template(
            "client_feedback_prompt_template",
            _DEFAULT_CLIENT_FEEDBACK_PROMPT_TEMPLATE,
        ),
        client_context_json=client_context_json,
        ratings_json=ratings_json,
        final_client_internal_state_json=final_client_internal_state_json,
        session_log_text=session_log_text,
    )

    supervisor_feedback_prompt = _format_session_eval_prompt(
        _get_session_eval_prompt_template(
            "supervisor_feedback_prompt_template",
            _DEFAULT_SUPERVISOR_FEEDBACK_PROMPT_TEMPLATE,
        ),
        mi_knowledge_text=mi_knowledge_text,
        ratings_json=ratings_json,
        final_client_internal_state_json=final_client_internal_state_json,
        session_log_text=session_log_text,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        client_feedback_future = executor.submit(
            _generate_session_eval_json,
            llm_bundle["client_feedback"]["llm"],
            prompt=client_feedback_prompt,
            temperature=float(llm_bundle["client_feedback"]["temperature"]),
            request_label="session_eval:client_feedback",
            fallback={"client_feedback": ""},
        )
        supervisor_feedback_future = executor.submit(
            _generate_session_eval_json,
            llm_bundle["supervisor_feedback"]["llm"],
            prompt=supervisor_feedback_prompt,
            temperature=float(llm_bundle["supervisor_feedback"]["temperature"]),
            request_label="session_eval:supervisor_feedback",
            fallback={"supervisor_feedback": ""},
        )
        client_feedback_data = client_feedback_future.result()
        supervisor_feedback_data = supervisor_feedback_future.result()

    data: Dict[str, Any] = {
        "ratings": dict(ratings),
        "client_feedback": str(client_feedback_data.get("client_feedback", "") or ""),
        "supervisor_feedback": str(supervisor_feedback_data.get("supervisor_feedback", "") or ""),
    }

    if internal_state_summary:
        data["client_internal_state_summary"] = internal_state_summary
        data["initial_client_internal_state"] = initial_state
        data["final_client_internal_state"] = final_state
        data["client_internal_state_delta_from_start"] = delta_state

    data.setdefault("session_id", session_id)
    return data


def save_client_evaluation_json(evaluation: Dict[str, Any], path: str) -> None:
    """
    evaluate_session_from_client_pov の結果を JSON ファイルに保存する簡易ユーティリティ。
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)


# ==============================
# 簡単なデモ
# ==============================

def _demo() -> None:
    """
    conversation_environment.demo_two_agents() などで作られた env を想定して、
    解析と保存を一通り走らせるミニデモです。
    実際には、あなたの環境で対話を終えた ConversationEnvironment を
    渡して使ってください。
    """
    from .mi_counselor_agent import MIRhythmBot, DummyLLM
    from .conversation_environment import ConversationEnvironment
    from .perma_client_agent import SimpleClientLLM

    counselor = MIRhythmBot(llm=DummyLLM())
    client = SimpleClientLLM(llm=DummyLLM())
    env = ConversationEnvironment(counselor=counselor, client=client)

    # 簡単なシミュレーション
    env.simulate(
        first_client_utterance="最近、生活リズムが崩れてしまって、気持ちも落ち込んでいます。",
        max_turns=5,
    )

    # 解析結果を表示
    print_basic_analysis(env)

    # JSONL / CSV に保存（ファイル名は例ですので、適宜変更してください）
    save_log_jsonl(env, "session_example.jsonl")
    save_log_csv(env, "session_example.csv")


if __name__ == "__main__":
    _demo()
