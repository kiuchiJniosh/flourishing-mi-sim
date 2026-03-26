from __future__ import annotations

import dataclasses
import inspect
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from .paths import resolve_config_path
from .openai_llm import OpenAIResponsesLLM
from .mi_prompt_knowledge import inject_mi_knowledge
import yaml


# ============================
# Nine phases
# ============================
class Phase(str, Enum):
    GREETING = "Greeting"
    PURPOSE_CONFIRMATION = "Purpose Confirmation"
    CURRENT_STATUS_CHECK = "Current Status Check"
    FOCUSING_TARGET_BEHAVIOR = "Focus Target Behavior"
    IMPORTANCE_PROMOTION = "Importance Promotion"
    CONFIDENCE_PROMOTION = "Confidence Promotion"
    NEXT_STEP_DECISION = "Next Step Decision"
    REVIEW_REFLECTION = "Review Reflection"
    CLOSING = "Closing"


_PHASE_SLOT_SCHEMA: Dict[Phase, Tuple[str, ...]] = {
    Phase.GREETING: ("greeting_exchange", "rapport_cue"),
    Phase.PURPOSE_CONFIRMATION: ("presenting_problem_raw", "today_focus_topic", "process_need"),
    Phase.CURRENT_STATUS_CHECK: ("current_situation", "problem_scene", "emotion_state", "background_context"),
    Phase.FOCUSING_TARGET_BEHAVIOR: ("target_behavior", "change_direction", "focus_agreement"),
    Phase.IMPORTANCE_PROMOTION: ("importance_reasons", "core_values", "importance_scale"),
    Phase.CONFIDENCE_PROMOTION: ("barrier_coping_strategy", "supports_strengths", "past_success_experience", "confidence_scale"),
    Phase.NEXT_STEP_DECISION: ("next_step_action", "execution_context", "commitment_level"),
    Phase.REVIEW_REFLECTION: ("session_learning", "key_takeaway", "carry_forward_intent"),
    Phase.CLOSING: ("closing_end_signal",),
}

_PHASE_SLOT_LABELS: Dict[str, str] = {
    "greeting_exchange": "Greeting exchange",
    "rapport_cue": "Rapport cue / reassurance",
    "presenting_problem_raw": "Initial presenting concern",
    "process_need": "Preference for pace / process",
    "today_focus_topic": "Today's focus topic",
    "current_situation": "Current facts",
    "problem_scene": "Concrete problem scene",
    "emotion_state": "Emotion",
    "background_context": "Background / context",
    "target_behavior": "Target behavior",
    "change_direction": "Direction of change",
    "focus_agreement": "Focus agreement",
    "importance_reasons": "Reasons it matters",
    "core_values": "Core values / what matters",
    "importance_scale": "Importance scale",
    "barrier_coping_strategy": "Barrier coping strategy",
    "supports_strengths": "Supports / strengths",
    "past_success_experience": "Past success experience",
    "confidence_scale": "Confidence scale",
    "next_step_action": "Next step",
    "execution_context": "Execution context (when / where)",
    "commitment_level": "Commitment / likelihood",
    "session_learning": "Today's insight / learning",
    "key_takeaway": "Key takeaway",
    "carry_forward_intent": "Intent to carry forward",
    "closing_end_signal": "Farewell / implied closing agreement",
}
_PHASE_BY_SLOT_KEY: Dict[str, Phase] = {
    slot_key: phase
    for phase, slot_keys in _PHASE_SLOT_SCHEMA.items()
    for slot_key in slot_keys
}

_PHASE_SLOT_QUALITY_RUBRIC_PATH = resolve_config_path("phase_slot_quality_rubrics.yaml")
_LAYER_ACTION_PROMPT_PATH = resolve_config_path("layer3_layer4_action_prompts.yaml")


def _init_phase_slot_memory() -> Dict[str, Dict[str, str]]:
    memory: Dict[str, Dict[str, str]] = {}
    for phase in Phase:
        memory[phase.name] = {slot_key: "" for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ())}
    return memory


def _init_phase_slot_meta() -> Dict[str, Dict[str, Dict[str, Any]]]:
    meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for phase in Phase:
        phase_meta: Dict[str, Dict[str, Any]] = {}
        for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
            phase_meta[slot_key] = {
                "status": "inactive",
                "evidence_quote": "",
                "evidence_turn_ids": [],
                "user_quote": "",
                "user_evidence_turn_ids": [],
                "assistant_quote": "",
                "assistant_evidence_turn_ids": [],
                "extraction_confidence": 0.0,
                "quality_score": 0.0,
                "issue_codes": [],
                "review_note": "",
                "review_source": "",
                "promote_on_phase_entry": False,
                "reviewed_at_turn": 0,
                "review_origin_phase": "",
                "review_decision_raw": "",
            }
        meta[phase.name] = phase_meta
    return meta


@lru_cache(maxsize=1)
def _load_phase_slot_quality_rubrics() -> Dict[str, Any]:
    """
    Load `config/phase_slot_quality_rubrics.yaml` and return the phase-specific rubric.
    Return an empty dict on failure so prompt construction can fall back safely.
    """
    try:
        if not _PHASE_SLOT_QUALITY_RUBRIC_PATH.exists():
            return {}
        loaded = yaml.safe_load(_PHASE_SLOT_QUALITY_RUBRIC_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            return dict(loaded)
    except Exception:
        return {}
    return {}


@lru_cache(maxsize=1)
def _load_layer_action_prompts() -> Dict[str, Any]:
    """
    Load `config/layer3_layer4_action_prompts.yaml` and return the Layer3/Layer4
    action prompt definitions. Return an empty dict on failure.
    """
    try:
        if not _LAYER_ACTION_PROMPT_PATH.exists():
            return {}
        loaded = yaml.safe_load(_LAYER_ACTION_PROMPT_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            return dict(loaded)
    except Exception:
        return {}
    return {}


def _get_layer3_action_prompt(*keys: str, default: str = "") -> str:
    node = _get_layer3_action_prompt_node(*keys)
    if not isinstance(node, str):
        return default
    text = node.strip("\n")
    if not text.strip():
        return default
    return text


def _get_layer3_action_prompt_formatted(
    *keys: str,
    default: str = "",
    **kwargs: Any,
) -> str:
    template = _get_layer3_action_prompt(*keys, default=default)
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def _get_layer3_action_prompt_node(*keys: str) -> Any:
    node: Any = _load_layer_action_prompts()
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _get_layer4_writer_prompt(*keys: str, default: str = "") -> str:
    return _get_layer3_action_prompt("layer4_writer_prompts", *keys, default=default)


def _get_layer4_writer_prompt_node(*keys: str) -> Any:
    return _get_layer3_action_prompt_node("layer4_writer_prompts", *keys)


def _get_layer4_writer_prompt_formatted(
    *keys: str,
    default: str = "",
    **kwargs: Any,
) -> str:
    template = _get_layer4_writer_prompt(*keys, default=default)
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def _get_layer4_writer_prompt_lines(*keys: str, default_lines: Sequence[str]) -> List[str]:
    default_text = "\n".join(default_lines)
    raw = _get_layer4_writer_prompt(*keys, default=default_text)
    lines = [line.strip() for line in str(raw or "").splitlines() if str(line).strip()]
    if lines:
        return lines
    return [line.strip() for line in default_lines if str(line).strip()]


def _split_nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text or "").splitlines() if str(line).strip()]


def _merge_issue_codes(
    base_codes: Sequence[str],
    extra_codes: Sequence[str],
    *,
    max_codes: int = 12,
) -> List[str]:
    merged: List[str] = []
    for item in list(base_codes) + list(extra_codes):
        code = re.sub(r"[^a-zA-Z0-9_-]", "", str(item or "").strip())
        if not code or code in merged:
            continue
        merged.append(code)
        if len(merged) >= max(1, int(max_codes)):
            break
    return merged


def _build_phase_slot_quality_rubric_prompt_text(phase: Phase) -> str:
    rubrics = _load_phase_slot_quality_rubrics()
    phase_raw = None
    if isinstance(rubrics, Mapping):
        phase_raw = rubrics.get(phase.name) or rubrics.get(phase.value)
    if not isinstance(phase_raw, Mapping):
        return "- Rubric file unavailable (temporary fallback: judge grounding, specificity, and feasibility)."

    lines: List[str] = []
    for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
        slot_label = _PHASE_SLOT_LABELS.get(slot_key, slot_key)
        raw = phase_raw.get(slot_key)
        if not isinstance(raw, Mapping):
            lines.append(f"- {slot_label} ({slot_key}): rubric not defined")
            continue
        min_condition = _normalize_slot_text(raw.get("min_condition") or raw.get("minimum_condition"))
        min_example = _normalize_slot_text(raw.get("min_example") or raw.get("minimum_example"))
        max_condition = _normalize_slot_text(raw.get("max_condition") or raw.get("maximum_condition"))
        max_example = _normalize_slot_text(raw.get("max_example") or raw.get("maximum_example"))
        lines.append(
            (
                f"- {slot_label} ({slot_key})\n"
                f"  Minimum condition: {min_condition or 'undefined'}\n"
                f"  Minimum example: {min_example or 'undefined'}\n"
                f"  Maximum condition: {max_condition or 'undefined'}\n"
                f"  Maximum example: {max_example or 'undefined'}"
            )
        )
    return "\n".join(lines) if lines else "- Rubric not defined"


def _rubric_hint_for_slot(*, phase: Phase, slot_key: str) -> str:
    rubrics = _load_phase_slot_quality_rubrics()
    phase_raw = None
    if isinstance(rubrics, Mapping):
        phase_raw = rubrics.get(phase.name) or rubrics.get(phase.value)
    if isinstance(phase_raw, Mapping):
        slot_raw = phase_raw.get(slot_key)
        if isinstance(slot_raw, Mapping):
            # Layer 1 only needs a lightweight hint, so prefer the maximum condition.
            hint = _normalize_slot_text(slot_raw.get("max_condition") or slot_raw.get("min_condition"))
            if hint:
                return hint
    return ""


def _append_numeric_scale_explicitness_guard(*, slot_key: str, hint: str) -> str:
    normalized_hint = _normalize_slot_text(hint)
    if slot_key == "commitment_level":
        guard_text = (
            "commitment_level does not have to be numeric. A 0-10 score or a qualitative"
            " expression such as low/medium/high is acceptable. Focus on how clearly the"
            " execution conditions and up/down factors are described, rather than the number itself."
        )
        if not normalized_hint:
            return guard_text
        first_sentence = normalized_hint.split("。", 1)[0].strip()
        if first_sentence:
            first_sentence = f"{first_sentence}。"
        if first_sentence and first_sentence in guard_text:
            return guard_text
        return f"{first_sentence} {guard_text}".strip()

    guard_text = ""
    if slot_key in {"importance_scale", "confidence_scale"}:
        guard_text = (
            "Only populate the numeric value when the user explicitly states a 0-10 score,"
            " or when it can be confirmed through explicit assent to the assistant's immediately"
            " preceding scale prompt."
        )
    if not guard_text:
        return normalized_hint
    if not normalized_hint:
        return guard_text
    if guard_text in normalized_hint:
        return normalized_hint
    return f"{normalized_hint} {guard_text}"


def _build_lightweight_slot_definition_text(
    *,
    phases: Sequence[Phase],
    allowed_slot_keys_by_phase: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    lines: List[str] = []
    seen_phase_names: Set[str] = set()
    for phase in phases:
        if phase.name in seen_phase_names:
            continue
        seen_phase_names.add(phase.name)
        configured_slot_keys = list(_PHASE_SLOT_SCHEMA.get(phase, ()))
        if allowed_slot_keys_by_phase and phase.name in allowed_slot_keys_by_phase:
            allowed = {
                _normalize_slot_text(slot_key)
                for slot_key in (allowed_slot_keys_by_phase.get(phase.name) or [])
                if _normalize_slot_text(slot_key)
            }
            slot_keys = [slot_key for slot_key in configured_slot_keys if slot_key in allowed]
        else:
            slot_keys = configured_slot_keys
        if not slot_keys:
            continue
        lines.append(f"- {phase.value}[{phase.name}]")
        for slot_key in slot_keys:
            slot_label = _PHASE_SLOT_LABELS.get(slot_key, slot_key)
            hint = _append_numeric_scale_explicitness_guard(
                slot_key=slot_key,
                hint=_rubric_hint_for_slot(phase=phase, slot_key=slot_key),
            )
            if hint:
                lines.append(f"  - {slot_key}({slot_label}): {hint}")
            else:
                lines.append(f"  - {slot_key}({slot_label})")
    return "\n".join(lines) if lines else "- none"


def _build_current_phase_slot_quality_context(state: "DialogueState") -> Dict[str, Dict[str, Any]]:
    phase = state.phase
    slot_values = state.phase_slots.get(phase.name) or {}
    slot_meta = state.phase_slot_meta.get(phase.name) or {}
    context: Dict[str, Dict[str, Any]] = {}
    for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
        value = _normalize_slot_text(slot_values.get(slot_key))
        meta = slot_meta.get(slot_key) if isinstance(slot_meta.get(slot_key), Mapping) else {}
        status = _normalize_slot_text((meta or {}).get("status"))
        if status not in {"confirmed", "tentative", "needs_confirmation", "inactive"}:
            status = "inactive"
        quality_score = _clamp01((meta or {}).get("quality_score"), 0.0)
        context[slot_key] = {
            "value": value,
            "status": status,
            "quality_score": round(float(quality_score), 3),
        }
    return context


class MainAction(str, Enum):
    REFLECT = "REFLECT"                # reflective listening / paraphrase
    REFLECT_SIMPLE = "REFLECT_SIMPLE"  # short reflection centered on explicit facts
    REFLECT_COMPLEX = "REFLECT_COMPLEX"  # one-step inference of implied meaning
    REFLECT_DOUBLE = "REFLECT_DOUBLE"  # double-sided reflection for ambivalence
    QUESTION = "QUESTION"              # question
    SCALING_QUESTION = "SCALING_QUESTION"  # 0-10 scaling question
    CLARIFY_PREFERENCE = "CLARIFY_PREFERENCE"  # preference check when permission is unclear
    SUMMARY = "SUMMARY"                # summary
    ASK_PERMISSION_TO_SHARE_INFO = "ASK_PERMISSION_TO_SHARE_INFO"  # EPE elicit: ask permission to share information
    PROVIDE_INFO = "PROVIDE_INFO"      # EPE provide+elicit: share neutrally and check response


class InfoMode(str, Enum):
    NONE = "NONE"
    WAITING_PERMISSION = "WAITING_PERMISSION"
    READY_TO_PROVIDE = "READY_TO_PROVIDE"


class ReflectionStyle(str, Enum):
    SIMPLE = "simple"        # mainly paraphrases explicit facts
    COMPLEX = "complex"      # weaves in emotion or value
    DOUBLE_SIDED = "double"  # holds both sides of ambivalence together


class AffirmationMode(str, Enum):
    NONE = "NONE"
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"

    def __bool__(self) -> bool:
        return self != AffirmationMode.NONE


_REFLECT_ACTIONS: Tuple[MainAction, ...] = (
    MainAction.REFLECT,
    MainAction.REFLECT_SIMPLE,
    MainAction.REFLECT_COMPLEX,
    MainAction.REFLECT_DOUBLE,
)


def _is_reflect_action(action: MainAction) -> bool:
    return action in _REFLECT_ACTIONS


def _reflection_style_from_action(action: MainAction) -> ReflectionStyle:
    if action == MainAction.REFLECT_SIMPLE:
        return ReflectionStyle.SIMPLE
    if action == MainAction.REFLECT_DOUBLE:
        return ReflectionStyle.DOUBLE_SIDED
    # Treat REFLECT / REFLECT_COMPLEX as complex reflections.
    return ReflectionStyle.COMPLEX


def _action_from_reflection_style(style: ReflectionStyle) -> MainAction:
    if style == ReflectionStyle.SIMPLE:
        return MainAction.REFLECT_SIMPLE
    if style == ReflectionStyle.DOUBLE_SIDED:
        return MainAction.REFLECT_DOUBLE
    return MainAction.REFLECT_COMPLEX


def _collapse_reflect_action(action: MainAction) -> MainAction:
    return MainAction.REFLECT if _is_reflect_action(action) else action


def _normalize_affirmation_mode(value: Any) -> AffirmationMode:
    if isinstance(value, AffirmationMode):
        return value
    if isinstance(value, bool):
        return AffirmationMode.SIMPLE if value else AffirmationMode.NONE

    raw = str(value or "").strip().upper().replace("-", "_")
    if raw in {"", "NONE", "NO", "FALSE", "0", "OFF"}:
        return AffirmationMode.NONE
    if raw in {"SIMPLE", "SINGLE", "TRUE", "1", "ON", "YES", "ADD_AFFIRM"}:
        return AffirmationMode.SIMPLE
    if raw in {"COMPLEX", "DEEP"}:
        return AffirmationMode.COMPLEX
    return AffirmationMode.NONE


def coerce_main_action(value: Any) -> Optional[MainAction]:
    if isinstance(value, MainAction):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return MainAction(raw)
    except Exception:
        return None


class RiskLevel(str, Enum):
    NONE = "none"
    MILD = "mild"
    HIGH = "high"


@dataclass
class RiskAssessment:
    level: RiskLevel = RiskLevel.NONE
    reason: Optional[str] = None
    raw_output: Optional[str] = None


@dataclass
class OutputEvaluation:
    score: float
    feedback: Optional[str] = None
    raw_output: Optional[str] = None
    rewrite: Optional[str] = None


class LLMClient(Protocol):
    """Replace the LLM call implementation to match the runtime environment."""

    def generate(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
        ...


@dataclass
class PlannerConfig:
    """
    Rule-based tuning knobs used as the pre-DsPy baseline.
    - `stochastic=False` favors deterministic, reproducible behavior
    - `reflect_*` controls suppress repetitive reflections
    - `summary_*` controls summary timing and triggers
    - `allow_override_*` relaxes reflection caps when change talk / resistance / novelty is high
    """
    # Softmax temperature. Still affects weighting even when stochastic=False.
    temperature: float = 0.7
    # Whether to sample after converting scores to probabilities.
    stochastic: bool = False
    # Random seed for reproducible variation when stochastic=True.
    seed: int = 42

    # Consecutive-reflection penalty starting from the third reflection.
    reflect_decay_base: float = 0.65
    # Upper cap when the reflection streak gets long.
    reflect_prob_cap_after_streak: float = 0.15
    reflect_streak_cap_threshold: int = 5

    # Threshold for encouraging a summary.
    summary_interval_turns: int = 9
    # If reflection streak reaches this count, push toward summary/question.
    summary_reflect_streak_trigger: int = 3

    # Thresholds for temporarily lifting reflection caps.
    allow_override_change_talk: float = 0.6
    allow_override_resistance: float = 0.6
    allow_override_novelty: float = 0.7
    # Relationship-signal thresholds that temporarily pause phase progression.
    phase_hold_resistance_threshold: float = 0.6
    phase_hold_discord_threshold: float = 0.6
    # Slot-based phase transition gates.
    phase_slot_fill_rate_threshold: float = 1.0
    # Keep quality_mean for diagnostics only.
    phase_slot_quality_mean_threshold: float = 0.62
    # Effective threshold used by the transition gate.
    phase_slot_quality_min_threshold: float = 0.8
    # Maximum number of turns allowed in CLOSING.
    closing_max_turns: int = 3

    # Reflection-style control
    complex_reflection_confidence_threshold: float = 0.58
    ambivalence_reflection_threshold: float = 0.68
    double_sided_max_ratio: float = 1.0 / 3.0
    double_sided_min_reflections_before_use: int = 2
    double_sided_min_turn_gap: int = 3

    # Output-validation mode:
    # - "warn": log warnings only
    # - "strict": still log rule violations, but do not regenerate text
    output_validation_mode: str = "warn"
    # JSON parsing policy for Layer1/2/3:
    # - "loose": allow JSON fragment extraction
    # - "strict": accept only strict JSON
    layer1_json_mode: str = "loose"
    layer2_json_mode: str = "loose"
    layer3_json_mode: str = "loose"


@dataclass
class DialogueState:
    # Initial phase for the nine-phase flow.
    phase: Phase = Phase.GREETING
    # Counselor turns spent inside the current phase.
    phase_turns: int = 0

    # Rhythm-control state
    r_since_q: int = 0              # non-question turns since the last question
    reflect_streak: int = 0         # consecutive REFLECT count
    turns_since_summary: int = 0    # turns since the last summary
    turns_since_affirm: int = 0     # turns since the last affirmation
    turns_since_complex_affirm: int = 99  # turns since the last complex affirmation

    # Information-sharing state (EPE: Elicit-Provide-Elicit)
    info_mode: InfoMode = InfoMode.NONE

    # Reference fields
    last_user_text: str = ""
    last_substantive_user_text: str = ""
    turn_index: int = 0
    last_actions: List[MainAction] = field(default_factory=list)
    reflection_turn_count: int = 0
    double_sided_reflection_count: int = 0
    turns_since_double_sided_reflection: int = 99
    last_reflection_styles: List[ReflectionStyle] = field(default_factory=list)
    # Scaling-question follow-up control (reason / plus-one)
    scale_followup_pending_phase: Optional[Phase] = None
    scale_followup_score: Optional[float] = None
    scale_followup_pending_step: Optional[str] = None
    scale_followup_last_asked_step: Optional[str] = None
    scale_followup_done_in_phase: Dict[str, bool] = field(default_factory=dict)

    # Phase-slot state
    phase_slots: Dict[str, Dict[str, str]] = field(default_factory=_init_phase_slot_memory)
    phase_slot_meta: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=_init_phase_slot_meta)
    phase_slot_pending: Dict[str, Dict[str, str]] = field(default_factory=_init_phase_slot_memory)
    phase_slot_pending_meta: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=_init_phase_slot_meta)

    # Safety-related notes
    risk_level: RiskLevel = RiskLevel.NONE
    last_risk_reason: Optional[str] = None


@dataclass
class PlannerFeatures:
    user_is_question: bool
    user_requests_info: bool
    has_permission: Optional[bool]  # None=unknown, True/False=explicit
    resistance: float               # 0-1, higher means more resistance-like
    change_talk: float              # 0-1, higher means more change-talk-like
    novelty: float                  # 0-1, higher means more novel information
    is_short_reply: bool
    topic_shift: bool
    need_summary: bool
    allow_reflect_override: bool
    discord: float = 0.0            # 0-1, higher means more discord / relationship strain
    importance_estimate: Optional[float] = None  # 0-1, or None if unavailable
    confidence_estimate: Optional[float] = None  # 0-1, or None if unavailable


class FirstTurnHint(str, Enum):
    GREETING_ONLY = "greeting_only"
    GREETING_WITH_TOPIC = "greeting_with_topic"
    TOPIC_ONLY = "topic_only"


_CHANGE_TALK_KINDS: Tuple[str, ...] = (
    "desire",
    "ability",
    "reason",
    "need",
    "commitment",
    "activation",
    "taking_step",
)
_CHANGE_TALK_DARN_KINDS: Tuple[str, ...] = ("desire", "ability", "reason", "need")
_CHANGE_TALK_CAT_KINDS: Tuple[str, ...] = ("commitment", "activation", "taking_step")
_CHANGE_TALK_ORIGIN_TYPES: Tuple[str, ...] = ("user_utterance", "system_reframed")
_CHANGE_TALK_MOTIVATION_FOCUS_TYPES: Tuple[str, ...] = ("importance_related", "confidence_related")
_CHANGE_TALK_IMPORTANCE_FOCUS_THRESHOLD: float = 0.8


@dataclass
class ChangeTalkCandidate:
    id: str
    kind: str
    normalized_text: str
    evidence_quote: str = ""
    evidence_turn: Optional[int] = None
    explicitness: str = "inferred"  # explicit / inferred
    confidence: float = 0.0
    slot_relevance: float = 0.0
    all_phase_slot_relevance: float = 0.0
    target_behavior_relevance: float = 0.0
    origin_type: str = "user_utterance"  # user_utterance / system_reframed
    motivation_focus: str = "importance_related"  # importance_related / confidence_related
    linked_slots: List[str] = field(default_factory=list)


@dataclass
class Decision:
    phase: Phase
    main_action: MainAction
    add_affirm: AffirmationMode
    # Expected next state after applying this decision.
    next_state: DialogueState
    reflection_style: Optional[ReflectionStyle] = None
    risk_mode: RiskLevel = RiskLevel.NONE
    focus_candidates: List[ChangeTalkCandidate] = field(default_factory=list)
    slot_target: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)


# ============================
# Swappable classifier / planner interfaces
# ============================
class ActionRanker(Protocol):
    """
    Priority proposer for main actions (OARS + exceptions).
    - First return value: ranked list of MainAction values
    - Second return value: debug payload such as proposed_main_action
    - If allowed_actions is provided, propose only within that set
    """

    def rank(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
        allowed_actions: Optional[Sequence[MainAction]] = None,
    ) -> Tuple[List[MainAction], Dict[str, Any]]:
        ...


class FeatureExtractor(Protocol):
    """
    Feature extractor (rule-based or LLM-based).
    - Returns: (PlannerFeatures, debug)
    """

    def extract(
        self,
        *,
        user_text: str,
        state: DialogueState,
        cfg: PlannerConfig,
        history: List[Tuple[str, str]],
    ) -> Tuple[PlannerFeatures, Dict[str, Any]]:
        ...


class PhaseSlotFiller(Protocol):
    """
    Phase-slot filler (LLM or rule-based).
    - Returns: (candidate updates, debug)
    """

    def fill_slots(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        user_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        ...


class RiskDetector(Protocol):
    """
    Safety-risk detector (self-harm, violence, etc.).
    - Returns: RiskAssessment
    """

    def detect(
        self,
        *,
        user_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
    ) -> RiskAssessment:
        ...


class OutputEvaluator(Protocol):
    """
    MI-alignment evaluator for generated responses.
    """

    def evaluate(
        self,
        *,
        action: MainAction,
        assistant_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
    ) -> OutputEvaluation:
        ...


class ChangeTalkInferer(Protocol):
    """
    Inference hook for explicit or implied change-talk candidates in the user utterance.
    - The first return value should preferably be structured, though `str` is still accepted for compatibility.
    """

    def infer(
        self,
        *,
        user_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
    ) -> Tuple[Any, Dict[str, Any]]:
        ...


class AffirmationDecider(Protocol):
    """
    Inference hook for the affirmation mode (NONE / SIMPLE / COMPLEX).
    """

    def decide(
        self,
        *,
        user_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
    ) -> Tuple[AffirmationMode, Dict[str, Any]]:
        ...


@dataclass
class Layer1Bundle:
    current_phase: str = ""
    required_slot_keys: List[str] = field(default_factory=list)
    current_slot_values_before_review: Dict[str, str] = field(default_factory=dict)
    review_target_slot_keys: List[str] = field(default_factory=list)
    review_target_slot_refs: List[str] = field(default_factory=list)
    review_scope: str = "mixed"
    target_phase_codes: List[str] = field(default_factory=list)
    precheck_quality_by_slot: Dict[str, float] = field(default_factory=dict)
    review_target_quality_threshold: float = 0.8
    candidate_updates: List[Dict[str, Any]] = field(default_factory=list)
    need_previous_phase_slot_updates: bool = False
    need_future_phase_slot_updates: bool = False
    layer1_note: str = ""
    evidence_validation: List[Dict[str, Any]] = field(default_factory=list)
    missing_slot_keys_precheck: List[str] = field(default_factory=list)
    slot_focus_candidates_precheck: List[str] = field(default_factory=list)
    explicit_correction_slot_keys: List[str] = field(default_factory=list)
    explicit_correction_slot_refs: List[str] = field(default_factory=list)
    focus_choice_needed: bool = False
    focus_choice_candidates: List[str] = field(default_factory=list)
    explicit_focus_preference_present: bool = False
    focus_choice_reason: str = ""


class SlotReviewer(Protocol):
    """
    Layer 2 reviewer for Layer 1 candidates.
    - Reviews Layer1Bundle and returns quality / repair hints.
    - Admissibility remains a Layer 1 responsibility.
    """

    def review(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
        layer1_bundle: Layer1Bundle,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...


@dataclass
class ResponseBrief:
    meta: Dict[str, str] = field(default_factory=dict)
    dialogue_digest: List[str] = field(default_factory=list)
    primary_focus: Dict[str, Any] = field(default_factory=dict)
    utterance_target: str = ""
    question_reflect_seed: str = ""
    evocation_move: str = ""
    phase_goal_this_turn: str = ""
    missing_info_candidate: str = ""
    slot_quality_target_examples: List[Dict[str, str]] = field(default_factory=list)
    repair_goal_text: str = ""
    success_criterion: str = ""
    ct_anchors: List[str] = field(default_factory=list)
    ct_operation_goal: str = "Strengthen"
    slot_goal: str = ""
    draft_response_text: str = ""
    question_shape: str = "none"
    use_reflection_first: bool = False
    writer_plan: Dict[str, Any] = field(default_factory=dict)
    must_include: List[str] = field(default_factory=list)
    must_avoid: List[str] = field(default_factory=list)
    language_constraints: Dict[str, Any] = field(default_factory=dict)
    brief_confidence: float = 0.0
    note_for_debug: str = ""

    @property
    def slot_repair_hints(self) -> List[Dict[str, str]]:
        return self.slot_quality_target_examples

    @slot_repair_hints.setter
    def slot_repair_hints(self, value: Sequence[Mapping[str, Any]] | None) -> None:
        normalized: List[Dict[str, str]] = []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, Mapping):
                    normalized.append(dict(item))
        self.slot_quality_target_examples = normalized


class ResponseIntegrator(Protocol):
    """
    Layer 3 response-brief generator.
    - Input: history, state, Layer 2 decisions, internal signals
    - Output: ResponseBrief (contract-like constraints + completed draft)
    """

    def integrate(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        action: MainAction,
        add_affirm: AffirmationMode,
        reflection_style: Optional[ReflectionStyle],
        risk_assessment: Optional[RiskAssessment],
        focus_candidates: Optional[Sequence[ChangeTalkCandidate]],
        slot_target: str,
        first_turn_hint: Optional[FirstTurnHint],
        allow_question_without_preface: bool,
        current_user_is_short: bool,
        phase_prediction_debug: Optional[Dict[str, Any]],
        action_ranking_debug: Optional[Dict[str, Any]],
        features: Optional[PlannerFeatures],
        change_talk_hint: Optional[str],
        focus_choice_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ResponseBrief, Dict[str, Any]]:
        ...


# ----------------------------
# Feature extraction (lightweight)
# ----------------------------
_RE_QUESTION = re.compile(r"[?？]|(?:could you|would you|can you|what|how|why|when|where)\b", re.IGNORECASE)
_RE_INFO_REQ = re.compile(
    r"(?:tell me|want to know|information|advice|suggestion|options|approach|tips|how should I|what should I do)",
    re.IGNORECASE,
)
_RE_YES = re.compile(
    r"^(?:yes|yeah|yep|sure|please|sounds good|that's fine|that's okay|okay|ok|got it|absolutely)",
    re.IGNORECASE,
)
_RE_NO = re.compile(
    r"^(?:no|nope|stop|don't|do not|not now|not needed|no thanks|rather not)",
    re.IGNORECASE,
)
_ASK_PERMISSION_REQUIRED_KEYWORDS = ("information", "share", "suggest", "option", "approach")
_ASK_PERMISSION_FORBIDDEN_PATTERNS = (
    re.compile(r"let me go ahead", re.IGNORECASE),
    re.compile(r"with your permission", re.IGNORECASE),
    re.compile(r"i'?ll decide", re.IGNORECASE),
    re.compile(r"we'?ll do that", re.IGNORECASE),
)
_PROVIDE_INFO_OPTION_PATTERNS = (
    re.compile(r"options?", re.IGNORECASE),
    re.compile(r"(?:A\s*/\s*B(?:\s*/\s*C)?)", re.IGNORECASE),
    re.compile(r"(?:\([ABCabc123]\)|[ABCabc123][\)）]|[①-③])"),
)
_PROVIDE_INFO_NEUTRAL_PHRASES = ("as one option", "as one possible approach")
_PROVIDE_INFO_REACTION_MARKERS = (
    "how does that feel",
    "what do you think",
    "how does that sound",
    "which one",
    "what feels doable",
    "what might work",
    "what seems realistic",
    "what could you try",
    "what seems to fit",
)
_AUTONOMY_FORBIDDEN_PATTERNS = (
    re.compile(r"we'?ll do that", re.IGNORECASE),
    re.compile(r"i'?ll take care of that", re.IGNORECASE),
    re.compile(r"this week we'?ll", re.IGNORECASE),
)
_ACTION_MISMATCH_SUGGESTION_PATTERNS = (
    re.compile(r"let'?s", re.IGNORECASE),
    re.compile(r"you should", re.IGNORECASE),
    re.compile(r"it would be good to", re.IGNORECASE),
    re.compile(r"i recommend", re.IGNORECASE),
    re.compile(r"i'?d recommend", re.IGNORECASE),
    re.compile(r"try\b", re.IGNORECASE),
    re.compile(r"how about\b", re.IGNORECASE),
)
_ACTION_MISMATCH_CHOICE_PATTERNS = (
    re.compile(r"which one", re.IGNORECASE),
    re.compile(r"which would you choose", re.IGNORECASE),
    re.compile(r"which side", re.IGNORECASE),
    re.compile(r"options?", re.IGNORECASE),
)
_SUMMARY_FORWARD_PROMPT_PATTERNS = (
    re.compile(r"(?:next|after this|moving forward|from here).{0,20}(?:step|action|plan).{0,20}(?:what|how|decide|think|tell)", re.IGNORECASE),
    re.compile(r"(?:specific|next).{0,12}(?:step|action|plan).{0,20}(?:what|how|decide|think|tell)", re.IGNORECASE),
    re.compile(r"(?:where|what) would you start", re.IGNORECASE),
)
_SUMMARY_EMOTION_MARKERS: Tuple[str, ...] = (
    "feeling",
    "emotion",
    "anxious",
    "worry",
    "uncertain",
    "torn",
    "hard",
    "painful",
    "scared",
    "confused",
    "overwhelmed",
)
_SUMMARY_VALUE_MARKERS: Tuple[str, ...] = (
    "value",
    "important",
    "meaning",
    "protect",
    "want to be",
    "priority",
    "reason",
    "want",
)
_SUMMARY_AMBIVALENCE_MARKERS: Tuple[str, ...] = (
    "on the other hand",
    "at the same time",
    "but",
    "however",
    "both",
    "torn",
    "uncertain",
)
_REVIEW_FIRST_TURN_SUMMARY_MARKERS: Tuple[str, ...] = (
    "so far",
    "up to now",
    "today's session",
    "this session",
)
_REVIEW_FIRST_TURN_QUESTION_MARKERS: Tuple[str, ...] = (
    "notice",
    "learning",
    "reflection",
    "what stands out",
    "takeaway",
)
_CLOSING_FIRST_TURN_ENDING_MARKERS: Tuple[str, ...] = (
    "end",
    "wrap up",
    "finish",
    "close",
)
_CLOSING_FIRST_TURN_SESSION_MARKERS: Tuple[str, ...] = (
    "today",
    "this time",
    "this session",
    "session",
)
_FINAL_CLOSING_MESSAGE = "Thank you. We will end today's session here."
_CT_STRENGTH_VALUE_REASON_MARKERS: Tuple[str, ...] = (
    "reason",
    "because",
    "value",
    "important",
    "meaning",
    "want",
)
_CT_STRENGTH_NEXT_STEP_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(next step|first step|to start|first)", re.IGNORECASE),
    re.compile(r"(try|start|continue|work on|decide|put into practice)", re.IGNORECASE),
)
_CT_STRENGTH_SCENE_TRIGGER_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(when|if|right before|right after|situation|timing)", re.IGNORECASE),
)
_OBSERVABLE_BEHAVIOR_CONTEXT_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(today|tomorrow|this week|next week|morning|night|weekday|weekend|every day|every week|\d+\s*(?:times?|minutes?|hours?)|where|when|situation)", re.IGNORECASE),
)
_NEXT_STEP_AUTONOMY_MARKERS: Tuple[str, ...] = (
    "for you",
    "on your own",
    "feels like a fit",
    "seems doable",
    "seems realistic",
    "choose",
    "without forcing it",
)
_RE_HAS_JAPANESE_CHAR = re.compile(r"[\u3041-\u3093\u30a1-\u30f6\u4e00-\u9fa5\u3005]")
_RE_JSON_LIKE_TEXT = re.compile(r"^\s*[\[{].*[\]}]\s*$", re.DOTALL)
_RE_DIALOGUE_ROLE_PREFIX = re.compile(r"^\s*(assistant|system|user)\s*[:：]", re.IGNORECASE)
_RE_CASUAL_ENDING = re.compile(r"(?:yeah|yep|kinda|sorta|gonna|wanna|ain't)(?:[.?!\s]*)$", re.IGNORECASE)
_RE_CHAT_SLANG = re.compile(r"(?:\blol+\b|\blmao\b|\brofl\b)")
_RE_SUSPICIOUS_ASCII_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]{2,20})(?![A-Za-z0-9_])")
_ALLOWED_INLINE_ASCII_TOKENS: Set[str] = {
    "ok",
    "sns",
    "line",
}
_REFLECTION_ELLIPSIS_ENDING_PATTERNS = (
    re.compile(r".+that'?s what it is$", re.IGNORECASE),
    re.compile(r".+that'?s how it feels$", re.IGNORECASE),
    re.compile(r".+want to$", re.IGNORECASE),
    re.compile(r".+don'?t want to$", re.IGNORECASE),
    re.compile(r".+wanted to$", re.IGNORECASE),
    re.compile(r".+didn'?t want to$", re.IGNORECASE),
    re.compile(r".{2,}but.{2,}$", re.IGNORECASE),
)
_REFLECT_ENDING_FAMILY_PATTERNS: Tuple[Tuple[str, Tuple[re.Pattern[str], ...]], ...] = (
    ("sounds_like_family", (re.compile(r"sounds like$", re.IGNORECASE),)),
    (
        "might_be_family",
        (
            re.compile(r"might be$", re.IGNORECASE),
            re.compile(r"could be$", re.IGNORECASE),
        ),
    ),
    (
        "that_means_family",
        (
            re.compile(r"that means$", re.IGNORECASE),
            re.compile(r"that'?s how it feels$", re.IGNORECASE),
        ),
    ),
    ("is_that_right_family", (re.compile(r"isn'?t it$", re.IGNORECASE),)),
    (
        "elliptical_family",
        (
            re.compile(r".+that$", re.IGNORECASE),
            re.compile(r".+to$", re.IGNORECASE),
        ),
    ),
)
_REFLECT_ADVICE_LIKE_PATTERNS = (
    re.compile(r"how about trying", re.IGNORECASE),
    re.compile(r"is effective", re.IGNORECASE),
    re.compile(r"is appropriate", re.IGNORECASE),
    re.compile(r"let'?s keep", re.IGNORECASE),
    re.compile(r"recommend.{0,40}", re.IGNORECASE),
)
_OVERINTERPRETATION_INFERENCE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"sounds like", re.IGNORECASE),
    re.compile(r"somewhere you feel", re.IGNORECASE),
    re.compile(r"(?:connects|becomes).{0,8}(?:feeling|thought)", re.IGNORECASE),
)
_OVERINTERPRETATION_STORY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"foundation of safety", re.IGNORECASE),
    re.compile(r"use as a foundation", re.IGNORECASE),
    re.compile(r"as a strength", re.IGNORECASE),
    re.compile(r"starting to take shape", re.IGNORECASE),
    re.compile(r"grow", re.IGNORECASE),
    re.compile(r"build up", re.IGNORECASE),
    re.compile(r"from here", re.IGNORECASE),
    re.compile(r"connects to what matters", re.IGNORECASE),
    re.compile(r"(?:connects|becomes).{0,6}entry point", re.IGNORECASE),
)

_SOFT_WARNING_LAYER4_REWRITE_ISSUE_MAP: Dict[str, str] = {
    "layer4_not_shorter_than_draft": "layer4_not_shorter_than_draft",
    "overinterpretation_risk": "overinterpretation_risk",
    "reflect_ending_family_bias": "reflect_ending_family_bias",
    "reflect_missing_change_talk_focus": "reflect_missing_change_talk_focus",
    "reflect_change_talk_not_strengthened": "reflect_change_talk_not_strengthened",
    "summary_missing_change_talk_focus": "summary_missing_change_talk_focus",
    "summary_change_talk_not_strengthened": "summary_change_talk_not_strengthened",
    "next_step_missing_concrete_step": "next_step_missing_concrete_step",
    "next_step_missing_autonomy_support": "next_step_missing_autonomy_support",
    "focusing_missing_observable_target_behavior": "focusing_missing_observable_target_behavior",
}
_SOFT_WARNING_LAYER4_REWRITE_ACTIONS: Set[MainAction] = {
    MainAction.REFLECT,
    MainAction.REFLECT_SIMPLE,
    MainAction.REFLECT_COMPLEX,
    MainAction.REFLECT_DOUBLE,
    MainAction.SUMMARY,
}
_LAYER4_GLOBAL_REWRITE_WARNING_CODES: Set[str] = {
    "layer4_not_shorter_than_draft",
}
_LAYER4_FALLBACK_ISSUE_REPAIR_GUIDANCE: Dict[str, str] = {
    "draft_response_text_missing": "draft_response_text is missing. Rebuild only one minimal natural sentence using utterance_target and must_include.",
    "layer4_not_shorter_than_draft": "The final sentence is not shorter than draft_response_text. Shorten it without adding meaning, while keeping the concrete wording.",
    "reflect_ending_family_bias": "Recent assistant reflections keep repeating the same ending family. Prioritize writer_plan.avoid_recent_ending_families and avoid repeating the same family.",
    "question_preface_missing": "QUESTION / ASK_PERMISSION(add_affirm=NONE) must use a simple reflection in sentence 1 and place the question in the final sentence. Do not omit sentence 1.",
    "question_preface_not_grounded_in_change_talk": "Return sentence 1 of QUESTION / ASK_PERMISSION(add_affirm=NONE) to question_reflect_seed / change_talk and do not drift to another focus.",
}
_LAYER4_EDIT_AUDIT_ABSTRACT_TERMS: Tuple[str, ...] = (
    "meaning-making",
    "foundation",
    "link point",
    "support",
    "direction",
    "sense",
    "approach",
    "connection",
    "entry point",
)
_LAYER4_EDIT_AUDIT_META_BRIDGE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\"[^\"]{2,80}\" is an entry point to \"[^\"]{2,120}\"", re.IGNORECASE),
)
_LAYER4_EDIT_AUDIT_QUOTED_PHRASE_PATTERN = re.compile(r"「([^」]{2,80})」")
_LAYER4_EDIT_AUDIT_CONCRETE_SPAN_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"((?:tonight|today|tomorrow|morning|night|after getting home|before a meeting|prep time|break time|quiet seat|meeting room)[^.]{0,24}?"
        r"(?:review|write down|say out loud|practice|tell someone|try|continue|start))",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:note|notebook|one sentence)[^.]{0,18}?"
        r"(?:review|write down|say out loud|practice|tell someone))",
        re.IGNORECASE,
    ),
    re.compile(r"((?:[0-9]+(?:\s*minutes?)?[^.]{0,18}?(?:say out loud|practice|review|continue)))", re.IGNORECASE),
)
_LAYER4_EDIT_AUDIT_SUBJECT_REVERSAL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("your", "my"),
    ("you", "I"),
    ("your", "our"),
)
_LAYER4_EDIT_AUDIT_VIEWPOINT_REVERSAL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("you", "I"),
    ("you", "we"),
)
_LAYER4_EDIT_AUDIT_AUTONOMY_ANCHORS: Tuple[str, ...] = (
    "you decide",
    "your sense",
    "for you",
    "decide for yourself",
    "your own sense",
)
_LAYER4_EDIT_AUDIT_FIRST_PERSON_MARKERS: Tuple[str, ...] = (
    "I",
    "we",
    "our side",
)
_LAYER4_DRAFT_HARD_CONTRACT_ISSUE_CODES: Set[str] = {
    "scale_followup_question_missing_plus_one_probe",
    "scale_followup_question_missing_reason_probe",
    "scale_followup_question_missing_zero_anchor",
    "question_preface_missing",
    "question_preface_not_grounded_in_change_talk",
    "ask_permission_not_question",
    "ask_permission_not_single_sentence",
    "summary_contains_question",
    "summary_sentence_count_out_of_range",
    "contract_question_count_exceeded",
    "affirmation_unexpected",
}
_LAYER4_DRAFT_SOFT_HINT_WARNING_CODES: Set[str] = {
    "affirmation_simple_missing",
    "affirmation_complex_missing",
    "reflect_ending_family_bias",
    "reflect_missing_change_talk_focus",
    "reflect_change_talk_not_strengthened",
    "summary_missing_emotion_layer",
    "summary_missing_value_layer",
    "summary_missing_ambivalence_layer",
    "summary_missing_change_talk_focus",
    "summary_change_talk_not_strengthened",
    "next_step_missing_concrete_step",
    "next_step_missing_autonomy_support",
    "focusing_missing_observable_target_behavior",
}

_INTERNAL_LABEL_LITERAL_TOKENS: Tuple[str, ...] = (
    "fixed_turn_focus",
    "fixed_focus_candidates",
    "focus_candidates",
    "primary_focus_id",
    "secondary_focus_id",
    "linked_slots",
    "selection_reason",
    "feature_extractor",
    "phase_slots",
    "phase_slots_pending",
    "current_phase_slots_pending",
    "change_talk_inference",
)
_RE_INTERNAL_CT_ID = re.compile(r"(?<![A-Za-z0-9_])ct_\d+(?![A-Za-z0-9_])", re.IGNORECASE)
_RE_INTERNAL_SNAKE_CASE_LONG = re.compile(r"\b[a-z]{2,}(?:_[a-z0-9]{2,}){2,}\b", re.IGNORECASE)

# Strong advice-request markers, checked on whitespace-stripped text.
_INFO_REQUEST_STRONG_MARKERS = [
    "please tell me",
    "i want advice",
    "can you advise me",
    "what do you recommend",
    "please suggest something",
    "i want suggestions",
    "what are the options",
    "how should i do this",
    "what should i do",
]


def _find_internal_label_leak(text: Any) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    for token in _INTERNAL_LABEL_LITERAL_TOKENS:
        if token in lowered:
            return token
    ct_match = _RE_INTERNAL_CT_ID.search(raw)
    if ct_match:
        return ct_match.group(0)
    snake_match = _RE_INTERNAL_SNAKE_CASE_LONG.search(raw)
    if snake_match:
        return snake_match.group(0)
    return None


def _strip_internal_labels(text: Any) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return ""
    cleaned = _RE_INTERNAL_CT_ID.sub("", cleaned)
    cleaned = _RE_INTERNAL_SNAKE_CASE_LONG.sub("", cleaned)
    for token in _INTERNAL_LABEL_LITERAL_TOKENS:
        cleaned = re.sub(re.escape(token), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,，:：;；/|[]{}()<>")
    return cleaned


def _sanitize_human_text(value: Any) -> str:
    text = _normalize_slot_text(value)
    if not text:
        return ""
    if not _find_internal_label_leak(text):
        return text
    return _normalize_slot_text(_strip_internal_labels(text))


def _sanitize_human_draft_text(value: Any, *, max_len: int = 420) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if _find_internal_label_leak(text):
        text = _strip_internal_labels(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "unknown", "n/a", "na"}:
        return ""
    if text in {"unknown", "unset", "none"}:
        return ""
    return text[: max(80, int(max_len))]


def _sanitize_slot_quality_target_example_detail(value: Any, *, max_len: int = 420) -> str:
    return _sanitize_human_draft_text(value, max_len=max_len)


def _extract_slot_quality_target_example_detail(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    for field in ("detail", "quality_upgrade_model_text", "quality_upgrade_hint", "quality_upgrade_text"):
        detail = _sanitize_slot_quality_target_example_detail(item.get(field), max_len=420)
        if detail:
            return detail
    return ""


def _extract_slot_quality_target_example_detail_raw(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    for field in ("detail", "quality_upgrade_model_text", "quality_upgrade_hint", "quality_upgrade_text"):
        value = item.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lower = text.lower()
        if lower in {"none", "null", "unknown", "n/a", "na"}:
            continue
        if text in {"unknown", "unset"}:
            continue
        return text
    return ""


def _extract_slot_quality_target_information(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    target_information = _sanitize_human_text(item.get("target_information"))
    if target_information:
        return target_information
    return ""


def _extract_slot_repair_hint_detail(item: Any) -> str:
    return _extract_slot_quality_target_example_detail(item)


def _extract_slot_repair_hint_detail_raw(item: Any) -> str:
    return _extract_slot_quality_target_example_detail_raw(item)


def _contains_internal_label_in_any(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _find_internal_label_leak(str(key)):
                return True
            if _contains_internal_label_in_any(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_internal_label_in_any(item) for item in value)
    return _find_internal_label_leak(value) is not None

# Resistance signals (coarse heuristic is acceptable here).
_RESIST_MARKERS = [
    "but", "however", "can't", "cannot", "hard", "difficult", "don't want to", "hate it", "pointless",
    "don't need", "waste", "what's the point", "too much", "exhausting", "annoying", "no motivation",
    "failed before", "won't work", "can't keep it up", "not feeling up for it",
]

# Change talk signals.
_CHANGE_TALK_MARKERS = [
    "want to", "try", "would like to try", "change", "be different", "work on it", "seems doable",
    "need", "important", "goal", "challenge", "improve", "should do", "keep going",
    "a little more", "want to test", "want to work on", "be able to",
]

# Discord / alliance-rupture signals.
_DISCORD_MARKERS = [
    "you don't get it",
    "that's not it",
    "that's not what I mean",
    "you're pushing me",
    "don't blame me",
    "why",
    "enough",
    "don't want to talk",
    "annoying",
    "irritated",
    "angry",
    "can't accept that",
    "can't trust that",
]

_AMBIVALENCE_SUSTAIN_HINTS = [
    "scared", "anxious", "hard", "impossible", "can't", "won't keep it up", "exhausting", "not feeling up for it",
    "not confident", "unsure",
]


def _char_bigrams(text: str) -> set:
    t = re.sub(r"\s+", "", text)
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _estimate_novelty(user_text: str, last_user_text: str) -> float:
    clean_len = len(re.sub(r"\s+", "", user_text))

    # 1) Character bigram distance.
    sim = _jaccard(_char_bigrams(user_text), _char_bigrams(last_user_text))
    novelty = 1.0 - sim

    # 2) Boost when new numbers or time-like details appear.
    nums_now = set(re.findall(r"\d+", user_text))
    nums_prev = set(re.findall(r"\d+", last_user_text))
    if nums_now - nums_prev:
        novelty = min(1.0, novelty + 0.15)

    # 3) Small boost when new proper-noun-like katakana spans appear.
    kata_now = set(re.findall(r"[\u30a1-\u30f4\u30fc]{3,}", user_text))
    kata_prev = set(re.findall(r"[\u30a1-\u30f4\u30fc]{3,}", last_user_text))
    if kata_now - kata_prev:
        novelty = min(1.0, novelty + 0.10)

    # 4) Even very short inputs should not drop fully to 0 because reflection can still build on context.
    if clean_len <= 8:
        # Cap novelty so short backchannels without new information do not look maximally novel.
        novelty = min(novelty, 0.45)
        novelty = max(novelty, 0.15)
    elif clean_len <= 14:
        novelty = min(novelty, 0.8)

    return max(0.0, min(1.0, novelty))


def _score_from_markers(text: str, markers: Sequence[str], *, cap: int = 4) -> float:
    hits = sum(1 for m in markers if m in text)
    return min(1.0, hits / cap)


def _detect_permission(text: str) -> Optional[bool]:
    t = text.strip()
    if _RE_YES.search(t):
        return True
    if _RE_NO.search(t):
        return False
    if "that's fine" in t or "that's okay" in t:
        return True
    if "stop" in t or "please stop" in t:
        return False
    normalized = re.sub(r"\s+", "", t)
    soft_yes_markers = [
        "a little is okay",
        "keep it brief",
        "if it's simple",
        "for now, sure",
        "i want to hear a bit",
        "i'd like to hear it",
        "part of it is okay",
        "i'm okay with it now",
        "it would help if you told me",
        "i'd appreciate hearing it",
        "i want to know",
        "i want details",
        "i'd use that",
        "if possible i'd like to hear it",
        "it would help to have an approach",
        "it would help to have an example",
        "i want the key points",
        "i could try it tonight",
    ]
    soft_no_markers = [
        "not yet",
        "not now",
        "i'm okay without it for now",
        "i want to stop for today",
        "i'd rather leave it",
        "maybe another time",
        "i want to sort things out first",
        "i'm not feeling ready",
        "because i'm scared",
        "because i'm anxious",
    ]
    yes_hits = sum(1 for marker in soft_yes_markers if marker in normalized)
    no_hits = sum(1 for marker in soft_no_markers if marker in normalized)
    if yes_hits > no_hits and yes_hits > 0:
        return True
    if no_hits > yes_hits and no_hits > 0:
        return False
    return None


def _detect_info_request(text: str) -> bool:
    """
    Return True only when advice is being explicitly requested.
    - Exact strong-request phrases have top priority.
    - Otherwise limit it to question-like utterances that also contain information-request keywords.
    """
    normalized = re.sub(r"\s+", "", text)
    if any(marker in normalized for marker in _INFO_REQUEST_STRONG_MARKERS):
        return True
    return bool(_RE_INFO_REQ.search(text) and _RE_QUESTION.search(text))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _coerce_bool_with_ja(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = re.sub(r"\s+", "", value).lower()
        if s in {
            "true",
            "1",
            "yes",
            "y",
            "on",
            "t",
            "yes",
            "true",
            "needed",
            "need",
            "present",
            "have",
        }:
            return True
        if s in {
            "false",
            "0",
            "no",
            "n",
            "off",
            "f",
            "no",
            "false",
            "not needed",
            "none",
            "absent",
        }:
            return False
    return default


def _coerce_permission(value: Any, default: Optional[bool]) -> Optional[bool]:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
        if s in {"null", "none", "unknown"}:
            return default
    return default


def _clamp01(value: Any, default: float) -> float:
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return float(max(0.0, min(1.0, default)))


def _coerce_optional_scale_0_1(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        score = float(value)
    except Exception:
        return default
    if score <= 1.0:
        return float(max(0.0, min(1.0, score)))
    # Backward compatibility: accept legacy 0-10 inputs after normalizing them to 0-1.
    return float(max(0.0, min(1.0, score / 10.0)))


def _compute_need_summary(*, state: DialogueState, topic_shift: bool, cfg: PlannerConfig) -> bool:
    """
    Make SUMMARY more likely once roughly `interval` turns have passed since the previous one.
    - Stay in cooldown before reaching `interval`
    - Just before `interval` (-1), allow it when there is a reflection streak or a topic shift
    """
    turns = max(0, int(state.turns_since_summary))
    interval = max(1, int(cfg.summary_interval_turns))
    if turns >= interval:
        return True

    preheat_floor = max(0, interval - 1)
    if turns < preheat_floor:
        return False

    return state.reflect_streak >= cfg.summary_reflect_streak_trigger or topic_shift


def _summary_cadence_multiplier(
    *,
    turns_since_summary: int,
    cfg: PlannerConfig,
    need_summary: bool = False,
) -> Tuple[float, str]:
    """
    Softly control SUMMARY cadence.
    - Suppress it during cooldown
    - Boost it once `interval` is reached
    """
    turns = max(0, int(turns_since_summary))
    interval = max(1, int(cfg.summary_interval_turns))

    if turns >= (interval + 3):
        return 1.15, "interval_plus_3_or_more"
    if turns >= interval:
        return 1.35, "interval_reached"

    preheat_turn = max(0, interval - 1)
    if turns >= preheat_turn:
        if need_summary:
            return 1.04, "preheat_ready"
        return 0.88, "preheat_hold"

    progress = turns / float(interval)
    return (0.45 + 0.45 * progress), "cooldown"


def _merge_feature_overlay(
    *,
    base_features: PlannerFeatures,
    overlay: Mapping[str, Any],
    user_text: str,
    state: DialogueState,
    cfg: PlannerConfig,
) -> Tuple[PlannerFeatures, Dict[str, Any]]:
    """
    Build PlannerFeatures by layering an LLM-derived overlay on top of rule-based baseline features.
    - Prefer LLM values, and fall back to rule values only when they are missing or invalid
    """
    resistance = _clamp01(overlay.get("resistance", base_features.resistance), base_features.resistance)
    discord = _clamp01(overlay.get("discord", base_features.discord), base_features.discord)
    change_talk = _clamp01(overlay.get("change_talk", base_features.change_talk), base_features.change_talk)
    novelty = _clamp01(overlay.get("novelty", base_features.novelty), base_features.novelty)
    importance_estimate = _coerce_optional_scale_0_1(
        overlay.get(
            "importance_estimate",
            overlay.get("importance_scale", base_features.importance_estimate),
        ),
        base_features.importance_estimate,
    )
    confidence_estimate = _coerce_optional_scale_0_1(
        overlay.get(
            "confidence_estimate",
            overlay.get("confidence_scale", base_features.confidence_estimate),
        ),
        base_features.confidence_estimate,
    )

    clean_len = len(re.sub(r"\s+", "", user_text))
    fallback_is_short_reply = clean_len <= 12
    fallback_topic_shift = novelty >= 0.85 and clean_len >= 15
    fallback_need_summary = _compute_need_summary(state=state, topic_shift=fallback_topic_shift, cfg=cfg)
    fallback_allow_reflect_override = (
        change_talk >= cfg.allow_override_change_talk
        or resistance >= cfg.allow_override_resistance
        or novelty >= cfg.allow_override_novelty
    )

    user_is_question = _coerce_bool(
        overlay.get("user_is_question"),
        base_features.user_is_question,
    )
    user_requests_info = _coerce_bool(
        overlay.get("user_requests_info"),
        base_features.user_requests_info,
    )
    has_permission = _coerce_permission(
        overlay.get("has_permission"),
        base_features.has_permission,
    )
    is_short_reply = _coerce_bool(
        overlay.get("is_short_reply"),
        fallback_is_short_reply,
    )
    topic_shift = _coerce_bool(
        overlay.get("topic_shift"),
        fallback_topic_shift,
    )
    need_summary = _coerce_bool(
        overlay.get("need_summary"),
        fallback_need_summary,
    )
    allow_reflect_override = _coerce_bool(
        overlay.get("allow_reflect_override"),
        fallback_allow_reflect_override,
    )

    merged = dataclasses.replace(
        base_features,
        resistance=resistance,
        discord=discord,
        change_talk=change_talk,
        novelty=novelty,
        user_is_question=user_is_question,
        user_requests_info=user_requests_info,
        has_permission=has_permission,
        is_short_reply=is_short_reply,
        topic_shift=topic_shift,
        need_summary=need_summary,
        allow_reflect_override=allow_reflect_override,
        importance_estimate=importance_estimate,
        confidence_estimate=confidence_estimate,
    )
    debug = {
        "llm_first_keys": [
            "user_is_question",
            "user_requests_info",
            "has_permission",
            "is_short_reply",
            "topic_shift",
            "need_summary",
            "allow_reflect_override",
            "importance_estimate",
            "confidence_estimate",
        ],
    }
    return merged, debug


def _extract_phase_feature_overlay(payload: Mapping[str, Any]) -> Dict[str, Any]:
    features = payload.get("features")
    if isinstance(features, dict):
        return dict(features)

    keys = {
        "resistance",
        "discord",
        "change_talk",
        "novelty",
        "importance_estimate",
        "confidence_estimate",
        "importance_scale",
        "confidence_scale",
        "user_is_question",
        "user_requests_info",
        "has_permission",
        "is_short_reply",
        "topic_shift",
        "need_summary",
        "allow_reflect_override",
    }
    overlay: Dict[str, Any] = {}
    for k in keys:
        if k in payload:
            overlay[k] = payload.get(k)
    return overlay


def _parse_phase_from_any(value: Any) -> Optional[Phase]:
    if isinstance(value, Phase):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for phase in Phase:
        if text == phase.value or text == phase.name:
            return phase
    upper = text.upper()
    if upper in Phase.__members__:
        return Phase[upper]
    for phase in Phase:
        if phase.value in text:
            return phase
    return None


def _normalize_slot_text(value: Any) -> str:
    if value is None:
        return ""

    text = ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    elif isinstance(value, list):
        items = []
        for item in value:
            s = _normalize_slot_text(item)
            if s:
                items.append(s)
        text = " / ".join(items[:4])
    elif isinstance(value, dict):
        parts: List[str] = []
        for k, v in value.items():
            sv = _normalize_slot_text(v)
            if sv:
                parts.append(f"{k}:{sv}")
        text = " / ".join(parts[:4])
    else:
        text = str(value)

    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"none", "null", "unknown", "n/a", "na"}:
        return ""
    if text in {"unknown", "unset", "none"}:
        return ""
    return text[:120]


def _normalize_turn_id_list(
    value: Any,
    *,
    min_turn: int = 1,
    max_turn: Optional[int] = None,
) -> List[int]:
    turn_ids: List[int] = []
    items: List[Any]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    elif value is None:
        items = []
    else:
        items = [value]
    for item in items:
        try:
            turn_id = int(item)
        except Exception:
            continue
        if turn_id < min_turn:
            continue
        if max_turn is not None and turn_id > max_turn:
            continue
        if turn_id in turn_ids:
            continue
        turn_ids.append(turn_id)
        if len(turn_ids) >= 6:
            break
    return turn_ids


_SLOT_APPLY_CONFIDENCE_THRESHOLD = 0.70
_SLOT_LAYER1_ADMISSIBILITY_CONFIDENCE_THRESHOLD = 0.55
_SLOT_CONFIRM_QUALITY_THRESHOLD = 0.50
_SLOT_LAYER2_REVIEW_TARGET_QUALITY_THRESHOLD = 0.80
_SLOT_REPAIR_HINT_TARGET_QUALITY = 1.00
_EXPLICIT_NUMERIC_SCALE_QUALITY_FLOOR = 0.80
# Minimum fallback value when the LLM reviewer does not return `quality_score`
_SLOT_FALLBACK_QUALITY_SCORE = 0.62


def _normalize_slot_candidate(value: Any) -> Dict[str, Any]:
    raw_value: Any = value
    raw_evidence: Any = ""
    raw_user_quote: Any = ""
    raw_assistant_quote: Any = ""
    raw_confidence: Any = 0.0
    raw_evidence_turn_ids: Any = []
    raw_evidence_turn_ids_provided: bool = False
    raw_user_evidence_turn_ids: Any = []
    raw_user_evidence_turn_ids_provided: bool = False
    raw_assistant_evidence_turn_ids: Any = []
    raw_assistant_evidence_turn_ids_provided: bool = False
    raw_assent_bridge: Any = False
    raw_assent_bridge_source_turn_id: Any = 0
    raw_explicit_correction: Any = False
    is_structured = False

    if isinstance(value, Mapping):
        marker_keys = {
            "value",
            "slot_value",
            "text",
            "content",
            "candidate",
            "evidence_quote",
            "evidence",
            "quote",
            "evidence_turn_ids",
            "turn_ids",
            "evidence_turns",
            "user_quote",
            "user_evidence_quote",
            "assistant_quote",
            "assistant_evidence_quote",
            "user_evidence_turn_ids",
            "user_evidence_turn_id",
            "user_turn_ids",
            "assistant_evidence_turn_ids",
            "assistant_evidence_turn_id",
            "assistant_turn_ids",
            "confidence",
            "assent_bridge",
            "assent_bridge_source_turn_id",
            "explicit_correction",
            "correction",
            "correction_flag",
        }
        is_structured = any(key in value for key in marker_keys)
        if is_structured:
            raw_value = (
                value.get("value")
                or value.get("slot_value")
                or value.get("text")
                or value.get("content")
                or value.get("candidate")
            )
            raw_evidence = (
                value.get("evidence_quote")
                or value.get("evidence")
                or value.get("quote")
                or ""
            )
            raw_user_quote = (
                value.get("user_quote")
                or value.get("user_evidence_quote")
                or ""
            )
            raw_assistant_quote = (
                value.get("assistant_quote")
                or value.get("assistant_evidence_quote")
                or ""
            )
            raw_evidence_turn_ids = (
                value.get("evidence_turn_ids")
                or value.get("turn_ids")
                or value.get("evidence_turns")
                or []
            )
            raw_evidence_turn_ids_provided = any(
                key in value for key in ("evidence_turn_ids", "turn_ids", "evidence_turns")
            )
            raw_user_evidence_turn_ids = (
                value.get("user_evidence_turn_ids")
                or value.get("user_evidence_turn_id")
                or value.get("user_turn_ids")
                or []
            )
            raw_user_evidence_turn_ids_provided = any(
                key in value for key in ("user_evidence_turn_ids", "user_evidence_turn_id", "user_turn_ids")
            )
            raw_assistant_evidence_turn_ids = (
                value.get("assistant_evidence_turn_ids")
                or value.get("assistant_evidence_turn_id")
                or value.get("assistant_turn_ids")
                or []
            )
            raw_assistant_evidence_turn_ids_provided = any(
                key in value for key in ("assistant_evidence_turn_ids", "assistant_evidence_turn_id", "assistant_turn_ids")
            )
            raw_confidence = value.get("confidence")
            raw_assent_bridge = value.get("assent_bridge")
            raw_assent_bridge_source_turn_id = value.get("assent_bridge_source_turn_id")
            if "explicit_correction" in value:
                raw_explicit_correction = value.get("explicit_correction")
            elif "correction" in value:
                raw_explicit_correction = value.get("correction")
            elif "correction_flag" in value:
                raw_explicit_correction = value.get("correction_flag")
        else:
            raw_value = value

    normalized_value = _normalize_slot_text(raw_value)
    normalized_user_quote = _normalize_slot_text(raw_user_quote) or _normalize_slot_text(raw_evidence)
    normalized_assistant_quote = _normalize_slot_text(raw_assistant_quote)
    normalized_user_turn_ids = _normalize_turn_id_list(
        raw_user_evidence_turn_ids if raw_user_evidence_turn_ids_provided else raw_evidence_turn_ids,
        min_turn=1,
    )
    normalized_assistant_turn_ids = _normalize_turn_id_list(raw_assistant_evidence_turn_ids, min_turn=1)
    normalized_evidence = normalized_user_quote or normalized_assistant_quote
    normalized_turn_ids = list(normalized_user_turn_ids)
    normalized_confidence = _clamp01(raw_confidence, 0.0)
    if isinstance(raw_assent_bridge, bool):
        normalized_assent_bridge = raw_assent_bridge
    else:
        normalized_assent_bridge = _coerce_bool_with_ja(raw_assent_bridge, False)
    normalized_assent_bridge_turn_ids = _normalize_turn_id_list(
        raw_assent_bridge_source_turn_id,
        min_turn=1,
    )
    normalized_assent_bridge_source_turn_id = (
        normalized_assent_bridge_turn_ids[0] if normalized_assent_bridge_turn_ids else 0
    )
    if isinstance(raw_explicit_correction, bool):
        normalized_explicit_correction = raw_explicit_correction
    else:
        normalized_explicit_correction = _coerce_bool_with_ja(raw_explicit_correction, False)
    return {
        "value": normalized_value,
        "evidence_quote": normalized_evidence,
        "evidence_turn_ids": normalized_turn_ids,
        "evidence_turn_ids_provided": raw_evidence_turn_ids_provided or raw_user_evidence_turn_ids_provided,
        "user_quote": normalized_user_quote,
        "user_evidence_turn_ids": normalized_user_turn_ids,
        "user_evidence_turn_ids_provided": raw_user_evidence_turn_ids_provided or raw_evidence_turn_ids_provided,
        "assistant_quote": normalized_assistant_quote,
        "assistant_evidence_turn_ids": normalized_assistant_turn_ids,
        "assistant_evidence_turn_ids_provided": raw_assistant_evidence_turn_ids_provided,
        "confidence": normalized_confidence,
        "assent_bridge": normalized_assent_bridge,
        "assent_bridge_source_turn_id": normalized_assent_bridge_source_turn_id,
        "explicit_correction": normalized_explicit_correction,
        "is_structured": is_structured,
    }


def _extract_phase_slot_updates(payload: Mapping[str, Any]) -> Dict[str, Any]:
    phase_slots = payload.get("phase_slots")
    if not isinstance(phase_slots, dict):
        return {}

    phase = _parse_phase_from_any(
        phase_slots.get("phase")
        or phase_slots.get("phase_code")
        or phase_slots.get("phase_label")
        or payload.get("phase")
        or payload.get("predicted_phase")
    )

    raw_slots = phase_slots.get("slots")
    if not isinstance(raw_slots, dict):
        raw_slots = {
            k: v
            for k, v in phase_slots.items()
            if k not in {"phase", "phase_code", "phase_label", "slots"}
        }

    if not raw_slots:
        return {}

    allowed_keys = set(_PHASE_SLOT_LABELS.keys())
    if phase is not None:
        allowed_keys = set(_PHASE_SLOT_SCHEMA.get(phase, ()))

    slots: Dict[str, Dict[str, Any]] = {}
    for key, value in raw_slots.items():
        k = str(key).strip()
        if k not in allowed_keys:
            continue
        candidate = _normalize_slot_candidate(value)
        slot_value = _normalize_slot_text(candidate.get("value"))
        if slot_value:
            slots[k] = {
                "value": slot_value,
                "evidence_quote": _normalize_slot_text(candidate.get("evidence_quote")),
                "evidence_turn_ids": _normalize_turn_id_list(candidate.get("evidence_turn_ids"), min_turn=1),
                "evidence_turn_ids_provided": bool(candidate.get("evidence_turn_ids_provided")),
                "user_quote": _normalize_slot_text(candidate.get("user_quote")),
                "user_evidence_turn_ids": _normalize_turn_id_list(candidate.get("user_evidence_turn_ids"), min_turn=1),
                "user_evidence_turn_ids_provided": bool(candidate.get("user_evidence_turn_ids_provided")),
                "assistant_quote": _normalize_slot_text(candidate.get("assistant_quote")),
                "assistant_evidence_turn_ids": _normalize_turn_id_list(candidate.get("assistant_evidence_turn_ids"), min_turn=1),
                "assistant_evidence_turn_ids_provided": bool(candidate.get("assistant_evidence_turn_ids_provided")),
                "confidence": _clamp01(candidate.get("confidence"), 0.0),
                "assent_bridge": bool(candidate.get("assent_bridge")),
                "assent_bridge_source_turn_id": (
                    _normalize_turn_id_list(candidate.get("assent_bridge_source_turn_id"), min_turn=1)[:1] or [0]
                )[0],
                "explicit_correction": bool(candidate.get("explicit_correction")),
            }

    if not slots:
        return {}

    return {
        "phase": phase.value if phase else None,
        "phase_code": phase.name if phase else None,
        "slots": slots,
    }


def _copy_phase_slot_memory(memory: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    copied = _init_phase_slot_memory()
    for phase in Phase:
        phase_key = phase.name
        existing = memory.get(phase_key)
        if not isinstance(existing, dict):
            continue
        for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
            copied[phase_key][slot_key] = _normalize_slot_text(existing.get(slot_key))
    return copied


def _copy_phase_slot_meta(meta: Mapping[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    copied = _init_phase_slot_meta()
    for phase in Phase:
        phase_key = phase.name
        existing_phase_meta = meta.get(phase_key)
        if not isinstance(existing_phase_meta, Mapping):
            continue
        for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
            existing = existing_phase_meta.get(slot_key)
            if not isinstance(existing, Mapping):
                continue
            status = _normalize_slot_text(existing.get("status")) or "inactive"
            if status not in {"confirmed", "tentative", "needs_confirmation", "inactive"}:
                status = "inactive"
            copied[phase_key][slot_key] = {
                "status": status,
                "evidence_quote": _normalize_slot_text(existing.get("evidence_quote")),
                "evidence_turn_ids": _normalize_turn_id_list(existing.get("evidence_turn_ids"), min_turn=1),
                "user_quote": _normalize_slot_text(existing.get("user_quote")),
                "user_evidence_turn_ids": _normalize_turn_id_list(existing.get("user_evidence_turn_ids"), min_turn=1),
                "assistant_quote": _normalize_slot_text(existing.get("assistant_quote")),
                "assistant_evidence_turn_ids": _normalize_turn_id_list(existing.get("assistant_evidence_turn_ids"), min_turn=1),
                "extraction_confidence": _clamp01(existing.get("extraction_confidence"), 0.0),
                "quality_score": _clamp01(existing.get("quality_score"), 0.0),
                "issue_codes": [
                    _normalize_slot_text(item)
                    for item in (existing.get("issue_codes") or [])
                    if _normalize_slot_text(item)
                ][:6]
                if isinstance(existing.get("issue_codes"), Sequence) and not isinstance(existing.get("issue_codes"), (str, bytes))
                else [],
                "review_note": _normalize_slot_text(existing.get("review_note")),
                "review_source": _normalize_slot_text(existing.get("review_source")),
                "promote_on_phase_entry": bool(existing.get("promote_on_phase_entry")),
                "reviewed_at_turn": max(0, int(existing.get("reviewed_at_turn") or 0)),
                "review_origin_phase": _normalize_slot_text(existing.get("review_origin_phase")),
                "review_decision_raw": _normalize_slot_text(existing.get("review_decision_raw")).lower(),
            }
    return copied


def _build_all_phase_slot_status_for_prompt(
    state: DialogueState,
    *,
    current_phase: Optional[Phase] = None,
    current_phase_slot_keys: Optional[Sequence[str]] = None,
) -> str:
    memory = _copy_phase_slot_memory(state.phase_slots)
    filtered_current_slot_keys: Optional[Set[str]] = None
    if current_phase is not None and current_phase_slot_keys is not None:
        filtered_current_slot_keys = {
            _normalize_slot_text(slot_key)
            for slot_key in current_phase_slot_keys
            if _normalize_slot_text(slot_key)
        }
    lines: List[str] = []
    for phase in Phase:
        expected_items: List[str] = []
        current_items: List[str] = []
        missing_items: List[str] = []
        phase_bucket = memory.get(phase.name, {})
        slot_keys: Sequence[str] = _PHASE_SLOT_SCHEMA.get(phase, ())
        if (
            filtered_current_slot_keys is not None
            and current_phase is not None
            and phase == current_phase
        ):
            slot_keys = [
                slot_key
                for slot_key in slot_keys
                if _normalize_slot_text(slot_key) in filtered_current_slot_keys
            ]
        for slot_key in slot_keys:
            label = _PHASE_SLOT_LABELS.get(slot_key, slot_key)
            expected_items.append(f"{slot_key}({label})")
            current_value = _normalize_slot_text(phase_bucket.get(slot_key))
            if current_value:
                current_items.append(f"{slot_key}({label})={current_value}")
            else:
                missing_items.append(f"{slot_key}({label})")

        lines.append(f"- {phase.value} [{phase.name}]")
        lines.append(f"  Required slots: {' / '.join(expected_items) if expected_items else 'none'}")
        lines.append(f"  Current values: {' / '.join(current_items) if current_items else 'none'}")
        lines.append(f"  Missing: {' / '.join(missing_items) if missing_items else 'none'}")
    return "\n".join(lines)


def _build_current_phase_slot_context_for_prompt(
    state: DialogueState,
    *,
    phase: Phase,
    slot_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, str]]:
    committed_memory = _copy_phase_slot_memory(state.phase_slots).get(phase.name, {})
    pending_memory = _copy_phase_slot_memory(state.phase_slot_pending).get(phase.name, {})
    target_slot_keys = (
        [
            _normalize_slot_text(slot_key)
            for slot_key in slot_keys
            if _normalize_slot_text(slot_key) in _PHASE_SLOT_SCHEMA.get(phase, ())
        ]
        if slot_keys is not None
        else list(_PHASE_SLOT_SCHEMA.get(phase, ()))
    )
    committed: Dict[str, str] = {}
    pending: Dict[str, str] = {}
    for slot_key in target_slot_keys:
        committed[slot_key] = _normalize_slot_text(committed_memory.get(slot_key))
        pending[slot_key] = _normalize_slot_text(pending_memory.get(slot_key))
    return {
        "committed": committed,
        "pending": pending,
    }


def _extract_phase_slot_update_list(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []

    parsed_updates: List[Dict[str, Any]] = []
    single = _extract_phase_slot_updates(payload)
    if single:
        parsed_updates.append(single)
    elif any(key in payload for key in ("phase", "phase_code", "phase_label", "slots")):
        direct_single = _extract_phase_slot_updates({"phase_slots": dict(payload)})
        if direct_single:
            parsed_updates.append(direct_single)

    for list_key in ("phase_slot_updates", "slot_updates", "updates"):
        raw_block = payload.get(list_key)
        if isinstance(raw_block, Mapping):
            # Format A: treat {"phase":"...","slots":{...}} as one item
            if any(key in raw_block for key in ("phase", "phase_code", "phase_label", "slots")):
                parsed_single = _extract_phase_slot_updates({"phase_slots": dict(raw_block)})
                if parsed_single:
                    parsed_updates.append(parsed_single)
            else:
                # Format B: {"PHASE_CODE":{"slot_key":{value,evidence_quote,confidence}}}
                for phase_hint, slots_hint in raw_block.items():
                    if not isinstance(slots_hint, Mapping):
                        continue
                    parsed_map = _extract_phase_slot_updates(
                        {"phase_slots": {"phase": phase_hint, "slots": dict(slots_hint)}}
                    )
                    if parsed_map:
                        parsed_updates.append(parsed_map)
            continue

        if not isinstance(raw_block, list):
            continue
        for item in raw_block:
            if not isinstance(item, Mapping):
                continue
            if isinstance(item.get("phase_slots"), Mapping):
                parsed = _extract_phase_slot_updates(item)
            else:
                parsed = _extract_phase_slot_updates({"phase_slots": dict(item)})
            if parsed:
                parsed_updates.append(parsed)

    merged_by_phase: Dict[str, Dict[str, Any]] = {}
    phase_order: List[str] = []
    for update in parsed_updates:
        phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
        if phase is None:
            continue
        phase_key = phase.name
        if phase_key not in merged_by_phase:
            merged_by_phase[phase_key] = {
                "phase": phase.value,
                "phase_code": phase.name,
                "slots": {},
            }
            phase_order.append(phase_key)

        slots = update.get("slots")
        if not isinstance(slots, Mapping):
            continue
        target_slots = merged_by_phase[phase_key]["slots"]
        for slot_key, slot_value in slots.items():
            key = str(slot_key).strip()
            if key not in _PHASE_SLOT_SCHEMA.get(phase, ()):
                continue
            candidate = _normalize_slot_candidate(slot_value)
            value = _normalize_slot_text(candidate.get("value"))
            if value:
                target_slots[key] = {
                    "value": value,
                    "evidence_quote": _normalize_slot_text(candidate.get("evidence_quote")),
                    "evidence_turn_ids": _normalize_turn_id_list(candidate.get("evidence_turn_ids"), min_turn=1),
                    "evidence_turn_ids_provided": bool(candidate.get("evidence_turn_ids_provided")),
                    "user_quote": _normalize_slot_text(candidate.get("user_quote")),
                    "user_evidence_turn_ids": _normalize_turn_id_list(candidate.get("user_evidence_turn_ids"), min_turn=1),
                    "user_evidence_turn_ids_provided": bool(candidate.get("user_evidence_turn_ids_provided")),
                    "assistant_quote": _normalize_slot_text(candidate.get("assistant_quote")),
                    "assistant_evidence_turn_ids": _normalize_turn_id_list(candidate.get("assistant_evidence_turn_ids"), min_turn=1),
                    "assistant_evidence_turn_ids_provided": bool(candidate.get("assistant_evidence_turn_ids_provided")),
                    "confidence": _clamp01(candidate.get("confidence"), 0.0),
                    "assent_bridge": bool(candidate.get("assent_bridge")),
                    "assent_bridge_source_turn_id": (
                        _normalize_turn_id_list(candidate.get("assent_bridge_source_turn_id"), min_turn=1)[:1] or [0]
                    )[0],
                    "explicit_correction": bool(candidate.get("explicit_correction")),
                }

    updates: List[Dict[str, Any]] = []
    for phase_key in phase_order:
        candidate = merged_by_phase.get(phase_key, {})
        slots = candidate.get("slots")
        if isinstance(slots, Mapping) and slots:
            updates.append(candidate)
    return updates


def _extract_phase_slot_update_list_from_any(raw_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_payload, Mapping):
        payload = dict(raw_payload)
        parsed = _extract_phase_slot_update_list(payload)
        if parsed:
            return parsed
        if any(key in payload for key in ("phase", "phase_code", "phase_label", "slots")):
            return _extract_phase_slot_update_list({"phase_slot_updates": [payload]})
        return []
    if isinstance(raw_payload, list):
        return _extract_phase_slot_update_list({"phase_slot_updates": raw_payload})
    return []


def _merge_phase_slot_updates(*update_groups: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged_by_phase: Dict[str, Dict[str, Any]] = {}
    phase_order: List[str] = []
    for updates in update_groups:
        for update in updates:
            if not isinstance(update, Mapping):
                continue
            phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
            if phase is None:
                continue
            phase_key = phase.name
            if phase_key not in merged_by_phase:
                merged_by_phase[phase_key] = {
                    "phase": phase.value,
                    "phase_code": phase.name,
                    "slots": {},
                }
                phase_order.append(phase_key)

            slots = update.get("slots")
            if not isinstance(slots, Mapping):
                continue

            bucket = merged_by_phase[phase_key]["slots"]
            for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
                if slot_key not in slots:
                    continue
                candidate = _normalize_slot_candidate(slots.get(slot_key))
                value = _normalize_slot_text(candidate.get("value"))
                if not value:
                    continue
                bucket[slot_key] = {
                    "value": value,
                    "evidence_quote": _normalize_slot_text(candidate.get("evidence_quote")),
                    "evidence_turn_ids": _normalize_turn_id_list(candidate.get("evidence_turn_ids"), min_turn=1),
                    "evidence_turn_ids_provided": bool(candidate.get("evidence_turn_ids_provided")),
                    "user_quote": _normalize_slot_text(candidate.get("user_quote")),
                    "user_evidence_turn_ids": _normalize_turn_id_list(candidate.get("user_evidence_turn_ids"), min_turn=1),
                    "user_evidence_turn_ids_provided": bool(candidate.get("user_evidence_turn_ids_provided")),
                    "assistant_quote": _normalize_slot_text(candidate.get("assistant_quote")),
                    "assistant_evidence_turn_ids": _normalize_turn_id_list(candidate.get("assistant_evidence_turn_ids"), min_turn=1),
                    "assistant_evidence_turn_ids_provided": bool(candidate.get("assistant_evidence_turn_ids_provided")),
                    "confidence": _clamp01(candidate.get("confidence"), 0.0),
                    "assent_bridge": bool(candidate.get("assent_bridge")),
                    "assent_bridge_source_turn_id": (
                        _normalize_turn_id_list(candidate.get("assent_bridge_source_turn_id"), min_turn=1)[:1] or [0]
                    )[0],
                    "explicit_correction": bool(candidate.get("explicit_correction")),
                }

    merged: List[Dict[str, Any]] = []
    for phase_key in phase_order:
        item = merged_by_phase.get(phase_key, {})
        slots = item.get("slots")
        if isinstance(slots, Mapping) and slots:
            merged.append(item)
    return merged


def _history_turn_index_map(history: List[Tuple[str, str]]) -> Dict[int, Dict[str, str]]:
    turn_map: Dict[int, Dict[str, str]] = {}
    for idx, (role, text) in enumerate(history, start=1):
        turn_map[idx] = {
            "role": str(role or ""),
            "text": str(text or ""),
        }
    return turn_map


def _normalize_match_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _validate_phase_slot_evidence_against_history(
    *,
    updates: Sequence[Mapping[str, Any]],
    history: List[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    turn_map = _history_turn_index_map(history)
    max_turn = len(history)
    validated_updates: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []

    for update in updates:
        if not isinstance(update, Mapping):
            continue
        phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
        if phase is None:
            continue
        raw_slots = update.get("slots")
        if not isinstance(raw_slots, Mapping):
            continue
        validated_slots: Dict[str, Dict[str, Any]] = {}
        for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
            if slot_key not in raw_slots:
                continue
            candidate = _normalize_slot_candidate(raw_slots.get(slot_key))
            candidate_value = _normalize_slot_text(candidate.get("value"))
            if not candidate_value:
                continue
            user_quote = _normalize_slot_text(candidate.get("user_quote")) or _normalize_slot_text(candidate.get("evidence_quote"))
            user_evidence_turn_ids = _normalize_turn_id_list(
                candidate.get("user_evidence_turn_ids") or candidate.get("evidence_turn_ids"),
                min_turn=1,
                max_turn=max_turn if max_turn > 0 else None,
            )
            user_evidence_turn_ids_provided = bool(candidate.get("user_evidence_turn_ids_provided")) or bool(
                candidate.get("evidence_turn_ids_provided")
            )
            assistant_quote = _normalize_slot_text(candidate.get("assistant_quote"))
            assistant_evidence_turn_ids = _normalize_turn_id_list(
                candidate.get("assistant_evidence_turn_ids"),
                min_turn=1,
                max_turn=max_turn if max_turn > 0 else None,
            )
            assistant_evidence_turn_ids_provided = bool(candidate.get("assistant_evidence_turn_ids_provided"))
            candidate_value_normalized = _normalize_slot_text(candidate.get("value"))
            raw_assent_bridge = candidate.get("assent_bridge")
            assent_bridge_requested = False
            if isinstance(raw_assent_bridge, bool):
                assent_bridge_requested = raw_assent_bridge
            elif raw_assent_bridge not in (None, ""):
                assent_bridge_requested = _coerce_bool_with_ja(raw_assent_bridge, False)
            assent_bridge_source_turn_ids = _normalize_turn_id_list(
                candidate.get("assent_bridge_source_turn_id"),
                min_turn=1,
                max_turn=max_turn if max_turn > 0 else None,
            )
            assent_bridge_source_turn_id = assent_bridge_source_turn_ids[0] if assent_bridge_source_turn_ids else 0
            assent_bridge_source_turn = turn_map.get(assent_bridge_source_turn_id) if assent_bridge_source_turn_id > 0 else {}
            assent_bridge_source_role = _normalize_slot_text((assent_bridge_source_turn or {}).get("role")).lower()

            if (
                assent_bridge_requested
                and assent_bridge_source_turn_id > 0
                and assent_bridge_source_role == "assistant"
                and assent_bridge_source_turn_id not in assistant_evidence_turn_ids
            ):
                assistant_evidence_turn_ids.append(assent_bridge_source_turn_id)

            user_target_turn_ids = (
                list(user_evidence_turn_ids)
                if user_evidence_turn_ids
                else [turn_id for turn_id, info in turn_map.items() if info.get("role") == "user"]
            )
            if not user_target_turn_ids and turn_map:
                user_target_turn_ids = [turn_id for turn_id, info in turn_map.items() if info.get("role") == "user"]
            assistant_target_turn_ids = (
                list(assistant_evidence_turn_ids)
                if assistant_evidence_turn_ids
                else (
                    [assent_bridge_source_turn_id]
                    if assent_bridge_source_turn_id > 0 and assent_bridge_source_role == "assistant"
                    else []
                )
            )

            structural_wrong_turn = False
            if user_evidence_turn_ids_provided and not user_evidence_turn_ids:
                structural_wrong_turn = True
            if not structural_wrong_turn and user_evidence_turn_ids:
                for turn_id in user_evidence_turn_ids:
                    role = _normalize_slot_text((turn_map.get(turn_id) or {}).get("role")).lower()
                    if role == "user":
                        continue
                    structural_wrong_turn = True
                    break
            if not structural_wrong_turn and assistant_evidence_turn_ids and not assent_bridge_requested:
                structural_wrong_turn = True
            if (
                not structural_wrong_turn
                and assistant_evidence_turn_ids_provided
                and not assistant_evidence_turn_ids
                and assent_bridge_requested
            ):
                structural_wrong_turn = True
            if not structural_wrong_turn and assistant_evidence_turn_ids:
                for turn_id in assistant_evidence_turn_ids:
                    role = _normalize_slot_text((turn_map.get(turn_id) or {}).get("role")).lower()
                    if role == "assistant":
                        continue
                    structural_wrong_turn = True
                    break

            user_matched_turn_ids: List[int] = []
            user_quote_found = False
            user_partial_match = False
            canonical_user_quote = user_quote
            assistant_matched_turn_ids: List[int] = []
            assistant_quote_found = False
            assistant_partial_match = False
            canonical_assistant_quote = assistant_quote
            assent_bridge_used = False
            semantic_review_pending = False
            bridge_probe_source = ""
            if user_quote and user_target_turn_ids:
                normalized_user_quote = _normalize_match_text(user_quote)
                for turn_id in user_target_turn_ids:
                    turn_info = turn_map.get(turn_id)
                    if not isinstance(turn_info, Mapping):
                        continue
                    turn_text = str(turn_info.get("text") or "")
                    normalized_turn_text = _normalize_match_text(turn_text)
                    if normalized_user_quote and normalized_user_quote in normalized_turn_text:
                        user_quote_found = True
                        user_matched_turn_ids.append(turn_id)
                        continue
                    if normalized_user_quote and normalized_turn_text and (
                        normalized_turn_text in normalized_user_quote or normalized_user_quote in normalized_turn_text
                    ):
                        user_quote_found = True
                        user_partial_match = True
                        user_matched_turn_ids.append(turn_id)
                if user_matched_turn_ids:
                    first_turn_text = str((turn_map.get(user_matched_turn_ids[0]) or {}).get("text") or "")
                    if first_turn_text and user_quote and user_quote not in first_turn_text:
                        canonical_user_quote = user_quote

            if assistant_quote and assistant_target_turn_ids:
                normalized_assistant_quote = _normalize_match_text(assistant_quote)
                for turn_id in assistant_target_turn_ids:
                    turn_info = turn_map.get(turn_id)
                    if not isinstance(turn_info, Mapping):
                        continue
                    turn_text = str(turn_info.get("text") or "")
                    normalized_turn_text = _normalize_match_text(turn_text)
                    if normalized_assistant_quote and normalized_assistant_quote in normalized_turn_text:
                        assistant_quote_found = True
                        assistant_matched_turn_ids.append(turn_id)
                        continue
                    if normalized_assistant_quote and normalized_turn_text and (
                        normalized_turn_text in normalized_assistant_quote
                        or normalized_assistant_quote in normalized_turn_text
                    ):
                        assistant_quote_found = True
                        assistant_partial_match = True
                        assistant_matched_turn_ids.append(turn_id)
                if assistant_matched_turn_ids:
                    first_turn_text = str((turn_map.get(assistant_matched_turn_ids[0]) or {}).get("text") or "")
                    if first_turn_text and assistant_quote and assistant_quote not in first_turn_text:
                        canonical_assistant_quote = assistant_quote

            assistant_quote_too_short = len(_normalize_match_text(assistant_quote)) <= 2 if assistant_quote else True
            if (
                assent_bridge_requested
                and assent_bridge_source_turn_id > 0
                and assent_bridge_source_role == "assistant"
                and (not assistant_quote_found or assistant_quote_too_short)
            ):
                bridge_turn_info = assent_bridge_source_turn or {}
                bridge_turn_text = str(bridge_turn_info.get("text") or "")
                normalized_bridge_turn_text = _normalize_match_text(bridge_turn_text)

                bridge_candidates: List[Tuple[str, str]] = []
                if assistant_quote:
                    bridge_candidates.append(("assistant_quote", assistant_quote))
                if candidate_value_normalized and candidate_value_normalized != assistant_quote:
                    bridge_candidates.append(("candidate_value", candidate_value_normalized))
                if user_quote and user_quote != assistant_quote:
                    bridge_candidates.append(("user_quote", user_quote))

                for source, probe_text in bridge_candidates:
                    normalized_probe = _normalize_match_text(probe_text)
                    if len(normalized_probe) <= 2:
                        continue
                    if not normalized_bridge_turn_text:
                        continue
                    direct_match = (
                        normalized_probe in normalized_bridge_turn_text
                        or normalized_bridge_turn_text in normalized_probe
                    )
                    fuzzy_match = False
                    if not direct_match:
                        similarity = _jaccard(_char_bigrams(probe_text), _char_bigrams(bridge_turn_text))
                        fuzzy_match = similarity >= 0.45
                    if direct_match or fuzzy_match:
                        assistant_quote_found = True
                        assistant_partial_match = (not direct_match) or (normalized_probe not in normalized_bridge_turn_text)
                        assistant_matched_turn_ids = [assent_bridge_source_turn_id]
                        canonical_assistant_quote = probe_text
                        assent_bridge_used = True
                        bridge_probe_source = source
                        break

            effective_user_quote = canonical_user_quote if user_quote_found and canonical_user_quote else user_quote
            effective_user_turn_ids = (
                _normalize_turn_id_list(
                    [*list(user_evidence_turn_ids), *list(user_matched_turn_ids)],
                    min_turn=1,
                    max_turn=max_turn if max_turn > 0 else None,
                )
                if user_quote_found and user_matched_turn_ids
                else list(user_evidence_turn_ids)
            )
            effective_assistant_quote = (
                canonical_assistant_quote if assistant_quote_found and canonical_assistant_quote else assistant_quote
            )
            effective_assistant_turn_ids = (
                _normalize_turn_id_list(
                    [*list(assistant_evidence_turn_ids), *list(assistant_matched_turn_ids)],
                    min_turn=1,
                    max_turn=max_turn if max_turn > 0 else None,
                )
                if assistant_quote_found and assistant_matched_turn_ids
                else list(assistant_evidence_turn_ids)
            )
            if (
                assent_bridge_requested
                and assent_bridge_source_turn_id > 0
                and assent_bridge_source_role == "assistant"
                and assent_bridge_source_turn_id not in effective_assistant_turn_ids
            ):
                effective_assistant_turn_ids.append(assent_bridge_source_turn_id)
            effective_evidence_quote = effective_user_quote or effective_assistant_quote
            effective_evidence_turn_ids = list(effective_user_turn_ids)
            has_explicit_numeric_scale_evidence = _has_explicit_numeric_scale_evidence(
                slot_key=slot_key,
                user_quote=effective_user_quote,
                user_turn_ids=effective_user_turn_ids,
                assistant_quote=effective_assistant_quote,
                assistant_turn_ids=effective_assistant_turn_ids,
                turn_map=turn_map,
            )
            if (
                not structural_wrong_turn
                and (
                    (effective_user_quote and (not user_quote_found or user_partial_match))
                    or (
                        effective_assistant_quote
                        and assent_bridge_requested
                        and (not assistant_quote_found or (assistant_partial_match and not assent_bridge_used))
                    )
                )
            ):
                semantic_review_pending = True

            validation_issue = ""
            if not effective_user_quote and not effective_assistant_quote:
                validation_issue = "missing_evidence"
            elif structural_wrong_turn:
                validation_issue = "wrong_turn"
            elif (
                len(_normalize_match_text(effective_user_quote)) <= 2
                and len(_normalize_match_text(effective_assistant_quote)) <= 2
            ):
                validation_issue = "too_short"
            else:
                if effective_assistant_turn_ids and not assent_bridge_requested:
                    validation_issue = "non_user_source"
                elif slot_key in _SLOT_NUMERIC_REQUIRED_KEYS and not has_explicit_numeric_scale_evidence:
                    validation_issue = "scale_missing_number"

            if validation_issue == "":
                slot_format_issue = _validate_slot_value_format(
                    slot_key=slot_key,
                    value=candidate_value,
                )
                if slot_format_issue:
                    validation_issue = slot_format_issue

            validated_slots[slot_key] = {
                "value": candidate_value,
                "evidence_quote": effective_evidence_quote,
                "evidence_turn_ids": effective_evidence_turn_ids,
                "user_quote": effective_user_quote,
                "user_evidence_turn_ids": effective_user_turn_ids,
                "assistant_quote": effective_assistant_quote,
                "assistant_evidence_turn_ids": effective_assistant_turn_ids,
                "confidence": _clamp01(candidate.get("confidence"), 0.0),
                "evidence_valid": validation_issue == "",
                "matched_turn_ids": _normalize_turn_id_list(
                    [*user_matched_turn_ids, *assistant_matched_turn_ids],
                    min_turn=1,
                    max_turn=max_turn if max_turn > 0 else None,
                ),
                "matched_user_turn_ids": _normalize_turn_id_list(user_matched_turn_ids, min_turn=1, max_turn=max_turn if max_turn > 0 else None),
                "matched_assistant_turn_ids": _normalize_turn_id_list(
                    assistant_matched_turn_ids,
                    min_turn=1,
                    max_turn=max_turn if max_turn > 0 else None,
                ),
                "canonical_evidence_quote": effective_evidence_quote,
                "canonical_user_quote": canonical_user_quote if user_quote_found else "",
                "canonical_assistant_quote": canonical_assistant_quote if assistant_quote_found else "",
                "validation_issue": validation_issue,
                "quote_found": bool(user_quote_found or assistant_quote_found),
                "assent_bridge_used": assent_bridge_used,
                "assent_bridge_source_turn_id": assent_bridge_source_turn_id if assent_bridge_used else 0,
                "assent_bridge_probe_source": bridge_probe_source if assent_bridge_used else "",
                "assent_bridge_requested": assent_bridge_requested,
                "semantic_review_pending": semantic_review_pending,
                "explicit_correction": bool(candidate.get("explicit_correction")),
            }
            validations.append(
                {
                    "phase": phase.value,
                    "phase_code": phase.name,
                    "slot_key": slot_key,
                    "quote_found": bool(user_quote_found or assistant_quote_found),
                    "matched_turn_ids": _normalize_turn_id_list(
                        [*user_matched_turn_ids, *assistant_matched_turn_ids],
                        min_turn=1,
                        max_turn=max_turn if max_turn > 0 else None,
                    ),
                    "matched_user_turn_ids": _normalize_turn_id_list(
                        user_matched_turn_ids,
                        min_turn=1,
                        max_turn=max_turn if max_turn > 0 else None,
                    ),
                    "matched_assistant_turn_ids": _normalize_turn_id_list(
                        assistant_matched_turn_ids,
                        min_turn=1,
                        max_turn=max_turn if max_turn > 0 else None,
                    ),
                    "canonical_evidence_quote": effective_evidence_quote,
                    "canonical_user_quote": canonical_user_quote if user_quote_found else "",
                    "canonical_assistant_quote": canonical_assistant_quote if assistant_quote_found else "",
                    "validation_issue": validation_issue,
                    "assent_bridge_used": assent_bridge_used,
                    "assent_bridge_source_turn_id": assent_bridge_source_turn_id if assent_bridge_used else 0,
                    "assent_bridge_probe_source": bridge_probe_source if assent_bridge_used else "",
                    "assent_bridge_requested": assent_bridge_requested,
                    "semantic_review_pending": semantic_review_pending,
                }
            )

        if validated_slots:
            validated_updates.append(
                {
                    "phase": phase.value,
                    "phase_code": phase.name,
                    "slots": validated_slots,
                }
            )

    return validated_updates, validations


def _flatten_layer1_candidates(
    *,
    updates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for update in updates:
        if not isinstance(update, Mapping):
            continue
        phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
        if phase is None:
            continue
        slots = update.get("slots")
        if not isinstance(slots, Mapping):
            continue
        for slot_key, candidate in slots.items():
            if not isinstance(candidate, Mapping):
                continue
            normalized_slot_key = str(slot_key or "").strip()
            if normalized_slot_key not in _PHASE_SLOT_SCHEMA.get(phase, ()):
                continue
            flattened.append(
                {
                    "phase": phase.value,
                    "phase_code": phase.name,
                    "slot_key": normalized_slot_key,
                    "candidate_value": _normalize_slot_text(candidate.get("value")),
                    "evidence_quote": _normalize_slot_text(candidate.get("evidence_quote")),
                    "evidence_turn_ids": _normalize_turn_id_list(candidate.get("evidence_turn_ids"), min_turn=1),
                    "user_quote": _normalize_slot_text(candidate.get("user_quote")),
                    "user_evidence_turn_ids": _normalize_turn_id_list(candidate.get("user_evidence_turn_ids"), min_turn=1),
                    "assistant_quote": _normalize_slot_text(candidate.get("assistant_quote")),
                    "assistant_evidence_turn_ids": _normalize_turn_id_list(
                        candidate.get("assistant_evidence_turn_ids"),
                        min_turn=1,
                    ),
                    "extraction_confidence": _clamp01(candidate.get("confidence"), 0.0),
                    "evidence_valid": bool(candidate.get("evidence_valid")),
                    "matched_turn_ids": _normalize_turn_id_list(candidate.get("matched_turn_ids"), min_turn=1),
                    "canonical_evidence_quote": _normalize_slot_text(candidate.get("canonical_evidence_quote")),
                    "validation_issue": _normalize_slot_text(candidate.get("validation_issue")),
                    "semantic_review_pending": bool(candidate.get("semantic_review_pending")),
                    "explicit_correction": bool(candidate.get("explicit_correction")),
                }
            )
    return flattened


def _precheck_slot_quality_for_layer2_target(
    *,
    slot_key: str,
    slot_value: str,
    slot_meta: Any,
) -> float:
    meta_quality = (
        _clamp01(slot_meta.get("quality_score"), 0.0)
        if isinstance(slot_meta, Mapping)
        else 0.0
    )
    # Even on the first turn, if quality is unset (0.0), use the minimum fallback quality when a value exists.
    if meta_quality <= 0.0 and slot_value:
        return _estimate_slot_text_quality(slot_key, slot_value)
    return meta_quality


def _build_layer2_review_candidate_from_state(
    *,
    phase: Phase,
    slot_key: str,
    slot_value: str,
    slot_meta: Any,
    quality_precheck: float,
    quality_target_threshold: float,
) -> Dict[str, Any]:
    slot_meta_map = slot_meta if isinstance(slot_meta, Mapping) else {}
    user_quote = _normalize_slot_text(slot_meta_map.get("user_quote"))
    user_evidence_turn_ids = _normalize_turn_id_list(slot_meta_map.get("user_evidence_turn_ids"), min_turn=1)
    assistant_quote = _normalize_slot_text(slot_meta_map.get("assistant_quote"))
    assistant_evidence_turn_ids = _normalize_turn_id_list(slot_meta_map.get("assistant_evidence_turn_ids"), min_turn=1)
    evidence_quote = user_quote or assistant_quote or _normalize_slot_text(slot_meta_map.get("evidence_quote"))
    evidence_turn_ids = (
        list(user_evidence_turn_ids)
        if user_evidence_turn_ids
        else _normalize_turn_id_list(slot_meta_map.get("evidence_turn_ids"), min_turn=1)
    )
    evidence_valid = bool(evidence_quote or evidence_turn_ids or assistant_evidence_turn_ids)
    validation_issue = "" if evidence_valid else "missing_evidence"
    return {
        "phase": phase.value,
        "phase_code": phase.name,
        "slot_key": slot_key,
        "candidate_value": slot_value,
        "evidence_quote": evidence_quote,
        "evidence_turn_ids": evidence_turn_ids,
        "user_quote": user_quote,
        "user_evidence_turn_ids": user_evidence_turn_ids,
        "assistant_quote": assistant_quote,
        "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
        "extraction_confidence": _clamp01(slot_meta_map.get("extraction_confidence"), 0.0),
        "evidence_valid": evidence_valid,
        "matched_turn_ids": [],
        "canonical_evidence_quote": evidence_quote,
        "validation_issue": validation_issue,
        "semantic_review_pending": False,
        "explicit_correction": False,
        "review_target_reason": "quality_below_threshold_or_unset",
        "candidate_source": "state_slot_memory",
        "quality_precheck": round(float(_clamp01(quality_precheck, 0.0)), 3),
        "quality_target_threshold": round(float(_clamp01(quality_target_threshold, 0.0)), 3),
    }


def _build_layer2_candidate_variant(
    *,
    source: str,
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_value = _normalize_slot_text(candidate.get("candidate_value"))
    user_quote = _normalize_slot_text(candidate.get("user_quote"))
    user_evidence_turn_ids = _normalize_turn_id_list(candidate.get("user_evidence_turn_ids"), min_turn=1)
    assistant_quote = _normalize_slot_text(candidate.get("assistant_quote"))
    assistant_evidence_turn_ids = _normalize_turn_id_list(candidate.get("assistant_evidence_turn_ids"), min_turn=1)
    evidence_quote = (
        user_quote
        or assistant_quote
        or _normalize_slot_text(candidate.get("evidence_quote"))
        or _normalize_slot_text(candidate.get("canonical_evidence_quote"))
    )
    evidence_turn_ids = _normalize_turn_id_list(
        candidate.get("evidence_turn_ids") or candidate.get("matched_turn_ids"),
        min_turn=1,
    )
    if user_evidence_turn_ids:
        evidence_turn_ids = _normalize_turn_id_list([*user_evidence_turn_ids, *evidence_turn_ids], min_turn=1)
    return {
        "variant_source": _normalize_slot_text(source),
        "candidate_value": candidate_value,
        "evidence_quote": evidence_quote,
        "evidence_turn_ids": evidence_turn_ids,
        "user_quote": user_quote,
        "user_evidence_turn_ids": user_evidence_turn_ids,
        "assistant_quote": assistant_quote,
        "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
        "extraction_confidence": _clamp01(candidate.get("extraction_confidence"), 0.0),
        "evidence_valid": bool(candidate.get("evidence_valid")),
        "validation_issue": _normalize_slot_text(candidate.get("validation_issue")),
        "semantic_review_pending": bool(candidate.get("semantic_review_pending")),
        "explicit_correction": bool(candidate.get("explicit_correction")),
    }


def _merge_layer2_review_candidate_pair(
    *,
    phase: Phase,
    slot_key: str,
    existing_candidate: Mapping[str, Any],
    latest_candidate: Optional[Mapping[str, Any]],
    quality_precheck: float,
    quality_target_threshold: float,
) -> Dict[str, Any]:
    existing_variant = _build_layer2_candidate_variant(
        source="existing_slot_memory",
        candidate=existing_candidate,
    )
    latest_variant = (
        _build_layer2_candidate_variant(
            source="latest_user_update",
            candidate=latest_candidate,
        )
        if isinstance(latest_candidate, Mapping)
        else None
    )

    has_existing_value = bool(existing_variant.get("candidate_value"))
    has_latest_value = bool((latest_variant or {}).get("candidate_value"))
    has_explicit_correction = bool(existing_variant.get("explicit_correction")) or bool(
        (latest_variant or {}).get("explicit_correction")
    )
    pair_state = "existing_and_latest" if has_existing_value and has_latest_value else "latest_only" if has_latest_value else "existing_only"
    selected_variant = latest_variant if has_latest_value and latest_variant is not None else existing_variant
    selected_source_raw = _normalize_slot_text(selected_variant.get("variant_source"))
    selected_source = "state_slot_memory" if selected_source_raw == "existing_slot_memory" else selected_source_raw or "state_slot_memory"

    candidate_variants: List[Dict[str, Any]] = [existing_variant]
    if latest_variant is not None:
        candidate_variants.append(latest_variant)

    merged_user_turn_ids = _normalize_turn_id_list(
        [
            *existing_variant.get("user_evidence_turn_ids", []),
            *(latest_variant.get("user_evidence_turn_ids", []) if latest_variant else []),
        ],
        min_turn=1,
    )
    merged_assistant_turn_ids = _normalize_turn_id_list(
        [
            *existing_variant.get("assistant_evidence_turn_ids", []),
            *(latest_variant.get("assistant_evidence_turn_ids", []) if latest_variant else []),
        ],
        min_turn=1,
    )
    merged_evidence_turn_ids = _normalize_turn_id_list(
        [
            *existing_variant.get("evidence_turn_ids", []),
            *(latest_variant.get("evidence_turn_ids", []) if latest_variant else []),
        ],
        min_turn=1,
    )
    if merged_user_turn_ids:
        merged_evidence_turn_ids = _normalize_turn_id_list([*merged_user_turn_ids, *merged_evidence_turn_ids], min_turn=1)
    merged_user_quote = (
        _normalize_slot_text(selected_variant.get("user_quote"))
        or _normalize_slot_text((latest_variant or {}).get("user_quote"))
        or _normalize_slot_text(existing_variant.get("user_quote"))
    )
    merged_assistant_quote = (
        _normalize_slot_text(selected_variant.get("assistant_quote"))
        or _normalize_slot_text((latest_variant or {}).get("assistant_quote"))
        or _normalize_slot_text(existing_variant.get("assistant_quote"))
    )
    merged_evidence_quote = (
        merged_user_quote
        or merged_assistant_quote
        or _normalize_slot_text(selected_variant.get("evidence_quote"))
        or _normalize_slot_text((latest_variant or {}).get("evidence_quote"))
        or _normalize_slot_text(existing_variant.get("evidence_quote"))
    )
    evidence_valid = bool(any(bool(item.get("evidence_valid")) for item in candidate_variants))
    validation_issue = _normalize_slot_text(selected_variant.get("validation_issue"))
    if not validation_issue and not evidence_valid:
        validation_issue = "missing_evidence"

    return {
        "phase": phase.value,
        "phase_code": phase.name,
        "slot_key": slot_key,
        "candidate_value": _normalize_slot_text(selected_variant.get("candidate_value")),
        "evidence_quote": merged_evidence_quote,
        "evidence_turn_ids": merged_evidence_turn_ids,
        "user_quote": merged_user_quote,
        "user_evidence_turn_ids": merged_user_turn_ids,
        "assistant_quote": merged_assistant_quote,
        "assistant_evidence_turn_ids": merged_assistant_turn_ids,
        "extraction_confidence": _clamp01(selected_variant.get("extraction_confidence"), 0.0),
        "evidence_valid": evidence_valid,
        "matched_turn_ids": [],
        "canonical_evidence_quote": merged_evidence_quote,
        "validation_issue": validation_issue,
        "semantic_review_pending": bool(any(bool(item.get("semantic_review_pending")) for item in candidate_variants)),
        "explicit_correction": has_explicit_correction,
        "review_target_reason": (
            "explicit_user_correction"
            if has_explicit_correction
            else "quality_below_threshold_or_unset"
        ),
        "candidate_source": "merged_existing_and_latest" if has_existing_value and has_latest_value else selected_source,
        "selected_candidate_source": selected_source,
        "candidate_pair_state": pair_state,
        "candidate_variants": candidate_variants,
        "existing_slot_value": _normalize_slot_text(existing_variant.get("candidate_value")),
        "latest_slot_value": _normalize_slot_text((latest_variant or {}).get("candidate_value")),
        "quality_precheck": round(float(_clamp01(quality_precheck, 0.0)), 3),
        "quality_target_threshold": round(float(_clamp01(quality_target_threshold, 0.0)), 3),
    }


def _latest_user_turn_from_history(history: Sequence[Tuple[str, str]]) -> Tuple[int, str]:
    latest_turn_id = 0
    latest_user_text = ""
    for turn_id, turn in enumerate(history, start=1):
        if not isinstance(turn, Sequence) or len(turn) < 2:
            continue
        role = _normalize_slot_text(turn[0]).lower()
        if role != "user":
            continue
        latest_turn_id = turn_id
        latest_user_text = _normalize_slot_text(turn[1])
    return latest_turn_id, latest_user_text


def _build_scale_followup_backfill_candidate(
    *,
    history: Sequence[Tuple[str, str]],
    state: DialogueState,
) -> Optional[Dict[str, Any]]:
    if not _is_scaling_question_phase(state.phase):
        return None
    slot_key = _scaling_slot_key_for_phase(state.phase)
    if slot_key not in _SLOT_NUMERIC_REQUIRED_KEYS:
        return None
    last_action = state.last_actions[-1] if state.last_actions else None
    if last_action != MainAction.SCALING_QUESTION:
        return None
    latest_user_turn_id, latest_user_text = _latest_user_turn_from_history(history)
    if latest_user_turn_id <= 0:
        return None
    scale_score = state.scale_followup_score
    if scale_score is None:
        scale_score = _extract_scale_0_10_value(latest_user_text)
    if scale_score is None:
        return None
    normalized_score = float(max(0.0, min(10.0, scale_score)))
    score_text = _format_scale_score_text(normalized_score)
    if not score_text:
        return None
    evidence_quote = latest_user_text or score_text
    return {
        "phase": state.phase.value,
        "phase_code": state.phase.name,
        "slot_key": slot_key,
        "candidate_value": score_text,
        "evidence_quote": evidence_quote,
        "evidence_turn_ids": [latest_user_turn_id],
        "user_quote": evidence_quote,
        "user_evidence_turn_ids": [latest_user_turn_id],
        "assistant_quote": "",
        "assistant_evidence_turn_ids": [],
        "extraction_confidence": 0.99,
        "evidence_valid": True,
        "matched_turn_ids": [latest_user_turn_id],
        "canonical_evidence_quote": evidence_quote,
        "validation_issue": "",
        "semantic_review_pending": False,
        "explicit_correction": False,
        "candidate_source": "scale_followup_score_backfill",
    }


def _build_layer1_slot_bundle(
    *,
    history: List[Tuple[str, str]],
    state: DialogueState,
    slot_updates: Sequence[Mapping[str, Any]],
    slot_fill_debug: Optional[Mapping[str, Any]],
) -> Tuple[Layer1Bundle, Dict[str, Any]]:
    current_phase = state.phase
    phase_slot_memory = _copy_phase_slot_memory(state.phase_slots)
    phase_slot_meta = _copy_phase_slot_meta(state.phase_slot_meta)
    current_phase_values = phase_slot_memory.get(current_phase.name, {})
    need_previous_phase_updates = False
    need_future_phase_updates = False
    include_non_current_updates_in_layer2_review = False
    layer1_note = ""
    focus_choice_hint = _normalize_focus_choice_hint(None)
    if isinstance(slot_fill_debug, Mapping):
        need_previous_phase_updates = bool(slot_fill_debug.get("need_previous_phase_slot_updates"))
        need_future_phase_updates = bool(slot_fill_debug.get("need_future_phase_slot_updates"))
        include_non_current_updates_in_layer2_review = bool(
            slot_fill_debug.get("include_non_current_updates_in_layer2_review")
        )
        layer1_note = _normalize_slot_text(slot_fill_debug.get("note"))
        focus_choice_hint = _normalize_focus_choice_hint(slot_fill_debug.get("focus_choice_hint"))

    current_idx = _phase_index(current_phase)
    previous_phases = _PHASE_ORDER[:current_idx]
    future_phases = _PHASE_ORDER[current_idx + 1 :]
    review_phase_order: List[Phase] = [current_phase]
    if need_previous_phase_updates:
        review_phase_order.extend(previous_phases)
    if need_future_phase_updates:
        review_phase_order.extend(future_phases)
    update_phase_order: List[Phase] = []
    for update in slot_updates:
        if not isinstance(update, Mapping):
            continue
        phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
        if phase is None or phase in update_phase_order:
            continue
        update_phase_order.append(phase)
    if include_non_current_updates_in_layer2_review:
        for phase in update_phase_order:
            if phase not in review_phase_order:
                review_phase_order.append(phase)
    allowed_phase_codes = {phase.name for phase in review_phase_order}

    filtered_updates: List[Dict[str, Any]] = []
    dropped_updates_non_allowed_phase: List[Dict[str, Any]] = []
    dropped_updates_unknown_phase: List[Dict[str, Any]] = []
    for update in slot_updates:
        if not isinstance(update, Mapping):
            continue
        phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
        if phase is None:
            dropped_updates_unknown_phase.append(dict(update))
            continue
        if phase.name not in allowed_phase_codes:
            dropped_updates_non_allowed_phase.append(
                {
                    "phase": phase.value,
                    "phase_code": phase.name,
                    "reason": "phase_not_allowed_for_current_layer1_policy",
                }
            )
            continue
        filtered_updates.append(dict(update))

    validated_updates, evidence_validation = _validate_phase_slot_evidence_against_history(
        updates=filtered_updates,
        history=history,
    )
    flattened_candidate_updates = _flatten_layer1_candidates(updates=validated_updates)
    required_slot_keys = list(_PHASE_SLOT_SCHEMA.get(current_phase, ()))
    explicit_correction_slot_keys: List[str] = []
    explicit_correction_slot_refs: List[str] = []

    def _phase_slot_quality_precheck(phase: Phase, slot_key: str) -> float:
        phase_values = phase_slot_memory.get(phase.name, {})
        phase_meta_bucket = phase_slot_meta.get(phase.name, {})
        slot_value = _normalize_slot_text(phase_values.get(slot_key))
        slot_meta = phase_meta_bucket.get(slot_key) if isinstance(phase_meta_bucket, Mapping) else {}
        return _precheck_slot_quality_for_layer2_target(
            slot_key=slot_key,
            slot_value=slot_value,
            slot_meta=slot_meta,
        )

    for candidate in flattened_candidate_updates:
        if not isinstance(candidate, Mapping):
            continue
        phase = _parse_phase_from_any(candidate.get("phase_code") or candidate.get("phase"))
        if phase is None or phase.name not in allowed_phase_codes:
            continue
        slot_key = _normalize_slot_text(candidate.get("slot_key"))
        if not slot_key or slot_key not in _PHASE_SLOT_SCHEMA.get(phase, ()):
            continue
        if not bool(candidate.get("explicit_correction")):
            continue
        slot_ref = f"{phase.name}.{slot_key}"
        if slot_ref not in explicit_correction_slot_refs:
            explicit_correction_slot_refs.append(slot_ref)
        if phase == current_phase and slot_key not in explicit_correction_slot_keys:
            explicit_correction_slot_keys.append(slot_key)

    precheck_quality_by_slot: Dict[str, float] = {}
    for slot_key in required_slot_keys:
        quality_precheck = _phase_slot_quality_precheck(current_phase, slot_key)
        precheck_quality_by_slot[slot_key] = round(float(_clamp01(quality_precheck, 0.0)), 3)

    review_target_slot_keys: List[str] = []
    review_target_slot_refs: List[str] = []
    review_target_refs: List[Tuple[Phase, str]] = []
    review_target_quality_threshold = _SLOT_LAYER2_REVIEW_TARGET_QUALITY_THRESHOLD
    latest_candidate_by_ref: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dropped_candidates_outside_review_target: List[Dict[str, Any]] = []
    for candidate in flattened_candidate_updates:
        if not isinstance(candidate, Mapping):
            continue
        phase = _parse_phase_from_any(candidate.get("phase_code") or candidate.get("phase"))
        if phase is None or phase.name not in allowed_phase_codes:
            continue
        phase_code = phase.name
        slot_key = _normalize_slot_text(candidate.get("slot_key"))
        if slot_key not in _PHASE_SLOT_SCHEMA.get(phase, ()):
            continue
        slot_ref_tuple = (phase_code, slot_key)
        quality_precheck = _phase_slot_quality_precheck(phase, slot_key)
        if quality_precheck >= review_target_quality_threshold:
            dropped_candidates_outside_review_target.append(
                {
                    "phase_code": phase_code,
                    "slot_key": slot_key,
                    "reason": "quality_at_or_above_target_threshold",
                }
            )
            continue
        enriched = dict(candidate)
        enriched["candidate_source"] = "latest_user_update"
        enriched["explicit_correction"] = bool(candidate.get("explicit_correction"))
        enriched["review_target_reason"] = (
            "explicit_user_correction"
            if f"{phase_code}.{slot_key}" in explicit_correction_slot_refs
            else "quality_below_threshold_or_unset"
        )
        enriched["quality_precheck"] = round(float(_clamp01(quality_precheck, 0.0)), 3)
        enriched["quality_target_threshold"] = round(float(review_target_quality_threshold), 3)
        latest_candidate_by_ref[slot_ref_tuple] = enriched

    for phase in review_phase_order:
        for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
            slot_ref_tuple = (phase.name, slot_key)
            if slot_ref_tuple not in latest_candidate_by_ref:
                continue
            review_target_refs.append((phase, slot_key))
            if slot_key not in review_target_slot_keys:
                review_target_slot_keys.append(slot_key)
            slot_ref = f"{phase.name}.{slot_key}"
            if slot_ref not in review_target_slot_refs:
                review_target_slot_refs.append(slot_ref)

    review_target_ref_set = {(phase.name, slot_key) for phase, slot_key in review_target_refs}

    scale_followup_backfill_debug: Dict[str, Any] = {"eligible": False, "applied": False}
    scale_backfill_candidate = _build_scale_followup_backfill_candidate(
        history=history,
        state=state,
    )
    if scale_backfill_candidate is not None:
        scale_followup_backfill_debug["eligible"] = True
        backfill_slot_key = _normalize_slot_text(scale_backfill_candidate.get("slot_key"))
        backfill_ref = (current_phase.name, backfill_slot_key)
        existing_candidate = latest_candidate_by_ref.get(backfill_ref, {})
        existing_value = (
            _normalize_slot_text(existing_candidate.get("candidate_value"))
            if isinstance(existing_candidate, Mapping)
            else ""
        )
        should_apply_backfill = (
            current_phase.name in allowed_phase_codes
            and _phase_slot_quality_precheck(current_phase, backfill_slot_key) < review_target_quality_threshold
            and (
                not isinstance(existing_candidate, Mapping)
                or not existing_value
                or _extract_scale_0_10_value(existing_value) is None
            )
        )
        if should_apply_backfill:
            enriched_backfill = dict(scale_backfill_candidate)
            enriched_backfill["review_target_reason"] = "quality_below_threshold_or_unset"
            enriched_backfill["quality_precheck"] = round(
                float(_clamp01(_phase_slot_quality_precheck(current_phase, backfill_slot_key), 0.0)),
                3,
            )
            enriched_backfill["quality_target_threshold"] = round(float(review_target_quality_threshold), 3)
            latest_candidate_by_ref[backfill_ref] = enriched_backfill
            if backfill_ref not in review_target_ref_set:
                review_target_refs.append((current_phase, backfill_slot_key))
                review_target_ref_set.add(backfill_ref)
                if backfill_slot_key not in review_target_slot_keys:
                    review_target_slot_keys.append(backfill_slot_key)
                backfill_ref_text = f"{current_phase.name}.{backfill_slot_key}"
                if backfill_ref_text not in review_target_slot_refs:
                    review_target_slot_refs.append(backfill_ref_text)
            scale_followup_backfill_debug["applied"] = True
            scale_followup_backfill_debug["slot_key"] = backfill_slot_key
            scale_followup_backfill_debug["value"] = _normalize_slot_text(
                enriched_backfill.get("candidate_value")
            )
        else:
            if current_phase.name not in allowed_phase_codes:
                scale_followup_backfill_debug["reason"] = "phase_not_allowed"
            elif _phase_slot_quality_precheck(current_phase, backfill_slot_key) >= review_target_quality_threshold:
                scale_followup_backfill_debug["reason"] = "quality_at_or_above_target_threshold"
            else:
                scale_followup_backfill_debug["reason"] = "latest_candidate_already_numeric"

    candidate_updates: List[Dict[str, Any]] = []
    candidate_pair_state_by_slot: Dict[str, str] = {}
    for phase, slot_key in review_target_refs:
        phase_values = phase_slot_memory.get(phase.name, {})
        phase_meta_bucket = phase_slot_meta.get(phase.name, {})
        current_value = _normalize_slot_text(phase_values.get(slot_key))
        slot_meta = phase_meta_bucket.get(slot_key) if isinstance(phase_meta_bucket, Mapping) else {}
        precheck_quality = _phase_slot_quality_precheck(phase, slot_key)
        existing_candidate = _build_layer2_review_candidate_from_state(
            phase=phase,
            slot_key=slot_key,
            slot_value=current_value,
            slot_meta=slot_meta,
            quality_precheck=precheck_quality,
            quality_target_threshold=review_target_quality_threshold,
        )
        latest_candidate = latest_candidate_by_ref.get((phase.name, slot_key))
        candidate = _merge_layer2_review_candidate_pair(
            phase=phase,
            slot_key=slot_key,
            existing_candidate=existing_candidate,
            latest_candidate=latest_candidate,
            quality_precheck=precheck_quality,
            quality_target_threshold=review_target_quality_threshold,
        )
        slot_ref = f"{phase.name}.{slot_key}"
        candidate_pair_state_by_slot[slot_ref] = (
            _normalize_slot_text(candidate.get("candidate_pair_state")) or "existing_only"
        )
        candidate_updates.append(candidate)

    candidate_slot_keys = [
        str(item.get("slot_key") or "").strip()
        for item in candidate_updates
        if (
            str(item.get("slot_key") or "").strip()
            and _normalize_slot_text(item.get("phase_code") or item.get("phase")) == current_phase.name
        )
    ]

    missing_slot_keys_precheck: List[str] = []
    for slot_key in required_slot_keys:
        current_value = _normalize_slot_text(current_phase_values.get(slot_key))
        has_candidate = slot_key in candidate_slot_keys
        if current_value:
            continue
        if has_candidate:
            continue
        missing_slot_keys_precheck.append(slot_key)

    slot_focus_candidates_precheck: List[str] = []
    for slot_key in missing_slot_keys_precheck + candidate_slot_keys:
        if not slot_key or slot_key in slot_focus_candidates_precheck:
            continue
        slot_focus_candidates_precheck.append(slot_key)
        if len(slot_focus_candidates_precheck) >= 4:
            break

    bundle = Layer1Bundle(
        current_phase=current_phase.name,
        required_slot_keys=required_slot_keys,
        current_slot_values_before_review={key: _normalize_slot_text(value) for key, value in current_phase_values.items()},
        review_target_slot_keys=review_target_slot_keys,
        review_target_slot_refs=review_target_slot_refs,
        review_scope="mixed",
        target_phase_codes=[phase.name for phase in review_phase_order],
        precheck_quality_by_slot=precheck_quality_by_slot,
        review_target_quality_threshold=round(float(review_target_quality_threshold), 3),
        candidate_updates=candidate_updates,
        need_previous_phase_slot_updates=need_previous_phase_updates,
        need_future_phase_slot_updates=need_future_phase_updates,
        layer1_note=layer1_note,
        evidence_validation=evidence_validation,
        missing_slot_keys_precheck=missing_slot_keys_precheck,
        slot_focus_candidates_precheck=slot_focus_candidates_precheck,
        explicit_correction_slot_keys=explicit_correction_slot_keys,
        explicit_correction_slot_refs=explicit_correction_slot_refs,
        focus_choice_needed=bool(focus_choice_hint.get("needed")),
        focus_choice_candidates=list(focus_choice_hint.get("candidate_topics") or []),
        explicit_focus_preference_present=bool(focus_choice_hint.get("explicit_preference_present")),
        focus_choice_reason=_normalize_slot_text(focus_choice_hint.get("reason")),
    )
    debug = {
        "current_phase": current_phase.value,
        "candidate_count": len(candidate_updates),
        "candidate_count_raw_from_layer1": len(flattened_candidate_updates),
        "review_target_slot_keys": review_target_slot_keys,
        "review_target_slot_refs": review_target_slot_refs,
        "precheck_quality_by_slot": precheck_quality_by_slot,
        "review_target_quality_threshold": round(float(review_target_quality_threshold), 3),
        "candidate_pair_state_by_slot": candidate_pair_state_by_slot,
        "missing_slot_keys_precheck": missing_slot_keys_precheck,
        "slot_focus_candidates_precheck": slot_focus_candidates_precheck,
        "focus_choice_hint": focus_choice_hint,
        "explicit_correction_slot_keys": explicit_correction_slot_keys,
        "explicit_correction_slot_refs": explicit_correction_slot_refs,
        "need_previous_phase_slot_updates": need_previous_phase_updates,
        "need_future_phase_slot_updates": need_future_phase_updates,
        "include_non_current_updates_in_layer2_review": include_non_current_updates_in_layer2_review,
        "review_phase_order": [phase.name for phase in review_phase_order],
        "update_phase_order_from_slot_updates": [phase.name for phase in update_phase_order],
        "allowed_phase_codes_for_apply": sorted(allowed_phase_codes),
        "filtered_update_count": len(filtered_updates),
        "dropped_candidates_outside_review_target": dropped_candidates_outside_review_target,
        "dropped_updates_non_allowed_phase": dropped_updates_non_allowed_phase,
        "dropped_updates_unknown_phase": dropped_updates_unknown_phase,
        "layer1_note": layer1_note,
        "scale_followup_score_backfill": scale_followup_backfill_debug,
    }
    return bundle, debug


def _apply_phase_slot_update_list_to_state(
    state: DialogueState,
    updates: Sequence[Mapping[str, Any]],
    *,
    allowed_phases: Optional[Sequence[Phase]] = None,
) -> Dict[str, Any]:
    if not isinstance(updates, Sequence) or not updates:
        return {"applied": False, "reason": "no_slot_updates"}

    update_results: List[Dict[str, Any]] = []
    applied_count = 0
    confirmation_notes: List[str] = []
    for update in updates:
        result = _apply_phase_slot_updates_to_state(
            state,
            update,
            allowed_phases=allowed_phases,
        )
        update_results.append(
            {
                "phase": update.get("phase") if isinstance(update, Mapping) else None,
                "phase_code": update.get("phase_code") if isinstance(update, Mapping) else None,
                **result,
            }
        )
        note = _normalize_slot_text(result.get("confirmation_note"))
        if note:
            confirmation_notes.append(note)
        if bool(result.get("applied")):
            applied_count += 1

    if applied_count <= 0:
        payload = {
            "applied": False,
            "reason": "no_effective_slot_updates",
            "update_count": len(update_results),
            "applied_count": 0,
            "update_results": update_results,
        }
        if confirmation_notes:
            payload["confirmation_notes"] = confirmation_notes
        return payload

    payload = {
        "applied": True,
        "update_count": len(update_results),
        "applied_count": applied_count,
        "update_results": update_results,
    }
    if confirmation_notes:
        payload["confirmation_notes"] = confirmation_notes
    return payload


def _apply_phase_slot_updates_to_state(
    state: DialogueState,
    slot_payload: Optional[Mapping[str, Any]],
    *,
    allowed_phases: Optional[Sequence[Phase]] = None,
) -> Dict[str, Any]:
    if not isinstance(slot_payload, Mapping):
        return {"applied": False, "reason": "no_slot_payload"}

    phase = _parse_phase_from_any(slot_payload.get("phase_code") or slot_payload.get("phase"))
    raw_slots = slot_payload.get("slots")
    if not isinstance(raw_slots, Mapping):
        return {"applied": False, "reason": "slots_missing"}
    if phase is None:
        return {"applied": False, "reason": "phase_missing"}
    if allowed_phases is not None:
        allowed_phase_codes = {
            phase_item.name
            for phase_item in allowed_phases
            if isinstance(phase_item, Phase)
        }
        if phase.name not in allowed_phase_codes:
            return {
                "applied": False,
                "reason": "phase_not_allowed",
                "phase": phase.value,
                "phase_code": phase.name,
                "allowed_phase_codes": sorted(allowed_phase_codes),
            }

    memory = _copy_phase_slot_memory(state.phase_slots)
    phase_bucket = memory.get(phase.name)
    if phase_bucket is None:
        return {"applied": False, "reason": "phase_bucket_missing"}

    applied_slots: Dict[str, str] = {}
    applied_slot_meta: Dict[str, Dict[str, Any]] = {}
    blocked_clear_slots: List[str] = []
    blocked_missing_evidence_slots: List[str] = []
    blocked_wrong_format_slots: List[Dict[str, Any]] = []
    low_confidence_slots: List[Dict[str, Any]] = []
    confirmation_items: List[str] = []
    for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
        if slot_key not in raw_slots:
            continue
        existing = _normalize_slot_text(phase_bucket.get(slot_key))
        candidate = _normalize_slot_candidate(raw_slots.get(slot_key))
        sv = _normalize_slot_text(candidate.get("value"))
        user_quote = _normalize_slot_text(candidate.get("user_quote")) or _normalize_slot_text(candidate.get("evidence_quote"))
        user_evidence_turn_ids = _normalize_turn_id_list(candidate.get("user_evidence_turn_ids"), min_turn=1)
        assistant_quote = _normalize_slot_text(candidate.get("assistant_quote"))
        assistant_evidence_turn_ids = _normalize_turn_id_list(candidate.get("assistant_evidence_turn_ids"), min_turn=1)
        evidence_quote = user_quote or assistant_quote
        confidence = _clamp01(candidate.get("confidence"), 0.0)
        if not sv:
            if existing:
                blocked_clear_slots.append(slot_key)
            continue
        if not evidence_quote:
            blocked_missing_evidence_slots.append(slot_key)
            confirmation_items.append(f"{slot_key}: no evidence quote (candidate={sv})")
            continue
        if confidence < _SLOT_APPLY_CONFIDENCE_THRESHOLD:
            low_confidence_slots.append(
                {
                    "slot_key": slot_key,
                    "value": sv,
                    "evidence_quote": evidence_quote,
                    "confidence": confidence,
                }
            )
            confirmation_items.append(
                f"{slot_key}: confidence {confidence:.2f} (candidate={sv})"
            )
            continue
        slot_format_issue = _validate_slot_value_format(slot_key=slot_key, value=sv)
        if slot_format_issue:
            blocked_wrong_format_slots.append(
                {
                    "slot_key": slot_key,
                    "value": sv,
                    "issue": slot_format_issue,
                }
            )
            confirmation_items.append(
                f"{slot_key}: format mismatch ({slot_format_issue}) (candidate={sv})"
            )
            continue
        phase_bucket[slot_key] = sv
        applied_slots[slot_key] = sv
        applied_slot_meta[slot_key] = {
            "evidence_quote": evidence_quote,
            "evidence_turn_ids": list(user_evidence_turn_ids),
            "user_quote": user_quote,
            "user_evidence_turn_ids": user_evidence_turn_ids,
            "assistant_quote": assistant_quote,
            "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
            "confidence": confidence,
        }

    confirmation_note = ""
    if confirmation_items:
        confirmation_note = "Slots requiring confirmation: " + " / ".join(confirmation_items[:3])

    if not applied_slots:
        result = {"applied": False, "reason": "no_effective_slots", "phase": phase.value}
        if blocked_clear_slots:
            result["blocked_clear_slots"] = blocked_clear_slots
        if blocked_missing_evidence_slots:
            result["blocked_missing_evidence_slots"] = blocked_missing_evidence_slots
        if blocked_wrong_format_slots:
            result["blocked_wrong_format_slots"] = blocked_wrong_format_slots
        if low_confidence_slots:
            result["low_confidence_slots"] = low_confidence_slots
        if confirmation_note:
            result["confirmation_note"] = confirmation_note
        return result

    state.phase_slots = memory

    result: Dict[str, Any] = {
        "applied": True,
        "phase": phase.value,
        "phase_code": phase.name,
        "applied_slots": applied_slots,
        "applied_slot_meta": applied_slot_meta,
    }
    if blocked_clear_slots:
        result["blocked_clear_slots"] = blocked_clear_slots
    if blocked_missing_evidence_slots:
        result["blocked_missing_evidence_slots"] = blocked_missing_evidence_slots
    if blocked_wrong_format_slots:
        result["blocked_wrong_format_slots"] = blocked_wrong_format_slots
    if low_confidence_slots:
        result["low_confidence_slots"] = low_confidence_slots
    if confirmation_note:
        result["confirmation_note"] = confirmation_note
    return result


def _review_candidate_key(phase_code: Any, slot_key: Any) -> Tuple[str, str]:
    phase = _parse_phase_from_any(phase_code)
    return (
        phase.name if phase is not None else "",
        str(slot_key or "").strip(),
    )


def _review_target_refs_from_bundle(layer1_bundle: Layer1Bundle) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    for raw_ref in layer1_bundle.review_target_slot_refs:
        phase_code_raw, _, slot_key_raw = str(raw_ref or "").partition(".")
        key = _review_candidate_key(phase_code_raw, slot_key_raw)
        if not key[0] or not key[1] or key in refs:
            continue
        refs.append(key)
    return refs


def _layer1_bundle_target_phase_codes(
    layer1_bundle: Layer1Bundle,
) -> List[str]:
    phase_codes: List[str] = []
    for phase_code, _slot_key in _review_target_refs_from_bundle(layer1_bundle):
        if phase_code not in phase_codes:
            phase_codes.append(phase_code)
    return phase_codes


def _normalize_focus_choice_hint(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "needed": False,
            "explicit_preference_present": False,
            "candidate_topics": [],
            "reason": "",
        }

    needed_raw: Any = None
    for key in ("needed", "focus_choice_needed", "clarify_needed"):
        if key in raw:
            needed_raw = raw.get(key)
            break

    explicit_raw: Any = None
    for key in (
        "explicit_preference_present",
        "explicit_focus_preference_present",
        "preference_already_stated",
    ):
        if key in raw:
            explicit_raw = raw.get(key)
            break

    candidate_topics_raw: Any = None
    for key in ("candidate_topics", "focus_candidates", "topic_candidates"):
        if key in raw:
            candidate_topics_raw = raw.get(key)
            break

    candidate_topics: List[str] = []
    if isinstance(candidate_topics_raw, Sequence) and not isinstance(candidate_topics_raw, (str, bytes)):
        for item in candidate_topics_raw:
            text = _normalize_slot_text(item)
            if not text or text in candidate_topics:
                continue
            candidate_topics.append(text)
            if len(candidate_topics) >= 3:
                break

    reason = _normalize_slot_text(raw.get("reason") or raw.get("note"))
    explicit_preference_present = _coerce_bool_with_ja(explicit_raw, False)
    needed = _coerce_bool_with_ja(needed_raw, False)
    if explicit_preference_present:
        needed = False

    return {
        "needed": needed,
        "explicit_preference_present": explicit_preference_present,
        "candidate_topics": candidate_topics,
        "reason": reason,
    }


def _extract_focus_choice_options(
    focus_choice_context: Optional[Mapping[str, Any]],
    *,
    max_items: int = 3,
) -> List[str]:
    normalized = _normalize_focus_choice_hint(focus_choice_context)
    options: List[str] = []
    for item in normalized.get("candidate_topics") or []:
        text = _sanitize_human_text(item)
        if not text or text in options:
            continue
        options.append(text)
        if len(options) >= max(1, int(max_items)):
            break
    return options


def _resolve_clarify_preference_mode(
    *,
    action: MainAction,
    focus_choice_context: Optional[Mapping[str, Any]],
) -> Tuple[str, List[str]]:
    if action != MainAction.CLARIFY_PREFERENCE:
        return "default", []
    normalized = _normalize_focus_choice_hint(focus_choice_context)
    options = _extract_focus_choice_options(normalized)
    if (
        normalized.get("needed")
        and not normalized.get("explicit_preference_present")
        and len(options) >= 2
    ):
        return "focus_choice", options
    return "default", []


def _copy_layer1_bundle_with_scope(
    layer1_bundle: Layer1Bundle,
    *,
    review_scope: str,
    review_target_refs: Sequence[Tuple[str, str]],
) -> Layer1Bundle:
    allowed_ref_set = set(review_target_refs)
    allowed_slot_keys = [slot_key for _, slot_key in review_target_refs]
    candidate_updates: List[Dict[str, Any]] = []
    for item in layer1_bundle.candidate_updates:
        if not isinstance(item, Mapping):
            continue
        key = _review_candidate_key(item.get("phase_code") or item.get("phase"), item.get("slot_key"))
        if key not in allowed_ref_set:
            continue
        candidate_updates.append(dict(item))

    target_phase_codes = _layer1_bundle_target_phase_codes(
        Layer1Bundle(review_target_slot_refs=[f"{phase_code}.{slot_key}" for phase_code, slot_key in review_target_refs])
    )
    current_phase = _parse_phase_from_any(layer1_bundle.current_phase)
    precheck_quality_by_slot = (
        {
            slot_key: layer1_bundle.precheck_quality_by_slot.get(slot_key, 0.0)
            for slot_key in allowed_slot_keys
            if slot_key in layer1_bundle.precheck_quality_by_slot
        }
        if review_scope == "current"
        else {}
    )
    current_slot_values_before_review = (
        {
            slot_key: layer1_bundle.current_slot_values_before_review.get(slot_key, "")
            for slot_key in allowed_slot_keys
            if slot_key in layer1_bundle.current_slot_values_before_review
        }
        if review_scope == "current"
        else {}
    )
    required_slot_keys = (
        list(_PHASE_SLOT_SCHEMA.get(current_phase, ()))
        if review_scope == "current" and isinstance(current_phase, Phase)
        else []
    )
    return Layer1Bundle(
        current_phase=layer1_bundle.current_phase,
        required_slot_keys=required_slot_keys,
        current_slot_values_before_review=current_slot_values_before_review,
        review_target_slot_keys=list(dict.fromkeys(allowed_slot_keys)),
        review_target_slot_refs=[f"{phase_code}.{slot_key}" for phase_code, slot_key in review_target_refs],
        review_scope=review_scope,
        target_phase_codes=target_phase_codes,
        precheck_quality_by_slot=precheck_quality_by_slot,
        review_target_quality_threshold=layer1_bundle.review_target_quality_threshold,
        candidate_updates=candidate_updates,
        need_previous_phase_slot_updates=layer1_bundle.need_previous_phase_slot_updates,
        need_future_phase_slot_updates=layer1_bundle.need_future_phase_slot_updates,
        layer1_note=layer1_bundle.layer1_note,
        evidence_validation=[dict(item) for item in layer1_bundle.evidence_validation if isinstance(item, Mapping)],
        missing_slot_keys_precheck=list(layer1_bundle.missing_slot_keys_precheck),
        slot_focus_candidates_precheck=list(layer1_bundle.slot_focus_candidates_precheck),
        explicit_correction_slot_keys=list(layer1_bundle.explicit_correction_slot_keys),
        explicit_correction_slot_refs=list(layer1_bundle.explicit_correction_slot_refs),
        focus_choice_needed=layer1_bundle.focus_choice_needed,
        focus_choice_candidates=list(layer1_bundle.focus_choice_candidates),
        explicit_focus_preference_present=layer1_bundle.explicit_focus_preference_present,
        focus_choice_reason=layer1_bundle.focus_choice_reason,
    )


def _split_layer1_bundle_for_layer2(
    layer1_bundle: Layer1Bundle,
    *,
    current_phase: Phase,
) -> Tuple[Layer1Bundle, Layer1Bundle]:
    current_refs: List[Tuple[str, str]] = []
    non_current_refs: List[Tuple[str, str]] = []
    for phase_code, slot_key in _review_target_refs_from_bundle(layer1_bundle):
        if phase_code == current_phase.name:
            current_refs.append((phase_code, slot_key))
        else:
            non_current_refs.append((phase_code, slot_key))
    return (
        _copy_layer1_bundle_with_scope(
            layer1_bundle,
            review_scope="current",
            review_target_refs=current_refs,
        ),
        _copy_layer1_bundle_with_scope(
            layer1_bundle,
            review_scope="non_current",
            review_target_refs=non_current_refs,
        ),
    )


def _focus_choice_context_from_layer1_bundle(
    layer1_bundle: Optional[Layer1Bundle],
) -> Dict[str, Any]:
    if not isinstance(layer1_bundle, Layer1Bundle):
        return _normalize_focus_choice_hint(None)
    return _normalize_focus_choice_hint(
        {
            "needed": layer1_bundle.focus_choice_needed,
            "explicit_preference_present": layer1_bundle.explicit_focus_preference_present,
            "candidate_topics": list(layer1_bundle.focus_choice_candidates),
            "reason": layer1_bundle.focus_choice_reason,
        }
    )


def _normalize_review_evidence_role(value: Any) -> str:
    role = _normalize_slot_text(value).lower()
    if role in {"user", "client", "u"}:
        return "user"
    if role in {"assistant", "counselor", "therapist", "a"}:
        return "assistant"
    if "user" in role or "client" in role:
        return "user"
    if "assistant" in role or "counselor" in role or "therapist" in role:
        return "assistant"
    return ""


def _normalize_review_evidence_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, Tuple[int, ...], str]] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        source = _normalize_slot_text(raw.get("source"))
        role = _normalize_review_evidence_role(raw.get("role") or raw.get("speaker") or source)
        quote = _normalize_slot_text(raw.get("quote") or raw.get("evidence_quote") or raw.get("text"))
        turn_ids = _normalize_turn_id_list(
            raw.get("turn_ids") or raw.get("evidence_turn_ids") or raw.get("evidence_ids"),
            min_turn=1,
        )
        if not quote and not turn_ids:
            continue
        dedup_key = (role, quote, tuple(turn_ids), source)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        item: Dict[str, Any] = {
            "role": role or "unknown",
            "quote": quote,
            "turn_ids": turn_ids,
        }
        if source:
            item["source"] = source
        normalized.append(item)
        if len(normalized) >= 10:
            break
    return normalized


def _aggregate_review_evidence_items(
    evidence_items: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[int], str, List[int]]:
    user_quote = ""
    assistant_quote = ""
    user_turn_ids: List[int] = []
    assistant_turn_ids: List[int] = []
    for item in evidence_items:
        if not isinstance(item, Mapping):
            continue
        role = _normalize_review_evidence_role(item.get("role") or item.get("source"))
        quote = _normalize_slot_text(item.get("quote") or item.get("evidence_quote"))
        turn_ids = _normalize_turn_id_list(item.get("turn_ids") or item.get("evidence_turn_ids"), min_turn=1)
        if role == "user":
            if quote and not user_quote:
                user_quote = quote
            user_turn_ids = _normalize_turn_id_list([*user_turn_ids, *turn_ids], min_turn=1)
            continue
        if role == "assistant":
            if quote and not assistant_quote:
                assistant_quote = quote
            assistant_turn_ids = _normalize_turn_id_list([*assistant_turn_ids, *turn_ids], min_turn=1)
    return user_quote, user_turn_ids, assistant_quote, assistant_turn_ids


def _build_review_evidence_items_from_candidate(candidate: Mapping[str, Any]) -> List[Dict[str, Any]]:
    evidence_items: List[Dict[str, Any]] = []
    variants = candidate.get("candidate_variants")
    if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            source = _normalize_slot_text(variant.get("variant_source") or variant.get("source"))
            user_quote = _normalize_slot_text(variant.get("user_quote"))
            user_turn_ids = _normalize_turn_id_list(variant.get("user_evidence_turn_ids"), min_turn=1)
            assistant_quote = _normalize_slot_text(variant.get("assistant_quote"))
            assistant_turn_ids = _normalize_turn_id_list(variant.get("assistant_evidence_turn_ids"), min_turn=1)
            if user_quote or user_turn_ids:
                evidence_items.append(
                    {
                        "role": "user",
                        "source": source,
                        "quote": user_quote,
                        "turn_ids": user_turn_ids,
                    }
                )
            if assistant_quote or assistant_turn_ids:
                evidence_items.append(
                    {
                        "role": "assistant",
                        "source": source,
                        "quote": assistant_quote,
                        "turn_ids": assistant_turn_ids,
                    }
                )
    if not evidence_items:
        fallback_source = _normalize_slot_text(candidate.get("selected_candidate_source") or candidate.get("candidate_source"))
        user_quote = _normalize_slot_text(candidate.get("user_quote"))
        user_turn_ids = _normalize_turn_id_list(candidate.get("user_evidence_turn_ids"), min_turn=1)
        assistant_quote = _normalize_slot_text(candidate.get("assistant_quote"))
        assistant_turn_ids = _normalize_turn_id_list(candidate.get("assistant_evidence_turn_ids"), min_turn=1)
        if user_quote or user_turn_ids:
            evidence_items.append(
                {
                    "role": "user",
                    "source": fallback_source,
                    "quote": user_quote,
                    "turn_ids": user_turn_ids,
                }
            )
        if assistant_quote or assistant_turn_ids:
            evidence_items.append(
                {
                    "role": "assistant",
                    "source": fallback_source,
                    "quote": assistant_quote,
                    "turn_ids": assistant_turn_ids,
                }
            )
    return _normalize_review_evidence_items(evidence_items)


def _merge_issue_codes(*groups: Sequence[Any]) -> List[str]:
    merged: List[str] = []
    for group in groups:
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            continue
        for item in group:
            code = _normalize_slot_text(item)
            if not code or code in merged:
                continue
            merged.append(code)
            if len(merged) >= 6:
                return merged
    return merged


def _is_review_evidence_sufficient(
    *,
    user_quote: str,
    user_turn_ids: Sequence[int],
    assistant_quote: str,
    assistant_turn_ids: Sequence[int],
) -> bool:
    normalized_user_quote = _normalize_slot_text(user_quote)
    normalized_assistant_quote = _normalize_slot_text(assistant_quote)
    normalized_user_turn_ids = _normalize_turn_id_list(user_turn_ids, min_turn=1)
    normalized_assistant_turn_ids = _normalize_turn_id_list(assistant_turn_ids, min_turn=1)
    has_user_evidence = bool(normalized_user_quote and normalized_user_turn_ids)
    has_assistant_evidence = bool(normalized_assistant_quote and normalized_assistant_turn_ids)
    return has_user_evidence or has_assistant_evidence


def _drop_stale_missing_evidence_issue(
    *,
    issue_codes: Sequence[str],
    admissible: bool,
    admissibility_reason: str,
    evidence_sufficient: bool,
) -> Tuple[List[str], bool, str]:
    normalized_issue_codes = _merge_issue_codes(issue_codes)
    normalized_reason = _normalize_slot_text(admissibility_reason)
    if not evidence_sufficient:
        return normalized_issue_codes, admissible, normalized_reason

    filtered_issue_codes = [code for code in normalized_issue_codes if code != "missing_evidence"]
    if normalized_reason == "missing_evidence":
        normalized_reason = ""
    if not admissible and not filtered_issue_codes and not normalized_reason:
        admissible = True
    return filtered_issue_codes, admissible, normalized_reason


def _compute_layer1_admissibility(
    *,
    slot_key: str,
    candidate_value: str,
    extraction_confidence: float,
    evidence_valid: bool,
    validation_issue: str,
) -> Tuple[bool, str, List[str]]:
    normalized_slot_key = _normalize_slot_text(slot_key)
    normalized_value = _normalize_slot_text(candidate_value)
    normalized_issue = _normalize_slot_text(validation_issue)
    confidence = _clamp01(extraction_confidence, 0.0)
    issue_codes: List[str] = []

    if not normalized_value:
        return False, "empty_candidate", ["too_vague"]
    if normalized_issue == "missing_evidence":
        return False, normalized_issue, ["missing_evidence"]
    if normalized_issue in {"wrong_turn", "unsupported_by_quote", "partial_match", "non_user_source", "too_short"}:
        issue_codes.append("low_confidence")
    if normalized_issue in _SLOT_VALIDATION_FORMAT_ISSUES:
        if normalized_slot_key in _SLOT_NUMERIC_REQUIRED_KEYS and normalized_issue == "scale_missing_number":
            return False, normalized_issue, ["wrong_format", "low_confidence"]
        return False, normalized_issue, ["wrong_format"]
    if not evidence_valid and "low_confidence" not in issue_codes:
        issue_codes.append("low_confidence")
    if confidence < _SLOT_LAYER1_ADMISSIBILITY_CONFIDENCE_THRESHOLD:
        return False, "low_extraction_confidence", issue_codes or ["low_confidence"]
    if normalized_slot_key in _SLOT_NUMERIC_REQUIRED_KEYS and not re.search(r"\d", normalized_value):
        return False, "wrong_format", ["wrong_format"]
    return True, "", issue_codes


def _has_numeric_scale_value(value: str) -> bool:
    score = _extract_scale_0_10_value(value)
    return score is not None and 0.0 <= float(score) <= 10.0


def _derive_review_decision(
    *,
    admissible: bool,
    quality_score: float,
    issue_codes: Sequence[str],
) -> str:
    if not admissible:
        return "needs_confirmation"
    if quality_score < _SLOT_CONFIRM_QUALITY_THRESHOLD or "too_vague" in issue_codes:
        return "revise"
    return "accept"


def _harmonize_review_outcome(
    *,
    requested_decision: str,
    admissible: bool,
    admissibility_reason: str,
    issue_codes: Sequence[str],
) -> Tuple[str, bool, str]:
    """
    Enforce minimal consistency for the review outcome.
    - If `missing_evidence` is present, downgrade `accept/revise` to `needs_confirmation`.
    - Resolve contradictions between non-admissible results and `accept/revise`.
    """
    decision = _normalize_slot_text(requested_decision).lower()
    if decision not in {"accept", "revise", "reject", "needs_confirmation", "clear_request"}:
        decision = ""
    reason = _normalize_slot_text(admissibility_reason)
    merged_issue_codes = _merge_issue_codes(issue_codes)

    # `clear_request` expresses an intent to clear the value, so preserve it with highest priority.
    if decision == "clear_request":
        return decision, True, reason

    has_missing_evidence = "missing_evidence" in merged_issue_codes
    if has_missing_evidence:
        admissible = False
        if not reason:
            reason = "missing_evidence"
        if decision in {"", "accept", "revise"}:
            decision = "needs_confirmation"

    if decision == "reject":
        admissible = False
        if not reason:
            reason = "llm_decision_non_accept"

    if not admissible and decision in {"", "accept", "revise"}:
        decision = "needs_confirmation"

    return decision, admissible, reason


def _resolve_review_outcome(
    *,
    slot_key: str,
    candidate_value: str,
    quality_score: float,
    requested_decision: str,
    requested_quality_action: str,
    admissible: bool,
    admissibility_reason: str,
    issue_codes: Sequence[str],
    layer1_validation_issue: str = "",
    format_only_block_hint: bool = False,
    honor_requested_quality_action: bool = True,
) -> Tuple[str, str, bool, str, List[str]]:
    """
    Resolve the final decision for a reviewed slot in one place.
    Returns: (decision, quality_action, admissible, admissibility_reason, issue_codes)
    """
    normalized_slot_key = _normalize_slot_text(slot_key)
    normalized_value = _normalize_slot_text(candidate_value)
    normalized_validation_issue = _normalize_slot_text(layer1_validation_issue)
    normalized_reason = _normalize_slot_text(admissibility_reason)
    normalized_issue_codes = _merge_issue_codes(issue_codes)

    if not admissible and not normalized_issue_codes:
        if normalized_reason in {"missing_evidence", "wrong_turn", "unsupported_by_quote", "invalid_evidence"}:
            normalized_issue_codes.append("missing_evidence")
        elif normalized_reason in {"wrong_format", *_SLOT_VALIDATION_FORMAT_ISSUES}:
            normalized_issue_codes.append("wrong_format")
        else:
            normalized_issue_codes.append("low_confidence")

    if quality_score < _SLOT_CONFIRM_QUALITY_THRESHOLD and "too_vague" not in normalized_issue_codes:
        normalized_issue_codes.append("too_vague")

    if normalized_slot_key in _SLOT_NUMERIC_REQUIRED_KEYS:
        has_numeric_scale_value = _has_numeric_scale_value(normalized_value)
        had_wrong_format = "wrong_format" in normalized_issue_codes
        format_only_block = had_wrong_format or normalized_reason == "wrong_format" or bool(format_only_block_hint)
        if has_numeric_scale_value:
            has_non_format_blocker = any(
                code in {"missing_evidence", "low_confidence"} for code in normalized_issue_codes
            )
            if had_wrong_format and not has_non_format_blocker:
                normalized_issue_codes = [
                    code for code in normalized_issue_codes if code != "wrong_format"
                ]
            if format_only_block and not has_non_format_blocker and normalized_value:
                admissible = True
                if normalized_reason == "wrong_format":
                    normalized_reason = ""
                if _normalize_slot_text(requested_decision).lower() == "needs_confirmation":
                    requested_decision = ""
        else:
            if "wrong_format" not in normalized_issue_codes:
                normalized_issue_codes.append("wrong_format")
            admissible = False
            if not normalized_reason or normalized_reason == "llm_decision_non_accept":
                normalized_reason = "wrong_format"
            if _normalize_slot_text(requested_decision).lower() in {"", "accept", "revise"}:
                requested_decision = "needs_confirmation"

    if normalized_slot_key == "commitment_level":
        had_wrong_format = "wrong_format" in normalized_issue_codes
        if had_wrong_format:
            normalized_issue_codes = [code for code in normalized_issue_codes if code != "wrong_format"]
        format_only_block = had_wrong_format or normalized_reason == "wrong_format" or bool(format_only_block_hint)
        has_non_format_blocker = any(
            code in {"missing_evidence", "low_confidence"} for code in normalized_issue_codes
        )
        if format_only_block and not has_non_format_blocker and normalized_value:
            admissible = True
            if normalized_reason == "wrong_format":
                normalized_reason = ""
            if _normalize_slot_text(requested_decision).lower() == "needs_confirmation":
                requested_decision = ""

    if normalized_slot_key == "target_behavior":
        non_actionable_flag = (
            "target_behavior_not_actionable" in normalized_issue_codes
            or normalized_reason == "target_behavior_not_actionable"
            or normalized_validation_issue == "target_behavior_not_actionable"
        )
        if non_actionable_flag:
            if "wrong_format" not in normalized_issue_codes:
                normalized_issue_codes.append("wrong_format")
            if "target_behavior_not_actionable" not in normalized_issue_codes:
                normalized_issue_codes.append("target_behavior_not_actionable")
            admissible = False
            if not normalized_reason or normalized_reason in {"wrong_format", "llm_decision_non_accept"}:
                normalized_reason = "target_behavior_not_actionable"
            if _normalize_slot_text(requested_decision).lower() in {"", "accept", "revise"}:
                requested_decision = "needs_confirmation"

    decision, admissible, normalized_reason = _harmonize_review_outcome(
        requested_decision=requested_decision,
        admissible=admissible,
        admissibility_reason=normalized_reason,
        issue_codes=normalized_issue_codes,
    )
    quality_action = _normalize_slot_text(requested_quality_action).lower()
    if not honor_requested_quality_action or quality_action not in {"keep", "refine", "clarify"}:
        quality_action = _derive_quality_action(
            admissible=admissible,
            quality_score=quality_score,
            issue_codes=normalized_issue_codes,
        )
    final_decision = decision or _derive_review_decision(
        admissible=admissible,
        quality_score=quality_score,
        issue_codes=normalized_issue_codes,
    )
    return final_decision, quality_action, admissible, normalized_reason, normalized_issue_codes


def _derive_quality_action(
    *,
    admissible: bool,
    quality_score: float,
    issue_codes: Sequence[str],
) -> str:
    if not admissible:
        return "clarify"
    if quality_score < _SLOT_CONFIRM_QUALITY_THRESHOLD or "too_vague" in issue_codes:
        return "refine"
    return "keep"


def _build_rule_slot_review(layer1_bundle: Layer1Bundle) -> Dict[str, Any]:
    reviewed_updates: List[Dict[str, Any]] = []
    slot_quality: Dict[str, float] = {}
    blocking_issue_codes: List[str] = []
    slot_quality_target_examples: List[Dict[str, str]] = []
    hinted_slot_keys: Set[str] = set()
    current_phase = _parse_phase_from_any(layer1_bundle.current_phase)
    current_phase_slot_keys = (
        set(_PHASE_SLOT_SCHEMA.get(current_phase, ()))
        if isinstance(current_phase, Phase)
        else set()
    )
    low_quality_hint_threshold = _clamp01(
        layer1_bundle.review_target_quality_threshold,
        _SLOT_LAYER2_REVIEW_TARGET_QUALITY_THRESHOLD,
    )

    for candidate in layer1_bundle.candidate_updates:
        phase_code = _normalize_slot_text(candidate.get("phase_code") or candidate.get("phase"))
        slot_key = _normalize_slot_text(candidate.get("slot_key"))
        if not phase_code or not slot_key:
            continue
        evidence_items = _build_review_evidence_items_from_candidate(candidate)
        (
            evidence_user_quote,
            evidence_user_turn_ids,
            evidence_assistant_quote,
            evidence_assistant_turn_ids,
        ) = _aggregate_review_evidence_items(evidence_items)
        candidate_value = _normalize_slot_text(candidate.get("candidate_value"))
        extraction_confidence = _clamp01(candidate.get("extraction_confidence"), 0.0)
        user_quote = (
            evidence_user_quote
            or _normalize_slot_text(candidate.get("user_quote"))
            or _normalize_slot_text(candidate.get("evidence_quote"))
        )
        user_evidence_turn_ids = _normalize_turn_id_list(
            [
                *_normalize_turn_id_list(
                    candidate.get("user_evidence_turn_ids") or candidate.get("evidence_turn_ids"),
                    min_turn=1,
                ),
                *evidence_user_turn_ids,
            ],
            min_turn=1,
        )
        assistant_quote = evidence_assistant_quote or _normalize_slot_text(candidate.get("assistant_quote"))
        assistant_evidence_turn_ids = _normalize_turn_id_list(
            [
                *_normalize_turn_id_list(candidate.get("assistant_evidence_turn_ids"), min_turn=1),
                *evidence_assistant_turn_ids,
            ],
            min_turn=1,
        )
        evidence_quote = user_quote or assistant_quote
        evidence_turn_ids = list(user_evidence_turn_ids)
        evidence_sufficient = _is_review_evidence_sufficient(
            user_quote=user_quote,
            user_turn_ids=user_evidence_turn_ids,
            assistant_quote=assistant_quote,
            assistant_turn_ids=assistant_evidence_turn_ids,
        )
        evidence_valid = bool(candidate.get("evidence_valid"))
        validation_issue = _normalize_slot_text(candidate.get("validation_issue"))

        admissible, admissibility_reason, admissibility_issue_codes = _compute_layer1_admissibility(
            slot_key=slot_key,
            candidate_value=candidate_value,
            extraction_confidence=extraction_confidence,
            evidence_valid=evidence_valid,
            validation_issue=validation_issue,
        )
        quality_score = _estimate_slot_text_quality(slot_key, candidate_value)
        issue_codes: List[str] = _merge_issue_codes(admissibility_issue_codes)
        issue_codes, admissible, admissibility_reason = _drop_stale_missing_evidence_issue(
            issue_codes=issue_codes,
            admissible=admissible,
            admissibility_reason=admissibility_reason,
            evidence_sufficient=evidence_sufficient,
        )
        if admissible:
            quality_score = _apply_explicit_numeric_scale_quality_floor(
                slot_key=slot_key,
                candidate_value=candidate_value,
                quality_score=quality_score,
                user_quote=user_quote,
                user_turn_ids=user_evidence_turn_ids,
            )
        if not admissible:
            if "missing_evidence" in issue_codes:
                quality_score = min(quality_score, 0.2)
            elif "wrong_format" in issue_codes:
                quality_score = min(quality_score, 0.48)
            else:
                quality_score = min(quality_score, 0.45)

        if quality_score < _SLOT_CONFIRM_QUALITY_THRESHOLD and "too_vague" not in issue_codes:
            issue_codes.append("too_vague")

        quality_action = _derive_quality_action(
            admissible=admissible,
            quality_score=quality_score,
            issue_codes=issue_codes,
        )
        decision = _derive_review_decision(
            admissible=admissible,
            quality_score=quality_score,
            issue_codes=issue_codes,
        )

        if not admissible:
            for issue_code in issue_codes:
                if issue_code not in blocking_issue_codes:
                    blocking_issue_codes.append(issue_code)

        slot_quality[slot_key] = quality_score
        review_note = ""
        if issue_codes:
            review_note = " / ".join(issue_codes)
        reviewed_updates.append(
            {
                "phase": _parse_phase_from_any(phase_code).value if _parse_phase_from_any(phase_code) else phase_code,
                "phase_code": phase_code,
                "slot_key": slot_key,
                "decision": decision,
                "quality_action": quality_action,
                "admissible": admissible,
                "admissibility_reason": admissibility_reason,
                "layer1_validation_issue": validation_issue,
                "original_value": candidate_value,
                "revised_value": candidate_value if decision in {"accept", "needs_confirmation"} else "",
                "evidence_quote": evidence_quote,
                "evidence_turn_ids": evidence_turn_ids,
                "user_quote": user_quote,
                "user_evidence_turn_ids": user_evidence_turn_ids,
                "assistant_quote": assistant_quote,
                "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
                "evidence_items": evidence_items,
                "extraction_confidence": extraction_confidence,
                "quality_score": quality_score,
                "issue_codes": issue_codes,
                "review_note": review_note,
            }
        )

        # Even in rule fallback, keep Layer3-facing hints for low-quality slots in the current phase.
        should_add_current_phase_hint = (
            slot_key in current_phase_slot_keys
            and quality_score < low_quality_hint_threshold
            and slot_key not in hinted_slot_keys
            and len(slot_quality_target_examples) < _MAX_SLOT_REPAIR_HINTS_PER_TURN
        )
        if should_add_current_phase_hint:
            primary_issue = issue_codes[0] if issue_codes else "too_vague"
            slot_label = _PHASE_SLOT_LABELS.get(slot_key, slot_key)
            probe_style = _SLOT_REPAIR_PROBE_STYLE.get(primary_issue, _SLOT_REPAIR_PROBE_STYLE["too_vague"])
            target_information = _sanitize_human_text(candidate_value) or _sanitize_human_text(
                f"{slot_label} details"
            )
            detail_seed = _sanitize_human_text(candidate_value) or _sanitize_human_text(
                f"I want to talk about {slot_label} a bit more concretely"
            )
            detail = (
                detail_seed
                if detail_seed.startswith("「") and detail_seed.endswith("」")
                else f"「{detail_seed}」"
            )
            slot_quality_target_examples.append(
                {
                    "slot_key": slot_key,
                    "slot_label": slot_label,
                    "issue_type": primary_issue,
                    "preferred_probe_style": probe_style,
                    "target_information": target_information,
                    "detail": detail,
                    "quality_upgrade_model_text": detail,
                }
            )
            hinted_slot_keys.add(slot_key)

    quality_values = [score for score in slot_quality.values() if score >= 0.0]
    quality_mean = sum(quality_values) / len(quality_values) if quality_values else 0.0
    quality_min = min(quality_values) if quality_values else 0.0
    return {
        "reviewed_updates": reviewed_updates,
        "slot_quality": {key: round(value, 3) for key, value in slot_quality.items()},
        "quality_mean": round(float(quality_mean), 3),
        "quality_min": round(float(quality_min), 3),
        "blocking_issue_codes": blocking_issue_codes[:6],
        "slot_quality_target_examples": slot_quality_target_examples,
        "slot_repair_hints": slot_quality_target_examples,
        "review_summary_note": _normalize_slot_text(layer1_bundle.layer1_note),
        "review_confidence": 0.62,
    }


def _empty_slot_review_payload() -> Dict[str, Any]:
    return {
        "reviewed_updates": [],
        "slot_quality": {},
        "quality_mean": 0.0,
        "quality_min": 0.0,
        "blocking_issue_codes": [],
        "slot_quality_target_examples": [],
        "slot_repair_hints": [],
        "review_summary_note": "",
        "review_confidence": 0.0,
    }


def _normalize_slot_review_payload(
    *,
    payload: Any,
    layer1_bundle: Layer1Bundle,
    allow_slot_quality_target_examples: bool = True,
    validate_current_phase_slot_repair_hints: bool = True,
    require_current_phase_slot_repair_hints: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    raw = payload if isinstance(payload, Mapping) else {}
    if not isinstance(payload, Mapping):
        issues.append("payload_not_object")

    current_phase = _parse_phase_from_any(layer1_bundle.current_phase)
    current_phase_slot_keys = (
        set(_PHASE_SLOT_SCHEMA.get(current_phase, ()))
        if isinstance(current_phase, Phase)
        else set()
    )

    candidate_keys = {
        _review_candidate_key(item.get("phase_code") or item.get("phase"), item.get("slot_key"))
        for item in layer1_bundle.candidate_updates
    }
    candidate_by_key = {
        _review_candidate_key(item.get("phase_code") or item.get("phase"), item.get("slot_key")): item
        for item in layer1_bundle.candidate_updates
    }

    reviewed_updates: List[Dict[str, Any]] = []
    raw_reviewed_updates = raw.get("reviewed_updates")
    if isinstance(raw_reviewed_updates, Sequence) and not isinstance(raw_reviewed_updates, (str, bytes)):
        for item in raw_reviewed_updates:
            if not isinstance(item, Mapping):
                continue
            key = _review_candidate_key(item.get("phase_code") or item.get("phase"), item.get("slot_key"))
            if key not in candidate_keys:
                issues.append("review_contains_unknown_slot")
                continue
            candidate = candidate_by_key.get(key, {})
            candidate_value = _normalize_slot_text(candidate.get("candidate_value"))
            original_value = _normalize_slot_text(item.get("original_value")) or candidate_value
            revised_value = _normalize_slot_text(item.get("revised_value"))
            evidence_items = _normalize_review_evidence_items(item.get("evidence_items"))
            if not evidence_items and isinstance(candidate, Mapping):
                evidence_items = _build_review_evidence_items_from_candidate(candidate)
            (
                evidence_user_quote,
                evidence_user_turn_ids,
                evidence_assistant_quote,
                evidence_assistant_turn_ids,
            ) = _aggregate_review_evidence_items(evidence_items)
            user_quote = (
                _normalize_slot_text(item.get("user_quote"))
                or evidence_user_quote
                or _normalize_slot_text(candidate.get("user_quote"))
                or _normalize_slot_text(item.get("evidence_quote"))
                or _normalize_slot_text(candidate.get("evidence_quote"))
            )
            user_evidence_turn_ids = _normalize_turn_id_list(
                [
                    *_normalize_turn_id_list(
                        item.get("user_evidence_turn_ids")
                        or candidate.get("user_evidence_turn_ids")
                        or item.get("evidence_turn_ids")
                        or candidate.get("evidence_turn_ids"),
                        min_turn=1,
                    ),
                    *evidence_user_turn_ids,
                ],
                min_turn=1,
            )
            assistant_quote = (
                _normalize_slot_text(item.get("assistant_quote"))
                or evidence_assistant_quote
                or _normalize_slot_text(candidate.get("assistant_quote"))
            )
            assistant_evidence_turn_ids = _normalize_turn_id_list(
                [
                    *_normalize_turn_id_list(
                        item.get("assistant_evidence_turn_ids")
                        or candidate.get("assistant_evidence_turn_ids"),
                        min_turn=1,
                    ),
                    *evidence_assistant_turn_ids,
                ],
                min_turn=1,
            )
            evidence_quote = user_quote or assistant_quote
            evidence_turn_ids = list(user_evidence_turn_ids)
            extraction_confidence = _clamp01(
                item.get("extraction_confidence", candidate.get("extraction_confidence")),
                _clamp01(candidate.get("extraction_confidence"), 0.0),
            )
            quality_score = _clamp01(
                item.get("quality_score"),
                extraction_confidence if extraction_confidence > 0.0 else 0.55,
            )
            layer1_admissible, layer1_admissibility_reason, admissibility_issue_codes = _compute_layer1_admissibility(
                slot_key=key[1],
                candidate_value=candidate_value or original_value,
                extraction_confidence=_clamp01(candidate.get("extraction_confidence"), extraction_confidence),
                evidence_valid=bool(candidate.get("evidence_valid")),
                validation_issue=_normalize_slot_text(candidate.get("validation_issue")),
            )

            # Ask the Layer2 LLM to return only merged results plus quality/evidence, and keep decision logic rule-derived.
            llm_review_decision = _normalize_slot_text(item.get("decision")).lower()
            if llm_review_decision != "clear_request":
                llm_review_decision = ""
            admissible = layer1_admissible
            admissibility_reason = layer1_admissibility_reason

            raw_issue_codes_normalized: List[str] = []
            raw_issue_codes = item.get("issue_codes")
            if isinstance(raw_issue_codes, Sequence) and not isinstance(raw_issue_codes, (str, bytes)):
                for code in raw_issue_codes:
                    normalized = _normalize_slot_text(code)
                    if not normalized or normalized in raw_issue_codes_normalized:
                        continue
                    raw_issue_codes_normalized.append(normalized)
                    if len(raw_issue_codes_normalized) >= 6:
                        break

            issue_codes = _merge_issue_codes(
                raw_issue_codes_normalized,
                admissibility_issue_codes,
            )
            evidence_sufficient = _is_review_evidence_sufficient(
                user_quote=user_quote,
                user_turn_ids=user_evidence_turn_ids,
                assistant_quote=assistant_quote,
                assistant_turn_ids=assistant_evidence_turn_ids,
            )
            issue_codes, admissible, admissibility_reason = _drop_stale_missing_evidence_issue(
                issue_codes=issue_codes,
                admissible=admissible,
                admissibility_reason=admissibility_reason,
                evidence_sufficient=evidence_sufficient,
            )
            if not _has_explicit_numeric_scale_evidence(
                slot_key=key[1],
                user_quote=user_quote,
                user_turn_ids=user_evidence_turn_ids,
                assistant_quote=assistant_quote,
                assistant_turn_ids=assistant_evidence_turn_ids,
            ):
                if "wrong_format" not in issue_codes:
                    issue_codes.append("wrong_format")
                if "low_confidence" not in issue_codes:
                    issue_codes.append("low_confidence")
                admissible = False
                if not admissibility_reason or admissibility_reason == "llm_decision_non_accept":
                    admissibility_reason = "scale_missing_number"
            elif admissible:
                quality_score = _apply_explicit_numeric_scale_quality_floor(
                    slot_key=key[1],
                    candidate_value=(revised_value or original_value or candidate_value),
                    quality_score=quality_score,
                    user_quote=user_quote,
                    user_turn_ids=user_evidence_turn_ids,
                )
            decision, quality_action, admissible, admissibility_reason, issue_codes = _resolve_review_outcome(
                slot_key=key[1],
                candidate_value=(revised_value or original_value or candidate_value),
                quality_score=quality_score,
                requested_decision=llm_review_decision,
                requested_quality_action="",
                admissible=admissible,
                admissibility_reason=admissibility_reason,
                issue_codes=issue_codes,
                layer1_validation_issue=_normalize_slot_text(candidate.get("validation_issue")),
                format_only_block_hint=(
                    "wrong_format" in raw_issue_codes_normalized
                    or "wrong_format" in admissibility_issue_codes
                ),
                honor_requested_quality_action=False,
            )
            review_note = _normalize_slot_text(item.get("review_note"))
            reviewed_updates.append(
                {
                    "phase": _parse_phase_from_any(key[0]).value if _parse_phase_from_any(key[0]) else key[0],
                    "phase_code": key[0],
                    "slot_key": key[1],
                    "decision": decision,
                    "review_decision_raw": decision,
                    "quality_action": quality_action,
                    "admissible": admissible,
                    "admissibility_reason": admissibility_reason,
                    "layer1_validation_issue": _normalize_slot_text(candidate.get("validation_issue")),
                    "original_value": original_value,
                    "revised_value": revised_value,
                    "evidence_quote": evidence_quote,
                    "evidence_turn_ids": evidence_turn_ids,
                    "user_quote": user_quote,
                    "user_evidence_turn_ids": user_evidence_turn_ids,
                    "assistant_quote": assistant_quote,
                    "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
                    "evidence_items": evidence_items,
                    "extraction_confidence": extraction_confidence,
                    "quality_score": quality_score,
                    "issue_codes": issue_codes,
                    "review_note": review_note,
                }
            )
    if not reviewed_updates:
        issues.append("reviewed_updates_missing")

    slot_quality: Dict[str, float] = {}
    for item in reviewed_updates:
        if not isinstance(item, Mapping):
            continue
        slot_key = _normalize_slot_text(item.get("slot_key"))
        if slot_key not in _PHASE_SLOT_LABELS:
            continue
        score = _clamp01(item.get("quality_score"), -1.0)
        if score >= 0.0:
            slot_quality[slot_key] = score

    blocking_issue_codes: List[str] = []
    for item in reviewed_updates:
        item_admissible = item.get("admissible")
        has_admissible_flag = isinstance(item_admissible, bool)
        legacy_decision = _normalize_slot_text(item.get("decision")).lower()
        should_block = (
            (has_admissible_flag and not bool(item_admissible))
            or (not has_admissible_flag and legacy_decision in {"reject", "needs_confirmation"})
        )
        if not should_block:
            continue
        for code in item.get("issue_codes", []):
            normalized = _normalize_slot_text(code)
            if normalized and normalized not in blocking_issue_codes:
                blocking_issue_codes.append(normalized)
                if len(blocking_issue_codes) >= 6:
                    break

    normalized_hints: List[Dict[str, Any]] = []
    if allow_slot_quality_target_examples:
        slot_quality_target_examples = raw.get("slot_quality_target_examples")
        if slot_quality_target_examples is None:
            slot_quality_target_examples = raw.get("slot_repair_hints")
        if isinstance(slot_quality_target_examples, Sequence) and not isinstance(slot_quality_target_examples, (str, bytes)):
            for item in slot_quality_target_examples:
                if not isinstance(item, Mapping):
                    continue
                slot_key = _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
                if not slot_key:
                    continue
                if (
                    validate_current_phase_slot_repair_hints
                    and current_phase_slot_keys
                    and slot_key not in current_phase_slot_keys
                ):
                    issues.append(f"{_SLOT_REPAIR_HINT_OUTSIDE_CURRENT_PHASE_ISSUE_PREFIX}{slot_key}")
                hint = dict(item)
                hint.setdefault("slot_key", slot_key)
                detail = _extract_slot_quality_target_example_detail_raw(item)
                target_information = _extract_slot_quality_target_information(item)
                if target_information:
                    hint["target_information"] = target_information
                if detail:
                    hint["detail"] = detail
                normalized_hints.append(hint)
    if require_current_phase_slot_repair_hints:
        low_quality_hint_threshold = _clamp01(
            layer1_bundle.review_target_quality_threshold,
            _SLOT_LAYER2_REVIEW_TARGET_QUALITY_THRESHOLD,
        )
        current_phase_low_quality_slot_keys = [
            slot_key
            for slot_key, score in slot_quality.items()
            if (
                slot_key in current_phase_slot_keys
                and _clamp01(score, -1.0) >= 0.0
                and _clamp01(score, -1.0) < low_quality_hint_threshold
            )
        ]
        hinted_current_phase_slot_keys = {
            _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
            for item in normalized_hints
            if (
                isinstance(item, Mapping)
                and _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label")) in current_phase_slot_keys
            )
        }
        hinted_current_phase_slot_keys_with_detail = {
            _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
            for item in normalized_hints
            if (
                isinstance(item, Mapping)
                and _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label")) in current_phase_slot_keys
                and bool(_extract_slot_repair_hint_detail_raw(item))
            )
        }
        for slot_key in current_phase_low_quality_slot_keys:
            if slot_key in hinted_current_phase_slot_keys:
                if slot_key not in hinted_current_phase_slot_keys_with_detail:
                    issues.append(f"{_CURRENT_PHASE_SLOT_REPAIR_HINT_DETAIL_MISSING_ISSUE_PREFIX}{slot_key}")
                continue
            issues.append(f"{_CURRENT_PHASE_SLOT_REPAIR_HINT_MISSING_ISSUE_PREFIX}{slot_key}")

    quality_values = [score for score in slot_quality.values() if score >= 0.0]
    quality_mean = sum(quality_values) / len(quality_values) if quality_values else 0.0
    quality_min = min(quality_values) if quality_values else 0.0
    review_summary_note = _normalize_slot_text(raw.get("review_summary_note"))
    review_confidence = 0.0
    if reviewed_updates:
        review_confidence = sum(
            _clamp01(item.get("extraction_confidence"), 0.0)
            for item in reviewed_updates
            if isinstance(item, Mapping)
        ) / float(len(reviewed_updates))

    return {
        "reviewed_updates": reviewed_updates,
        "slot_quality": {k: round(float(v), 3) for k, v in slot_quality.items()},
        "quality_mean": round(float(quality_mean), 3),
        "quality_min": round(float(quality_min), 3),
        "blocking_issue_codes": blocking_issue_codes[:6],
        "slot_quality_target_examples": normalized_hints,
        "slot_repair_hints": normalized_hints,
        "review_summary_note": review_summary_note,
        "review_confidence": round(float(review_confidence), 3),
    }, issues


def _is_slot_review_contract_issue(issue: Any) -> bool:
    normalized_issue = _normalize_slot_text(issue)
    return normalized_issue.startswith(
        (
            _CURRENT_PHASE_SLOT_REPAIR_HINT_MISSING_ISSUE_PREFIX,
            _CURRENT_PHASE_SLOT_REPAIR_HINT_DETAIL_MISSING_ISSUE_PREFIX,
            _SLOT_REPAIR_HINT_OUTSIDE_CURRENT_PHASE_ISSUE_PREFIX,
        )
    )


def _apply_reviewed_phase_slot_updates_to_state(
    *,
    state: DialogueState,
    reviewed_updates: Sequence[Mapping[str, Any]],
    apply_mode: str = "commit",
    review_source: str = "",
    source_phase: Optional[Phase] = None,
) -> Dict[str, Any]:
    if not isinstance(reviewed_updates, Sequence) or not reviewed_updates:
        return {"applied": False, "reason": "no_reviewed_updates", "applied_count": 0, "reviewed_count": 0}

    normalized_apply_mode = str(apply_mode or "").strip().lower()
    if normalized_apply_mode not in {"commit", "pending_non_current"}:
        normalized_apply_mode = "commit"
    source_phase_name = source_phase.name if isinstance(source_phase, Phase) else ""
    slot_memory = _copy_phase_slot_memory(state.phase_slots)
    slot_meta = _copy_phase_slot_meta(state.phase_slot_meta)
    pending_memory = _copy_phase_slot_memory(state.phase_slot_pending)
    pending_meta = _copy_phase_slot_meta(state.phase_slot_pending_meta)
    default_meta = _init_phase_slot_meta()
    applied_count = 0
    pending_count = 0
    staged_count = 0
    reviewed_count = 0
    update_results: List[Dict[str, Any]] = []

    for reviewed in reviewed_updates:
        if not isinstance(reviewed, Mapping):
            continue
        phase = _parse_phase_from_any(reviewed.get("phase_code") or reviewed.get("phase"))
        slot_key = _normalize_slot_text(reviewed.get("slot_key"))
        if phase is None or slot_key not in _PHASE_SLOT_SCHEMA.get(phase, ()):
            continue
        reviewed_count += 1
        review_decision_raw = _normalize_slot_text(reviewed.get("review_decision_raw") or reviewed.get("decision")).lower()
        if review_decision_raw not in {"accept", "revise", "reject", "needs_confirmation", "clear_request"}:
            review_decision_raw = ""
        original_value = _normalize_slot_text(reviewed.get("original_value"))
        revised_value = _normalize_slot_text(reviewed.get("revised_value"))
        # Prefer `revised_value` with explicit update intent so Layer1 candidates can be applied each turn more easily.
        final_value = revised_value or original_value
        extraction_confidence = _clamp01(reviewed.get("extraction_confidence"), 0.0)
        quality_score = _clamp01(reviewed.get("quality_score"), _estimate_slot_text_quality(slot_key, final_value))
        issue_codes: List[str] = []
        raw_issue_codes = reviewed.get("issue_codes")
        if isinstance(raw_issue_codes, Sequence) and not isinstance(raw_issue_codes, (str, bytes)):
            for item in raw_issue_codes:
                code = _normalize_slot_text(item)
                if not code or code in issue_codes:
                    continue
                issue_codes.append(code)
                if len(issue_codes) >= 6:
                    break
        admissible_raw = reviewed.get("admissible")
        if isinstance(admissible_raw, bool):
            admissible = bool(admissible_raw)
        else:
            admissible = review_decision_raw in {"accept", "revise"}
        admissibility_reason = _normalize_slot_text(reviewed.get("admissibility_reason"))
        review_note = _normalize_slot_text(reviewed.get("review_note"))
        evidence_items = _normalize_review_evidence_items(reviewed.get("evidence_items"))
        (
            evidence_user_quote,
            evidence_user_turn_ids,
            evidence_assistant_quote,
            evidence_assistant_turn_ids,
        ) = _aggregate_review_evidence_items(evidence_items)
        user_quote = (
            _normalize_slot_text(reviewed.get("user_quote"))
            or evidence_user_quote
            or _normalize_slot_text(reviewed.get("evidence_quote"))
        )
        user_evidence_turn_ids = _normalize_turn_id_list(
            [
                *_normalize_turn_id_list(
                    reviewed.get("user_evidence_turn_ids") or reviewed.get("evidence_turn_ids"),
                    min_turn=1,
                ),
                *evidence_user_turn_ids,
            ],
            min_turn=1,
        )
        assistant_quote = _normalize_slot_text(reviewed.get("assistant_quote")) or evidence_assistant_quote
        assistant_evidence_turn_ids = _normalize_turn_id_list(
            [
                *_normalize_turn_id_list(
                    reviewed.get("assistant_evidence_turn_ids"),
                    min_turn=1,
                ),
                *evidence_assistant_turn_ids,
            ],
            min_turn=1,
        )
        evidence_quote = user_quote or assistant_quote
        evidence_turn_ids = list(user_evidence_turn_ids)
        committed_value_before = _normalize_slot_text(slot_memory[phase.name].get(slot_key))
        committed_status_before = _normalize_slot_text((slot_meta.get(phase.name, {}).get(slot_key) or {}).get("status"))
        if committed_status_before not in {"confirmed", "inactive"}:
            committed_status_before = "inactive"
        committed_quality_before = _clamp01((slot_meta.get(phase.name, {}).get(slot_key) or {}).get("quality_score"), 0.0)

        evidence_sufficient = _is_review_evidence_sufficient(
            user_quote=user_quote,
            user_turn_ids=user_evidence_turn_ids,
            assistant_quote=assistant_quote,
            assistant_turn_ids=assistant_evidence_turn_ids,
        )
        if not evidence_sufficient and "missing_evidence" not in issue_codes:
            issue_codes.append("missing_evidence")
        issue_codes, admissible, admissibility_reason = _drop_stale_missing_evidence_issue(
            issue_codes=issue_codes,
            admissible=admissible,
            admissibility_reason=admissibility_reason,
            evidence_sufficient=evidence_sufficient,
        )
        if not _has_explicit_numeric_scale_evidence(
            slot_key=slot_key,
            user_quote=user_quote,
            user_turn_ids=user_evidence_turn_ids,
            assistant_quote=assistant_quote,
            assistant_turn_ids=assistant_evidence_turn_ids,
        ):
            if "wrong_format" not in issue_codes:
                issue_codes.append("wrong_format")
            if "low_confidence" not in issue_codes:
                issue_codes.append("low_confidence")
            admissible = False
            if not admissibility_reason or admissibility_reason == "llm_decision_non_accept":
                admissibility_reason = "scale_missing_number"
        elif admissible:
            quality_score = _apply_explicit_numeric_scale_quality_floor(
                slot_key=slot_key,
                candidate_value=final_value,
                quality_score=quality_score,
                user_quote=user_quote,
                user_turn_ids=user_evidence_turn_ids,
            )

        decision, quality_action, admissible, admissibility_reason, issue_codes = _resolve_review_outcome(
            slot_key=slot_key,
            candidate_value=final_value,
            quality_score=quality_score,
            requested_decision=review_decision_raw,
            requested_quality_action=_normalize_slot_text(reviewed.get("quality_action")).lower(),
            admissible=admissible,
            admissibility_reason=admissibility_reason,
            issue_codes=issue_codes,
            layer1_validation_issue=_normalize_slot_text(reviewed.get("layer1_validation_issue")),
            format_only_block_hint=False,
            honor_requested_quality_action=False,
        )
        review_decision_raw = decision

        candidate_status = "inactive"
        can_apply = bool(final_value)
        clear_requested = review_decision_raw == "clear_request"
        blocked_by_review_decision = review_decision_raw in {"reject", "needs_confirmation"}
        blocked_by_evidence = "missing_evidence" in issue_codes or not evidence_sufficient
        commit_allowed = can_apply and admissible and not blocked_by_review_decision and not blocked_by_evidence
        overwrite_blocked_by_confirmed = False
        applied_to_committed = False
        staged_to_pending = False
        stage_for_phase_entry = bool(
            normalized_apply_mode == "pending_non_current"
            and isinstance(source_phase, Phase)
            and _phase_index(phase) > _phase_index(source_phase)
        )

        if stage_for_phase_entry:
            if clear_requested:
                pending_memory[phase.name][slot_key] = ""
                pending_meta[phase.name][slot_key] = dict(default_meta[phase.name][slot_key])
                pending_count += 1
                candidate_status = "inactive"
            elif commit_allowed:
                pending_memory[phase.name][slot_key] = final_value
                pending_meta[phase.name][slot_key] = {
                    "status": "tentative",
                    "evidence_quote": evidence_quote,
                    "evidence_turn_ids": evidence_turn_ids,
                    "user_quote": user_quote,
                    "user_evidence_turn_ids": user_evidence_turn_ids,
                    "assistant_quote": assistant_quote,
                    "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
                    "extraction_confidence": extraction_confidence,
                    "quality_score": quality_score,
                    "issue_codes": issue_codes,
                    "review_note": review_note,
                    "review_source": review_source or "layer2_non_current",
                    "promote_on_phase_entry": True,
                    "reviewed_at_turn": max(0, int(state.turn_index)),
                    "review_origin_phase": source_phase_name,
                    "review_decision_raw": review_decision_raw,
                }
                pending_count += 1
                staged_count += 1
                candidate_status = "tentative"
                staged_to_pending = True
            else:
                pending_memory[phase.name][slot_key] = ""
                pending_meta[phase.name][slot_key] = dict(default_meta[phase.name][slot_key])
                overwrite_blocked_by_confirmed = bool(
                    committed_status_before == "confirmed"
                    and committed_value_before
                    and can_apply
                    and not clear_requested
                )
                pending_count += 1
        elif clear_requested:
            slot_memory[phase.name][slot_key] = ""
            slot_meta[phase.name][slot_key] = dict(default_meta[phase.name][slot_key])
            pending_memory[phase.name][slot_key] = ""
            pending_meta[phase.name][slot_key] = dict(default_meta[phase.name][slot_key])
            applied_count += 1
            pending_count += 1
            candidate_status = "inactive"
            applied_to_committed = True
        elif commit_allowed:
            # Never lower the quality score; update it only when it increases.
            committed_quality_after = (
                max(committed_quality_before, quality_score)
                if committed_status_before == "confirmed" and committed_value_before
                else quality_score
            )
            slot_memory[phase.name][slot_key] = final_value
            slot_meta[phase.name][slot_key] = {
                "status": "confirmed",
                "evidence_quote": evidence_quote,
                "evidence_turn_ids": evidence_turn_ids,
                "user_quote": user_quote,
                "user_evidence_turn_ids": user_evidence_turn_ids,
                "assistant_quote": assistant_quote,
                "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
                "extraction_confidence": extraction_confidence,
                "quality_score": committed_quality_after,
                "issue_codes": issue_codes,
                "review_note": review_note,
                "review_source": review_source or "layer2_current",
                "promote_on_phase_entry": False,
                "reviewed_at_turn": max(0, int(state.turn_index)),
                "review_origin_phase": source_phase_name,
                "review_decision_raw": review_decision_raw,
            }
            # Clear pending values when a confirmed value is adopted.
            pending_memory[phase.name][slot_key] = ""
            pending_meta[phase.name][slot_key] = dict(default_meta[phase.name][slot_key])
            applied_count += 1
            pending_count += 1
            candidate_status = "confirmed"
            applied_to_committed = True
        else:
            # Do not promote reject / needs_confirmation / evidence-poor candidates into state, and do not keep them in pending either.
            pending_memory[phase.name][slot_key] = ""
            pending_meta[phase.name][slot_key] = dict(default_meta[phase.name][slot_key])
            overwrite_blocked_by_confirmed = bool(
                committed_status_before == "confirmed"
                and committed_value_before
                and can_apply
                and not clear_requested
            )
            pending_count += 1

        committed_value_after = _normalize_slot_text(slot_memory[phase.name].get(slot_key))
        committed_status_after = _normalize_slot_text((slot_meta.get(phase.name, {}).get(slot_key) or {}).get("status"))
        if committed_status_after not in {"confirmed", "inactive"}:
            committed_status_after = "inactive"

        update_results.append(
            {
                "phase": phase.value,
                "phase_code": phase.name,
                "slot_key": slot_key,
                "decision": decision,
                "review_decision_raw": review_decision_raw,
                "quality_action": quality_action,
                "status": committed_status_after,
                "candidate_status": candidate_status,
                "final_value": final_value,
                "applied": applied_to_committed,
                "staged_to_pending": staged_to_pending,
                "overwrite_blocked_by_confirmed": overwrite_blocked_by_confirmed,
                "user_quote": user_quote,
                "user_evidence_turn_ids": user_evidence_turn_ids,
                "assistant_quote": assistant_quote,
                "assistant_evidence_turn_ids": assistant_evidence_turn_ids,
                "evidence_items": evidence_items,
                "committed_value_before": committed_value_before,
                "committed_value_after": committed_value_after,
                "committed_status_before": committed_status_before,
                "committed_status_after": committed_status_after,
                "pending_value": _normalize_slot_text(pending_memory[phase.name].get(slot_key)),
                "pending_status": _normalize_slot_text((pending_meta.get(phase.name, {}).get(slot_key) or {}).get("status")),
                "issue_codes": issue_codes,
            }
        )

    state.phase_slots = slot_memory
    state.phase_slot_meta = slot_meta
    state.phase_slot_pending = pending_memory
    state.phase_slot_pending_meta = pending_meta
    return {
        "applied": applied_count > 0,
        "applied_count": applied_count,
        "pending_count": pending_count,
        "staged_count": staged_count,
        "reviewed_count": reviewed_count,
        "apply_mode": normalized_apply_mode,
        "update_results": update_results,
    }


def _promote_pending_phase_slot_updates_on_phase_entry(
    *,
    state: DialogueState,
    entered_phase: Phase,
) -> Dict[str, Any]:
    slot_memory = _copy_phase_slot_memory(state.phase_slots)
    slot_meta = _copy_phase_slot_meta(state.phase_slot_meta)
    pending_memory = _copy_phase_slot_memory(state.phase_slot_pending)
    pending_meta = _copy_phase_slot_meta(state.phase_slot_pending_meta)
    default_meta = _init_phase_slot_meta()

    promoted_slots: Dict[str, Dict[str, Any]] = {}
    for slot_key in _PHASE_SLOT_SCHEMA.get(entered_phase, ()):
        pending_value = _normalize_slot_text(pending_memory.get(entered_phase.name, {}).get(slot_key))
        pending_slot_meta = pending_meta.get(entered_phase.name, {}).get(slot_key) or {}
        if not pending_value or not isinstance(pending_slot_meta, Mapping):
            continue
        if not bool(pending_slot_meta.get("promote_on_phase_entry")):
            continue
        committed_value_before = _normalize_slot_text(slot_memory[entered_phase.name].get(slot_key))
        committed_quality_before = _clamp01(
            (slot_meta.get(entered_phase.name, {}).get(slot_key) or {}).get("quality_score"),
            0.0,
        )
        pending_quality = _clamp01(pending_slot_meta.get("quality_score"), 0.0)
        committed_quality_after = (
            max(committed_quality_before, pending_quality)
            if committed_value_before
            else pending_quality
        )
        slot_memory[entered_phase.name][slot_key] = pending_value
        slot_meta[entered_phase.name][slot_key] = {
            "status": "confirmed",
            "evidence_quote": _normalize_slot_text(pending_slot_meta.get("evidence_quote")),
            "evidence_turn_ids": _normalize_turn_id_list(pending_slot_meta.get("evidence_turn_ids"), min_turn=1),
            "user_quote": _normalize_slot_text(pending_slot_meta.get("user_quote")),
            "user_evidence_turn_ids": _normalize_turn_id_list(
                pending_slot_meta.get("user_evidence_turn_ids"),
                min_turn=1,
            ),
            "assistant_quote": _normalize_slot_text(pending_slot_meta.get("assistant_quote")),
            "assistant_evidence_turn_ids": _normalize_turn_id_list(
                pending_slot_meta.get("assistant_evidence_turn_ids"),
                min_turn=1,
            ),
            "extraction_confidence": _clamp01(pending_slot_meta.get("extraction_confidence"), 0.0),
            "quality_score": committed_quality_after,
            "issue_codes": [
                _normalize_slot_text(item)
                for item in (pending_slot_meta.get("issue_codes") or [])
                if _normalize_slot_text(item)
            ][:6]
            if isinstance(pending_slot_meta.get("issue_codes"), Sequence)
            and not isinstance(pending_slot_meta.get("issue_codes"), (str, bytes))
            else [],
            "review_note": _normalize_slot_text(pending_slot_meta.get("review_note")),
            "review_source": _normalize_slot_text(pending_slot_meta.get("review_source")) or "layer2_non_current_promoted",
            "promote_on_phase_entry": False,
            "reviewed_at_turn": max(0, int(pending_slot_meta.get("reviewed_at_turn") or 0)),
            "review_origin_phase": _normalize_slot_text(pending_slot_meta.get("review_origin_phase")),
            "review_decision_raw": _normalize_slot_text(pending_slot_meta.get("review_decision_raw")).lower(),
        }
        pending_memory[entered_phase.name][slot_key] = ""
        pending_meta[entered_phase.name][slot_key] = dict(default_meta[entered_phase.name][slot_key])
        promoted_slots[slot_key] = {
            "value": pending_value,
            "quality_score": round(float(committed_quality_after), 3),
            "review_source": slot_meta[entered_phase.name][slot_key].get("review_source"),
        }

    state.phase_slots = slot_memory
    state.phase_slot_meta = slot_meta
    state.phase_slot_pending = pending_memory
    state.phase_slot_pending_meta = pending_meta
    return {
        "promoted": bool(promoted_slots),
        "phase": entered_phase.value,
        "phase_code": entered_phase.name,
        "promoted_count": len(promoted_slots),
        "promoted_slots": promoted_slots,
    }


def _normalize_json_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "strict":
        return "strict"
    return "loose"


def _parse_json_from_text(text: str, *, strict: bool = False) -> Optional[Any]:
    """
    A permissive helper that extracts and parses the first JSON object/array from LLM output.
    - Strip ```json ... ``` code fences and surrounding explanatory text.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    if strict:
        try:
            return json.loads(cleaned)
        except Exception:
            return None

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def _extract_text_from_payload(
    payload: Any,
    *,
    path: str = "",
    preferred_keys: Sequence[str] = (
        "draft_response_text",
        "assistant_response_text",
        "response",
        "reply",
        "text",
        "utterance",
        "message",
        "content",
        "output",
    ),
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a string that can be used as dialogue text from a JSON-like structure.
    """
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            return text, (path or "<root>")
        return None, None

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            item_path = f"{path}[{idx}]" if path else f"[{idx}]"
            extracted, extracted_path = _extract_text_from_payload(
                item,
                path=item_path,
                preferred_keys=preferred_keys,
            )
            if extracted:
                return extracted, extracted_path
        return None, None

    if isinstance(payload, dict):
        # Prefer common key names first.
        for key in preferred_keys:
            if key not in payload:
                continue
            item_path = f"{path}.{key}" if path else key
            extracted, extracted_path = _extract_text_from_payload(
                payload.get(key),
                path=item_path,
                preferred_keys=preferred_keys,
            )
            if extracted:
                return extracted, extracted_path

        # Then scan the remaining keys.
        for key, value in payload.items():
            if key in preferred_keys:
                continue
            item_path = f"{path}.{key}" if path else str(key)
            extracted, extracted_path = _extract_text_from_payload(
                value,
                path=item_path,
                preferred_keys=preferred_keys,
            )
            if extracted:
                return extracted, extracted_path

    return None, None


def _extract_text_from_payload_preferred_only(
    payload: Any,
    *,
    path: str = "",
    under_preferred_path: bool = False,
    preferred_keys: Sequence[str] = (
        "draft_response_text",
        "assistant_response_text",
        "response",
        "reply",
        "text",
        "utterance",
        "message",
        "content",
        "output",
    ),
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract body candidates only from subtrees under the keys listed in `preferred_keys`.
    """
    if isinstance(payload, str):
        text = payload.strip()
        if text and under_preferred_path:
            return text, (path or "<root>")
        return None, None

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            item_path = f"{path}[{idx}]" if path else f"[{idx}]"
            extracted, extracted_path = _extract_text_from_payload_preferred_only(
                item,
                path=item_path,
                under_preferred_path=under_preferred_path,
                preferred_keys=preferred_keys,
            )
            if extracted:
                return extracted, extracted_path
        return None, None

    if isinstance(payload, dict):
        for key in preferred_keys:
            if key not in payload:
                continue
            item_path = f"{path}.{key}" if path else key
            extracted, extracted_path = _extract_text_from_payload_preferred_only(
                payload.get(key),
                path=item_path,
                under_preferred_path=True,
                preferred_keys=preferred_keys,
            )
            if extracted:
                return extracted, extracted_path

        for key, value in payload.items():
            item_path = f"{path}.{key}" if path else str(key)
            extracted, extracted_path = _extract_text_from_payload_preferred_only(
                value,
                path=item_path,
                under_preferred_path=under_preferred_path,
                preferred_keys=preferred_keys,
            )
            if extracted:
                return extracted, extracted_path
    return None, None


def _recover_draft_response_text_from_raw_output(
    raw_output_text: Any,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Recover the body corresponding to `draft_response_text` from raw Layer3 output.
    """
    cleaned = str(raw_output_text or "").strip()
    if not cleaned:
        return "", None

    parsed = _parse_json_from_text(cleaned)
    if parsed is not None:
        extracted, extracted_path = _extract_text_from_payload_preferred_only(parsed)
        recovered = _sanitize_human_draft_text(extracted)
        if recovered:
            return recovered, {
                "method": "parsed_json_preferred_keys",
                "source_path": extracted_path or "",
            }

    # Recover the text even if the JSON is broken, as long as the key string still remains.
    quoted_key_match = re.search(
        r'"draft_response_text"\s*:\s*"((?:\\.|[^"\\])*)"',
        cleaned,
        flags=re.DOTALL,
    )
    if quoted_key_match:
        token = quoted_key_match.group(1)
        try:
            candidate = json.loads(f'"{token}"')
        except Exception:
            candidate = token.replace('\\"', '"')
        recovered = _sanitize_human_draft_text(candidate)
        if recovered:
            return recovered, {
                "method": "regex_draft_response_text_quoted",
                "source_path": "draft_response_text",
            }

    unquoted_key_match = re.search(
        r"draft_response_text\s*[:=]\s*(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if unquoted_key_match:
        candidate = _sanitize_human_draft_text(unquoted_key_match.group(1))
        if candidate:
            return candidate, {
                "method": "regex_draft_response_text_unquoted",
                "source_path": "draft_response_text",
            }

    looks_jsonish = cleaned.startswith("{") or cleaned.startswith("[") or cleaned.startswith("```")
    if not looks_jsonish:
        candidate = _sanitize_human_draft_text(cleaned)
        if candidate:
            return candidate, {
                "method": "raw_plain_text_fallback",
                "source_path": "<raw_text>",
            }

    return "", None


def _normalize_assistant_output_text(raw_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    When the generation result is JSON, extract and return the body candidate (`response`, `text`, etc.).
    """

    def _flatten_output_whitespace(text: Any) -> str:
        t = str(text or "")
        if not t:
            return ""
        # Collapse both escaped newlines ("\\n") and literal newlines into a single space.
        t = t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n").replace("\\t", " ")
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"[ \t]*\n+[ \t]*", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        # Remove spaces immediately after punctuation for cleaner sentence formatting.
        t = re.sub(r"([。．！？])\s+", r"\1", t)
        return t

    cleaned = (raw_text or "").strip()
    parsed = _parse_json_from_text(cleaned)
    if parsed is None:
        return _flatten_output_whitespace(cleaned), None

    if isinstance(parsed, dict) and not parsed:
        return "", {
            "raw_format": "json",
            "extracted_text_path": "",
            "normalized": False,
            "empty_payload": True,
        }
    if isinstance(parsed, list) and not parsed:
        return "", {
            "raw_format": "json",
            "extracted_text_path": "",
            "normalized": False,
            "empty_payload": True,
        }

    extracted, extracted_path = _extract_text_from_payload(parsed)
    if extracted:
        return _flatten_output_whitespace(extracted), {
            "raw_format": "json",
            "extracted_text_path": extracted_path or "",
            "normalized": True,
        }

    # If JSON is parsed but no body candidate is found, treat it as empty and trigger regeneration.
    return "", {
        "raw_format": "json",
        "extracted_text_path": "",
        "normalized": False,
        "no_text_extracted": True,
    }


def _summarize_assistant_generation_exception(exc: Exception) -> Dict[str, Any]:
    error_type = exc.__class__.__name__
    error_reason = str(exc).strip() or error_type
    lowered = f"{error_type} {error_reason}".lower()
    timeout = any(marker in lowered for marker in ("timeout", "timed out", "time out"))
    return {
        "error_type": error_type,
        "error_reason": error_reason,
        "timeout": timeout,
    }


def extract_features_rule(user_text: str, state: DialogueState, cfg: PlannerConfig) -> PlannerFeatures:
    """
    Extract rule-based features from the input utterance.
    - Score `resistance` / `change_talk` from marker lists (to be extended with real logs later)
    - Estimate `novelty` using bi-gram similarity plus numbers/proper-term appearances
    - If `allow_reflect_override` is True, relax the cap on consecutive reflections
    """
    user_is_question = bool(_RE_QUESTION.search(user_text))
    user_requests_info = _detect_info_request(user_text)
    has_permission = _detect_permission(user_text) if state.info_mode == InfoMode.WAITING_PERMISSION else None

    resistance = _score_from_markers(user_text, _RESIST_MARKERS)
    discord = _score_from_markers(user_text, _DISCORD_MARKERS, cap=3)
    change_talk = _score_from_markers(user_text, _CHANGE_TALK_MARKERS)

    novelty = _estimate_novelty(user_text, state.last_user_text)
    topic_shift = novelty >= 0.85 and len(re.sub(r"\s+", "", user_text)) >= 15
    clean_len = len(re.sub(r"\s+", "", user_text))
    is_short_reply = clean_len <= 12

    need_summary = _compute_need_summary(state=state, topic_shift=topic_shift, cfg=cfg)

    allow_reflect_override = (
        change_talk >= cfg.allow_override_change_talk
        or resistance >= cfg.allow_override_resistance
        or novelty >= cfg.allow_override_novelty
    )

    return PlannerFeatures(
        user_is_question=user_is_question,
        user_requests_info=user_requests_info,
        has_permission=has_permission,
        resistance=resistance,
        change_talk=change_talk,
        novelty=novelty,
        is_short_reply=is_short_reply,
        topic_shift=topic_shift,
        need_summary=need_summary,
        allow_reflect_override=allow_reflect_override,
        discord=discord,
    )


def extract_features(user_text: str, state: DialogueState, cfg: PlannerConfig) -> PlannerFeatures:
    """
    Backward-compatible alias that calls the rule-based feature extractor.
    """
    return extract_features_rule(user_text, state, cfg)


@dataclass
class RuleBasedFeatureExtractor:
    """
    Wrapper around the existing rule-based feature extractor.
    """

    def extract(
        self,
        *,
        user_text: str,
        state: DialogueState,
        cfg: PlannerConfig,
        history: List[Tuple[str, str]],
    ) -> Tuple[PlannerFeatures, Dict[str, Any]]:
        features = extract_features_rule(user_text, state, cfg)
        return features, {"method": "rule"}


@dataclass
class LLMFeatureExtractor:
    """
    Extractor that scores resistance / change talk / novelty and related features with an LLM.
    Use the rule-based version as both fallback and baseline.
    """

    llm: LLMClient
    temperature: float = 0.2
    max_history_turns: int = 6
    rule_fallback: RuleBasedFeatureExtractor = field(default_factory=RuleBasedFeatureExtractor)
    novelty_neutral_default: float = 0.5

    def extract(
        self,
        *,
        user_text: str,
        state: DialogueState,
        cfg: PlannerConfig,
        history: List[Tuple[str, str]],
    ) -> Tuple[PlannerFeatures, Dict[str, Any]]:
        base_features, base_debug = self.rule_fallback.extract(
            user_text=user_text, state=state, cfg=cfg, history=history
        )

        dialogue = _history_to_dialogue(history, max_turns=self.max_history_turns)
        last_assistant_text = ""
        for role, text in reversed(history):
            if role == "assistant":
                last_assistant_text = str(text or "").strip()
                break
        permission_priority = state.info_mode == InfoMode.WAITING_PERMISSION
        system = (
            "You are an analyst that estimates numeric MI dialogue features from counselor-client conversations.\n"
            "Estimate resistance (0-1), discord (0-1), change talk orientation (0-1), and novelty (0-1).\n"
            "Also return importance_estimate and confidence_estimate (0-1) when inferable; otherwise use null.\n"
            "Return a JSON object only. Example:\n"
            '{"resistance":0.2,"discord":0.1,"change_talk":0.6,"novelty":0.4,"importance_estimate":0.7,"confidence_estimate":0.4,"user_is_question":false,"user_requests_info":false,"has_permission":null,"note":"brief rationale"}\n'
            "- Resistance: closer to 1.0 when the client shows pushback, refusal, or giving up.\n"
            "- Score resistance slightly high (+0.2 from your default instinct) and clip at 1.0.\n"
            "- Do not underestimate resistance. Hesitation, low self-efficacy, or reluctance can justify 0.3-0.5.\n"
            "- Clear refusal or rejection (for example: impossible, cannot, pointless) should usually be 0.6 or above.\n"
            "- Discord: closer to 1.0 when the relationship feels misaligned, pressured, blamed, or oppositional.\n"
            "- Change talk: closer to 1.0 when the client expresses wanting change, trying something, or acting in line with values.\n"
            "- Novelty: closer to 1.0 when the current utterance adds meaningfully new information versus the recent context.\n"
            "- user_is_question: true when the client utterance is a question.\n"
            "- user_requests_info: true when the client is clearly asking for information, advice, or options.\n"
            "- has_permission: true/false when permission intent is inferable, otherwise null.\n"
            "- Examples of true: \"A little would be okay\", \"I'd like to hear something manageable\".\n"
            "- Examples of false: \"I don't want to hear that yet\", \"I'd rather not do that now\".\n"
            "- If both acceptance and reservation are present, prioritize true when receiving the information still seems primary.\n"
            "- When info_mode is WAITING_PERMISSION, prioritize the has_permission judgment over everything else.\n"
            "- In WAITING_PERMISSION, clear acceptance such as yes / please / that's okay / OK must produce has_permission=true.\n"
            "- Even if the acceptance is conditional or tentative, set has_permission=true when a willingness to receive is readable.\n"
            "- In WAITING_PERMISSION, clear refusal such as no / not yet / stop must produce has_permission=false.\n"
            "- When clear acceptance or refusal is present, do not return has_permission=null.\n"
        )
        prev_assistant_text = last_assistant_text if last_assistant_text else "(none)"
        user = (
            f"[Current phase]\n{state.phase.value}\n"
            f"[Current info-sharing mode]\n{state.info_mode.value}\n"
            f"[Permission judgment prioritized]\n{'yes' if permission_priority else 'no'}\n"
            f"[Previous counselor utterance]\n{prev_assistant_text}\n"
            f"[Recent dialogue in chronological order]\n{dialogue}\n"
            f"[Current client utterance]\n{user_text}\n"
            "Return JSON only."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label="layer1:feature_extractor",
        )
        parsed = _parse_json_from_text(raw)
        overlay_source: Dict[str, Any] = parsed if isinstance(parsed, dict) else {}
        overlay = _extract_phase_feature_overlay(overlay_source)
        novelty_source = "llm"
        novelty_raw = overlay.get("novelty")
        if novelty_raw in (None, ""):
            # Even if `novelty` is missing on an LLM success path, do not fall back to the old heuristic estimate.
            overlay["novelty"] = _clamp01(self.novelty_neutral_default, 0.5)
            novelty_source = "neutral_default_missing"
        else:
            try:
                overlay["novelty"] = _clamp01(float(novelty_raw), self.novelty_neutral_default)
            except Exception:
                overlay["novelty"] = _clamp01(self.novelty_neutral_default, 0.5)
                novelty_source = "neutral_default_invalid"
        merged, merge_debug = _merge_feature_overlay(
            base_features=base_features,
            overlay=overlay,
            user_text=user_text,
            state=state,
            cfg=cfg,
        )

        debug: Dict[str, Any] = {
            "method": "llm",
            "raw_output": raw,
            "parsed": overlay_source,
            "parsed_overlay": overlay,
            "fallback": base_debug,
            "merge_debug": merge_debug,
            "novelty_source": novelty_source,
            "novelty_neutral_default": _clamp01(self.novelty_neutral_default, 0.5),
            "request_label": "layer1:feature_extractor",
        }
        return merged, debug


# ----------------------------
# Planning (policy: forced ranker adoption + action mask)
# ----------------------------
_RANKER_ACTION_SPACE: Tuple[MainAction, ...] = (
    MainAction.REFLECT_SIMPLE,
    MainAction.REFLECT_COMPLEX,
    MainAction.REFLECT_DOUBLE,
    MainAction.QUESTION,
    MainAction.SCALING_QUESTION,
    MainAction.SUMMARY,
    MainAction.CLARIFY_PREFERENCE,
    MainAction.ASK_PERMISSION_TO_SHARE_INFO,
    MainAction.PROVIDE_INFO,
)


_QUESTION_LIKE_ACTIONS: Tuple[MainAction, ...] = (
    MainAction.QUESTION,
    MainAction.SCALING_QUESTION,
    MainAction.CLARIFY_PREFERENCE,
    MainAction.ASK_PERMISSION_TO_SHARE_INFO,
    MainAction.PROVIDE_INFO,
)
_INFO_SHARE_PHASE_ORDER: Tuple[Phase, ...] = (
    Phase.GREETING,
    Phase.PURPOSE_CONFIRMATION,
    Phase.CURRENT_STATUS_CHECK,
    Phase.FOCUSING_TARGET_BEHAVIOR,
    Phase.IMPORTANCE_PROMOTION,
    Phase.CONFIDENCE_PROMOTION,
    Phase.NEXT_STEP_DECISION,
    Phase.REVIEW_REFLECTION,
    Phase.CLOSING,
)
_CLARIFY_PREFERENCE_PRIORITY_INTERVAL_TURNS = 10


def _estimate_ambivalence_score(features: PlannerFeatures) -> float:
    """
    Approximate ambivalence (coexistence of change-side and sustain-side signals) on a 0-1 scale.
    - change_talk: change-side signal
    - sustain side: the stronger of resistance / discord
    """
    change_side = _clamp01(features.change_talk, 0.0)
    sustain_side = max(_clamp01(features.resistance, 0.0), _clamp01(features.discord, 0.0))
    return _clamp01((change_side + sustain_side) / 2.0, 0.0)


def _should_prioritize_clarify_preference_for_focus_choice(
    *,
    state: DialogueState,
    allowed_actions: Sequence[MainAction],
    focus_choice_context: Optional[Mapping[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    allowed_set = set(allowed_actions)
    if state.phase not in {Phase.PURPOSE_CONFIRMATION, Phase.CURRENT_STATUS_CHECK}:
        return False, {"enabled": False, "reason": "phase_not_eligible"}
    if MainAction.CLARIFY_PREFERENCE not in allowed_set or MainAction.QUESTION not in allowed_set:
        return False, {"enabled": False, "reason": "required_actions_not_allowed"}

    normalized_context = _normalize_focus_choice_hint(focus_choice_context)
    candidate_topics = list(normalized_context.get("candidate_topics") or [])
    if normalized_context.get("explicit_preference_present"):
        return False, {
            "enabled": False,
            "reason": "explicit_preference_present",
            "candidate_topics": candidate_topics,
        }
    if not normalized_context.get("needed"):
        return False, {
            "enabled": False,
            "reason": "focus_choice_not_needed",
            "candidate_topics": candidate_topics,
        }
    if len(candidate_topics) < 2:
        return False, {
            "enabled": False,
            "reason": "candidate_topics_lt_2",
            "candidate_topics": candidate_topics,
        }

    return True, {
        "enabled": True,
        "reason": "layer1_focus_choice_needed",
        "candidate_topics": candidate_topics,
        "focus_choice_reason": _normalize_slot_text(normalized_context.get("reason")),
    }


def _should_prioritize_clarify_preference_over_question(
    *,
    state: DialogueState,
    features: PlannerFeatures,
    cfg: PlannerConfig,
    allowed_actions: Sequence[MainAction],
) -> Tuple[bool, Dict[str, Any]]:
    allowed_set = set(allowed_actions)
    if MainAction.CLARIFY_PREFERENCE not in allowed_set or MainAction.QUESTION not in allowed_set:
        return False, {"enabled": False, "reason": "required_actions_not_allowed"}

    ambivalence_score = _estimate_ambivalence_score(features)
    ambivalence_threshold = _clamp01(cfg.ambivalence_reflection_threshold, 0.68)
    if ambivalence_score < ambivalence_threshold:
        return False, {
            "enabled": False,
            "reason": "ambivalence_below_threshold",
            "ambivalence_score": round(float(ambivalence_score), 3),
            "ambivalence_threshold": round(float(ambivalence_threshold), 3),
        }

    interval_turns = max(2, int(_CLARIFY_PREFERENCE_PRIORITY_INTERVAL_TURNS))
    cadence_due = state.turn_index > 0 and (state.turn_index % interval_turns == 0)
    if not cadence_due:
        return False, {
            "enabled": False,
            "reason": "cadence_not_due",
            "turn_index": int(state.turn_index),
            "interval_turns": int(interval_turns),
            "ambivalence_score": round(float(ambivalence_score), 3),
            "ambivalence_threshold": round(float(ambivalence_threshold), 3),
        }

    return True, {
        "enabled": True,
        "reason": "high_ambivalence_cadence_due",
        "turn_index": int(state.turn_index),
        "interval_turns": int(interval_turns),
        "ambivalence_score": round(float(ambivalence_score), 3),
        "ambivalence_threshold": round(float(ambivalence_threshold), 3),
    }


def _should_prioritize_reflect_double_for_ambivalence(
    *,
    state: DialogueState,
    features: PlannerFeatures,
    cfg: PlannerConfig,
    allowed_actions: Sequence[MainAction],
) -> Tuple[bool, Dict[str, Any]]:
    allowed_set = set(allowed_actions)
    if MainAction.REFLECT_DOUBLE not in allowed_set:
        return False, {"enabled": False, "reason": "reflect_double_not_allowed"}

    ambivalence_score = _estimate_ambivalence_score(features)
    ambivalence_threshold = _clamp01(cfg.ambivalence_reflection_threshold, 0.68)
    if ambivalence_score < ambivalence_threshold:
        return False, {
            "enabled": False,
            "reason": "ambivalence_below_threshold",
            "ambivalence_score": round(float(ambivalence_score), 3),
            "ambivalence_threshold": round(float(ambivalence_threshold), 3),
        }

    interval_turns = max(2, int(_CLARIFY_PREFERENCE_PRIORITY_INTERVAL_TURNS))
    cadence_due = state.turn_index > 0 and (state.turn_index % interval_turns == 0)
    if not cadence_due:
        return False, {
            "enabled": False,
            "reason": "cadence_not_due",
            "turn_index": int(state.turn_index),
            "interval_turns": int(interval_turns),
            "ambivalence_score": round(float(ambivalence_score), 3),
            "ambivalence_threshold": round(float(ambivalence_threshold), 3),
        }

    return True, {
        "enabled": True,
        "reason": "high_ambivalence_cadence_due",
        "turn_index": int(state.turn_index),
        "interval_turns": int(interval_turns),
        "ambivalence_score": round(float(ambivalence_score), 3),
        "ambivalence_threshold": round(float(ambivalence_threshold), 3),
    }


def _normalize_ranker_action_space(actions: Optional[Sequence[MainAction]]) -> List[MainAction]:
    base = list(actions) if actions else list(_RANKER_ACTION_SPACE)
    normalized: List[MainAction] = []
    seen = set()
    for item in base:
        action = item if isinstance(item, MainAction) else coerce_main_action(item)
        if action is None:
            continue
        if action == MainAction.REFLECT:
            action = MainAction.REFLECT_COMPLEX
        if action in seen:
            continue
        if action not in _RANKER_ACTION_SPACE:
            continue
        seen.add(action)
        normalized.append(action)
    if not normalized:
        return [MainAction.REFLECT_COMPLEX]
    return normalized


def _neutralize_allowed_actions_for_ranker(
    actions: Optional[Sequence[MainAction]],
) -> List[MainAction]:
    normalized = _normalize_ranker_action_space(actions)
    allowed_set = set(normalized)
    neutralized = [action for action in _RANKER_ACTION_SPACE if action in allowed_set]
    if neutralized:
        return neutralized
    return normalized


def _prioritize_ask_permission_for_ranker(
    *,
    actions: Optional[Sequence[MainAction]],
    state: DialogueState,
    features: PlannerFeatures,
    info_gate_open: bool,
) -> List[MainAction]:
    normalized = _normalize_ranker_action_space(actions)
    if not normalized:
        return normalized
    if not info_gate_open or not features.user_requests_info:
        return normalized
    if state.info_mode == InfoMode.READY_TO_PROVIDE or features.has_permission is True:
        return normalized
    if state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is None:
        return normalized
    if state.phase == Phase.CLOSING:
        prioritized_head = (
            MainAction.REFLECT_COMPLEX,
            MainAction.REFLECT_SIMPLE,
            MainAction.QUESTION,
            MainAction.ASK_PERMISSION_TO_SHARE_INFO,
            MainAction.PROVIDE_INFO,
        )
    else:
        prioritized_head = (
            MainAction.REFLECT_DOUBLE,
            MainAction.QUESTION,
            MainAction.REFLECT_COMPLEX,
            MainAction.ASK_PERMISSION_TO_SHARE_INFO,
            MainAction.PROVIDE_INFO,
        )
    prioritized = [action for action in prioritized_head if action in normalized]
    prioritized.extend(action for action in normalized if action not in prioritized)
    return prioritized


def _is_info_share_phase_gate_open(phase: Phase) -> bool:
    return _INFO_SHARE_PHASE_ORDER.index(phase) >= _INFO_SHARE_PHASE_ORDER.index(
        Phase.CURRENT_STATUS_CHECK
    )


def _build_mask_fallback_priority(
    *,
    state: DialogueState,
    features: PlannerFeatures,
    cfg: PlannerConfig,
    allowed_actions: Sequence[MainAction],
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[MainAction], Dict[str, Any]]:
    allowed_set = set(allowed_actions)
    priority: List[MainAction] = []
    reasons: List[str] = []
    focus_choice_priority_enabled, focus_choice_priority_debug = (
        _should_prioritize_clarify_preference_for_focus_choice(
            state=state,
            allowed_actions=allowed_actions,
            focus_choice_context=focus_choice_context,
        )
    )
    clarify_priority_enabled, clarify_priority_debug = _should_prioritize_clarify_preference_over_question(
        state=state,
        features=features,
        cfg=cfg,
        allowed_actions=allowed_actions,
    )
    reflect_double_priority_enabled, reflect_double_priority_debug = (
        _should_prioritize_reflect_double_for_ambivalence(
            state=state,
            features=features,
            cfg=cfg,
            allowed_actions=allowed_actions,
        )
    )

    def push(action: MainAction, reason: str) -> None:
        if action not in allowed_set:
            return
        if action in priority:
            return
        priority.append(action)
        reasons.append(f"{action.value}:{reason}")

    summary_mul, summary_stage = _summary_cadence_multiplier(
        turns_since_summary=state.turns_since_summary,
        cfg=cfg,
        need_summary=features.need_summary,
    )
    info_request_priority_mode = (
        _is_info_share_phase_gate_open(state.phase)
        and features.user_requests_info
        and not (state.info_mode == InfoMode.READY_TO_PROVIDE or features.has_permission is True)
        and not (state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is None)
    )

    if state.phase == Phase.CLOSING:
        push(MainAction.REFLECT_COMPLEX, "closing_phase_reflect_priority")
        push(MainAction.REFLECT_SIMPLE, "closing_phase_reflect_priority")
        push(MainAction.QUESTION, "closing_phase_reflect_priority")
    if state.info_mode == InfoMode.READY_TO_PROVIDE or features.has_permission is True:
        push(MainAction.PROVIDE_INFO, "permission_ready")
    if state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is None:
        push(MainAction.CLARIFY_PREFERENCE, "waiting_permission_unclear")
    if focus_choice_priority_enabled:
        push(MainAction.CLARIFY_PREFERENCE, "focus_choice_needed")
    if info_request_priority_mode:
        push(MainAction.REFLECT_DOUBLE, "info_request_ranker_priority")
        push(MainAction.QUESTION, "info_request_ranker_priority")
        push(MainAction.REFLECT_COMPLEX, "info_request_ranker_priority")
        push(MainAction.ASK_PERMISSION_TO_SHARE_INFO, "info_request_ranker_priority")
        push(MainAction.PROVIDE_INFO, "info_request_ranker_priority")
    if features.user_requests_info:
        push(MainAction.ASK_PERMISSION_TO_SHARE_INFO, "user_requests_info")

    if reflect_double_priority_enabled:
        push(MainAction.REFLECT_DOUBLE, "ambivalence_priority_window")
    if clarify_priority_enabled:
        push(MainAction.CLARIFY_PREFERENCE, "ambivalence_priority_window")

    if features.need_summary or summary_mul >= 1.2:
        push(MainAction.SUMMARY, f"summary_{summary_stage}")
    if state.r_since_q >= 2:
        if _is_scaling_question_phase(state.phase):
            push(MainAction.SCALING_QUESTION, "rrq_scaling_question_due")
        push(MainAction.QUESTION, "rrq_question_due")
    if features.is_short_reply:
        if _is_scaling_question_phase(state.phase):
            push(MainAction.SCALING_QUESTION, "short_reply_scaling_phase")
        else:
            push(MainAction.QUESTION, "short_reply")
    if features.user_is_question:
        push(MainAction.REFLECT_COMPLEX, "user_question_reflect_first")
    if features.resistance >= 0.6:
        push(MainAction.REFLECT_COMPLEX, "high_resistance")
    if features.change_talk >= 0.55 or features.novelty >= 0.55:
        push(MainAction.REFLECT_COMPLEX, "change_talk_or_novelty")

    for action in (
        MainAction.REFLECT_COMPLEX,
        MainAction.SCALING_QUESTION,
        MainAction.QUESTION,
        MainAction.SUMMARY,
        MainAction.REFLECT_SIMPLE,
        MainAction.REFLECT_DOUBLE,
        MainAction.CLARIFY_PREFERENCE,
        MainAction.ASK_PERMISSION_TO_SHARE_INFO,
        MainAction.PROVIDE_INFO,
    ):
        push(action, "base_order")

    if not priority:
        priority = [MainAction.REFLECT_COMPLEX]
        reasons = ["REFLECT_COMPLEX:mask_empty_fallback"]

    return priority, {
        "summary_cadence_multiplier": round(float(summary_mul), 3),
        "summary_cadence_stage": summary_stage,
        "info_request_priority_mode": bool(info_request_priority_mode),
        "priority_reasons": reasons,
        "focus_choice_priority": focus_choice_priority_debug,
        "clarify_preference_priority": clarify_priority_debug,
        "reflect_double_priority": reflect_double_priority_debug,
    }


def _should_force_clarify_preference_for_focus_choice(
    *,
    state: DialogueState,
    action: MainAction,
    allowed_actions: Sequence[MainAction],
    focus_choice_context: Optional[Mapping[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    priority_enabled, priority_debug = _should_prioritize_clarify_preference_for_focus_choice(
        state=state,
        allowed_actions=allowed_actions,
        focus_choice_context=focus_choice_context,
    )
    if not priority_enabled:
        return False, {
            "applied": False,
            "reason": priority_debug.get("reason"),
            "priority_debug": priority_debug,
        }
    if action != MainAction.QUESTION:
        return False, {
            "applied": False,
            "reason": "selected_action_not_question",
            "priority_debug": priority_debug,
            "from_action": action.value,
        }
    return True, {
        "applied": True,
        "reason": "focus_choice_needed_before_question",
        "priority_debug": priority_debug,
        "from_action": action.value,
        "to_action": MainAction.CLARIFY_PREFERENCE.value,
    }


def _is_review_reflection_first_turn(state: DialogueState) -> bool:
    return state.phase == Phase.REVIEW_REFLECTION and state.phase_turns <= 0


def _is_review_reflection_second_turn(state: DialogueState) -> bool:
    return state.phase == Phase.REVIEW_REFLECTION and state.phase_turns == 1


def _is_closing_first_turn(state: DialogueState) -> bool:
    return state.phase == Phase.CLOSING and state.phase_turns <= 0


def _is_scaling_question_phase(phase: Phase) -> bool:
    return phase in {Phase.IMPORTANCE_PROMOTION, Phase.CONFIDENCE_PROMOTION}


_SCALE_FOLLOWUP_STEP_REASON = "reason"
_SCALE_FOLLOWUP_STEP_PLUS_ONE = "plus_one"


def _normalize_scale_followup_step(value: Any) -> str:
    raw = _normalize_slot_text(value).lower().replace("-", "_")
    if raw in {"reason", "reason_probe", "why_not_zero"}:
        return _SCALE_FOLLOWUP_STEP_REASON
    if raw in {"plus_one", "up_one", "one_point_up"}:
        return _SCALE_FOLLOWUP_STEP_PLUS_ONE
    return ""


def _clear_scale_followup_runtime_state(state: DialogueState) -> None:
    state.scale_followup_pending_phase = None
    state.scale_followup_score = None
    state.scale_followup_pending_step = None
    state.scale_followup_last_asked_step = None


def _effective_scale_followup_pending_step(
    *,
    phase: Phase,
    score: Optional[float],
    pending_step: Any,
) -> str:
    step = _normalize_scale_followup_step(pending_step)
    normalized_score = _normalize_scale_score_0_10(score)
    if step != _SCALE_FOLLOWUP_STEP_PLUS_ONE or normalized_score is None:
        return step
    # Do not continue `plus_one` when importance is 8+ and confidence is already positive.
    if phase == Phase.IMPORTANCE_PROMOTION and normalized_score >= 8.0:
        return _SCALE_FOLLOWUP_STEP_REASON
    if phase == Phase.CONFIDENCE_PROMOTION and normalized_score > 0.0:
        return _SCALE_FOLLOWUP_STEP_REASON
    return step


def _current_scale_followup_step(state: DialogueState) -> str:
    if not _is_scaling_question_phase(state.phase):
        return ""
    if state.scale_followup_pending_phase != state.phase:
        return ""
    return _effective_scale_followup_pending_step(
        phase=state.phase,
        score=state.scale_followup_score,
        pending_step=state.scale_followup_pending_step,
    )


def _is_scale_followup_pending(state: DialogueState) -> bool:
    if not _is_scaling_question_phase(state.phase):
        return False
    if state.scale_followup_pending_phase != state.phase:
        return False
    if state.scale_followup_score is None:
        return False
    if not _current_scale_followup_step(state):
        return False
    phase_key = state.phase.name
    return not bool((state.scale_followup_done_in_phase or {}).get(phase_key, False))


def _is_scale_followup_incomplete(state: DialogueState) -> bool:
    if not _is_scaling_question_phase(state.phase):
        return False
    phase_key = state.phase.name
    if bool((state.scale_followup_done_in_phase or {}).get(phase_key, False)):
        return False
    if _is_scale_followup_pending(state):
        return True
    last_action = state.last_actions[-1] if state.last_actions else None
    if last_action == MainAction.SCALING_QUESTION:
        return True
    if (
        last_action == MainAction.QUESTION
        and _normalize_scale_followup_step(state.scale_followup_last_asked_step)
    ):
        return True
    return False


def _format_scale_score_text(score: Optional[float]) -> str:
    if score is None:
        return ""
    rounded = round(float(score), 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.1f}".rstrip("0").rstrip(".")


def _normalize_scale_score_0_10(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    return float(max(0.0, min(10.0, score)))


def _choose_scale_followup_step_after_score(*, phase: Phase, score: float) -> str:
    normalized_score = _normalize_scale_score_0_10(score) or 0.0
    if normalized_score <= 0.0:
        return _SCALE_FOLLOWUP_STEP_PLUS_ONE
    return _SCALE_FOLLOWUP_STEP_REASON


def _should_schedule_plus_one_after_reason(*, phase: Phase, score: Optional[float]) -> bool:
    normalized_score = _normalize_scale_score_0_10(score)
    if normalized_score is None or normalized_score <= 0.0:
        return False
    # For confidence, usually stop at the reason check (if the score is 0, branch to plus-one from the start).
    if phase == Phase.CONFIDENCE_PROMOTION:
        return False
    # Skip `plus_one` when importance is 8 or higher.
    if phase == Phase.IMPORTANCE_PROMOTION and normalized_score >= 8.0:
        return False
    return True


def _update_scale_followup_state_from_user_turn(state: DialogueState, user_text: str) -> Dict[str, Any]:
    debug: Dict[str, Any] = {
        "phase": state.phase.value,
        "last_action": (state.last_actions[-1].value if state.last_actions else None),
        "pending_step_before": _normalize_scale_followup_step(state.scale_followup_pending_step) or None,
        "last_asked_step_before": _normalize_scale_followup_step(state.scale_followup_last_asked_step) or None,
    }
    phase_key = state.phase.name

    if state.scale_followup_pending_phase is not None and state.scale_followup_pending_phase != state.phase:
        debug["cleared_by_phase_change"] = {
            "from_phase": state.scale_followup_pending_phase.value,
            "to_phase": state.phase.value,
        }
        _clear_scale_followup_runtime_state(state)

    if not _is_scaling_question_phase(state.phase):
        if state.scale_followup_pending_phase is not None:
            debug["cleared_non_scaling_phase"] = True
        _clear_scale_followup_runtime_state(state)
        return debug

    if bool((state.scale_followup_done_in_phase or {}).get(phase_key, False)):
        _clear_scale_followup_runtime_state(state)
        debug["skip_due_to_done_in_phase"] = True
        return debug

    last_action = state.last_actions[-1] if state.last_actions else None
    if last_action == MainAction.SCALING_QUESTION:
        # Right after a phase transition (`phase_turns == 0`), treat the preceding SCALING_QUESTION
        # as a question that belonged to the previous phase, so do not create a follow-up in the current phase.
        if state.phase_turns <= 0:
            _clear_scale_followup_runtime_state(state)
            state.scale_followup_done_in_phase[phase_key] = True
            debug["pending_set"] = False
            debug["reason"] = "last_scaling_question_not_in_current_phase"
            debug["done_marked"] = True
            return debug

        score = _extract_scale_0_10_value(user_text)
        if score is None:
            _clear_scale_followup_runtime_state(state)
            state.scale_followup_done_in_phase[phase_key] = True
            debug["pending_set"] = False
            debug["reason"] = "score_not_found_in_user_reply"
            debug["done_marked"] = True
            return debug

        normalized_score = _normalize_scale_score_0_10(score) or 0.0
        pending_step = _choose_scale_followup_step_after_score(
            phase=state.phase,
            score=normalized_score,
        )
        state.scale_followup_pending_phase = state.phase
        state.scale_followup_score = normalized_score
        state.scale_followup_pending_step = pending_step
        state.scale_followup_last_asked_step = None
        debug["pending_set"] = True
        debug["pending_step"] = pending_step
        debug["score"] = state.scale_followup_score
        debug["reason"] = "pending_reason_after_scale" if pending_step == _SCALE_FOLLOWUP_STEP_REASON else "skip_reason_pending_plus_one"
        debug["reason_marker_present"] = any(
            marker in _normalize_slot_text(user_text) for marker in _SLOT_SCALE_REASON_MARKERS
        )
        return debug

    if last_action == MainAction.QUESTION:
        asked_step = _normalize_scale_followup_step(state.scale_followup_last_asked_step)
        if asked_step == _SCALE_FOLLOWUP_STEP_REASON:
            followup_score = state.scale_followup_score
            if followup_score is None:
                _clear_scale_followup_runtime_state(state)
                state.scale_followup_done_in_phase[phase_key] = True
                debug["pending_set"] = False
                debug["reason"] = "reason_followup_missing_score"
                debug["done_marked"] = True
                return debug
            if _should_schedule_plus_one_after_reason(
                phase=state.phase,
                score=followup_score,
            ):
                state.scale_followup_pending_phase = state.phase
                state.scale_followup_pending_step = _SCALE_FOLLOWUP_STEP_PLUS_ONE
                state.scale_followup_last_asked_step = None
                debug["pending_set"] = True
                debug["pending_step"] = _SCALE_FOLLOWUP_STEP_PLUS_ONE
                debug["score"] = followup_score
                debug["reason"] = "reason_followup_completed_pending_plus_one"
                return debug

            _clear_scale_followup_runtime_state(state)
            state.scale_followup_done_in_phase[phase_key] = True
            debug["pending_set"] = False
            debug["score"] = followup_score
            debug["reason"] = "reason_followup_completed_done"
            debug["done_marked"] = True
            return debug
        if asked_step == _SCALE_FOLLOWUP_STEP_PLUS_ONE:
            _clear_scale_followup_runtime_state(state)
            state.scale_followup_done_in_phase[phase_key] = True
            debug["pending_set"] = False
            debug["reason"] = "plus_one_followup_completed"
            debug["done_marked"] = True
            return debug

    debug["pending_unchanged"] = _is_scale_followup_pending(state)
    return debug


def _scaling_estimate_for_phase(features: PlannerFeatures, phase: Phase) -> Optional[float]:
    if phase == Phase.IMPORTANCE_PROMOTION:
        return features.importance_estimate
    if phase == Phase.CONFIDENCE_PROMOTION:
        return features.confidence_estimate
    return None


def _scaling_slot_key_for_phase(phase: Phase) -> str:
    if phase == Phase.IMPORTANCE_PROMOTION:
        return "importance_scale"
    if phase == Phase.CONFIDENCE_PROMOTION:
        return "confidence_scale"
    return ""


def _should_enable_scaling_question(
    *,
    state: DialogueState,
    features: PlannerFeatures,
    phase_recent_actions: Sequence[MainAction],
) -> Tuple[bool, str]:
    if not _is_scaling_question_phase(state.phase):
        return False, "phase_not_scaling_question_phase"
    if MainAction.SCALING_QUESTION in phase_recent_actions:
        return False, "scaling_question_already_used_in_phase"

    scale_estimate = _scaling_estimate_for_phase(features, state.phase)
    if state.phase_turns <= 0:
        if scale_estimate is not None and scale_estimate >= 0.5:
            return True, "phase_entry_scale_ge_0_5"
        return False, "phase_entry_scale_lt_0_5"
    if state.phase_turns >= 2:
        return True, "phase_turn_gte_3"
    return False, "phase_turn_lt_3"


def compute_allowed_actions(
    *,
    state: DialogueState,
    features: PlannerFeatures,
    cfg: PlannerConfig,
    first_turn_hint: Optional[FirstTurnHint] = None,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    info_gate_open = _is_info_share_phase_gate_open(state.phase)

    allowed = _normalize_ranker_action_space(None)
    disallowed: Dict[str, str] = {}
    mask_steps: List[str] = []

    def drop(action: MainAction, reason: str) -> None:
        if action not in allowed:
            return
        allowed.remove(action)
        disallowed[action.value] = reason

    if state.phase == Phase.REVIEW_REFLECTION:
        review_first_turn = _is_review_reflection_first_turn(state)
        review_info_request_priority = (not review_first_turn) and bool(features.user_requests_info)
        if review_first_turn:
            allowed = [MainAction.QUESTION]
            mask_steps.append("review_first_turn_question_only")
        elif review_info_request_priority:
            if (state.info_mode == InfoMode.READY_TO_PROVIDE or features.has_permission is True) and info_gate_open:
                allowed = [MainAction.PROVIDE_INFO]
                mask_steps.append("permission_ready_provide_info_only")
            elif state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is None:
                allowed = [MainAction.CLARIFY_PREFERENCE]
                mask_steps.append("waiting_permission_unclear_clarify_only")
            else:
                allowed = _normalize_ranker_action_space(None)
                can_provide_info = (
                    state.info_mode == InfoMode.READY_TO_PROVIDE
                    or features.has_permission is True
                )
                if not can_provide_info:
                    drop(MainAction.PROVIDE_INFO, "permission_not_ready")
            mask_steps.append("review_info_request_priority")
        elif _is_review_reflection_second_turn(state):
            allowed = [MainAction.REFLECT_COMPLEX]
            mask_steps.append("review_second_turn_reflect_complex_only")
        else:
            allowed = [MainAction.REFLECT_COMPLEX]
            mask_steps.append("review_feedback_turn_reflect_complex_only")
    elif _is_closing_first_turn(state):
        allowed = [MainAction.REFLECT_SIMPLE, MainAction.REFLECT_COMPLEX]
        mask_steps.append("closing_first_turn_reflect_simple_or_complex_only")
    elif (state.info_mode == InfoMode.READY_TO_PROVIDE or features.has_permission is True) and info_gate_open:
        allowed = [MainAction.PROVIDE_INFO]
        mask_steps.append("permission_ready_provide_info_only")
    elif state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is None:
        allowed = [MainAction.CLARIFY_PREFERENCE]
        mask_steps.append("waiting_permission_unclear_clarify_only")
    else:
        if not info_gate_open:
            drop(MainAction.ASK_PERMISSION_TO_SHARE_INFO, "phase_gate_closed")
            drop(MainAction.PROVIDE_INFO, "phase_gate_closed")

        can_ask_permission = info_gate_open and (
            features.user_requests_info
            or state.info_mode in (InfoMode.WAITING_PERMISSION, InfoMode.READY_TO_PROVIDE)
        )
        can_provide_info = info_gate_open and (
            state.info_mode == InfoMode.READY_TO_PROVIDE
            or features.has_permission is True
        )
        if not can_ask_permission:
            drop(MainAction.ASK_PERMISSION_TO_SHARE_INFO, "missing_info_request_or_mode")
        if not can_provide_info:
            drop(MainAction.PROVIDE_INFO, "permission_not_ready")

        last_action = state.last_actions[-1] if state.last_actions else None
        recent_actions = [a for a in state.last_actions if isinstance(a, MainAction)]
        phase_recent_len = min(max(state.phase_turns, 0), len(recent_actions))
        phase_recent_actions = recent_actions[-phase_recent_len:] if phase_recent_len > 0 else []
        scale_followup_pending_turn = _is_scale_followup_pending(state)

        if scale_followup_pending_turn:
            allowed = [MainAction.QUESTION]
            mask_steps.append("scale_followup_pending_question_only")
        else:
            scaling_enabled, scaling_policy_reason = _should_enable_scaling_question(
                state=state,
                features=features,
                phase_recent_actions=phase_recent_actions,
            )
            if scaling_enabled:
                drop(MainAction.QUESTION, "scaling_question_policy_replace_question")
                mask_steps.append(f"scaling_question_enabled:{scaling_policy_reason}")
            else:
                drop(MainAction.SCALING_QUESTION, scaling_policy_reason)
                if _is_scaling_question_phase(state.phase):
                    mask_steps.append(f"scaling_question_disabled:{scaling_policy_reason}")

        allow_question_back_for_info_request = info_gate_open and features.user_requests_info
        if (
            (not scale_followup_pending_turn)
            and features.user_is_question
            and not allow_question_back_for_info_request
        ):
            drop(MainAction.QUESTION, "user_question_avoid_question_back")
            drop(MainAction.SCALING_QUESTION, "user_question_avoid_question_back")
            if state.phase in (
                Phase.GREETING,
                Phase.PURPOSE_CONFIRMATION,
                Phase.CURRENT_STATUS_CHECK,
            ):
                drop(MainAction.SUMMARY, "user_question_early_phase_no_summary")

        if (not scale_followup_pending_turn) and last_action == MainAction.CLARIFY_PREFERENCE:
            drop(MainAction.QUESTION, "post_clarify_preference_no_immediate_question")
            drop(MainAction.SCALING_QUESTION, "post_clarify_preference_no_immediate_question")

        last_clarify_index: Optional[int] = None
        for idx in range(len(recent_actions) - 1, -1, -1):
            if recent_actions[idx] == MainAction.CLARIFY_PREFERENCE:
                last_clarify_index = idx
                break
        if (not scale_followup_pending_turn) and last_clarify_index is not None:
            actions_after_last_clarify = recent_actions[last_clarify_index + 1 :]
            reflect_count_after_last_clarify = sum(
                1 for action in actions_after_last_clarify if _is_reflect_action(action)
            )
            if reflect_count_after_last_clarify < 2:
                drop(MainAction.QUESTION, "post_clarify_preference_reflect_window_lt2")
                drop(MainAction.SCALING_QUESTION, "post_clarify_preference_reflect_window_lt2")

        recent_three_actions = recent_actions[-3:]
        clarify_guard_reasons: List[str] = []
        if last_action in (MainAction.QUESTION, MainAction.SCALING_QUESTION):
            clarify_guard_reasons.append("prev_question")
        if state.phase_turns < 2:
            clarify_guard_reasons.append("phase_turns_lt_2")
        if MainAction.CLARIFY_PREFERENCE in recent_three_actions:
            clarify_guard_reasons.append("used_in_recent_3_turns")
        if clarify_guard_reasons:
            drop(
                MainAction.CLARIFY_PREFERENCE,
                "clarify_preference_guard:" + ",".join(clarify_guard_reasons),
            )

        if (
            (not scale_followup_pending_turn)
            and last_action in _QUESTION_LIKE_ACTIONS
            and not features.user_is_question
            and not features.is_short_reply
        ):
            drop(MainAction.QUESTION, "suppress_consecutive_questions")
            drop(MainAction.SCALING_QUESTION, "suppress_consecutive_questions")

        if (
            state.reflect_streak >= cfg.reflect_streak_cap_threshold
            and not features.allow_reflect_override
        ):
            drop(MainAction.REFLECT_SIMPLE, "reflect_streak_cap")
            drop(MainAction.REFLECT_COMPLEX, "reflect_streak_cap")
            drop(MainAction.REFLECT_DOUBLE, "reflect_streak_cap")

        phase_reflect_simple_or_complex = sum(
            1
            for action in phase_recent_actions
            if action in (MainAction.REFLECT, MainAction.REFLECT_SIMPLE, MainAction.REFLECT_COMPLEX)
        )
        phase_reflect_double_count = sum(
            1 for action in phase_recent_actions if action == MainAction.REFLECT_DOUBLE
        )
        if phase_reflect_simple_or_complex < cfg.double_sided_min_reflections_before_use:
            drop(MainAction.REFLECT_DOUBLE, "double_sided_min_reflections_not_met")
        if phase_reflect_double_count >= 1:
            drop(MainAction.REFLECT_DOUBLE, "double_sided_already_used_in_phase")
        if state.turns_since_double_sided_reflection < cfg.double_sided_min_turn_gap:
            drop(MainAction.REFLECT_DOUBLE, "double_sided_turn_gap_guardrail")

    if state.phase == Phase.CLOSING:
        if MainAction.SUMMARY in allowed:
            allowed.remove(MainAction.SUMMARY)
        if MainAction.REFLECT_DOUBLE in allowed:
            allowed.remove(MainAction.REFLECT_DOUBLE)
        disallowed[MainAction.SUMMARY.value] = "closing_phase_no_summary"
        disallowed[MainAction.REFLECT_DOUBLE.value] = "closing_phase_no_reflect_double"

    if not allowed:
        allowed = [MainAction.REFLECT_COMPLEX]
        mask_steps.append("mask_empty_fallback_reflect_complex")

    fallback_priority, fallback_debug = _build_mask_fallback_priority(
        state=state,
        features=features,
        cfg=cfg,
        allowed_actions=allowed,
        focus_choice_context=focus_choice_context,
    )
    ordered_allowed = [action for action in fallback_priority if action in set(allowed)]
    for action in allowed:
        if action not in ordered_allowed:
            ordered_allowed.append(action)

    return {
        "allowed_actions": ordered_allowed,
        "fallback_action": ordered_allowed[0],
        "mask_steps": mask_steps,
        "disallowed_actions": disallowed,
        "info_share_phase_gate_open": info_gate_open,
        "fallback_debug": fallback_debug,
    }


def _action_ranker_supports_allowed_actions(action_ranker: ActionRanker) -> bool:
    try:
        sig = inspect.signature(action_ranker.rank)
    except Exception:
        return False
    return "allowed_actions" in sig.parameters


_AFFIRM_COMPLEX_SIGNAL_PATTERNS: Sequence[str] = (
    "value",
    "important",
    "meaning",
    "authentic",
    "belief",
    "responsibility",
    "family",
    "future",
    "work",
    "role",
    "want to continue",
    "want to protect",
    "pride",
    "resourcefulness",
    "overcome",
    "face",
)


def _simple_affirm_interval(state: DialogueState) -> int:
    """
    Vary the spacing of simple affirmations between 1 and 2 turns, preferring 2.
    Derive it from state rather than randomness so behavior stays reproducible.
    """
    mix = state.turn_index + state.phase_turns + state.r_since_q + len(state.last_actions)
    return 1 if (mix % 7) in (1, 4) else 2


def _estimate_complex_affirm_signal(*, user_text: str, features: PlannerFeatures, state: DialogueState) -> float:
    text = user_text or ""
    hit_count = sum(1 for w in _AFFIRM_COMPLEX_SIGNAL_PATTERNS if w in text)
    importance_slots = state.phase_slots.get(Phase.IMPORTANCE_PROMOTION.name, {})
    core_values_text = _normalize_slot_text(
        importance_slots.get("core_values") if isinstance(importance_slots, Mapping) else ""
    )
    score = 0.0
    score += min(0.45, 0.15 * hit_count)
    score += min(0.25, 0.35 * max(0.0, features.change_talk))
    score += min(0.20, 0.25 * max(0.0, features.novelty))
    score += 0.15 if core_values_text else 0.0
    if len(re.sub(r"\s+", "", text)) >= 20:
        score += 0.05
    return max(0.0, min(1.0, score))


def _decide_affirm_mode(
    *,
    user_text: str,
    features: PlannerFeatures,
    state: DialogueState,
) -> Tuple[AffirmationMode, Dict[str, Any]]:
    """
    Three-way affirmation decision:
    - NONE
    - SIMPLE: add lightly every 2-3 turns
    - COMPLEX: use sparingly only when the context is rich and values/strengths are clearly present
    """
    complex_signal = _estimate_complex_affirm_signal(user_text=user_text, features=features, state=state)
    simple_interval = _simple_affirm_interval(state)

    has_dialogue_warmup = state.turn_index >= 4
    importance_slots = state.phase_slots.get(Phase.IMPORTANCE_PROMOTION.name, {})
    core_values_text = _normalize_slot_text(
        importance_slots.get("core_values") if isinstance(importance_slots, Mapping) else ""
    )
    has_value_context = bool(core_values_text) or features.change_talk >= 0.55
    complex_frequency_ok = state.turns_since_complex_affirm >= 10
    complex_ready = (
        has_dialogue_warmup
        and has_value_context
        and complex_frequency_ok
        and complex_signal >= 0.68
    )

    if complex_ready:
        debug = {
            "selected_mode": AffirmationMode.COMPLEX.value,
            "reason": "value_strength_salient_and_low_frequency",
            "complex_signal": complex_signal,
            "has_dialogue_warmup": has_dialogue_warmup,
            "has_value_context": has_value_context,
            "complex_frequency_ok": complex_frequency_ok,
            "simple_interval": simple_interval,
        }
        return AffirmationMode.COMPLEX, debug

    simple_due = state.turns_since_affirm >= simple_interval
    if simple_due:
        debug = {
            "selected_mode": AffirmationMode.SIMPLE.value,
            "reason": "cadence_due",
            "complex_signal": complex_signal,
            "simple_interval": simple_interval,
            "turns_since_affirm": state.turns_since_affirm,
        }
        return AffirmationMode.SIMPLE, debug

    debug = {
        "selected_mode": AffirmationMode.NONE.value,
        "reason": "cadence_not_due",
        "complex_signal": complex_signal,
        "simple_interval": simple_interval,
        "turns_since_affirm": state.turns_since_affirm,
    }
    return AffirmationMode.NONE, debug


def _normalize_affirm_score_label(value: Any) -> str:
    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if raw in {"NONE", "NO", "FALSE", "0", "OFF"}:
        return "NONE"
    if raw in {"SIMPLE", "SINGLE", "TRUE", "YES", "1", "ON"}:
        return "SIMPLE"
    if raw in {"COMPLEX", "DEEP"}:
        return "COMPLEX"
    return ""


def _parse_affirm_mode_scores(payload: Any) -> Tuple[Dict[str, float], List[str], Optional[str]]:
    scores: Dict[str, float] = {"NONE": 0.0, "SIMPLE": 0.0, "COMPLEX": 0.0}
    issues: List[str] = []
    selected_mode_hint: Optional[str] = None
    parsed_count = 0

    def _assign(label: Any, raw_score: Any, *, source: str) -> None:
        nonlocal parsed_count
        normalized_label = _normalize_affirm_score_label(label)
        if not normalized_label:
            issues.append(f"unknown_score_label:{source}:{label}")
            return
        score = _clamp01(raw_score, -1.0)
        if score < 0.0:
            issues.append(f"invalid_score_value:{source}:{label}:{raw_score}")
            return
        scores[normalized_label] = max(scores[normalized_label], float(score))
        parsed_count += 1

    if isinstance(payload, Mapping):
        for selected_key in ("selected_mode", "selected_affirm_mode", "mode", "predicted_mode"):
            if selected_key not in payload:
                continue
            normalized = _normalize_affirm_score_label(payload.get(selected_key))
            if normalized:
                selected_mode_hint = normalized
                break
        nested_scores = None
        for nested_key in ("scores", "mode_scores", "affirm_scores", "affirmation_scores"):
            nested = payload.get(nested_key)
            if isinstance(nested, Mapping):
                nested_scores = nested
                break
        if isinstance(nested_scores, Mapping):
            for label, raw_score in nested_scores.items():
                _assign(label, raw_score, source=f"nested:{label}")
        for key, label in (
            ("none_score", "NONE"),
            ("no_affirm_score", "NONE"),
            ("simple_score", "SIMPLE"),
            ("single_score", "SIMPLE"),
            ("complex_score", "COMPLEX"),
        ):
            if key in payload:
                _assign(label, payload.get(key), source=f"flat:{key}")
        for label in ("NONE", "SIMPLE", "COMPLEX"):
            if label in payload:
                _assign(label, payload.get(label), source=f"flat:{label}")

    if parsed_count <= 0 and selected_mode_hint:
        scores[selected_mode_hint] = 1.0
        parsed_count = 1
        issues.append("scores_missing_used_selected_mode_hint")

    if parsed_count <= 0:
        uniform = round(1.0 / 3.0, 6)
        scores = {"NONE": uniform, "SIMPLE": uniform, "COMPLEX": uniform}
        issues.append("scores_missing_default_uniform")
        return scores, issues, selected_mode_hint

    total = float(sum(max(0.0, score) for score in scores.values()))
    if total <= 1e-9:
        uniform = round(1.0 / 3.0, 6)
        scores = {"NONE": uniform, "SIMPLE": uniform, "COMPLEX": uniform}
        issues.append("scores_non_positive_default_uniform")
        return scores, issues, selected_mode_hint

    normalized_scores = {label: max(0.0, score) / total for label, score in scores.items()}
    return normalized_scores, issues, selected_mode_hint


def _select_affirm_mode_from_scores(
    *,
    user_text: str,
    features: PlannerFeatures,
    state: DialogueState,
    mode_scores: Mapping[str, float],
) -> Tuple[AffirmationMode, Dict[str, Any]]:
    base_scores = {
        "NONE": _clamp01(mode_scores.get("NONE"), 0.0),
        "SIMPLE": _clamp01(mode_scores.get("SIMPLE"), 0.0),
        "COMPLEX": _clamp01(mode_scores.get("COMPLEX"), 0.0),
    }
    total = float(sum(base_scores.values()))
    if total > 1e-9:
        base_scores = {label: score / total for label, score in base_scores.items()}
    else:
        uniform = 1.0 / 3.0
        base_scores = {"NONE": uniform, "SIMPLE": uniform, "COMPLEX": uniform}

    simple_interval = _simple_affirm_interval(state)
    simple_due = state.turns_since_affirm >= simple_interval
    has_dialogue_warmup = state.turn_index >= 4
    importance_slots = state.phase_slots.get(Phase.IMPORTANCE_PROMOTION.name, {})
    core_values_text = _normalize_slot_text(
        importance_slots.get("core_values") if isinstance(importance_slots, Mapping) else ""
    )
    has_value_context = bool(core_values_text) or features.change_talk >= 0.55
    complex_frequency_ok = state.turns_since_complex_affirm >= 10
    complex_gate_open = has_dialogue_warmup and has_value_context and complex_frequency_ok

    gated_scores = dict(base_scores)
    if not simple_due:
        gated_scores["SIMPLE"] = 0.0
    if not complex_gate_open:
        gated_scores["COMPLEX"] = 0.0

    gated_total = float(sum(gated_scores.values()))
    if gated_total <= 1e-9:
        gated_scores = {"NONE": 1.0, "SIMPLE": 0.0, "COMPLEX": 0.0}
    else:
        gated_scores = {label: score / gated_total for label, score in gated_scores.items()}

    tie_priority = {"NONE": 0, "SIMPLE": 1, "COMPLEX": 2}
    selected_label = max(
        ("NONE", "SIMPLE", "COMPLEX"),
        key=lambda label: (gated_scores.get(label, 0.0), tie_priority[label]),
    )
    selected_mode = _normalize_affirmation_mode(selected_label)
    debug = {
        "method": "llm_score_selector",
        "selected_mode": selected_mode.value,
        "selection_source": "llm_mode_scores_with_state_gate",
        "mode_scores": {label: round(score, 4) for label, score in base_scores.items()},
        "gated_mode_scores": {label: round(score, 4) for label, score in gated_scores.items()},
        "simple_due": simple_due,
        "simple_interval": simple_interval,
        "has_dialogue_warmup": has_dialogue_warmup,
        "has_value_context": has_value_context,
        "complex_frequency_ok": complex_frequency_ok,
        "complex_gate_open": complex_gate_open,
        "rule_complex_signal_reference": round(
            _estimate_complex_affirm_signal(user_text=user_text, features=features, state=state),
            4,
        ),
    }
    return selected_mode, debug


def decide_affirm(
    features: PlannerFeatures,
    state: DialogueState,
    user_text: str = "",
) -> AffirmationMode:
    """
    For compatibility, the public function returns only the mode.
    `bool(decide_affirm(...))` stays backward-compatible via `__bool__` (`NONE` only is False).
    """
    mode, _ = _decide_affirm_mode(user_text=user_text, features=features, state=state)
    return mode


def apply_action_to_state(
    *,
    state: DialogueState,
    features: PlannerFeatures,
    action: MainAction,
    add_affirm: AffirmationMode | bool | str,
    reflection_style: Optional[ReflectionStyle] = None,
) -> DialogueState:
    ns = dataclasses.replace(state)
    ns.scale_followup_done_in_phase = dict(state.scale_followup_done_in_phase or {})
    affirm_mode = _normalize_affirmation_mode(add_affirm)
    ns.turn_index += 1
    # Number of turns spent in the current phase (+1 on counselor output).
    ns.phase_turns += 1

    # info_mode transitions
    if action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        ns.info_mode = InfoMode.WAITING_PERMISSION
    elif action == MainAction.CLARIFY_PREFERENCE:
        # Exit the waiting state here to avoid permission re-ask loops.
        ns.info_mode = InfoMode.NONE
    elif action == MainAction.PROVIDE_INFO:
        ns.info_mode = InfoMode.NONE
    elif state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is False:
        ns.info_mode = InfoMode.NONE
    elif state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is True:
        ns.info_mode = InfoMode.READY_TO_PROVIDE

    phase_key = state.phase.name
    if action == MainAction.SCALING_QUESTION and _is_scaling_question_phase(state.phase):
        ns.scale_followup_done_in_phase[phase_key] = False
        _clear_scale_followup_runtime_state(ns)
    elif action == MainAction.QUESTION and _is_scale_followup_pending(state):
        pending_step = _current_scale_followup_step(state)
        ns.scale_followup_pending_phase = state.phase
        ns.scale_followup_score = state.scale_followup_score
        ns.scale_followup_pending_step = None
        ns.scale_followup_last_asked_step = pending_step or None
    elif state.scale_followup_pending_phase is not None and state.scale_followup_pending_phase != state.phase:
        _clear_scale_followup_runtime_state(ns)

    # Rhythm counters
    emitted_double_sided = False
    if action in (
        MainAction.QUESTION,
        MainAction.SCALING_QUESTION,
        MainAction.CLARIFY_PREFERENCE,
        MainAction.ASK_PERMISSION_TO_SHARE_INFO,
        MainAction.PROVIDE_INFO,
    ):
        ns.r_since_q = 0
        ns.reflect_streak = 0
    elif _is_reflect_action(action):
        ns.r_since_q += 1
        ns.reflect_streak += 1
        ns.reflection_turn_count += 1
        style = reflection_style or _reflection_style_from_action(action)
        ns.last_reflection_styles = (ns.last_reflection_styles + [style])[-12:]
        if style == ReflectionStyle.DOUBLE_SIDED:
            emitted_double_sided = True
            ns.double_sided_reflection_count += 1
    elif action == MainAction.SUMMARY:
        ns.r_since_q += 1
        ns.reflect_streak = 0

    if emitted_double_sided:
        ns.turns_since_double_sided_reflection = 0
    else:
        ns.turns_since_double_sided_reflection += 1

    # Summary counters
    if action == MainAction.SUMMARY:
        ns.turns_since_summary = 0
    else:
        ns.turns_since_summary += 1

    # Affirmation counters
    if affirm_mode != AffirmationMode.NONE:
        ns.turns_since_affirm = 0
    else:
        ns.turns_since_affirm += 1
    if affirm_mode == AffirmationMode.COMPLEX:
        ns.turns_since_complex_affirm = 0
    else:
        ns.turns_since_complex_affirm += 1

    ns.last_actions = (ns.last_actions + [action])[-10:]
    return ns


# Simple first-turn hint detection
_FIRST_TURN_GREETING_PATTERNS: Sequence[str] = [
    "hello",
    "good evening",
    "good morning",
    "good morning",
    "nice to meet you",
    "nice to meet you",
    "looking forward to working with you",
    "thank you for your help",
    "thanks for your hard work",
]
_FIRST_TURN_CONSULT_PATTERNS: Sequence[str] = [
    "consult",
    "worry",
    "trouble",
    "want to talk",
    "want you to listen",
    "want you to listen",
    "please listen",
    "tell me",
    "help me",
    "help me",
    "about",
    "question",
]


def detect_first_turn_hint(user_text: str) -> Tuple[FirstTurnHint, Dict[str, bool]]:
    text = user_text or ""
    has_greeting = any(pat in text for pat in _FIRST_TURN_GREETING_PATTERNS)
    has_consult = any(pat in text for pat in _FIRST_TURN_CONSULT_PATTERNS)

    if has_greeting and has_consult:
        hint = FirstTurnHint.GREETING_WITH_TOPIC
    elif has_greeting:
        hint = FirstTurnHint.GREETING_ONLY
    else:
        hint = FirstTurnHint.TOPIC_ONLY

    return hint, {"has_greeting": has_greeting, "has_consultation": has_consult}


# Phase progression order (do not skip by default)
_PHASE_ORDER: List[Phase] = [
    Phase.GREETING,
    Phase.PURPOSE_CONFIRMATION,
    Phase.CURRENT_STATUS_CHECK,
    Phase.FOCUSING_TARGET_BEHAVIOR,
    Phase.IMPORTANCE_PROMOTION,
    Phase.CONFIDENCE_PROMOTION,
    Phase.NEXT_STEP_DECISION,
    Phase.REVIEW_REFLECTION,
    Phase.CLOSING,
]

_TARGET_BEHAVIOR_ALIGNMENT_PHASES: Tuple[Phase, ...] = (
    Phase.IMPORTANCE_PROMOTION,
    Phase.CONFIDENCE_PROMOTION,
    Phase.NEXT_STEP_DECISION,
)

_PHASE_PLANNING_BOUNDARY: Dict[Phase, str] = {
    Phase.GREETING: (
        "Phase goal: greet the client, create safety, and establish a base for easy conversation. "
        "Planning note: do not make plans yet. Prioritize rapport and felt safety."
    ),
    Phase.PURPOSE_CONFIRMATION: (
        "Phase goal: clarify what the client wants to work on today. "
        "Planning note: stop at goal alignment; do not move into concrete execution details, timing, or frequency."
    ),
    Phase.CURRENT_STATUS_CHECK: (
        "Phase goal: understand the current situation, concrete problem scenes, difficulties, feelings, and context with care. "
        "Planning note: stay with understanding and organizing; do not propose even tiny next steps or agree on timing."
    ),
    Phase.FOCUSING_TARGET_BEHAVIOR: (
        "Phase goal: narrow the direction of the target behavior and focus the conversation. "
        "Planning note: focus the behavioral domain only. Do not decide dates, frequency, or deadlines yet."
    ),
    Phase.IMPORTANCE_PROMOTION: (
        "Phase goal: help the client explore and put into words why this goal matters, including reasons and values. "
        "Planning note: keep the work at the level of clarifying reasons and values; do not move into execution planning. "
        "In QUESTION, use an importance scale to evoke importance. "
        "In REFLECT, deepen DARN material: Desire, Ability, Reasons, and Need."
    ),
    Phase.CONFIDENCE_PROMOTION: (
        "Phase goal: organize coping strategies, resources, and past successes, and find places where change feels possible so confidence can grow. "
        "Planning note: stay with coping strategies, resources, and successful experiences. Leave exact dates and procedures to the next phase. "
        "In QUESTION, use a confidence scale to evoke confidence. "
        "In REFLECT, deepen CAT material: Commitment, Activation, and Taking steps."
    ),
    Phase.NEXT_STEP_DECISION: (
        "Phase goal: agree on the next step as a concrete action. "
        "Planning note: only here is it acceptable to specify when, what, and how much. "
        "In QUESTION, make execution conditions, barriers, and restart rules concrete so willingness and likelihood can be expressed. "
        "In REFLECT, deepen the client's wording about whether the next step feels doable and under what conditions."
    ),
    Phase.REVIEW_REFLECTION: (
        "Phase goal: reflect on what stands out most from today and what the client wants to keep in mind next. "
        "Planning note: do not add new plans; revisit scenes, reasons, and what the client wants to carry forward."
    ),
    Phase.CLOSING: (
        "Phase goal: summarize key points and close with connection and appreciation. "
        "Planning note: do not add new planning; confirm what has been agreed and bring the session to a close."
    ),
}

def _phase_index(p: Phase) -> int:
    return _PHASE_ORDER.index(p)

def _is_mid_phase(p: Phase) -> bool:
    return p not in (Phase.GREETING, Phase.CLOSING)

def _min_turns_required_for_exit(p: Phase) -> int:
    if p == Phase.REVIEW_REFLECTION:
        return 2
    if p == Phase.CLOSING:
        # Handle CLOSING in two turns: first ask whether to end, then close on the next turn.
        return 1
    return 3 if _is_mid_phase(p) else 0


_SLOT_QUALITY_LOW_INFO_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(do not know|not set|blank|nothing in particular|none$|unknown|cannot decide yet|cannot quite decide yet)"),
)
_SLOT_QUALITY_ACTIONABLE_VERB_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(do|want to do|try|continue|start|reduce|increase|talk|rest|move forward)"),
)
_SLOT_SCALE_KEYS: Tuple[str, ...] = ("importance_scale", "confidence_scale")
# Only importance/confidence strictly require numeric values. `commitment_level` accepts both numeric and qualitative forms and is handled separately.
_SLOT_NUMERIC_REQUIRED_KEYS: Tuple[str, ...] = ("importance_scale", "confidence_scale")
_SLOT_SCALE_REASON_MARKERS: Tuple[str, ...] = (
    "because",
    "because",
    "because",
    "reason",
    "basis",
    "if so",
    "if",
    "prospect",
    "why",
    "why",
)
_SLOT_SCALE_REASON_PROBE_STRICT_MARKERS: Tuple[str, ...] = (
    "reason",
    "basis",
    "why",
    "why",
)
_SCALE_FOLLOWUP_ZERO_ANCHOR_MARKERS: Tuple[str, ...] = (
    "0 points",
    "zero points",
    "rather than 0",
    "rather than zero",
)
_SCALE_FOLLOWUP_PLUS_ONE_MARKERS: Tuple[str, ...] = (
    "go up by 1 point",
    "go up by 1 point",
    "raise by 1 point",
    "raise by 1 point",
    "go up by one",
    "go up by one",
    "go up by one",
    "go up by one",
    "go up by one",
    "go up by one",
    "go up by one level",
    "go up by one level",
)
_SCALE_VALUE_BLOCKED_SUFFIXES: Tuple[str, ...] = (
    "minutes",
    "hours",
    "days",
    "weeks",
    "week",
    "months",
    "months",
    "month",
    "year",
    "times",
    "o'clock",
    "seconds",
    "yen",
    "people",
    "items",
    "items",
    "items",
    "sheets",
    "years old",
    "years old",
    "%",
    "％",
)
_SLOT_CONTEXT_TIME_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{1,2}\s*(?:o'clock|am|pm|min(?:ute)?s?)|morning|afternoon|evening|late night|this week|next week|weekday|weekend|sunday|monday|tuesday|wednesday|thursday|friday|saturday|every day|daily|every week|weekly|week\s*\d+)"),
)
_SLOT_CONTEXT_PLACE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(home|house|workplace|office|company|school|desk|bedroom|bed|living room|kitchen|commute|train|on the go|cafe)"),
)
_SLOT_CONTEXT_TRIGGER_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(when|if|once|right before|right after|whenever I feel|when I notice|when I slump|when I get anxious)"),
)
_SLOT_SCENE_DISRUPTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(freeze|cannot move|get stuck|overthink|procrastinat|avoid|interrupt|stop making progress|come to a halt)"),
)
_SLOT_ACTION_OBSERVABLE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(write|record|note|walk|sleep|wake up|close|open|breathe|contact|do|try|start|continue|stop|reduce|increase)"),
)
_SLOT_VALIDATION_FORMAT_ISSUES: Tuple[str, ...] = (
    "scale_missing_number",
    "scale_out_of_range",
    "scale_missing_reason",
    "target_behavior_not_observable",
    "target_behavior_not_actionable",
    "problem_scene_not_reproducible",
    "execution_context_missing_context",
    "next_step_action_not_concrete",
)
_SLOT_REPAIR_ISSUE_PRIORITY: Dict[str, int] = {
    "missing_evidence": 0,
    "wrong_format": 1,
    "low_confidence": 2,
    "too_vague": 3,
}
_MAX_SLOT_REPAIR_HINTS_PER_TURN = max(len(slot_keys) for slot_keys in _PHASE_SLOT_SCHEMA.values())
_CURRENT_PHASE_SLOT_REPAIR_HINT_MISSING_ISSUE_PREFIX = "current_phase_slot_repair_hint_missing_for_low_quality:"
_CURRENT_PHASE_SLOT_REPAIR_HINT_DETAIL_MISSING_ISSUE_PREFIX = (
    "current_phase_slot_repair_hint_detail_missing_for_low_quality:"
)
_SLOT_REPAIR_HINT_OUTSIDE_CURRENT_PHASE_ISSUE_PREFIX = "slot_repair_hint_outside_current_phase:"
_SLOT_REPAIR_PROBE_STYLE: Dict[str, str] = {
    "missing_evidence": "ask for a concrete example or a recent instance to add evidence",
    "low_confidence": "check briefly and tentatively whether it fits, without stating it as certain",
    "wrong_format": "ask with an explicit format such as number, timing, or frequency",
    "too_vague": "make it more concrete through time, place, person, or action",
}


def _slot_key_from_focus_text(value: Any) -> str:
    text = _normalize_slot_text(value)
    if not text:
        return ""
    alias_map = {
        "barriers": "barrier_coping_strategy",
        "barrier_coping": "barrier_coping_strategy",
        "barrier_coping_strategy": "barrier_coping_strategy",
        "barrier": "barrier_coping_strategy",
        "barrier factor": "barrier_coping_strategy",
        "barrier coping strategy": "barrier_coping_strategy",
    }
    normalized = alias_map.get(text.lower(), alias_map.get(text, text))
    if normalized in _PHASE_SLOT_LABELS:
        return normalized
    for slot_key, slot_label in _PHASE_SLOT_LABELS.items():
        if normalized == slot_label:
            return slot_key
    return normalized


def _slot_label_from_target(slot_target: Any) -> str:
    slot_key = _slot_key_from_focus_text(slot_target)
    if not slot_key:
        return ""
    if slot_key in _PHASE_SLOT_LABELS:
        return _PHASE_SLOT_LABELS.get(slot_key, slot_key)
    return _sanitize_human_text(slot_key)


def _phase_from_slot_key(slot_key_raw: Any) -> Optional[Phase]:
    slot_key = _slot_key_from_focus_text(slot_key_raw)
    if not slot_key:
        return None
    return _PHASE_BY_SLOT_KEY.get(slot_key)


def _extract_slot_keys_from_confirmation_note(note: Any) -> List[str]:
    text = _normalize_slot_text(note)
    if not text:
        return []
    keys: List[str] = []
    for match in re.finditer(r"([a-z][a-z0-9_]{2,})\s*:", text):
        slot_key = _slot_key_from_focus_text(match.group(1))
        if slot_key not in _PHASE_SLOT_LABELS:
            continue
        if slot_key in keys:
            continue
        keys.append(slot_key)
        if len(keys) >= 4:
            break
    return keys


def _has_any_pattern(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _extract_scale_0_10_value(text: str) -> Optional[float]:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return None
    for match in re.finditer(r"(?<!\d)(10(?:\.\d+)?|[0-9](?:\.\d+)?)(?!\d)", normalized):
        suffix = normalized[match.end(1) : match.end(1) + 2]
        if any(suffix.startswith(unit) for unit in _SCALE_VALUE_BLOCKED_SUFFIXES):
            continue
        try:
            value = float(match.group(1))
        except Exception:
            continue
        return value
    return None


def _evidence_text_has_explicit_scale_value(
    *,
    quote: str,
    turn_ids: Sequence[int],
    turn_map: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> bool:
    normalized_turn_ids = _normalize_turn_id_list(turn_ids, min_turn=1)
    if turn_map is not None and normalized_turn_ids:
        saw_turn_text = False
        for turn_id in normalized_turn_ids:
            turn_info = turn_map.get(turn_id)
            if not isinstance(turn_info, Mapping):
                continue
            saw_turn_text = True
            turn_text = _normalize_slot_text(turn_info.get("text"))
            if _extract_scale_0_10_value(turn_text) is not None:
                return True
        if saw_turn_text:
            return False
    return _extract_scale_0_10_value(quote) is not None


def _has_explicit_numeric_scale_evidence(
    *,
    slot_key: str,
    user_quote: str,
    user_turn_ids: Sequence[int],
    assistant_quote: str,
    assistant_turn_ids: Sequence[int],
    turn_map: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> bool:
    normalized_slot_key = _normalize_slot_text(slot_key)
    if normalized_slot_key not in _SLOT_NUMERIC_REQUIRED_KEYS:
        return True
    if _evidence_text_has_explicit_scale_value(
        quote=user_quote,
        turn_ids=user_turn_ids,
        turn_map=turn_map,
    ):
        return True
    if assistant_turn_ids and _evidence_text_has_explicit_scale_value(
        quote=assistant_quote,
        turn_ids=assistant_turn_ids,
        turn_map=turn_map,
    ):
        return True
    return False


def _apply_explicit_numeric_scale_quality_floor(
    *,
    slot_key: str,
    candidate_value: str,
    quality_score: float,
    user_quote: str,
    user_turn_ids: Sequence[int],
    turn_map: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> float:
    normalized_score = _clamp01(quality_score, 0.0)
    normalized_slot_key = _normalize_slot_text(slot_key)
    if normalized_slot_key not in _SLOT_NUMERIC_REQUIRED_KEYS:
        return normalized_score
    if _extract_scale_0_10_value(candidate_value) is None:
        return normalized_score
    if not _evidence_text_has_explicit_scale_value(
        quote=user_quote,
        turn_ids=user_turn_ids,
        turn_map=turn_map,
    ):
        return normalized_score
    return max(normalized_score, _EXPLICIT_NUMERIC_SCALE_QUALITY_FLOOR)


def _contains_scale_zero_anchor(text: str) -> bool:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _SCALE_FOLLOWUP_ZERO_ANCHOR_MARKERS)


def _contains_scale_plus_one_probe(text: str) -> bool:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _SCALE_FOLLOWUP_PLUS_ONE_MARKERS)


def _count_slot_context_categories(text: str) -> int:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return 0
    categories = 0
    if _has_any_pattern(normalized, _SLOT_CONTEXT_TIME_PATTERNS):
        categories += 1
    if _has_any_pattern(normalized, _SLOT_CONTEXT_PLACE_PATTERNS):
        categories += 1
    if _has_any_pattern(normalized, _SLOT_CONTEXT_TRIGGER_PATTERNS):
        categories += 1
    return categories


def _validate_slot_value_format(*, slot_key: str, value: str) -> str:
    # Classical NLP-style format checks (regex / lexical heuristics) are no longer used.
    # Let the Layer2 reviewer judge format validity; do not block here.
    _ = slot_key
    _ = value
    return ""


def _collect_slot_quality_target_examples(
    *,
    phase_prediction_debug: Optional[Mapping[str, Any]],
    slot_target: str,
    current_phase: Optional[Phase] = None,
) -> List[Dict[str, Any]]:
    if _did_phase_transition_for_current_turn(phase_prediction_debug):
        return []
    if not isinstance(phase_prediction_debug, Mapping):
        return []
    if "slot_quality_target_examples" in phase_prediction_debug:
        raw_review_hints = phase_prediction_debug.get("slot_quality_target_examples")
    elif "slot_repair_hints" in phase_prediction_debug:
        raw_review_hints = phase_prediction_debug.get("slot_repair_hints")
    else:
        slot_review = phase_prediction_debug.get("slot_review")
        raw_review_hints = (
            slot_review.get("slot_quality_target_examples")
            if isinstance(slot_review, Mapping) and slot_review.get("slot_quality_target_examples") is not None
            else (
                slot_review.get("slot_repair_hints")
                if isinstance(slot_review, Mapping)
                else None
            )
        )
    if not isinstance(raw_review_hints, Sequence) or isinstance(raw_review_hints, (str, bytes)):
        return []
    normalized_slot_target = _slot_key_from_focus_text(slot_target)
    slot_focus_priority = phase_prediction_debug.get("slot_focus_priority")
    priority_rank: Dict[str, int] = {}
    for slot_key_raw in [normalized_slot_target] + (
        list(slot_focus_priority)
        if isinstance(slot_focus_priority, Sequence) and not isinstance(slot_focus_priority, (str, bytes))
        else []
    ):
        slot_key = _slot_key_from_focus_text(slot_key_raw)
        if not slot_key or slot_key in priority_rank:
            continue
        priority_rank[slot_key] = len(priority_rank)

    current_phase_slot_keys = (
        set(_PHASE_SLOT_SCHEMA.get(current_phase, ()))
        if isinstance(current_phase, Phase)
        else set()
    )
    ranked_hints: List[Tuple[Tuple[int, int, int, int, int], Dict[str, Any]]] = []
    for idx, item in enumerate(raw_review_hints):
        if not isinstance(item, Mapping):
            continue
        slot_key = _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
        if not slot_key or slot_key not in _PHASE_SLOT_LABELS:
            continue
        if current_phase_slot_keys and slot_key not in current_phase_slot_keys:
            continue
        hint_phase = _parse_phase_from_any(item.get("phase"))
        if hint_phase is None:
            hint_phase = _parse_phase_from_any(item.get("phase_code"))
        if hint_phase is None:
            hint_phase = _phase_from_slot_key(slot_key)
        slot_label = _sanitize_human_text(item.get("slot_label")) or _sanitize_human_text(
            _PHASE_SLOT_LABELS.get(slot_key, slot_key)
        )
        issue_type = _sanitize_human_text(item.get("issue_type")) or "too_vague"
        probe_style = (
            _sanitize_human_text(item.get("preferred_probe_style"))
            or _SLOT_REPAIR_PROBE_STYLE.get(issue_type, _SLOT_REPAIR_PROBE_STYLE["too_vague"])
        )
        detail = _extract_slot_quality_target_example_detail(item)
        target_information = _extract_slot_quality_target_information(item)
        hint: Dict[str, Any] = {
            "slot_key": slot_key,
            "slot_label": slot_label,
            "issue_type": issue_type,
            "preferred_probe_style": probe_style,
        }
        if hint_phase is not None:
            hint["phase"] = hint_phase.value
            hint["phase_code"] = hint_phase.name
        if target_information:
            hint["target_information"] = target_information
        if detail:
            hint["detail"] = detail
            hint["quality_upgrade_model_text"] = detail
        ranked_hints.append(
            (
                (
                    0 if normalized_slot_target and slot_key == normalized_slot_target else 1,
                    priority_rank.get(slot_key, len(priority_rank) + idx),
                    _SLOT_REPAIR_ISSUE_PRIORITY.get(issue_type, len(_SLOT_REPAIR_ISSUE_PRIORITY)),
                    0 if detail else 1,
                    idx,
                ),
                hint,
            )
        )

    ranked_hints.sort(key=lambda item: item[0])
    hints: List[Dict[str, Any]] = []
    seen_slot_keys: set[str] = set()
    for _, hint in ranked_hints:
        slot_key = str(hint.get("slot_key") or "").strip()
        if not slot_key or slot_key in seen_slot_keys:
            continue
        hints.append(hint)
        seen_slot_keys.add(slot_key)
        if len(hints) >= _MAX_SLOT_REPAIR_HINTS_PER_TURN:
            break
    return hints


def _collect_slot_repair_hints(
    *,
    phase_prediction_debug: Optional[Mapping[str, Any]],
    slot_target: str,
    current_phase: Optional[Phase] = None,
) -> List[Dict[str, Any]]:
    return _collect_slot_quality_target_examples(
        phase_prediction_debug=phase_prediction_debug,
        slot_target=slot_target,
        current_phase=current_phase,
    )


def _did_phase_transition_for_current_turn(
    phase_prediction_debug: Optional[Mapping[str, Any]],
) -> bool:
    if not isinstance(phase_prediction_debug, Mapping):
        return False
    phase_override = phase_prediction_debug.get("phase_override_for_scale_followup")
    if isinstance(phase_override, Mapping):
        from_raw = phase_override.get("from_phase")
        forced_raw = phase_override.get("forced_phase")
        from_phase = _parse_phase_from_any(from_raw)
        forced_phase = _parse_phase_from_any(forced_raw)
        if from_phase is not None and forced_phase is not None:
            return forced_phase != from_phase
        from_text = _normalize_slot_text(from_raw)
        forced_text = _normalize_slot_text(forced_raw)
        if from_text and forced_text:
            return forced_text != from_text
    enforce = phase_prediction_debug.get("enforce")
    if isinstance(enforce, Mapping):
        current_raw = enforce.get("current")
        predicted_raw = enforce.get("predicted")
        current_phase = _parse_phase_from_any(current_raw)
        predicted_phase = _parse_phase_from_any(predicted_raw)
        if current_phase is not None and predicted_phase is not None:
            return predicted_phase != current_phase
        current_text = _normalize_slot_text(current_raw)
        predicted_text = _normalize_slot_text(predicted_raw)
        if current_text and predicted_text:
            return current_text != predicted_text
    transition_code = _normalize_slot_text(phase_prediction_debug.get("phase_transition_code"))
    if transition_code == "phase_advanced":
        return True
    transition_note = _normalize_slot_text(phase_prediction_debug.get("phase_transition_note"))
    return bool(transition_note and transition_note.startswith("New phase started"))


def _next_phase_or_self(current: Phase) -> Phase:
    idx = _phase_index(current)
    if idx >= len(_PHASE_ORDER) - 1:
        return current
    return _PHASE_ORDER[idx + 1]


def _derive_slot_focus_priority_from_slot_gate(
    *,
    phase: Phase,
    slot_gate: Optional[Mapping[str, Any]],
    quality_min_threshold: float,
) -> List[str]:
    required_slots = list(_PHASE_SLOT_SCHEMA.get(phase, ()))
    required_set = set(required_slots)
    priority: List[str] = []

    def _push(slot_key_raw: Any) -> None:
        slot_key = _slot_key_from_focus_text(slot_key_raw)
        if not slot_key or slot_key not in required_set or slot_key in priority:
            return
        priority.append(slot_key)

    if isinstance(slot_gate, Mapping):
        missing_slot_keys = slot_gate.get("missing_slot_keys")
        if isinstance(missing_slot_keys, Sequence) and not isinstance(missing_slot_keys, (str, bytes)):
            for item in missing_slot_keys:
                _push(item)

        slot_quality = slot_gate.get("slot_quality")
        if isinstance(slot_quality, Mapping):
            scored: List[Tuple[str, float]] = []
            for key, score in slot_quality.items():
                slot_key = _slot_key_from_focus_text(key)
                quality = _clamp01(score, -1.0)
                if not slot_key or slot_key not in required_set or quality < 0.0:
                    continue
                scored.append((slot_key, quality))
            scored.sort(key=lambda item: item[1])
            for slot_key, score in scored:
                if score < quality_min_threshold:
                    _push(slot_key)

    for slot_key in required_slots:
        _push(slot_key)
    return priority


def _estimate_slot_text_quality(slot_key: str, value: str) -> float:
    """
    Legacy length- and token-pattern-based estimation is no longer used.
    Quality should normally come from the Layer2 LLM reviewer's `quality_score`;
    this function only serves as a fallback when reviewer information is unavailable.
    """
    _ = slot_key  # Retained for backward compatibility.
    text = _normalize_slot_text(value)
    if not text:
        return 0.0
    return _clamp01(_SLOT_FALLBACK_QUALITY_SCORE, 0.0)


def evaluate_phase_slot_readiness(
    *,
    state: DialogueState,
    cfg: PlannerConfig,
    user_text: str,
    features: Optional[PlannerFeatures] = None,
    slot_review_debug: Optional[Mapping[str, Any]] = None,
    phase_slot_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    current = state.phase
    next_phase = _next_phase_or_self(current)
    required_slots = _PHASE_SLOT_SCHEMA.get(current, ())
    phase_slot_memory = _copy_phase_slot_memory(state.phase_slots)
    effective_slot_meta = _copy_phase_slot_meta(phase_slot_meta or state.phase_slot_meta)
    slot_bucket = phase_slot_memory.get(current.name, {})
    slot_meta_bucket = effective_slot_meta.get(current.name, {})

    slot_values: Dict[str, str] = {}
    effective_slot_status: Dict[str, str] = {}
    filled_slot_keys: List[str] = []
    slot_quality: Dict[str, float] = {}
    slot_review_quality = (
        slot_review_debug.get("slot_quality")
        if isinstance(slot_review_debug, Mapping)
        else {}
    )
    if not isinstance(slot_review_quality, Mapping):
        slot_review_quality = {}

    for slot_key in required_slots:
        slot_text = _normalize_slot_text(slot_bucket.get(slot_key))
        slot_values[slot_key] = slot_text
        slot_meta = slot_meta_bucket.get(slot_key)
        status = (
            _normalize_slot_text((slot_meta or {}).get("status"))
            if isinstance(slot_meta, Mapping)
            else ""
        )
        if status not in {"confirmed", "inactive"}:
            status = "inactive"
        effective_slot_status[slot_key] = status

        committed_quality_floor = (
            _clamp01((slot_meta or {}).get("quality_score"), 0.0)
            if isinstance(slot_meta, Mapping)
            else 0.0
        )
        should_apply_quality_floor = bool(slot_text)
        if slot_key in slot_review_quality:
            reviewed_quality = _clamp01(slot_review_quality.get(slot_key), 0.0)
            slot_quality[slot_key] = (
                max(reviewed_quality, committed_quality_floor)
                if should_apply_quality_floor
                else reviewed_quality
            )
        else:
            # Do not fabricate a quality estimate when the LLM reviewer does not return one.
            # Only preserve a previously confirmed quality value if it exists.
            slot_quality[slot_key] = committed_quality_floor if should_apply_quality_floor else 0.0

        if slot_text:
            filled_slot_keys.append(slot_key)

    required_count = len(required_slots)
    filled_count = len(filled_slot_keys)
    missing_slot_keys = [slot_key for slot_key in required_slots if slot_key not in filled_slot_keys]
    missing_slot_labels = [
        _PHASE_SLOT_LABELS.get(slot_key, slot_key)
        for slot_key in missing_slot_keys
    ]
    missing_slot_texts = [f"Need information about {label}" for label in missing_slot_labels]
    fill_rate = 1.0 if required_count == 0 else round(float(filled_count / required_count), 3)
    quality_values = [slot_quality.get(slot_key, 0.0) for slot_key in required_slots if slot_key in slot_quality]
    quality_mean: Optional[float] = (
        round(float(sum(quality_values) / len(quality_values)), 3)
        if quality_values
        else None
    )
    quality_min: Optional[float] = (
        round(float(min(quality_values)), 3)
        if quality_values
        else None
    )

    min_turns_required = _min_turns_required_for_exit(current)
    turn_gate_pass = state.phase_turns >= min_turns_required

    hold_reasons: List[str] = []
    if features is not None:
        if features.resistance >= cfg.phase_hold_resistance_threshold:
            hold_reasons.append("high_resistance")
        if features.discord >= cfg.phase_hold_discord_threshold:
            hold_reasons.append("high_discord")

    pass_fill = fill_rate >= cfg.phase_slot_fill_rate_threshold
    pass_quality_mean = (
        True
        if quality_mean is None
        else bool(quality_mean >= float(cfg.phase_slot_quality_mean_threshold))
    )
    pass_quality_min = (
        True
        if quality_min is None
        else bool(quality_min >= float(cfg.phase_slot_quality_min_threshold))
    )
    pass_relational = not hold_reasons
    unresolved_required_slot_keys = set(missing_slot_keys)
    blocking_issue_codes: List[str] = []
    blocking_issue_slot_keys: List[str] = []
    if isinstance(slot_review_debug, Mapping):
        reviewed_updates = slot_review_debug.get("reviewed_updates")
        reviewed_update_seen = False
        if isinstance(reviewed_updates, Sequence) and not isinstance(reviewed_updates, (str, bytes)):
            for item in reviewed_updates:
                if not isinstance(item, Mapping):
                    continue
                reviewed_update_seen = True
                slot_key = _normalize_slot_text(item.get("slot_key"))
                if not slot_key or slot_key not in unresolved_required_slot_keys:
                    continue
                issue_codes_for_item: List[str] = []
                raw_issue_codes = item.get("issue_codes")
                if isinstance(raw_issue_codes, Sequence) and not isinstance(raw_issue_codes, (str, bytes)):
                    issue_codes_for_item = _merge_issue_codes(raw_issue_codes)

                admissible_raw = item.get("admissible")
                has_admissible_flag = isinstance(admissible_raw, bool)
                decision = _normalize_slot_text(item.get("decision")).lower()
                is_blocking_update = (
                    (has_admissible_flag and not bool(admissible_raw))
                    or (not has_admissible_flag and decision in {"reject", "needs_confirmation"})
                )
                if not is_blocking_update and issue_codes_for_item:
                    is_blocking_update = any(
                        code in {"missing_evidence", "low_confidence", "wrong_format"}
                        for code in issue_codes_for_item
                    )
                if not is_blocking_update:
                    continue
                if slot_key not in blocking_issue_slot_keys:
                    blocking_issue_slot_keys.append(slot_key)
                for code in issue_codes_for_item:
                    if not code or code in blocking_issue_codes:
                        continue
                    blocking_issue_codes.append(code)
                    if len(blocking_issue_codes) >= 6:
                        break
                if len(blocking_issue_codes) >= 6:
                    break

        # Backward compatibility with the older format (`blocking_issue_codes` only):
        # only use it when unresolved required slots remain and `reviewed_updates` is absent.
        if unresolved_required_slot_keys and not reviewed_update_seen and not blocking_issue_codes:
            raw_blocking = slot_review_debug.get("blocking_issue_codes")
            if isinstance(raw_blocking, Sequence) and not isinstance(raw_blocking, (str, bytes)):
                for item in raw_blocking:
                    code = _normalize_slot_text(item)
                    if not code or code in blocking_issue_codes:
                        continue
                    blocking_issue_codes.append(code)
                    if len(blocking_issue_codes) >= 6:
                        break
    no_blocking_review_issue = not blocking_issue_codes

    has_user_content = len(re.sub(r"\s+", "", user_text or "")) >= 3
    if current == Phase.CLOSING:
        # CLOSING does not advance to another phase, but the final completion check still
        # uses slot readiness and minimum stay turns (relationship signals do not block closing here).
        passed = pass_fill and turn_gate_pass
        decision = "advance_by_closing_readiness" if passed else "stay_by_closing_readiness"
    elif current == Phase.GREETING:
        # GREETING should not linger; move past it shortly after the start.
        passed = state.turn_index >= 1 and has_user_content
        decision = "advance_by_greeting_completion" if passed else "stay_greeting_phase"
    elif current == Phase.REVIEW_REFLECTION:
        # REVIEW_REFLECTION is fixed to two turns and always moves to CLOSING afterward.
        passed = turn_gate_pass
        decision = "advance_by_review_fixed_two_turns" if passed else "stay_review_until_two_turns"
    else:
        passed = (
            pass_fill
            and pass_quality_min
            and turn_gate_pass
            and pass_relational
            and no_blocking_review_issue
        )
        decision = "advance_by_slot_readiness" if passed else "stay_by_slot_readiness"

    selected_phase = next_phase if passed else current
    gate_fail_reasons: List[str] = []
    if current == Phase.REVIEW_REFLECTION:
        if not turn_gate_pass:
            gate_fail_reasons.append("insufficient_min_turns")
    else:
        if not pass_fill:
            gate_fail_reasons.append("insufficient_slot_fill")
        if not pass_quality_min:
            gate_fail_reasons.append("insufficient_slot_quality")
        if not turn_gate_pass:
            gate_fail_reasons.append("insufficient_min_turns")
        if hold_reasons and current != Phase.CLOSING:
            gate_fail_reasons.append("relational_hold")
        if not no_blocking_review_issue:
            gate_fail_reasons.append("blocking_review_issue")

    return {
        "current_phase": current.value,
        "selected_phase": selected_phase.value,
        "next_phase_candidate": next_phase.value,
        "passed": passed,
        "decision": decision,
        "required_slots": list(required_slots),
        "required_slot_count": required_count,
        "filled_slot_count": filled_count,
        "filled_slot_keys": filled_slot_keys,
        "missing_slot_keys": missing_slot_keys,
        "missing_slot_labels": missing_slot_labels,
        "missing_slot_texts": missing_slot_texts,
        "fill_rate": fill_rate,
        "fill_rate_threshold": cfg.phase_slot_fill_rate_threshold,
        "slot_values": slot_values,
        "effective_slot_status": effective_slot_status,
        "slot_quality": slot_quality,
        "quality_mean": quality_mean,
        "quality_min": quality_min,
        "quality_mean_threshold": cfg.phase_slot_quality_mean_threshold,
        "quality_min_threshold": cfg.phase_slot_quality_min_threshold,
        "quality_gate_basis": "quality_min_only",
        "phase_turns": state.phase_turns,
        "min_turns_required": min_turns_required,
        "review_fixed_two_turn_phase": current == Phase.REVIEW_REFLECTION,
        "turn_gate_pass": turn_gate_pass,
        "pass_fill": pass_fill,
        "pass_quality_mean": pass_quality_mean,
        "pass_quality_min": pass_quality_min,
        "pass_relational": pass_relational,
        "no_blocking_review_issue": no_blocking_review_issue,
        "blocking_issue_codes": blocking_issue_codes,
        "blocking_issue_slot_keys": blocking_issue_slot_keys,
        "unresolved_required_slot_keys": sorted(unresolved_required_slot_keys),
        "gate_fail_reasons": gate_fail_reasons,
        "relational_hold_reasons": hold_reasons,
    }


def _build_phase_transition_feedback(
    *,
    current_phase: Phase,
    selected_phase: Phase,
    llm_phase_intent: str,
    llm_missing_requirements: Sequence[str],
    slot_gate_debug: Mapping[str, Any],
) -> Dict[str, Any]:
    missing_from_llm: List[str] = []
    for item in llm_missing_requirements:
        text = _normalize_slot_text(item)
        if not text:
            continue
        if text.endswith("information is needed"):
            missing_from_llm.append(text)
        else:
            missing_from_llm.append(f"{text} information is needed")

    missing_from_slots = [
        _normalize_slot_text(item)
        for item in (slot_gate_debug.get("missing_slot_texts") if isinstance(slot_gate_debug, Mapping) else [])
    ]
    missing_from_slots = [item for item in missing_from_slots if item]

    if selected_phase != current_phase:
        return {
            "phase_transition_code": "phase_advanced",
            "phase_transition_note": f"New phase started: {selected_phase.value}",
        }

    pass_fill = bool(slot_gate_debug.get("pass_fill"))
    turn_gate_pass = bool(slot_gate_debug.get("turn_gate_pass"))
    hold_reasons_raw = slot_gate_debug.get("relational_hold_reasons")
    hold_reasons = hold_reasons_raw if isinstance(hold_reasons_raw, list) else []

    if not pass_fill:
        # When fill is insufficient, prioritize concrete missing-slot information.
        missing_texts = missing_from_slots or missing_from_llm
        note = " / ".join(missing_texts[:3]) if missing_texts else "Required slot information is missing"
        return {
            "phase_transition_code": "need_slot_information",
            "phase_transition_note": note,
        }

    if not turn_gate_pass:
        if llm_phase_intent == "stay":
            stay_reasons = missing_from_llm or missing_from_slots
            if stay_reasons:
                note = " / ".join(stay_reasons[:3]) + " (stay: below the minimum required turns)"
                return {
                    "phase_transition_code": "llm_stay_intent",
                    "phase_transition_note": note,
                }
        return {
            "phase_transition_code": "insufficient_min_turns",
            "phase_transition_note": "Stay in phase (below the minimum required turns)",
        }

    if any(reason in {"high_resistance", "high_discord"} for reason in hold_reasons):
        return {
            "phase_transition_code": "need_resistance_or_discord_response",
            "phase_transition_note": "Need to respond to resistance or discord",
        }

    if llm_phase_intent == "stay":
        stay_reasons = missing_from_llm or missing_from_slots
        note = " / ".join(stay_reasons[:3]) if stay_reasons else "Stay in phase"
        return {
            "phase_transition_code": "llm_stay_intent",
            "phase_transition_note": note,
        }

    return {
        "phase_transition_code": "phase_stay",
        "phase_transition_note": "Stay in phase",
    }


def _build_phase_transition_feedback_from_review(
    *,
    current_phase: Phase,
    selected_phase: Phase,
    slot_gate_debug: Mapping[str, Any],
    slot_review_debug: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    review_debug = slot_review_debug if isinstance(slot_review_debug, Mapping) else {}
    slot_quality_target_examples = review_debug.get("slot_quality_target_examples")
    if slot_quality_target_examples is None:
        slot_quality_target_examples = review_debug.get("slot_repair_hints")
    if not isinstance(slot_quality_target_examples, Sequence) or isinstance(slot_quality_target_examples, (str, bytes)):
        slot_quality_target_examples = []
    normalized_hints: List[Dict[str, Any]] = []
    for item in slot_quality_target_examples:
        if not isinstance(item, Mapping):
            continue
        normalized_hints.append(dict(item))

    if selected_phase != current_phase:
        return {
            "phase_transition_code": "phase_advanced",
            "phase_transition_note": f"New phase started: {selected_phase.value}",
            "slot_focus_priority": [],
            "slot_quality_target_examples": [],
            "slot_repair_hints": [],
        }

    missing_slot_keys = slot_gate_debug.get("missing_slot_keys")
    missing_slot_keys = (
        [str(item).strip() for item in missing_slot_keys if str(item).strip()]
        if isinstance(missing_slot_keys, Sequence) and not isinstance(missing_slot_keys, (str, bytes))
        else []
    )
    quality_slots: List[str] = []
    low_quality_slots: List[str] = []
    slot_quality = slot_gate_debug.get("slot_quality")
    quality_min_threshold = _clamp01(slot_gate_debug.get("quality_min_threshold"), -1.0)
    if quality_min_threshold < 0.0:
        quality_min_threshold = 0.8
    if isinstance(slot_quality, Mapping):
        scored = sorted(
            [
                (str(key).strip(), _clamp01(value, -1.0))
                for key, value in slot_quality.items()
                if str(key).strip() and _clamp01(value, -1.0) >= 0.0
            ],
            key=lambda item: item[1],
        )
        quality_slots = [slot_key for slot_key, _ in scored[:3]]
        low_quality_slots = [
            slot_key
            for slot_key, score in scored
            if score >= 0.0 and score < quality_min_threshold
        ][:3]

    blocking_issue_codes = slot_gate_debug.get("blocking_issue_codes")
    blocking_issue_codes = (
        [str(item).strip() for item in blocking_issue_codes if str(item).strip()]
        if isinstance(blocking_issue_codes, Sequence) and not isinstance(blocking_issue_codes, (str, bytes))
        else []
    )
    gate_fail_reasons = slot_gate_debug.get("gate_fail_reasons")
    gate_fail_reasons = (
        [str(item).strip() for item in gate_fail_reasons if str(item).strip()]
        if isinstance(gate_fail_reasons, Sequence) and not isinstance(gate_fail_reasons, (str, bytes))
        else []
    )
    missing_slot_texts = slot_gate_debug.get("missing_slot_texts")
    missing_slot_texts = (
        [_normalize_slot_text(item) for item in missing_slot_texts if _normalize_slot_text(item)]
        if isinstance(missing_slot_texts, Sequence) and not isinstance(missing_slot_texts, (str, bytes))
        else []
    )
    review_summary_note = _normalize_slot_text(review_debug.get("review_summary_note"))

    if "insufficient_slot_fill" in gate_fail_reasons:
        note = " / ".join(missing_slot_texts[:3]) if missing_slot_texts else "Required slot information is missing"
        return {
            "phase_transition_code": "need_slot_information",
            "phase_transition_note": note,
            "slot_focus_priority": missing_slot_keys[:3],
            "slot_quality_target_examples": normalized_hints,
            "slot_repair_hints": normalized_hints,
        }
    if "insufficient_slot_quality" in gate_fail_reasons or "blocking_review_issue" in gate_fail_reasons:
        quality_note = review_summary_note or "Slot information quality needs improvement"
        if blocking_issue_codes:
            quality_note = f"{quality_note} ({', '.join(blocking_issue_codes[:3])})"
        return {
            "phase_transition_code": "need_slot_quality_improvement",
            "phase_transition_note": quality_note,
            "slot_focus_priority": (low_quality_slots or quality_slots or missing_slot_keys)[:3],
            "slot_quality_target_examples": normalized_hints,
            "slot_repair_hints": normalized_hints,
        }
    if "insufficient_min_turns" in gate_fail_reasons:
        return {
            "phase_transition_code": "insufficient_min_turns",
            "phase_transition_note": "Stay in phase (below the minimum required turns)",
            "slot_focus_priority": (quality_slots or missing_slot_keys)[:3],
            "slot_quality_target_examples": normalized_hints,
            "slot_repair_hints": normalized_hints,
        }
    if "relational_hold" in gate_fail_reasons:
        return {
            "phase_transition_code": "need_resistance_or_discord_response",
            "phase_transition_note": "Need to respond to resistance or discord",
            "slot_focus_priority": (quality_slots or missing_slot_keys)[:3],
            "slot_quality_target_examples": normalized_hints,
            "slot_repair_hints": normalized_hints,
        }
    return {
        "phase_transition_code": "phase_stay",
        "phase_transition_note": review_summary_note or "Stay in phase",
        "slot_focus_priority": (quality_slots or missing_slot_keys)[:3],
        "slot_quality_target_examples": normalized_hints,
        "slot_repair_hints": normalized_hints,
    }


def enforce_phase_progression(
    *,
    current: Phase,
    predicted: Phase,
    state: DialogueState,
    phase_features: Optional[PlannerFeatures] = None,
    cfg: Optional[PlannerConfig] = None,
) -> Tuple[Phase, Dict[str, Any]]:
    """
    Advance phases one step at a time in order and never move backward.
    When resistance/discord is strong, stay in the current phase instead of advancing.
    Except for greeting/closing, stay at least three turns before transitioning
    (review reflection may end after two turns).
    Returns: (selected_phase, debug)
    """
    dbg: Dict[str, Any] = {"enforce": True, "current": current.value, "predicted": predicted.value, "phase_turns": state.phase_turns}

    # Do not move backward (exception: risk-driven closing is handled in another layer).
    if _phase_index(predicted) <= _phase_index(current):
        dbg["decision"] = "no_backward"
        return current, dbg

    skip_relational_hold = current == Phase.GREETING and state.turn_index >= 1
    dbg["skip_relational_hold_for_greeting"] = skip_relational_hold

    if phase_features is not None and not skip_relational_hold:
        res_th = cfg.phase_hold_resistance_threshold if cfg is not None else 0.6
        dis_th = cfg.phase_hold_discord_threshold if cfg is not None else 0.6
        hold_reasons: List[str] = []
        if phase_features.resistance >= res_th:
            hold_reasons.append("high_resistance")
        if phase_features.discord >= dis_th:
            hold_reasons.append("high_discord")
        dbg["phase_hold_signals"] = {
            "resistance": phase_features.resistance,
            "topic_shift": phase_features.topic_shift,
            "discord": phase_features.discord,
            "resistance_threshold": res_th,
            "discord_threshold": dis_th,
        }
        if hold_reasons:
            dbg["decision"] = "hold_by_relational_signal"
            dbg["hold_reasons"] = hold_reasons
            return current, dbg
    elif phase_features is not None and skip_relational_hold:
        dbg["phase_hold_signals_ignored"] = {
            "reason": "greeting_should_not_span_multiple_turns",
            "turn_index": state.turn_index,
        }

    # Do not allow skipping more than one phase.
    next_idx = _phase_index(current) + 1
    desired = _PHASE_ORDER[next_idx]
    if _phase_index(predicted) > next_idx:
        dbg["clamped_from"] = predicted.value
        predicted = desired

    # Check the minimum-stay requirement (excluding greeting/closing).
    min_turns = _min_turns_required_for_exit(current)
    dbg["min_turns_required"] = min_turns
    if state.phase_turns < min_turns:
        dbg["decision"] = "stay_until_min_turns"
        return current, dbg

    # Progress allowed: move to the next phase (one step).
    dbg["decision"] = "advance_one"
    return predicted, dbg

def _history_to_dialogue(history: List[Tuple[str, str]], *, max_turns: int = 8) -> str:
    lines: List[str] = []
    for role, text in history[-max_turns:]:
        prefix = "Client" if role == "user" else "Counselor"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


def _parse_phase_from_text(text: str) -> Tuple[Optional[Phase], float]:
    t = (text or "").strip().strip("「」\"'`")
    if not t:
        return None, 0.0

    # Exact match (highest confidence).
    for p in Phase:
        if t == p.value:
            return p, 1.0

    # Label appears somewhere in the text (medium confidence).
    for p in Phase:
        if p.value in t:
            return p, 0.7

    # Rescue omitted or variant forms (low confidence).
    if "greeting" in t:
        return Phase.GREETING, 0.4
    if "purpose" in t:
        return Phase.PURPOSE_CONFIRMATION, 0.4
    if "current status" in t:
        return Phase.CURRENT_STATUS_CHECK, 0.4
    if "target" in t or "behavior" in t or "focus" in t:
        return Phase.FOCUSING_TARGET_BEHAVIOR, 0.35
    if "importance" in t:
        return Phase.IMPORTANCE_PROMOTION, 0.35
    if "confidence" in t:
        return Phase.CONFIDENCE_PROMOTION, 0.35
    if "next" in t or "next step" in t:
        return Phase.NEXT_STEP_DECISION, 0.35
    if "review" in t or "insight" in t or "learning" in t:
        return Phase.REVIEW_REFLECTION, 0.35
    if "closing" in t or "end" in t:
        return Phase.CLOSING, 0.35

    return None, 0.0


_SLOT_FILL_SCOPE_ALL = "all"
_SLOT_FILL_SCOPE_CURRENT_ONLY = "current_only"
_SLOT_FILL_SCOPE_NON_CURRENT_ONLY = "non_current_only"


def _normalize_slot_fill_scope_mode(value: Any) -> str:
    normalized = _normalize_slot_text(value).lower().replace("-", "_")
    if normalized in {
        _SLOT_FILL_SCOPE_CURRENT_ONLY,
        "current",
        "current_phase",
        "current_phase_only",
    }:
        return _SLOT_FILL_SCOPE_CURRENT_ONLY
    if normalized in {
        _SLOT_FILL_SCOPE_NON_CURRENT_ONLY,
        "non_current",
        "cross_phase",
        "other_phases",
        "non_current_phase_only",
    }:
        return _SLOT_FILL_SCOPE_NON_CURRENT_ONLY
    return _SLOT_FILL_SCOPE_ALL


def _slot_fill_previous_phases_for_scope(*, current_phase: Phase, scope_mode: str) -> List[Phase]:
    normalized_scope = _normalize_slot_fill_scope_mode(scope_mode)
    if normalized_scope == _SLOT_FILL_SCOPE_CURRENT_ONLY:
        return []
    current_idx = _phase_index(current_phase)
    previous = list(_PHASE_ORDER[:current_idx])
    if normalized_scope != _SLOT_FILL_SCOPE_NON_CURRENT_ONLY:
        return previous
    if current_idx < _phase_index(Phase.CURRENT_STATUS_CHECK):
        return []
    min_allowed_idx = _phase_index(Phase.PURPOSE_CONFIRMATION)
    return [
        phase
        for phase in previous
        if _phase_index(phase) >= min_allowed_idx and phase != Phase.GREETING
    ]


def _slot_fill_future_phases_for_scope(*, current_phase: Phase, scope_mode: str) -> List[Phase]:
    normalized_scope = _normalize_slot_fill_scope_mode(scope_mode)
    if normalized_scope == _SLOT_FILL_SCOPE_CURRENT_ONLY:
        return []
    current_idx = _phase_index(current_phase)
    future = list(_PHASE_ORDER[current_idx + 1 :])
    if normalized_scope != _SLOT_FILL_SCOPE_NON_CURRENT_ONLY:
        return future
    return [
        phase
        for phase in future
        if phase not in {Phase.REVIEW_REFLECTION, Phase.CLOSING}
    ]


@dataclass
class LLMPhaseSlotFiller:
    """
    Use the LLM to extract phase-specific slot-update candidates.
    - Return a `phase_slot_updates` list containing only slots targeted for update.
    """

    llm: LLMClient
    temperature: float = 0.0
    max_history_turns: int = 8
    json_mode: str = "loose"
    scope_mode: str = _SLOT_FILL_SCOPE_ALL

    def fill_slots(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        user_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        scope_mode = _normalize_slot_fill_scope_mode(self.scope_mode)
        dialogue, turn_start, turn_end = _history_to_numbered_dialogue(
            history,
            max_turns=self.max_history_turns,
        )
        current_phase = state.phase
        target_phase_codes = (
            {current_phase.name}
            if scope_mode != _SLOT_FILL_SCOPE_NON_CURRENT_ONLY
            else set()
        )
        current_phase_required_slot_keys = (
            list(_PHASE_SLOT_SCHEMA.get(current_phase, ()))
            if target_phase_codes
            else []
        )
        current_phase_committed = _copy_phase_slot_memory(state.phase_slots).get(current_phase.name, {})
        current_phase_meta = _copy_phase_slot_meta(state.phase_slot_meta).get(current_phase.name, {})
        layer1_target_quality_threshold = _SLOT_LAYER2_REVIEW_TARGET_QUALITY_THRESHOLD
        layer1_precheck_quality_by_slot: Dict[str, float] = {}
        layer1_prompt_target_slot_keys: List[str] = []
        for slot_key in current_phase_required_slot_keys:
            slot_value = _normalize_slot_text(current_phase_committed.get(slot_key))
            slot_meta = current_phase_meta.get(slot_key) if isinstance(current_phase_meta, Mapping) else {}
            quality_precheck = _precheck_slot_quality_for_layer2_target(
                slot_key=slot_key,
                slot_value=slot_value,
                slot_meta=slot_meta,
            )
            rounded_quality = round(float(_clamp01(quality_precheck, 0.0)), 3)
            layer1_precheck_quality_by_slot[slot_key] = rounded_quality
            if rounded_quality >= layer1_target_quality_threshold:
                continue
            layer1_prompt_target_slot_keys.append(slot_key)

        prompt_user_turn_ids = [
            idx
            for idx, (role, _text) in enumerate(history, start=1)
            if turn_start <= idx <= turn_end and str(role or "").strip().lower() == "user"
        ]
        prompt_assistant_turn_ids_for_assent_bridge = [
            idx
            for idx, (role, _text) in enumerate(history, start=1)
            if turn_start <= idx <= turn_end and str(role or "").strip().lower() == "assistant"
        ]
        prompt_user_turn_ids_json = json.dumps(prompt_user_turn_ids, ensure_ascii=False)
        prompt_assistant_turn_ids_for_assent_bridge_json = json.dumps(
            prompt_assistant_turn_ids_for_assent_bridge,
            ensure_ascii=False,
        )
        previous_phases = _slot_fill_previous_phases_for_scope(
            current_phase=current_phase,
            scope_mode=scope_mode,
        )
        future_phases = _slot_fill_future_phases_for_scope(
            current_phase=current_phase,
            scope_mode=scope_mode,
        )
        previous_phase_codes = {phase.name for phase in previous_phases}
        future_phase_codes = {phase.name for phase in future_phases}
        target_phase_text = (
            f"{current_phase.value}[{current_phase.name}]"
            if target_phase_codes
            else "none (current-phase updates disabled)"
        )
        regular_target_slot_keys = ", ".join(layer1_prompt_target_slot_keys) or "none (all slot quality >= 0.8)"
        all_current_phase_slot_keys = ", ".join(current_phase_required_slot_keys) or "none"
        target_slot_schema_text = (
            f"- {current_phase.value} [{current_phase.name}]: {all_current_phase_slot_keys}"
            if target_phase_codes
            else "- none"
        )
        previous_phase_text = (
            " / ".join(f"{phase.value}[{phase.name}]" for phase in previous_phases) if previous_phases else "none"
        )
        future_phase_text = (
            " / ".join(f"{phase.value}[{phase.name}]" for phase in future_phases) if future_phases else "none"
        )
        previous_slot_schema_lines: List[str] = []
        for phase in previous_phases:
            slot_keys = ", ".join(_PHASE_SLOT_SCHEMA.get(phase, ())) or "none"
            previous_slot_schema_lines.append(f"- {phase.value} [{phase.name}]: {slot_keys}")
        previous_slot_schema_text = "\n".join(previous_slot_schema_lines) if previous_slot_schema_lines else "- none"
        future_slot_schema_lines: List[str] = []
        for phase in future_phases:
            slot_keys = ", ".join(_PHASE_SLOT_SCHEMA.get(phase, ())) or "none"
            future_slot_schema_lines.append(f"- {phase.value} [{phase.name}]: {slot_keys}")
        future_slot_schema_text = "\n".join(future_slot_schema_lines) if future_slot_schema_lines else "- none"
        current_phase_lightweight_definition_text = _build_lightweight_slot_definition_text(
            phases=[current_phase],
            allowed_slot_keys_by_phase={current_phase.name: current_phase_required_slot_keys},
        )
        non_current_phase_lightweight_definition_text = _build_lightweight_slot_definition_text(
            phases=[*previous_phases, *future_phases],
        )
        if scope_mode == _SLOT_FILL_SCOPE_CURRENT_ONLY:
            lightweight_definition_scope = "current_only"
            lightweight_definition_prompt_text = (
                "Current-phase lightweight slot definitions:\n"
                f"{current_phase_lightweight_definition_text}\n"
            )
            user_lightweight_definition_text = (
                f"[Current-phase lightweight slot definitions]\n{current_phase_lightweight_definition_text}\n"
            )
        elif scope_mode == _SLOT_FILL_SCOPE_NON_CURRENT_ONLY:
            lightweight_definition_scope = "non_current_only"
            lightweight_definition_prompt_text = (
                "Non-current-phase lightweight slot definitions:\n"
                f"{non_current_phase_lightweight_definition_text}\n"
            )
            user_lightweight_definition_text = (
                f"[Non-current-phase lightweight slot definitions]\n{non_current_phase_lightweight_definition_text}\n"
            )
        else:
            lightweight_definition_scope = "mixed"
            lightweight_definition_prompt_text = (
                "Current-phase lightweight slot definitions:\n"
                f"{current_phase_lightweight_definition_text}\n"
                "Non-current-phase lightweight slot definitions:\n"
                f"{non_current_phase_lightweight_definition_text}\n"
            )
            user_lightweight_definition_text = (
                f"[Current-phase lightweight slot definitions]\n{current_phase_lightweight_definition_text}\n"
                f"[Non-current-phase lightweight slot definitions]\n{non_current_phase_lightweight_definition_text}\n"
            )
        target_behavior_focus = _get_confirmed_target_behavior_for_change_talk(state=state)
        target_behavior_alignment_active = bool(
            target_behavior_focus and current_phase in _TARGET_BEHAVIOR_ALIGNMENT_PHASES
        )
        target_behavior_alignment_phase_text = " / ".join(
            f"{phase.value}[{phase.name}]"
            for phase in _TARGET_BEHAVIOR_ALIGNMENT_PHASES
        )
        target_behavior_alignment_anchor_text = target_behavior_focus or "none"
        target_behavior_alignment_active_text = (
            "enabled" if target_behavior_alignment_active else "disabled"
        )
        scope_rules_text = ""
        if scope_mode == _SLOT_FILL_SCOPE_CURRENT_ONLY:
            scope_rules_text = (
                "- Current-phase-only mode: put only current-phase updates into `phase_slot_updates`.\n"
                "- Always keep `previous_phase_slot_updates` / `future_phase_slot_updates` as empty arrays.\n"
                "- Set `need_previous_phase_slot_updates` / `need_future_phase_slot_updates` to false.\n"
            )
        elif scope_mode == _SLOT_FILL_SCOPE_NON_CURRENT_ONLY:
            scope_rules_text = (
                "- Non-current-phase-only mode: current-phase updates are prohibited (`phase_slot_updates` must be empty).\n"
                "- Put updates for earlier phases into `previous_phase_slot_updates`.\n"
                "- Put updates for later phases into `future_phase_slot_updates`.\n"
                "- If none apply, keep each array empty.\n"
            )
        else:
            scope_rules_text = (
                "- Normally, place only current-phase updates into `phase_slot_updates`.\n"
                "- Exception: only when the dialogue history clearly shows that the user wants it and a previous phase needs completion, set `need_previous_phase_slot_updates=true` and place the update in `previous_phase_slot_updates`.\n"
                "- If that exception does not apply, set `need_previous_phase_slot_updates=false` and `previous_phase_slot_updates=[]`.\n"
                "- Exception: only when information directly relevant to a future phase is explicit (for example next action, importance, confidence, or closing intent), set `need_future_phase_slot_updates=true` and place the update in `future_phase_slot_updates`.\n"
                "- When emitting future-phase updates, allow at most one phase and two slots. Otherwise use `need_future_phase_slot_updates=false` and `future_phase_slot_updates=[]`.\n"
            )

        system = (
            "You are a phase-aware MI slot filling agent.\n"
            "Goal: extract candidate slot updates from the latest client utterance and the dialogue history, centered on the current phase.\n"
            "Return exactly one JSON object and nothing else.\n"
            f"Extraction scope: {scope_mode}\n"
            f"Target phase this turn: {target_phase_text}\n"
            f"Previous phases: {previous_phase_text}\n"
            f"Future phases: {future_phase_text}\n"
            "Update rules:\n"
            f"{scope_rules_text}"
            f"{lightweight_definition_prompt_text}"
            f"- Target-behavior alignment phase: {target_behavior_alignment_phase_text}.\n"
            f"- Target-behavior alignment anchor (confirmed, quality>=0.8): {target_behavior_alignment_anchor_text}.\n"
            f"- Target-behavior alignment enabled: {target_behavior_alignment_active_text}.\n"
            "- When target-behavior alignment is enabled, prefer updates directly related to FOCUSING_TARGET_BEHAVIOR.target_behavior.\n"
            "- Do not fill slots with unrelated generalities unless explicit_correction=true clearly justifies it.\n"
            "- Each slot value must use the shape {\"value\":\"...\",\"user_quote\":\"...\",\"user_evidence_turn_ids\":[1],\"assistant_quote\":\"...\",\"assistant_evidence_turn_ids\":[2],\"assent_bridge\":false,\"assent_bridge_source_turn_id\":0,\"confidence\":0.0-1.0,\"explicit_correction\":false}.\n"
            "- In Layer1, do not rely on existing slot values except to check the target-behavior anchor. Extract from the latest user turn, or from the immediately previous assistant turn only when assent_bridge=true.\n"
            "- Update importance_scale / confidence_scale only when the user_quote explicitly contains a 0-10 number.\n"
            "- Even with assent_bridge=true, scale values may be updated only when the previous assistant utterance named the number and the latest user utterance clearly assents to that number.\n"
            "- Final integration is handled by Layer2 Reviewer; Layer1 should focus on candidate extraction plus evidence linkage.\n"
            "- When the client explicitly corrects an earlier understanding, set explicit_correction=true.\n"
            "- user_quote / user_evidence_turn_ids must always point to user turns.\n"
            "- assistant_quote / assistant_evidence_turn_ids may be used only when assent_bridge=true.\n"
            "- Maintain alignment between quotes and turn ids.\n"
            "- Do not use counselor utterances as evidence except in the assent-bridge case.\n"
            "Required procedure:\n"
            "1) Build slot candidates from the latest user utterance and attach user evidence.\n"
            "2) If the latest user utterance is an assent response, set assent_bridge=true.\n"
            "3) When assent_bridge=true, inspect the immediately previous assistant utterance and merge only the information required for the slot value.\n"
            "4) When assent_bridge=false, set assistant_quote=\"\" and assistant_evidence_turn_ids=[].\n"
            "- For assent_bridge, do not write a bare assent token like \"yes\" as assistant_quote. Quote the relevant phrase from the previous assistant utterance or the finalized slot value phrase.\n"
            f"- Candidates with confidence < {_SLOT_APPLY_CONFIDENCE_THRESHOLD:.2f} will not be applied to state. Put any needed confirmation guidance in note.\n"
            "- Never overwrite an existing slot value with None/null/unknown/none.\n"
            "- Do not force weak guesses into slots.\n"
            "- Keep each slot value short (within 120 characters).\n"
            "- Use only allowed slot keys.\n"
            "- Human-readable fields such as value and note must be natural English. Avoid placeholders or stray code tokens.\n"
            "- Preserve user_quote / assistant_quote as quotations from the source when used.\n"
            "- In PURPOSE_CONFIRMATION, do not confuse today_focus_topic with process_need.\n"
            "- today_focus_topic should name only what the client wants to work on today.\n"
            "- process_need should describe only how the client wants the conversation to proceed.\n"
            "- If the latest user utterance contains both, return both in separate slots.\n"
            "- In PURPOSE_CONFIRMATION / CURRENT_STATUS_CHECK, if there are multiple plausible entry points and the client has not stated a preference, set focus_choice_hint.needed=true.\n"
            "- Put 2-3 short noun phrases in focus_choice_hint.candidate_topics when a choice prompt is needed.\n"
            "- If the client has already made the entry preference explicit, set focus_choice_hint.explicit_preference_present=true and focus_choice_hint.needed=false.\n"
            "- In FOCUSING_TARGET_BEHAVIOR, if the client spontaneously states a concrete behavior, target_behavior may be returned even if focus_agreement is still uncertain.\n"
            "- Do not invent new coping actions or plans. Concrete behavior must be grounded in user evidence or assent_bridge.\n"
            f"- Normally return only slot keys with quality < {layer1_target_quality_threshold:.2f}. If explicit_correction=true, already-high-quality slots may also be returned.\n"
            "- If there are no updates, return empty arrays.\n"
            "Output format example:\n"
            '{"phase_slot_updates":[{"phase":"CURRENT_STATUS_CHECK","slots":{"current_situation":{"value":"Overtime has been continuing and fatigue feels intense","user_quote":"The overtime just keeps going and I cannot shake the fatigue","user_evidence_turn_ids":[3],"assistant_quote":"","assistant_evidence_turn_ids":[],"assent_bridge":false,"assent_bridge_source_turn_id":0,"confidence":0.91,"explicit_correction":false}}}],"need_previous_phase_slot_updates":false,"previous_phase_slot_updates":[],"need_future_phase_slot_updates":false,"future_phase_slot_updates":[],"focus_choice_hint":{"needed":false,"explicit_preference_present":false,"candidate_topics":[],"reason":""},"note":""}\n'
            "Available slot keys in the current phase:\n"
            f"{target_slot_schema_text}\n"
            "Available slot keys in previous phases:\n"
            f"{previous_slot_schema_text}\n"
            "Available slot keys in future phases:\n"
            f"{future_slot_schema_text}\n"
        )
        system = inject_mi_knowledge(system, agent_name="phase_slot_filler")

        user = (
            f"[Current phase]\n{state.phase.value}\n"
            f"[Extraction scope]\n{scope_mode}\n"
            f"[Target phase this turn]\n{target_phase_text}\n"
            f"[Lightweight definition scope]\n{lightweight_definition_scope}\n"
            f"{user_lightweight_definition_text}"
            f"[FOCUSING_TARGET_BEHAVIOR.target_behavior anchor]\n{target_behavior_alignment_anchor_text}\n"
            f"[Target-behavior alignment enabled]\n{target_behavior_alignment_active_text}\n"
            f"[Layer1 target slot keys (quality<{layer1_target_quality_threshold:.2f})]\n{json.dumps(layer1_prompt_target_slot_keys, ensure_ascii=False)}\n"
            "[Exception rule]\nSlots outside the list above may also be updated when explicit_correction=true.\n"
            f"[Previous phases]\n{previous_phase_text}\n"
            f"[Future phases]\n{future_phase_text}\n"
            "[Future-phase exception]\nUse need_future_phase_slot_updates=true only when future-phase relevance is explicit.\n"
            f"[Turn index]\n{state.turn_index}\n"
            f"[User turn ids allowed as evidence]\n{prompt_user_turn_ids_json}\n"
            f"[Assistant turn ids allowed for assent bridge]\n{prompt_assistant_turn_ids_for_assent_bridge_json}\n"
            f"[Recent dialogue T{turn_start}-T{turn_end}]\n{dialogue}\n"
            f"[Current client utterance]\n{user_text}\n"
            "Return JSON containing updated slots only."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label="layer1:phase_slot_filler",
        )
        json_mode = _normalize_json_mode(self.json_mode)
        strict_json = json_mode == "strict"
        parsed = _parse_json_from_text(raw, strict=strict_json)
        strict_json_schema_rejected = False
        if isinstance(parsed, Mapping):
            parsed_payload: Mapping[str, Any] = parsed
        elif isinstance(parsed, list):
            if strict_json:
                strict_json_schema_rejected = True
                parsed_payload = {}
            else:
                parsed_payload = {"phase_slot_updates": parsed}
        else:
            if strict_json:
                strict_json_schema_rejected = True
            parsed_payload = {}
        updates = _extract_phase_slot_update_list(parsed_payload)
        model_note = _normalize_slot_text(parsed_payload.get("note") if isinstance(parsed_payload, Mapping) else "")
        focus_choice_hint = _normalize_focus_choice_hint(
            parsed_payload.get("focus_choice_hint") if isinstance(parsed_payload, Mapping) else None
        )

        previous_need_raw: Any = None
        for key in (
            "need_previous_phase_slot_updates",
            "need_prior_phase_slot_updates",
            "previous_phase_update_needed",
            "requires_previous_phase_slot_updates",
        ):
            if key in parsed_payload:
                previous_need_raw = parsed_payload.get(key)
                break
        need_previous_phase_updates_requested = _coerce_bool_with_ja(previous_need_raw, False)
        future_need_raw: Any = None
        for key in (
            "need_future_phase_slot_updates",
            "need_next_phase_slot_updates",
            "future_phase_update_needed",
            "requires_future_phase_slot_updates",
        ):
            if key in parsed_payload:
                future_need_raw = parsed_payload.get(key)
                break
        need_future_phase_updates_requested = _coerce_bool_with_ja(future_need_raw, False)

        previous_update_sources: List[List[Dict[str, Any]]] = []
        for key in (
            "previous_phase_slot_updates",
            "prior_phase_slot_updates",
            "previous_slot_updates",
            "past_phase_slot_updates",
            "previous_phase_updates",
        ):
            if key not in parsed_payload:
                continue
            parsed_prev = _extract_phase_slot_update_list_from_any(parsed_payload.get(key))
            if parsed_prev:
                previous_update_sources.append(parsed_prev)
        explicit_previous_updates = (
            _merge_phase_slot_updates(*previous_update_sources) if previous_update_sources else []
        )
        future_update_sources: List[List[Dict[str, Any]]] = []
        for key in (
            "future_phase_slot_updates",
            "next_phase_slot_updates",
            "upcoming_phase_slot_updates",
            "future_phase_updates",
        ):
            if key not in parsed_payload:
                continue
            parsed_future = _extract_phase_slot_update_list_from_any(parsed_payload.get(key))
            if parsed_future:
                future_update_sources.append(parsed_future)
        explicit_future_updates = (
            _merge_phase_slot_updates(*future_update_sources) if future_update_sources else []
        )

        target_updates: List[Dict[str, Any]] = []
        previous_updates_from_main: List[Dict[str, Any]] = []
        future_updates_from_main: List[Dict[str, Any]] = []
        dropped_non_current_phase_updates: List[Dict[str, Any]] = []
        dropped_unknown_phase_updates: List[Dict[str, Any]] = []
        for update in updates:
            phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
            if phase is None:
                dropped_unknown_phase_updates.append(update)
                continue
            if phase.name in target_phase_codes:
                target_updates.append(update)
                continue
            if phase.name in previous_phase_codes:
                previous_updates_from_main.append(update)
                continue
            if phase.name in future_phase_codes:
                future_updates_from_main.append(update)
                continue
            dropped_non_current_phase_updates.append(update)

        previous_candidate_updates = _merge_phase_slot_updates(previous_updates_from_main, explicit_previous_updates)
        previous_updates_in_range: List[Dict[str, Any]] = []
        dropped_previous_updates_out_of_range: List[Dict[str, Any]] = []
        for update in previous_candidate_updates:
            phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
            if phase is None or phase.name not in previous_phase_codes:
                dropped_previous_updates_out_of_range.append(update)
                continue
            previous_updates_in_range.append(update)
        future_candidate_updates = _merge_phase_slot_updates(future_updates_from_main, explicit_future_updates)
        future_updates_in_range: List[Dict[str, Any]] = []
        dropped_future_updates_out_of_range: List[Dict[str, Any]] = []
        for update in future_candidate_updates:
            phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
            if phase is None or phase.name not in future_phase_codes:
                dropped_future_updates_out_of_range.append(update)
                continue
            future_updates_in_range.append(update)

        if scope_mode == _SLOT_FILL_SCOPE_NON_CURRENT_ONLY:
            previous_updates_applied = list(previous_updates_in_range)
            dropped_previous_updates_by_policy = []
            future_updates_applied = list(future_updates_in_range)
            dropped_future_updates_by_policy = []
            dropped_future_updates_by_limit: List[Dict[str, Any]] = []
        else:
            previous_updates_applied = (
                previous_updates_in_range if need_previous_phase_updates_requested else []
            )
            dropped_previous_updates_by_policy = (
                [] if need_previous_phase_updates_requested else previous_updates_in_range
            )
            future_updates_allowed = (
                future_updates_in_range if need_future_phase_updates_requested else []
            )
            dropped_future_updates_by_policy = (
                [] if need_future_phase_updates_requested else future_updates_in_range
            )
            future_updates_applied = []
            dropped_future_updates_by_limit = []
            max_future_phase_count = 1
            max_future_slots_per_phase = 2
            for update in future_updates_allowed:
                if len(future_updates_applied) >= max_future_phase_count:
                    dropped_future_updates_by_limit.append(dict(update))
                    continue
                if not isinstance(update, Mapping):
                    dropped_future_updates_by_limit.append(dict(update))
                    continue
                phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
                if phase is None:
                    dropped_future_updates_by_limit.append(dict(update))
                    continue
                slots = update.get("slots")
                if not isinstance(slots, Mapping):
                    dropped_future_updates_by_limit.append(dict(update))
                    continue
                limited_slots: Dict[str, Any] = {}
                for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):
                    if slot_key not in slots:
                        continue
                    limited_slots[slot_key] = slots.get(slot_key)
                    if len(limited_slots) >= max_future_slots_per_phase:
                        break
                if not limited_slots:
                    dropped_future_updates_by_limit.append(dict(update))
                    continue
                limited_update = dict(update)
                limited_update["slots"] = limited_slots
                future_updates_applied.append(limited_update)
        layer1_prompt_target_slot_key_set = set(layer1_prompt_target_slot_keys)
        filtered_target_updates: List[Dict[str, Any]] = []
        explicit_correction_slot_keys_from_layer1: List[str] = []
        dropped_current_phase_updates_out_of_prompt_target: List[Dict[str, Any]] = []
        for update in target_updates:
            if not isinstance(update, Mapping):
                continue
            slots = update.get("slots")
            if not isinstance(slots, Mapping):
                dropped_current_phase_updates_out_of_prompt_target.append(dict(update))
                continue
            filtered_slots: Dict[str, Any] = {}
            for slot_key, slot_value in slots.items():
                normalized_slot_key = _normalize_slot_text(slot_key)
                if not normalized_slot_key:
                    continue
                explicit_correction = False
                if isinstance(slot_value, Mapping):
                    raw_explicit_correction = slot_value.get("explicit_correction")
                    if isinstance(raw_explicit_correction, bool):
                        explicit_correction = raw_explicit_correction
                    elif raw_explicit_correction not in (None, ""):
                        explicit_correction = _coerce_bool_with_ja(raw_explicit_correction, False)
                if (
                    normalized_slot_key not in layer1_prompt_target_slot_key_set
                    and not explicit_correction
                ):
                    continue
                filtered_slots[slot_key] = slot_value
                if (
                    explicit_correction
                    and normalized_slot_key not in explicit_correction_slot_keys_from_layer1
                ):
                    explicit_correction_slot_keys_from_layer1.append(normalized_slot_key)
            if not filtered_slots:
                dropped_current_phase_updates_out_of_prompt_target.append(dict(update))
                continue
            filtered_update = dict(update)
            filtered_update["slots"] = filtered_slots
            filtered_target_updates.append(filtered_update)

        filtered_updates = _merge_phase_slot_updates(
            filtered_target_updates,
            previous_updates_applied,
            future_updates_applied,
        )
        dropped_updates = (
            dropped_non_current_phase_updates
            + dropped_unknown_phase_updates
            + dropped_previous_updates_out_of_range
            + dropped_previous_updates_by_policy
            + dropped_future_updates_out_of_range
            + dropped_future_updates_by_policy
            + dropped_future_updates_by_limit
            + dropped_current_phase_updates_out_of_prompt_target
        )

        debug: Dict[str, Any] = {
            "method": "llm",
            "raw_output": raw,
            "parsed_json": parsed if isinstance(parsed, (Mapping, list)) else None,
            "json_mode": json_mode,
            "strict_json_schema_rejected": strict_json_schema_rejected,
            "scope_mode": scope_mode,
            "target_phases": sorted(target_phase_codes),
            "layer1_target_quality_threshold": round(float(layer1_target_quality_threshold), 3),
            "layer1_precheck_quality_by_slot": layer1_precheck_quality_by_slot,
            "layer1_prompt_target_slot_keys": layer1_prompt_target_slot_keys,
            "layer1_regular_target_slot_keys_text": regular_target_slot_keys,
            "current_phase_slot_context": {},
            "uses_existing_slot_context_in_prompt": target_behavior_alignment_active,
            "target_behavior_alignment_focus": target_behavior_alignment_anchor_text,
            "target_behavior_alignment_active": target_behavior_alignment_active,
            "target_behavior_alignment_phase_codes": [phase.name for phase in _TARGET_BEHAVIOR_ALIGNMENT_PHASES],
            "request_label": "layer1:phase_slot_filler",
            "lightweight_definition_scope": lightweight_definition_scope,
            "current_phase_lightweight_slot_definitions": current_phase_lightweight_definition_text,
            "non_current_phase_lightweight_slot_definitions": non_current_phase_lightweight_definition_text,
            "prompt_user_turn_ids": prompt_user_turn_ids,
            "prompt_assistant_turn_ids_for_assent_bridge": prompt_assistant_turn_ids_for_assent_bridge,
            "previous_phases": [phase.name for phase in previous_phases],
            "future_phases": [phase.name for phase in future_phases],
            "need_previous_phase_slot_updates_raw": previous_need_raw,
            "need_previous_phase_slot_updates_requested": need_previous_phase_updates_requested,
            "need_previous_phase_slot_updates": bool(previous_updates_applied),
            "need_future_phase_slot_updates_raw": future_need_raw,
            "need_future_phase_slot_updates_requested": need_future_phase_updates_requested,
            "need_future_phase_slot_updates": bool(future_updates_applied),
            "target_phase_updates": filtered_target_updates,
            "previous_phase_updates_from_main": previous_updates_from_main,
            "explicit_previous_phase_updates": explicit_previous_updates,
            "previous_phase_updates_candidate": previous_updates_in_range,
            "previous_phase_updates_applied": previous_updates_applied,
            "dropped_previous_phase_updates_by_policy": dropped_previous_updates_by_policy,
            "dropped_previous_phase_updates_out_of_range": dropped_previous_updates_out_of_range,
            "future_phase_updates_from_main": future_updates_from_main,
            "explicit_future_phase_updates": explicit_future_updates,
            "future_phase_updates_candidate": future_updates_in_range,
            "future_phase_updates_applied": future_updates_applied,
            "dropped_future_phase_updates_by_policy": dropped_future_updates_by_policy,
            "dropped_future_phase_updates_out_of_range": dropped_future_updates_out_of_range,
            "dropped_future_phase_updates_by_limit": dropped_future_updates_by_limit,
            "dropped_current_phase_updates_out_of_prompt_target": dropped_current_phase_updates_out_of_prompt_target,
            "explicit_correction_slot_keys_from_layer1": explicit_correction_slot_keys_from_layer1,
            "dropped_updates_non_current_phase": dropped_non_current_phase_updates,
            "dropped_updates_unknown_phase": dropped_unknown_phase_updates,
            "updates": filtered_updates,
            "dropped_updates_out_of_target_phase": dropped_updates,
            "note": model_note,
            "focus_choice_hint": focus_choice_hint,
        }
        return filtered_updates, debug


@dataclass
class RuleBasedSlotReviewer:
    """
    Deterministic reviewer for candidates contained in Layer1Bundle.
    """

    def review(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
        layer1_bundle: Layer1Bundle,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        reviewed = _build_rule_slot_review(layer1_bundle)
        debug = {
            "method": "rule_slot_reviewer",
            "reviewed_count": len(reviewed.get("reviewed_updates") or []),
            "quality_mean": reviewed.get("quality_mean"),
            "quality_min": reviewed.get("quality_min"),
        }
        return reviewed, debug


@dataclass
class RuleBasedNonCurrentSlotReviewer:
    """
    Layer2 deterministic reviewer for non-current-phase candidates.
    - Return only reviewed_updates and slot quality, without repair hints.
    """

    def review(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
        layer1_bundle: Layer1Bundle,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        _ = history
        _ = state
        _ = features
        reviewed = _build_rule_slot_review(layer1_bundle)
        reviewed["slot_quality_target_examples"] = []
        reviewed["slot_repair_hints"] = []
        debug = {
            "method": "rule_non_current_slot_reviewer",
            "reviewed_count": len(reviewed.get("reviewed_updates") or []),
            "quality_mean": reviewed.get("quality_mean"),
            "quality_min": reviewed.get("quality_min"),
        }
        return reviewed, debug


@dataclass
class LLMSlotReviewer:
    """
    Layer2 reviewer that audits Layer1 extraction candidates and returns quality plus repair hints.
    - Do not invent new slots; review only Layer1 candidates.
    - Prefer the LLM judgment and fall back only for schema normalization issues.
    """

    llm: LLMClient
    temperature: float = 0.0
    max_history_turns: int = 8
    quality_pass_threshold: float = 0.62
    json_mode: str = "loose"

    def review(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
        layer1_bundle: Layer1Bundle,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dialogue, turn_start, turn_end = _history_to_numbered_dialogue(
            history,
            max_turns=self.max_history_turns,
        )
        bundle_json = json.dumps(dataclasses.asdict(layer1_bundle), ensure_ascii=False)
        feature_json = json.dumps(dataclasses.asdict(features), ensure_ascii=False)
        review_target_threshold = _clamp01(
            layer1_bundle.review_target_quality_threshold,
            _SLOT_LAYER2_REVIEW_TARGET_QUALITY_THRESHOLD,
        )
        review_target_slot_keys_json = json.dumps(layer1_bundle.review_target_slot_keys, ensure_ascii=False)
        review_target_slot_refs_json = json.dumps(layer1_bundle.review_target_slot_refs, ensure_ascii=False)
        current_quality_context_json = json.dumps(
            _build_current_phase_slot_quality_context(state),
            ensure_ascii=False,
        )
        rubric_text = _build_phase_slot_quality_rubric_prompt_text(state.phase)

        system = (
            "You are the Layer2 Slot Reviewer for MI counseling.\n"
            "Audit only the slot candidates extracted by Layer1 and return quality ratings plus repair hints.\n"
            "Constraints:\n"
            f"- Review only Layer1Bundle.review_target_slot_keys, defined as quality < {review_target_threshold:.2f} or unset.\n"
            "- Stay within Layer1Bundle.review_target_slot_refs and candidate_updates.\n"
            "- Do not invent new slot keys.\n"
            "- Return reviewed_updates with revised_value, quality_score, issue_codes, evidence_items, and review_note.\n"
            "- Do not return decision / admissible / quality_action / slot_quality / quality_mean / quality_min / blocking_issue_codes.\n"
            "- Do not re-estimate features or decide phase progression.\n"
            "- When candidate_variants include both existing memory and a latest user update, compare them and output one final integrated value.\n"
            "- Keep concrete details. Do not abstract away time, place, frequency, duration, or conditions when they are grounded.\n"
            "- For importance_scale / confidence_scale, accept only explicit 0-10 numeric evidence from user turns.\n"
            f"- If explicit numeric evidence exists for importance_scale / confidence_scale, keep quality_score >= {_EXPLICIT_NUMERIC_SCALE_QUALITY_FLOOR:.2f}.\n"
            "- If no numeric evidence exists, keep the value provisional, mark wrong_format and low_confidence, and keep quality_score <= 0.59.\n"
            "- For commitment_level, accept either a 0-10 number or qualitative labels such as low / medium / high.\n"
            "- Evaluate commitment_level by clarity of intent and conditions, not by the magnitude of the number itself.\n"
            "- today_focus_topic deserves quality >= 0.80 only when the client explicitly confirms the topic, either spontaneously or by clear assent to the previous assistant summary.\n"
            "- focus_agreement deserves quality >= 0.80 only when the client clearly assents or re-states the focus.\n"
            "- target_behavior must be an observable action, reproducible coping response, or proximal action that improves the state.\n"
            "- Do not invent new coping plans for target_behavior.\n"
            "- quality_score may rise but must not go below the existing value.\n"
            f"- If an existing quality_score is already >= {self.quality_pass_threshold:.2f} and evidence is not contradicted, keep it as pass.\n"
            "- Always return quality_score for every reviewed slot.\n"
            "- evidence_items must include role, quote, and turn_ids.\n"
            "- Apply missing_evidence only when no valid quote+turn_ids evidence exists.\n"
            "- slot_quality_target_examples must be actionable missing-information candidates for the next response, not audit notes.\n"
            f"- Return one slot_quality_target_examples item per current-phase slot with quality < {review_target_threshold:.2f}; if none, return [].\n"
            f"- Return at most {_MAX_SLOT_REPAIR_HINTS_PER_TURN} hint items and do not include other-phase hints.\n"
            "- slot_label must be a human-readable content label, not an internal schema label.\n"
            f"- target_information should briefly name the missing information needed to move the slot toward quality {_SLOT_REPAIR_HINT_TARGET_QUALITY:.2f}.\n"
            "- detail must be one natural first-person user utterance example, not a question or counselor instruction.\n"
            "- detail must not contain internal labels, meta commentary, or generic placeholders.\n"
            "- If you output slot_quality_target_examples, mirror detail into quality_upgrade_model_text.\n"
            "Return exactly one JSON object.\n"
            "Output format:\n"
            "{\n"
            '  "reviewed_updates":[{"phase":"...","phase_code":"...","slot_key":"...","original_value":"...","revised_value":"...","evidence_items":[{"role":"user","quote":"...","turn_ids":[1]},{"role":"assistant","quote":"...","turn_ids":[2]}],"extraction_confidence":0.8,"quality_score":0.7,"issue_codes":["..."],"review_note":"..."}],\n'
            '  "slot_quality_target_examples":[{"slot_key":"...","slot_label":"...","issue_type":"missing_evidence|low_confidence|wrong_format|too_vague","preferred_probe_style":"...","target_information":"missing information needed for the slot","detail":"\"If the client said this next, the slot quality would improve\"","quality_upgrade_model_text":"same as detail"}],\n'
            '  "review_summary_note":"..."\n'
            "}"
        )
        system = inject_mi_knowledge(system, agent_name="slot_reviewer")
        user = (
            f"[Current phase]\n{state.phase.value}\n"
            f"[Features]\n{feature_json}\n"
            f"[Confirmed current-phase slots: value / status / current quality]\n{current_quality_context_json}\n"
            f"[Phase-specific slot rubric]\n{rubric_text}\n"
            f"[Layer2 review target slot keys]\n{review_target_slot_keys_json}\n"
            f"[Layer2 review target refs]\n{review_target_slot_refs_json}\n"
            f"[Dialogue T{turn_start}-T{turn_end}]\n{dialogue}\n"
            f"[Layer1Bundle]\n{bundle_json}\n"
            "Audit only Layer1Bundle.review_target_slot_keys and return JSON."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        request_label = "layer2:slot_reviewer"
        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label=request_label,
        )
        json_mode = _normalize_json_mode(self.json_mode)
        strict_json = json_mode == "strict"
        parsed = _parse_json_from_text(raw, strict=strict_json)
        strict_json_schema_rejected = strict_json and not isinstance(parsed, Mapping)
        normalized, issues = _normalize_slot_review_payload(
            payload=parsed,
            layer1_bundle=layer1_bundle,
        )
        contract_issues = [issue for issue in issues if _is_slot_review_contract_issue(issue)]
        if contract_issues:
            debug = {
                "method": "llm_slot_reviewer_contract_reject",
                "raw_output": raw,
                "parsed": parsed if isinstance(parsed, Mapping) else None,
                "schema_issues": issues,
                "contract_issues": contract_issues,
                "json_mode": json_mode,
                "strict_json_schema_rejected": strict_json_schema_rejected,
                "allow_rule_fallback": False,
                "reviewed_count": 0,
                "quality_mean": 0.0,
                "quality_min": 0.0,
                "request_label": request_label,
            }
            return _empty_slot_review_payload(), debug
        if strict_json and (strict_json_schema_rejected or issues):
            debug = {
                "method": "llm_slot_reviewer_strict_reject",
                "raw_output": raw,
                "parsed": parsed if isinstance(parsed, Mapping) else None,
                "schema_issues": issues,
                "json_mode": json_mode,
                "strict_json_schema_rejected": strict_json_schema_rejected,
                "allow_rule_fallback": False,
                "reviewed_count": 0,
                "quality_mean": 0.0,
                "quality_min": 0.0,
                "request_label": request_label,
            }
            return _empty_slot_review_payload(), debug

        debug = {
            "method": "llm_slot_reviewer",
            "raw_output": raw,
            "parsed": parsed if isinstance(parsed, Mapping) else None,
            "schema_issues": issues,
            "json_mode": json_mode,
            "strict_json_schema_rejected": strict_json_schema_rejected,
            "allow_rule_fallback": True,
            "reviewed_count": len(normalized.get("reviewed_updates") or []),
            "quality_mean": normalized.get("quality_mean"),
            "quality_min": normalized.get("quality_min"),
            "request_label": request_label,
        }
        return normalized, debug


@dataclass
class LLMNonCurrentSlotReviewer:
    """
    Layer2 reviewer dedicated to auditing non-current-phase candidates.
    - Return only reviewed_updates / quality and omit slot_quality_target_examples.
    """

    llm: LLMClient
    temperature: float = 0.0
    max_history_turns: int = 8
    quality_pass_threshold: float = 0.62
    json_mode: str = "loose"

    def review(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
        layer1_bundle: Layer1Bundle,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dialogue, turn_start, turn_end = _history_to_numbered_dialogue(
            history,
            max_turns=self.max_history_turns,
        )
        bundle_json = json.dumps(dataclasses.asdict(layer1_bundle), ensure_ascii=False)
        feature_json = json.dumps(dataclasses.asdict(features), ensure_ascii=False)
        review_target_threshold = _clamp01(
            layer1_bundle.review_target_quality_threshold,
            _SLOT_LAYER2_REVIEW_TARGET_QUALITY_THRESHOLD,
        )
        review_target_slot_keys_json = json.dumps(layer1_bundle.review_target_slot_keys, ensure_ascii=False)
        review_target_slot_refs_json = json.dumps(layer1_bundle.review_target_slot_refs, ensure_ascii=False)
        target_phase_codes = layer1_bundle.target_phase_codes or _layer1_bundle_target_phase_codes(layer1_bundle)
        target_phase_labels = [
            f"{phase.value}[{phase.name}]"
            for phase in (
                _parse_phase_from_any(phase_code)
                for phase_code in target_phase_codes
            )
            if isinstance(phase, Phase)
        ]
        all_phase_status = _build_all_phase_slot_status_for_prompt(state)

        system = (
            "You are the Layer2 reviewer for non-current-phase MI slot candidates.\n"
            "Role: review only Layer1 candidates outside the current phase and return merged `reviewed_updates`.\n"
            "Constraints:\n"
            f"- Limit review targets to `quality < {review_target_threshold:.2f}` (including unset quality) using `Layer1Bundle.review_target_slot_keys` / `review_target_slot_refs`.\n"
            "- Restrict each `phase_code+slot_key` pair to the range covered by `Layer1Bundle.review_target_slot_refs` / `candidate_updates`.\n"
            "- Do not add any new `slot_key` that is not already present in Layer1 candidates.\n"
            "- Do not make phase-progression decisions.\n"
            "- Do not re-estimate `features`.\n"
            "- Return merged results (`revised_value`), `quality_score`, `issue_codes`, `evidence_items`, and `review_note` inside `reviewed_updates`.\n"
            "- Do not return `slot_quality_target_examples` / `slot_repair_hints`.\n"
            "- When a slot has both `existing_slot_memory` and `latest_user_update` in `candidate_updates[].candidate_variants`, compare them and merge to one final value.\n"
            "- `reviewed_updates[].original_value / revised_value` must reflect the final judgment after comparison.\n"
            "- Preserve specificity: `revised_value` must not become more abstract than the specific information already present in `original_value` or candidate variants.\n"
            "- `today_focus_topic` may reach `quality_score >= 0.80` only when the topic is confirmed by the client's own spontaneous statement or explicit assent to the immediately preceding assistant summary.\n"
            "- If the client is merely following the interviewer's direction, or only choosing which topic to start with, keep `today_focus_topic <= 0.79`.\n"
            "- `target_behavior` is admissible only when it names an observable behavior, a reproducible coping action in a concrete situation, or a proximal action that improves the state.\n"
            "- If `target_behavior` stays at the level of exploring values/meaning only, assign `target_behavior_not_actionable` and `wrong_format`.\n"
            "- `focus_agreement` may reach `quality_score >= 0.80` only when there is a clear yes, spontaneous restatement, or explicit assent to the immediately preceding assistant summary.\n"
            "- Choosing which subtopic to address first is only a subtopic preference and does not count as evidence for `focus_agreement`.\n"
            "- When `target_behavior` is not yet settled, `focus_agreement` may refer to `today_focus_topic` or `change_direction`, but if there is a `target_behavior` candidate it must explicitly include that behavior or a close paraphrase.\n"
            "- If the client spontaneously states a concrete behavior, `target_behavior` may still reach `quality_score >= 0.80` even if `focus_agreement` is unresolved.\n"
            "- When explicit assent to the preceding assistant summary includes `target_behavior`, both `target_behavior` and `focus_agreement` may reach `quality_score >= 0.80`.\n"
            "- If `focus_agreement` does not actually point to that content, keep it at `<= 0.79` and add `low_confidence`.\n"
            "- Do not invent new coping actions or plans as fixed `target_behavior` values. Use only content supported by user evidence or an assent bridge.\n"
            "- For `semantic_review_pending=true`, judge whether `user_quote` is semantically supported by the specified user turn, including paraphrase/summary cases.\n"
            "- Never lower an existing `quality_score`; update only when it rises.\n"
            f"- Pass criterion: if existing `quality_score` is already >= {self.quality_pass_threshold:.2f}, keep it as pass unless the evidence contradicts it.\n"
            "- List evidence in `reviewed_updates[].evidence_items` with `role`, `quote`, and `turn_ids`.\n"
            "- Add `missing_evidence` only when `reviewed_updates[].evidence_items` contains no valid quote+turn_ids evidence at all.\n"
            "- Return exactly one JSON object.\n"
            "Output format:\n"
            "{\n"
            '  "reviewed_updates":[{"phase":"...","phase_code":"...","slot_key":"...","original_value":"...","revised_value":"...","evidence_items":[{"role":"user","quote":"...","turn_ids":[1]}],"extraction_confidence":0.8,"quality_score":0.7,"issue_codes":["..."],"review_note":"..."}],\n'
            '  "review_summary_note":"..."\n'
            "}"
        )
        system = inject_mi_knowledge(system, agent_name="slot_reviewer")
        user = (
            f"[Current phase] {state.phase.value}\n"
            f"[Review target phases] {' / '.join(target_phase_labels) if target_phase_labels else 'none'}\n"
            f"[Features] {feature_json}\n"
            f"[Confirmed slot state across all phases]\n{all_phase_status}\n"
            f"[Layer2 review target slots (quality<{review_target_threshold:.2f}, including unset)] {review_target_slot_keys_json}\n"
            f"[Layer2 review target refs (phase_code.slot_key)] {review_target_slot_refs_json}\n"
            f"[History (T{turn_start}-T{turn_end})]\n{dialogue}\n"
            f"[Layer1Bundle] {bundle_json}\n"
            "Review only the non-current-phase `review_target_slot_refs` and return JSON."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        request_label = "layer2:slot_reviewer_non_current"
        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label=request_label,
        )
        json_mode = _normalize_json_mode(self.json_mode)
        strict_json = json_mode == "strict"
        parsed = _parse_json_from_text(raw, strict=strict_json)
        strict_json_schema_rejected = strict_json and not isinstance(parsed, Mapping)
        normalized, issues = _normalize_slot_review_payload(
            payload=parsed,
            layer1_bundle=layer1_bundle,
            allow_slot_quality_target_examples=False,
            validate_current_phase_slot_repair_hints=False,
            require_current_phase_slot_repair_hints=False,
        )
        normalized["slot_quality_target_examples"] = []
        normalized["slot_repair_hints"] = []
        if strict_json and (strict_json_schema_rejected or issues):
            debug = {
                "method": "llm_non_current_slot_reviewer_strict_reject",
                "raw_output": raw,
                "parsed": parsed if isinstance(parsed, Mapping) else None,
                "schema_issues": issues,
                "json_mode": json_mode,
                "strict_json_schema_rejected": strict_json_schema_rejected,
                "allow_rule_fallback": False,
                "reviewed_count": 0,
                "quality_mean": 0.0,
                "quality_min": 0.0,
                "request_label": request_label,
            }
            return _empty_slot_review_payload(), debug

        debug = {
            "method": "llm_non_current_slot_reviewer",
            "raw_output": raw,
            "parsed": parsed if isinstance(parsed, Mapping) else None,
            "schema_issues": issues,
            "json_mode": json_mode,
            "strict_json_schema_rejected": strict_json_schema_rejected,
            "allow_rule_fallback": True,
            "reviewed_count": len(normalized.get("reviewed_updates") or []),
            "quality_mean": normalized.get("quality_mean"),
            "quality_min": normalized.get("quality_min"),
            "request_label": request_label,
        }
        return normalized, debug


@dataclass
class LLMActionRanker:
    """
    Use the LLM to score candidate main actions on the same footing and propose the strongest one.
    Returns:
      - ordered_main_actions: ranking used as a rule-side bias (MainAction)
      - debug: metadata such as proposed_main_action
    """

    llm: LLMClient
    temperature: float = 0.2
    max_history_turns: int = 6
    summary_interval_turns: int = 9

    def rank(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
        allowed_actions: Optional[Sequence[MainAction]] = None,
    ) -> Tuple[List[MainAction], Dict[str, Any]]:
        dialogue = _history_to_dialogue(history, max_turns=self.max_history_turns)
        feature_json = json.dumps(dataclasses.asdict(features), ensure_ascii=False)
        allowed = _normalize_ranker_action_space(allowed_actions)
        allowed_labels = [action.value for action in allowed]

        phase_order: Tuple[Phase, ...] = (
            Phase.GREETING,
            Phase.PURPOSE_CONFIRMATION,
            Phase.CURRENT_STATUS_CHECK,
            Phase.FOCUSING_TARGET_BEHAVIOR,
            Phase.IMPORTANCE_PROMOTION,
            Phase.CONFIDENCE_PROMOTION,
            Phase.NEXT_STEP_DECISION,
            Phase.REVIEW_REFLECTION,
            Phase.CLOSING,
        )
        current_phase_idx = phase_order.index(state.phase)
        ask_permission_phase_gate_open = current_phase_idx >= phase_order.index(Phase.CURRENT_STATUS_CHECK)

        recent_actions = [a for a in state.last_actions if isinstance(a, MainAction)]
        phase_recent_len = min(max(state.phase_turns, 0), len(recent_actions))
        phase_recent_actions = recent_actions[-phase_recent_len:] if phase_recent_len > 0 else []
        phase_simple_or_complex_reflect_count = sum(
            1
            for a in phase_recent_actions
            if a in (MainAction.REFLECT, MainAction.REFLECT_SIMPLE, MainAction.REFLECT_COMPLEX)
        )
        phase_double_reflect_count = sum(1 for a in phase_recent_actions if a == MainAction.REFLECT_DOUBLE)
        last10_actions = recent_actions[-10:]
        has_ask_permission_in_last10 = any(a == MainAction.ASK_PERMISSION_TO_SHARE_INFO for a in last10_actions)
        question_like_actions: Tuple[MainAction, ...] = (
            MainAction.QUESTION,
            MainAction.SCALING_QUESTION,
            MainAction.CLARIFY_PREFERENCE,
            MainAction.ASK_PERMISSION_TO_SHARE_INFO,
            MainAction.PROVIDE_INFO,
        )
        last_action = recent_actions[-1] if recent_actions else None

        def _count_rrq_cycles(actions: Sequence[MainAction]) -> int:
            cycles = 0
            i = 0
            while i + 2 < len(actions):
                if (
                    _is_reflect_action(actions[i])
                    and _is_reflect_action(actions[i + 1])
                    and actions[i + 2] in (MainAction.QUESTION, MainAction.SCALING_QUESTION)
                ):
                    cycles += 1
                    i += 3
                else:
                    i += 1
            return cycles

        def _build_cadence_prior() -> Tuple[Dict[str, float], Dict[str, Any]]:
            prior: Dict[MainAction, float] = {a: 1.0 for a in MainAction}
            reasons: List[str] = []
            question_actions: Tuple[MainAction, ...] = (MainAction.QUESTION, MainAction.SCALING_QUESTION)

            def apply_question_weight(weight: float) -> None:
                for question_action in question_actions:
                    prior[question_action] *= weight

            if _is_scaling_question_phase(state.phase):
                prior[MainAction.SCALING_QUESTION] *= 1.12
                prior[MainAction.QUESTION] *= 0.96
                reasons.append("prefer_scaling_question_in_scaling_phase")
            else:
                prior[MainAction.SCALING_QUESTION] *= 0.80
                reasons.append("scaling_question_down_outside_scaling_phase")

            if state.r_since_q <= 0:
                apply_question_weight(0.62)
                prior[MainAction.REFLECT_SIMPLE] *= 1.15
                prior[MainAction.REFLECT_COMPLEX] *= 1.15
                prior[MainAction.REFLECT_DOUBLE] *= 1.10
                reasons.append("question_down_when_r_since_q_0")
            elif state.r_since_q == 1:
                apply_question_weight(0.82)
                prior[MainAction.REFLECT_SIMPLE] *= 1.08
                prior[MainAction.REFLECT_COMPLEX] *= 1.08
                reasons.append("question_slight_down_when_r_since_q_1")
            elif state.r_since_q >= 2:
                apply_question_weight(1.12)
                reasons.append("question_up_after_two_non_question_turns")

            if last_action in question_like_actions:
                apply_question_weight(0.55)
                prior[MainAction.REFLECT_SIMPLE] *= 1.12
                prior[MainAction.REFLECT_COMPLEX] *= 1.12
                reasons.append("downweight_consecutive_question")

            if state.reflect_streak >= 2:
                apply_question_weight(1.08)
                reasons.append("question_rebound_after_reflect_streak")

            recent_two_actions = recent_actions[-2:]
            if (
                len(recent_two_actions) == 2
                and state.phase not in {Phase.REVIEW_REFLECTION, Phase.CLOSING}
                and all(_is_reflect_action(action) for action in recent_two_actions)
                and all(
                    _reflection_style_from_action(action) == ReflectionStyle.COMPLEX
                    for action in recent_two_actions
                )
            ):
                prior[MainAction.REFLECT_COMPLEX] *= 0.82
                prior[MainAction.REFLECT_SIMPLE] *= 1.12
                apply_question_weight(1.04)
                reasons.append("downweight_third_consecutive_complex_reflect")

            if features.resistance >= 0.6:
                apply_question_weight(0.80)
                prior[MainAction.REFLECT_COMPLEX] *= 1.10
                reasons.append("question_down_for_high_resistance")

            if features.user_is_question:
                apply_question_weight(0.55)
                prior[MainAction.REFLECT_SIMPLE] *= 1.10
                prior[MainAction.REFLECT_COMPLEX] *= 1.12
                reasons.append("question_down_for_user_question")

            if features.is_short_reply:
                apply_question_weight(1.06)
                reasons.append("question_up_for_short_reply")

            if features.novelty < 0.25:
                prior[MainAction.REFLECT_SIMPLE] *= 0.94
                prior[MainAction.REFLECT_COMPLEX] *= 0.90
                reasons.append("reflect_down_for_low_novelty")
                if state.reflect_streak >= 1:
                    apply_question_weight(1.08)
                    prior[MainAction.SUMMARY] *= 1.12
                    reasons.append("progress_push_after_reflect_on_low_novelty")

            if state.phase == Phase.CLOSING:
                apply_question_weight(0.72)
                prior[MainAction.REFLECT_SIMPLE] *= 1.04
                prior[MainAction.REFLECT_COMPLEX] *= 1.12
                reasons.append("closing_phase_reflect_over_question")

            summary_interval = max(1, int(self.summary_interval_turns))
            summary_preheat_floor = max(0, summary_interval - 1)
            turns_since_summary = max(0, int(state.turns_since_summary))
            if turns_since_summary < summary_preheat_floor:
                prior[MainAction.SUMMARY] *= 0.72
                reasons.append("summary_down_during_cooldown")
            elif turns_since_summary >= summary_interval:
                prior[MainAction.SUMMARY] *= 1.25
                reasons.append("summary_up_after_interval")
            else:
                if features.need_summary:
                    prior[MainAction.SUMMARY] *= 1.08
                    reasons.append("summary_slight_up_near_interval_with_need_summary")
                else:
                    prior[MainAction.SUMMARY] *= 0.92
                    reasons.append("summary_hold_near_interval_without_need_summary")

            if features.need_summary:
                prior[MainAction.SUMMARY] *= 1.10
                reasons.append("summary_up_for_need_summary")

            if features.novelty < 0.25 and rrq_cycles_in_last10 >= 2:
                prior[MainAction.SUMMARY] *= 1.10
                reasons.append("summary_up_for_low_novelty_after_rrq")

            normalized = {a.value: round(max(0.25, min(2.0, w)), 3) for a, w in prior.items()}
            debug = {
                "r_since_q": state.r_since_q,
                "reflect_streak": state.reflect_streak,
                "turns_since_summary": state.turns_since_summary,
                "summary_interval_turns": self.summary_interval_turns,
                "last_action": last_action.value if isinstance(last_action, MainAction) else None,
                "reasons": reasons,
            }
            return normalized, debug

        rrq_cycles_in_last10 = _count_rrq_cycles(last10_actions)
        cadence_prior, cadence_prior_debug = _build_cadence_prior()
        ranker_signal = {
            "phase_code": state.phase.name,
            "allowed_actions": allowed_labels,
            "phase_turns": state.phase_turns,
            "phase_recent_actions": [a.value for a in phase_recent_actions],
            "phase_reflect_simple_or_complex_count": phase_simple_or_complex_reflect_count,
            "phase_reflect_double_count": phase_double_reflect_count,
            "last10_actions": [a.value for a in last10_actions],
            "has_ask_permission_in_last10": has_ask_permission_in_last10,
            "rrq_cycles_in_last10_actions": rrq_cycles_in_last10,
            "ask_permission_phase_gate_open": ask_permission_phase_gate_open,
            "info_mode": state.info_mode.value,
            "has_permission_signal": features.has_permission,
            "user_requests_info_signal": features.user_requests_info,
            "novelty_signal": features.novelty,
            "turns_since_summary": state.turns_since_summary,
            "summary_interval_turns": self.summary_interval_turns,
            "cadence_prior": cadence_prior,
            "cadence_prior_debug": cadence_prior_debug,
        }
        ranker_signal_json = json.dumps(ranker_signal, ensure_ascii=False)

        system = (
            "You are an assistant that proposes the counselor's next main action in an MI dialogue.\n"
            "Choose exactly one `main_action`, and it must come from `allowed_actions`.\n"
            "Never choose anything outside `allowed_actions`.\n"
            "Prioritize MI principles (empathy, autonomy support, rolling with resistance) and the flow of the dialogue.\n"
            "If useful, return up to the top 3 ranked candidates in `rank`.\n"
            "`confidence` may be returned as a number from 0.00 to 1.00 (optional).\n"
            "Output JSON only. Example:\n"
            '{"main_action":"REFLECT_COMPLEX","rank":["REFLECT_COMPLEX","QUESTION","SUMMARY"],"confidence":0.82}\n'
            "If you return `rank`, use labels from `allowed_actions` only."
        )
        system = inject_mi_knowledge(system, agent_name="action_ranker")
        user = (
            f"[Current phase]{state.phase.value}\n"
            f"[allowed_actions]{', '.join(allowed_labels)}\n"
            f"[Features / reference]{feature_json}\n"
            f"[Action-history signals / constraints]{ranker_signal_json}\n"
            f"[Recent dialogue]\n{dialogue}\n"
            "Make the proposal in line with MI principles (empathy, rolling with resistance, autonomy support)."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        request_label = "layer2:action_ranker"
        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label=request_label,
        )
        parsed = _parse_json_from_text(raw)
        rank_list: List[Any] = []
        main_action_raw: Any = None
        confidence_raw: Any = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("rank"), list):
                rank_list = parsed.get("rank", [])
            main_action_raw = parsed.get("main_action")
            if main_action_raw is None:
                main_action_raw = parsed.get("action")
            confidence_raw = parsed.get("confidence")
            if confidence_raw is None:
                confidence_raw = parsed.get("main_action_confidence")
        elif isinstance(parsed, list):
            rank_list = parsed

        def _to_choice_label(val: Any) -> Optional[str]:
            if val is None:
                return None
            s = str(val).strip()
            if not s:
                return None
            u = s.upper().replace("-", "_").replace(" ", "_")

            # Accept legacy localized labels as well.
            if ("simple" in s.lower() or "\u5358\u7d14" in s or "\u30b7\u30f3\u30d7\u30eb" in s) and ("reflection" in s.lower() or "\u805e\u304d\u8fd4\u3057" in s):
                return "REFLECT_SIMPLE"
            if ("complex" in s.lower() or "\u8907\u96d1" in s) and ("reflection" in s.lower() or "\u805e\u304d\u8fd4\u3057" in s):
                return "REFLECT_COMPLEX"
            if ("double" in s.lower() or "double-sided" in s.lower() or "\u4e21\u4fa1" in s or "\u4e21\u9762" in s) and ("reflection" in s.lower() or "\u805e\u304d\u8fd4\u3057" in s):
                return "REFLECT_DOUBLE"
            if "scaling" in s.lower() or "\u30b9\u30b1\u30fc\u30ea\u30f3\u30b0" in s or "scale" in s:
                return "SCALING_QUESTION"
            if "question" in s.lower():
                return "QUESTION"
            if "summary" in s.lower():
                return "SUMMARY"
            if ("preference" in s.lower() or "intent" in s.lower() or "which" in s.lower() or "\u9078\u597d" in s or "\u610f\u5411" in s or "\u3069\u3061\u3089" in s) and ("check" in s.lower() or "question" in s.lower() or "\u78ba\u8a8d" in s):
                return "CLARIFY_PREFERENCE"
            if "permission" in s.lower() or "\u8a31\u53ef" in s:
                return "ASK_PERMISSION_TO_SHARE_INFO"
            if "provide_info" in s.lower() or "information sharing" in s.lower() or "\u60c5\u5831\u63d0\u4f9b" in s:
                return "PROVIDE_INFO"

            if u in {"REFLECT_SIMPLE", "SIMPLE_REFLECT", "SIMPLE_REFLECTION", "REFLECTION_SIMPLE"}:
                return "REFLECT_SIMPLE"
            if u in {"REFLECT_COMPLEX", "COMPLEX_REFLECT", "COMPLEX_REFLECTION", "REFLECTION_COMPLEX"}:
                return "REFLECT_COMPLEX"
            if u in {
                "REFLECT_DOUBLE",
                "DOUBLE_REFLECT",
                "DOUBLE_SIDED_REFLECT",
                "DOUBLE_SIDED_REFLECTION",
                "AMBIVALENCE_REFLECT",
                "AMBIVALENT_REFLECTION",
            }:
                return "REFLECT_DOUBLE"
            if u in {
                "SCALING_QUESTION",
                "SCALE_QUESTION",
                "SCALEQUESTION",
                "SCALINGQUESTION",
                "IMPORTANCE_SCALE_QUESTION",
                "CONFIDENCE_SCALE_QUESTION",
            }:
                return "SCALING_QUESTION"
            if u in {"QUESTION", "ASK_QUESTION"}:
                return "QUESTION"
            if u in {"SUMMARY", "SUMMARIZE", "SUMMARISE"}:
                return "SUMMARY"
            if u in {
                "CLARIFY_PREFERENCE",
                "CLARIFY",
                "CLARIFY_INTENT",
                "PREFERENCE_CLARIFY",
                "CLARIFY_CHOICE",
            }:
                return "CLARIFY_PREFERENCE"
            if u == "ASK_PERMISSION_TO_SHARE_INFO":
                return "ASK_PERMISSION_TO_SHARE_INFO"
            if u in {"PROVIDE_INFO", "PROVIDE_INFORMATION", "INFO", "PROVIDE"}:
                return "PROVIDE_INFO"
            if u == "REFLECT":
                return "REFLECT_COMPLEX"
            return None

        def _choice_to_action(choice: str) -> MainAction:
            if choice == "REFLECT_SIMPLE":
                return MainAction.REFLECT_SIMPLE
            if choice == "REFLECT_COMPLEX":
                return MainAction.REFLECT_COMPLEX
            if choice == "REFLECT_DOUBLE":
                return MainAction.REFLECT_DOUBLE
            if choice == "SCALING_QUESTION":
                return MainAction.SCALING_QUESTION
            if choice == "QUESTION":
                return MainAction.QUESTION
            if choice == "SUMMARY":
                return MainAction.SUMMARY
            if choice == "CLARIFY_PREFERENCE":
                return MainAction.CLARIFY_PREFERENCE
            if choice == "ASK_PERMISSION_TO_SHARE_INFO":
                return MainAction.ASK_PERMISSION_TO_SHARE_INFO
            if choice == "PROVIDE_INFO":
                return MainAction.PROVIDE_INFO
            return MainAction.REFLECT_COMPLEX

        choice_rank: List[str] = []
        seen_choice = set()
        for v in rank_list:
            c = _to_choice_label(v)
            if c and c not in seen_choice:
                choice_rank.append(c)
                seen_choice.add(c)

        proposed_choice = _to_choice_label(main_action_raw)
        if proposed_choice is None and choice_rank:
            proposed_choice = choice_rank[0]
        if proposed_choice is None:
            proposed_choice = allowed_labels[0]
        if proposed_choice not in seen_choice:
            choice_rank = [proposed_choice] + choice_rank
            seen_choice.add(proposed_choice)

        proposed_main_action = _choice_to_action(proposed_choice)
        allowed_set = set(allowed)
        proposed_in_allowed = proposed_main_action in allowed_set

        ordered: List[MainAction] = []
        seen_actions = set()
        for choice in choice_rank:
            action = _choice_to_action(choice)
            if action not in allowed_set:
                continue
            if action in seen_actions:
                continue
            ordered.append(action)
            seen_actions.add(action)

        used_default = False
        if not ordered:
            ordered = list(allowed)
            used_default = True
        elif proposed_in_allowed and ordered[0] != proposed_main_action:
            ordered = [proposed_main_action] + [a for a in ordered if a != proposed_main_action]

        confidence = _clamp01(confidence_raw, -1.0)
        if confidence_raw in (None, ""):
            confidence = -1.0

        debug = {
            "raw_output": raw,
            "parsed": parsed,
            "used_default": used_default,
            "choice_rank": choice_rank,
            "proposed_choice": proposed_choice,
            "proposed_main_action": proposed_main_action.value,
            "allowed_actions": allowed_labels,
            "invalid_action": not proposed_in_allowed,
            "proposed_action_outside_allowed": None if proposed_in_allowed else _choice_to_action(proposed_choice).value,
            "confidence": None if confidence < 0.0 else round(float(confidence), 3),
            "cadence_prior": cadence_prior,
            "cadence_prior_debug": cadence_prior_debug,
            "request_label": request_label,
        }
        return ordered, debug


# ----------------------------
# Change-talk inference (Layer2 helper)
# ----------------------------
_CHANGE_TALK_DIRECTION_RULES: Dict[Phase, Dict[str, Any]] = {
    Phase.GREETING: {
        "priority_slots": ("rapport_cue", "greeting_exchange"),
        "focus_hint": "Prioritize small forward-looking openings that help build rapport.",
        "phase_focus": "Within rapport-building, look for openings that suggest willingness to start talking.",
    },
    Phase.PURPOSE_CONFIRMATION: {
        "priority_slots": ("today_focus_topic", "presenting_problem_raw", "process_need"),
        "focus_hint": "Prioritize reasons, felt need, and intention related to today's goal.",
        "phase_focus": "Pick up reasons and felt need connected to today's purpose.",
    },
    Phase.CURRENT_STATUS_CHECK: {
        "priority_slots": ("problem_scene", "emotion_state", "current_situation", "background_context"),
        "focus_hint": "From the current difficulties, prioritize reasons for change and felt need.",
        "phase_focus": "From concrete problem scenes and current strain, pick up need and reasons for change.",
    },
    Phase.FOCUSING_TARGET_BEHAVIOR: {
        "priority_slots": ("target_behavior", "change_direction", "focus_agreement"),
        "focus_hint": "Prioritize intention and feasibility for the target behavior.",
        "phase_focus": "Pick up intention and feasibility related to the target behavior.",
    },
    Phase.IMPORTANCE_PROMOTION: {
        "priority_slots": ("importance_reasons", "core_values", "importance_scale"),
        "focus_hint": "Prioritize talk about values, importance, and reasons.",
        "phase_focus": "From talk about values and importance, pick up reasons and felt need for change.",
    },
    Phase.CONFIDENCE_PROMOTION: {
        "priority_slots": ("supports_strengths", "past_success_experience", "barrier_coping_strategy", "confidence_scale"),
        "focus_hint": "Prioritize self-efficacy, execution outlook, and the next step.",
        "phase_focus": "From strengths, past successes, and coping with barriers, pick up feasibility and the next step.",
    },
    Phase.NEXT_STEP_DECISION: {
        "priority_slots": ("next_step_action", "execution_context", "commitment_level"),
        "focus_hint": "Prioritize concrete definition of the next step and intention to carry it out.",
        "phase_focus": "Pick up concrete details of the next step and implementation intent.",
    },
    Phase.REVIEW_REFLECTION: {
        "priority_slots": ("session_learning", "key_takeaway", "carry_forward_intent"),
        "focus_hint": "Prioritize putting learning into words and intention to carry it forward.",
        "phase_focus": "From learning and carry-forward intent, pick up willingness to continue.",
    },
    Phase.CLOSING: {
        "priority_slots": ("closing_end_signal",),
        "focus_hint": "Prioritize agreement to close and intention to end.",
        "phase_focus": "Check for goodbye signals or consent to end and consolidate closing agreement.",
    },
}

_CHANGE_TALK_TARGET_ANCHOR_PHASES: Set[Phase] = {
    Phase.FOCUSING_TARGET_BEHAVIOR,
    Phase.IMPORTANCE_PROMOTION,
    Phase.CONFIDENCE_PROMOTION,
    Phase.NEXT_STEP_DECISION,
}
_CHANGE_TALK_DIRECTION_ONLY_PHASES: Set[Phase] = {
    Phase.IMPORTANCE_PROMOTION,
    Phase.CONFIDENCE_PROMOTION,
    Phase.REVIEW_REFLECTION,
    Phase.CLOSING,
}
_CHANGE_TALK_PROCESS_PREFERENCE_MARKERS: Tuple[str, ...] = (
    "my own pace",
    "without rushing",
    "while seeing how it goes",
    "process preference",
    "how should I proceed",
    "where should I start",
    "where should I begin",
)
_CHANGE_TALK_DIRECTION_MARKERS: Tuple[str, ...] = (
    "sort out",
    "reduce",
    "clarify",
    "use",
    "support",
    "condition",
    "stability",
    "one step",
    "action",
    "priority",
    "important",
    "continue",
    "carry forward",
    "learning",
)
_CHANGE_TALK_UNCERTAINTY_MARKERS: Tuple[str, ...] = (
    "do not know",
    "cannot grasp",
    "cannot see",
    "anxious",
    "uncertain",
    "confidence",
)


def _should_prioritize_target_behavior_for_phase(phase: Optional[Phase]) -> bool:
    return phase in _CHANGE_TALK_TARGET_ANCHOR_PHASES


def _get_confirmed_target_behavior_for_change_talk(
    *,
    state: DialogueState,
    quality_threshold: float = 0.8,
) -> str:
    focusing_slots = state.phase_slots.get(Phase.FOCUSING_TARGET_BEHAVIOR.name, {})
    target_behavior = _normalize_slot_text(
        focusing_slots.get("target_behavior") if isinstance(focusing_slots, Mapping) else ""
    )
    if not target_behavior:
        return ""

    focusing_meta = state.phase_slot_meta.get(Phase.FOCUSING_TARGET_BEHAVIOR.name, {})
    target_meta = focusing_meta.get("target_behavior") if isinstance(focusing_meta, Mapping) else None
    if not isinstance(target_meta, Mapping):
        return ""

    status = _normalize_slot_text(target_meta.get("status")).lower()
    quality = _clamp01(target_meta.get("quality_score"), -1.0)
    if status != "confirmed" or quality < quality_threshold:
        return ""
    return target_behavior


def _collect_confirmed_slots_for_change_talk(
    *,
    state: DialogueState,
    quality_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    phase_slot_memory = _copy_phase_slot_memory(state.phase_slots)
    phase_slot_meta = _copy_phase_slot_meta(state.phase_slot_meta)
    confirmed_slots: List[Dict[str, Any]] = []
    for phase in Phase:
        phase_slots = phase_slot_memory.get(phase.name, {})
        phase_meta = phase_slot_meta.get(phase.name, {})
        for slot_key in _PHASE_SLOT_SCHEMA.get(phase, ()):  # pragma: no branch
            value = _normalize_slot_text(phase_slots.get(slot_key) if isinstance(phase_slots, Mapping) else "")
            if not value:
                continue
            slot_meta = phase_meta.get(slot_key) if isinstance(phase_meta, Mapping) else None
            status = _normalize_slot_text((slot_meta or {}).get("status")).lower()
            quality = _clamp01((slot_meta or {}).get("quality_score"), 0.0)
            if status != "confirmed" or quality < quality_threshold:
                continue
            confirmed_slots.append(
                {
                    "phase": phase.value,
                    "phase_code": phase.name,
                    "slot_key": slot_key,
                    "slot_label": _PHASE_SLOT_LABELS.get(slot_key, slot_key),
                    "value": value,
                    "quality_score": round(float(quality), 3),
                }
            )
    return confirmed_slots


def _build_review_reflection_bridge_slots(
    *,
    phase_slot_memory: Mapping[str, Mapping[str, str]],
    raw_target_behavior: str,
) -> List[Dict[str, str]]:
    purpose_slots = phase_slot_memory.get(Phase.PURPOSE_CONFIRMATION.name, {})
    current_status_slots = phase_slot_memory.get(Phase.CURRENT_STATUS_CHECK.name, {})
    importance_slots = phase_slot_memory.get(Phase.IMPORTANCE_PROMOTION.name, {})
    confidence_slots = phase_slot_memory.get(Phase.CONFIDENCE_PROMOTION.name, {})
    anchor_specs = (
        (
            Phase.PURPOSE_CONFIRMATION,
            "presenting_problem_raw",
            "Presenting concern",
            purpose_slots.get("presenting_problem_raw") if isinstance(purpose_slots, Mapping) else "",
        ),
        (
            Phase.CURRENT_STATUS_CHECK,
            "problem_scene",
            "Problem scene",
            current_status_slots.get("problem_scene") if isinstance(current_status_slots, Mapping) else "",
        ),
        (
            Phase.IMPORTANCE_PROMOTION,
            "core_values",
            "Values / what matters",
            importance_slots.get("core_values") if isinstance(importance_slots, Mapping) else "",
        ),
        (
            Phase.CONFIDENCE_PROMOTION,
            "supports_strengths",
            "Resources / strengths",
            confidence_slots.get("supports_strengths") if isinstance(confidence_slots, Mapping) else "",
        ),
        (
            Phase.FOCUSING_TARGET_BEHAVIOR,
            "target_behavior",
            "Target behavior",
            "" if isinstance(confidence_slots, Mapping) and confidence_slots.get("supports_strengths") else raw_target_behavior,
        ),
    )

    bridge_slots: List[Dict[str, str]] = []
    seen_fields = set()
    for phase, slot_key, slot_label, raw_value in anchor_specs:
        value = _normalize_slot_text(raw_value)
        if not value:
            continue
        field_text = f"{slot_label}={value}"
        if field_text in seen_fields:
            continue
        seen_fields.add(field_text)
        bridge_slots.append(
            {
                "phase": phase.value,
                "phase_code": phase.name,
                "slot_key": slot_key,
                "slot_label": slot_label,
                "value": value,
            }
        )
    return bridge_slots


def _build_review_reflection_bridge_fields(
    *,
    phase_slot_memory: Mapping[str, Mapping[str, str]],
    raw_target_behavior: str,
) -> List[str]:
    bridge_slots = _build_review_reflection_bridge_slots(
        phase_slot_memory=phase_slot_memory,
        raw_target_behavior=raw_target_behavior,
    )
    bridge_fields: List[str] = []
    for item in bridge_slots:
        slot_label = _normalize_slot_text(item.get("slot_label"))
        value = _normalize_slot_text(item.get("value"))
        if not slot_label or not value:
            continue
        field_text = f"{slot_label}={value}"
        bridge_fields.append(field_text)
    return bridge_fields


def _build_macro_bridge_anchor(
    *,
    state: DialogueState,
    primary_focus_topic: str,
) -> str:
    if state.phase not in {
        Phase.IMPORTANCE_PROMOTION,
        Phase.CONFIDENCE_PROMOTION,
        Phase.REVIEW_REFLECTION,
        Phase.CLOSING,
    }:
        return ""

    micro_focus = _sanitize_human_text(primary_focus_topic)
    if micro_focus in {"", "recent difficulty", "current theme"}:
        return ""

    purpose_slots = state.phase_slots.get(Phase.PURPOSE_CONFIRMATION.name, {})
    macro_candidates: List[str] = []
    if isinstance(purpose_slots, Mapping):
        for raw_value in (
            purpose_slots.get("today_focus_topic"),
            purpose_slots.get("presenting_problem_raw"),
        ):
            value = _sanitize_human_text(raw_value)
            if not value or value in macro_candidates:
                continue
            macro_candidates.append(value)

    normalized_micro = _normalize_slot_text(micro_focus)
    for macro_theme in macro_candidates:
        if _normalize_slot_text(macro_theme) == normalized_micro:
            continue
        return f"'{micro_focus}' is also connected to '{macro_theme}'"
    return ""


def _summarize_phase_slots_for_change_talk(
    *,
    state: DialogueState,
    quality_threshold: float = 0.8,
) -> Dict[str, Any]:
    phase_slot_memory = _copy_phase_slot_memory(state.phase_slots)
    use_target_behavior_only_policy = state.phase in _CHANGE_TALK_TARGET_ANCHOR_PHASES
    use_review_closing_bridge_policy = state.phase in {Phase.REVIEW_REFLECTION, Phase.CLOSING}
    use_today_focus_priority_policy = (
        state.phase in {Phase.GREETING, Phase.PURPOSE_CONFIRMATION, Phase.CURRENT_STATUS_CHECK}
        and not use_target_behavior_only_policy
        and not use_review_closing_bridge_policy
    )
    confirmed_slots = _collect_confirmed_slots_for_change_talk(
        state=state,
        quality_threshold=quality_threshold,
    )

    confirmed_slot_items: List[str] = []
    confirmed_slot_keys: List[str] = []
    for item in confirmed_slots:
        phase_code = _normalize_slot_text(item.get("phase_code"))
        slot_key = _normalize_slot_text(item.get("slot_key"))
        slot_label = _normalize_slot_text(item.get("slot_label")) or slot_key
        value = _normalize_slot_text(item.get("value"))
        if not phase_code or not slot_key or not value:
            continue
        confirmed_slot_items.append(f"{phase_code}.{slot_label}={value}")
        if slot_key not in confirmed_slot_keys:
            confirmed_slot_keys.append(slot_key)

    focusing_slots = phase_slot_memory.get(Phase.FOCUSING_TARGET_BEHAVIOR.name, {})
    raw_target_behavior = _normalize_slot_text(
        focusing_slots.get("target_behavior") if isinstance(focusing_slots, Mapping) else ""
    )

    confirmed_target_behavior = _get_confirmed_target_behavior_for_change_talk(state=state)
    target_behavior_focus = confirmed_target_behavior
    target_behavior_focus_source = "confirmed" if confirmed_target_behavior else ""
    if not target_behavior_focus and use_target_behavior_only_policy and raw_target_behavior:
        target_behavior_focus = raw_target_behavior
        target_behavior_focus_source = "phase_slot_state"
    target_behavior_focus_text = f"target_behavior={target_behavior_focus}" if target_behavior_focus else "none"

    bridge_slots: List[Dict[str, str]] = []
    bridge_fields: List[str] = []
    if use_review_closing_bridge_policy:
        bridge_slots = _build_review_reflection_bridge_slots(
            phase_slot_memory=phase_slot_memory,
            raw_target_behavior=target_behavior_focus or raw_target_behavior,
        )
        bridge_fields = [f"{item.get('slot_label')}={item.get('value')}" for item in bridge_slots]

    prompt_slots: List[Dict[str, Any]] = []
    prompt_slot_keys: List[str] = []
    if use_review_closing_bridge_policy:
        prompt_slot_policy = "bridge_slots_review_closing"
        prompt_slots = [dict(item) for item in bridge_slots]
        for item in prompt_slots:
            slot_key = _normalize_slot_text(item.get("slot_key"))
            if slot_key and slot_key not in prompt_slot_keys:
                prompt_slot_keys.append(slot_key)
    elif use_today_focus_priority_policy:
        prompt_slot_policy = "today_focus_topic_priority_before_focusing"
        purpose_slots = phase_slot_memory.get(Phase.PURPOSE_CONFIRMATION.name, {})
        confirmed_purpose_slots: Dict[str, Dict[str, Any]] = {}
        for item in confirmed_slots:
            if _normalize_slot_text(item.get("phase_code")) != Phase.PURPOSE_CONFIRMATION.name:
                continue
            slot_key = _normalize_slot_text(item.get("slot_key"))
            if not slot_key:
                continue
            confirmed_purpose_slots[slot_key] = dict(item)

        def _append_purpose_slot(slot_key: str, slot_label: str) -> None:
            confirmed_item = confirmed_purpose_slots.get(slot_key)
            if isinstance(confirmed_item, Mapping):
                value = _normalize_slot_text(confirmed_item.get("value"))
                if not value:
                    return
                prompt_slots.append(
                    {
                        "phase": Phase.PURPOSE_CONFIRMATION.value,
                        "phase_code": Phase.PURPOSE_CONFIRMATION.name,
                        "slot_key": slot_key,
                        "slot_label": slot_label,
                        "value": value,
                        "source": "confirmed",
                    }
                )
                if slot_key not in prompt_slot_keys:
                    prompt_slot_keys.append(slot_key)
                return

            value = _normalize_slot_text(
                purpose_slots.get(slot_key) if isinstance(purpose_slots, Mapping) else ""
            )
            if not value:
                return
            prompt_slots.append(
                {
                    "phase": Phase.PURPOSE_CONFIRMATION.value,
                    "phase_code": Phase.PURPOSE_CONFIRMATION.name,
                    "slot_key": slot_key,
                    "slot_label": slot_label,
                    "value": value,
                    "source": "phase_slot_state",
                }
            )
            if slot_key not in prompt_slot_keys:
                prompt_slot_keys.append(slot_key)

        _append_purpose_slot("today_focus_topic", "today's focus")
        _append_purpose_slot("presenting_problem_raw", "presenting concern")
        if not prompt_slots:
            prompt_slot_policy = "all_confirmed_slots"
            prompt_slots = list(confirmed_slots)
            prompt_slot_keys = list(confirmed_slot_keys)
    elif use_target_behavior_only_policy:
        prompt_slot_policy = "target_behavior_only_after_focusing"
        if target_behavior_focus:
            prompt_slots.append(
                {
                    "phase": Phase.FOCUSING_TARGET_BEHAVIOR.value,
                    "phase_code": Phase.FOCUSING_TARGET_BEHAVIOR.name,
                    "slot_key": "target_behavior",
                    "slot_label": _PHASE_SLOT_LABELS.get("target_behavior", "target_behavior"),
                    "value": target_behavior_focus,
                    "source": target_behavior_focus_source,
                }
            )
            prompt_slot_keys.append("target_behavior")
    else:
        prompt_slot_policy = "all_confirmed_slots"
        prompt_slots = list(confirmed_slots)
        prompt_slot_keys = list(confirmed_slot_keys)

    prompt_slot_items: List[str] = []
    for item in prompt_slots:
        phase_code = _normalize_slot_text(item.get("phase_code"))
        slot_label = _normalize_slot_text(item.get("slot_label")) or _normalize_slot_text(item.get("slot_key"))
        value = _normalize_slot_text(item.get("value"))
        if not phase_code or not slot_label or not value:
            continue
        prompt_slot_items.append(f"{phase_code}.{slot_label}={value}")

    return {
        "confirmed_slots": confirmed_slots,
        "confirmed_slots_text": " / ".join(confirmed_slot_items) if confirmed_slot_items else "none",
        "confirmed_slot_keys": confirmed_slot_keys,
        "prompt_slots": prompt_slots,
        "prompt_slots_text": " / ".join(prompt_slot_items) if prompt_slot_items else "none",
        "prompt_slot_keys": prompt_slot_keys,
        "prompt_slot_policy": prompt_slot_policy,
        "bridge_slots": bridge_slots,
        "bridge_fields": bridge_fields,
        "bridge_fields_text": " / ".join(bridge_fields) if bridge_fields else "none",
        "target_behavior_focus": target_behavior_focus,
        "target_behavior_focus_text": target_behavior_focus_text,
        "target_behavior_focus_enabled": bool(target_behavior_focus),
        "target_behavior_focus_source": target_behavior_focus_source or "none",
        "target_anchor_policy": prompt_slot_policy,
    }


def _decide_change_talk_direction(
    *,
    state: DialogueState,
    features: PlannerFeatures,
    slot_context: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    rule = _CHANGE_TALK_DIRECTION_RULES.get(
        state.phase,
        {
            "priority_slots": (),
            "focus_hint": "Carefully pick up openings on the change side.",
            "phase_focus": "From the recent talk, carefully pick up openings on the change side.",
        },
    )
    priority_slots = tuple(rule.get("priority_slots") or ())
    current_slots = slot_context.get("current_slots")
    if not isinstance(current_slots, Mapping):
        current_slots = {}

    target_behavior_focus = _normalize_slot_text(slot_context.get("target_behavior_focus"))
    target_behavior_focus_enabled = bool(target_behavior_focus)
    target_behavior_focus_source = _normalize_slot_text(slot_context.get("target_behavior_focus_source")) or "none"
    prioritize_target_behavior_anchor = (
        target_behavior_focus_enabled and _should_prioritize_target_behavior_for_phase(state.phase)
    )

    selected_anchors: List[str] = []
    selected_slot_keys: List[str] = []
    if prioritize_target_behavior_anchor:
        selected_slot_keys.append("target_behavior")
        selected_anchors.append(f"{_PHASE_SLOT_LABELS.get('target_behavior', 'target_behavior')}={target_behavior_focus}")
    for slot_key in priority_slots:
        value = _normalize_slot_text(current_slots.get(slot_key))
        if not value:
            continue
        if slot_key == "target_behavior" and target_behavior_focus_enabled:
            continue
        selected_slot_keys.append(slot_key)
        selected_anchors.append(f"{_PHASE_SLOT_LABELS.get(slot_key, slot_key)}={value}")
        if len(selected_anchors) >= 2:
            break

    source = "phase_slots"
    if not selected_anchors:
        bridge_fields = slot_context.get("bridge_fields")
        if isinstance(bridge_fields, list):
            selected_anchors = [str(item).strip() for item in bridge_fields if str(item).strip()][:2]
        if selected_anchors:
            source = "cross_phase_slots"
    elif prioritize_target_behavior_anchor:
        source = "target_behavior_anchor"

    if not selected_anchors:
        other_phase_text = str(slot_context.get("other_phase_text") or "").strip()
        if other_phase_text and other_phase_text != "none":
            selected_anchors = [part.strip() for part in other_phase_text.split(" / ") if part.strip()][:2]
            source = "other_phase_slots"

    if not selected_anchors:
        selected_anchors = ["the latest utterance and its context"]
        source = "user_text_only"

    if features.resistance >= 0.6 or features.discord >= 0.5:
        tone_hint = "Respect resistance or discord, avoid negating sustain talk, and pick up the change side in a small, careful way."
    elif features.change_talk >= 0.65:
        tone_hint = "Make explicit change language more concrete and prioritize wording that connects to action."
    else:
        tone_hint = "Gently supplement implied intention or reasons and avoid overstatement."
    if prioritize_target_behavior_anchor:
        if target_behavior_focus_source == "confirmed":
            tone_hint += " Prioritize candidates tied to the confirmed target behavior."
        else:
            tone_hint += " Prioritize candidates tied to the target-behavior anchor."
    elif state.phase in {Phase.REVIEW_REFLECTION, Phase.CLOSING}:
        tone_hint += " Reference the presenting concern, problem scene, values, and target behavior with equal weight rather than over-focusing on one."

    preferred_focus = _preferred_change_talk_motivation_focus(
        importance_estimate=features.importance_estimate,
    )
    if preferred_focus == "importance_related":
        scoring_hint = (
            f"Because the estimated importance is below {_CHANGE_TALK_IMPORTANCE_FOCUS_THRESHOLD:.1f}, "
            "score importance-related content (values, reasons, need) more strongly."
        )
    elif preferred_focus == "confidence_related":
        scoring_hint = (
            f"Because the estimated importance is at least {_CHANGE_TALK_IMPORTANCE_FOCUS_THRESHOLD:.1f}, "
            "score confidence-related content (feasibility, execution outlook) more strongly."
        )
    else:
        scoring_hint = "Because the importance estimate is unknown, treat importance and confidence as equally weighted."

    phase_focus = str(rule.get("phase_focus") or "")
    focus_hint = str(rule.get("focus_hint") or "Carefully pick up openings on the change side.")
    anchor_text = " / ".join(selected_anchors)
    direction = (
        f"{phase_focus} {focus_hint} "
        f"Focus slots: {anchor_text}. {tone_hint} {scoring_hint}"
    )
    debug = {
        "phase": state.phase.value,
        "priority_slots": list(priority_slots),
        "selected_slot_keys": selected_slot_keys,
        "selected_anchors": selected_anchors,
        "source": source,
        "target_behavior_focus": target_behavior_focus,
        "target_behavior_focus_enabled": target_behavior_focus_enabled,
        "target_behavior_focus_source": target_behavior_focus_source,
        "focus_hint": focus_hint,
        "preferred_motivation_focus": preferred_focus or "none",
        "tone_hint": tone_hint,
        "scoring_hint": scoring_hint,
    }
    return direction, debug


def _parse_change_talk_items(
    text: str,
    *,
    max_items: int = 6,
    for_focus_terms: bool = False,
) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    clauses: List[str] = []
    for line in raw.splitlines():
        cleaned = line.strip().strip("「」\"'`")
        if not cleaned:
            continue
        cleaned = re.sub(
            r"^(?:change[_\-\s]?talk)\s*[:：]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:[-*•・]+|\d+[.)．、])\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            sentence_parts = [part.strip() for part in re.split(r"[。．!?？！]+", cleaned) if part.strip()]
            if sentence_parts:
                clauses.extend(sentence_parts)

    if not clauses:
        compact = re.sub(r"\s+", " ", raw).strip().strip("「」\"'`")
        compact = re.sub(
            r"^(?:change[_\-\s]?talk)\s*[:：]\s*",
            "",
            compact,
            flags=re.IGNORECASE,
        )
        compact = re.sub(
            r"(?:change[_\-\s]?talk(?:\s+is)?[、,\s]*){2,}",
            "change talk is ",
            compact,
            flags=re.IGNORECASE,
        )
        clauses = [part.strip().strip("「」\"'`") for part in re.split(r"[。．!?？！\n]+", compact) if part.strip()]

    items: List[str] = []
    for clause in clauses:
        seed = clause.strip().strip("「」\"'`")
        seed = re.sub(r"^change[_\-\s]?talk(?:\s+is)?[、,\s]*", "", seed, flags=re.IGNORECASE)
        seed = re.sub(r"^change[_\-\s]?talk", "", seed, flags=re.IGNORECASE).lstrip(":： ")
        seed = seed.strip().rstrip("。．!?？！")
        if not seed:
            continue

        raw_parts = [part.strip() for part in re.split(r"[、,，/・]", seed) if part.strip()]
        if for_focus_terms and len(raw_parts) <= 1 and " and " in seed.lower():
            tentative = [part.strip() for part in re.split(r"\band\b", seed, flags=re.IGNORECASE) if part.strip()]
            if len(tentative) >= 2 and all(len(re.sub(r"\s+", "", part)) >= 2 for part in tentative):
                raw_parts = tentative
        if not raw_parts:
            raw_parts = [seed]

        for part in raw_parts:
            token = re.sub(r"\s+", " ", part or "").strip().strip("「」\"'`")
            token = token.rstrip("。．!?？！")
            if for_focus_terms:
                token = re.sub(r"^(?:a little|slightly|still|for now|honestly)\s+", "", token, flags=re.IGNORECASE)
                token = re.sub(
                    r"(?:is showing up|is present|is there|seems present|appears present|interest in|signs of|mainly)$",
                    "",
                    token,
                    flags=re.IGNORECASE,
                )
            token = token.strip("。．!?？！「」\"'` ")
            if len(re.sub(r"\s+", "", token)) < 2:
                continue
            if token in items:
                continue
            items.append(token)
            if len(items) >= max_items:
                return items
    return items


def _normalize_change_talk_kind(value: Any, *, text: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "darn_desire": "desire",
        "darn_ability": "ability",
        "darn_reason": "reason",
        "darn_need": "need",
        "cat_commitment": "commitment",
        "cat_activation": "activation",
        "cat_taking_step": "taking_step",
    }
    raw = alias_map.get(raw, raw)
    if raw in _CHANGE_TALK_KINDS:
        return raw
    return _infer_change_talk_kind(text)


def _infer_change_talk_kind(text: str) -> str:
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return "desire"
    if re.search(r"(willdo|doit|keepgoing|continue|decidedto|planto|intendto)", normalized):
        return "commitment"
    if re.search(r"(tryit|giveitatry|startdoing|workonit)", normalized):
        return "activation"
    if re.search(r"(didit|havedone|keptitup|started|triedit)", normalized):
        return "taking_step"
    if re.search(r"(mightbeableto|candoit|seemsdoable|confidence)", normalized):
        return "ability"
    if re.search(r"(needto|haveto|should|otherwise|ifthiskeepsgoing)", normalized):
        return "need"
    if re.search(r"(because|reason|sothat|thatswhy)", normalized):
        return "reason"
    has_change = bool(re.search(r"(wantto|change|stop|improve|getbackontrack)", normalized))
    has_sustain = bool(any(marker in normalized for marker in _AMBIVALENCE_SUSTAIN_HINTS))
    if has_change and has_sustain:
        return "reason"
    if has_sustain:
        return "need"
    return "desire"


def _infer_change_talk_motivation_focus(*, text: str) -> str:
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return "importance_related"

    confidence_patterns = (
        r"(confidence|cando|doable|keepitup|carryout|tryit|nextstep|concrete|preparation)",
    )
    importance_patterns = (
        r"(important|meaning|reason|value|need|otherwise|trouble|avoid|change|improve|getbackontrack)",
    )
    if any(re.search(pattern, normalized) for pattern in confidence_patterns):
        return "confidence_related"
    if any(re.search(pattern, normalized) for pattern in importance_patterns):
        return "importance_related"
    return "importance_related"


def _normalize_change_talk_motivation_focus(
    value: Any,
    *,
    text: str,
    legacy_category: Any = None,
) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "importance": "importance_related",
        "importance_side": "importance_related",
        "value_related": "importance_related",
        "values_related": "importance_related",
        "need_related": "importance_related",
        "reason_related": "importance_related",
        "confidence": "confidence_related",
        "confidence_side": "confidence_related",
        "self_efficacy": "confidence_related",
        "efficacy_related": "confidence_related",
        "action_related": "confidence_related",
        "execution_related": "confidence_related",
    }
    raw = alias_map.get(raw, raw)
    if raw in _CHANGE_TALK_MOTIVATION_FOCUS_TYPES:
        return raw

    # Accept the old DARN/CAT schema for backward compatibility and map it onto the newer internal axis.
    legacy_raw = str(legacy_category or "").strip().lower().replace("-", "_").replace(" ", "_")
    if legacy_raw in {"cat", "action", "planning", "mobilizing"}:
        return "confidence_related"
    if legacy_raw in {"darn", "preparation", "preparatory", "evoking"}:
        return "importance_related"

    return _infer_change_talk_motivation_focus(text=text)


def _normalize_importance_estimate_to_0_1(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except Exception:
        return None
    if score < 0.0:
        return None
    if score <= 1.0:
        return _clamp01(score, 0.0)
    return _clamp01(score / 10.0, 0.0)


def _preferred_change_talk_motivation_focus(*, importance_estimate: Any) -> str:
    normalized_importance = _normalize_importance_estimate_to_0_1(importance_estimate)
    if normalized_importance is None:
        return ""
    if normalized_importance < _CHANGE_TALK_IMPORTANCE_FOCUS_THRESHOLD:
        return "importance_related"
    return "confidence_related"


def _is_change_talk_text_grounded_in_user_utterance(text: Any, user_text: Any) -> bool:
    normalized_text = re.sub(r"\s+", "", str(text or "")).rstrip("。．!?？！")
    normalized_user = re.sub(r"\s+", "", str(user_text or ""))
    if not normalized_text or not normalized_user:
        return False
    return normalized_text in normalized_user


def _normalize_change_talk_origin(value: Any, *, text: str, user_text: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "user": "user_utterance",
        "user_direct": "user_utterance",
        "direct_user": "user_utterance",
        "quoted_user": "user_utterance",
        "system": "system_reframed",
        "reframed": "system_reframed",
        "inferred": "system_reframed",
        "sustain_reframed": "system_reframed",
        "reframed_from_sustain": "system_reframed",
        "sustain_talk_reframed": "system_reframed",
    }
    raw = alias_map.get(raw, raw)
    if raw in _CHANGE_TALK_ORIGIN_TYPES:
        if raw == "user_utterance" and not _is_change_talk_text_grounded_in_user_utterance(text, user_text):
            return "system_reframed"
        return raw

    if _is_change_talk_text_grounded_in_user_utterance(text, user_text):
        return "user_utterance"
    return "system_reframed"


def _normalize_change_talk_explicitness(value: Any, *, text: str, user_text: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"explicit", "inferred"}:
        if raw == "explicit" and not _is_change_talk_text_grounded_in_user_utterance(text, user_text):
            return "inferred"
        return raw
    if _is_change_talk_text_grounded_in_user_utterance(text, user_text):
        return "explicit"
    return "inferred"


def _normalize_evidence_turn(value: Any, *, default_turn: int, min_turn: int, max_turn: int) -> Optional[int]:
    try:
        turn = int(value)
    except Exception:
        turn = default_turn
    if turn < min_turn or turn > max_turn:
        turn = default_turn
    return turn if turn > 0 else None


def _normalize_linked_slots(value: Any, *, default_slots: Sequence[str]) -> List[str]:
    linked: List[str] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            slot_key = str(item or "").strip()
            if not slot_key or slot_key in linked:
                continue
            linked.append(slot_key)
            if len(linked) >= 4:
                break
    if not linked:
        for slot_key in default_slots:
            sk = str(slot_key or "").strip()
            if not sk or sk in linked:
                continue
            linked.append(sk)
            if len(linked) >= 4:
                break
    return linked


def _normalize_change_talk_candidate(
    raw: Any,
    *,
    idx: int,
    default_evidence_turn: int,
    min_turn: int,
    max_turn: int,
    user_text: str,
    default_linked_slots: Sequence[str],
    default_confidence: float,
) -> Optional[ChangeTalkCandidate]:
    if isinstance(raw, Mapping):
        text = _normalize_slot_text(
            raw.get("normalized_text")
            or raw.get("text")
            or raw.get("content")
            or raw.get("value")
            or raw.get("summary")
        )
        evidence_quote = _normalize_slot_text(raw.get("evidence_quote") or raw.get("quote") or "")
        confidence = _clamp01(raw.get("confidence"), default_confidence)
        slot_relevance = _clamp01(
            raw.get("slot_relevance", raw.get("slot_relatedness", raw.get("relevance", 0.0))),
            0.0,
        )
        all_phase_slot_relevance = _clamp01(
            raw.get(
                "all_phase_slot_relevance",
                raw.get(
                    "global_slot_relevance",
                    raw.get(
                        "overall_slot_relevance",
                        raw.get("cross_phase_slot_relevance", slot_relevance),
                    ),
                ),
            ),
            slot_relevance,
        )
        target_behavior_relevance = _clamp01(
            raw.get(
                "target_behavior_relevance",
                raw.get(
                    "target_behavior_relatedness",
                    raw.get("target_behavior_alignment", slot_relevance),
                ),
            ),
            slot_relevance,
        )
        candidate_id = _normalize_slot_text(raw.get("id")) or f"ct_{idx}"
        kind = _normalize_change_talk_kind(raw.get("kind"), text=text)
        legacy_category = raw.get("ct_category") or raw.get("darn_cat") or raw.get("category")
        motivation_focus = _normalize_change_talk_motivation_focus(
            raw.get("motivation_focus")
            or raw.get("importance_confidence_focus")
            or raw.get("focus_domain"),
            text=text,
            legacy_category=legacy_category,
        )
        origin_type = _normalize_change_talk_origin(
            raw.get("origin_type") or raw.get("origin") or raw.get("source_type") or raw.get("source_flag"),
            text=text,
            user_text=user_text,
        )
        explicitness = _normalize_change_talk_explicitness(
            raw.get("explicitness"),
            text=text,
            user_text=user_text,
        )
        evidence_turn = _normalize_evidence_turn(
            raw.get("evidence_turn"),
            default_turn=default_evidence_turn,
            min_turn=min_turn,
            max_turn=max_turn,
        )
        linked_slots = _normalize_linked_slots(
            raw.get("linked_slots"),
            default_slots=default_linked_slots,
        )
    else:
        text = _normalize_slot_text(raw)
        evidence_quote = text
        confidence = _clamp01(default_confidence, default_confidence)
        slot_relevance = 0.0
        all_phase_slot_relevance = 0.0
        target_behavior_relevance = 0.0
        candidate_id = f"ct_{idx}"
        kind = _infer_change_talk_kind(text)
        motivation_focus = _infer_change_talk_motivation_focus(text=text)
        origin_type = _normalize_change_talk_origin(None, text=text, user_text=user_text)
        explicitness = _normalize_change_talk_explicitness(None, text=text, user_text=user_text)
        evidence_turn = _normalize_evidence_turn(
            None,
            default_turn=default_evidence_turn,
            min_turn=min_turn,
            max_turn=max_turn,
        )
        linked_slots = _normalize_linked_slots([], default_slots=default_linked_slots)

    if not text:
        return None
    if not evidence_quote:
        evidence_quote = text

    return ChangeTalkCandidate(
        id=candidate_id,
        kind=kind,
        normalized_text=text,
        evidence_quote=evidence_quote,
        evidence_turn=evidence_turn,
        explicitness=explicitness,
        confidence=confidence,
        slot_relevance=slot_relevance,
        all_phase_slot_relevance=all_phase_slot_relevance,
        target_behavior_relevance=target_behavior_relevance,
        origin_type=origin_type,
        motivation_focus=motivation_focus,
        linked_slots=linked_slots,
    )


def _build_change_talk_candidates_from_legacy_text(
    text: str,
    *,
    default_evidence_turn: int,
    min_turn: int,
    max_turn: int,
    user_text: str,
    default_linked_slots: Sequence[str],
    max_candidates: int = 4,
) -> List[ChangeTalkCandidate]:
    items = _parse_change_talk_items(text, max_items=max_candidates, for_focus_terms=False)
    candidates: List[ChangeTalkCandidate] = []
    seen_text = set()
    for idx, item in enumerate(items, start=1):
        candidate = _normalize_change_talk_candidate(
            item,
            idx=idx,
            default_evidence_turn=default_evidence_turn,
            min_turn=min_turn,
            max_turn=max_turn,
            user_text=user_text,
            default_linked_slots=default_linked_slots,
            default_confidence=0.55,
        )
        if candidate is None:
            continue
        normalized = re.sub(r"\s+", "", candidate.normalized_text)
        if not normalized or normalized in seen_text:
            continue
        seen_text.add(normalized)
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _normalize_change_talk_infer_output(
    payload: Any,
    *,
    user_text: str,
    history_start_turn: int,
    history_end_turn: int,
    linked_slots_hint: Sequence[str],
    default_text: str = "",
) -> Tuple[List[ChangeTalkCandidate], str]:
    raw = payload
    legacy_text = ""
    candidates: List[ChangeTalkCandidate] = []
    default_turn = history_end_turn if history_end_turn > 0 else history_start_turn

    if isinstance(raw, Mapping):
        candidate_block = (
            raw.get("focus_candidates")
            or raw.get("candidates")
            or raw.get("change_talk_candidates")
            or []
        )
        if isinstance(candidate_block, Sequence) and not isinstance(candidate_block, (str, bytes)):
            for idx, item in enumerate(candidate_block, start=1):
                candidate = _normalize_change_talk_candidate(
                    item,
                    idx=idx,
                    default_evidence_turn=default_turn,
                    min_turn=history_start_turn,
                    max_turn=history_end_turn,
                    user_text=user_text,
                    default_linked_slots=linked_slots_hint,
                    default_confidence=0.55,
                )
                if candidate is None:
                    continue
                if any(existing.id == candidate.id for existing in candidates):
                    candidate.id = f"ct_{idx}"
                candidates.append(candidate)
                if len(candidates) >= 4:
                    break

        raw_legacy = (
            raw.get("change_talk_inference")
            or raw.get("inference")
            or raw.get("summary")
            or raw.get("text")
            or ""
        )
        legacy_text = _normalize_slot_text(raw_legacy)
    elif isinstance(raw, str):
        legacy_text = _normalize_slot_text(raw)
    else:
        legacy_text = _normalize_slot_text(default_text)

    if not candidates and legacy_text:
        candidates = _build_change_talk_candidates_from_legacy_text(
            legacy_text,
            default_evidence_turn=default_turn,
            min_turn=history_start_turn,
            max_turn=history_end_turn,
            user_text=user_text,
            default_linked_slots=linked_slots_hint,
        )
    if not legacy_text and default_text:
        legacy_text = _normalize_slot_text(default_text)
    return candidates[:4], legacy_text


def _collect_linked_slot_priority_from_focus_candidates(
    focus_candidates: Optional[Sequence[ChangeTalkCandidate]],
    *,
    max_candidates: int = 2,
    max_slots: int = 4,
) -> List[str]:
    if not isinstance(focus_candidates, Sequence):
        return []
    linked_slot_keys: List[str] = []
    for candidate in list(focus_candidates)[: max(1, int(max_candidates))]:
        if not isinstance(candidate, ChangeTalkCandidate):
            continue
        for linked_slot in candidate.linked_slots:
            slot_key = _slot_key_from_focus_text(linked_slot)
            if not slot_key or slot_key in linked_slot_keys:
                continue
            linked_slot_keys.append(slot_key)
            if len(linked_slot_keys) >= max(1, int(max_slots)):
                return linked_slot_keys
    return linked_slot_keys


def _derive_slot_target_from_phase_debug(
    phase_debug: Optional[Mapping[str, Any]],
    *,
    focus_candidates: Optional[Sequence[ChangeTalkCandidate]] = None,
) -> str:
    if not isinstance(phase_debug, Mapping):
        return ""

    phase_transitioned = _did_phase_transition_for_current_turn(phase_debug)
    slot_review = None if phase_transitioned else phase_debug.get("slot_review")
    slot_gate = (
        phase_debug.get("post_transition_slot_gate")
        if phase_transitioned and isinstance(phase_debug.get("post_transition_slot_gate"), Mapping)
        else phase_debug.get("slot_gate")
    )
    linked_slot_priority = _collect_linked_slot_priority_from_focus_candidates(focus_candidates)
    linked_slot_priority_set = set(linked_slot_priority)

    if linked_slot_priority:
        missing_slot_keys: List[str] = []
        if isinstance(slot_gate, Mapping):
            raw_missing_slot_keys = slot_gate.get("missing_slot_keys")
            if isinstance(raw_missing_slot_keys, Sequence) and not isinstance(raw_missing_slot_keys, (str, bytes)):
                for item in raw_missing_slot_keys:
                    slot_key = _slot_key_from_focus_text(item)
                    if not slot_key or slot_key in missing_slot_keys:
                        continue
                    missing_slot_keys.append(slot_key)
        for slot_key in linked_slot_priority:
            if slot_key in missing_slot_keys:
                return slot_key

        quality_min_threshold = 0.8
        if isinstance(slot_gate, Mapping):
            gate_threshold = _clamp01(slot_gate.get("quality_min_threshold"), -1.0)
            if gate_threshold >= 0.0:
                quality_min_threshold = gate_threshold
        quality_by_slot: Dict[str, float] = {}
        quality_sources = []
        if isinstance(slot_review, Mapping):
            quality_sources.append(slot_review.get("slot_quality"))
        if isinstance(slot_gate, Mapping):
            quality_sources.append(slot_gate.get("slot_quality"))
        for source in quality_sources:
            if not isinstance(source, Mapping):
                continue
            for key, score in source.items():
                slot_key = _slot_key_from_focus_text(key)
                quality = _clamp01(score, -1.0)
                if not slot_key or quality < 0.0:
                    continue
                if slot_key in quality_by_slot:
                    quality_by_slot[slot_key] = min(quality_by_slot[slot_key], quality)
                else:
                    quality_by_slot[slot_key] = quality
        preferred_low_quality: List[Tuple[str, float]] = []
        for slot_key in linked_slot_priority:
            if slot_key not in quality_by_slot:
                continue
            quality = quality_by_slot[slot_key]
            if quality < quality_min_threshold:
                preferred_low_quality.append((slot_key, quality))
        if preferred_low_quality:
            preferred_low_quality.sort(key=lambda item: item[1])
            return preferred_low_quality[0][0]

        phase_level_hints = phase_debug.get("slot_quality_target_examples")
        if phase_level_hints is None:
            phase_level_hints = phase_debug.get("slot_repair_hints")
        if isinstance(phase_level_hints, Sequence) and not isinstance(phase_level_hints, (str, bytes)):
            for item in phase_level_hints:
                if not isinstance(item, Mapping):
                    continue
                slot_key = _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
                if slot_key and slot_key in linked_slot_priority_set:
                    return slot_key

        if isinstance(slot_review, Mapping):
            review_hints = slot_review.get("slot_quality_target_examples")
            if review_hints is None:
                review_hints = slot_review.get("slot_repair_hints")
            if isinstance(review_hints, Sequence) and not isinstance(review_hints, (str, bytes)):
                for item in review_hints:
                    if not isinstance(item, Mapping):
                        continue
                    slot_key = _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
                    if slot_key and slot_key in linked_slot_priority_set:
                        return slot_key

            reviewed_updates = slot_review.get("reviewed_updates")
            if isinstance(reviewed_updates, Sequence) and not isinstance(reviewed_updates, (str, bytes)):
                for item in reviewed_updates:
                    if not isinstance(item, Mapping):
                        continue
                    admissible = item.get("admissible")
                    if isinstance(admissible, bool):
                        if admissible:
                            continue
                    else:
                        decision = _normalize_slot_text(item.get("decision")).lower()
                        if decision not in {"reject", "needs_confirmation"}:
                            continue
                    slot_key = _slot_key_from_focus_text(item.get("slot_key"))
                    if slot_key and slot_key in linked_slot_priority_set:
                        return slot_key

        return linked_slot_priority[0]

    slot_focus_priority = phase_debug.get("slot_focus_priority")
    if isinstance(slot_focus_priority, Sequence) and not isinstance(slot_focus_priority, (str, bytes)):
        for item in slot_focus_priority:
            slot_key = _slot_key_from_focus_text(item)
            if slot_key:
                return slot_key

    phase_level_hints = phase_debug.get("slot_quality_target_examples")
    if phase_level_hints is None:
        phase_level_hints = phase_debug.get("slot_repair_hints")
    if isinstance(phase_level_hints, Sequence) and not isinstance(phase_level_hints, (str, bytes)):
        for item in phase_level_hints:
            if not isinstance(item, Mapping):
                continue
            slot_key = _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
            if slot_key:
                return slot_key

    if isinstance(slot_review, Mapping):
        review_hints = slot_review.get("slot_quality_target_examples")
        if review_hints is None:
            review_hints = slot_review.get("slot_repair_hints")
        if isinstance(review_hints, Sequence) and not isinstance(review_hints, (str, bytes)):
            for item in review_hints:
                if not isinstance(item, Mapping):
                    continue
                slot_key = _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
                if slot_key:
                    return slot_key

        reviewed_updates = slot_review.get("reviewed_updates")
        if isinstance(reviewed_updates, Sequence) and not isinstance(reviewed_updates, (str, bytes)):
            for item in reviewed_updates:
                if not isinstance(item, Mapping):
                    continue
                admissible = item.get("admissible")
                if isinstance(admissible, bool):
                    if admissible:
                        continue
                else:
                    decision = _normalize_slot_text(item.get("decision")).lower()
                    if decision not in {"reject", "needs_confirmation"}:
                        continue
                slot_key = _slot_key_from_focus_text(item.get("slot_key"))
                if slot_key:
                    return slot_key

        slot_quality = slot_review.get("slot_quality")
        if isinstance(slot_quality, Mapping):
            scored: List[Tuple[str, float]] = []
            for key, score in slot_quality.items():
                slot_key = _slot_key_from_focus_text(key)
                quality = _clamp01(score, -1.0)
                if not slot_key or quality < 0.0:
                    continue
                scored.append((slot_key, quality))
            if scored:
                scored.sort(key=lambda item: item[1])
                return scored[0][0]

    if not isinstance(slot_gate, Mapping):
        return ""
    keys = slot_gate.get("missing_slot_keys")
    if isinstance(keys, Sequence) and not isinstance(keys, (str, bytes)):
        for key in keys:
            slot_key = str(key or "").strip()
            if not slot_key:
                continue
            return slot_key
    texts = slot_gate.get("missing_slot_texts")
    if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
        for text in texts:
            normalized = _normalize_slot_text(text)
            if normalized:
                return normalized
    return ""


def _detect_layer1_target_slot_miss(phase_debug: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "detected": False,
        "reason": "",
        "target_slot_keys": [],
        "returned_slot_keys": [],
        "matched_slot_keys": [],
    }
    if not isinstance(phase_debug, Mapping):
        result["reason"] = "phase_debug_missing"
        return result

    layer1_slot_fill = phase_debug.get("layer1_slot_fill")
    if not isinstance(layer1_slot_fill, Mapping):
        result["reason"] = "layer1_slot_fill_missing"
        return result

    slot_fill_debug = layer1_slot_fill.get("slot_fill_debug")
    if not isinstance(slot_fill_debug, Mapping):
        result["reason"] = "slot_fill_debug_missing"
        return result

    debug_sources: List[Mapping[str, Any]] = [slot_fill_debug]
    scope_current_debug = slot_fill_debug.get("scope_current_debug")
    if isinstance(scope_current_debug, Mapping):
        debug_sources.append(scope_current_debug)
    scope_non_current_debug = slot_fill_debug.get("scope_non_current_debug")
    if isinstance(scope_non_current_debug, Mapping):
        debug_sources.append(scope_non_current_debug)

    def _first_debug_value(key: str) -> Any:
        for source in debug_sources:
            if key in source:
                return source.get(key)
        return None

    raw_target_slot_keys = _first_debug_value("layer1_prompt_target_slot_keys")
    if not isinstance(raw_target_slot_keys, Sequence) or isinstance(raw_target_slot_keys, (str, bytes)):
        result["reason"] = "target_slot_keys_missing"
        return result

    target_slot_keys: List[str] = []
    for item in raw_target_slot_keys:
        slot_key = _slot_key_from_focus_text(item)
        if not slot_key or slot_key in target_slot_keys:
            continue
        target_slot_keys.append(slot_key)
    result["target_slot_keys"] = target_slot_keys
    if not target_slot_keys:
        result["reason"] = "target_slot_keys_empty"
        return result

    raw_target_phase_updates = _first_debug_value("target_phase_updates")
    if not isinstance(raw_target_phase_updates, Sequence) or isinstance(raw_target_phase_updates, (str, bytes)):
        raw_target_phase_updates = _first_debug_value("scope_current_updates")
    returned_slot_keys: List[str] = []
    if isinstance(raw_target_phase_updates, Sequence) and not isinstance(raw_target_phase_updates, (str, bytes)):
        for update in raw_target_phase_updates:
            if not isinstance(update, Mapping):
                continue
            slots = update.get("slots")
            if not isinstance(slots, Mapping):
                continue
            for slot_key_raw in slots.keys():
                slot_key = _slot_key_from_focus_text(slot_key_raw)
                if not slot_key or slot_key in returned_slot_keys:
                    continue
                returned_slot_keys.append(slot_key)
    result["returned_slot_keys"] = returned_slot_keys

    target_slot_key_set = set(target_slot_keys)
    matched_slot_keys = [slot_key for slot_key in returned_slot_keys if slot_key in target_slot_key_set]
    result["matched_slot_keys"] = matched_slot_keys
    if matched_slot_keys:
        result["reason"] = "target_slot_returned"
        return result

    result["detected"] = True
    result["reason"] = "no_target_slot_returned" if returned_slot_keys else "no_target_phase_updates"
    return result


def _score_change_talk_candidate_for_turn(
    candidate: ChangeTalkCandidate,
    *,
    target_behavior_focus: str = "",
    prioritize_target_behavior: bool = True,
    importance_estimate: Optional[float] = None,
) -> float:
    score = 0.32 * _clamp01(candidate.confidence, 0.0)
    score += 0.18 * _clamp01(candidate.slot_relevance, 0.0)
    score += 0.30 * _clamp01(candidate.all_phase_slot_relevance, 0.0)
    target_behavior_relevance = _clamp01(candidate.target_behavior_relevance, 0.0)
    if _normalize_slot_text(target_behavior_focus) and prioritize_target_behavior:
        score += 0.42 * target_behavior_relevance
    else:
        score += 0.08 * target_behavior_relevance

    origin_raw = str(candidate.origin_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    origin_alias = {
        "user": "user_utterance",
        "user_direct": "user_utterance",
        "direct_user": "user_utterance",
        "quoted_user": "user_utterance",
        "system": "system_reframed",
        "reframed": "system_reframed",
        "inferred": "system_reframed",
        "sustain_reframed": "system_reframed",
        "reframed_from_sustain": "system_reframed",
        "sustain_talk_reframed": "system_reframed",
    }
    origin = origin_alias.get(origin_raw, origin_raw)
    if origin not in _CHANGE_TALK_ORIGIN_TYPES:
        origin = "user_utterance" if candidate.explicitness == "explicit" else "system_reframed"
    if origin == "user_utterance":
        score += 0.22
    else:
        score += 0.04

    candidate_focus = _normalize_change_talk_motivation_focus(
        candidate.motivation_focus,
        text=candidate.normalized_text,
    )
    preferred_focus = _preferred_change_talk_motivation_focus(importance_estimate=importance_estimate)
    if preferred_focus:
        if candidate_focus == preferred_focus:
            score += 0.16
        else:
            score += 0.06

    if candidate.explicitness == "explicit":
        score += 0.06

    return max(0.0, min(2.0, score))


def _candidate_linked_slot_keys(candidate: ChangeTalkCandidate) -> set[str]:
    keys: set[str] = set()
    for item in candidate.linked_slots:
        slot_key = _slot_key_from_focus_text(item)
        if slot_key:
            keys.add(slot_key)
    return keys


def _is_change_talk_process_preference_text(text: str) -> bool:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return False
    if not any(marker in normalized for marker in _CHANGE_TALK_PROCESS_PREFERENCE_MARKERS):
        return False
    return not any(marker in normalized for marker in _CHANGE_TALK_DIRECTION_MARKERS)


def _is_change_talk_uncertainty_only_text(text: str) -> bool:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return False
    if not any(marker in normalized for marker in _CHANGE_TALK_UNCERTAINTY_MARKERS):
        return False
    return not any(marker in normalized for marker in _CHANGE_TALK_DIRECTION_MARKERS)


def _phase_adjust_change_talk_candidate_score(
    candidate: ChangeTalkCandidate,
    *,
    current_phase: Optional[Phase],
) -> float:
    if current_phase is None:
        return 0.0

    slot_keys = _candidate_linked_slot_keys(candidate)
    explicit_user_candidate = (
        str(candidate.origin_type or "").strip().lower() == "user_utterance"
        and str(candidate.explicitness or "").strip().lower() == "explicit"
    )
    uses_target_or_next_step = bool(
        slot_keys & {"target_behavior", "next_step_action", "execution_context", "commitment_level"}
    )
    uses_direction_or_support = bool(
        slot_keys
        & {
            "change_direction",
            "importance_reasons",
            "core_values",
            "supports_strengths",
            "past_success_experience",
            "barrier_coping_strategy",
            "session_learning",
            "key_takeaway",
            "carry_forward_intent",
        }
    )
    normalized_text = _normalize_slot_text(candidate.normalized_text)
    adjustment = 0.0

    if _is_change_talk_uncertainty_only_text(normalized_text):
        adjustment -= 0.45
    if _is_change_talk_process_preference_text(normalized_text):
        adjustment -= 0.35 if current_phase in _CHANGE_TALK_DIRECTION_ONLY_PHASES else 0.18

    if current_phase in {Phase.PURPOSE_CONFIRMATION, Phase.CURRENT_STATUS_CHECK}:
        if uses_target_or_next_step and not explicit_user_candidate:
            adjustment -= 0.28
    elif current_phase in {Phase.IMPORTANCE_PROMOTION, Phase.CONFIDENCE_PROMOTION}:
        if uses_target_or_next_step and not explicit_user_candidate:
            adjustment -= 0.48
        if uses_direction_or_support:
            adjustment += 0.2
    elif current_phase in {Phase.REVIEW_REFLECTION, Phase.CLOSING}:
        if uses_target_or_next_step and not explicit_user_candidate:
            adjustment -= 0.52
        if uses_direction_or_support or explicit_user_candidate:
            adjustment += 0.18
    return adjustment


def _should_surface_change_talk_candidate(
    candidate: ChangeTalkCandidate,
    *,
    current_phase: Optional[Phase],
) -> bool:
    normalized_text = _normalize_slot_text(candidate.normalized_text)
    if not normalized_text:
        return False
    if _is_change_talk_process_preference_text(normalized_text):
        if current_phase in _CHANGE_TALK_DIRECTION_ONLY_PHASES:
            return False
        if (
            str(candidate.origin_type or "").strip().lower() != "user_utterance"
            or str(candidate.explicitness or "").strip().lower() != "explicit"
        ):
            return False
    return True


def _select_change_talk_candidates_for_action(
    *,
    candidates: Sequence[ChangeTalkCandidate],
    max_items: int = 2,
    current_phase: Optional[Phase] = None,
    target_behavior_focus: str = "",
    prioritize_target_behavior: bool = True,
    importance_estimate: Optional[float] = None,
) -> List[ChangeTalkCandidate]:
    if not candidates:
        return []
    item_limit = max(1, int(max_items))
    effective_target_behavior_focus = (
        target_behavior_focus if prioritize_target_behavior else ""
    )
    target_behavior_focus_enabled = bool(_normalize_slot_text(effective_target_behavior_focus))

    def _target_behavior_bucket(candidate: ChangeTalkCandidate) -> int:
        if not target_behavior_focus_enabled:
            return 0
        explicit_user_candidate = (
            str(candidate.origin_type or "").strip().lower() == "user_utterance"
            and str(candidate.explicitness or "").strip().lower() == "explicit"
        )
        if (
            current_phase in {Phase.IMPORTANCE_PROMOTION, Phase.CONFIDENCE_PROMOTION}
            and not explicit_user_candidate
            and _candidate_linked_slot_keys(candidate)
            & {"target_behavior", "next_step_action", "execution_context", "commitment_level"}
        ):
            return 0
        relevance = _clamp01(candidate.target_behavior_relevance, 0.0)
        if relevance >= 0.7:
            return 2
        if relevance >= 0.4:
            return 1
        return 0

    def _tie_break_key(candidate: ChangeTalkCandidate) -> Tuple[int, int]:
        linked_slots: List[str] = []
        for item in candidate.linked_slots:
            slot_key = str(item).strip()
            if not slot_key or slot_key in linked_slots:
                continue
            linked_slots.append(slot_key)
        if not linked_slots:
            return (0, 0)
        return (1, min(6, len(linked_slots)))

    scored = sorted(
        candidates,
        key=lambda candidate: (
            _target_behavior_bucket(candidate),
            _score_change_talk_candidate_for_turn(
                candidate,
                target_behavior_focus=effective_target_behavior_focus,
                prioritize_target_behavior=prioritize_target_behavior,
                importance_estimate=importance_estimate,
            )
            + _phase_adjust_change_talk_candidate_score(candidate, current_phase=current_phase),
            _tie_break_key(candidate),
        ),
        reverse=True,
    )
    selected: List[ChangeTalkCandidate] = []
    seen_text: set[str] = set()

    def _normalized_key(candidate: ChangeTalkCandidate) -> str:
        return re.sub(r"\s+", "", candidate.normalized_text)

    for candidate in scored:
        if len(selected) >= item_limit:
            break
        if not _should_surface_change_talk_candidate(candidate, current_phase=current_phase):
            continue
        key = re.sub(r"\s+", "", candidate.normalized_text)
        if not key or key in seen_text:
            continue
        selected.append(candidate)
        seen_text.add(key)
    return selected


def _change_talk_hint_max_items_for_phase(current_phase: Optional[Phase], requested_max_items: int) -> int:
    max_items = max(1, int(requested_max_items))
    if current_phase in _CHANGE_TALK_DIRECTION_ONLY_PHASES:
        return min(1, max_items)
    return max_items


def _build_change_talk_hint_from_candidates(
    *,
    candidates: Sequence[ChangeTalkCandidate],
    current_phase: Optional[Phase] = None,
    max_items: int = 2,
) -> str:
    phrases: List[str] = []
    item_limit = _change_talk_hint_max_items_for_phase(current_phase, max_items)
    for candidate in list(candidates):
        if not _should_surface_change_talk_candidate(candidate, current_phase=current_phase):
            continue
        text = _normalize_slot_text(candidate.normalized_text)
        if not text:
            continue
        normalized = text.rstrip("。．!?？！")
        if not normalized:
            continue
        phrases.append(normalized)
        if len(phrases) >= item_limit:
            break
    if not phrases:
        return ""
    return "Change talk: " + " / ".join(phrases) + "."


@dataclass
class LLMChangeTalkInferer:
    """
    Helper agent that structures and infers explicit or implied change-talk candidates.
    - Keep multiple candidates rather than collapsing to one; Layer2 should focus on extraction.
    - Include change_talk_inference as a string hint for backward compatibility.
    """

    llm: LLMClient
    temperature: float = 0.0
    max_history_turns: int = 6

    def infer(
        self,
        *,
        user_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        slot_context = _summarize_phase_slots_for_change_talk(state=state)
        prompt_slots_raw = slot_context.get("prompt_slots")
        prompt_slots = (
            list(prompt_slots_raw)
            if isinstance(prompt_slots_raw, Sequence) and not isinstance(prompt_slots_raw, (str, bytes))
            else []
        )
        confirmed_slots_raw = slot_context.get("confirmed_slots")
        confirmed_slots = (
            list(confirmed_slots_raw)
            if isinstance(confirmed_slots_raw, Sequence) and not isinstance(confirmed_slots_raw, (str, bytes))
            else []
        )
        raw_prompt_slot_keys = slot_context.get("prompt_slot_keys")
        linked_slots_hint: List[str] = []
        if isinstance(raw_prompt_slot_keys, Sequence) and not isinstance(raw_prompt_slot_keys, (str, bytes)):
            for item in raw_prompt_slot_keys:
                slot_key = _normalize_slot_text(item)
                if not slot_key or slot_key in linked_slots_hint:
                    continue
                linked_slots_hint.append(slot_key)
        prompt_slot_policy = _normalize_slot_text(slot_context.get("prompt_slot_policy")) or "all_confirmed_slots"
        direction_hint = "Extract change-talk candidates from the latest utterance, prioritizing links to the inference slots."
        direction_debug = {
            "source": prompt_slot_policy,
            "selected_slot_keys": list(linked_slots_hint),
            "target_behavior_focus_source": _normalize_slot_text(slot_context.get("target_behavior_focus_source")) or "none",
        }

        system = (
            "You are a change-talk inference agent for motivational interviewing (MI).\n"
            "Goal: extract and structure explicit or implied change-talk candidates from the latest client utterance.\n"
            "Inference target: DARN-CAT (Desire / Ability / Reasons / Need / Commitment / Activation / Taking steps).\n"
            "Output constraint: return exactly one JSON object. Do not include explanations or code blocks.\n"
            "Format:\n"
            "{\n"
            '  "focus_candidates":[{"id":"ct_1","motivation_focus":"importance_related|confidence_related","origin_type":"user_utterance|system_reframed","normalized_text":"...","evidence_quote":"...","evidence_turn":12,"explicitness":"explicit|inferred","confidence":0.0,"slot_relevance":0.0,"all_phase_slot_relevance":0.0,"target_behavior_relevance":0.0,"linked_slots":["..."]}],\n'
            '  "change_talk_inference":"..."\n'
            "}\n"
            "- Return 0-4 items in `focus_candidates`.\n"
            "- For each candidate, return `slot_relevance` as a 0.0-1.0 score for relevance to the slot context.\n"
            "- For each candidate, return `all_phase_slot_relevance` as a 0.0-1.0 score for relevance to all confirmed slots.\n"
            "- For each candidate, return `target_behavior_relevance` as a 0.0-1.0 score for relevance to the target-behavior anchor (prefer confirmed anchors).\n"
            "- `origin_type` is required. Use `user_utterance` when the candidate directly uses the user's wording, and `system_reframed` when it is reframed from sustain talk.\n"
            "- `motivation_focus` is required. Always assign either `importance_related` or `confidence_related`.\n"
            "- `importance_related` means why change matters (values, reasons, need). `confidence_related` means how change might be carried out (ability, confidence, next-step feasibility).\n"
            "- Write human-readable fields such as `normalized_text` and `change_talk_inference` in natural English. Do not mix in placeholders or internal labels.\n"
            "- `evidence_quote` is a direct quote and may remain unchanged.\n"
            "- Extract primarily from the latest client utterance; do not fill gaps from older history.\n"
            "- Prioritize content that is strongly related to the provided inference slots.\n"
            "- If a target-behavior anchor is available, prioritize candidates with high `target_behavior_relevance` and include at least one candidate at 0.70 or above when possible.\n"
            "- Do not extract process-preference requests or preferences about pacing, style, or counselor involvement (for example: wanting to go slowly, wanting the listener to just listen first, not wanting suggestions yet) as change-talk candidates.\n"
            "- In particular, always exclude phrases like 'I want to go at my own pace', 'without rushing', or 'I do not know where to start' unless the same context also contains a concrete situation (when/where) and behavior (what to do).\n"
            "- Do not extract 'I want to protect my own pace' by itself unless it is tied to a concrete action, such as stating key points in one sentence before a deadline meeting.\n"
            "- In `focus_candidates.normalized_text`, never copy sustain-talk or process-preference sentences as-is. Write only the change-side proposition connected to what the client wants to change and how.\n"
            "- In `change_talk_inference`, never include sustain-talk or process-preference sentences directly. Keep it to 1-2 propositions that state only the change direction.\n"
            "- If the latest utterance contains broken wording or stray English tokens (for example `night`), do not mirror them verbatim in `normalized_text` or `change_talk_inference`. If the meaning is clear, normalize it into natural English; if unclear, avoid the token and keep only the change direction.\n"
            "- Do not leave phrases like 'at my own pace', 'without rushing', or 'while feeling things out' standing on their own. If kept, keep only the part tied to the change goal.\n"
            "- If the latest utterance contains concrete cues such as situations, moments, felt sense, bodily sensations, or action clues, preserve those specifics in `normalized_text` / `change_talk_inference` rather than abstracting them away.\n"
            "- `normalized_text` / `change_talk_inference` should not sound like writer meta-language (intent, flow, focus, support, stabilization, organizing). Write a sentence the client could plausibly say from lived experience.\n"
            "- If the utterance is a single word or short phrase (for example, 'sense of continuity'), do not invent a story beyond the evidence. Add only a short directional framing if needed.\n"
            "- If the latest utterance contains only process-preference wishes and no direction of change, it is acceptable to return `focus_candidates=[]` and `change_talk_inference=\"\"`. Do not force change talk.\n"
            "- Even when the user's wording is only weakly explicit, you may infer implied meaning within what the context reasonably supports.\n"
            "- Even for sustain-leaning statements such as ambivalence, uncertainty about importance, or low confidence, imagine the change-side motivation behind them and reframe it into DARN-CAT candidates.\n"
            "- When problems, burdens, negative emotions, worries, or ambivalence are described, always include at least one candidate about how the person wants that difficulty relieved, reduced, or improved.\n"
            "- Example: 'At night I get anxious and cannot stop overthinking' -> 'I want to ease that nighttime anxiety' (Need/Desire).\n"
            "- Example: 'The meaning of working feels thin right now' -> 'I want to put the meaning of working into words' (Reasons/Need).\n"
            "- Example: 'I am wavering because I cannot put my values into words' -> 'I want to put my values into words and sort them out' (Desire/Activation).\n"
            "- Example: 'During certification study I sometimes feel this really matters to me' -> 'I want to use those moments in certification study when something feels important to me as clues to put what I care about into words' (Reasons/Desire).\n"
            "- Example: 'I want to decide my speaking pace myself' alone is just a process preference and should not be extracted.\n"
            "[Strict extraction gate]\n"
            "- Only adopt content as a change-talk candidate when the desired direction of change can be stated in a single sentence.\n"
            "- Do not extract the following:\n"
            "  1. Mere descriptions of anxiety, confusion, or low confidence (for example: not knowing how to begin, lacking confidence, not feeling sure).\n"
            "  2. Evaluations or preferences about consultation process, speaking style, or expression style (for example: whether it feels like a good fit, whether keeping it short works better).\n"
            "  3. Meta-evaluations about meaning or interpersonal impression that do not include a direction for improving behavior or state.\n"
            "- Exclude sentences containing 'do not know', 'cannot grasp it', 'not confident', or 'cannot feel sure' unless the same context also explicitly states what the person wants to change and how.\n"
            "- If concrete cues are present, do not replace them with only abstract paraphrases such as 'I want to organize my values' or 'I want to stabilize the meaning-making'.\n"
            "- When in doubt, prefer not adopting the candidate.\n"
            "- Do not use explicitly double-sided formulations containing 'on the one hand'. If ambivalence exists, keep only the side that expresses DARN-CAT.\n"
        )
        system = inject_mi_knowledge(system, agent_name="change_talk_inferer")
        user = (
            f"[Current phase]{state.phase.value}\n"
            f"[Slot input policy]{prompt_slot_policy}\n"
            f"[All confirmed slots]{slot_context.get('confirmed_slots_text')}\n"
            f"[All confirmed slots (JSON)]{json.dumps(confirmed_slots, ensure_ascii=False)}\n"
            f"[Inference slots]{slot_context.get('prompt_slots_text')}\n"
            f"[Inference slots (JSON)]{json.dumps(prompt_slots, ensure_ascii=False)}\n"
            f"[Target-behavior anchor (prefer confirmed)]{slot_context.get('target_behavior_focus_text')}\n"
            f"[Bridge slots]{slot_context.get('bridge_fields_text')}\n"
            f"[Bridge slots (JSON)]{json.dumps(slot_context.get('bridge_slots') or [], ensure_ascii=False)}\n"
            f"[Inference direction]{direction_hint}\n"
            f"[Current client utterance]{user_text}\n"
            "Use the latest utterance as the primary evidence source and return JSON only."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        request_label = "layer2:change_talk_inferer"
        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label=request_label,
        )
        parsed = _parse_json_from_text(raw)
        history_end_turn = len(history)
        history_start_turn = max(1, history_end_turn - max(1, int(self.max_history_turns)) + 1)
        payload_source: Any = parsed if isinstance(parsed, Mapping) else raw
        candidates, legacy_text = _normalize_change_talk_infer_output(
            payload_source,
            user_text=user_text,
            history_start_turn=history_start_turn,
            history_end_turn=history_end_turn,
            linked_slots_hint=[str(item) for item in linked_slots_hint if str(item).strip()],
            default_text="",
        )
        if not candidates and features.change_talk >= 0.5:
            fallback_candidates = _build_change_talk_candidates_from_legacy_text(
                user_text,
                default_evidence_turn=history_end_turn,
                min_turn=history_start_turn,
                max_turn=history_end_turn,
                user_text=user_text,
                default_linked_slots=[str(item) for item in linked_slots_hint if str(item).strip()],
                max_candidates=2,
            )
            candidates = fallback_candidates

        target_behavior_focus = _normalize_slot_text(slot_context.get("target_behavior_focus"))
        selected_candidates = _select_change_talk_candidates_for_action(
            candidates=candidates,
            max_items=2,
            current_phase=state.phase,
            target_behavior_focus=target_behavior_focus,
            prioritize_target_behavior=_should_prioritize_target_behavior_for_phase(state.phase),
            importance_estimate=features.importance_estimate,
        )
        legacy_hint = _build_change_talk_hint_from_candidates(
            candidates=selected_candidates or candidates,
            current_phase=state.phase,
        )
        if not legacy_hint and candidates:
            legacy_hint = legacy_text
        if not candidates:
            legacy_hint = ""
        normalized_payload = {
            "focus_candidates": [dataclasses.asdict(candidate) for candidate in candidates],
            "change_talk_inference": legacy_hint,
        }
        debug = {
            "method": "llm",
            "raw_output": raw,
            "parsed": parsed,
            "output": legacy_hint,
            "focus_candidates": [dataclasses.asdict(candidate) for candidate in candidates],
            "selected_focus_candidates": [dataclasses.asdict(candidate) for candidate in selected_candidates],
            "slot_context": slot_context,
            "direction_hint": direction_hint,
            "direction_debug": direction_debug,
            "request_label": request_label,
        }
        return normalized_payload, debug


@dataclass
class LLMAffirmationDecider:
    """
    Use the LLM to estimate NONE/SIMPLE/COMPLEX scores and choose the affirmation mode from them.
    Fall back to the existing rule-based decision on failure.
    """

    llm: LLMClient
    temperature: float = 0.0
    max_history_turns: int = 6

    def decide(
        self,
        *,
        user_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
        features: PlannerFeatures,
    ) -> Tuple[AffirmationMode, Dict[str, Any]]:
        dialogue = _history_to_dialogue(history, max_turns=self.max_history_turns)
        feature_json = json.dumps(dataclasses.asdict(features), ensure_ascii=False)
        importance_slots = state.phase_slots.get(Phase.IMPORTANCE_PROMOTION.name, {})
        core_values_text = _normalize_slot_text(
            importance_slots.get("core_values") if isinstance(importance_slots, Mapping) else ""
        )
        gate_signal = {
            "phase_code": state.phase.name,
            "turn_index": state.turn_index,
            "phase_turns": state.phase_turns,
            "turns_since_affirm": state.turns_since_affirm,
            "turns_since_complex_affirm": state.turns_since_complex_affirm,
            "r_since_q": state.r_since_q,
            "reflect_streak": state.reflect_streak,
            "has_core_values": bool(core_values_text),
            "core_values_text": core_values_text,
        }
        gate_signal_json = json.dumps(gate_signal, ensure_ascii=False)

        try:
            system = (
                "You are the affirmation-mode scoring agent for an MI dialogue.\n"
                "Goal: score the fit of the three `add_affirm` modes (NONE / SIMPLE / COMPLEX) for the latest client utterance.\n"
                "Output constraint: return exactly one JSON object and nothing else; no explanation or code block.\n"
                "Scoring policy:\n"
                "- NONE: better to omit affirmation because it is more natural, safer, or better calibrated.\n"
                "- SIMPLE: a short affirmation helps preserve rapport and support forward movement.\n"
                "- COMPLEX: an affirmation that weaves in values, strengths, or meaning is appropriate without becoming excessive.\n"
                "Notes:\n"
                "- Treat `SINGLE` as synonymous with `SIMPLE`.\n"
                "- All three scores range from 0.0 to 1.0; higher means the mode is more appropriate.\n"
                "- `selected_mode` is optional, but it must be consistent with `scores`.\n"
                "Output format:\n"
                '{'
                '"scores":{"NONE":0.0,"SIMPLE":0.0,"COMPLEX":0.0},'
                '"selected_mode":"NONE|SIMPLE|COMPLEX",'
                '"confidence":0.0,'
                '"rationale":"brief reason"'
                '}'
            )
            system = inject_mi_knowledge(system, agent_name="affirmation_decider")
            user = (
                f"[Current phase]{state.phase.value}\n"
                f"[Features]{feature_json}\n"
                f"[Affirmation-decision signals]{gate_signal_json}\n"
                f"[Recent dialogue]\n{dialogue}\n"
                f"[Current client utterance]{user_text}\n"
                "Return JSON only."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            raw = self.llm.generate(
                messages,
                temperature=self.temperature,
                request_label="layer2:affirmation_decider",
            )
            parsed = _parse_json_from_text(raw)
            mode_scores, score_issues, selected_mode_hint = _parse_affirm_mode_scores(parsed)
            selected_mode, selector_debug = _select_affirm_mode_from_scores(
                user_text=user_text,
                features=features,
                state=state,
                mode_scores=mode_scores,
            )

            llm_confidence_raw = parsed.get("confidence") if isinstance(parsed, Mapping) else None
            llm_confidence = _clamp01(llm_confidence_raw, -1.0)
            debug = {
                "method": "llm_affirmation_decider",
                "raw_output": raw,
                "parsed": parsed if isinstance(parsed, Mapping) else None,
                "score_parse_issues": score_issues,
                "llm_selected_mode_hint": selected_mode_hint,
                "llm_confidence": round(float(llm_confidence), 4) if llm_confidence >= 0.0 else None,
                "mode_scores": selector_debug.get("mode_scores"),
                "gated_mode_scores": selector_debug.get("gated_mode_scores"),
                "selector_debug": selector_debug,
                "request_label": "layer2:affirmation_decider",
            }
            if isinstance(parsed, Mapping):
                rationale_text = _normalize_slot_text(parsed.get("rationale") or parsed.get("reason"))
                if rationale_text:
                    debug["llm_rationale"] = rationale_text
            return selected_mode, debug
        except Exception as exc:
            fallback_mode, fallback_debug = _decide_affirm_mode(
                user_text=user_text,
                features=features,
                state=state,
            )
            return fallback_mode, {
                "method": "llm_affirmation_decider_error_rule_fallback",
                "error": str(exc),
                "rule_debug": fallback_debug,
                "request_label": "layer2:affirmation_decider",
            }


# ----------------------------
# Risk detection (safety layer)
# ----------------------------
_RISK_SELF_HARM = [
    "want to die", "want to disappear", "suicide", "hang myself", "end my life", "do not want to keep living", "suicidal ideation", "self-cutting", "overdose",
]
_RISK_HARM_OTHERS = ["kill", "hurt you", "revenge", "harm"]
_RISK_SEVERE = ["auditory hallucination", "hallucination", "delusion", "cannot control it", "in panic", "out of control"]

# Simple patterns for detecting affirmation
_AFFIRM_PATTERNS = [
    r"great",
    r"great",
    r"working hard",
    r"working hard",
    r"keeping at it",
    r"engaging with",
    r"despite how hard it has been",
    r"effort",
    r"courage",
    r"have been able to",
    r"impressive",
    r"impressive",
    r"resourcefulness",
    r"important step",
    r"facing",
    r"moving forward",
]

_AFFIRM_COMPLEX_PATTERNS = [
    r"value",
    r"treasuring",
    r"valuing",
    r"authentic",
    r"belief",
    r"sense of responsibility",
    r"sincerity",
    r"persistence",
    r"stance",
    r"choosing",
    r"core axis",
]


@dataclass
class RuleBasedRiskDetector:
    """
    Simple rule-based detector for self-harm / harm-to-others risk (lightweight fallback).
    """

    def detect(
        self,
        *,
        user_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
    ) -> RiskAssessment:
        text = user_text
        hits = sum(1 for kw in _RISK_SELF_HARM if kw in text)
        hits += sum(1 for kw in _RISK_HARM_OTHERS if kw in text)
        severe_hits = [kw for kw in _RISK_SEVERE if kw in text]
        level = RiskLevel.NONE
        if hits >= 2 or any(kw in text for kw in _RISK_HARM_OTHERS):
            level = RiskLevel.HIGH
        elif hits == 1 or severe_hits:
            level = RiskLevel.MILD
        reason = None
        if level != RiskLevel.NONE:
            terms = {kw for kw in _RISK_SELF_HARM + _RISK_HARM_OTHERS + _RISK_SEVERE if kw in text}
            reason = "rule detection: " + ", ".join(sorted(terms)) if terms else "rule detection"
        return RiskAssessment(level=level, reason=reason)


@dataclass
class LLMRiskDetector:
    """
    Detector that uses the LLM to assess risk severity and returns one of none/mild/high.
    """

    llm: LLMClient
    temperature: float = 0.0
    max_history_turns: int = 8
    rule_fallback: RuleBasedRiskDetector = field(default_factory=RuleBasedRiskDetector)

    def detect(
        self,
        *,
        user_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
    ) -> RiskAssessment:
        dialogue = _history_to_dialogue(history, max_turns=self.max_history_turns)
        system = (
            "You are a classifier that judges safety risk in dialogue.\n"
            "Return one of `none / mild / high` in JSON.\n"
            "- high: imminent danger such as suicide attempt, concrete plan, or intent to harm others\n"
            "- mild: signs of suicidal ideation or suspected severe deterioration\n"
            "- none: none of the above\n"
            'Example output: {"level":"high","reason":"repeatedly says they want to die"}'
        )
        system = inject_mi_knowledge(system, agent_name="risk_detector")
        user = (
            f"[Current phase]{state.phase.value}\n"
            f"[Recent dialogue]\n{dialogue}\n"
            f"[Current client utterance]{user_text}\n"
            "Assess the safety risk and add a short reason."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label="safety:risk_detector",
        )
        parsed = _parse_json_from_text(raw)
        level_text: Optional[str] = None
        reason: Optional[str] = None
        if isinstance(parsed, dict):
            level_text = str(parsed.get("level") or parsed.get("risk") or "").lower()
            reason = parsed.get("reason") or parsed.get("note")

        mapping = {"none": RiskLevel.NONE, "low": RiskLevel.MILD, "mild": RiskLevel.MILD, "moderate": RiskLevel.MILD, "high": RiskLevel.HIGH}
        if level_text in mapping:
            level = mapping[level_text]
        else:
            fallback = self.rule_fallback.detect(user_text=user_text, history=history, state=state)
            return RiskAssessment(level=fallback.level, reason=fallback.reason, raw_output=raw)

        return RiskAssessment(level=level, reason=reason, raw_output=raw)


@dataclass
class LLMMIEvaluator:
    """
    LLM evaluator that gives a lightweight score for whether the response follows MI principles.
    """

    llm: LLMClient
    temperature: float = 0.0
    max_history_turns: int = 6

    def evaluate(
        self,
        *,
        action: MainAction,
        assistant_text: str,
        history: List[Tuple[str, str]],
        state: DialogueState,
    ) -> OutputEvaluation:
        dialogue = _history_to_dialogue(history, max_turns=self.max_history_turns)
        system = (
            "You are a reviewer that scores MI dialogue responses.\n"
            "Rate from 0 to 10 whether empathy, non-directiveness, autonomy support, and rolling with resistance are maintained.\n"
            'Output JSON only. Example: {"score":8.5,"feedback":"careful reflection but the question is leading"}\n'
            "- If the score is low, put a short improvement point in `feedback`.\n"
        )
        system = inject_mi_knowledge(system, agent_name="mi_evaluator")
        user = (
            f"[Current phase]{state.phase.value}\n"
            f"[Main action]{action.value}\n"
            f"[Recent dialogue]\n{dialogue}\n"
            f"[Assistant response for this turn]{assistant_text}\n"
            "Score MI adherence and, if needed, write a short improvement hint."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = self.llm.generate(
            messages,
            temperature=self.temperature,
            request_label="safety:mi_evaluator",
        )
        parsed = _parse_json_from_text(raw)
        score = 10.0
        feedback = None
        rewrite = None
        if isinstance(parsed, dict):
            try:
                score = float(parsed.get("score", score))
            except Exception:
                pass
            feedback_val = parsed.get("feedback") or parsed.get("comment")
            rewrite_val = parsed.get("rewrite")
            feedback = str(feedback_val) if feedback_val is not None else None
            rewrite = str(rewrite_val) if rewrite_val is not None else None

        score = max(0.0, min(10.0, score))
        return OutputEvaluation(score=score, feedback=feedback, raw_output=raw, rewrite=rewrite)


@dataclass
class _PromptNormalizedInputs:
    phase: Phase
    affirm_mode: AffirmationMode
    reflection_style_effective: Optional[ReflectionStyle]
    phase_slot_memory: Dict[str, Dict[str, str]]
    current_phase_slots: Dict[str, str]
    current_phase_slot_desc: str
    target_behavior_focus: str
    target_behavior_focus_enabled: bool
    phase_transition_note: str
    phase_hint: str
    rank_hint: str
    proposed_action_hint: str
    feature_hint: str
    change_talk_hint: str
    has_change_talk_hint: bool
    risk_note: str
    is_first_turn_greeting: bool
    first_turn_note: str


@dataclass(frozen=True)
class _ActionRuleContext:
    state: DialogueState
    reflection_style_effective: Optional[ReflectionStyle]
    current_user_is_short: bool
    allow_question_without_preface: bool
    closing_phase_complete: bool = False
    clarify_preference_mode: str = "default"
    clarify_preference_options: Tuple[str, ...] = ()


def _render_first_turn_note(
    *,
    state: DialogueState,
    first_turn_hint: Optional[FirstTurnHint],
) -> Tuple[bool, str]:
    is_first_turn_greeting = state.phase == Phase.GREETING and state.turn_index == 0
    if not is_first_turn_greeting:
        return False, ""
    if first_turn_hint == FirstTurnHint.GREETING_ONLY:
        return (
            True,
            "First-turn exception: the client only greeted you. Briefly return the greeting and appreciation, do not ask a question, and add one short reflection if possible.",
        )
    if first_turn_hint == FirstTurnHint.GREETING_WITH_TOPIC:
        return (
            True,
            "First-turn exception: the client greeted you and raised a topic. Return the greeting, express appreciation, and start with one short reflection only; do not ask a question yet.",
        )
    return (
        True,
        "First-turn exception: the client opened with a topic only. Offer a polite greeting and appreciation from your side, then give one short reflection only; do not ask a question yet.",
    )


def _render_risk_note(risk_assessment: Optional[RiskAssessment]) -> str:
    if risk_assessment is None:
        return ""
    if risk_assessment.level == RiskLevel.HIGH:
        return "High risk has been detected. Pause the normal flow and prioritize immediate safety and referral to appropriate support."
    if risk_assessment.level == RiskLevel.MILD:
        return "There is some safety concern. Proceed carefully and prioritize a sense of safety."
    return ""


def _extract_phase_debug_hints(
    phase_prediction_debug: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    phase_hint = "none"
    phase_transition_note = "none"
    if not isinstance(phase_prediction_debug, Mapping):
        return phase_hint, phase_transition_note

    phase_label = (
        phase_prediction_debug.get("parsed_phase")
        or phase_prediction_debug.get("fallback_phase")
        or phase_prediction_debug.get("predicted_phase")
        or phase_prediction_debug.get("current")
    )
    phase_method = phase_prediction_debug.get("method", "unknown")
    confidence = phase_prediction_debug.get("confidence")
    confidence_text = ""
    try:
        if confidence is not None:
            confidence_text = f", conf={float(confidence):.2f}"
    except Exception:
        confidence_text = ""
    if phase_label:
        phase_hint = f"{phase_label} (method={phase_method}{confidence_text})"
    else:
        phase_hint = f"method={phase_method}{confidence_text}"

    transition_text = _normalize_slot_text(phase_prediction_debug.get("phase_transition_note"))
    if transition_text:
        phase_transition_note = transition_text

    return phase_hint, phase_transition_note


def _extract_rank_and_feature_hints(
    *,
    action_ranking_debug: Optional[Dict[str, Any]],
    features: Optional[PlannerFeatures],
) -> Tuple[str, str, str]:
    rank_hint = "none"
    proposed_action_hint = "none"

    if action_ranking_debug:
        if action_ranking_debug.get("error"):
            rank_hint = f"error:{action_ranking_debug.get('error')}"
        elif action_ranking_debug.get("used_default"):
            rank_hint = "allowed_action_fallback(default)"
        elif action_ranking_debug.get("choice_rank"):
            rank_hint = ", ".join(str(v) for v in action_ranking_debug.get("choice_rank", []))
        elif action_ranking_debug.get("parsed") is not None:
            rank_hint = str(action_ranking_debug.get("parsed"))
        proposed_main_action = action_ranking_debug.get("proposed_main_action")
        if proposed_main_action:
            proposed_action_hint = str(proposed_main_action)

    feature_hint = "none"
    if features is not None:
        feature_hint = json.dumps(dataclasses.asdict(features), ensure_ascii=False)
    return rank_hint, proposed_action_hint, feature_hint


def _normalize_prompt_inputs(
    *,
    history: List[Tuple[str, str]],
    state: DialogueState,
    action: MainAction,
    add_affirm: AffirmationMode | bool | str,
    reflection_style: Optional[ReflectionStyle],
    risk_assessment: Optional[RiskAssessment],
    first_turn_hint: Optional[FirstTurnHint],
    phase_prediction_debug: Optional[Dict[str, Any]],
    action_ranking_debug: Optional[Dict[str, Any]],
    features: Optional[PlannerFeatures],
    change_talk_inference: Optional[str],
) -> _PromptNormalizedInputs:
    affirm_mode = _normalize_affirmation_mode(add_affirm)
    phase_slot_memory = _copy_phase_slot_memory(state.phase_slots)
    current_phase_slots = phase_slot_memory.get(state.phase.name, {})
    current_phase_slot_desc = ", ".join(
        f"{_PHASE_SLOT_LABELS.get(slot_key, slot_key)}={current_phase_slots.get(slot_key) or 'unset'}"
        for slot_key in _PHASE_SLOT_SCHEMA.get(state.phase, ())
    )
    target_behavior_focus = _get_confirmed_target_behavior_for_change_talk(state=state)
    phase_hint, phase_transition_note = _extract_phase_debug_hints(
        phase_prediction_debug
    )
    rank_hint, proposed_action_hint, feature_hint = _extract_rank_and_feature_hints(
        action_ranking_debug=action_ranking_debug,
        features=features,
    )

    reflection_style_effective: Optional[ReflectionStyle] = reflection_style
    if reflection_style_effective is None and _is_reflect_action(action):
        reflection_style_effective = _reflection_style_from_action(action)

    change_talk_hint = (change_talk_inference or "").strip() or "none"
    has_change_talk_hint = change_talk_hint != "none"

    is_first_turn_greeting, first_turn_note = _render_first_turn_note(
        state=state,
        first_turn_hint=first_turn_hint,
    )
    risk_note = _render_risk_note(risk_assessment)

    return _PromptNormalizedInputs(
        phase=state.phase,
        affirm_mode=affirm_mode,
        reflection_style_effective=reflection_style_effective,
        phase_slot_memory=phase_slot_memory,
        current_phase_slots=current_phase_slots,
        current_phase_slot_desc=current_phase_slot_desc,
        target_behavior_focus=target_behavior_focus,
        target_behavior_focus_enabled=bool(target_behavior_focus),
        phase_transition_note=phase_transition_note,
        phase_hint=phase_hint,
        rank_hint=rank_hint,
        proposed_action_hint=proposed_action_hint,
        feature_hint=feature_hint,
        change_talk_hint=change_talk_hint,
        has_change_talk_hint=has_change_talk_hint,
        risk_note=risk_note,
        is_first_turn_greeting=is_first_turn_greeting,
        first_turn_note=first_turn_note,
    )


def _render_role_and_style() -> str:
    return (
        "[Role and Tone]\n"
        "- You are a dialogue agent who supports people in the style of Motivational Interviewing (MI).\n"
        "- Your tone is polite, respectful, non-presumptive, non-confrontational, and supportive of autonomy."
    )


def _render_safety_rules(*, risk_note: str) -> str:
    lines = [
        "- If self-harm, harm to others, or imminent danger is suspected, pause the normal flow and prioritize safety and referral to appropriate support.",
    ]
    if risk_note:
        lines.append(f"- Safety note for this turn: {risk_note}")
    return "[Safety Rules: Highest-Priority Exception]\n" + "\n".join(lines)


def _render_hard_constraints(
    *,
    state: DialogueState,
    action: MainAction,
    normalized: _PromptNormalizedInputs,
) -> str:
    reflection_style_line = (
        normalized.reflection_style_effective.value
        if normalized.reflection_style_effective
        else "not_specified"
    )
    lines = [
        f"- Current phase: {state.phase.value}",
        f"- Main action for this turn: {action.value}",
        f"- Affirmation instruction: {normalized.affirm_mode.value}",
        f"- Reflection style for this turn: {reflection_style_line}",
        "- Do not re-decide the current phase, the main action for this turn, or the affirmation instruction during generation.",
    ]
    if normalized.first_turn_note:
        lines.append(f"- {normalized.first_turn_note}")
    return "[Locked Instructions for This Turn: Hard Constraints]\n" + "\n".join(lines)


def _render_phase_guidance(phase: Phase) -> str:
    planning = _PHASE_PLANNING_BOUNDARY.get(phase, "")
    if not planning:
        return ""
    return (
        "[Phase Goal and Planning Boundary]\n"
        f"- {phase.value} [{phase.name}]: {planning}"
    )


def _render_context_summary(
    *,
    normalized: _PromptNormalizedInputs,
) -> str:
    lines = [
        f"- Phase note: {normalized.phase_transition_note} (use: keep the response aligned with the stay/advance rationale)",
        f"- feature_extractor output: {normalized.feature_hint} (use: capture dialogue signals such as resistance, change talk, and short replies)",
        f"- Change talk summary: {normalized.change_talk_hint} (use: align the focus candidates for reflections, questions, and summaries)",
        f"- Current phase slots (JSON): {json.dumps(normalized.current_phase_slots, ensure_ascii=False)} (use: find missing information in the current phase)",
        f"- All phase slots (JSON): {json.dumps(normalized.phase_slot_memory, ensure_ascii=False)} (use: check consistency with existing information; the main focus is still the current phase)",
    ]
    if normalized.current_phase_slot_desc:
        lines.insert(
            3,
            f"- Required slots in the current phase: {normalized.current_phase_slot_desc}",
        )
    usage_lines = [
        "- Usage priority: dialogue history > required slots in the current phase > cross-phase consistency checks.",
        "- Response policy: choose the focus of reflections, questions, and summaries so that missing current-phase slots can be filled.",
        "- Prohibited: do not copy JSON key names or label names into the final response.",
    ]
    if normalized.phase in {Phase.REVIEW_REFLECTION, Phase.CLOSING}:
        raw_target_behavior = ""
        focusing_slots = normalized.phase_slot_memory.get(Phase.FOCUSING_TARGET_BEHAVIOR.name, {})
        if isinstance(focusing_slots, Mapping):
            raw_target_behavior = _normalize_slot_text(focusing_slots.get("target_behavior"))
        review_anchor_fields = _build_review_reflection_bridge_fields(
            phase_slot_memory=normalized.phase_slot_memory,
            raw_target_behavior=(raw_target_behavior or normalized.target_behavior_focus),
        )
        review_anchor_text = " / ".join(review_anchor_fields) if review_anchor_fields else "none"
        phase_code = normalized.phase.name
        lines.append(
            f"- {phase_code} integration anchors: {review_anchor_text}"
            " (use: review the conversation by referencing the presenting problem, problem scene, values, and target behavior in parallel)"
        )
        usage_lines.insert(
            1,
            f"- Exception: in {phase_code}, treat presenting_problem_raw / problem_scene / core_values / target_behavior with equal weight and avoid over-centering only one of them.",
        )
    elif normalized.target_behavior_focus_enabled:
        lines.append(
            f"- FOCUSING_TARGET_BEHAVIOR.target_behavior(confirmed, quality>=0.80): {normalized.target_behavior_focus}"
            " (use: highest-priority anchor for response consistency across all phases)"
        )
        usage_lines.insert(
            1,
            "- Exception: if the target_behavior above is already confirmed (quality>=0.80), prioritize consistency with that target behavior above all else.",
        )
    return (
        "[Context Summary]\n"
        + "\n".join(lines)
        + "\n[How to Use This Section]\n"
        + "\n".join(usage_lines)
    )


def _render_layer3_shared_inputs(
    *,
    state: DialogueState,
    action: MainAction,
    add_affirm: AffirmationMode,
    reflection_style: Optional[ReflectionStyle],
    normalized: _PromptNormalizedInputs,
    selected_focus_lines: Sequence[str],
    ct_anchor_seed_text: str,
    ct_operation_goal: str,
    focus_contract_hint_text: str,
    slot_target_label: str,
    slot_quality_target_examples_text: str,
    slot_quality_target_example_details: str,
    selected_slot_quality_target_example_text: str,
    selected_target_information_seed: str,
    evocation_move_seed: str,
    target_behavior_focus: str,
    macro_bridge_anchor: str = "",
    clarify_preference_mode: str = "default",
    focus_choice_options: Sequence[str] = (),
) -> str:
    reflection_style_text = reflection_style.value if reflection_style else "none"
    phase_boundary = _PHASE_PLANNING_BOUNDARY.get(state.phase, "none")
    current_phase_required_slots = normalized.current_phase_slot_desc or "none"
    current_phase_slots_json = json.dumps(normalized.current_phase_slots, ensure_ascii=False)
    all_phase_slots_json = json.dumps(normalized.phase_slot_memory, ensure_ascii=False)

    primary_focus_seed = selected_focus_lines[0] if selected_focus_lines else "none"
    change_talk_seed = focus_contract_hint_text or "none"
    missing_info_seed = slot_target_label or "none"
    target_behavior_anchor = target_behavior_focus or "none"
    focus_choice_options_text = _join_writer_items(list(focus_choice_options), fallback="none")
    end_phase_equal_anchors = "none"
    if state.phase in {Phase.REVIEW_REFLECTION, Phase.CLOSING}:
        raw_target_behavior = ""
        focusing_slots = normalized.phase_slot_memory.get(Phase.FOCUSING_TARGET_BEHAVIOR.name, {})
        if isinstance(focusing_slots, Mapping):
            raw_target_behavior = _normalize_slot_text(focusing_slots.get("target_behavior"))
        review_anchor_fields = _build_review_reflection_bridge_fields(
            phase_slot_memory=normalized.phase_slot_memory,
            raw_target_behavior=(raw_target_behavior or target_behavior_focus),
        )
        end_phase_equal_anchors = (
            " / ".join(review_anchor_fields) if review_anchor_fields else "none"
        )
    last_user_text = state.last_user_text or "none"
    last_substantive_user_text = state.last_substantive_user_text or "none"

    return (
        "[Layer3 Shared Variables]\n"
        f"- fixed_phase: {state.phase.value} [{state.phase.name}]\n"
        f"- fixed_main_action: {action.value}\n"
        f"- fixed_affirm_mode: {add_affirm.value}\n"
        f"- fixed_reflection_style: {reflection_style_text}\n"
        f"- primary_focus_seed: {primary_focus_seed}\n"
        f"- change_talk_seed: {change_talk_seed}\n"
        f"- ct_anchor_seed: {ct_anchor_seed_text or 'none'}\n"
        f"- ct_operation_goal_seed: {ct_operation_goal or 'none'}\n"
        f"- slot_target_label: {slot_target_label or 'none'}\n"
        f"- missing_info_seed: {missing_info_seed}\n"
        f"- clarify_preference_mode: {clarify_preference_mode or 'default'}\n"
        f"- focus_choice_options_seed: {focus_choice_options_text}\n"
        f"- slot_quality_target_examples_json: {slot_quality_target_examples_text}\n"
        f"- selected_slot_quality_target_example_json: {selected_slot_quality_target_example_text}\n"
        f"- selected_target_information_seed: {selected_target_information_seed}\n"
        f"- evocation_move_seed: {evocation_move_seed}\n"
        f"- macro_bridge_anchor: {macro_bridge_anchor or 'none'}\n"
        f"- slot_quality_target_example_detail_seed: {slot_quality_target_example_details}\n"
        "- slot_quality_target_example_detail_seed gives concrete anchor candidates such as time, place, quantity, tools, or steps. Keep anything that can be used naturally in the response without abstracting it.\n"
        "- If slot_quality_target_example_detail_seed looks like a question, agreement sentence, or meta sentence, do not copy the sentence pattern. Extract only the concrete wording about scene, action, or felt sense.\n"
        f"- current_phase_required_slots: {current_phase_required_slots}\n"
        f"- Current phase slots (JSON): {current_phase_slots_json}\n"
        f"- current_phase_slots_json: {current_phase_slots_json}\n"
        f"- all_phase_slots_json: {all_phase_slots_json}\n"
        f"- target_behavior_anchor(confirmed, quality>=0.80): {target_behavior_anchor}\n"
        f"- end_phase_equal_anchors: {end_phase_equal_anchors}\n"
        f"- repair_detail_target_quality: quality={_SLOT_REPAIR_HINT_TARGET_QUALITY:.2f}\n"
        f"- last_user_text: {last_user_text}\n"
        f"- last_substantive_user_text: {last_substantive_user_text}\n"
        f"- phase_boundary: {phase_boundary}\n"
        f"- Use quality>=0.80 as a confirmed/consistency anchor, and use quality={_SLOT_REPAIR_HINT_TARGET_QUALITY:.2f} as the threshold for repair-target phrasing.\n"
        "- Do not expose internal labels or variable names in the response body."
    )


def _render_action_rule_safety_override() -> str:
    return _get_layer3_action_prompt(
        "action_rules",
        "safety_override",
        default=(
            "Output requirements (safety mode):\n"
            "- Prioritize immediate safety first.\n"
            "- Focus on crisis response and pause the normal MI flow for now."
        ),
    )


def _render_action_rule_first_turn(
    first_turn_hint: Optional[FirstTurnHint],
) -> str:
    if first_turn_hint == FirstTurnHint.GREETING_ONLY:
        return _get_layer3_action_prompt(
            "action_rules",
            "first_turn",
            "greeting_only",
            default=(
                "Output requirements (first turn: greeting only):\n"
                "- Respond with a polite greeting in 1-2 concise sentences."
            ),
        )
    if first_turn_hint == FirstTurnHint.GREETING_WITH_TOPIC:
        return _get_layer3_action_prompt(
            "action_rules",
            "first_turn",
            "greeting_with_topic",
            default=(
                "Output requirements (first turn: greeting plus topic):\n"
                "- Include a brief greeting and reflection in 2-3 concise sentences."
            ),
        )
    return _get_layer3_action_prompt(
        "action_rules",
        "first_turn",
        "topic_only",
        default=(
            "Output requirements (first turn: topic only, no greeting):\n"
            "- Add a polite greeting and keep it concise in 2-3 sentences."
        ),
    )


def _render_reflect_action_rule(context: _ActionRuleContext) -> str:
    if context.closing_phase_complete:
        return _render_closing_final_turn_action_rule(context)
    if _is_closing_first_turn(context.state):
        return _render_closing_first_turn_action_rule(context)

    action_rule = _get_layer3_action_prompt(
        "action_rules",
        "reflect",
        "base",
        default=(
            "Output requirements (reflection / paraphrase):\n"
            "- Select and reflect the key point related to change talk.\n"
            "- Do not ask a question (do not end with a question mark)."
        ),
    )
    if context.state.phase == Phase.REVIEW_REFLECTION and context.state.phase_turns >= 1:
        action_rule += (
            "\n"
            + _get_layer3_action_prompt(
                "action_rules",
                "reflect",
                "review_feedback_append",
                default=(
                    "- In the review phase, anchor the response in what the client says is staying with them most or what they want to keep in mind next."
                ),
            )
        )
    if context.current_user_is_short:
        action_rule += (
            "\n"
            + _get_layer3_action_prompt(
                "action_rules",
                "reflect",
                "short_user_append",
                default=(
                    "- If the latest user turn is only a short reply or backchannel, do not merely paraphrase that turn by itself."
                ),
            )
        )

    style_specific_rule = {
        ReflectionStyle.SIMPLE: _get_layer3_action_prompt(
            "action_rules",
            "reflect",
            "style_simple",
            default="- Style: simple",
        ),
        ReflectionStyle.COMPLEX: _get_layer3_action_prompt(
            "action_rules",
            "reflect",
            "style_complex",
            default="- Style: complex",
        ),
        ReflectionStyle.DOUBLE_SIDED: _get_layer3_action_prompt(
            "action_rules",
            "reflect",
            "style_double",
            default="- Style: double",
        ),
    }.get(context.reflection_style_effective or ReflectionStyle.COMPLEX)
    if style_specific_rule:
        action_rule += "\n" + style_specific_rule
    return action_rule


def _render_question_action_rule(context: _ActionRuleContext) -> str:
    if _is_review_reflection_first_turn(context.state):
        return _render_review_first_turn_action_rule(context)

    if context.allow_question_without_preface:
        action_rule = _get_layer3_action_prompt(
            "action_rules",
            "question",
            "short_reply",
            default=(
                "Output requirements (Question: simple version for short replies):\n"
                "- Use at most two sentences in the basic structure.\n"
                "- If add_affirm != NONE, make sentence 1 a short affirmation and sentence 2 only a question about utterance_target.\n"
                "- If add_affirm == NONE, make sentence 1 a simple reflection grounded in question_reflect_seed and sentence 2 only a question about utterance_target."
            ),
        )
    else:
        action_rule = _get_layer3_action_prompt(
            "action_rules",
            "question",
            "default",
            default=(
                "Output requirements (Question):\n"
                "- Use at most two sentences in the basic structure.\n"
                "- If add_affirm != NONE, make sentence 1 a short affirmation and sentence 2 only a question about utterance_target.\n"
                "- If add_affirm == NONE, make sentence 1 a simple reflection grounded in question_reflect_seed and sentence 2 only a question about utterance_target.\n"
                "- Use only one question in principle."
            ),
        )
    if _is_scale_followup_pending(context.state):
        followup_step = _current_scale_followup_step(context.state)
        score_text = _format_scale_score_text(context.state.scale_followup_score)
        if followup_step == _SCALE_FOLLOWUP_STEP_PLUS_ONE:
            action_rule += (
                "\n"
                "- Briefly acknowledge the previous answer, then focus only on what would raise it by one point.\n"
                f"- Ask exactly one question: what would move it from {score_text} up by one point?\n"
                "- Do not stack a deeper reason-question here."
            )
        elif (context.state.scale_followup_score or 0.0) > 0.0:
            score = context.state.scale_followup_score
            should_continue_plus_one = _should_schedule_plus_one_after_reason(
                phase=context.state.phase,
                score=score,
            )
            action_rule += (
                "\n"
                "- Briefly acknowledge the previous scaling answer, and in this turn focus only on the reason question.\n"
                f"- Ask exactly one question: why is it {score_text} rather than 0?\n"
                + (
                    "- Save the one-point-up follow-up for the next turn."
                    if should_continue_plus_one
                    else "- Complete the reason check in this turn and do not stack additional follow-up questions."
                )
            )
        else:
            action_rule += (
                "\n"
                "- Briefly acknowledge the previous scaling answer, then ask only one question about what would raise it by one point.\n"
                "- Limit the question to what would move it up by one point."
            )
    return action_rule


def _render_scaling_question_action_rule(context: _ActionRuleContext) -> str:
    action_rule = _get_layer3_action_prompt(
        "action_rules",
        "scaling_question",
        "base",
        default=(
            "Output requirements (Scaling question):\n"
            "- Explicitly use a 0-10 scale and ask only for the number in a single question."
        ),
    )
    phase_specific_rule = _get_layer3_action_prompt(
        "action_rules",
        "scaling_question",
        "by_phase",
        context.state.phase.name,
        default="",
    )
    if phase_specific_rule:
        action_rule = f"{action_rule}\n{phase_specific_rule}"
    return action_rule


def _render_summary_action_rule(context: _ActionRuleContext) -> str:
    action_rule = _get_layer3_action_prompt(
        "action_rules",
        "summary",
        "base",
        default=(
            "Output requirements (Summary):\n"
            "- Organize the key points and do not include a question."
        ),
    )
    return action_rule


def _render_clarify_preference_action_rule(context: _ActionRuleContext) -> str:
    action_rule = _get_layer3_action_prompt(
        "action_rules",
        "clarify_preference",
        "base",
        default=(
            "Output requirements (Preference clarification: reflection + choice question):\n"
            "- One reflection sentence plus one choice question."
        ),
    )
    if (
        context.clarify_preference_mode == "focus_choice"
        and context.clarify_preference_options
    ):
        focus_choice_append = _get_layer3_action_prompt_formatted(
            "action_rules",
            "clarify_preference",
            "focus_choice",
            default=(
                "- In this turn, check which topic to start with rather than discussing process preference.\n"
                "- In the choice question, list 2-3 items from {focus_choice_options} briefly.\n"
                "- Do not imply that unchosen topics disappear."
            ),
            focus_choice_options=_join_writer_items(
                list(context.clarify_preference_options),
                fallback="none",
            ),
        )
        if focus_choice_append:
            action_rule = f"{action_rule}\n{focus_choice_append}"
    return action_rule


def _render_ask_permission_action_rule(context: _ActionRuleContext) -> str:
    if context.allow_question_without_preface:
        return _get_layer3_action_prompt(
            "action_rules",
            "ask_permission",
            "short_reply",
            default=(
                "Output requirements (Permission check: simple version for short replies):\n"
                "- Use at most two sentences in the basic structure.\n"
                "- If add_affirm != NONE, make sentence 1 a short affirmation and sentence 2 only a question checking whether information sharing is okay.\n"
                "- If add_affirm == NONE, make sentence 1 a simple reflection grounded in question_reflect_seed and sentence 2 only a question checking whether information sharing is okay.\n"
                "- Do not include the details of the information yet."
            ),
        )
    return _get_layer3_action_prompt(
        "action_rules",
        "ask_permission",
        "base",
        default=(
            "Output requirements (Permission check):\n"
            "- Use at most two sentences in the basic structure.\n"
            "- If add_affirm != NONE, make sentence 1 a short affirmation and sentence 2 only a question checking whether information sharing is okay.\n"
            "- If add_affirm == NONE, make sentence 1 a simple reflection grounded in question_reflect_seed and sentence 2 only a question checking whether information sharing is okay.\n"
            "- Do not include the details of the information yet.\n"
            "- Only the final sentence may be a question."
        ),
    )


def _render_provide_info_action_rule(context: _ActionRuleContext) -> str:
    action_rule = _get_layer3_action_prompt(
        "action_rules",
        "provide_info",
        "base",
        default=(
            "Output requirements (Information sharing: EPE Provide -> Elicit):\n"
            "- Share information neutrally and place one question at the end asking for the reaction."
        ),
    )
    return action_rule


def _render_unknown_action_rule(_: _ActionRuleContext) -> str:
    return _get_layer3_action_prompt(
        "action_rules",
        "unknown",
        default="Output requirements: unknown",
    )


def _render_review_first_turn_action_rule(context: _ActionRuleContext) -> str:
    action_rule = _get_layer3_action_prompt(
        "action_rules",
        "review_first_turn",
        "base",
        default=(
            "Output requirements (Review phase, turn 1: fixed structure):\n"
            "- Structure it as one complex reflection, then one summary, then one review question."
        ),
    )
    return action_rule


def _render_closing_first_turn_action_rule(context: _ActionRuleContext) -> str:
    style_name = (context.reflection_style_effective or ReflectionStyle.COMPLEX).value
    action_rule = _get_layer3_action_prompt_formatted(
        "action_rules",
        "closing_first_turn",
        "base",
        default=(
            "Output requirements (Closing phase, first turn: fixed structure):\n"
            "- After one short reflection sentence, place one end-check question."
        ),
        style_name=style_name,
    )
    return action_rule


def _render_closing_final_turn_action_rule(context: _ActionRuleContext) -> str:
    style_name = (context.reflection_style_effective or ReflectionStyle.COMPLEX).value
    return _get_layer3_action_prompt_formatted(
        "action_rules",
        "closing_final_turn",
        "base",
        default=(
            "Output requirements (Closing phase, final turn: fixed structure):\n"
            "- Use the selected reflection style ({style_name}) for one short reflective sentence.\n"
            "- Then add one sentence explicitly stating that today's session will end here.\n"
            "- Do not include a question (zero question marks).\n"
            "- Include a brief thank-you.\n"
            "- Do not add advice or suggestions."
        ),
        style_name=style_name,
    )


_ACTION_RULE_RENDERERS: Dict[MainAction, Callable[[_ActionRuleContext], str]] = {
    MainAction.REFLECT: _render_reflect_action_rule,
    MainAction.REFLECT_SIMPLE: _render_reflect_action_rule,
    MainAction.REFLECT_COMPLEX: _render_reflect_action_rule,
    MainAction.REFLECT_DOUBLE: _render_reflect_action_rule,
    MainAction.QUESTION: _render_question_action_rule,
    MainAction.SCALING_QUESTION: _render_scaling_question_action_rule,
    MainAction.SUMMARY: _render_summary_action_rule,
    MainAction.CLARIFY_PREFERENCE: _render_clarify_preference_action_rule,
    MainAction.ASK_PERMISSION_TO_SHARE_INFO: _render_ask_permission_action_rule,
    MainAction.PROVIDE_INFO: _render_provide_info_action_rule,
}


def _render_action_rule(
    *,
    action: MainAction,
    state: DialogueState,
    risk_assessment: Optional[RiskAssessment],
    first_turn_hint: Optional[FirstTurnHint],
    normalized: _PromptNormalizedInputs,
    allow_question_without_preface: bool,
    current_user_is_short: bool,
    closing_phase_complete: bool = False,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> str:
    if risk_assessment and risk_assessment.level == RiskLevel.HIGH:
        return _render_action_rule_safety_override()
    if normalized.is_first_turn_greeting:
        return _render_action_rule_first_turn(first_turn_hint)

    clarify_preference_mode, clarify_preference_options = _resolve_clarify_preference_mode(
        action=action,
        focus_choice_context=focus_choice_context,
    )
    context = _ActionRuleContext(
        state=state,
        reflection_style_effective=normalized.reflection_style_effective,
        current_user_is_short=current_user_is_short,
        allow_question_without_preface=allow_question_without_preface,
        closing_phase_complete=closing_phase_complete,
        clarify_preference_mode=clarify_preference_mode,
        clarify_preference_options=tuple(clarify_preference_options),
    )
    if _is_review_reflection_first_turn(state):
        return _render_review_first_turn_action_rule(context)
    if closing_phase_complete:
        return _render_closing_final_turn_action_rule(context)
    if _is_closing_first_turn(state):
        return _render_closing_first_turn_action_rule(context)
    renderer = _ACTION_RULE_RENDERERS.get(action, _render_unknown_action_rule)
    return renderer(context)


def _render_layer3_action_rule_for_writer(
    *,
    action: MainAction,
    state: DialogueState,
    risk_assessment: Optional[RiskAssessment],
    first_turn_hint: Optional[FirstTurnHint],
    reflection_style: Optional[ReflectionStyle],
    closing_phase_complete: bool = False,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> str:
    if risk_assessment and risk_assessment.level == RiskLevel.HIGH:
        return _render_action_rule_safety_override()
    if state.phase == Phase.GREETING and state.turn_index == 0:
        return _render_action_rule_first_turn(first_turn_hint)

    clarify_preference_mode, clarify_preference_options = _resolve_clarify_preference_mode(
        action=action,
        focus_choice_context=focus_choice_context,
    )
    context = _ActionRuleContext(
        state=state,
        reflection_style_effective=reflection_style or ReflectionStyle.COMPLEX,
        current_user_is_short=False,
        allow_question_without_preface=False,
        closing_phase_complete=closing_phase_complete,
        clarify_preference_mode=clarify_preference_mode,
        clarify_preference_options=tuple(clarify_preference_options),
    )
    if _is_review_reflection_first_turn(state):
        return _render_review_first_turn_action_rule(context)
    if closing_phase_complete:
        return _render_closing_final_turn_action_rule(context)
    if _is_closing_first_turn(state):
        return _render_closing_first_turn_action_rule(context)
    renderer = _ACTION_RULE_RENDERERS.get(action, _render_unknown_action_rule)
    return renderer(context)


def _render_task_section(action_rule: str) -> str:
    return _get_layer3_action_prompt_formatted(
        "common_prompts",
        "task_section_template",
        default=(
            "[Task for this turn: action rules]\n"
            "- Use not only the most recent turn but also the flow of the conversation so far.\n"
            "- Avoid lecturing, pushing, or speaking in absolute terms.\n"
            "- Offer advice, suggestions, or concrete options only in PROVIDE_INFO. Do not suggest actions in other main actions.\n"
            "- Forbidden style: avoid counselor-led wording such as 'I'll make it ...', 'I'll take care of ...', or 'Let's set this week's plan as ...'.\n"
            "- Keep agency with the user. When needed, briefly check whether the counselor's understanding is correct.\n"
            "- Outside PROVIDE_INFO, do not make proposal-style moves such as 'let's do ...'. Even in FOCUSING_TARGET_BEHAVIOR or NEXT_STEP_DECISION, draw out the user's own thinking rather than proposing from the counselor side. In REFLECT, respond as a reflective statement rather than turning it into a question; in complex reflection, add only one small implied meaning to support evocation.\n"
            "{action_rule}"
        ),
        action_rule=action_rule,
    )


def _render_output_format_constraints() -> str:
    return _get_layer3_action_prompt(
        "common_prompts",
        "output_format_constraints",
        default=(
            "[Output format constraints]\n"
            "- Keep each sentence from becoming too long.\n"
            "- If a sentence feels crowded, shorten it by removing unnecessary words or simplifying the clause structure rather than relying on punctuation tricks.\n"
            "- Do not dodge brevity constraints by splitting one idea into many short sentences. Keep the whole response concise while prioritizing natural flow.\n"
            "- Use plain language that a teenager could understand.\n"
            "- Avoid jargon, or explain it briefly if it is truly necessary.\n"
            "- Right before output, do a final naturalness check. If there are abrupt pointers, repetitive wording, or padded lead-ins, rewrite them more simply.\n"
            "- Prioritize a final sentence that connects naturally to the conversation and reads as natural English."
        ),
    )


def _render_writer_validation_checks(
    *,
    action: MainAction,
    add_affirm: AffirmationMode,
    state: DialogueState,
    first_turn_hint: Optional[FirstTurnHint],
    closing_phase_complete: bool = False,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> str:
    common_default = [
        "- Output exactly one response body only. Do not output JSON, role names, or code blocks.",
        "- Do not expose internal labels or English snake_case identifiers (internal IDs or key names) in the response body.",
        "- Use everyday language in a polite but not over-formal style. Do not use slang such as `w`, `www`, or `lol`-style markers.",
        "- `draft_response_text` is a meaning anchor, not a wording anchor. If abstract nouns remain, recast them into everyday language closer to `last_user_text` while preserving meaning.",
        "- Keep each sentence concise.",
        "- When tightening a sentence, do not mechanically delete commas. Shorten it naturally by removing unnecessary words and adjusting word order.",
        "- Forbidden style: wording such as 'I'll make it ...', 'I'll take care of ...', or 'Let's make this week's plan ...'.",
        "- Do not add new themes that drift away from the main point of the completed Layer3 draft.",
        "- Do not change agency, viewpoint, or ownership. Do not swap autonomy anchors such as 'you', 'your own sense', or 'decide for yourself' into counselor-centered phrasing such as 'I' or 'from our side'.",
        "- Preserve at least one concrete term from the user's wording. If `must_include` or `slot_quality_target_examples.detail` contains concrete language, keep at least one of those items.",
        "- Even if `must_include` is empty, keep at least one concrete cue from `last_user_text` when it contains bodily sensations, actions, settings, sounds, or times of day.",
        "- Do not erase the concrete focus of change talk. Do not flatten the specific direction signaled by `ct_anchors`, `utterance_target`, or `evocation_move` into generalities.",
        "- Strongly avoid replacing concrete content with abstraction. Even for brevity, do not delete the specifics and leave only abstract nouns.",
        "- Abstract nouns such as 'put into words', 'organize', 'reconsider', 'meaning', 'trigger', 'stance', 'awareness', 'learning', or 'takeaway' should be paraphrased into verbs, felt sense, actions, or situations unless the user used that wording themselves.",
        "- Even when abstract wording is needed, aim for no more than one such term in a turn and do not stack them.",
        "- In the final sentence, check for agency reversal, viewpoint reversal, loss of concrete terms, or over-abstraction, and fix any of those while keeping the sentence short and natural.",
    ]
    lines = _get_layer4_writer_prompt_lines(
        "validation_checks",
        "common",
        default_lines=common_default,
    )

    if _is_reflect_action(action):
        if closing_phase_complete:
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "closing_final_turn",
                    default_lines=[
                        "- In the final closing turn, do not include a question (zero question marks).",
                        "- Include one sentence that explicitly signals ending (for example: ending here / this is where we stop today).",
                    ],
                )
            )
        if _is_closing_first_turn(state):
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "reflect_closing",
                    default_lines=[
                        "- In the first closing turn, place one reflection sentence followed by one question checking whether it is okay to end.",
                        "- Make the final question about whether to end today's session here.",
                    ],
                )
            )
        else:
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "reflect_non_closing",
                    default_lines=["- In reflection actions, do not use a question mark."],
                )
            )
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "reflect_common",
                default_lines=[
                    "- Do not make the reflection longer than `draft_response_text`. Do not add meaning; if needed, trim it to keep it natural.",
                    "- Do not end with a mere paraphrase. Move the meaning forward by only one small step.",
                    "- In reflection, work only with the main proposition of `draft_response_text`.",
                    "- In reflection actions, do not give advice, recommendations, or decisions.",
                    "- Forbidden examples include phrasing such as 'How about trying ...', '... is effective', '... is appropriate', 'Let's keep doing ...', or 'I recommend ...'.",
                ],
            )
        )

    allow_questionless = (
        action == MainAction.QUESTION
        and state.turn_index == 0
        and state.phase == Phase.GREETING
        and first_turn_hint == FirstTurnHint.GREETING_ONLY
    )
    if action in {MainAction.QUESTION, MainAction.SCALING_QUESTION}:
        if allow_questionless:
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "question_allow_questionless",
                    default_lines=["- If this is only an initial greeting, a turn without a question is acceptable."],
                )
            )
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "question_target",
                    default_lines=["- Limit the question target to one unresolved point or request from `draft_response_text`."],
                )
            )
        else:
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "question_general",
                    default_lines=[
                        "- Ask exactly one question, and make it an explicit interrogative sentence.",
                        "- Limit the question target to one unresolved point or request from `draft_response_text`.",
                    ],
                )
            )
        if action == MainAction.QUESTION and _is_scale_followup_pending(state):
            followup_step = _current_scale_followup_step(state)
            score_text = _format_scale_score_text(state.scale_followup_score)
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "question_followup_intro",
                    default_lines=["- Briefly acknowledge the previous answer, then place only one follow-up question."],
                )
            )
            if followup_step == _SCALE_FOLLOWUP_STEP_PLUS_ONE:
                lines.extend(
                    _split_nonempty_lines(
                        _get_layer4_writer_prompt_formatted(
                            "validation_checks",
                            "question_followup_plus_one_template",
                            default=(
                                f"- In this turn, ask only one question: what would raise it from {score_text} by one point?\n"
                                "- Do not stack a reason question in this same turn."
                            ),
                            score_text=score_text,
                        )
                    )
                )
            elif (state.scale_followup_score or 0.0) > 0.0:
                lines.extend(
                    _split_nonempty_lines(
                        _get_layer4_writer_prompt_formatted(
                            "validation_checks",
                            "question_followup_reason_template",
                            default=(
                                f"- In this turn, ask only one question: why is it {score_text} rather than 0?\n"
                                "- Do not stack the one-point-up question in this same turn."
                            ),
                            score_text=score_text,
                        )
                    )
                )
            else:
                lines.extend(
                    _get_layer4_writer_prompt_lines(
                        "validation_checks",
                        "question_followup_default",
                        default_lines=["- In this turn, ask only one question about what would raise it by one point."],
                    )
                )
        if action == MainAction.SCALING_QUESTION:
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "scaling_question",
                    default_lines=[
                        "- Explicitly use a 0-10 scale and ask only for the number.",
                        "- Do not combine a reason question or one-point-up question into the same question.",
                        "- Do not abstract the scale target into vague labels such as 'this practice' or 'this method'.",
                    ],
                )
            )
        if _is_review_reflection_first_turn(state):
            lines.extend(
                _get_layer4_writer_prompt_lines(
                    "validation_checks",
                    "review_first_turn",
                    default_lines=[
                        "- In the first review-reflection turn, structure the turn as one complex reflection, one summary, then one review question.",
                        "- The final question should open naturally in everyday language, such as asking what stands out most from today's talk or what the client wants to keep in mind next.",
                        "- Do not include an affirmation sentence in this turn.",
                    ],
                )
            )

    if action == MainAction.SUMMARY:
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "summary",
                default_lines=[
                    "- In SUMMARY, do not include a question.",
                    "- Follow Layer3 action_rules for sentence count, structure, and ending in SUMMARY; do not add extra Layer4-only structure.",
                ],
            )
        )

    if action == MainAction.CLARIFY_PREFERENCE:
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "clarify_preference",
                default_lines=[
                    "- Ask exactly one choice question.",
                    "- Summarize each option as a short phrase and avoid overloading them with modifiers.",
                    "- Avoid heavy repeated paired constructions such as 'between ... and ...'.",
                ],
            )
        )
        clarify_preference_mode, focus_choice_options = _resolve_clarify_preference_mode(
            action=action,
            focus_choice_context=focus_choice_context,
        )
        if clarify_preference_mode == "focus_choice" and focus_choice_options:
            lines.extend(
                _split_nonempty_lines(
                    _get_layer4_writer_prompt_formatted(
                        "validation_checks",
                        "clarify_preference_focus_choice",
                        default=(
                            "- In this turn, ask a question about which topic to start with rather than about process preference.\n"
                            "- Use {focus_choice_options} as the options so the client can choose where to begin."
                        ),
                        focus_choice_options=_join_writer_items(
                            focus_choice_options,
                            fallback="none",
                        ),
                    )
                )
            )

    if action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "ask_permission",
                default_lines=[
                    "- Keep the basic structure to at most two sentences, and make only the final sentence a question.",
                    "- If add_affirm != NONE, use one affirmation sentence plus one permission-check question; if add_affirm == NONE, use one simple reflection sentence plus one permission-check question.",
                    "- Include at least one of these ideas in the final question: information, sharing, options, methods, or suggestions.",
                ],
            )
        )

    if action == MainAction.PROVIDE_INFO:
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "provide_info",
                default_lines=[
                    "- Include wording such as 'as one option' or present explicit options (A/B/C, etc.).",
                    "- End with exactly one question asking for the client's reaction.",
                ],
            )
        )

    if add_affirm == AffirmationMode.NONE:
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "affirm_none",
                default_lines=[
                    "- Do not include an affirmation sentence in this turn.",
                    "- When there is no affirmation, avoid praise-oriented wording such as 'great', 'trying hard', 'effort', 'resourcefulness', 'facing it', or 'keeping at it'.",
                ],
            )
        )
    elif add_affirm == AffirmationMode.SIMPLE:
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "affirm_simple",
                default_lines=[
                    "- Include one short affirmation sentence.",
                    "- Ground it in one observable piece of speech, choice, or action from this turn.",
                    "- A basic form is: 'Even with the uncertainty or hesitation, you are trying to ...'.",
                ],
            )
        )
    else:
        lines.extend(
            _get_layer4_writer_prompt_lines(
                "validation_checks",
                "affirm_complex",
                default_lines=[
                    "- Include one affirmation sentence grounded in the context.",
                    "- Start from the observed fact.",
                    "- Then add only one short phrase about the direction that action points toward or what seems important to preserve.",
                    "- Do not force a nominalized strength or value label. When unsure, prefer a behavior-based affirmation over a personality label.",
                ],
            )
        )

    header = _get_layer4_writer_prompt(
        "validation_checks",
        "header",
        default="[Pre-output check: required]",
    )
    return f"{header}\n" + "\n".join(lines)


def _render_affirm_rule(affirm_mode: AffirmationMode) -> str:
    if affirm_mode == AffirmationMode.NONE:
        return _get_layer4_writer_prompt(
            "affirmation_rules",
            "none",
            default=(
                "[Affirmation]\n"
                "- Do not include an affirmation sentence in this turn.\n"
                "- Do not use stock praise phrases that evaluate effort or strengths (for example: great, working hard, being resourceful)."
            ),
        )
    if affirm_mode == AffirmationMode.SIMPLE:
        return _get_layer4_writer_prompt(
            "affirmation_rules",
            "simple",
            default=(
                "[Affirmation: SIMPLE]\n"
                "- Add one short affirmation sentence.\n"
                "- Ground it in only one observable sign of effort, resourcefulness, persistence, or willingness to face the issue.\n"
                "- A basic form is: 'Even with the uncertainty or hesitation, you are trying to ...'.\n"
                "- Avoid excessive praise or overconfident labeling.\n"
                "- Do not say 'I acknowledge' or 'I evaluate'; reflect it briefly as a statement.\n"
                "- In QUESTION / ASK_PERMISSION_TO_SHARE_INFO, place the affirmation before the question.\n"
            ),
        )
    return _get_layer4_writer_prompt(
        "affirmation_rules",
        "complex",
        default=(
            "[Affirmation: COMPLEX]\n"
            "- Add one context-grounded affirmation sentence.\n"
            "- Prefer a two-part structure: first the observed fact, then one short phrase about the direction it points toward or what seems important.\n"
            "- Do not force a nominalized strength or value label. When unsure, prefer a behavior-based affirmation over a personality label.\n"
            "- If a label feels too strong, soften it by compressing it into forms like 'trying to ...' or 'trying to keep ... going' rather than deleting it entirely.\n"
            "- Base it on concrete evidence in the client's wording rather than on flattery.\n"
            "- Do not say 'I acknowledge' or 'I evaluate'; reflect it briefly as a statement.\n"
            "- In QUESTION / ASK_PERMISSION_TO_SHARE_INFO, place the affirmation before the question.\n"
        ),
    )


def _render_debug_block(normalized: _PromptNormalizedInputs) -> str:
    return (
        "[Debug information: do not reflect in the output text]\n"
        "- This section is only for log reference. Never output it directly in the response.\n"
        f"- phase_debug output: {normalized.phase_hint}\n"
        f"- action_ranker output: {normalized.rank_hint}\n"
        f"- action_ranker proposed main_action: {normalized.proposed_action_hint}\n"
        f"- feature_extractor output: {normalized.feature_hint}"
    )


def _assemble_system(sections: Sequence[str]) -> str:
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _assemble_system_with_runtime_tail(
    *,
    fixed_sections: Sequence[str],
    runtime_sections: Sequence[str],
    runtime_header: str = "[Runtime context: updated every turn]",
) -> str:
    fixed = _assemble_system(fixed_sections)
    runtime = _assemble_system(runtime_sections)
    if not runtime:
        return fixed
    if not fixed:
        return runtime
    return f"{fixed}\n\n{runtime_header}\n{runtime}"


def _render_runtime_risk_note(risk_note: str) -> str:
    risk = _sanitize_human_text(risk_note)
    if not risk:
        return ""
    return "[Runtime safety note]\n" f"- Safety note for this turn: {risk}"


def _build_messages_with_history_limit(
    *,
    system: str,
    history: List[Tuple[str, str]],
    max_turns: int,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    for role, text in history[-max(1, int(max_turns)):]:
        messages.append({"role": role, "content": text})
    return messages


def _build_messages(*, system: str, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    return _build_messages_with_history_limit(system=system, history=history, max_turns=20)


def _history_to_numbered_dialogue(
    history: List[Tuple[str, str]],
    *,
    max_turns: int,
) -> Tuple[str, int, int]:
    total = len(history)
    start = max(0, total - max(1, int(max_turns)))
    lines: List[str] = []
    for idx, (role, text) in enumerate(history[start:], start=start + 1):
        prefix = "Client" if role == "user" else "Counselor"
        lines.append(f"T{idx} {prefix}: {text}")
    return "\n".join(lines), start + 1, total


def _default_question_count_max(action: MainAction) -> int:
    if action in {
        MainAction.QUESTION,
        MainAction.SCALING_QUESTION,
        MainAction.CLARIFY_PREFERENCE,
        MainAction.ASK_PERMISSION_TO_SHARE_INFO,
        MainAction.PROVIDE_INFO,
    }:
        return 1
    return 0


def _default_question_shape_for_action(action: MainAction) -> str:
    if action == MainAction.SCALING_QUESTION:
        return "scale_question"
    if _default_question_count_max(action) > 0:
        return "open_question"
    return "reflective_check"


def _response_focus_contains_internal_label(focus: Mapping[str, Any]) -> bool:
    for key in ("topic", "meaning", "why_now"):
        if _find_internal_label_leak(focus.get(key, "")):
            return True
    quotes = focus.get("evidence_quotes")
    if isinstance(quotes, Sequence) and not isinstance(quotes, (str, bytes)):
        for item in quotes:
            if _find_internal_label_leak(item):
                return True
    return False


def _format_focus_candidate_for_prompt(candidate: ChangeTalkCandidate) -> str:
    topic = _sanitize_human_text(candidate.normalized_text)
    if not topic:
        return ""
    evidence = _sanitize_human_text(candidate.evidence_quote)
    detail_parts: List[str] = []
    if evidence and evidence != topic:
        detail_parts.append(f'Evidence: "{evidence}"')
    if isinstance(candidate.evidence_turn, int) and candidate.evidence_turn > 0:
        detail_parts.append(f"Source: T{candidate.evidence_turn}")
    if detail_parts:
        return f"{topic}（{' / '.join(detail_parts)}）"
    return topic


def _format_focus_candidates_for_prompt(
    candidates: Sequence[ChangeTalkCandidate],
    *,
    max_items: int,
) -> List[str]:
    lines: List[str] = []
    for candidate in list(candidates)[:max_items]:
        line = _format_focus_candidate_for_prompt(candidate)
        if not line:
            continue
        if line in lines:
            continue
        lines.append(line)
    return lines


def _format_focus_for_writer(focus: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(focus, Mapping):
        return "none"
    topic = _sanitize_human_text(focus.get("topic"))
    meaning = _sanitize_human_text(focus.get("meaning"))
    why_now = _sanitize_human_text(focus.get("why_now"))
    core = ""
    if topic and meaning:
        core = f"{topic}（{meaning}）"
    else:
        core = topic or meaning
    if core and why_now:
        return f"{core} / Priority reason: {why_now}"
    if why_now:
        return f"Priority reason: {why_now}"
    return core or "none"


def _join_writer_items(values: Sequence[str], *, fallback: str = "none") -> str:
    cleaned: List[str] = []
    for value in values:
        text = _sanitize_human_text(value)
        if not text:
            continue
        if text in cleaned:
            continue
        cleaned.append(text)
        if len(cleaned) >= 5:
            break
    if not cleaned:
        return fallback
    return " / ".join(cleaned)


def _format_writer_plan_for_prompt(
    plan: Optional[Mapping[str, Any]],
    *,
    fallback: str = "none",
) -> str:
    if not isinstance(plan, Mapping):
        return fallback
    parts: List[str] = []
    opening = _sanitize_human_text(plan.get("opening_function"))
    core = _sanitize_human_text(plan.get("core_function"))
    closing = _sanitize_human_text(plan.get("closing_function"))
    if opening:
        parts.append(f"Opening={opening}")
    if core:
        parts.append(f"Core={core}")
    if closing:
        parts.append(f"Closing={closing}")
    try:
        question_count_max = int(plan.get("question_count_max"))
    except Exception:
        question_count_max = -1
    if question_count_max >= 0:
        parts.append(f"Question limit={question_count_max}")
    preferred_ending_family = _sanitize_human_text(plan.get("preferred_ending_family"))
    if preferred_ending_family:
        parts.append(f"Preferred ending family={preferred_ending_family}")
    raw_avoid_recent_ending_families = plan.get("avoid_recent_ending_families")
    avoid_recent_ending_families: List[str] = []
    if isinstance(raw_avoid_recent_ending_families, Sequence) and not isinstance(
        raw_avoid_recent_ending_families, (str, bytes)
    ):
        for item in raw_avoid_recent_ending_families:
            text = _sanitize_human_text(item)
            if not text or text in avoid_recent_ending_families:
                continue
            avoid_recent_ending_families.append(text)
            if len(avoid_recent_ending_families) >= 3:
                break
    if avoid_recent_ending_families:
        parts.append(f"Avoid ending families={_join_writer_items(avoid_recent_ending_families)}")
    return " / ".join(parts) if parts else fallback


def _format_layer4_repair_issue_codes_for_prompt(
    issue_codes: Optional[Sequence[Any]],
    *,
    fallback: str = "none",
) -> str:
    if not isinstance(issue_codes, Sequence) or isinstance(issue_codes, (str, bytes)):
        return fallback
    codes: List[str] = []
    for item in issue_codes:
        code = _normalize_slot_text(item)
        if not code:
            continue
        if code in codes:
            continue
        codes.append(code)
        if len(codes) >= 6:
            break
    return " / ".join(codes) if codes else fallback


def _build_layer4_repair_issue_codes_for_draft(
    *,
    schema_issues: Sequence[Any],
    validation_ok: bool,
    validation_reason: str,
    soft_warnings: Sequence[str],
) -> Dict[str, List[str]]:
    hard_contract = _merge_issue_codes(schema_issues)
    normalized_reason = _normalize_slot_text(validation_reason)
    if not validation_ok and normalized_reason in _LAYER4_DRAFT_HARD_CONTRACT_ISSUE_CODES:
        hard_contract = _merge_issue_codes(hard_contract, [normalized_reason])

    soft_hint: List[str] = []
    if isinstance(soft_warnings, Sequence) and not isinstance(soft_warnings, (str, bytes)):
        for warning in soft_warnings:
            normalized_warning = _normalize_slot_text(warning)
            if not normalized_warning or normalized_warning not in _LAYER4_DRAFT_SOFT_HINT_WARNING_CODES:
                continue
            if normalized_warning in soft_hint:
                continue
            soft_hint.append(normalized_warning)
            if len(soft_hint) >= 6:
                break

    return {
        "hard_contract": hard_contract,
        "soft_hint": soft_hint,
    }


def _normalize_layer4_repair_issue_codes(
    issue_groups: Optional[Mapping[str, Sequence[Any]]],
) -> Dict[str, List[str]]:
    if not isinstance(issue_groups, Mapping):
        return {
            "hard_contract": [],
            "soft_hint": [],
        }
    return {
        "hard_contract": _merge_issue_codes(issue_groups.get("hard_contract")),
        "soft_hint": _merge_issue_codes(issue_groups.get("soft_hint")),
    }


def _copy_layer4_repair_issue_codes(
    issue_groups: Optional[Mapping[str, Sequence[Any]]],
) -> Dict[str, List[str]]:
    normalized = _normalize_layer4_repair_issue_codes(issue_groups)
    return {
        "hard_contract": list(normalized.get("hard_contract") or []),
        "soft_hint": list(normalized.get("soft_hint") or []),
    }


def _merge_layer4_repair_issue_codes(
    base_issue_groups: Optional[Mapping[str, Sequence[Any]]],
    *,
    hard_contract: Optional[Sequence[Any]] = None,
    soft_hint: Optional[Sequence[Any]] = None,
) -> Dict[str, List[str]]:
    normalized = _normalize_layer4_repair_issue_codes(base_issue_groups)
    return {
        "hard_contract": _merge_issue_codes(normalized.get("hard_contract"), hard_contract or []),
        "soft_hint": _merge_issue_codes(normalized.get("soft_hint"), soft_hint or []),
    }


def _format_layer4_repair_issue_code_groups_for_prompt(
    issue_groups: Optional[Mapping[str, Sequence[Any]]],
    *,
    fallback: str = "none",
) -> Dict[str, str]:
    normalized = _normalize_layer4_repair_issue_codes(issue_groups)
    return {
        "hard_contract": _format_layer4_repair_issue_codes_for_prompt(
            normalized.get("hard_contract"),
            fallback=fallback,
        ),
        "soft_hint": _format_layer4_repair_issue_codes_for_prompt(
            normalized.get("soft_hint"),
            fallback=fallback,
        ),
    }


def _render_layer4_issue_repair_guidance(
    issue_groups: Optional[Mapping[str, Sequence[Any]]],
    *,
    action: MainAction,
) -> str:
    normalized = _normalize_layer4_repair_issue_codes(issue_groups)
    issue_codes = _merge_issue_codes(
        normalized.get("hard_contract") or [],
        normalized.get("soft_hint") or [],
    )
    if not issue_codes:
        return ""

    issue_repair_node = _get_layer4_writer_prompt_node("issue_repair")
    if not isinstance(issue_repair_node, Mapping):
        issue_repair_node = {}

    header = _normalize_slot_text(issue_repair_node.get("header")) or "[Layer3 repair guide by violation code]"
    intro_line = (
        _normalize_slot_text(issue_repair_node.get("intro_line"))
        or "Fix the matching issues below first. Do not add repairs for codes that are not present."
    )
    guidance_node = issue_repair_node.get("guidance")
    guidance_map = guidance_node if isinstance(guidance_node, Mapping) else {}

    lines: List[str] = [header, f"- {intro_line.lstrip('-').strip()}"]
    fallback_line = ""
    if action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        fallback_line = _normalize_slot_text(issue_repair_node.get("fallback_ask_permission"))
    elif action == MainAction.SUMMARY:
        fallback_line = _normalize_slot_text(issue_repair_node.get("fallback_summary"))
    elif _is_reflect_action(action):
        fallback_line = _normalize_slot_text(issue_repair_node.get("fallback_reflect"))
    elif action in _QUESTION_LIKE_ACTIONS:
        fallback_line = _normalize_slot_text(issue_repair_node.get("fallback_question_like"))
    if fallback_line:
        lines.append(f"- {fallback_line.lstrip('-').strip()}")

    for issue_code in issue_codes:
        raw_line = _normalize_slot_text(guidance_map.get(issue_code))
        if not raw_line:
            raw_line = _normalize_slot_text(_LAYER4_FALLBACK_ISSUE_REPAIR_GUIDANCE.get(issue_code))
        if not raw_line:
            continue
        line = f"- {raw_line.lstrip('-').strip()}"
        if line in lines:
            continue
        lines.append(line)

    if len(lines) <= 2 and not fallback_line:
        return ""
    return "\n".join(lines)


def _format_slot_quality_target_examples_for_prompt(
    examples: Sequence[Mapping[str, Any]],
    *,
    fallback: str = "none",
) -> str:
    lines: List[str] = []
    for example in examples:
        if not isinstance(example, Mapping):
            continue
        slot_label = _sanitize_human_text(example.get("slot_label"))
        slot_key = _slot_key_from_focus_text(example.get("slot_key"))
        if not slot_label and slot_key:
            slot_label = _sanitize_human_text(_PHASE_SLOT_LABELS.get(slot_key, slot_key))
        issue_type = _sanitize_human_text(example.get("issue_type"))
        if issue_type not in _SLOT_REPAIR_PROBE_STYLE:
            issue_type = "too_vague"
        probe_style = (
            _sanitize_human_text(example.get("preferred_probe_style"))
            or _SLOT_REPAIR_PROBE_STYLE.get(issue_type, _SLOT_REPAIR_PROBE_STYLE["too_vague"])
        )
        if _find_internal_label_leak(slot_label):
            continue
        target_information = _extract_slot_quality_target_information(example)
        detail = _extract_slot_quality_target_example_detail(example)
        line = f"{slot_label or 'unspecified slot'} [{issue_type}] {probe_style}"
        if target_information:
            line += f" / Needed information: {target_information}"
        if detail:
            line += f" / Utterance example: {detail}"
        if line in lines:
            continue
        lines.append(line)
        if len(lines) >= _MAX_SLOT_REPAIR_HINTS_PER_TURN:
            break
    if not lines:
        return fallback
    return " / ".join(lines)


def _format_slot_quality_target_example_details_for_prompt(
    examples: Sequence[Mapping[str, Any]],
    *,
    fallback: str = "none",
) -> str:
    lines: List[str] = []
    for example in examples:
        if not isinstance(example, Mapping):
            continue
        detail = _extract_slot_quality_target_example_detail(example)
        if not detail:
            continue
        slot_label = _sanitize_human_text(example.get("slot_label"))
        slot_key = _slot_key_from_focus_text(example.get("slot_key"))
        if not slot_label and slot_key:
            slot_label = _sanitize_human_text(_PHASE_SLOT_LABELS.get(slot_key, slot_key))
        if _find_internal_label_leak(slot_label):
            continue
        line = f"{slot_label or 'unspecified slot'}: {detail}"
        if line in lines:
            continue
        lines.append(line)
        if len(lines) >= _MAX_SLOT_REPAIR_HINTS_PER_TURN:
            break
    if not lines:
        return fallback
    return " / ".join(lines)


def _slot_quality_target_examples_from_phase_debug(
    phase_prediction_debug: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if _did_phase_transition_for_current_turn(phase_prediction_debug):
        return []
    if not isinstance(phase_prediction_debug, Mapping):
        return []
    if "slot_quality_target_examples" in phase_prediction_debug:
        raw = phase_prediction_debug.get("slot_quality_target_examples")
    elif "slot_repair_hints" in phase_prediction_debug:
        raw = phase_prediction_debug.get("slot_repair_hints")
    else:
        slot_review = phase_prediction_debug.get("slot_review")
        raw = (
            slot_review.get("slot_quality_target_examples")
            if isinstance(slot_review, Mapping) and slot_review.get("slot_quality_target_examples") is not None
            else (
                slot_review.get("slot_repair_hints")
                if isinstance(slot_review, Mapping)
                else None
            )
        )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    examples: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        example = dict(item)
        target_information = _extract_slot_quality_target_information(item)
        detail = _extract_slot_quality_target_example_detail(item)
        if target_information:
            example["target_information"] = target_information
        if detail:
            example["detail"] = detail
        examples.append(example)
    return examples


def _format_slot_repair_hints_for_prompt(
    hints: Sequence[Mapping[str, Any]],
    *,
    fallback: str = "none",
) -> str:
    return _format_slot_quality_target_examples_for_prompt(hints, fallback=fallback)


def _format_slot_repair_detail_targets_for_prompt(
    hints: Sequence[Mapping[str, Any]],
    *,
    fallback: str = "none",
) -> str:
    return _format_slot_quality_target_example_details_for_prompt(hints, fallback=fallback)


def _slot_repair_hints_from_phase_debug(
    phase_prediction_debug: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return _slot_quality_target_examples_from_phase_debug(phase_prediction_debug)


def _select_slot_quality_target_example(
    examples: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    first_example: Optional[Dict[str, Any]] = None
    for item in examples:
        if not isinstance(item, Mapping):
            continue
        example = dict(item)
        if first_example is None:
            first_example = example
        if _extract_slot_quality_target_information(example) or _extract_slot_quality_target_example_detail(example):
            return example
    return first_example


_TURN_OBJECTIVE_QUESTION_SHAPES: Tuple[str, ...] = (
    "none",
    "open_question",
    "gentle_clarification",
    "scale_question",
    "example_probe",
    "scope_narrowing_question",
    "reflective_check",
)

_CT_OPERATION_GOALS: Tuple[str, ...] = (
    "Elicit",
    "Strengthen",
    "Deepen",
    "Shift",
)


def _normalize_turn_objective_question_shape(value: Any, *, default: str) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if raw in _TURN_OBJECTIVE_QUESTION_SHAPES:
        return raw
    return default if default in _TURN_OBJECTIVE_QUESTION_SHAPES else "none"


def _normalize_ct_operation_goal(value: Any, *, default: str = "Strengthen") -> str:
    raw = str(value or "").strip()
    if not raw:
        default_clean = str(default or "").strip()
        if default_clean in _CT_OPERATION_GOALS:
            return default_clean
        return default_clean or "Strengthen"
    lowered = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "elicit": "Elicit",
        "drawout": "Elicit",
        "strengthen": "Strengthen",
        "reinforce": "Strengthen",
        "deepen": "Deepen",
        "clarify": "Deepen",
        "shift": "Shift",
        "transition": "Shift",
    }
    normalized = aliases.get(lowered)
    if normalized in _CT_OPERATION_GOALS:
        return normalized
    default_clean = str(default or "").strip()
    if default_clean in _CT_OPERATION_GOALS:
        return default_clean
    return default_clean or "Strengthen"


def _default_ct_operation_goal_for_action(action: MainAction) -> str:
    if _is_reflect_action(action):
        return "Strengthen"
    if action in {MainAction.QUESTION, MainAction.SCALING_QUESTION, MainAction.CLARIFY_PREFERENCE}:
        return "Deepen"
    if action in {MainAction.ASK_PERMISSION_TO_SHARE_INFO, MainAction.PROVIDE_INFO, MainAction.SUMMARY}:
        return "Shift"
    return "Strengthen"


def _extract_ct_anchors_from_sources(
    *,
    selected_candidates: Sequence[ChangeTalkCandidate],
    change_talk_hint: str,
    max_items: int = 2,
) -> List[str]:
    anchors: List[str] = []
    for candidate in selected_candidates:
        topic = _sanitize_human_text(candidate.normalized_text)
        if not topic:
            continue
        if topic in anchors:
            continue
        anchors.append(topic)
        if len(anchors) >= max(1, int(max_items)):
            return anchors
    for term in _extract_change_talk_focus_terms(change_talk_hint, max_terms=max(2, max_items * 2)):
        cleaned = _sanitize_human_text(term)
        if not cleaned:
            continue
        if cleaned in anchors:
            continue
        anchors.append(cleaned)
        if len(anchors) >= max(1, int(max_items)):
            break
    return anchors


def _derive_turn_objective_from_slot_quality_target_examples(
    *,
    slot_quality_target_examples: Sequence[Mapping[str, Any]],
    slot_target: str,
    action: MainAction,
    phase: Phase,
) -> Dict[str, Any]:
    question_allowed = _default_question_count_max(action) > 0
    default_use_reflection_first = _is_reflect_action(action) or action == MainAction.SUMMARY
    slot_focus_label = _slot_label_from_target(slot_target) or "this topic"
    default_goal = (
        f"Check one easy-to-answer point about {slot_focus_label}"
        if question_allowed
        else f"Make the meaning of {slot_focus_label} more concrete through a checking reflection"
    )
    default_success = (
        f"One concrete piece of information about {slot_focus_label} comes back"
        if question_allowed
        else f"The talk about {slot_focus_label} becomes slightly more concrete"
    )
    default_shape = _default_question_shape_for_action(action)

    primary_example = _select_slot_quality_target_example(slot_quality_target_examples)
    if primary_example is None:
        return {
            "repair_goal_text": default_goal,
            "success_criterion": default_success,
            "question_shape": default_shape,
            "use_reflection_first": default_use_reflection_first,
        }

    slot_key = _slot_key_from_focus_text(primary_example.get("slot_key") or primary_example.get("slot_label"))
    slot_label = _sanitize_human_text(primary_example.get("slot_label")) or _sanitize_human_text(
        _PHASE_SLOT_LABELS.get(slot_key, "")
    ) or slot_focus_label
    issue_type = _sanitize_human_text(primary_example.get("issue_type"))
    if issue_type not in _SLOT_REPAIR_PROBE_STYLE:
        issue_type = "too_vague"
    target_information = _extract_slot_quality_target_information(primary_example)

    repair_goal_text = default_goal
    success_criterion = default_success
    question_shape = default_shape
    use_reflection_first = default_use_reflection_first

    if target_information:
        if question_allowed:
            repair_goal_text = f"Check one point about {target_information}"
            success_criterion = f"One new concrete piece of information about {target_information} comes back"
        else:
            repair_goal_text = f"Work with one point about the meaning of {target_information}"
            success_criterion = f"The talk about {target_information} becomes slightly more concrete"
        if issue_type == "missing_evidence":
            question_shape = "example_probe" if question_allowed else "reflective_check"
        elif issue_type == "low_confidence":
            question_shape = "gentle_clarification" if question_allowed else "reflective_check"
            use_reflection_first = True
            success_criterion = "A yes/no correction or short rewording comes back"
        elif issue_type == "wrong_format":
            if slot_key in _SLOT_SCALE_KEYS:
                repair_goal_text = f"Check {target_information} in the required format such as 0-10"
                success_criterion = "A number from 0 to 10 comes back"
                question_shape = "scale_question"
            else:
                repair_goal_text = f"Check {target_information} in an easy-to-answer format"
                success_criterion = f"The information needed for {target_information} comes back in the required format"
                question_shape = _default_question_shape_for_action(action)
        else:
            question_shape = _default_question_shape_for_action(action)
    else:
        if issue_type == "missing_evidence":
            repair_goal_text = f"Ask for one recent instance or concrete example about {slot_label} to add evidence"
            success_criterion = f"One concrete example related to {slot_label} comes back"
            question_shape = "example_probe"
        elif issue_type == "low_confidence":
            repair_goal_text = f"Offer a tentative understanding of {slot_label} and briefly check whether it fits"
            success_criterion = "A yes/no correction or short rewording comes back"
            question_shape = "gentle_clarification"
            use_reflection_first = True
        elif issue_type == "wrong_format":
            if slot_key in _SLOT_SCALE_KEYS:
                repair_goal_text = f"Check {slot_label} in the required format such as 0-10"
                success_criterion = "A number from 0 to 10 comes back"
                question_shape = "scale_question"
            else:
                repair_goal_text = f"Ask again about {slot_label} in an easy-to-answer format"
                success_criterion = f"The information needed for {slot_label} comes back in the required format"
                question_shape = _default_question_shape_for_action(action)
        else:
            if slot_key == "problem_scene":
                repair_goal_text = "Ask about one recent concrete scene"
                success_criterion = "One recent scene comes back, including when, where, and what happened"
            elif slot_key == "execution_context":
                repair_goal_text = "Make the implementation context concrete, including when or where it happens"
                success_criterion = "One time window or location for carrying it out is made explicit"
            elif phase == Phase.PURPOSE_CONFIRMATION:
                if slot_key == "today_focus_topic":
                    repair_goal_text = "Narrow today's focus to one thing to work on here"
                    success_criterion = "One topic for today becomes clear"
                elif slot_key == "process_need":
                    repair_goal_text = "Briefly check what speaking style or process preference would help"
                    success_criterion = "A short preference about pace, style, or process becomes clear"
                elif slot_key == "presenting_problem_raw":
                    repair_goal_text = "Confirm the original concern in the client's own words"
                    success_criterion = "The initial presenting concern can be stated in one sentence"
                else:
                    repair_goal_text = "Narrow today's focus to one thing to work on here"
                    success_criterion = "One topic for today becomes clear"
            question_shape = _default_question_shape_for_action(action)

    if not question_allowed:
        question_shape = "reflective_check" if use_reflection_first else "none"

    return {
        "repair_goal_text": _sanitize_human_text(repair_goal_text),
        "success_criterion": _sanitize_human_text(success_criterion),
        "question_shape": _normalize_turn_objective_question_shape(question_shape, default=default_shape),
        "use_reflection_first": bool(use_reflection_first),
    }


def _derive_turn_objective_from_slot_repair(
    *,
    slot_repair_hints: Sequence[Mapping[str, Any]],
    slot_target: str,
    action: MainAction,
    phase: Phase,
) -> Dict[str, Any]:
    return _derive_turn_objective_from_slot_quality_target_examples(
        slot_quality_target_examples=slot_repair_hints,
        slot_target=slot_target,
        action=action,
        phase=phase,
    )


def _derive_utterance_target(
    *,
    action: MainAction,
    question_required: bool,
    primary_focus_topic: str,
    slot_target: str,
    missing_info_candidate: str,
    repair_goal_text: str,
    selected_slot_quality_target_example: Optional[Mapping[str, Any]] = None,
    change_talk_hint: str = "",
    ct_operation_goal: str = "Strengthen",
    clarify_preference_mode: str = "default",
    focus_choice_options: Sequence[str] = (),
) -> str:
    topic = _sanitize_human_text(primary_focus_topic) or "the current topic"
    slot_label = _slot_label_from_target(slot_target) or _sanitize_human_text(missing_info_candidate)
    target_information = _extract_slot_quality_target_information(selected_slot_quality_target_example)
    operation_goal = _normalize_ct_operation_goal(ct_operation_goal, default="Strengthen")
    change_talk_items = _parse_change_talk_items(change_talk_hint, max_items=2)
    change_talk_focus = ""
    if change_talk_items:
        if len(change_talk_items) >= 2 and operation_goal in {"Deepen", "Shift"}:
            change_talk_focus = _sanitize_human_text(f"{change_talk_items[0]} and {change_talk_items[1]}")
        else:
            change_talk_focus = _sanitize_human_text(change_talk_items[0])

    def _experience_probe_from_focus() -> str:
        if not change_talk_focus:
            return ""
        if operation_goal == "Elicit":
            return _sanitize_human_text(f"Check one scene that opens a path toward {change_talk_focus}")
        if operation_goal == "Shift":
            return _sanitize_human_text(f"Check one trigger that nudges things toward {change_talk_focus}")
        return _sanitize_human_text(f"Check one scene or cue that connects to {change_talk_focus}")

    def _experience_reflect_from_focus() -> str:
        if not change_talk_focus:
            return ""
        if operation_goal == "Shift":
            return _sanitize_human_text(f"Reflect one moment when things start leaning toward {change_talk_focus}")
        return _sanitize_human_text(f"Reflect one moment or felt sense where {change_talk_focus} showed up")

    if target_information:
        if question_required:
            return _sanitize_human_text(f"Check one point about {target_information}")
        return _sanitize_human_text(f"Reflect one point about {target_information}")
    if (
        action == MainAction.CLARIFY_PREFERENCE
        and clarify_preference_mode == "focus_choice"
        and len([item for item in focus_choice_options if _sanitize_human_text(item)]) >= 2
    ):
        return "Check which available entry point feels easiest to start from"
    if question_required:
        focus_probe = _experience_probe_from_focus()
        if focus_probe:
            return focus_probe
        anchor = slot_label or topic
        return _sanitize_human_text(f"Check one point about {anchor}")
    goal = _sanitize_human_text(repair_goal_text)
    focus_reflect = _experience_reflect_from_focus()
    if focus_reflect:
        return focus_reflect
    if goal:
        return _sanitize_human_text(f"Reflect one point where {goal} becomes visible")
    return _sanitize_human_text(f"Reflect one point where {topic} becomes visible")


def _derive_question_reflect_seed(
    *,
    action: MainAction,
    add_affirm: AffirmationMode | bool | str,
    change_talk_hint: str = "",
    ct_anchors: Sequence[str] = (),
) -> str:
    affirm_mode = _normalize_affirmation_mode(add_affirm)
    if action not in {MainAction.QUESTION, MainAction.ASK_PERMISSION_TO_SHARE_INFO}:
        return ""
    if affirm_mode != AffirmationMode.NONE:
        return ""

    for item in _parse_change_talk_items(change_talk_hint, max_items=3):
        text = _sanitize_human_text(item)
        if text:
            return text

    for item in ct_anchors:
        text = _sanitize_human_text(item)
        if text:
            return text

    return ""


def _derive_evocation_move(
    *,
    action: MainAction,
    add_affirm: AffirmationMode | bool | str,
    reflection_style: Optional[ReflectionStyle],
    primary_focus_topic: str,
    slot_target: str,
    missing_info_candidate: str,
    repair_goal_text: str,
    selected_slot_quality_target_example: Optional[Mapping[str, Any]] = None,
    change_talk_hint: str = "",
    ct_operation_goal: str = "Strengthen",
    clarify_preference_mode: str = "default",
    focus_choice_options: Sequence[str] = (),
) -> str:
    affirm_mode = _normalize_affirmation_mode(add_affirm)
    operation_goal = _normalize_ct_operation_goal(ct_operation_goal, default="Strengthen")
    focus_topic = _sanitize_human_text(primary_focus_topic)
    slot_label = _slot_label_from_target(slot_target) or _sanitize_human_text(missing_info_candidate)
    target_information = _extract_slot_quality_target_information(selected_slot_quality_target_example)
    change_talk_items = _parse_change_talk_items(change_talk_hint, max_items=2)
    focus_label = _sanitize_human_text(change_talk_items[0]) if change_talk_items else (target_information or slot_label or focus_topic or "the change side")

    move = ""
    if action == MainAction.REFLECT_DOUBLE or reflection_style == ReflectionStyle.DOUBLE_SIDED:
        move = (
            f"Use a double-sided reflection: briefly acknowledge the side that slows {focus_label} in the first half, "
            f"then end on the side moving toward {focus_label} in line with utterance_target"
        )
    elif _is_reflect_action(action):
        if action == MainAction.REFLECT_SIMPLE or reflection_style == ReflectionStyle.SIMPLE:
            move = f"Use a concise reflection that briefly mirrors only one point about {focus_label}"
        else:
            if operation_goal == "Deepen":
                move = f"Use a complex reflection to show, by half a step, the meaning of choosing what protects {focus_label}"
            elif operation_goal == "Shift":
                move = f"Use a complex reflection to show, by half a step, the footing that could move toward {focus_label}"
            else:
                move = f"Use a complex reflection to show, by half a step, the meaning of moving toward {focus_label}"
    elif action == MainAction.QUESTION:
        if affirm_mode == AffirmationMode.NONE:
            move = (
                f"Start with a simple reflection that acknowledges one point about {focus_label} selected from change_talk_inferer, "
                "then ask exactly one question about utterance_target"
            )
        else:
            move = f"Place one short affirmation first, then ask exactly one utterance_target question about {focus_label}"
    elif action == MainAction.SCALING_QUESTION:
        move = f"Use a scaling question to check {slot_label or focus_label} with only a number"
    elif action == MainAction.SUMMARY:
        move = f"Use a summary that gathers {focus_label} while briefly keeping the ambivalence or background in view"
    elif action == MainAction.CLARIFY_PREFERENCE:
        if (
            clarify_preference_mode == "focus_choice"
            and len([item for item in focus_choice_options if _sanitize_human_text(item)]) >= 2
        ):
            focus_choice_label = _join_writer_items(list(focus_choice_options), fallback="the visible entry points")
            move = f"Briefly reflect the ambivalence, then ask one question about which of {focus_choice_label} feels easiest to work with now"
        else:
            move = "Briefly reflect the ambivalence, then check with one binary question which process preference would help right now"
    elif action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        if affirm_mode == AffirmationMode.NONE:
            move = (
                f"Start with a simple reflection that acknowledges one point about {focus_label} selected from change_talk_inferer, "
                "then ask one question about whether information sharing is okay"
            )
        else:
            move = f"Place one short affirmation first, then ask one question about whether information sharing fits {focus_label}"
    elif action == MainAction.PROVIDE_INFO:
        move = "Share brief neutral information, then ask one question about the client's reaction"
    else:
        move = f"Keep the turn centered on {focus_label} and narrow it to one conversational function"

    if affirm_mode == AffirmationMode.SIMPLE:
        move = f"{move}. Add a short affirmation naturally."
    elif affirm_mode == AffirmationMode.COMPLEX:
        move = f"{move}. Add an affirmation grounded in observed evidence naturally."
    return _sanitize_human_text(move)


def _build_brief_must_include(
    *,
    primary_focus_topic: str,
    change_talk_hint: str,
    selected_slot_quality_target_example: Optional[Mapping[str, Any]],
) -> List[str]:
    items: List[str] = []

    def _push(value: Any) -> None:
        text = _sanitize_human_text(value)
        if not text or text in items:
            return
        items.append(text)

    target_information = _extract_slot_quality_target_information(selected_slot_quality_target_example)
    if target_information:
        _push(target_information)

    for item in _parse_change_talk_items(change_talk_hint, max_items=2):
        _push(item)

    _push(primary_focus_topic)
    return items[:3]


def _merge_priority_text_items(
    *,
    priority_items: Sequence[Any],
    fallback_items: Sequence[Any],
    max_items: int = 3,
) -> List[str]:
    merged: List[str] = []

    def _push(value: Any) -> None:
        text = _sanitize_human_text(value)
        if not text or text in merged:
            return
        merged.append(text)

    for item in priority_items:
        _push(item)
        if len(merged) >= max(1, int(max_items)):
            return merged
    for item in fallback_items:
        _push(item)
        if len(merged) >= max(1, int(max_items)):
            break
    return merged


def _build_minimum_draft_response_for_action(
    *,
    action: MainAction,
    focus_topic: str,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> str:
    topic = _sanitize_human_text(focus_topic) or "what you are describing right now"
    clarify_preference_mode, focus_choice_options = _resolve_clarify_preference_mode(
        action=action,
        focus_choice_context=focus_choice_context,
    )
    if action == MainAction.SCALING_QUESTION:
        return "Given what we have talked through so far, where would you place your confidence right now on a 0 to 10 scale?"
    if action == MainAction.QUESTION:
        return f"Could you say a little more concretely about {topic}?"
    if action == MainAction.CLARIFY_PREFERENCE:
        if clarify_preference_mode == "focus_choice" and len(focus_choice_options) >= 2:
            option_text = "、".join(focus_choice_options[:3])
            return f"I can see a few important entry points here. Which of {option_text} feels closest right now?"
        return "Would it help more right now to hear a possible option, or to keep sorting through the feeling first?"
    if action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        return "Would it be okay if I shared one option that might be useful?"
    if action == MainAction.PROVIDE_INFO:
        return "One option is to break the effort into smaller pieces. How does that land with you?"
    if action == MainAction.SUMMARY:
        return "So far, it sounds like both the wish to move forward and the hesitation are present, and that you want to work through this step by step in a way that feels right."
    return f"It sounds like you want to sort through {topic} little by little."


def _canonicalize_clause_ending(text: str) -> str:
    """
    Normalize clause endings for comparison.
    Use the final segment after stripping punctuation and closing brackets.
    """
    normalized = _normalize_slot_text(text)
    if not normalized:
        return ""
    tail = re.sub(r"[。．!?？！\s]+$", "", normalized)
    tail = re.sub(r"[」』】）》〉\)\]]+$", "", tail)
    if not tail:
        return ""
    for marker in ("it sounds like", "it seems", "that means", "that suggests", "that kind of feeling"):
        if tail.endswith(marker):
            return marker
    return tail[-6:] if len(tail) > 6 else tail


def _has_repetitive_reflect_ending(
    *,
    draft_response_text: str,
    history: Sequence[Tuple[str, str]],
    action: MainAction,
    lookback_assistant_turns: int = 4,
) -> bool:
    """
    Check whether the same reflection ending is being repeated in recent history.
    For now, focus mainly on repeated "it sounds like" endings.
    """
    if not (_is_reflect_action(action) or action == MainAction.SUMMARY):
        return False
    current_ending = _canonicalize_clause_ending(_last_output_clause(draft_response_text))
    if current_ending != "it sounds like":
        return False

    matched = 0
    seen_assistant_turns = 0
    for role, text in reversed(list(history)):
        if role != "assistant":
            continue
        ending = _canonicalize_clause_ending(_last_output_clause(text))
        if not ending:
            continue
        seen_assistant_turns += 1
        if ending == current_ending:
            matched += 1
        if seen_assistant_turns >= max(1, lookback_assistant_turns):
            break
    return matched >= 2


def _normalize_output_clause_tail(text: str) -> str:
    tail = _normalize_slot_text(_last_output_clause(text))
    if not tail:
        return ""
    tail = re.sub(r"[。．!?？！\s]+$", "", tail)
    tail = re.sub(r"[」』】）》〉\)\]]+$", "", tail)
    return tail


def _detect_reflect_ending_family(text: str) -> str:
    tail = _normalize_output_clause_tail(text)
    if not tail:
        return ""
    for family, patterns in _REFLECT_ENDING_FAMILY_PATTERNS:
        if any(pat.search(tail) for pat in patterns):
            return family
    if _is_reflection_ellipsis_ending(tail):
        return "elliptical_ending"
    return ""


def _analyze_reflect_ending_family_bias(
    *,
    text: str,
    history: Sequence[Tuple[str, str]],
    action: MainAction,
    lookback_assistant_turns: int = 4,
    repetition_threshold: int = 2,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "enabled": False,
        "warning": False,
        "current_family": "",
        "recent_families": [],
        "recent_family_counts": {},
        "avoid_recent_ending_families": [],
        "lookback_assistant_turns": max(1, int(lookback_assistant_turns)),
        "repetition_threshold": max(1, int(repetition_threshold)),
    }
    if not _is_reflect_action(action):
        return result

    result["enabled"] = True
    current_family = _detect_reflect_ending_family(text)
    result["current_family"] = current_family
    if not current_family:
        return result

    recent_families_reversed: List[str] = []
    seen_assistant_turns = 0
    for role, previous_text in reversed(list(history)):
        if role != "assistant":
            continue
        seen_assistant_turns += 1
        family = _detect_reflect_ending_family(previous_text)
        if family:
            recent_families_reversed.append(family)
        if seen_assistant_turns >= max(1, int(lookback_assistant_turns)):
            break
    recent_families = list(reversed(recent_families_reversed))
    result["recent_families"] = recent_families

    family_counts = Counter(recent_families)
    result["recent_family_counts"] = dict(family_counts)
    if family_counts.get(current_family, 0) < max(1, int(repetition_threshold)):
        return result

    avoid_recent_ending_families: List[str] = [current_family]
    for family, count in family_counts.most_common():
        if count < max(1, int(repetition_threshold)):
            continue
        if family in avoid_recent_ending_families:
            continue
        avoid_recent_ending_families.append(family)
        if len(avoid_recent_ending_families) >= 3:
            break
    result["warning"] = True
    result["avoid_recent_ending_families"] = avoid_recent_ending_families[:3]
    return result


def _apply_reflect_ending_family_guidance_to_brief(
    *,
    brief: ResponseBrief,
    guidance: Mapping[str, Any],
) -> None:
    if not isinstance(brief, ResponseBrief):
        return
    if not isinstance(guidance, Mapping) or not bool(guidance.get("warning")):
        return

    avoid_recent = guidance.get("avoid_recent_ending_families")
    if not isinstance(avoid_recent, Sequence) or isinstance(avoid_recent, (str, bytes)):
        return

    existing_plan = dict(brief.writer_plan) if isinstance(brief.writer_plan, Mapping) else {}
    existing_avoid = existing_plan.get("avoid_recent_ending_families")
    merged_avoid = _merge_priority_text_items(
        priority_items=avoid_recent,
        fallback_items=(
            existing_avoid
            if isinstance(existing_avoid, Sequence) and not isinstance(existing_avoid, (str, bytes))
            else []
        ),
        max_items=3,
    )
    if not merged_avoid:
        return
    existing_plan["avoid_recent_ending_families"] = merged_avoid
    brief.writer_plan = existing_plan


def _default_response_brief(
    *,
    state: DialogueState,
    action: MainAction,
    add_affirm: AffirmationMode,
    reflection_style: Optional[ReflectionStyle],
    risk_assessment: Optional[RiskAssessment],
    focus_candidates: Sequence[ChangeTalkCandidate],
    change_talk_hint: str,
    history: List[Tuple[str, str]],
    slot_target: str = "",
    slot_quality_target_examples: Optional[Sequence[Mapping[str, Any]]] = None,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> ResponseBrief:
    recent_user = ""
    for role, text in reversed(history):
        if role == "user":
            recent_user = _normalize_slot_text(text)
            if recent_user:
                break

    target_behavior_focus = _get_confirmed_target_behavior_for_change_talk(state=state)
    sorted_candidates = _select_change_talk_candidates_for_action(
        candidates=[
            candidate
            for candidate in focus_candidates
            if _normalize_slot_text(candidate.normalized_text)
        ],
        max_items=4,
        current_phase=state.phase,
        target_behavior_focus=target_behavior_focus,
        prioritize_target_behavior=_should_prioritize_target_behavior_for_phase(state.phase),
    )
    selected_primary = sorted_candidates[0] if sorted_candidates else None
    selected_secondary = sorted_candidates[1] if len(sorted_candidates) > 1 else None
    clarify_preference_mode, focus_choice_options = _resolve_clarify_preference_mode(
        action=action,
        focus_choice_context=focus_choice_context,
    )

    focus_terms = _extract_change_talk_focus_terms(change_talk_hint, max_terms=2)
    focus_topic = (
        _normalize_slot_text(selected_primary.normalized_text) if selected_primary is not None else ""
    ) or (focus_terms[0] if focus_terms else ("recent difficulty" if recent_user else "current theme"))
    macro_bridge_anchor = _build_macro_bridge_anchor(
        state=state,
        primary_focus_topic=focus_topic,
    )
    focus_meaning = (
        "an opening toward change in the latest utterance"
        if selected_primary is not None
        else (
            focus_terms[1]
            if len(focus_terms) > 1
            else (
                "both movement toward change and hesitation are visible"
                if change_talk_hint
                else "deepen the reflection in line with the latest talk"
            )
        )
    )
    writer_plan = {
        "opening_function": "reflection",
        "core_function": action.value,
        "closing_function": "no question" if _default_question_count_max(action) == 0 else "one question",
        "question_count_max": _default_question_count_max(action),
        "preferred_ending_family": "",
        "avoid_recent_ending_families": [],
    }
    if _is_closing_first_turn(state):
        writer_plan["opening_function"] = "shortReflection"
        writer_plan["core_function"] = "end check"
        writer_plan["closing_function"] = "one question checking whether to end"
        writer_plan["question_count_max"] = 1
    elif _is_review_reflection_first_turn(state):
        writer_plan["opening_function"] = "complexReflection"
        writer_plan["core_function"] = "sessionSummary"
        writer_plan["closing_function"] = "one review question"
        writer_plan["question_count_max"] = 1
    elif action == MainAction.QUESTION:
        writer_plan["opening_function"] = (
            "shortAffirmation" if add_affirm != AffirmationMode.NONE else "simpleReflection"
        )
        writer_plan["core_function"] = "question"
        writer_plan["closing_function"] = "one question about utterance_target"
    elif action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        writer_plan["opening_function"] = (
            "shortAffirmation" if add_affirm != AffirmationMode.NONE else "simpleReflection"
        )
        writer_plan["core_function"] = "permissionCheckForInformationSharing"
        writer_plan["closing_function"] = "one question checking whether information sharing is okay"
    slot_target_text = _slot_label_from_target(slot_target)
    normalized_slot_quality_target_examples = [
        {
            "slot_key": _slot_key_from_focus_text(item.get("slot_key")),
            "slot_label": _sanitize_human_text(item.get("slot_label")),
            "issue_type": _sanitize_human_text(item.get("issue_type")),
            "preferred_probe_style": _sanitize_human_text(item.get("preferred_probe_style")),
            "target_information": _extract_slot_quality_target_information(item),
            "detail": _extract_slot_quality_target_example_detail(item),
        }
        for item in (slot_quality_target_examples or [])
        if isinstance(item, Mapping)
    ]
    selected_slot_quality_target_example = _select_slot_quality_target_example(
        normalized_slot_quality_target_examples
    )
    turn_objective = _derive_turn_objective_from_slot_quality_target_examples(
        slot_quality_target_examples=normalized_slot_quality_target_examples,
        slot_target=slot_target,
        action=action,
        phase=state.phase,
    )
    language_constraints = {
        "style": "polite",
        "plain_language": True,
        "max_commas_per_sentence": 2,
        "target_sentence_length": "",
    }
    dialogue_digest: List[str] = []
    if recent_user:
        dialogue_digest.append(f"Latest client utterance: {recent_user}")
    dialogue_digest.append(f"Current phase: {state.phase.value}.")

    selected_for_anchor: List[ChangeTalkCandidate] = []
    if selected_primary is not None:
        selected_for_anchor.append(selected_primary)
    if selected_secondary is not None:
        selected_for_anchor.append(selected_secondary)
    ct_anchors = _extract_ct_anchors_from_sources(
        selected_candidates=selected_for_anchor,
        change_talk_hint=change_talk_hint,
        max_items=2,
    )
    ct_operation_goal = _default_ct_operation_goal_for_action(action)
    slot_goal = str(turn_objective.get("repair_goal_text") or "")
    phase_goal_this_turn = f"Take one step in line with the purpose of {state.phase.value}"
    if _is_closing_first_turn(state):
        phase_goal_this_turn = "Check whether it is okay to end today's session here and reach agreement on closing"
    elif _is_review_reflection_first_turn(state):
        phase_goal_this_turn = "Put into words what stands out most from today and what the client wants to keep in mind next"
    question_shape = str(turn_objective.get("question_shape") or "none")
    use_reflection_first = bool(turn_objective.get("use_reflection_first", False))
    question_required = _default_question_count_max(action) > 0
    if _is_closing_first_turn(state):
        question_shape = "open_question"
        use_reflection_first = True
        question_required = True
    elif _is_review_reflection_first_turn(state):
        question_required = True

    missing_info_candidate = slot_target_text if question_required else ""
    if question_required and not missing_info_candidate:
        missing_info_candidate = _sanitize_human_text(focus_topic)
    evocation_move = _derive_evocation_move(
        action=action,
        add_affirm=add_affirm,
        reflection_style=reflection_style,
        primary_focus_topic=focus_topic,
        slot_target=slot_target,
        missing_info_candidate=missing_info_candidate,
        repair_goal_text=str(turn_objective.get("repair_goal_text") or ""),
        selected_slot_quality_target_example=selected_slot_quality_target_example,
        change_talk_hint=change_talk_hint or " / ".join(ct_anchors),
        ct_operation_goal=ct_operation_goal,
        clarify_preference_mode=clarify_preference_mode,
        focus_choice_options=focus_choice_options,
    )

    draft_response_text = ""
    must_include = _build_brief_must_include(
        primary_focus_topic=focus_topic,
        change_talk_hint=change_talk_hint or " / ".join(ct_anchors),
        selected_slot_quality_target_example=selected_slot_quality_target_example,
    )
    if clarify_preference_mode == "focus_choice":
        must_include = _merge_priority_text_items(
            priority_items=focus_choice_options,
            fallback_items=must_include,
            max_items=3,
        )
    if action == MainAction.CLARIFY_PREFERENCE:
        draft_response_text = _build_minimum_draft_response_for_action(
            action=action,
            focus_topic=focus_topic,
            focus_choice_context=focus_choice_context,
        )

    brief = ResponseBrief(
        meta={
            "phase": state.phase.name,
            "main_action": action.value,
            "affirm_mode": add_affirm.value,
            "reflection_style": (reflection_style.value if reflection_style else "none"),
            "risk_mode": risk_assessment.level.value if risk_assessment else RiskLevel.NONE.value,
        },
        dialogue_digest=dialogue_digest[:4],
        primary_focus={
            "topic": focus_topic,
            "meaning": focus_meaning,
            "why_now": "highest-priority focus for this turn",
            "evidence_turn_ids": (
                [selected_primary.evidence_turn] if (selected_primary and selected_primary.evidence_turn) else ([len(history)] if history else [])
            ),
            "evidence_quotes": (
                [selected_primary.evidence_quote]
                if (selected_primary and selected_primary.evidence_quote)
                else ([recent_user] if recent_user else [])
            ),
        },
        utterance_target="",
        question_reflect_seed="",
        evocation_move=evocation_move,
        phase_goal_this_turn=phase_goal_this_turn,
        missing_info_candidate=missing_info_candidate,
        slot_quality_target_examples=normalized_slot_quality_target_examples,
        repair_goal_text=str(turn_objective.get("repair_goal_text") or ""),
        success_criterion=str(turn_objective.get("success_criterion") or ""),
        ct_anchors=ct_anchors[:2],
        ct_operation_goal=ct_operation_goal,
        slot_goal=slot_goal,
        draft_response_text=draft_response_text,
        question_shape=question_shape,
        use_reflection_first=use_reflection_first,
        writer_plan=writer_plan,
        must_include=must_include[:3],
        must_avoid=[
            "exposing internal labels or JSON keys",
            "presumptive or pushy phrasing",
            "offering suggestions without permission",
        ],
        language_constraints=language_constraints,
        brief_confidence=0.45,
        note_for_debug=(
            "fallback_brief_minimum_draft"
            if draft_response_text
            else "fallback_brief_missing_draft"
        ),
    )
    return brief


def _normalize_response_focus(
    value: Any,
    *,
    history_start_turn: int,
    history_end_turn: int,
) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    topic = _sanitize_human_text(raw.get("topic"))
    meaning = _sanitize_human_text(raw.get("meaning"))
    why_now = _sanitize_human_text(raw.get("why_now"))
    turn_ids: List[int] = []
    raw_turn_ids = raw.get("evidence_turn_ids")
    if isinstance(raw_turn_ids, Sequence) and not isinstance(raw_turn_ids, (str, bytes)):
        for item in raw_turn_ids:
            try:
                turn_id = int(item)
            except Exception:
                continue
            if turn_id < history_start_turn or turn_id > history_end_turn:
                continue
            if turn_id in turn_ids:
                continue
            turn_ids.append(turn_id)
            if len(turn_ids) >= 3:
                break

    quotes: List[str] = []
    raw_quotes = raw.get("evidence_quotes")
    if isinstance(raw_quotes, Sequence) and not isinstance(raw_quotes, (str, bytes)):
        for item in raw_quotes:
            text = _sanitize_human_text(item)
            if not text:
                continue
            if text in quotes:
                continue
            quotes.append(text)
            if len(quotes) >= 3:
                break
    elif isinstance(raw_quotes, str):
        text = _sanitize_human_text(raw_quotes)
        if text:
            quotes.append(text)

    return {
        "topic": topic,
        "meaning": meaning,
        "why_now": why_now,
        "evidence_turn_ids": turn_ids,
        "evidence_quotes": quotes,
    }


def _normalize_response_brief_payload(
    *,
    payload: Any,
    raw_output_text: str = "",
    state: DialogueState,
    action: MainAction,
    add_affirm: AffirmationMode,
    reflection_style: Optional[ReflectionStyle],
    risk_assessment: Optional[RiskAssessment],
    focus_candidates: Sequence[ChangeTalkCandidate],
    slot_target: str,
    slot_quality_target_examples: Sequence[Mapping[str, Any]],
    change_talk_hint: str,
    history: List[Tuple[str, str]],
    history_start_turn: int,
    history_end_turn: int,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[ResponseBrief, List[str]]:
    issues: List[str] = []

    def _add_issue(code: str) -> None:
        if code not in issues:
            issues.append(code)

    raw = payload if isinstance(payload, Mapping) else {}
    if not isinstance(payload, Mapping):
        _add_issue("payload_not_object")

    fallback = _default_response_brief(
        state=state,
        action=action,
        add_affirm=add_affirm,
        reflection_style=reflection_style,
        risk_assessment=risk_assessment,
        focus_candidates=focus_candidates,
        change_talk_hint=change_talk_hint,
        history=history,
        slot_target=slot_target,
        slot_quality_target_examples=slot_quality_target_examples,
        focus_choice_context=focus_choice_context,
    )
    clarify_preference_mode, focus_choice_options = _resolve_clarify_preference_mode(
        action=action,
        focus_choice_context=focus_choice_context,
    )

    digest: List[str] = []
    raw_digest = raw.get("dialogue_digest")
    if isinstance(raw_digest, Sequence) and not isinstance(raw_digest, (str, bytes)):
        for item in raw_digest:
            raw_text = _normalize_slot_text(item)
            if _find_internal_label_leak(raw_text):
                _add_issue("dialogue_digest_contains_internal_label")
                continue
            text = _sanitize_human_text(raw_text)
            if not text:
                continue
            if text in digest:
                continue
            digest.append(text)
            if len(digest) >= 4:
                break
    if not digest:
        _add_issue("dialogue_digest_missing")
        digest = [_sanitize_human_text(item) for item in fallback.dialogue_digest]
        digest = [item for item in digest if item]

    raw_primary_focus = raw.get("primary_focus")
    if _contains_internal_label_in_any(raw_primary_focus):
        _add_issue("primary_focus_contains_internal_label")
    primary_focus = _normalize_response_focus(
        raw_primary_focus,
        history_start_turn=history_start_turn,
        history_end_turn=history_end_turn,
    )
    if _response_focus_contains_internal_label(primary_focus):
        _add_issue("primary_focus_contains_internal_label")
        primary_focus = dict(fallback.primary_focus)
    if not primary_focus.get("topic") and not primary_focus.get("meaning"):
        _add_issue("primary_focus_missing")
        primary_focus = dict(fallback.primary_focus)
    macro_bridge_anchor = _build_macro_bridge_anchor(
        state=state,
        primary_focus_topic=str(primary_focus.get("topic") or ""),
    )

    phase_goal_raw = _normalize_slot_text(raw.get("phase_goal_this_turn"))
    if _find_internal_label_leak(phase_goal_raw):
        _add_issue("phase_goal_contains_internal_label")
        phase_goal_raw = ""
    phase_goal = _sanitize_human_text(phase_goal_raw) or fallback.phase_goal_this_turn
    if not phase_goal_raw:
        _add_issue("phase_goal_missing")

    missing_info_candidate_raw = _normalize_slot_text(raw.get("missing_info_candidate"))
    if _find_internal_label_leak(missing_info_candidate_raw):
        _add_issue("missing_info_candidate_contains_internal_label")
        missing_info_candidate_raw = ""
    missing_info_candidate = _sanitize_human_text(missing_info_candidate_raw)
    question_required = _default_question_count_max(action) > 0
    if _is_closing_first_turn(state):
        question_required = True
    slot_target_text = _slot_label_from_target(slot_target)
    if question_required and slot_target_text:
        missing_info_candidate = slot_target_text
    elif question_required and not missing_info_candidate:
        missing_info_candidate = (
            _sanitize_human_text(fallback.missing_info_candidate)
            or _sanitize_human_text(primary_focus.get("topic"))
        )
    if not question_required:
        missing_info_candidate = ""

    raw_slot_quality_target_examples = raw.get("slot_quality_target_examples")
    if raw_slot_quality_target_examples is None:
        raw_slot_quality_target_examples = raw.get("slot_repair_hints")
    slot_quality_target_examples: List[Dict[str, str]] = []
    seen_slot_quality_target_examples: set[Tuple[str, str]] = set()
    if isinstance(raw_slot_quality_target_examples, Sequence) and not isinstance(raw_slot_quality_target_examples, (str, bytes)):
        for item in raw_slot_quality_target_examples:
            if not isinstance(item, Mapping):
                continue
            slot_key = _slot_key_from_focus_text(item.get("slot_key") or item.get("slot_label"))
            if _find_internal_label_leak(slot_key):
                _add_issue("slot_quality_target_examples_contains_internal_label")
                continue
            slot_label = _sanitize_human_text(item.get("slot_label"))
            if not slot_label and slot_key:
                slot_label = _sanitize_human_text(_PHASE_SLOT_LABELS.get(slot_key, slot_key))
            if _find_internal_label_leak(slot_label):
                _add_issue("slot_quality_target_examples_contains_internal_label")
                continue
            issue_type = _sanitize_human_text(item.get("issue_type"))
            if issue_type not in _SLOT_REPAIR_PROBE_STYLE:
                issue_type = "too_vague"
            probe_style = (
                _sanitize_human_text(item.get("preferred_probe_style"))
                or _SLOT_REPAIR_PROBE_STYLE.get(issue_type, _SLOT_REPAIR_PROBE_STYLE["too_vague"])
            )
            detail = _extract_slot_quality_target_example_detail(item)
            target_information = _extract_slot_quality_target_information(item)
            hint = {
                "slot_key": slot_key,
                "slot_label": slot_label,
                "issue_type": issue_type,
                "preferred_probe_style": probe_style,
            }
            if target_information:
                hint["target_information"] = target_information
            if detail:
                hint["detail"] = detail
            dedup = (hint["slot_key"], hint["issue_type"])
            if not hint["slot_key"] or dedup in seen_slot_quality_target_examples:
                continue
            slot_quality_target_examples.append(hint)
            seen_slot_quality_target_examples.add(dedup)
            if len(slot_quality_target_examples) >= _MAX_SLOT_REPAIR_HINTS_PER_TURN:
                break
    if not slot_quality_target_examples:
        slot_quality_target_examples = [
            dict(item) for item in fallback.slot_quality_target_examples if isinstance(item, Mapping)
        ]
    if len(slot_quality_target_examples) > _MAX_SLOT_REPAIR_HINTS_PER_TURN:
        slot_quality_target_examples = slot_quality_target_examples[:_MAX_SLOT_REPAIR_HINTS_PER_TURN]
    selected_slot_quality_target_example = _select_slot_quality_target_example(
        slot_quality_target_examples
    )
    derived_turn_objective = _derive_turn_objective_from_slot_quality_target_examples(
        slot_quality_target_examples=slot_quality_target_examples,
        slot_target=slot_target,
        action=action,
        phase=state.phase,
    )

    raw_repair_goal_text = _normalize_slot_text(raw.get("repair_goal_text"))
    if _find_internal_label_leak(raw_repair_goal_text):
        _add_issue("repair_goal_contains_internal_label")
        raw_repair_goal_text = ""
    repair_goal_text = (
        _sanitize_human_text(raw_repair_goal_text)
        or _sanitize_human_text(fallback.repair_goal_text)
        or _sanitize_human_text(derived_turn_objective.get("repair_goal_text"))
    )

    raw_success_criterion = _normalize_slot_text(raw.get("success_criterion"))
    if _find_internal_label_leak(raw_success_criterion):
        _add_issue("success_criterion_contains_internal_label")
        raw_success_criterion = ""
    success_criterion = (
        _sanitize_human_text(raw_success_criterion)
        or _sanitize_human_text(fallback.success_criterion)
        or _sanitize_human_text(derived_turn_objective.get("success_criterion"))
    )

    raw_ct_anchors = raw.get("ct_anchors")
    ct_anchors: List[str] = []
    ct_anchors_from_payload = 0
    if isinstance(raw_ct_anchors, Sequence) and not isinstance(raw_ct_anchors, (str, bytes)):
        for item in raw_ct_anchors:
            text = _sanitize_human_text(item)
            if not text:
                continue
            if _find_internal_label_leak(text):
                _add_issue("ct_anchors_contains_internal_label")
                continue
            if text in ct_anchors:
                continue
            ct_anchors.append(text)
            if len(ct_anchors) >= 2:
                break
    elif isinstance(raw_ct_anchors, str):
        for item in re.split(r"[、,/|・]", raw_ct_anchors):
            text = _sanitize_human_text(item)
            if not text:
                continue
            if _find_internal_label_leak(text):
                _add_issue("ct_anchors_contains_internal_label")
                continue
            if text in ct_anchors:
                continue
            ct_anchors.append(text)
            if len(ct_anchors) >= 2:
                break
    ct_anchors_from_payload = len(ct_anchors)
    if ct_anchors_from_payload == 0:
        _add_issue("ct_anchors_missing")
    if not ct_anchors:
        ct_anchors = [_sanitize_human_text(item) for item in fallback.ct_anchors if _sanitize_human_text(item)]
    if len(ct_anchors) > 2:
        ct_anchors = ct_anchors[:2]

    raw_ct_operation_goal = str(raw.get("ct_operation_goal") or "").strip()
    ct_operation_goal = _normalize_ct_operation_goal(
        raw_ct_operation_goal,
        default=_normalize_ct_operation_goal(
            fallback.ct_operation_goal,
            default=_default_ct_operation_goal_for_action(action),
        ),
    )
    if not raw_ct_operation_goal:
        _add_issue("ct_operation_goal_missing")
    normalized_raw_ct_goal = _normalize_ct_operation_goal(raw_ct_operation_goal, default="__INVALID__")
    if raw_ct_operation_goal and normalized_raw_ct_goal == "__INVALID__":
        _add_issue("ct_operation_goal_invalid")

    slot_goal_raw = _normalize_slot_text(raw.get("slot_goal"))
    if _find_internal_label_leak(slot_goal_raw):
        _add_issue("slot_goal_contains_internal_label")
        slot_goal_raw = ""
    slot_goal = (
        _sanitize_human_text(slot_goal_raw)
        or _sanitize_human_text(repair_goal_text)
        or _sanitize_human_text(fallback.slot_goal)
    )
    if not _sanitize_human_text(slot_goal_raw):
        _add_issue("slot_goal_missing")

    raw_utterance_target = _normalize_slot_text(raw.get("utterance_target"))
    if _find_internal_label_leak(raw_utterance_target):
        _add_issue("utterance_target_contains_internal_label")
        raw_utterance_target = ""
    utterance_target = _sanitize_human_text(raw_utterance_target)
    if not raw_utterance_target:
        _add_issue("utterance_target_missing")

    raw_question_reflect_seed = _normalize_slot_text(raw.get("question_reflect_seed"))
    if _find_internal_label_leak(raw_question_reflect_seed):
        _add_issue("question_reflect_seed_contains_internal_label")
        raw_question_reflect_seed = ""
    question_reflect_seed = _sanitize_human_text(raw_question_reflect_seed)
    if (
        action in {MainAction.QUESTION, MainAction.ASK_PERMISSION_TO_SHARE_INFO}
        and add_affirm == AffirmationMode.NONE
        and not question_reflect_seed
    ):
        _add_issue("question_reflect_seed_missing")

    raw_evocation_move = _normalize_slot_text(raw.get("evocation_move"))
    if _find_internal_label_leak(raw_evocation_move):
        _add_issue("evocation_move_contains_internal_label")
        raw_evocation_move = ""
    evocation_move = (
        _sanitize_human_text(raw_evocation_move)
        or _sanitize_human_text(fallback.evocation_move)
        or _derive_evocation_move(
            action=action,
            add_affirm=add_affirm,
            reflection_style=reflection_style,
            primary_focus_topic=str(primary_focus.get("topic") or ""),
            slot_target=slot_target,
            missing_info_candidate=missing_info_candidate,
            repair_goal_text=repair_goal_text,
            selected_slot_quality_target_example=selected_slot_quality_target_example,
            change_talk_hint=" / ".join(ct_anchors),
            ct_operation_goal=ct_operation_goal,
            clarify_preference_mode=clarify_preference_mode,
            focus_choice_options=focus_choice_options,
        )
    )

    default_question_shape = (
        _normalize_turn_objective_question_shape(
            fallback.question_shape,
            default=str(derived_turn_objective.get("question_shape") or "none"),
        )
        if fallback.question_shape
        else str(derived_turn_objective.get("question_shape") or "none")
    )
    question_shape = _normalize_turn_objective_question_shape(
        raw.get("question_shape"),
        default=default_question_shape,
    )
    raw_question_shape = str(raw.get("question_shape") or "").strip().lower().replace(" ", "_").replace("-", "_")
    if raw_question_shape and raw_question_shape not in _TURN_OBJECTIVE_QUESTION_SHAPES:
        _add_issue("question_shape_invalid")

    raw_use_reflection_first = raw.get("use_reflection_first")
    if isinstance(raw_use_reflection_first, bool):
        use_reflection_first = bool(raw_use_reflection_first)
    elif isinstance(raw_use_reflection_first, str):
        lowered = _normalize_slot_text(raw_use_reflection_first).lower()
        if lowered in {"true", "1", "yes", "y"}:
            use_reflection_first = True
        elif lowered in {"false", "0", "no", "n"}:
            use_reflection_first = False
        else:
            use_reflection_first = bool(
                fallback.use_reflection_first
                or bool(derived_turn_objective.get("use_reflection_first", False))
            )
            _add_issue("use_reflection_first_invalid")
    else:
        use_reflection_first = bool(
            fallback.use_reflection_first
            or bool(derived_turn_objective.get("use_reflection_first", False))
        )

    if not question_required and question_shape not in {"none", "reflective_check"}:
        _add_issue("question_shape_inconsistent_with_action")
        question_shape = "reflective_check" if use_reflection_first else "none"
    if question_required and question_shape == "none":
        _add_issue("question_shape_inconsistent_with_action")
        question_shape = _default_question_shape_for_action(action)

    raw_plan = raw.get("writer_plan") if isinstance(raw.get("writer_plan"), Mapping) else {}
    raw_avoid_recent_ending_families = raw_plan.get("avoid_recent_ending_families")
    avoid_recent_ending_families: List[str] = []
    if isinstance(raw_avoid_recent_ending_families, Sequence) and not isinstance(
        raw_avoid_recent_ending_families, (str, bytes)
    ):
        for item in raw_avoid_recent_ending_families:
            if _find_internal_label_leak(item):
                _add_issue("writer_plan_avoid_recent_ending_families_contains_internal_label")
            text = _sanitize_human_text(item)
            if not text:
                continue
            if text in avoid_recent_ending_families:
                continue
            avoid_recent_ending_families.append(text)
            if len(avoid_recent_ending_families) >= 3:
                break
    writer_plan = {
        "opening_function": _sanitize_human_text(raw_plan.get("opening_function")) or fallback.writer_plan.get("opening_function"),
        "core_function": _sanitize_human_text(raw_plan.get("core_function")) or action.value,
        "closing_function": _sanitize_human_text(raw_plan.get("closing_function")) or fallback.writer_plan.get("closing_function"),
        "question_count_max": max(
            0,
            min(
                1,
                int(raw_plan.get("question_count_max"))
                if str(raw_plan.get("question_count_max", "")).strip().isdigit()
                else int(fallback.writer_plan.get("question_count_max", _default_question_count_max(action))),
            ),
        ),
        "preferred_ending_family": (
            _sanitize_human_text(raw_plan.get("preferred_ending_family"))
            or _sanitize_human_text(fallback.writer_plan.get("preferred_ending_family"))
            or ""
        ),
        "avoid_recent_ending_families": avoid_recent_ending_families
        or [
            _sanitize_human_text(item)
            for item in (fallback.writer_plan.get("avoid_recent_ending_families") or [])
            if _sanitize_human_text(item)
        ][:3],
    }
    for plan_key in ("opening_function", "core_function", "closing_function"):
        if _find_internal_label_leak(raw_plan.get(plan_key, "")):
            _add_issue(f"writer_plan_{plan_key}_contains_internal_label")
            writer_plan[plan_key] = str(fallback.writer_plan.get(plan_key) or "")
    if _find_internal_label_leak(raw_plan.get("preferred_ending_family", "")):
        _add_issue("writer_plan_preferred_ending_family_contains_internal_label")
        writer_plan["preferred_ending_family"] = _sanitize_human_text(
            fallback.writer_plan.get("preferred_ending_family")
        ) or ""
    required_qmax = _default_question_count_max(action)
    if _is_closing_first_turn(state):
        required_qmax = 1
    if required_qmax == 0 and writer_plan["question_count_max"] != 0:
        _add_issue("question_count_max_inconsistent_with_action")
        writer_plan["question_count_max"] = 0
    if required_qmax == 1 and writer_plan["question_count_max"] < 1:
        writer_plan["question_count_max"] = 1
    if _is_closing_first_turn(state):
        writer_plan["opening_function"] = "shortReflection"
        writer_plan["core_function"] = "end check"
        writer_plan["closing_function"] = "one question checking whether to end"
        writer_plan["question_count_max"] = 1
        question_shape = "open_question"
        use_reflection_first = True
        if not any(marker in phase_goal.lower() for marker in ("end", "close", "finish")):
            phase_goal = "Check whether it is okay to end today's session here"
    elif _is_review_reflection_first_turn(state):
        writer_plan["opening_function"] = "complexReflection"
        writer_plan["core_function"] = "sessionSummary"
        writer_plan["closing_function"] = "one review question"
        writer_plan["question_count_max"] = 1
        question_shape = "open_question"
        use_reflection_first = True
        if not any(marker in phase_goal.lower() for marker in ("notice", "learn", "review", "reflect")):
            phase_goal = "Put into words what stands out most from today and what the client wants to keep in mind next"

    must_include: List[str] = []
    raw_include = raw.get("must_include")
    if isinstance(raw_include, Sequence) and not isinstance(raw_include, (str, bytes)):
        for item in raw_include:
            raw_text = _normalize_slot_text(item)
            if _find_internal_label_leak(raw_text):
                _add_issue("must_include_contains_internal_label")
                continue
            text = _sanitize_human_text(raw_text)
            if not text:
                continue
            if text in must_include:
                continue
            must_include.append(text)
            if len(must_include) >= 3:
                break
    if not must_include:
        must_include = [_sanitize_human_text(item) for item in fallback.must_include]
        must_include = [item for item in must_include if item]
    if not must_include:
        must_include = _build_brief_must_include(
            primary_focus_topic=str(primary_focus.get("topic") or ""),
            change_talk_hint=" / ".join(ct_anchors),
            selected_slot_quality_target_example=selected_slot_quality_target_example,
        )
    if clarify_preference_mode == "focus_choice":
        must_include = _merge_priority_text_items(
            priority_items=focus_choice_options,
            fallback_items=must_include,
            max_items=3,
        )

    must_avoid: List[str] = []
    raw_avoid = raw.get("must_avoid")
    if isinstance(raw_avoid, Sequence) and not isinstance(raw_avoid, (str, bytes)):
        for item in raw_avoid:
            raw_text = _normalize_slot_text(item)
            if _find_internal_label_leak(raw_text):
                _add_issue("must_avoid_contains_internal_label")
                continue
            text = _sanitize_human_text(raw_text)
            if not text:
                continue
            if text in must_avoid:
                continue
            must_avoid.append(text)
            if len(must_avoid) >= 5:
                break
    if not must_avoid:
        must_avoid = [_sanitize_human_text(item) for item in fallback.must_avoid]
        must_avoid = [item for item in must_avoid if item]

    raw_lang = raw.get("language_constraints") if isinstance(raw.get("language_constraints"), Mapping) else {}
    language_constraints = {
        "style": _sanitize_human_text(raw_lang.get("style")) or fallback.language_constraints.get("style"),
        "plain_language": bool(raw_lang.get("plain_language", fallback.language_constraints.get("plain_language", True))),
        "max_commas_per_sentence": max(
            1,
            min(4, int(raw_lang.get("max_commas_per_sentence") or fallback.language_constraints.get("max_commas_per_sentence", 2))),
        ),
        "target_sentence_length": _sanitize_human_text(raw_lang.get("target_sentence_length"))
        or fallback.language_constraints.get("target_sentence_length"),
    }
    if _find_internal_label_leak(language_constraints.get("style", "")) or _find_internal_label_leak(
        language_constraints.get("target_sentence_length", "")
    ):
        _add_issue("language_constraints_contains_internal_label")
        language_constraints = dict(fallback.language_constraints)

    brief_confidence = _clamp01(raw.get("brief_confidence"), fallback.brief_confidence)
    note_for_debug = _sanitize_human_text(raw.get("note_for_debug")) or fallback.note_for_debug
    raw_draft_response_text = raw.get("draft_response_text")
    if raw_draft_response_text in (None, ""):
        recovered_from_payload, recovered_payload_path = _extract_text_from_payload_preferred_only(raw)
        if recovered_from_payload:
            raw_draft_response_text = recovered_from_payload
            _add_issue("draft_response_text_recovered_from_payload")
            if recovered_payload_path:
                note_for_debug = f"{note_for_debug} | recovered:{recovered_payload_path}".strip()
    if _contains_internal_label_in_any(raw_draft_response_text):
        _add_issue("draft_response_text_contains_internal_label")
    draft_response_text = _sanitize_human_draft_text(raw_draft_response_text)
    if not draft_response_text:
        recovered_from_raw, recover_debug = _recover_draft_response_text_from_raw_output(raw_output_text)
        if recovered_from_raw:
            draft_response_text = recovered_from_raw
            _add_issue("draft_response_text_recovered_from_raw_output")
            source_path = ""
            if isinstance(recover_debug, Mapping):
                source_path = str(recover_debug.get("source_path") or "")
            if source_path:
                note_for_debug = f"{note_for_debug} | recovered_raw:{source_path}".strip()
    if not draft_response_text:
        _add_issue("draft_response_text_missing")
        if action == MainAction.CLARIFY_PREFERENCE:
            draft_response_text = _build_minimum_draft_response_for_action(
                action=action,
                focus_topic=str(primary_focus.get("topic") or ""),
                focus_choice_context=focus_choice_context,
            )
        note_for_debug = f"{note_for_debug} | draft_missing".strip(" |") or "draft_missing"
    if _has_repetitive_reflect_ending(
        draft_response_text=draft_response_text,
        history=history,
        action=action,
    ):
        _add_issue("reflect_ending_repetition")

    brief = ResponseBrief(
        meta={
            "phase": state.phase.name,
            "main_action": action.value,
            "affirm_mode": add_affirm.value,
            "reflection_style": (reflection_style.value if reflection_style else "none"),
            "risk_mode": risk_assessment.level.value if risk_assessment else RiskLevel.NONE.value,
        },
        dialogue_digest=digest[:4],
        primary_focus=primary_focus,
        utterance_target=utterance_target,
        question_reflect_seed=question_reflect_seed,
        evocation_move=evocation_move,
        phase_goal_this_turn=phase_goal,
        missing_info_candidate=missing_info_candidate,
        slot_quality_target_examples=slot_quality_target_examples,
        repair_goal_text=repair_goal_text,
        success_criterion=success_criterion,
        ct_anchors=ct_anchors[:2],
        ct_operation_goal=ct_operation_goal,
        slot_goal=slot_goal,
        draft_response_text=draft_response_text,
        question_shape=question_shape,
        use_reflection_first=use_reflection_first,
        writer_plan=writer_plan,
        must_include=must_include[:3],
        must_avoid=must_avoid[:5],
        language_constraints=language_constraints,
        brief_confidence=brief_confidence,
        note_for_debug=note_for_debug,
    )
    return brief, issues


def build_prompt(
    *,
    history: List[Tuple[str, str]],
    state: DialogueState,
    action: MainAction,
    add_affirm: AffirmationMode | bool | str,
    reflection_style: Optional[ReflectionStyle] = None,
    risk_assessment: Optional[RiskAssessment] = None,
    first_turn_hint: Optional[FirstTurnHint] = None,
    allow_question_without_preface: bool = False,
    current_user_is_short: bool = False,
    phase_prediction_debug: Optional[Dict[str, Any]] = None,
    action_ranking_debug: Optional[Dict[str, Any]] = None,
    features: Optional[PlannerFeatures] = None,
    change_talk_inference: Optional[str] = None,
    focus_candidates: Optional[Sequence[ChangeTalkCandidate]] = None,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
    include_debug_block: bool = False,
) -> List[Dict[str, str]]:
    """
    history: [("user"|"assistant", text), ...]
    Returns messages in chat-completions-compatible format.
    """
    effective_change_talk_hint = (change_talk_inference or "").strip()
    focus_hint_from_contract = _build_change_talk_hint_from_candidates(
        candidates=list(focus_candidates or []),
        current_phase=state.phase,
    )
    if focus_hint_from_contract:
        effective_change_talk_hint = focus_hint_from_contract

    normalized = _normalize_prompt_inputs(
        history=history,
        state=state,
        action=action,
        add_affirm=add_affirm,
        reflection_style=reflection_style,
        risk_assessment=risk_assessment,
        first_turn_hint=first_turn_hint,
        phase_prediction_debug=phase_prediction_debug,
        action_ranking_debug=action_ranking_debug,
        features=features,
        change_talk_inference=effective_change_talk_hint,
    )

    action_rule = _render_action_rule(
        action=action,
        state=state,
        risk_assessment=risk_assessment,
        first_turn_hint=first_turn_hint,
        normalized=normalized,
        allow_question_without_preface=allow_question_without_preface,
        current_user_is_short=current_user_is_short,
        focus_choice_context=focus_choice_context,
    )

    fixed_sections: List[str] = [
        _render_role_and_style(),
        _render_safety_rules(risk_note=""),
    ]
    runtime_sections: List[str] = [
        _render_runtime_risk_note(normalized.risk_note),
        _render_hard_constraints(state=state, action=action, normalized=normalized),
        _render_phase_guidance(state.phase),
        _render_context_summary(normalized=normalized),
        _render_task_section(action_rule),
        _render_affirm_rule(normalized.affirm_mode),
        _render_output_format_constraints(),
    ]
    if include_debug_block:
        runtime_sections.append(_render_debug_block(normalized))

    fixed_system = _assemble_system(fixed_sections)
    knowledge_action_hint = (
        normalized.proposed_action_hint
        if normalized.proposed_action_hint != "none"
        else action.value
    )
    fixed_with_knowledge = inject_mi_knowledge(
        fixed_system,
        agent_name="response_generator",
        main_action=knowledge_action_hint,
    )
    system = _assemble_system_with_runtime_tail(
        fixed_sections=[fixed_with_knowledge],
        runtime_sections=runtime_sections,
    )
    system = (
        f"{system}\n\n"
        "Follow the guidance above, obey the fixed instructions for this turn (hard constraints), and produce a natural MI-counselor response."
    )
    return _build_messages(system=system, history=history)


def build_response_brief_messages(
    *,
    history: List[Tuple[str, str]],
    state: DialogueState,
    action: MainAction,
    add_affirm: AffirmationMode,
    reflection_style: Optional[ReflectionStyle],
    risk_assessment: Optional[RiskAssessment],
    first_turn_hint: Optional[FirstTurnHint],
    allow_question_without_preface: bool,
    current_user_is_short: bool,
    phase_prediction_debug: Optional[Dict[str, Any]],
    action_ranking_debug: Optional[Dict[str, Any]],
    features: Optional[PlannerFeatures],
    focus_candidates: Sequence[ChangeTalkCandidate],
    slot_target: str,
    change_talk_hint: Optional[str],
    focus_choice_context: Optional[Mapping[str, Any]] = None,
    max_history_turns: int = 120,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    slot_target_label = _slot_label_from_target(slot_target)
    slot_quality_target_examples = _slot_quality_target_examples_from_phase_debug(phase_prediction_debug)
    slot_quality_target_examples_text = (
        json.dumps(slot_quality_target_examples, ensure_ascii=False) if slot_quality_target_examples else "[]"
    )
    slot_quality_target_example_details = _format_slot_quality_target_example_details_for_prompt(
        slot_quality_target_examples,
        fallback="none",
    )
    selected_slot_quality_target_example = _select_slot_quality_target_example(slot_quality_target_examples)
    selected_slot_quality_target_example_text = (
        json.dumps(selected_slot_quality_target_example, ensure_ascii=False)
        if selected_slot_quality_target_example
        else "none"
    )
    selected_target_information_seed = (
        _extract_slot_quality_target_information(selected_slot_quality_target_example) or "none"
    )
    clarify_preference_mode, focus_choice_options = _resolve_clarify_preference_mode(
        action=action,
        focus_choice_context=focus_choice_context,
    )
    evocation_move_seed = (
        _derive_evocation_move(
            action=action,
            add_affirm=add_affirm,
            reflection_style=reflection_style,
            primary_focus_topic="",
            slot_target=slot_target,
            missing_info_candidate=slot_target_label,
            repair_goal_text="",
            selected_slot_quality_target_example=selected_slot_quality_target_example,
            change_talk_hint=change_talk_hint or "",
            ct_operation_goal=_default_ct_operation_goal_for_action(action),
            clarify_preference_mode=clarify_preference_mode,
            focus_choice_options=focus_choice_options,
        )
        or "none"
    )
    target_behavior_focus = _get_confirmed_target_behavior_for_change_talk(state=state)
    selected_focus_candidates = _select_change_talk_candidates_for_action(
        candidates=focus_candidates,
        max_items=2,
        current_phase=state.phase,
        target_behavior_focus=target_behavior_focus,
        prioritize_target_behavior=_should_prioritize_target_behavior_for_phase(state.phase),
        importance_estimate=(features.importance_estimate if isinstance(features, PlannerFeatures) else None),
    )
    focus_contract_hint = (
        (change_talk_hint or "").strip()
        or _build_change_talk_hint_from_candidates(
            candidates=selected_focus_candidates or focus_candidates,
            current_phase=state.phase,
        )
        or "none"
    )
    focus_contract_hint_text = _sanitize_human_text(focus_contract_hint) or "none"
    selected_focus_lines = _format_focus_candidates_for_prompt(
        selected_focus_candidates,
        max_items=2,
    )
    all_focus_lines = _format_focus_candidates_for_prompt(
        focus_candidates,
        max_items=4,
    )
    ct_anchor_seed_lines = selected_focus_lines if selected_focus_lines else all_focus_lines[:2]
    ct_anchor_seed_text = " / ".join(ct_anchor_seed_lines) if ct_anchor_seed_lines else "none"
    macro_bridge_anchor = _build_macro_bridge_anchor(
        state=state,
        primary_focus_topic=(selected_focus_lines[0] if selected_focus_lines else ""),
    )

    normalized = _normalize_prompt_inputs(
        history=history,
        state=state,
        action=action,
        add_affirm=add_affirm,
        reflection_style=reflection_style,
        risk_assessment=risk_assessment,
        first_turn_hint=first_turn_hint,
        phase_prediction_debug=phase_prediction_debug,
        action_ranking_debug=action_ranking_debug,
        features=features,
        change_talk_inference=focus_contract_hint,
    )
    action_rule = _render_action_rule(
        action=action,
        state=state,
        risk_assessment=risk_assessment,
        first_turn_hint=first_turn_hint,
        normalized=normalized,
        allow_question_without_preface=allow_question_without_preface,
        current_user_is_short=current_user_is_short,
        focus_choice_context=focus_choice_context,
    )
    dialogue, turn_start, turn_end = _history_to_numbered_dialogue(
        history,
        max_turns=max_history_turns,
    )
    schema = (
        "[Output schema: JSON only]\n"
        "{\n"
        '  "meta":{"phase":"...","main_action":"...","affirm_mode":"...","reflection_style":"...","risk_mode":"..."},\n'
        '  "dialogue_digest":["...","..."],\n'
        '  "primary_focus":{"topic":"...","meaning":"...","why_now":"...","evidence_turn_ids":[1],"evidence_quotes":["..."]},\n'
        '  "utterance_target":"...",\n'
        '  "question_reflect_seed":"...",\n'
        '  "evocation_move":"...",\n'
        '  "phase_goal_this_turn":"...",\n'
        '  "missing_info_candidate":"...",\n'
        '  "slot_quality_target_examples":[{"slot_key":"...","slot_label":"...","issue_type":"missing_evidence|low_confidence|wrong_format|too_vague","preferred_probe_style":"...","target_information":"...","detail":"..."}],\n'
        '  "ct_anchors":["...","..."],\n'
        '  "ct_operation_goal":"Elicit|Strengthen|Deepen|Shift",\n'
        '  "slot_goal":"...",\n'
        '  "repair_goal_text":"...",\n'
        '  "success_criterion":"...",\n'
        '  "question_shape":"none|open_question|gentle_clarification|scale_question|example_probe|scope_narrowing_question|reflective_check",\n'
        '  "use_reflection_first":true|false,\n'
        '  "writer_plan":{"opening_function":"...","core_function":"...","closing_function":"...","question_count_max":0|1,"preferred_ending_family":"...","avoid_recent_ending_families":["..."]},\n'
        '  "draft_response_text":"...",\n'
        '  "must_include":["..."],\n'
        '  "must_avoid":["..."],\n'
        '  "language_constraints":{"style":"polite","plain_language":true,"max_commas_per_sentence":2},\n'
        '  "brief_confidence":0.0,\n'
        '  "note_for_debug":"..."\n'
        "}\n"
        "- Keep focus to a single primary_focus item.\n"
        "- `utterance_target` should name only the one experiential proposition actually handled in this turn.\n"
        "- `utterance_target` is a planning label, not the response text itself. Do not make it a question, reflection, or summary sentence.\n"
        "- Do not put question marks, requests for explanation, agreement language, or counselor meta-language inside `utterance_target`.\n"
        "- Keep `utterance_target` close to something the client could experience or say, not writer-side intent such as flow or support.\n"
        "- Prefer short forms such as 'check one scene where ...' or 'reflect one moment when ... became visible'.\n"
        "- Use `question_reflect_seed` only for QUESTION / ASK_PERMISSION_TO_SHARE_INFO when affirm_mode=NONE. It should contain only the one change-talk point reflected in sentence 1.\n"
        "- `question_reflect_seed` must not be a question. Pick one item from change_talk_seed / ct_anchor_seed and keep the wording close to the source.\n"
        "- `evocation_move` must name only the main rhetorical move within fixed_main_action. Do not re-decide main_action.\n"
        "- For `slot_quality_target_examples.detail`, treat it as a hypothetical user utterance example and abstract it into target_information rather than copying it directly.\n"
        "- If `detail` is itself a question, agreement line, or counselor meta-direction, do not copy the sentence pattern; only keep the concrete content.\n"
        "- Do not output deferred_targets.\n"
        "- Limit `ct_anchors` to at most 2 human-readable items.\n"
        "- `slot_goal` should be a conversational goal sentence, not an internal key.\n"
        "- `missing_info_candidate` should only name the item needed for question-like actions.\n"
        "- evidence_turn_ids may contain only turn numbers from the input history.\n"
        "- In `must_include`, prioritize concrete anchors that should remain in the final text, such as time, place, amount, tools, steps, or implementation details.\n"
        "- Do not fill `must_include` only with abstractions. If a concrete term can stay, do not replace it with a generalized paraphrase.\n"
        "- If the client shows a priority contrast, implementation intent, or something important to protect, keep that change-core briefly in `must_include` as well.\n"
        "- Human-readable values (`utterance_target`, `evocation_move`, `ct_anchors`, `slot_goal`, `repair_goal_text`, `success_criterion`, `draft_response_text`, `must_include`, `must_avoid`, `note_for_debug`) should be written in English. JSON keys, enums, and phase codes stay unchanged.\n"
        "- Keep `evidence_quotes` as source quotations; do not rewrite them.\n"
        "- Do not write internal labels such as feature_extractor / phase_slots / change_talk_inference inside `must_avoid`.\n"
        "- `writer_plan.preferred_ending_family` is optional. Use a short human-readable label only when needed to avoid stiff endings.\n"
        "- `writer_plan.avoid_recent_ending_families` is optional. Use up to 2 human-readable family labels only when recent assistant reflections are ending too similarly.\n"
        "- `draft_response_text` should be a complete draft response that already contains the content worth preserving.\n"
        "- `draft_response_text` must implement `evocation_move`; do not stop at a slightly stronger paraphrase.\n"
        "- Treat `draft_response_text` as the upper anchor that Layer4 may trim, so front-load the necessary meaning and concrete anchors.\n"
        "- Write `draft_response_text` as natural English. Do not leave broken placeholders or stray ASCII fragments.\n"
        "- Preserve concrete anchors from selected_target_information_seed or slot_quality_target_examples.detail whenever possible.\n"
        "- Do not collapse concrete phrases such as '10-minute blocks', 'a cue', 'a one-week notebook', or 'quiet time in the morning' into abstractions like 'preparation' or 'approach'.\n"
        "- Output exactly one JSON object and nothing else."
    )
    if state.phase == Phase.REVIEW_REFLECTION:
        target_behavior_integration_rule = (
            "- In REVIEW_REFLECTION, refer to presenting_problem_raw / problem_scene / core_values / target_behavior with equal weight rather than over-focusing on just one.\n"
        )
    elif state.phase == Phase.CLOSING:
        target_behavior_integration_rule = (
            "- In CLOSING, keep presenting_problem_raw / problem_scene / core_values / target_behavior balanced while prioritizing the end-check structure.\n"
        )
    else:
        target_behavior_integration_rule = (
            "- If a target_behavior anchor exists, preserve consistency with it across all phases.\n"
        )
    integration_rules = (
        "[Layer3 core responsibilities]\n"
        "- Do not change phase / main_action / affirm_mode / reflection_style once Layer2 has fixed them.\n"
        "- Keep the turn to one focus and one experiential proposition in utterance_target.\n"
        "- `draft_response_text` must stay one-turn, one-proposition. Do not stack reflection, explanation, affirmation, and future orientation all at once.\n"
        "- Prefer primary_focus_seed when choosing the main focus; use only one fallback focus from focus_candidates if needed.\n"
        "- Use change_talk_seed / ct_anchor_seed only as change-side focus candidates. Do not make sustain talk or process-preference requests the main subject of the response.\n"
        "- If process-preference language appears, connect it to what it protects or what it makes slightly easier, rather than making the preference itself the star of the reflection.\n"
        "- If the client is already leaning toward one option, keep that side as the main focus and treat the unchosen side only as supporting context.\n"
        "- Human-facing text such as draft_response_text / utterance_target / evocation_move / must_include should not contain placeholders or broken ASCII fragments.\n"
        "- Keep concrete anchors by meaning, not only by literal surface form. Do not place placeholders or corrupted tokens into must_include.\n"
        "- Prioritize slot_target_label as the missing-information focus when relevant.\n"
        "- slot_quality_target_examples_json and selected_slot_quality_target_example_json may guide reflections and summaries, not only questions.\n"
        "- Build utterance_target from the single anchor that is closest to lived experience among change_talk_seed / selected_target_information_seed / slot_quality_target_example_detail_seed.\n"
        "- utterance_target should describe one client experience proposition, not writer-side intent such as flow or support.\n"
        "- evocation_move must specify only one main rhetorical move within fixed_main_action.\n"
        "- Apply question / reflection / summary form constraints to draft_response_text, but keep utterance_target itself non-interrogative.\n"
        "- When add_affirm == NONE for QUESTION / ASK_PERMISSION_TO_SHARE_INFO, question_reflect_seed should contain exactly one reflected item from change_talk_seed / ct_anchor_seed.\n"
        "- must_include should prioritize concrete anchors such as time, place, amount, tools, or steps, and should also preserve the core of change if there is a clear priority contrast or implementation intent.\n"
        "- Keep concrete anchors concrete in draft_response_text; do not abstract them away.\n"
        "- draft_response_text is the Layer4 upper anchor, so preserve needed content even if that makes it slightly longer.\n"
        "- draft_response_text must clearly enact evocation_move.\n"
        "- Limit stronger hypotheses to one at most. Do not pile interpretation, affirmation, and direction-setting into the same sentence.\n"
        "- In REFLECT_COMPLEX, keep to one explicit point plus at most one half-step implication, phrased reversibly rather than as a fixed conclusion.\n"
        "- If recent assistant reflections are ending too similarly, populate writer_plan.avoid_recent_ending_families and avoid repeating the same ending family.\n"
        "- If ambivalence is explicit in the same turn, keep the hesitant side briefly but end on the change side or the side already being chosen.\n"
        "- If the same change topic has already been reflected in the last two turns, shift the third reflection to one of value, reason, a successful moment, or the next sign of traction.\n"
        "- Avoid certainty, prediction, and over-storytelling. Prefer reversible language.\n"
        "- REFLECT_SIMPLE / REFLECT_COMPLEX should return to only one point in utterance_target.\n"
        "- REFLECT_DOUBLE should briefly hold the sustain-side first and end on the change-side that relates to utterance_target.\n"
        "- In QUESTION, prioritize evoking over information collection. Work with only one target: meaning, reason, what the client wants to protect, why the score is not zero, or what would raise it by one point.\n"
        "- Keep the default question structures fixed for QUESTION and ASK_PERMISSION_TO_SHARE_INFO.\n"
        "- If you add an affirmation, keep it grounded in observed speech, choices, or actions rather than abstract praise.\n"
        "- SUMMARY may be 2-3 sentences, but bundle change talk into at most two clusters and avoid repetitive recovery of the whole timeline.\n"
        "- Especially in NEXT_STEP_DECISION / FOCUSING_TARGET_BEHAVIOR, preserve either an observable behavior or a scene/time anchor.\n"
        "- Do not end draft_response_text on abstract nouns alone; land on a scene, sensation, choice, or action.\n"
        "- Poor utterance_target example: 'Reflect by half a step the intent to focus on what matters.'\n"
        "- Better utterance_target example: 'Reflect one moment in the certification study where the thought \"this might matter\" showed up.'\n"
        "- Poor draft_response_text example: 'Reduce the idea of trying 10-minute blocks and a cue to merely asking what preparation is needed.'\n"
        "- Better draft_response_text example: 'Keep concrete terms such as 10-minute blocks and a cue, and only smooth the sentence.'\n"
        "- Design the turn so that current_phase_required_slots / current_phase_slots_json move the current phase slightly forward.\n"
        f"{target_behavior_integration_rule}"
        "- When macro_bridge_anchor is present in IMPORTANCE_PROMOTION / CONFIDENCE_PROMOTION / REVIEW_REFLECTION / CLOSING, preserve that internal connection without filling must_include with abstract bridge phrases or adding new 'X is the entrance to Y' explanations.\n"
        f"- If there is a concrete wording change that would move quality toward {_SLOT_REPAIR_HINT_TARGET_QUALITY:.2f}, preserve that wording in at least one of repair_goal_text / success_criterion / slot_goal.\n"
        f"- Use quality>=0.80 as the confirmed-consistent anchor threshold, and quality={_SLOT_REPAIR_HINT_TARGET_QUALITY:.2f} as the repair-goal threshold.\n"
        "- Follow action-specific behavior from the action rule.\n"
        "- draft_response_text should be a natural response that could be sent as-is.\n"
        "- note_for_debug should keep only a short record of the judgment core.\n"
        "- Output exactly one JSON object.\n"
        "\n"
        "[Common workflow]\n"
        "1. Decide one primary_focus.\n"
        "2. Decide one utterance_target.\n"
        "3. Decide one evocation_move.\n"
        "4. Decide slot_goal / repair_goal_text / success_criterion in one sentence each.\n"
        "5. Build writer_plan / must_include / draft_response_text in line with the action rule.\n"
        "6. Set brief_confidence."
    )
    shared_inputs = _render_layer3_shared_inputs(
        state=state,
        action=action,
        add_affirm=add_affirm,
        reflection_style=reflection_style,
        normalized=normalized,
        selected_focus_lines=selected_focus_lines,
        ct_anchor_seed_text=ct_anchor_seed_text,
        ct_operation_goal=_default_ct_operation_goal_for_action(action),
        focus_contract_hint_text=focus_contract_hint_text,
        slot_target_label=slot_target_label,
        slot_quality_target_examples_text=slot_quality_target_examples_text,
        slot_quality_target_example_details=slot_quality_target_example_details,
        selected_slot_quality_target_example_text=selected_slot_quality_target_example_text,
        selected_target_information_seed=selected_target_information_seed,
        evocation_move_seed=evocation_move_seed,
        target_behavior_focus=target_behavior_focus,
        macro_bridge_anchor=macro_bridge_anchor,
        clarify_preference_mode=clarify_preference_mode,
        focus_choice_options=focus_choice_options,
    )
    fixed_system_sections = [
        "You are Layer3 (information integration) for an MI counselor.",
        _render_safety_rules(risk_note=""),
        integration_rules,
        schema,
    ]
    runtime_system_sections = [
        _render_runtime_risk_note(normalized.risk_note),
        _render_phase_guidance(state.phase),
        shared_inputs,
        f"{_render_task_section(action_rule)}\n\n{_render_affirm_rule(add_affirm)}",
    ]
    system = _assemble_system_with_runtime_tail(
        fixed_sections=fixed_system_sections,
        runtime_sections=runtime_system_sections,
    )
    system = inject_mi_knowledge(
        system,
        agent_name="response_integrator",
        main_action=action.value,
    )
    user = (
        f"[History: T{turn_start}-T{turn_end}]\n{dialogue}\n"
        "Integrate the information above and return exactly one JSON object."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, {
        "history_turn_start": turn_start,
        "history_turn_end": turn_end,
        "slot_quality_target_examples": slot_quality_target_examples,
        "slot_repair_hints": slot_quality_target_examples,
    }


def build_writer_messages(
    *,
    history: List[Tuple[str, str]],
    state: DialogueState,
    action: MainAction,
    add_affirm: AffirmationMode,
    reflection_style: Optional[ReflectionStyle],
    risk_assessment: Optional[RiskAssessment],
    first_turn_hint: Optional[FirstTurnHint],
    brief: ResponseBrief,
    layer4_repair_issue_codes: Optional[Mapping[str, Sequence[Any]]] = None,
    max_history_turns: int = 120,
    closing_phase_complete: bool = False,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    reflection_style_text = reflection_style.value if reflection_style else "none"
    first_turn_note = ""
    if state.phase == Phase.GREETING and state.turn_index == 0:
        _, first_turn_note = _render_first_turn_note(
            state=state,
            first_turn_hint=first_turn_hint,
        )

    last_user_text = ""
    for role, text in reversed(history):
        if role == "user":
            last_user_text = _sanitize_human_text(text)
            if last_user_text:
                break
    if not last_user_text:
        last_user_text = "none"

    draft_response_text = _sanitize_human_draft_text(brief.draft_response_text)
    draft_available = bool(draft_response_text)
    draft_length_cap_chars = _visible_text_length(draft_response_text) if draft_available else 0
    if not draft_response_text:
        draft_response_text = "none"
    draft_status_text = "available" if draft_available else "missing"

    primary_focus = brief.primary_focus if isinstance(brief.primary_focus, Mapping) else {}
    primary_focus_topic = _sanitize_human_text(primary_focus.get("topic")) or "none"
    primary_focus_meaning = _sanitize_human_text(primary_focus.get("meaning")) or "none"
    utterance_target_text = _sanitize_human_text(brief.utterance_target) or "none"
    question_reflect_seed_text = _sanitize_human_text(brief.question_reflect_seed) or "none"
    evocation_move_text = _sanitize_human_text(brief.evocation_move)
    if _find_internal_label_leak(evocation_move_text):
        evocation_move_text = ""
    evocation_move_text = evocation_move_text or "none"
    phase_goal_text = _sanitize_human_text(brief.phase_goal_this_turn) or "none"
    writer_plan_text = _format_writer_plan_for_prompt(brief.writer_plan)
    must_include_text = _join_writer_items(brief.must_include)
    must_avoid_text = _join_writer_items(brief.must_avoid)
    ct_anchors_text = _join_writer_items(brief.ct_anchors)
    ct_operation_goal_text = _sanitize_human_text(brief.ct_operation_goal) or "none"
    slot_quality_target_example_detail_text = _format_slot_quality_target_example_details_for_prompt(
        brief.slot_quality_target_examples
    )
    slot_goal_text = _sanitize_human_text(brief.slot_goal) or "none"
    repair_goal_text = _sanitize_human_text(brief.repair_goal_text) or "none"
    success_criterion_text = _sanitize_human_text(brief.success_criterion) or "none"
    layer4_repair_issue_code_texts = _format_layer4_repair_issue_code_groups_for_prompt(
        layer4_repair_issue_codes
    )
    clarify_preference_mode, focus_choice_options = _resolve_clarify_preference_mode(
        action=action,
        focus_choice_context=focus_choice_context,
    )
    focus_choice_options_text = _join_writer_items(focus_choice_options, fallback="none")
    macro_bridge_anchor_text = (
        _build_macro_bridge_anchor(
            state=state,
            primary_focus_topic=primary_focus_topic,
        )
        or "none"
    )

    writer_action_guardrails = _render_writer_action_guardrails(
        action=action,
        reflection_style=reflection_style,
        state=state,
        closing_phase_complete=closing_phase_complete,
        focus_choice_context=focus_choice_context,
    )
    writer_validation_checks = _render_writer_validation_checks(
        action=action,
        add_affirm=add_affirm,
        state=state,
        first_turn_hint=first_turn_hint,
        closing_phase_complete=closing_phase_complete,
        focus_choice_context=focus_choice_context,
    )
    layer3_action_rule_for_writer = _render_layer3_action_rule_for_writer(
        action=action,
        state=state,
        risk_assessment=risk_assessment,
        first_turn_hint=first_turn_hint,
        reflection_style=reflection_style,
        closing_phase_complete=closing_phase_complete,
        focus_choice_context=focus_choice_context,
    )
    layer3_action_rule_section = _get_layer4_writer_prompt_formatted(
        "layer3_action_rule_section_template",
        default=(
            "[Layer3 action rules: reference]\n"
            "- The rules below were fixed by Layer3. Keep their meaning intact while polishing the final wording.\n"
            "{action_rule}"
        ),
        action_rule=layer3_action_rule_for_writer,
    )

    minimal_mode_rules = _get_layer4_writer_prompt(
        "minimal_mode_rules",
        default=(
            "[Layer4 polishing mode]\n"
            "- Treat the completed Layer3 draft as the main anchor and polish it into a natural final response while obeying the fixed constraints and editor brief.\n"
            "- The top priority is to make the final response shorter than draft_response_text and more natural in dialogue. If you are unsure, favor brevity and naturalness over richer interpretation, but do not break the main action or the direction of change.\n"
            "- Preserve the meaning, topic, and structural intent of draft_response_text. Do not generate a new alternative.\n"
            "- draft_response_text is a meaning anchor, not a wording anchor. Keep the meaning and main action while shifting wording closer to the client's everyday language.\n"
            "- Do not change fixed_main_action.\n"
            "- Adjust affirmation and style in line with fixed_affirm_mode / fixed_reflection_style.\n"
            "- Do not change writer_plan. Respect the question limit.\n"
            "- Do not re-judge primary_focus / utterance_target / evocation_move / must_include / must_avoid once Layer3 has fixed them.\n"
            "- Limit edits to paraphrasing, redundancy reduction, word-order tuning, punctuation tuning, and natural sentence merging or splitting.\n"
            "- Add or remove a sentence only when minimally necessary to satisfy action rules or writer_plan.\n"
            "- Do not add new topics, new suggestions, new questions, history reinterpretations, or new strength labels.\n"
            "- If add_affirm != NONE, do not remove the affirmation itself. If it is too strong, compress it into a brief acknowledgment grounded in observable speech, choices, or actions.\n"
            "- If draft_response_text is present, the final text should not exceed its visible length and will usually be shorter.\n"
            "- Only if draft_response_text is missing may you rebuild one minimal natural response from utterance_target / evocation_move / must_include.\n"
            "- Edit priorities: 1) preserve the main action 2) respect the client's explicit latest preference 3) preserve the core of draft_response_text / evocation_move 4) keep it easy to receive, without pressure 5) maintain history consistency 6) reduce redundancy.\n"
            "- When shortening, prioritize: 1) the client's own words 2) the direction the client is trying to choose 3) both poles of ambivalence.\n"
            "- If several candidates preserve the same main action and change direction, choose the shorter, more natural, more everyday-language version.\n"
            "- If instructions conflict, prioritize: main-action guardrails / pre-output checks > affirmation rules > redundancy reduction.\n"
            "- If writer_plan.preferred_ending_family is present, favor that temperature of ending, but do not force unnatural explanatory wording.\n"
            "- If writer_plan.avoid_recent_ending_families is present, avoid repeating the same ending family when possible without breaking meaning.\n"
            "- Do not change agency, viewpoint, or ownership. Do not replace client-autonomy anchors such as 'you', 'your sense', or 'you decide' with counselor-centered wording.\n"
            "- Preserve at least one concrete client term whenever possible.\n"
            "- Even if must_include is empty, keep at least one concrete word from last_user_text when bodily sensation, action, scene, sound, or time-of-day language is present.\n"
            "- If possible, convert abstract nouns back into verbs, sensations, actions, or scenes unless the client used the abstraction directly.\n"
            "- Layer4 is not only about softening; it is also about returning language to lived everyday speech.\n"
            "- Do not let the final line land on abstract nouns alone. End on a scene, sensation, choice, or action.\n"
            "- layer4_repair_issue_codes.hard_contract are mandatory repairs; soft_hint are improvement hints. Do not expose code names in the response text.\n"
            "- If the draft over-interprets or over-stories, step it back one level toward the client's latest wording.\n"
            "- If the draft makes a stronger interpretation than the client explicitly stated, first weaken it into reversible wording, and if needed return to content closer to the explicit statement.\n"
            "- Do not erase the concrete focus of change talk.\n"
            "- Even while softening, preserve the change direction shown by ct_anchors / utterance_target / evocation_move rather than retreating to safe generalities.\n"
            "- If the client is already leaning to one side in an A/B choice or ambivalence, keep that side foregrounded and avoid promoting the unchosen side.\n"
            "- In SCALING_QUESTION, keep the scale tied to the concrete behavior that was named most recently rather than abstracting it.\n"
            "- If macro_bridge_anchor exists, you do not need to echo it literally; just keep the link between micro focus and macro theme intact.\n"
            "- Treat ct_anchors / ct_operation_goal / evocation_move / slot_goal / success_criterion as meaning-preservation notes, not wording to copy literally.\n"
            "- If possible, vary repeated counselor kernel words rather than repeating the same phrasing turn after turn.\n"
            "- If last_user_text explicitly says the client wants to keep talking, do not add new closing language or deferral to next time unless draft_response_text already contains it.\n"
            "- In REVIEW_REFLECTION / CLOSING, prefer everyday wording over added abstract terms the client did not use.\n"
            "- Keep warmth and understanding without overpraise, over-certainty, or promises about the future.\n"
            "- Make the final expression consistent with the latest client input and the dialogue history while trimming redundancy.\n"
            "- Before output, check for viewpoint reversal, loss of concrete terms, or over-abstraction, and correct those issues while staying short and natural.\n"
            "- Output only the response body, with no JSON, heading, bullet list, or role prefix."
        ),
    )
    hard_constraints = _get_layer4_writer_prompt_formatted(
        "hard_constraints_template",
        default=(
            "[Fixed constraints: do not re-decide]\n"
            "- phase: {phase_value} [{phase_name}]\n"
            "- turn_index: {turn_index}\n"
            "- fixed_main_action: {action_value}\n"
            "- fixed_affirm_mode: {affirm_mode}\n"
            "- fixed_reflection_style: {reflection_style}"
        ),
        phase_value=state.phase.value,
        phase_name=state.phase.name,
        turn_index=state.turn_index,
        action_value=action.value,
        affirm_mode=add_affirm.value,
        reflection_style=reflection_style_text,
    )
    editor_brief = _get_layer4_writer_prompt_formatted(
        "editor_brief_template",
        default=(
            "[Layer4 editor brief]\n"
            "- last_user_text: {last_user_text}\n"
            "- primary_focus.topic: {primary_focus_topic}\n"
            "- primary_focus.meaning: {primary_focus_meaning}\n"
            "- ct_anchors: {ct_anchors}\n"
            "- ct_operation_goal: {ct_operation_goal}\n"
            "- utterance_target: {utterance_target}\n"
            "- question_reflect_seed: {question_reflect_seed}\n"
            "- evocation_move: {evocation_move}\n"
            "- macro_bridge_anchor: {macro_bridge_anchor}\n"
            "- phase_goal_this_turn: {phase_goal_this_turn}\n"
            "- writer_plan: {writer_plan}\n"
            "- must_include: {must_include}\n"
            "- must_avoid: {must_avoid}\n"
            "- draft_status: {draft_status}\n"
            "- draft_length_cap_chars: {draft_length_cap_chars}\n"
            "- slot_goal: {slot_goal}\n"
            "- repair_goal_text: {repair_goal_text}\n"
            "- success_criterion: {success_criterion}\n"
            "- clarify_preference_mode: {clarify_preference_mode}\n"
            "- clarify_preference_options: {clarify_preference_options}\n"
            "- slot_quality_target_examples.detail: {slot_quality_target_examples_detail}\n"
            "- draft_response_text: {draft_response_text}\n"
            "- layer4_repair_issue_codes.hard_contract: {layer4_repair_issue_codes_hard_contract}\n"
            "- layer4_repair_issue_codes.soft_hint: {layer4_repair_issue_codes_soft_hint}\n"
            "- hard_contract must be resolved. Apply soft_hint only as far as naturalness permits.\n"
            "- The items above were fixed by Layer3. Layer4 should not re-decide them and should polish draft_response_text as the main anchor.\n"
            "- The full dialogue history is available for reference, but prioritize connection to the immediately preceding user input.\n"
            "- draft_response_text is the main meaning anchor, while wording may move back toward everyday language close to last_user_text.\n"
            "- Treat ct_anchors / ct_operation_goal / evocation_move / slot_goal / success_criterion as notes about what meaning must survive, not wording to surface directly.\n"
            "- macro_bridge_anchor is only a connection memo. Do not paraphrase it literally unless keeping the link through a concrete scene or action requires it."
        ),
        last_user_text=last_user_text,
        primary_focus_topic=primary_focus_topic,
        primary_focus_meaning=primary_focus_meaning,
        ct_anchors=ct_anchors_text,
        ct_operation_goal=ct_operation_goal_text,
        utterance_target=utterance_target_text,
        question_reflect_seed=question_reflect_seed_text,
        evocation_move=evocation_move_text,
        macro_bridge_anchor=macro_bridge_anchor_text,
        phase_goal_this_turn=phase_goal_text,
        writer_plan=writer_plan_text,
        must_include=must_include_text,
        must_avoid=must_avoid_text,
        draft_status=draft_status_text,
        draft_length_cap_chars=str(draft_length_cap_chars) if draft_length_cap_chars > 0 else "none",
        slot_goal=slot_goal_text,
        repair_goal_text=repair_goal_text,
        success_criterion=success_criterion_text,
        clarify_preference_mode=clarify_preference_mode,
        clarify_preference_options=focus_choice_options_text,
        slot_quality_target_examples_detail=slot_quality_target_example_detail_text,
        slot_repair_hints_detail=slot_quality_target_example_detail_text,
        draft_response_text=draft_response_text,
        layer4_repair_issue_codes_hard_contract=layer4_repair_issue_code_texts["hard_contract"],
        layer4_repair_issue_codes_soft_hint=layer4_repair_issue_code_texts["soft_hint"],
    )
    draft_length_contract = _get_layer4_writer_prompt_formatted(
        "draft_length_contract_template",
        default=(
            "【Draft Length Contract】\n"
            "- draft_status: {draft_status}\n"
            "- draft_length_cap_chars: {draft_length_cap_chars}\n"
            "- If draft_response_text exists, trim rather than add meaning and make the final sentence shorter than draft_response_text.\n"
            "- Prioritize brevity and naturalness. When in doubt, cut detail instead of adding interpretation.\n"
            "- Rebuild a single minimal natural sentence from utterance_target / evocation_move / must_include only if draft_response_text is missing."
        ),
        draft_status=draft_status_text,
        draft_length_cap_chars=str(draft_length_cap_chars) if draft_length_cap_chars > 0 else "none",
    )
    edit_task = _get_layer4_writer_prompt(
        "edit_task",
        default=(
            "【Edit Task】\n"
            "0. Highest priority: make the response shorter and more natural than draft_response_text. When in doubt, delete detail instead of adding interpretation. Keep the main action and the change direction.\n"
            "1. Use draft_response_text as the main anchor, fix hard-contract issues first, and use soft hints only as polish guidance.\n"
            "2. Respect utterance_target, evocation_move, must_include, must_avoid, and writer_plan while staying inside the main-action guardrails and affirmation rules.\n"
            "3. Polish mainly through paraphrasing, shortening, word-order adjustment, and natural sentence merging/splitting.\n"
            "3.1. Even when shortening, preserve the client's own wording, the direction they are leaning toward, and both poles of ambivalence whenever possible.\n"
            "3.2. Prefer everyday language close to last_user_text over abstract noun chains.\n"
            "3.3. Land the sentence on scene / sensation / choice / action wording rather than abstract meta language.\n"
            "3.4. If draft_response_text exists, trim without changing the meaning and keep the final non-space length under draft_length_cap_chars.\n"
            "3.5. Follow writer_plan.preferred_ending_family when present and avoid writer_plan.avoid_recent_ending_families when possible.\n"
            "4. Lower excessive interpretation or certainty by one step, but do not erase the evocative core or change-talk direction.\n"
            "4.1. Replace abstract meta language with verbs, sensations, actions, and scenes unless the user explicitly used the abstract wording.\n"
            "4.2. If add_affirm != NONE, compress affirmation into brief acknowledgment of observable speech, choice, or action rather than deleting it.\n"
            "4.3. If the draft interprets more strongly than the evidence supports, weaken it first into reversible phrasing.\n"
            "4.4. Do not change subject, viewpoint, or ownership. Do not turn autonomy anchors into counselor-centered wording.\n"
            "5. Let explicit wishes in last_user_text override weaker draft tendencies. Do not turn a continue signal into a closing signal.\n"
            "5.1. Preserve at least one concrete user term whenever possible.\n"
            "5.2. If macro_bridge_anchor exists, preserve the concrete connection without adding a new explanatory sentence that was absent from the draft.\n"
            "5.3. Keep the preferred side foregrounded when the user is already leaning toward one side of an A/B contrast.\n"
            "5.4. In SCALING_QUESTION, keep the concrete behavior name instead of abstracting it.\n"
            "6. Remove any extra explanation that breaks the one-turn, one-function shape.\n"
            "6.1. Do not add new topics, new proposals, new questions, history reinterpretation, or new strength labels.\n"
            "6.2. Do not delete the concrete focus of change talk.\n"
            "6.3. Avoid repeating the same core wording as the immediately previous counselor utterance.\n"
            "6.4. In QUESTION actions, keep sentence 1 as a brief reflection or grounded affirmation, and sentence 2 as the question only.\n"
            "6.5. In REVIEW_REFLECTION / CLOSING, prefer everyday language over meta reflection wording the user did not use.\n"
            "7. Before output, check for subject flip, viewpoint flip, loss of concrete wording, and over-abstraction, then fix any issue while staying short and natural.\n"
            "8. Output exactly one response text only. Do not expose issue codes or internal labels."
        ),
    )
    issue_repair_guidance = _render_layer4_issue_repair_guidance(
        layer4_repair_issue_codes,
        action=action,
    )

    fixed_system_sections = [
        _render_role_and_style(),
        _render_safety_rules(risk_note=""),
        minimal_mode_rules,
        _render_output_format_constraints(),
    ]
    runtime_system_sections = [
        _render_runtime_risk_note(_render_risk_note(risk_assessment)),
        hard_constraints,
    ]
    phase_guidance = _render_phase_guidance(state.phase)
    if phase_guidance:
        runtime_system_sections.append(phase_guidance)
    runtime_system_sections.extend(
        [
            layer3_action_rule_section,
            writer_action_guardrails,
            _render_affirm_rule(add_affirm),
            editor_brief,
            draft_length_contract,
        ]
    )
    if issue_repair_guidance:
        runtime_system_sections.append(issue_repair_guidance)
    runtime_system_sections.extend([edit_task, writer_validation_checks])
    if first_turn_note:
        runtime_system_sections.append(f"[First-turn exception]\n- {first_turn_note}")
    system = _assemble_system_with_runtime_tail(
        fixed_sections=fixed_system_sections,
        runtime_sections=runtime_system_sections,
    )
    system = inject_mi_knowledge(
        system,
        agent_name="response_writer",
        main_action=action.value,
    )
    return _build_messages_with_history_limit(
        system=system,
        history=history,
        max_turns=max_history_turns,
    )


@dataclass
class LLMResponseIntegrator:
    llm: LLMClient
    temperature: float = 0.1
    max_history_turns: int = 120
    retry_once_on_schema_error: bool = False
    json_mode: str = "loose"

    def integrate(
        self,
        *,
        history: List[Tuple[str, str]],
        state: DialogueState,
        action: MainAction,
        add_affirm: AffirmationMode,
        reflection_style: Optional[ReflectionStyle],
        risk_assessment: Optional[RiskAssessment],
        focus_candidates: Optional[Sequence[ChangeTalkCandidate]],
        slot_target: str,
        first_turn_hint: Optional[FirstTurnHint],
        allow_question_without_preface: bool,
        current_user_is_short: bool,
        phase_prediction_debug: Optional[Dict[str, Any]],
        action_ranking_debug: Optional[Dict[str, Any]],
        features: Optional[PlannerFeatures],
        change_talk_hint: Optional[str],
        focus_choice_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ResponseBrief, Dict[str, Any]]:
        focus_hint = (change_talk_hint or "").strip()
        candidate_list = list(focus_candidates or [])
        json_mode = _normalize_json_mode(self.json_mode)
        strict_json = json_mode == "strict"
        messages, meta = build_response_brief_messages(
            history=history,
            state=state,
            action=action,
            add_affirm=add_affirm,
            reflection_style=reflection_style,
            risk_assessment=risk_assessment,
            focus_candidates=candidate_list,
            slot_target=slot_target,
            first_turn_hint=first_turn_hint,
            allow_question_without_preface=allow_question_without_preface,
            current_user_is_short=current_user_is_short,
            phase_prediction_debug=phase_prediction_debug,
            action_ranking_debug=action_ranking_debug,
            features=features,
            change_talk_hint=focus_hint,
            focus_choice_context=focus_choice_context,
            max_history_turns=self.max_history_turns,
        )

        request_label = "layer3:response_integrator"

        def _run(messages_in: List[Dict[str, str]]) -> Tuple[str, Any, ResponseBrief, List[str], bool, bool]:
            raw = self.llm.generate(
                messages_in,
                temperature=self.temperature,
                request_label=request_label,
            )
            parsed = _parse_json_from_text(raw, strict=strict_json)
            strict_json_schema_rejected = strict_json and not isinstance(parsed, Mapping)
            slot_quality_target_examples = meta.get("slot_quality_target_examples")
            if slot_quality_target_examples is None:
                slot_quality_target_examples = meta.get("slot_repair_hints")
            if not isinstance(slot_quality_target_examples, Sequence) or isinstance(slot_quality_target_examples, (str, bytes)):
                slot_quality_target_examples = []
            brief, issues = _normalize_response_brief_payload(
                payload=parsed,
                raw_output_text=raw,
                state=state,
                action=action,
                add_affirm=add_affirm,
                reflection_style=reflection_style,
                risk_assessment=risk_assessment,
                focus_candidates=candidate_list,
                slot_target=slot_target,
                slot_quality_target_examples=slot_quality_target_examples,
                change_talk_hint=focus_hint,
                history=history,
                history_start_turn=int(meta.get("history_turn_start", 1)),
                history_end_turn=int(meta.get("history_turn_end", len(history))),
                focus_choice_context=focus_choice_context,
            )
            strict_rejected = bool(strict_json and (strict_json_schema_rejected or issues))
            # Fall back to default_response_brief only when Layer3 failed to return an object-shaped output at all.
            if strict_json and strict_json_schema_rejected:
                brief = _default_response_brief(
                    state=state,
                    action=action,
                    add_affirm=add_affirm,
                    reflection_style=reflection_style,
                    risk_assessment=risk_assessment,
                    focus_candidates=candidate_list,
                    change_talk_hint=focus_hint,
                    history=history,
                    slot_target=slot_target,
                    slot_quality_target_examples=slot_quality_target_examples,
                    focus_choice_context=focus_choice_context,
                )
            return raw, parsed, brief, issues, strict_json_schema_rejected, strict_rejected

        raw_output, parsed, brief, issues, strict_json_schema_rejected, strict_rejected = _run(messages)
        retried = False
        retry_raw = ""
        retry_parsed: Any = None
        retry_strict_json_schema_rejected = False
        retry_strict_rejected = False
        has_brief_shape = bool(
            isinstance(parsed, Mapping)
            and any(key in parsed for key in ("meta", "dialogue_digest", "primary_focus", "writer_plan"))
        )
        should_retry = bool(
            issues
            and self.retry_once_on_schema_error
            and has_brief_shape
            and not strict_json
        )
        if should_retry:
            retried = True
            repair_user = {
                "role": "user",
                "content": (
                    "The previous JSON had schema issues: "
                    + ", ".join(issues[:5])
                    + ". Re-output JSON only for the same input."
                ),
            }
            retry_messages = messages + [{"role": "assistant", "content": raw_output}, repair_user]
            (
                retry_raw,
                retry_parsed,
                retry_brief,
                retry_issues,
                retry_strict_json_schema_rejected,
                retry_strict_rejected,
            ) = _run(retry_messages)
            if len(retry_issues) <= len(issues):
                (
                    raw_output,
                    parsed,
                    brief,
                    issues,
                    strict_json_schema_rejected,
                    strict_rejected,
                ) = (
                    retry_raw,
                    retry_parsed,
                    retry_brief,
                    retry_issues,
                    retry_strict_json_schema_rejected,
                    retry_strict_rejected,
                )

        debug = {
            "method": "llm_response_integrator_strict_reject" if strict_rejected else "llm_response_integrator",
            "raw_output": raw_output,
            "parsed": parsed,
            "schema_issues": issues,
            "json_mode": json_mode,
            "strict_json_schema_rejected": strict_json_schema_rejected,
            "strict_json_rejected": strict_rejected,
            "focus_candidates": [dataclasses.asdict(candidate) for candidate in candidate_list],
            "slot_target": slot_target,
            "focus_choice_context": _normalize_focus_choice_hint(focus_choice_context),
            "retry_used": retried,
            "retry_raw_output": retry_raw if retried else None,
            "retry_parsed": retry_parsed if retried else None,
            "history_turn_start": meta.get("history_turn_start"),
            "history_turn_end": meta.get("history_turn_end"),
            "slot_quality_target_examples": meta.get("slot_quality_target_examples"),
            "slot_repair_hints": meta.get("slot_quality_target_examples"),
            "request_label": request_label,
        }
        return brief, debug


# ----------------------------
# Output validation (minimal)
# ----------------------------
def _contains_affirmation(text: str) -> bool:
    """Lightweight check for affirmation-like wording using partial matches to typical phrases."""
    for pat in _AFFIRM_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def _contains_complex_affirmation(text: str) -> bool:
    """
    Treat complex affirmations as lines that contain both affirmation-like phrasing and explicit value/strength language.
    """
    if not _contains_affirmation(text):
        return False
    return any(re.search(pat, text) for pat in _AFFIRM_COMPLEX_PATTERNS)


def _contains_question_form(text: str) -> bool:
    return bool(_RE_QUESTION.search(text))


_CHANGE_TALK_FOCUS_STOPWORDS = {
    "change",
    "intention",
    "reason",
    "feeling",
    "interest",
    "forward movement",
    "ambivalence",
    "sustain talk",
    "implementation",
    "concretization",
    "sign",
    "implied",
}


def _extract_change_talk_focus_terms(change_talk_inference: str, *, max_terms: int = 6) -> List[str]:
    raw = str(change_talk_inference or "").strip()
    if not raw or raw == "none":
        return []

    parsed_items = _parse_change_talk_items(raw, max_items=max(max_terms * 2, 8), for_focus_terms=True)
    terms: List[str] = []
    for item in parsed_items:
        token = re.sub(r"\s+", "", item or "")
        if not token:
            continue
        if token in _CHANGE_TALK_FOCUS_STOPWORDS:
            continue
        if token in terms:
            continue
        terms.append(token)
        if len(terms) >= max_terms:
            return terms
    return terms


def _count_change_talk_focus_mentions(
    text: str,
    change_talk_inference: str,
    *,
    max_terms: int = 10,
) -> Tuple[int, List[str]]:
    body = re.sub(r"\s+", "", text or "")
    if not body:
        return 0, []

    terms = sorted(
        _extract_change_talk_focus_terms(change_talk_inference, max_terms=max_terms),
        key=len,
        reverse=True,
    )
    matched: List[str] = []
    for term in terms:
        token = re.sub(r"\s+", "", term or "")
        if not token:
            continue
        if token not in body:
            continue
        if any(token in existing or existing in token for existing in matched):
            continue
        matched.append(token)
    return len(matched), matched


def _contains_change_talk_focus_reference(text: str, change_talk_inference: str) -> bool:
    body = re.sub(r"\s+", "", text or "")
    if not body:
        return False

    # If the reflection itself already contains clear change language, treat that as the minimum acceptable focus.
    if any(marker in body for marker in _CHANGE_TALK_MARKERS):
        return True
    if re.search(r"(?:wanttochange|wanttostop|wantto|want|wanttotry|wanttocontinue|seemsdoable|wanttoworkon)", body):
        return True

    focus_terms = _extract_change_talk_focus_terms(change_talk_inference)
    return any(term and term in body for term in focus_terms)


def _extract_change_talk_enrichment_categories(text: str) -> set[str]:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return set()
    categories: set[str] = set()
    has_scene_context = (
        _has_any_pattern(normalized, _SLOT_CONTEXT_TIME_PATTERNS)
        or _has_any_pattern(normalized, _SLOT_CONTEXT_PLACE_PATTERNS)
        or _has_any_pattern(normalized, _CT_STRENGTH_SCENE_TRIGGER_PATTERNS)
    )
    if has_scene_context:
        categories.add("scene")
    if _has_any_pattern(normalized, _SLOT_CONTEXT_TIME_PATTERNS) or re.search(r"(\d+\s*(times?|minutes?|hours?|days?|weeks?))", normalized):
        categories.add("frequency")
    if any(marker in normalized for marker in _SUMMARY_VALUE_MARKERS) or any(
        marker in normalized for marker in _CT_STRENGTH_VALUE_REASON_MARKERS
    ):
        categories.add("value_reason")
    if _contains_observable_behavior_reference(normalized) or _has_any_pattern(normalized, _CT_STRENGTH_NEXT_STEP_PATTERNS):
        categories.add("next_step")
    return categories


def _has_change_talk_strengthening(text: str, change_talk_inference: str) -> bool:
    normalized_output = _normalize_slot_text(text)
    if not normalized_output:
        return False

    output_categories = _extract_change_talk_enrichment_categories(normalized_output)
    normalized_hint = _normalize_slot_text(change_talk_inference)
    hint_categories = _extract_change_talk_enrichment_categories(normalized_hint) if normalized_hint else set()
    category_gain = bool(output_categories - hint_categories) if normalized_hint else bool(output_categories)

    hint_terms = _extract_change_talk_focus_terms(change_talk_inference, max_terms=4)
    hint_kinds = [_infer_change_talk_kind(term) for term in hint_terms]
    hint_has_darn = any(kind in {"desire", "ability", "reason", "need"} for kind in hint_kinds)
    hint_has_cat = any(kind in {"commitment", "activation", "taking_step"} for kind in hint_kinds)
    output_kind = _infer_change_talk_kind(normalized_output)
    darn_to_cat_progress = hint_has_darn and not hint_has_cat and output_kind in {"commitment", "activation", "taking_step"}
    return category_gain or darn_to_cat_progress


def _split_output_clauses(text: str) -> List[str]:
    return [chunk.strip() for chunk in re.split(r"[。\n]", text) if chunk.strip()]


def _visible_text_length(text: Any) -> int:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return len(normalized)


def _count_question_clauses(text: str) -> int:
    return sum(1 for chunk in _split_output_clauses(text) if _contains_question_form(chunk))


def _last_output_clause(text: str) -> str:
    chunks = _split_output_clauses(text)
    if not chunks:
        return text.strip()
    return chunks[-1]


def _extract_question_preface_clause(text: str) -> str:
    chunks = _split_output_clauses(text)
    if len(chunks) < 2:
        return ""
    if _contains_question_form(chunks[0]):
        return ""
    if not _contains_question_form(chunks[-1]):
        return ""
    return chunks[0]


def _has_sentence_with_too_many_commas(text: str, *, max_commas: int = 2) -> bool:
    for chunk in _split_output_clauses(text):
        comma_count = chunk.count("、") + chunk.count(",") + chunk.count("，")
        if comma_count > max_commas:
            return True
    return False


def _contains_autonomy_forbidden_pattern(text: str) -> bool:
    return any(pat.search(text) for pat in _AUTONOMY_FORBIDDEN_PATTERNS)


def _contains_action_mismatch_suggestion(text: str) -> bool:
    return any(pat.search(text) for pat in _ACTION_MISMATCH_SUGGESTION_PATTERNS)


def _contains_action_mismatch_choice_prompt(text: str) -> bool:
    return any(pat.search(text) for pat in _ACTION_MISMATCH_CHOICE_PATTERNS)


def _contains_summary_forward_prompt(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    return any(pat.search(compact) for pat in _SUMMARY_FORWARD_PROMPT_PATTERNS)


def _contains_observable_behavior_reference(text: str) -> bool:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return False
    has_actionable_verb = any(pat.search(normalized) for pat in _SLOT_QUALITY_ACTIONABLE_VERB_PATTERNS)
    has_context_anchor = any(pat.search(normalized) for pat in _OBSERVABLE_BEHAVIOR_CONTEXT_PATTERNS)
    return has_actionable_verb and has_context_anchor


def _contains_next_step_autonomy_support(text: str) -> bool:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return False
    if _contains_question_form(normalized):
        return True
    return any(marker in normalized for marker in _NEXT_STEP_AUTONOMY_MARKERS)


def _contains_reflect_advice_like_pattern(text: str) -> bool:
    return any(pat.search(text) for pat in _REFLECT_ADVICE_LIKE_PATTERNS)


def _contains_overinterpretation_risk(text: str) -> bool:
    normalized = _normalize_slot_text(text)
    if not normalized:
        return False
    inference_hit = any(pat.search(normalized) for pat in _OVERINTERPRETATION_INFERENCE_PATTERNS)
    story_hits = sum(1 for pat in _OVERINTERPRETATION_STORY_PATTERNS if pat.search(normalized))
    if inference_hit and story_hits >= 1:
        return True
    if story_hits >= 2 and any(marker in normalized for marker in ("from here", "foundation", "strength", "grow", "entry point")):
        return True
    return False


def _collect_layer4_rewrite_issue_codes_from_soft_warnings(
    *,
    action: MainAction,
    soft_warnings: Sequence[str],
) -> List[str]:
    normalized_warnings = [_normalize_slot_text(warning) for warning in soft_warnings]
    has_global_rewrite_warning = any(
        warning in _LAYER4_GLOBAL_REWRITE_WARNING_CODES for warning in normalized_warnings if warning
    )
    if action not in _SOFT_WARNING_LAYER4_REWRITE_ACTIONS and not has_global_rewrite_warning:
        return []
    issue_codes: List[str] = []
    for warning in normalized_warnings:
        code = _SOFT_WARNING_LAYER4_REWRITE_ISSUE_MAP.get(str(warning or "").strip())
        if not code or code in issue_codes:
            continue
        issue_codes.append(code)
    return issue_codes


def _normalize_output_validation_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "strict":
        return "strict"
    return "warn"


def _build_final_closing_response(
    *,
    action: MainAction,
    add_affirm: AffirmationMode,
) -> str:
    if action == MainAction.REFLECT_SIMPLE:
        reflect_line = "Today, your wish to check the pace that feels right for you came through clearly."
    else:
        reflect_line = (
            "Today, even while holding both the wish to move forward and the hesitation, "
            "you seemed to be carefully searching for a way that feels workable to you."
        )

    if add_affirm == AffirmationMode.SIMPLE:
        affirm_line = "Your care in putting these feelings into words has come through."
    elif add_affirm == AffirmationMode.COMPLEX:
        affirm_line = "There is real honesty and persistence in the way you have faced and named these mixed feelings."
    else:
        affirm_line = ""

    parts: List[str] = []
    if affirm_line:
        parts.append(affirm_line)
    parts.append(reflect_line)
    parts.append(_FINAL_CLOSING_MESSAGE)
    return "".join(parts)


def _is_reflection_ellipsis_ending(tail: str) -> bool:
    return any(pat.search(tail) for pat in _REFLECTION_ELLIPSIS_ENDING_PATTERNS)


def _find_suspicious_ascii_tokens_in_japanese_text(text: str) -> List[str]:
    t = (text or "").strip()
    if not t or not _RE_HAS_JAPANESE_CHAR.search(t):
        return []

    suspicious: List[str] = []
    seen: Set[str] = set()
    for match in _RE_SUSPICIOUS_ASCII_TOKEN.finditer(t):
        token = str(match.group(1) or "")
        lower = token.lower()
        if not token:
            continue
        if lower in _ALLOWED_INLINE_ASCII_TOKENS:
            continue
        if token.isupper() and len(token) <= 3:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        suspicious.append(token)
    return suspicious


def _looks_natural_counselor_japanese(text: str, *, action: Optional[MainAction] = None) -> bool:
    """
    Lightweight check for whether the response looks minimally natural as counselor language.
    This is not a full grammar check; it mainly blocks obviously unnatural output such as JSON fragments or chat slang.
    For reflection actions and SUMMARY only, allow elliptical endings that still read as natural reflective clauses.
    """
    t = (text or "").strip()
    if not t:
        return False
    if not _RE_HAS_JAPANESE_CHAR.search(t):
        return False
    if _RE_JSON_LIKE_TEXT.search(t):
        return False
    if _RE_DIALOGUE_ROLE_PREFIX.search(t):
        return False
    if "```" in t:
        return False
    if any(ch in t for ch in "{}[]<>`"):
        return False
    if _RE_CHAT_SLANG.search(t) or t.lower().endswith("w"):
        return False

    last_clause = _last_output_clause(t)
    tail = re.sub(r"[。．!?？！\s]+$", "", last_clause)
    if not tail:
        return False
    if _RE_CASUAL_ENDING.search(tail):
        return False

    polite_markers = ("thank you", "it sounds", "it seems", "could", "would", "may", "might", "can", "does")
    if (
        action is not None
        and (_is_reflect_action(action) or action == MainAction.SUMMARY)
        and _is_reflection_ellipsis_ending(tail)
    ):
        return True
    if not any(marker in tail for marker in polite_markers):
        return False
    return True


def collect_soft_validation_warnings(
    action: MainAction,
    text: str,
    *,
    add_affirm: AffirmationMode | bool | str = AffirmationMode.NONE,
    change_talk_inference: Optional[str] = None,
    history: Optional[Sequence[Tuple[str, str]]] = None,
    state: Optional[DialogueState] = None,
    brief: Optional[ResponseBrief] = None,
    enforce_draft_length_cap: bool = False,
    closing_phase_complete: bool = False,
) -> List[str]:
    t = text.strip()
    if not t:
        return []

    warnings: List[str] = []
    affirm_mode = _normalize_affirmation_mode(add_affirm)

    if _is_reflect_action(action) and action in {MainAction.REFLECT, MainAction.REFLECT_SIMPLE, MainAction.REFLECT_COMPLEX}:
        focus_hint = (change_talk_inference or "").strip()
        if focus_hint and focus_hint != "none":
            if not _contains_change_talk_focus_reference(t, focus_hint):
                warnings.append("reflect_missing_change_talk_focus")
            else:
                focus_count, _ = _count_change_talk_focus_mentions(t, focus_hint)
                if focus_count > 2:
                    warnings.append("reflect_too_many_change_talk_points")
                if not _has_change_talk_strengthening(t, focus_hint):
                    warnings.append("reflect_change_talk_not_strengthened")
    if _is_reflect_action(action) and _contains_reflect_advice_like_pattern(t):
        warnings.append("reflect_advice_like_phrase")
    if (_is_reflect_action(action) or action == MainAction.SUMMARY) and _contains_overinterpretation_risk(t):
        warnings.append("overinterpretation_risk")
    if _is_reflect_action(action):
        ending_family_guidance = _analyze_reflect_ending_family_bias(
            text=t,
            history=history or [],
            action=action,
        )
        if bool(ending_family_guidance.get("warning")):
            warnings.append("reflect_ending_family_bias")

    if action == MainAction.SUMMARY:
        summary_text = _normalize_slot_text(t)
        if not any(marker in summary_text for marker in _SUMMARY_EMOTION_MARKERS):
            warnings.append("summary_missing_emotion_layer")
        if not any(marker in summary_text for marker in _SUMMARY_VALUE_MARKERS):
            warnings.append("summary_missing_value_layer")
        if not any(marker in summary_text for marker in _SUMMARY_AMBIVALENCE_MARKERS):
            warnings.append("summary_missing_ambivalence_layer")
        focus_hint = (change_talk_inference or "").strip()
        if focus_hint and focus_hint != "none":
            if not _contains_change_talk_focus_reference(summary_text, focus_hint):
                warnings.append("summary_missing_change_talk_focus")
            elif not _has_change_talk_strengthening(summary_text, focus_hint):
                warnings.append("summary_change_talk_not_strengthened")

    if state is not None and state.phase == Phase.FOCUSING_TARGET_BEHAVIOR:
        if not _contains_observable_behavior_reference(t):
            warnings.append("focusing_missing_observable_target_behavior")

    if state is not None and state.phase == Phase.NEXT_STEP_DECISION:
        if not _contains_observable_behavior_reference(t):
            warnings.append("next_step_missing_concrete_step")
        if not _contains_next_step_autonomy_support(t):
            warnings.append("next_step_missing_autonomy_support")

    if closing_phase_complete:
        if not any(marker in t for marker in _CLOSING_FIRST_TURN_ENDING_MARKERS):
            warnings.append("closing_final_turn_missing_ending_signal")
        if not any(marker in t for marker in _CLOSING_FIRST_TURN_SESSION_MARKERS):
            warnings.append("closing_final_turn_missing_session_anchor")

    if enforce_draft_length_cap and isinstance(brief, ResponseBrief):
        draft_length_cap = _visible_text_length(brief.draft_response_text)
        if draft_length_cap > 0 and _visible_text_length(t) >= draft_length_cap:
            warnings.append("layer4_not_shorter_than_draft")

    if affirm_mode == AffirmationMode.SIMPLE and not _contains_affirmation(t):
        warnings.append("affirmation_simple_missing")
    if affirm_mode == AffirmationMode.COMPLEX and not _contains_complex_affirmation(t):
        warnings.append("affirmation_complex_missing")
    if _find_suspicious_ascii_tokens_in_japanese_text(t):
        warnings.append("mixed_script_noise_token")
    return warnings


def _normalize_layer4_edit_audit_text(value: Any) -> str:
    text = _sanitize_human_text(value)
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def _iter_layer4_edit_audit_anchor_segments(text: Any) -> List[str]:
    sanitized = _sanitize_human_text(text)
    if not sanitized:
        return []

    segments: List[str] = []
    seen: Set[str] = set()

    def _add(segment: str) -> None:
        normalized = _normalize_layer4_edit_audit_text(segment)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        segments.append(segment)

    for match in _LAYER4_EDIT_AUDIT_QUOTED_PHRASE_PATTERN.finditer(sanitized):
        _add(str(match.group(1) or ""))
    for pattern in _LAYER4_EDIT_AUDIT_CONCRETE_SPAN_PATTERNS:
        for match in pattern.finditer(sanitized):
            _add(str(match.group(1) or ""))
    return segments[:8]


def _collect_layer4_edit_audit_anchor_candidates(brief: Optional[ResponseBrief]) -> List[str]:
    if not isinstance(brief, ResponseBrief):
        return []

    candidates: List[str] = []
    seen: Set[str] = set()

    def _append(raw: Any) -> None:
        candidate = _sanitize_human_text(raw)
        normalized = _normalize_layer4_edit_audit_text(candidate)
        if not normalized or len(normalized) < 2 or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(candidate)

    for item in brief.must_include or []:
        _append(item)
    for item in brief.slot_quality_target_examples or []:
        _append(_extract_slot_quality_target_example_detail(item))
        _append(_extract_slot_quality_target_information(item))
    for item in _iter_layer4_edit_audit_anchor_segments(brief.utterance_target):
        _append(item)
    for item in _iter_layer4_edit_audit_anchor_segments(brief.draft_response_text):
        _append(item)
    return candidates[:8]


def _audit_layer4_edit(
    *,
    enabled: bool,
    draft_response_text: Any,
    assistant_text: Any,
    brief: Optional[ResponseBrief],
) -> Dict[str, Any]:
    audit: Dict[str, Any] = {
        "enabled": bool(enabled),
        "checked": False,
        "issues": [],
        "issue_count": 0,
        "subject_reversal": False,
        "viewpoint_reversal": False,
        "concrete_anchor_dropped": False,
        "over_abstracted": False,
        "meta_bridge_added": False,
        "checked_anchors": [],
        "kept_anchors": [],
        "dropped_anchors": [],
        "added_abstract_terms": [],
    }
    if not enabled:
        return audit

    draft_text = _sanitize_human_draft_text(draft_response_text)
    final_text = _sanitize_human_text(assistant_text)
    if not draft_text or not final_text:
        return audit

    draft_normalized = _normalize_layer4_edit_audit_text(draft_text)
    final_normalized = _normalize_layer4_edit_audit_text(final_text)
    if not draft_normalized or not final_normalized:
        return audit

    audit["checked"] = True

    checked_anchors: List[str] = []
    kept_anchors: List[str] = []
    dropped_anchors: List[str] = []
    for candidate in _collect_layer4_edit_audit_anchor_candidates(brief):
        candidate_normalized = _normalize_layer4_edit_audit_text(candidate)
        if not candidate_normalized or candidate_normalized not in draft_normalized:
            continue
        checked_anchors.append(candidate)
        if candidate_normalized in final_normalized:
            kept_anchors.append(candidate)
        else:
            dropped_anchors.append(candidate)

    subject_reversal = any(
        source in draft_normalized and target in final_normalized and source not in final_normalized
        for source, target in _LAYER4_EDIT_AUDIT_SUBJECT_REVERSAL_PAIRS
    )
    viewpoint_reversal = subject_reversal or any(
        source in draft_normalized and target in final_normalized and source not in final_normalized
        for source, target in _LAYER4_EDIT_AUDIT_VIEWPOINT_REVERSAL_PAIRS
    )
    if not viewpoint_reversal:
        has_autonomy_anchor = any(anchor in draft_normalized for anchor in _LAYER4_EDIT_AUDIT_AUTONOMY_ANCHORS)
        introduced_first_person = any(
            marker in final_normalized and marker not in draft_normalized
            for marker in _LAYER4_EDIT_AUDIT_FIRST_PERSON_MARKERS
        )
        viewpoint_reversal = bool(has_autonomy_anchor and introduced_first_person)

    draft_abstract_terms = [term for term in _LAYER4_EDIT_AUDIT_ABSTRACT_TERMS if term in draft_normalized]
    final_abstract_terms = [term for term in _LAYER4_EDIT_AUDIT_ABSTRACT_TERMS if term in final_normalized]
    added_abstract_terms = [term for term in final_abstract_terms if term not in draft_abstract_terms]
    concrete_anchor_dropped = bool(dropped_anchors)
    meta_bridge_added = any(
        pattern.search(final_text) and not pattern.search(draft_text)
        for pattern in _LAYER4_EDIT_AUDIT_META_BRIDGE_PATTERNS
    )
    over_abstracted = bool(
        meta_bridge_added
        or (
            added_abstract_terms
            and (concrete_anchor_dropped or len(final_abstract_terms) > len(draft_abstract_terms))
        )
    )

    issues: List[str] = []
    if subject_reversal:
        issues.append("subject_reversal")
    if viewpoint_reversal:
        issues.append("viewpoint_reversal")
    if concrete_anchor_dropped:
        issues.append("concrete_anchor_dropped")
    if meta_bridge_added:
        issues.append("meta_bridge_added")
    if over_abstracted:
        issues.append("over_abstracted")

    audit.update(
        {
            "issues": issues,
            "issue_count": len(issues),
            "subject_reversal": subject_reversal,
            "viewpoint_reversal": viewpoint_reversal,
            "concrete_anchor_dropped": concrete_anchor_dropped,
            "over_abstracted": over_abstracted,
            "meta_bridge_added": meta_bridge_added,
            "checked_anchors": checked_anchors,
            "kept_anchors": kept_anchors,
            "dropped_anchors": dropped_anchors,
            "added_abstract_terms": added_abstract_terms,
        }
    )
    return audit


def validate_output(
    action: MainAction,
    text: str,
    *,
    add_affirm: AffirmationMode | bool | str = AffirmationMode.NONE,
    state: Optional[DialogueState] = None,
    first_turn_hint: Optional[FirstTurnHint] = None,
    change_talk_inference: Optional[str] = None,
    brief: Optional[ResponseBrief] = None,
    closing_phase_complete: bool = False,
) -> Tuple[bool, str]:
    t = text.strip()
    if not t:
        return False, "empty"
    affirm_mode = _normalize_affirmation_mode(add_affirm)
    max_commas_per_sentence = 2
    if isinstance(brief, ResponseBrief):
        try:
            max_commas_per_sentence = int((brief.language_constraints or {}).get("max_commas_per_sentence", 2) or 2)
        except Exception:
            max_commas_per_sentence = 2
    max_commas_per_sentence = max(1, min(4, max_commas_per_sentence))
    if _contains_autonomy_forbidden_pattern(t):
        return False, "autonomy_forbidden_pattern"
    if _find_internal_label_leak(t):
        return False, "internal_label_leak"
    if (
        not _RE_JSON_LIKE_TEXT.search(t)
        and not _RE_DIALOGUE_ROLE_PREFIX.search(t)
        and _find_suspicious_ascii_tokens_in_japanese_text(t)
    ):
        return False, "mixed_script_noise_token"
    if action != MainAction.PROVIDE_INFO and _contains_action_mismatch_suggestion(t):
        return False, "action_mismatch_suggestion"
    if _is_reflect_action(action) and _contains_action_mismatch_choice_prompt(t):
        return False, "reflect_contains_choice_prompt"
    if not _looks_natural_counselor_japanese(t, action=action):
        return False, "unnatural_counselor_japanese"
    if _has_sentence_with_too_many_commas(t, max_commas=max_commas_per_sentence):
        return False, "sentence_too_many_commas"

    allow_reflect_question = state is not None and _is_closing_first_turn(state)
    if _is_reflect_action(action):
        if ("？" in t or "?" in t) and not allow_reflect_question:
            return False, "reflect_contains_question_mark"
    if action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        clauses = _split_output_clauses(t)
        if len(clauses) != 2:
            return False, "ask_permission_not_single_sentence"
        last_clause = _last_output_clause(t)
        if not _contains_question_form(last_clause):
            return False, "ask_permission_not_question"
        if _count_question_clauses(t) != 1:
            return False, "ask_permission_question_count_invalid"
        if any(pat.search(t) for pat in _ASK_PERMISSION_FORBIDDEN_PATTERNS):
            return False, "ask_permission_forbidden_phrase"
        if not any(keyword in last_clause for keyword in _ASK_PERMISSION_REQUIRED_KEYWORDS):
            return False, "ask_permission_missing_info_context"
    if action == MainAction.PROVIDE_INFO:
        has_option = any(pat.search(t) for pat in _PROVIDE_INFO_OPTION_PATTERNS)
        has_neutral_frame = any(phrase in t for phrase in _PROVIDE_INFO_NEUTRAL_PHRASES)
        if not (has_option or has_neutral_frame):
            return False, "provide_info_missing_option_or_neutral_frame"
        if _count_question_clauses(t) != 1:
            return False, "provide_info_question_count_invalid"
        last_clause = _last_output_clause(t)
        if not _contains_question_form(last_clause):
            return False, "provide_info_missing_final_question"
        if not any(marker in last_clause for marker in _PROVIDE_INFO_REACTION_MARKERS):
            return False, "provide_info_missing_reaction_prompt"
    if action == MainAction.CLARIFY_PREFERENCE:
        if not _contains_question_form(t):
            return False, "clarify_preference_missing_question"
        qcount = t.count("？") + t.count("?")
        if qcount >= 2:
            return False, "clarify_preference_too_many_questions"
        if not any(marker in t.lower() for marker in ("which", "would help more", "which side", "which feels closer")):
            return False, "clarify_preference_missing_choice"
    if action == MainAction.SCALING_QUESTION:
        if not _contains_question_form(t):
            return False, "scaling_question_missing_question"
        if _count_question_clauses(t) != 1:
            return False, "scaling_question_count_invalid"
        scale_value = _extract_scale_0_10_value(t)
        if scale_value is None:
            return False, "scaling_question_missing_scale"
        if scale_value < 0.0 or scale_value > 10.0:
            return False, "scaling_question_scale_out_of_range"
        if any(marker in t for marker in _SLOT_SCALE_REASON_PROBE_STRICT_MARKERS):
            return False, "scaling_question_should_not_ask_reason"
        if _contains_scale_plus_one_probe(t):
            return False, "scaling_question_should_not_ask_plus_one"
    allow_questionless = (
        action == MainAction.QUESTION
        and state is not None
        and state.turn_index == 0
        and state.phase == Phase.GREETING
        and first_turn_hint == FirstTurnHint.GREETING_ONLY
    )
    if action == MainAction.QUESTION and not allow_questionless:
        if not _contains_question_form(t):
            return False, "question_missing"
        qcount = t.count("？") + t.count("?")
        if qcount >= 2:
            return False, "too_many_questions"
    if action == MainAction.QUESTION and state is not None and _is_scale_followup_pending(state):
        followup_step = _current_scale_followup_step(state)
        pending_score = state.scale_followup_score
        if followup_step == _SCALE_FOLLOWUP_STEP_PLUS_ONE:
            if not _contains_scale_plus_one_probe(t):
                return False, "scale_followup_question_missing_plus_one_probe"
            if any(marker in t for marker in _SLOT_SCALE_REASON_PROBE_STRICT_MARKERS):
                return False, "scale_followup_plus_one_question_contains_reason_probe"
        else:
            if not any(marker in t for marker in _SLOT_SCALE_REASON_MARKERS):
                return False, "scale_followup_question_missing_reason_probe"
            if (
                pending_score is not None
                and pending_score > 0.0
                and not _contains_scale_zero_anchor(t)
            ):
                return False, "scale_followup_question_missing_zero_anchor"
            if _contains_scale_plus_one_probe(t):
                return False, "scale_followup_reason_question_contains_plus_one_probe"
    if state is not None and _is_review_reflection_first_turn(state):
        clauses = _split_output_clauses(t)
        if len(clauses) < 3:
            return False, "review_first_turn_structure_missing"
        if not any(marker in t for marker in _REVIEW_FIRST_TURN_SUMMARY_MARKERS):
            return False, "review_first_turn_summary_missing"
        last_clause = _last_output_clause(t)
        if not _contains_question_form(last_clause):
            return False, "review_first_turn_final_question_missing"
        if not any(marker in last_clause for marker in _REVIEW_FIRST_TURN_QUESTION_MARKERS):
            return False, "review_first_turn_reflection_prompt_missing"
    if state is not None and _is_closing_first_turn(state):
        clauses = _split_output_clauses(t)
        if len(clauses) < 2:
            return False, "closing_first_turn_structure_missing"
        last_clause = _last_output_clause(t)
        if not _contains_question_form(last_clause):
            return False, "closing_first_turn_final_question_missing"
        if not any(marker in last_clause for marker in _CLOSING_FIRST_TURN_ENDING_MARKERS):
            return False, "closing_first_turn_ending_confirmation_missing"
        if not any(marker in last_clause for marker in _CLOSING_FIRST_TURN_SESSION_MARKERS):
            return False, "closing_first_turn_session_anchor_missing"
    if closing_phase_complete:
        if _contains_question_form(t):
            return False, "closing_final_turn_contains_question"
    if action == MainAction.SUMMARY:
        if _contains_question_form(t):
            return False, "summary_contains_question"
        if _contains_summary_forward_prompt(t):
            return False, "summary_contains_forward_prompt"
        sentence_count = len(_split_output_clauses(t))
        if sentence_count < 2 or sentence_count > 3:
            return False, "summary_sentence_count_out_of_range"
    contract_question_count_max: Optional[int] = None
    if isinstance(brief, ResponseBrief):
        raw_qmax = (brief.writer_plan or {}).get("question_count_max")
        try:
            contract_question_count_max = max(0, min(1, int(raw_qmax)))
        except Exception:
            contract_question_count_max = None
    if contract_question_count_max is not None:
        if _count_question_clauses(t) > contract_question_count_max:
            return False, "contract_question_count_exceeded"
    requires_question_like_preface = (
        action in {MainAction.QUESTION, MainAction.ASK_PERMISSION_TO_SHARE_INFO}
        and affirm_mode == AffirmationMode.NONE
        and not allow_questionless
    )
    if requires_question_like_preface:
        preface_clause = _extract_question_preface_clause(t)
        if not preface_clause:
            return False, "question_preface_missing"
        grounding_hint = ""
        if isinstance(brief, ResponseBrief):
            grounding_hint = _sanitize_human_text(brief.question_reflect_seed)
        if not grounding_hint:
            grounding_hint = _sanitize_human_text(change_talk_inference)
        if grounding_hint and grounding_hint != "none":
            if not _contains_change_talk_focus_reference(preface_clause, grounding_hint):
                return False, "question_preface_not_grounded_in_change_talk"
    if affirm_mode == AffirmationMode.NONE and _contains_affirmation(t):
        return False, "affirmation_unexpected"
    return True, "ok"

def _render_writer_action_guardrails(
    *,
    action: MainAction,
    reflection_style: Optional[ReflectionStyle],
    state: Optional[DialogueState] = None,
    closing_phase_complete: bool = False,
    focus_choice_context: Optional[Mapping[str, Any]] = None,
) -> str:
    if (
        closing_phase_complete
        and _is_reflect_action(action)
        and isinstance(state, DialogueState)
        and state.phase == Phase.CLOSING
    ):
        style_line = "- ReflectionStyle(complex): work with only one proposition from utterance_target."
        if reflection_style == ReflectionStyle.SIMPLE:
            style_line = "- ReflectionStyle(simple): briefly restate only one explicit point."
        return _get_layer4_writer_prompt_formatted(
            "action_guardrails",
            "closing_final_turn",
            default=(
                "[Main-action guardrails: CLOSING final turn / explicit ending]\n"
                "- Sentence 1 is a short reflection and sentence 2 explicitly closes the session.\n"
                "- Do not ask a question.\n"
                "- The closing line must include both a session anchor such as today / this turn / session and an ending verb such as end / finish / close.\n"
                "- Do not turn it into advice, suggestion, or a forced choice.\n"
                "{style_line}"
            ),
            style_line=style_line,
        )
    if _is_reflect_action(action) and isinstance(state, DialogueState) and _is_closing_first_turn(state):
        style_line = "- ReflectionStyle(complex): work with only one proposition from utterance_target."
        if reflection_style == ReflectionStyle.SIMPLE:
            style_line = "- ReflectionStyle(simple): briefly restate only one explicit point."
        return _get_layer4_writer_prompt_formatted(
            "action_guardrails",
            "closing_reflect_template",
            default=(
                "[Main-action guardrails: CLOSING first turn / REFLECT + end-check question]\n"
                "- Sentence 1 is a short reflection and sentence 2 is a question checking whether to end.\n"
                "- The only question must be the final one, asking whether it is okay to end today's session here.\n"
                "- Do not turn it into advice, suggestion, or a forced choice.\n"
                "{style_line}"
            ),
            style_line=style_line,
        )

    if _is_reflect_action(action):
        style_line = ""
        if reflection_style == ReflectionStyle.SIMPLE:
            style_line = "- Reflection style (simple): briefly restate only one point from utterance_target."
        elif reflection_style == ReflectionStyle.DOUBLE_SIDED:
            style_line = "- Reflection style (double): place the sustain side first, then the utterance_target side, and end on the change side."
        else:
            style_line = "- Reflection style (complex): work with only one proposition from utterance_target."
        return _get_layer4_writer_prompt_formatted(
            "action_guardrails",
            "reflect_template",
            default=(
                "[Main-action guardrails: REFLECT]\n"
                "- The job is reflection / reframing, not questioning, advising, or forcing a choice.\n"
                "- Keep reflection to one sentence, at most two.\n"
                "- Add one small implied meaning (value, ambivalence, or intention) rather than ending as a pure paraphrase.\n"
                "- Avoid repeating the same opener or ending as the immediately previous turn.\n"
                "- Do not end with a question mark.\n"
                "- Avoid advisory language such as 'let's', 'you should', or other prescriptive phrasing.\n"
                "- Do not pressure the client into a choice.\n"
                "{style_line}"
            ),
            style_line=style_line,
        )
    if action == MainAction.QUESTION:
        if isinstance(state, DialogueState) and _is_scale_followup_pending(state):
            followup_step = _current_scale_followup_step(state)
            score_text = _format_scale_score_text(state.scale_followup_score)
            if followup_step == _SCALE_FOLLOWUP_STEP_PLUS_ONE:
                return _get_layer4_writer_prompt_formatted(
                    "action_guardrails",
                    "question_followup_plus_one_template",
                    default=(
                        "[Main-action guardrails: QUESTION / scaling follow-up]\n"
                        "- Briefly acknowledge the previous answer in one sentence before the question.\n"
                        "- Ask exactly one question.\n"
                        "- Ask what would move the score from {score_text} up by one point.\n"
                        "- Do not stack a reason question in the same turn.\n"
                        "- Do not mix in advice or instruction."
                    ),
                    score_text=score_text,
                )
            if (state.scale_followup_score or 0.0) > 0.0:
                return _get_layer4_writer_prompt_formatted(
                    "action_guardrails",
                    "question_followup_reason_template",
                    default=(
                        "[Main-action guardrails: QUESTION / scaling follow-up]\n"
                        "- Briefly acknowledge the previous answer in one sentence before the question.\n"
                        "- Ask exactly one question.\n"
                        "- Ask why it is {score_text} rather than 0.\n"
                        "- Do not stack the plus-one question in the same turn.\n"
                        "- Do not mix in advice or instruction."
                    ),
                    score_text=score_text,
                )
            return _get_layer4_writer_prompt(
                "action_guardrails",
                "question_followup_default",
                default=(
                    "[Main-action guardrails: QUESTION / scaling follow-up]\n"
                    "- Briefly acknowledge the previous answer in one sentence before the question.\n"
                    "- Ask exactly one question.\n"
                    "- Ask what would raise it by one point.\n"
                    "- Do not mix in advice or instruction."
                ),
            )
        return _get_layer4_writer_prompt(
            "action_guardrails",
            "question_default",
            default=(
                "[Main-action guardrails: QUESTION]\n"
                "- Keep the default question structure fixed.\n"
                "- If fixed_affirm_mode != NONE, sentence 1 is affirmation and sentence 2 is the question about utterance_target.\n"
                "- If fixed_affirm_mode == NONE, sentence 1 is a simple reflection grounded in question_reflect_seed and sentence 2 is the question about utterance_target.\n"
                "- Only the final sentence may be a question, and it must not include advice or instruction."
            ),
        )
    if action == MainAction.SCALING_QUESTION:
        return _get_layer4_writer_prompt(
            "action_guardrails",
            "scaling_question",
            default=(
                "[Main-action guardrails: SCALING_QUESTION]\n"
                "- Explicitly ask on a 0-10 scale in one sentence and one question.\n"
                "- Ask only for the number; do not add reason or plus-one probes in the same question.\n"
                "- Keep the scale anchored to the concrete behavior rather than abstracting it.\n"
                "- Do not mix in advice or instruction."
            ),
        )
    if action == MainAction.SUMMARY:
        return _get_layer4_writer_prompt(
            "action_guardrails",
            "summary",
            default=(
                "[Main-action guardrails: SUMMARY]\n"
                "- Follow the Layer3 action_rules for summary structure, sentence count, and ending. Do not add Layer4-only structure.\n"
                "- Do not drift away from the purpose of main_action."
            ),
        )
    if action == MainAction.CLARIFY_PREFERENCE:
        guardrail = _get_layer4_writer_prompt(
            "action_guardrails",
            "clarify_preference",
            default=(
                "[Main-action guardrails: CLARIFY_PREFERENCE]\n"
                "- Sentence 1 is a brief reflection; sentence 2 is a binary-choice question.\n"
                "- If the client is already leaning to one side, reflect that side first and present it first in the choice.\n"
                "- Keep each option label short and easy to read.\n"
                "- Ask in a clean A-or-B form rather than repeating long paired phrases.\n"
                "- Do not re-ask permission.\n"
                "- Do not add concrete suggestions here."
            ),
        )
        clarify_preference_mode, focus_choice_options = _resolve_clarify_preference_mode(
            action=action,
            focus_choice_context=focus_choice_context,
        )
        if clarify_preference_mode == "focus_choice" and focus_choice_options:
            focus_choice_append = _get_layer4_writer_prompt_formatted(
                "action_guardrails",
                "clarify_preference_focus_choice",
                default=(
                    "- In this case, ask the client to choose the entry topic.\n"
                    "- Present the options briefly, like {focus_choice_options}, so it is obvious which topic each option refers to."
                ),
                focus_choice_options=_join_writer_items(
                    focus_choice_options,
                    fallback="none",
                ),
            )
            if focus_choice_append:
                guardrail = f"{guardrail}\n{focus_choice_append}"
        return guardrail
    if action == MainAction.ASK_PERMISSION_TO_SHARE_INFO:
        return _get_layer4_writer_prompt(
            "action_guardrails",
            "ask_permission",
            default=(
                "[Main-action guardrails: ASK_PERMISSION_TO_SHARE_INFO]\n"
                "- Follow Layer3 action_rules for structure, sentence count, and question placement. Do not add Layer4-only structure.\n"
                "- If add_affirm != NONE, sentence 1 is affirmation and sentence 2 asks only whether sharing information is okay.\n"
                "- If add_affirm == NONE, sentence 1 is a simple reflection grounded in question_reflect_seed and sentence 2 asks only whether sharing information is okay.\n"
                "- Do not include the concrete advice content yet.\n"
                "- Only the final sentence may be a question, and do not use pushy wording."
            ),
        )
    if action == MainAction.PROVIDE_INFO:
        return _get_layer4_writer_prompt(
            "action_guardrails",
            "provide_info",
            default=(
                "[Main-action guardrails: PROVIDE_INFO]\n"
                "- Share short, neutral information and include either explicit options or phrasing such as 'as one option'.\n"
                "- End with one question that asks for the client's reaction.\n"
                "- Avoid pushy wording."
            ),
        )
    return _get_layer4_writer_prompt(
        "action_guardrails",
        "unknown",
        default=(
            "[Main-action guardrails]\n"
            "- Output only the response text in a form that matches the current main action."
        ),
    )


# ----------------------------
# Orchestrator (minimal implementation)
# ----------------------------
@dataclass
class MIRhythmBot:
    llm: LLMClient
    cfg: PlannerConfig = field(default_factory=PlannerConfig)
    state: DialogueState = field(default_factory=DialogueState)
    history: List[Tuple[str, str]] = field(default_factory=list)

    action_ranker: Optional[ActionRanker] = None
    affirmation_decider: Optional[AffirmationDecider] = None
    feature_extractor: Optional[FeatureExtractor] = None
    phase_slot_filler: Optional[PhaseSlotFiller] = None
    phase_slot_filler_current: Optional[PhaseSlotFiller] = None
    phase_slot_filler_non_current: Optional[PhaseSlotFiller] = None
    slot_reviewer: Optional[SlotReviewer] = None
    slot_reviewer_non_current: Optional[SlotReviewer] = None
    change_talk_inferer: Optional[ChangeTalkInferer] = None
    response_integrator: Optional[ResponseIntegrator] = None
    risk_detector: Optional[RiskDetector] = field(default_factory=RuleBasedRiskDetector)
    output_evaluator: Optional[OutputEvaluator] = None

    # Threshold for automatic regeneration when the MI-adherence score is low (None means log only)
    evaluation_rewrite_threshold: Optional[float] = None
    writer_history_max_turns: int = 120
    layer4_temperature: float = 0.2
    layer4_enabled: bool = True

    def __post_init__(self) -> None:
        # Default to the LLM FeatureExtractor. If it fails, fall back internally to rule-based extraction.
        if self.feature_extractor is None:
            self.feature_extractor = LLMFeatureExtractor(llm=self.llm, temperature=0.0)
        if self.phase_slot_filler_current is None and self.phase_slot_filler is not None:
            if isinstance(self.phase_slot_filler, LLMPhaseSlotFiller):
                self.phase_slot_filler_current = dataclasses.replace(
                    self.phase_slot_filler,
                    scope_mode=_SLOT_FILL_SCOPE_CURRENT_ONLY,
                )
            else:
                self.phase_slot_filler_current = self.phase_slot_filler
        if self.phase_slot_filler_non_current is None and isinstance(self.phase_slot_filler, LLMPhaseSlotFiller):
            self.phase_slot_filler_non_current = dataclasses.replace(
                self.phase_slot_filler,
                scope_mode=_SLOT_FILL_SCOPE_NON_CURRENT_ONLY,
            )
        # By default, add_affirm decisions use the LLM's three-mode score estimate.
        if self.affirmation_decider is None:
            self.affirmation_decider = LLMAffirmationDecider(llm=self.llm, temperature=0.0)
        # Slot review also defaults to the LLM and falls back to the rule reviewer only on runtime error.
        if self.slot_reviewer is None:
            self.slot_reviewer = LLMSlotReviewer(llm=self.llm, temperature=0.0)
        if self.slot_reviewer_non_current is None:
            self.slot_reviewer_non_current = LLMNonCurrentSlotReviewer(llm=self.llm, temperature=0.0)

    def reset(self) -> None:
        """
        Reset state for switching to a new session.
        - Reinitialize DialogueState
        - Clear the utterance history
        """
        self.state = DialogueState()
        self.history.clear()

    def step(self, user_text: str) -> Tuple[str, Decision]:
        # 1) Update history (user)
        self.history.append(("user", user_text))
        layer3_json_mode = _normalize_json_mode(getattr(self.cfg, "layer3_json_mode", "loose"))

        # 1.2) Lightweight first-turn detection (greeting-only vs greeting-plus-topic)
        first_turn_hint: Optional[FirstTurnHint] = None
        first_turn_hint_debug: Optional[Dict[str, bool]] = None
        if self.state.turn_index == 0:
            first_turn_hint, first_turn_hint_debug = detect_first_turn_hint(user_text)

        # 1.5) Safety check (force a phase override if risk is high)
        risk_assessment: Optional[RiskAssessment] = None
        if self.risk_detector is not None:
            try:
                risk_assessment = self.risk_detector.detect(
                    user_text=user_text,
                    history=self.history,
                    state=self.state,
                )
            except Exception as e:
                risk_assessment = RiskAssessment(level=self.state.risk_level, reason=f"risk_detector_error:{e}")

        if risk_assessment:
            self.state.risk_level = risk_assessment.level
            self.state.last_risk_reason = risk_assessment.reason
            if risk_assessment.level == RiskLevel.HIGH:
                # In emergencies, disable information-sharing mode and prioritize safety guidance.
                self.state.info_mode = InfoMode.NONE

        crisis_override = bool(risk_assessment and risk_assessment.level == RiskLevel.HIGH)

        # 2) Three-layer pipeline
        # Layer1: estimate features + fill slots (in parallel)
        feature_debug: Dict[str, Any] = {}
        phase_debug: Dict[str, Any] = {"method": "heuristic"}
        ranker_debug: Optional[Dict[str, Any]] = None
        ranker_ordered_actions: List[MainAction] = []
        ranker_proposed_main_action: Optional[MainAction] = None
        change_talk_inference: str = ""
        change_talk_candidates: List[ChangeTalkCandidate] = []
        change_talk_debug: Dict[str, Any] = {"method": "not_started"}
        affirm_mode: Optional[AffirmationMode] = None
        affirm_mode_decider_debug: Optional[Dict[str, Any]] = None
        affirm_debug: Dict[str, Any] = {
            "method": "llm_affirmation_decider" if self.affirmation_decider is not None else "rule",
            "predicted_affirm_mode": None,
            "predicted_add_affirm": None,
        }

        layer1_slot_fill_debug: Dict[str, Any] = {"method": "disabled"}
        applied_phase_slots_debug: Dict[str, Any] = {"applied": False, "reason": "not_executed"}
        history_snapshot = list(self.history)
        state_snapshot = dataclasses.replace(self.state)
        phase_before_step = self.state.phase

        if crisis_override:
            features = extract_features_rule(user_text, self.state, self.cfg)
            feature_debug = {"method": "risk_override_rule"}
            change_talk_inference = ""
            change_talk_candidates = []
            change_talk_debug = {
                "method": "risk_override_empty",
                "suppressed_by_risk_override": True,
                "output": change_talk_inference,
                "focus_candidates": [],
            }
            prev_phase = self.state.phase
            self.state.phase = Phase.CLOSING
            self.state.phase_turns = 0 if self.state.phase != prev_phase else self.state.phase_turns
            phase_debug = {
                "method": "risk_override",
                "risk_level": risk_assessment.level.value if risk_assessment else RiskLevel.HIGH.value,
                "risk_reason": risk_assessment.reason if risk_assessment else None,
                "parallel_with_action_ranker": False,
            }
            affirm_mode = AffirmationMode.NONE
            affirm_debug = {
                "method": "risk_override",
                "predicted_affirm_mode": AffirmationMode.NONE.value,
                "predicted_add_affirm": False,
                "parallel_with_action_ranker": False,
            }
        else:
            # Layer1: feature estimation + slot filling
            layer1_json_mode = _normalize_json_mode(getattr(self.cfg, "layer1_json_mode", "loose"))
            layer2_json_mode = _normalize_json_mode(getattr(self.cfg, "layer2_json_mode", "loose"))
            feature_result: Optional[Tuple[PlannerFeatures, Dict[str, Any]]] = None
            feature_error: Optional[str] = None
            slot_fill_result: Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = None
            slot_fill_error: Optional[str] = None
            slot_fill_current_result: Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = None
            slot_fill_current_error: Optional[str] = None
            slot_fill_non_current_result: Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = None
            slot_fill_non_current_error: Optional[str] = None
            current_slot_filler = self.phase_slot_filler_current
            non_current_slot_filler = self.phase_slot_filler_non_current
            if current_slot_filler is None and self.phase_slot_filler is not None:
                current_slot_filler = self.phase_slot_filler
            for filler in (current_slot_filler, non_current_slot_filler):
                if filler is not None and hasattr(filler, "json_mode"):
                    try:
                        setattr(filler, "json_mode", layer1_json_mode)
                    except Exception:
                        pass

            has_any_slot_filler = (
                current_slot_filler is not None
                or non_current_slot_filler is not None
            )

            with ThreadPoolExecutor(
                max_workers=(
                    1
                    + (1 if current_slot_filler is not None else 0)
                    + (1 if non_current_slot_filler is not None else 0)
                )
            ) as executor:
                feature_future = (
                    executor.submit(
                        self.feature_extractor.extract,
                        user_text=user_text,
                        state=state_snapshot,
                        cfg=self.cfg,
                        history=history_snapshot,
                    )
                    if self.feature_extractor is not None
                    else executor.submit(
                        lambda: (
                            extract_features_rule(user_text, state_snapshot, self.cfg),
                            {"method": "no_feature_extractor_fallback"},
                        )
                    )
                )
                slot_fill_current_future = (
                    executor.submit(
                        current_slot_filler.fill_slots,
                        history=history_snapshot,
                        state=state_snapshot,
                        user_text=user_text,
                    )
                    if current_slot_filler is not None
                    else None
                )
                slot_fill_non_current_future = (
                    executor.submit(
                        non_current_slot_filler.fill_slots,
                        history=history_snapshot,
                        state=state_snapshot,
                        user_text=user_text,
                    )
                    if non_current_slot_filler is not None
                    else None
                )

                try:
                    feature_result = feature_future.result()
                except Exception as e:
                    feature_error = str(e)

                if slot_fill_current_future is not None:
                    try:
                        slot_fill_current_result = slot_fill_current_future.result()
                    except Exception as e:
                        slot_fill_current_error = str(e)

                if slot_fill_non_current_future is not None:
                    try:
                        slot_fill_non_current_result = slot_fill_non_current_future.result()
                    except Exception as e:
                        slot_fill_non_current_error = str(e)

            if feature_result is not None and feature_error is None:
                features, feature_debug = feature_result
            else:
                features = extract_features_rule(user_text, self.state, self.cfg)
                feature_debug = {
                    "method": "feature_extractor_error_fallback",
                    "error": feature_error or "feature_result_missing",
                }

            merged_slot_updates: List[Dict[str, Any]] = []
            scope_current_debug_map: Mapping[str, Any] = {}
            scope_non_current_debug_map: Mapping[str, Any] = {}
            merged_slot_fill_debug: Dict[str, Any] = {
                "method": "parallel_phase_slot_filler",
                "feature_parallel_kept": True,
                "scope_current_enabled": bool(current_slot_filler is not None),
                "scope_non_current_enabled": bool(non_current_slot_filler is not None),
                "scope_current_error": slot_fill_current_error,
                "scope_non_current_error": slot_fill_non_current_error,
            }
            if slot_fill_current_result is not None and slot_fill_current_error is None:
                current_updates, current_debug = slot_fill_current_result
                merged_slot_updates = _merge_phase_slot_updates(merged_slot_updates, current_updates)
                merged_slot_fill_debug["scope_current_updates"] = current_updates
                merged_slot_fill_debug["scope_current_debug"] = current_debug
                if isinstance(current_debug, Mapping):
                    scope_current_debug_map = current_debug
            if slot_fill_non_current_result is not None and slot_fill_non_current_error is None:
                non_current_updates, non_current_debug = slot_fill_non_current_result
                merged_slot_updates = _merge_phase_slot_updates(merged_slot_updates, non_current_updates)
                merged_slot_fill_debug["scope_non_current_updates"] = non_current_updates
                merged_slot_fill_debug["scope_non_current_debug"] = non_current_debug
                if isinstance(non_current_debug, Mapping):
                    scope_non_current_debug_map = non_current_debug

            if (
                (slot_fill_current_result is not None and slot_fill_current_error is None)
                or (slot_fill_non_current_result is not None and slot_fill_non_current_error is None)
            ):
                current_need_previous_requested = _coerce_bool_with_ja(
                    scope_current_debug_map.get("need_previous_phase_slot_updates"),
                    False,
                )
                current_need_future_requested = _coerce_bool_with_ja(
                    scope_current_debug_map.get("need_future_phase_slot_updates"),
                    False,
                )
                non_current_need_previous_requested = _coerce_bool_with_ja(
                    scope_non_current_debug_map.get("need_previous_phase_slot_updates"),
                    False,
                )
                non_current_need_future_requested = _coerce_bool_with_ja(
                    scope_non_current_debug_map.get("need_future_phase_slot_updates"),
                    False,
                )
                non_current_scope_success = (
                    non_current_slot_filler is not None
                    and slot_fill_non_current_result is not None
                    and slot_fill_non_current_error is None
                )
                if non_current_scope_success:
                    current_phase_idx = _phase_index(self.state.phase)
                    need_previous_updates = False
                    need_future_updates = False
                    for update in merged_slot_updates:
                        if not isinstance(update, Mapping):
                            continue
                        phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
                        if phase is None:
                            continue
                        phase_idx = _phase_index(phase)
                        if phase_idx < current_phase_idx:
                            need_previous_updates = True
                        elif phase_idx > current_phase_idx:
                            need_future_updates = True
                else:
                    need_previous_updates = bool(current_need_previous_requested or non_current_need_previous_requested)
                    need_future_updates = bool(current_need_future_requested or non_current_need_future_requested)
                merged_slot_fill_debug["need_previous_phase_slot_updates_requested"] = bool(
                    current_need_previous_requested or non_current_need_previous_requested
                )
                merged_slot_fill_debug["need_future_phase_slot_updates_requested"] = bool(
                    current_need_future_requested or non_current_need_future_requested
                )
                merged_slot_fill_debug["need_previous_phase_slot_updates"] = need_previous_updates
                merged_slot_fill_debug["need_future_phase_slot_updates"] = need_future_updates
                merged_slot_fill_debug["include_non_current_updates_in_layer2_review"] = non_current_scope_success
                if "focus_choice_hint" in scope_current_debug_map:
                    merged_slot_fill_debug["focus_choice_hint"] = _normalize_focus_choice_hint(
                        scope_current_debug_map.get("focus_choice_hint")
                    )
                if "layer1_prompt_target_slot_keys" in scope_current_debug_map:
                    merged_slot_fill_debug["layer1_prompt_target_slot_keys"] = scope_current_debug_map.get(
                        "layer1_prompt_target_slot_keys"
                    )
                if "target_phase_updates" in scope_current_debug_map:
                    merged_slot_fill_debug["target_phase_updates"] = scope_current_debug_map.get(
                        "target_phase_updates"
                    )
                if "target_phase_updates" not in merged_slot_fill_debug:
                    current_phase_name = self.state.phase.name
                    current_phase_updates: List[Dict[str, Any]] = []
                    for update in merged_slot_updates:
                        if not isinstance(update, Mapping):
                            continue
                        phase = _parse_phase_from_any(update.get("phase_code") or update.get("phase"))
                        if phase is None or phase.name != current_phase_name:
                            continue
                        current_phase_updates.append(dict(update))
                    merged_slot_fill_debug["target_phase_updates"] = current_phase_updates
                slot_fill_result = (merged_slot_updates, merged_slot_fill_debug)
            elif has_any_slot_filler:
                slot_fill_error = (
                    " / ".join(
                        item
                        for item in (
                            slot_fill_current_error,
                            slot_fill_non_current_error,
                        )
                        if item
                    )
                    or "slot_fill_result_missing"
                )

            if slot_fill_result is not None and slot_fill_error is None:
                slot_updates, slot_fill_debug = slot_fill_result
                layer1_bundle, layer1_bundle_debug = _build_layer1_slot_bundle(
                    history=history_snapshot,
                    state=self.state,
                    slot_updates=slot_updates,
                    slot_fill_debug=slot_fill_debug if isinstance(slot_fill_debug, Mapping) else {},
                )
                applied_phase_slots_debug = {
                    "applied": False,
                    "reason": "deferred_until_slot_review",
                    "reviewed_count": 0,
                    "applied_count": 0,
                }
                layer1_slot_fill_debug = {
                    "method": "phase_slot_filler_layer1",
                    "slot_fill_debug": slot_fill_debug,
                    "slot_updates": slot_updates,
                    "layer1_bundle": dataclasses.asdict(layer1_bundle),
                    "layer1_bundle_summary": layer1_bundle_debug,
                    "allowed_phase_codes_for_apply": layer1_bundle_debug.get("allowed_phase_codes_for_apply"),
                    "dropped_updates_non_allowed_phase": layer1_bundle_debug.get("dropped_updates_non_allowed_phase"),
                    "dropped_updates_unknown_phase": layer1_bundle_debug.get("dropped_updates_unknown_phase"),
                    "applied_phase_slots": applied_phase_slots_debug,
                }
            elif not has_any_slot_filler:
                layer1_slot_fill_debug = {"method": "no_phase_slot_filler"}
                applied_phase_slots_debug = {"applied": False, "reason": "no_phase_slot_filler"}
                layer1_bundle, layer1_bundle_debug = _build_layer1_slot_bundle(
                    history=history_snapshot,
                    state=self.state,
                    slot_updates=[],
                    slot_fill_debug={},
                )
                layer1_slot_fill_debug["layer1_bundle"] = dataclasses.asdict(layer1_bundle)
                layer1_slot_fill_debug["layer1_bundle_summary"] = layer1_bundle_debug
                layer1_slot_fill_debug["allowed_phase_codes_for_apply"] = layer1_bundle_debug.get("allowed_phase_codes_for_apply")
                layer1_slot_fill_debug["dropped_updates_non_allowed_phase"] = layer1_bundle_debug.get("dropped_updates_non_allowed_phase")
                layer1_slot_fill_debug["dropped_updates_unknown_phase"] = layer1_bundle_debug.get("dropped_updates_unknown_phase")
            else:
                layer1_slot_fill_debug = {
                    "method": "phase_slot_filler_error",
                    "error": slot_fill_error or "slot_fill_result_missing",
                }
                applied_phase_slots_debug = {"applied": False, "reason": "phase_slot_filler_error"}
                layer1_bundle, layer1_bundle_debug = _build_layer1_slot_bundle(
                    history=history_snapshot,
                    state=self.state,
                    slot_updates=[],
                    slot_fill_debug={},
                )
                layer1_slot_fill_debug["layer1_bundle"] = dataclasses.asdict(layer1_bundle)
                layer1_slot_fill_debug["layer1_bundle_summary"] = layer1_bundle_debug
                layer1_slot_fill_debug["allowed_phase_codes_for_apply"] = layer1_bundle_debug.get("allowed_phase_codes_for_apply")
                layer1_slot_fill_debug["dropped_updates_non_allowed_phase"] = layer1_bundle_debug.get("dropped_updates_non_allowed_phase")
                layer1_slot_fill_debug["dropped_updates_unknown_phase"] = layer1_bundle_debug.get("dropped_updates_unknown_phase")

            current_review_bundle, non_current_review_bundle = _split_layer1_bundle_for_layer2(
                layer1_bundle,
                current_phase=self.state.phase,
            )
            layer1_slot_fill_debug["layer2_current_review_bundle"] = dataclasses.asdict(current_review_bundle)
            layer1_slot_fill_debug["layer2_non_current_review_bundle"] = dataclasses.asdict(non_current_review_bundle)
            focus_choice_context = _focus_choice_context_from_layer1_bundle(layer1_bundle)
            layer1_slot_fill_debug["focus_choice_hint"] = focus_choice_context

            # Update state used by later decisions, based on Layer1 features.
            cleaned_user = re.sub(r"\s+", "", user_text)
            if (not features.is_short_reply and len(cleaned_user) >= 6) or (
                features.change_talk >= 0.5 and len(cleaned_user) >= 3
            ):
                self.state.last_substantive_user_text = user_text
            if self.state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is True:
                self.state.info_mode = InfoMode.READY_TO_PROVIDE
            if self.state.info_mode == InfoMode.WAITING_PERMISSION and features.has_permission is False:
                self.state.info_mode = InfoMode.NONE

            layer2_state_snapshot = dataclasses.replace(self.state)
            provisional_action_mask = compute_allowed_actions(
                state=layer2_state_snapshot,
                features=features,
                cfg=self.cfg,
                first_turn_hint=first_turn_hint,
                focus_choice_context=focus_choice_context,
            )
            provisional_allowed_actions = _normalize_ranker_action_space(
                provisional_action_mask.get("allowed_actions")  # type: ignore[arg-type]
            )
            # Preserve the action-mask priority order even in the ranker input.
            provisional_allowed_actions_for_ranker = list(provisional_allowed_actions)
            provisional_allowed_actions_for_ranker = _prioritize_ask_permission_for_ranker(
                actions=provisional_allowed_actions_for_ranker,
                state=layer2_state_snapshot,
                features=features,
                info_gate_open=bool(provisional_action_mask.get("info_share_phase_gate_open")),
            )
            provisional_mask_steps_raw = provisional_action_mask.get("mask_steps")
            if isinstance(provisional_mask_steps_raw, Sequence) and not isinstance(
                provisional_mask_steps_raw,
                (str, bytes),
            ):
                provisional_mask_steps = [
                    str(step).strip()
                    for step in provisional_mask_steps_raw
                    if str(step).strip()
                ]
            else:
                provisional_mask_steps = []
            ranker_skip_reason: Optional[str] = None
            for skip_step in (
                "permission_ready_provide_info_only",
                "waiting_permission_unclear_clarify_only",
                "review_second_turn_reflect_complex_only",
                "closing_first_turn_reflect_simple_or_complex_only",
                "scale_followup_pending_question_only",
            ):
                if skip_step in provisional_mask_steps:
                    ranker_skip_reason = skip_step
                    break

            # Layer2: current/non-current slot review + main_action choice + affirmation decision + change-talk inference (parallel)
            phase_before_layer2 = self.state.phase
            has_current_slot_reviewer = self.slot_reviewer is not None
            has_non_current_slot_reviewer = self.slot_reviewer_non_current is not None
            has_slot_reviewer = has_current_slot_reviewer or has_non_current_slot_reviewer
            has_action_ranker = self.action_ranker is not None
            should_run_action_ranker = has_action_ranker and ranker_skip_reason is None
            has_change_talk_inferer = self.change_talk_inferer is not None
            should_run_current_slot_reviewer = bool(current_review_bundle.review_target_slot_keys)
            should_run_non_current_slot_reviewer = bool(non_current_review_bundle.review_target_slot_keys)
            should_run_slot_reviewer = should_run_current_slot_reviewer or should_run_non_current_slot_reviewer
            parallel_enabled = should_run_slot_reviewer and should_run_action_ranker

            current_slot_review_result: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
            current_slot_review_error: Optional[str] = None
            non_current_slot_review_result: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
            non_current_slot_review_error: Optional[str] = None
            rank_result: Optional[Tuple[List[MainAction], Dict[str, Any]]] = None
            rank_error: Optional[str] = None
            affirm_error: Optional[str] = None
            change_talk_result: Optional[Tuple[Any, Dict[str, Any]]] = None
            change_talk_error: Optional[str] = None
            non_current_slot_review_future = None
            non_current_slot_review_executor: Optional[ThreadPoolExecutor] = None
            strict_layer2_json = layer2_json_mode == "strict"

            def _run_current_slot_reviewer() -> Tuple[Dict[str, Any], Dict[str, Any]]:
                reviewer = self.slot_reviewer or RuleBasedSlotReviewer()
                if hasattr(reviewer, "json_mode"):
                    try:
                        setattr(reviewer, "json_mode", layer2_json_mode)
                    except Exception:
                        pass
                try:
                    reviewed, reviewed_debug = reviewer.review(
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                        layer1_bundle=current_review_bundle,
                    )
                except Exception as e:
                    fallback_reviewed, fallback_debug = RuleBasedSlotReviewer().review(
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                        layer1_bundle=current_review_bundle,
                    )
                    fallback_debug = dict(fallback_debug or {})
                    fallback_debug["fallback_from"] = "slot_reviewer_exception"
                    fallback_debug["fallback_error"] = str(e)
                    return fallback_reviewed, fallback_debug

                reviewed_updates = reviewed.get("reviewed_updates") if isinstance(reviewed, Mapping) else None
                has_reviewed_updates = (
                    isinstance(reviewed_updates, Sequence)
                    and not isinstance(reviewed_updates, (str, bytes))
                    and len(reviewed_updates) > 0
                )
                if (
                    isinstance(reviewer, LLMSlotReviewer)
                    and bool(reviewed_debug.get("allow_rule_fallback", True))
                    and not has_reviewed_updates
                    and bool(current_review_bundle.candidate_updates)
                ):
                    fallback_reviewed, fallback_debug = RuleBasedSlotReviewer().review(
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                        layer1_bundle=current_review_bundle,
                    )
                    fallback_debug = dict(fallback_debug or {})
                    fallback_debug["fallback_from"] = "llm_empty_reviewed_updates"
                    return fallback_reviewed, fallback_debug

                return reviewed, reviewed_debug

            def _run_non_current_slot_reviewer() -> Tuple[Dict[str, Any], Dict[str, Any]]:
                reviewer = self.slot_reviewer_non_current or RuleBasedNonCurrentSlotReviewer()
                if hasattr(reviewer, "json_mode"):
                    try:
                        setattr(reviewer, "json_mode", layer2_json_mode)
                    except Exception:
                        pass
                try:
                    reviewed, reviewed_debug = reviewer.review(
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                        layer1_bundle=non_current_review_bundle,
                    )
                except Exception as e:
                    fallback_reviewed, fallback_debug = RuleBasedNonCurrentSlotReviewer().review(
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                        layer1_bundle=non_current_review_bundle,
                    )
                    fallback_debug = dict(fallback_debug or {})
                    fallback_debug["fallback_from"] = "non_current_slot_reviewer_exception"
                    fallback_debug["fallback_error"] = str(e)
                    return fallback_reviewed, fallback_debug

                reviewed_updates = reviewed.get("reviewed_updates") if isinstance(reviewed, Mapping) else None
                has_reviewed_updates = (
                    isinstance(reviewed_updates, Sequence)
                    and not isinstance(reviewed_updates, (str, bytes))
                    and len(reviewed_updates) > 0
                )
                if (
                    isinstance(reviewer, LLMNonCurrentSlotReviewer)
                    and bool(reviewed_debug.get("allow_rule_fallback", True))
                    and not has_reviewed_updates
                    and bool(non_current_review_bundle.candidate_updates)
                ):
                    fallback_reviewed, fallback_debug = RuleBasedNonCurrentSlotReviewer().review(
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                        layer1_bundle=non_current_review_bundle,
                    )
                    fallback_debug = dict(fallback_debug or {})
                    fallback_debug["fallback_from"] = "llm_empty_reviewed_updates"
                    return fallback_reviewed, fallback_debug

                return reviewed, reviewed_debug

            def _resolve_non_current_slot_review_result(
                *,
                raw_result: Optional[Tuple[Dict[str, Any], Dict[str, Any]]],
                error: Optional[str],
                execution_timing: str,
            ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[Dict[str, Any]], Dict[str, Any]]:
                if not should_run_non_current_slot_reviewer:
                    payload = _empty_slot_review_payload()
                    debug_map = {
                        "method": "slot_reviewer_skipped_no_non_current_review_targets",
                        "request_label": "layer2:slot_reviewer_non_current",
                        "reason": "no_non_current_review_target_slot_keys",
                        "review_target_slot_keys": list(non_current_review_bundle.review_target_slot_keys),
                        "candidate_count": len(non_current_review_bundle.candidate_updates),
                        "json_mode": layer2_json_mode,
                        "strict_json_mode": strict_layer2_json,
                    }
                    staged_debug = {
                        "applied": False,
                        "applied_count": 0,
                        "pending_count": 0,
                        "staged_count": 0,
                        "reviewed_count": 0,
                        "apply_mode": "pending_non_current",
                        "update_results": [],
                        "reason": "no_non_current_review_target_slot_keys",
                        "source": "non_current_slot_reviewer",
                        "review_method": debug_map["method"],
                    }
                    return payload, debug_map, [], [], staged_debug

                schema_issues: List[str] = []
                if raw_result is not None and error is None:
                    raw_non_current_slot_review, raw_non_current_slot_review_debug = raw_result
                    payload, schema_issues = _normalize_slot_review_payload(
                        payload=raw_non_current_slot_review,
                        layer1_bundle=non_current_review_bundle,
                        allow_slot_quality_target_examples=False,
                        validate_current_phase_slot_repair_hints=False,
                        require_current_phase_slot_repair_hints=False,
                    )
                    payload["slot_quality_target_examples"] = []
                    payload["slot_repair_hints"] = []
                    debug_map = dict(raw_non_current_slot_review_debug or {})
                    if schema_issues:
                        debug_map["schema_issues"] = schema_issues
                    debug_map["json_mode"] = layer2_json_mode
                else:
                    payload = _empty_slot_review_payload()
                    debug_map = {
                        "method": "non_current_slot_reviewer_error_skip",
                        "request_label": "layer2:slot_reviewer_non_current",
                        "error": error or "non_current_slot_review_result_missing",
                        "json_mode": layer2_json_mode,
                        "strict_json_mode": strict_layer2_json,
                    }

                debug_map["execution_timing"] = execution_timing
                debug_map["waited_before_generation"] = (
                    execution_timing == "awaited_before_layer3_layer4_on_phase_transition"
                )
                debug_map["waited_after_generation"] = (
                    execution_timing == "deferred_parallel_with_layer3_layer4"
                )

                reviewed_updates_raw = payload.get("reviewed_updates")
                reviewed_updates = (
                    list(reviewed_updates_raw)
                    if isinstance(reviewed_updates_raw, Sequence)
                    and not isinstance(reviewed_updates_raw, (str, bytes))
                    else []
                )
                staged_debug = _apply_reviewed_phase_slot_updates_to_state(
                    state=self.state,
                    reviewed_updates=reviewed_updates,
                    apply_mode="pending_non_current",
                    review_source="layer2_non_current",
                    source_phase=phase_before_layer2,
                )
                staged_debug["reason"] = (
                    "non_current_slot_review_updates_staged"
                    if int(staged_debug.get("staged_count") or 0) > 0
                    else "non_current_slot_review_no_updates"
                )
                staged_debug["source"] = "non_current_slot_reviewer"
                staged_debug["review_method"] = debug_map.get("method")
                staged_debug["execution_timing"] = execution_timing
                return payload, debug_map, schema_issues, reviewed_updates, staged_debug

            with ThreadPoolExecutor(
                max_workers=(
                    (1 if should_run_current_slot_reviewer else 0)
                    + (1 if should_run_action_ranker else 0)
                    + 1
                    + (1 if has_change_talk_inferer else 0)
                )
            ) as executor:
                current_slot_review_future = (
                    executor.submit(_run_current_slot_reviewer)
                    if should_run_current_slot_reviewer
                    else None
                )
                def _run_action_ranker() -> Tuple[List[MainAction], Dict[str, Any]]:
                    if self.action_ranker is None:
                        return [], {"error": "action_ranker_missing"}
                    if _action_ranker_supports_allowed_actions(self.action_ranker):
                        return self.action_ranker.rank(
                            history=history_snapshot,
                            state=layer2_state_snapshot,
                            features=features,
                            allowed_actions=tuple(provisional_allowed_actions_for_ranker),
                        )
                    return self.action_ranker.rank(
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                    )

                rank_future = (
                    executor.submit(_run_action_ranker)
                    if should_run_action_ranker
                    else None
                )

                def _run_change_talk_inferer() -> Tuple[Any, Dict[str, Any]]:
                    if self.change_talk_inferer is None:
                        return "", {"error": "change_talk_inferer_missing"}
                    return self.change_talk_inferer.infer(
                        user_text=user_text,
                        history=history_snapshot,
                        state=layer2_state_snapshot,
                        features=features,
                    )

                def _run_affirmation_decider() -> Tuple[AffirmationMode, Dict[str, Any]]:
                    if self.affirmation_decider is not None:
                        return self.affirmation_decider.decide(
                            user_text=user_text,
                            history=history_snapshot,
                            state=layer2_state_snapshot,
                            features=features,
                        )
                    return _decide_affirm_mode(
                        user_text=user_text,
                        features=features,
                        state=layer2_state_snapshot,
                    )

                change_talk_future = (
                    executor.submit(_run_change_talk_inferer)
                    if self.change_talk_inferer is not None
                    else None
                )
                affirm_future = executor.submit(_run_affirmation_decider)

                if current_slot_review_future is not None:
                    try:
                        current_slot_review_result = current_slot_review_future.result()
                    except Exception as e:
                        current_slot_review_error = str(e)

                if rank_future is not None:
                    try:
                        rank_result = rank_future.result()
                    except Exception as e:
                        rank_error = str(e)

                if change_talk_future is not None:
                    try:
                        change_talk_result = change_talk_future.result()
                    except Exception as e:
                        change_talk_error = str(e)

                try:
                    affirm_mode, affirm_mode_decider_debug = affirm_future.result()
                except Exception as e:
                    affirm_error = str(e)

            review_schema_issues: List[str] = []
            non_current_review_schema_issues: List[str] = []
            if not should_run_current_slot_reviewer:
                slot_review_payload = _empty_slot_review_payload()
                slot_review_debug = {
                    "method": "slot_reviewer_skipped_no_review_targets",
                    "request_label": "layer2:slot_reviewer",
                    "reason": "no_current_review_target_slot_keys",
                    "review_target_slot_keys": list(current_review_bundle.review_target_slot_keys),
                    "candidate_count": len(current_review_bundle.candidate_updates),
                    "json_mode": layer2_json_mode,
                    "strict_json_mode": strict_layer2_json,
                }
            elif current_slot_review_result is not None and current_slot_review_error is None:
                raw_slot_review, raw_slot_review_debug = current_slot_review_result
                slot_review_payload, review_schema_issues = _normalize_slot_review_payload(
                    payload=raw_slot_review,
                    layer1_bundle=current_review_bundle,
                )
                slot_review_debug = dict(raw_slot_review_debug or {})
                if review_schema_issues:
                    slot_review_debug["schema_issues"] = review_schema_issues
                slot_review_debug["json_mode"] = layer2_json_mode
            else:
                slot_review_payload = _empty_slot_review_payload()
                slot_review_debug = {
                    "method": "slot_reviewer_error_skip",
                    "request_label": "layer2:slot_reviewer",
                    "error": current_slot_review_error or "slot_review_result_missing",
                    "json_mode": layer2_json_mode,
                    "strict_json_mode": strict_layer2_json,
                }
            if should_run_non_current_slot_reviewer:
                non_current_slot_review_payload = _empty_slot_review_payload()
                non_current_slot_review_debug = {
                    "method": "slot_reviewer_deferred_until_post_phase_decision",
                    "request_label": "layer2:slot_reviewer_non_current",
                    "reason": "deferred_until_phase_transition_check",
                    "review_target_slot_keys": list(non_current_review_bundle.review_target_slot_keys),
                    "candidate_count": len(non_current_review_bundle.candidate_updates),
                    "json_mode": layer2_json_mode,
                    "strict_json_mode": strict_layer2_json,
                    "execution_timing": "deferred_until_phase_transition_check",
                    "waited_before_generation": False,
                    "waited_after_generation": False,
                }
                non_current_reviewed_updates: List[Dict[str, Any]] = []
                staged_non_current_phase_slots_debug = {
                    "applied": False,
                    "applied_count": 0,
                    "pending_count": 0,
                    "staged_count": 0,
                    "reviewed_count": 0,
                    "apply_mode": "pending_non_current",
                    "update_results": [],
                    "reason": "deferred_until_phase_transition_check",
                    "source": "non_current_slot_reviewer",
                    "review_method": non_current_slot_review_debug["method"],
                    "execution_timing": "deferred_until_phase_transition_check",
                }
            else:
                (
                    non_current_slot_review_payload,
                    non_current_slot_review_debug,
                    non_current_review_schema_issues,
                    non_current_reviewed_updates,
                    staged_non_current_phase_slots_debug,
                ) = _resolve_non_current_slot_review_result(
                    raw_result=None,
                    error=None,
                    execution_timing="skipped_no_non_current_review_targets",
                )

            reviewed_updates_raw = slot_review_payload.get("reviewed_updates")
            reviewed_updates = (
                list(reviewed_updates_raw)
                if isinstance(reviewed_updates_raw, Sequence) and not isinstance(reviewed_updates_raw, (str, bytes))
                else []
            )
            applied_phase_slots_debug = _apply_reviewed_phase_slot_updates_to_state(
                state=self.state,
                reviewed_updates=reviewed_updates,
                apply_mode="commit",
                review_source="layer2_current",
                source_phase=phase_before_layer2,
            )
            applied_phase_slots_debug["reason"] = (
                "slot_review_updates_applied"
                if int(applied_phase_slots_debug.get("reviewed_count") or 0) > 0
                else "slot_review_no_updates"
            )
            applied_phase_slots_debug["source"] = "slot_reviewer"
            applied_phase_slots_debug["review_method"] = slot_review_debug.get("method")

            if isinstance(layer1_slot_fill_debug, dict):
                layer1_slot_fill_debug["applied_phase_slots"] = applied_phase_slots_debug
                layer1_slot_fill_debug["staged_non_current_phase_slots"] = staged_non_current_phase_slots_debug

            slot_gate_debug = evaluate_phase_slot_readiness(
                state=self.state,
                cfg=self.cfg,
                user_text=user_text,
                features=features,
                slot_review_debug=slot_review_payload,
                phase_slot_meta=self.state.phase_slot_meta,
            )
            enforced_phase = _parse_phase_from_any(slot_gate_debug.get("selected_phase")) or self.state.phase
            phase_transition_feedback = _build_phase_transition_feedback_from_review(
                current_phase=self.state.phase,
                selected_phase=enforced_phase,
                slot_gate_debug=slot_gate_debug,
                slot_review_debug=slot_review_payload,
            )
            phase_debug = {
                "method": "slot_review_rule_gate",
                "phase_decision_owner": "rule_after_slot_review",
                "parallel_with_slot_reviewer": should_run_slot_reviewer,
                "parallel_with_current_slot_reviewer": should_run_current_slot_reviewer,
                "parallel_with_non_current_slot_reviewer": should_run_non_current_slot_reviewer,
                "slot_reviewer_enabled": has_slot_reviewer,
                "current_slot_reviewer_enabled": has_current_slot_reviewer,
                "non_current_slot_reviewer_enabled": has_non_current_slot_reviewer,
                "slot_reviewer_skipped_no_targets": not should_run_slot_reviewer,
                "parallel_with_action_ranker": should_run_action_ranker,
                "action_ranker_enabled": has_action_ranker,
                "action_ranker_skipped_reason": ranker_skip_reason,
                "parallel_enabled": parallel_enabled,
                "layer1_slot_fill": layer1_slot_fill_debug,
                "layer1_slot_bundle_summary": layer1_bundle_debug,
                "applied_phase_slots": applied_phase_slots_debug,
                "staged_non_current_phase_slots": staged_non_current_phase_slots_debug,
                "slot_review": slot_review_payload,
                "slot_review_debug": slot_review_debug,
                "non_current_slot_review": non_current_slot_review_payload,
                "non_current_slot_review_debug": non_current_slot_review_debug,
                "reviewed_phase_slots": [
                    dict(item)
                    for item in [*reviewed_updates, *non_current_reviewed_updates]
                    if isinstance(item, Mapping)
                ],
                "reviewed_current_phase_slots": [
                    dict(item) for item in reviewed_updates if isinstance(item, Mapping)
                ],
                "reviewed_non_current_phase_slots": [
                    dict(item) for item in non_current_reviewed_updates if isinstance(item, Mapping)
                ],
                "slot_gate": slot_gate_debug,
                "predicted_phase": enforced_phase.value,
                "enforce": {
                    "decision": slot_gate_debug.get("decision"),
                    "current": self.state.phase.value,
                    "predicted": enforced_phase.value,
                },
            }
            if review_schema_issues:
                phase_debug["slot_review_schema_issues"] = review_schema_issues
            if non_current_review_schema_issues:
                phase_debug["non_current_slot_review_schema_issues"] = non_current_review_schema_issues
            phase_debug.update(phase_transition_feedback)

            last_action_before_phase = self.state.last_actions[-1] if self.state.last_actions else None
            if (
                _is_scale_followup_incomplete(self.state)
                and _is_scaling_question_phase(self.state.phase)
                and enforced_phase != self.state.phase
            ):
                phase_debug["phase_override_for_scale_followup"] = {
                    "from_phase": self.state.phase.value,
                    "predicted_phase": enforced_phase.value,
                    "forced_phase": self.state.phase.value,
                    "reason": "keep_phase_until_scale_followup_question",
                    "last_action": (last_action_before_phase.value if isinstance(last_action_before_phase, MainAction) else None),
                    "pending_step": _normalize_scale_followup_step(self.state.scale_followup_pending_step) or None,
                    "last_asked_step": _normalize_scale_followup_step(self.state.scale_followup_last_asked_step) or None,
                }
                phase_debug["predicted_phase"] = self.state.phase.value
                phase_debug["phase_transition_code"] = "stay_by_scale_followup_override"
                phase_debug["phase_transition_note"] = "Stay in the current phase until the post-scaling follow-up is completed."
                enforce_info = phase_debug.get("enforce")
                if isinstance(enforce_info, dict):
                    enforce_info["decision"] = "stay_by_scale_followup_override"
                    enforce_info["predicted"] = self.state.phase.value
                enforced_phase = self.state.phase

            if (
                self.state.info_mode == InfoMode.WAITING_PERMISSION
                and features.has_permission is None
                and _is_info_share_phase_gate_open(self.state.phase)
                and enforced_phase != self.state.phase
                and not crisis_override
            ):
                phase_debug["phase_override_for_waiting_permission"] = {
                    "from_phase": self.state.phase.value,
                    "predicted_phase": enforced_phase.value,
                    "forced_phase": self.state.phase.value,
                    "reason": "keep_phase_until_permission_is_resolved",
                }
                phase_debug["predicted_phase"] = self.state.phase.value
                phase_debug["phase_transition_code"] = "stay_by_waiting_permission"
                phase_debug["phase_transition_note"] = "Stay in the current phase because permission for information sharing is not resolved yet."
                enforce_info = phase_debug.get("enforce")
                if isinstance(enforce_info, dict):
                    enforce_info["decision"] = "stay_by_waiting_permission"
                    enforce_info["predicted"] = self.state.phase.value
                enforced_phase = self.state.phase

            if (
                features.user_requests_info
                and enforced_phase != self.state.phase
                and not crisis_override
            ):
                phase_debug["phase_override_for_info_request"] = {
                    "from_phase": self.state.phase.value,
                    "predicted_phase": enforced_phase.value,
                    "forced_phase": self.state.phase.value,
                    "reason": "keep_phase_until_info_request_is_handled",
                }
                phase_debug["predicted_phase"] = self.state.phase.value
                phase_debug["phase_transition_code"] = "stay_by_info_request"
                phase_debug["phase_transition_note"] = "Stay in the current phase so the request for information can be handled first."
                enforce_info = phase_debug.get("enforce")
                if isinstance(enforce_info, dict):
                    enforce_info["decision"] = "stay_by_info_request"
                    enforce_info["predicted"] = self.state.phase.value
                enforced_phase = self.state.phase

            if enforced_phase != self.state.phase:
                self.state.phase_turns = 0
            self.state.phase = enforced_phase

            if affirm_error is None:
                decider_method = _normalize_slot_text(
                    affirm_mode_decider_debug.get("method")
                    if isinstance(affirm_mode_decider_debug, Mapping)
                    else ""
                )
                affirm_debug = {
                    "method": decider_method or (
                        "parallel_rule" if (should_run_slot_reviewer or should_run_action_ranker) else "rule"
                    ),
                    "request_label": (
                        str(affirm_mode_decider_debug.get("request_label", "") or "")
                        if isinstance(affirm_mode_decider_debug, Mapping)
                        else ""
                    ),
                    "predicted_affirm_mode": (
                        affirm_mode.value if isinstance(affirm_mode, AffirmationMode) else None
                    ),
                    "predicted_add_affirm": bool(affirm_mode),
                    "decider_debug": affirm_mode_decider_debug,
                    "rule_debug": affirm_mode_decider_debug,
                    "parallel_with_slot_reviewer": should_run_slot_reviewer,
                    "parallel_with_action_ranker": should_run_action_ranker,
                    "action_ranker_enabled": has_action_ranker,
                    "action_ranker_skipped_reason": ranker_skip_reason,
                }
            else:
                affirm_mode = AffirmationMode.NONE
                affirm_debug = {
                    "method": "affirm_decider_error_fallback",
                    "request_label": "layer2:affirmation_decider",
                    "error": affirm_error,
                    "predicted_affirm_mode": AffirmationMode.NONE.value,
                    "predicted_add_affirm": False,
                    "parallel_with_slot_reviewer": should_run_slot_reviewer,
                    "parallel_with_action_ranker": should_run_action_ranker,
                    "action_ranker_enabled": has_action_ranker,
                    "action_ranker_skipped_reason": ranker_skip_reason,
                }

            if change_talk_result is not None and change_talk_error is None:
                inferred_payload, inferred_debug = change_talk_result
                change_talk_debug = dict(inferred_debug or {})
                change_talk_debug.setdefault("method", "parallel")
                direction_debug = (
                    change_talk_debug.get("direction_debug")
                    if isinstance(change_talk_debug.get("direction_debug"), Mapping)
                    else {}
                )
                linked_slots_hint = (
                    direction_debug.get("selected_slot_keys")
                    if isinstance(direction_debug, Mapping)
                    else []
                )
                if not isinstance(linked_slots_hint, Sequence) or isinstance(linked_slots_hint, (str, bytes)):
                    linked_slots_hint = []
                history_end_turn = len(history_snapshot)
                history_start_turn = max(1, history_end_turn - 10 + 1)
                default_text = _normalize_slot_text(change_talk_debug.get("output"))
                change_talk_candidates, change_talk_inference = _normalize_change_talk_infer_output(
                    inferred_payload,
                    user_text=user_text,
                    history_start_turn=history_start_turn,
                    history_end_turn=history_end_turn,
                    linked_slots_hint=[str(item) for item in linked_slots_hint if str(item).strip()],
                    default_text=default_text,
                )
                if not change_talk_inference:
                    change_talk_inference = _build_change_talk_hint_from_candidates(
                        candidates=change_talk_candidates,
                        current_phase=self.state.phase,
                    )
                change_talk_debug["used_fallback"] = False
                change_talk_debug["parallel_with_slot_reviewer"] = should_run_slot_reviewer
                change_talk_debug["parallel_with_action_ranker"] = should_run_action_ranker
                change_talk_debug["action_ranker_enabled"] = has_action_ranker
                change_talk_debug["action_ranker_skipped_reason"] = ranker_skip_reason
                change_talk_debug["parallel_with_affirmation_decider"] = True
                change_talk_debug["focus_candidates"] = [
                    dataclasses.asdict(candidate) for candidate in change_talk_candidates
                ]
                change_talk_debug["output"] = change_talk_inference
            else:
                change_talk_inference = ""
                change_talk_candidates = []
                change_talk_debug = {
                    "method": "change_talk_inferer_error_empty" if has_change_talk_inferer else "change_talk_inferer_disabled_empty",
                    "error": change_talk_error if has_change_talk_inferer else None,
                    "output": change_talk_inference,
                    "focus_candidates": [],
                    "used_fallback": False,
                    "parallel_with_slot_reviewer": should_run_slot_reviewer,
                    "parallel_with_action_ranker": should_run_action_ranker,
                    "action_ranker_enabled": has_action_ranker,
                    "action_ranker_skipped_reason": ranker_skip_reason,
                    "parallel_with_affirmation_decider": True,
                }

            if should_run_action_ranker:
                if rank_result is not None and rank_error is None:
                    ranker_ordered_actions, ranker_debug = rank_result
                    cleaned: List[MainAction] = []
                    seen = set()
                    for a in ranker_ordered_actions:
                        if not isinstance(a, MainAction):
                            continue
                        if a in seen:
                            continue
                        seen.add(a)
                        cleaned.append(a)
                    ranker_ordered_actions = cleaned
                    if ranker_debug is None:
                        ranker_debug = {}
                    ranker_debug["parallel_with_slot_reviewer"] = should_run_slot_reviewer
                    ranker_debug["provisional_allowed_actions"] = [a.value for a in provisional_allowed_actions]
                    ranker_debug["provisional_allowed_actions_for_ranker"] = [
                        a.value for a in provisional_allowed_actions_for_ranker
                    ]
                    ranker_debug["provisional_action_mask"] = provisional_action_mask
                    proposed_action_raw = ranker_debug.get("proposed_main_action")
                    parsed_action = coerce_main_action(proposed_action_raw)
                    if parsed_action is None and ranker_ordered_actions:
                        parsed_action = ranker_ordered_actions[0]
                    if parsed_action is not None:
                        ranker_proposed_main_action = (
                            MainAction.REFLECT_COMPLEX
                            if parsed_action == MainAction.REFLECT
                            else parsed_action
                        )
                else:
                    ranker_ordered_actions = []
                    ranker_debug = {
                        "error": rank_error or "rank_result_missing",
                        "parallel_with_slot_reviewer": should_run_slot_reviewer,
                    }
            elif has_action_ranker:
                ranker_ordered_actions = []
                ranker_debug = {
                    "method": "action_ranker_skipped",
                    "skip_reason": ranker_skip_reason or "action_ranker_not_scheduled",
                    "parallel_with_slot_reviewer": should_run_slot_reviewer,
                    "provisional_allowed_actions": [a.value for a in provisional_allowed_actions],
                    "provisional_allowed_actions_for_ranker": [
                        a.value for a in provisional_allowed_actions_for_ranker
                    ],
                    "provisional_action_mask": provisional_action_mask,
                }

        if (
            self.state.turn_index == 0
            and self.state.phase != Phase.GREETING
            and not crisis_override
        ):
            phase_before_guardrail = self.state.phase
            phase_debug["forced_first_turn_phase"] = phase_before_guardrail.value
            phase_debug["predicted_phase"] = Phase.GREETING.value
            phase_debug["phase_decision_owner"] = "first_turn_guardrail"
            enforce_info = phase_debug.get("enforce")
            if not isinstance(enforce_info, dict):
                enforce_info = {}
                phase_debug["enforce"] = enforce_info
            enforce_info["decision"] = "forced_first_turn_greeting"
            enforce_info["current"] = phase_before_guardrail.value
            enforce_info["predicted"] = Phase.GREETING.value
            phase_debug["phase_transition_code"] = "forced_first_turn_greeting"
            phase_debug["phase_transition_note"] = "Stay in the current phase because the first turn should prioritize the greeting phase."
            self.state.phase = Phase.GREETING
            self.state.phase_turns = 0

        if (
            self.state.turn_index >= 1
            and self.state.phase == Phase.GREETING
            and not crisis_override
        ):
            phase_before_guardrail = self.state.phase
            phase_debug["forced_post_greeting_phase"] = phase_before_guardrail.value
            phase_debug["predicted_phase"] = Phase.PURPOSE_CONFIRMATION.value
            phase_debug["phase_decision_owner"] = "post_greeting_guardrail"
            enforce_info = phase_debug.get("enforce")
            if not isinstance(enforce_info, dict):
                enforce_info = {}
                phase_debug["enforce"] = enforce_info
            enforce_info["decision"] = "forced_after_greeting_purpose_confirmation"
            enforce_info["current"] = phase_before_guardrail.value
            enforce_info["predicted"] = Phase.PURPOSE_CONFIRMATION.value
            phase_debug["phase_transition_code"] = "forced_post_greeting_purpose_confirmation"
            phase_debug["phase_transition_note"] = "New phase started: purpose confirmation after the greeting guardrail."
            self.state.phase = Phase.PURPOSE_CONFIRMATION
            self.state.phase_turns = 0

        pending_promotion_debug: Dict[str, Any] = {
            "promoted": False,
            "reason": "phase_unchanged",
            "phase_before": phase_before_step.value,
            "phase_after": self.state.phase.value,
        }
        if (
            not crisis_override
            and self.state.phase != phase_before_step
        ):
            pending_promotion_debug = _promote_pending_phase_slot_updates_on_phase_entry(
                state=self.state,
                entered_phase=self.state.phase,
            )
            pending_promotion_debug["phase_before"] = phase_before_step.value
            pending_promotion_debug["phase_after"] = self.state.phase.value
        if isinstance(phase_debug, dict):
            phase_debug["promoted_non_current_phase_slots"] = pending_promotion_debug
        if isinstance(layer1_slot_fill_debug, dict):
            layer1_slot_fill_debug["promoted_non_current_phase_slots"] = pending_promotion_debug

        phase_transitioned_this_turn = _did_phase_transition_for_current_turn(phase_debug)
        if phase_transitioned_this_turn and isinstance(phase_debug, dict):
            post_transition_slot_gate = evaluate_phase_slot_readiness(
                state=self.state,
                cfg=self.cfg,
                user_text=user_text,
                features=features,
                slot_review_debug=None,
                phase_slot_meta=self.state.phase_slot_meta,
            )
            quality_min_threshold = _clamp01(
                post_transition_slot_gate.get("quality_min_threshold"),
                self.cfg.phase_slot_quality_min_threshold,
            )
            post_transition_slot_focus_priority = _derive_slot_focus_priority_from_slot_gate(
                phase=self.state.phase,
                slot_gate=post_transition_slot_gate,
                quality_min_threshold=quality_min_threshold,
            )
            phase_debug["post_transition_slot_gate"] = post_transition_slot_gate
            phase_debug["slot_focus_priority"] = post_transition_slot_focus_priority
            phase_debug["slot_quality_target_examples"] = []
            phase_debug["slot_repair_hints"] = []
            phase_debug["phase_focus_resync"] = {
                "applied": True,
                "phase": self.state.phase.value,
                "phase_code": self.state.phase.name,
                "slot_focus_priority": post_transition_slot_focus_priority,
            }
            if (
                self.state.scale_followup_pending_phase is not None
                or self.state.scale_followup_score is not None
                or _normalize_scale_followup_step(self.state.scale_followup_pending_step)
                or _normalize_scale_followup_step(self.state.scale_followup_last_asked_step)
            ):
                phase_debug["scale_followup_cleared_by_phase_transition"] = {
                    "pending_phase_before": (
                        self.state.scale_followup_pending_phase.value
                        if isinstance(self.state.scale_followup_pending_phase, Phase)
                        else None
                    ),
                    "score_before": self.state.scale_followup_score,
                    "pending_step_before": _normalize_scale_followup_step(self.state.scale_followup_pending_step) or None,
                    "last_asked_step_before": _normalize_scale_followup_step(self.state.scale_followup_last_asked_step) or None,
                }
            _clear_scale_followup_runtime_state(self.state)

        scale_followup_debug: Dict[str, Any] = {"updated": False}
        if not crisis_override:
            scale_followup_debug = _update_scale_followup_state_from_user_turn(self.state, user_text)
            scale_followup_debug["updated"] = True
        if isinstance(phase_debug, dict):
            phase_debug["scale_followup_debug"] = scale_followup_debug

        action_mask_debug: Dict[str, Any] = {}
        review_first_turn = _is_review_reflection_first_turn(self.state)
        review_second_turn = _is_review_reflection_second_turn(self.state)
        review_info_request_priority = (
            self.state.phase == Phase.REVIEW_REFLECTION
            and not review_first_turn
            and bool(features.user_requests_info)
        )
        closing_first_turn = _is_closing_first_turn(self.state)
        if crisis_override:
            action = MainAction.PROVIDE_INFO
            debug = {"sampling": "crisis_override", "risk_level": risk_assessment.level.value}
            action_mask_debug = {
                "allowed_actions": [MainAction.PROVIDE_INFO.value],
                "fallback_action": MainAction.PROVIDE_INFO.value,
                "mask_steps": ["risk_override_provide_info_only"],
            }
        else:
            action_mask_debug = compute_allowed_actions(
                state=self.state,
                features=features,
                cfg=self.cfg,
                first_turn_hint=first_turn_hint,
                focus_choice_context=focus_choice_context,
            )
            allowed_actions = _normalize_ranker_action_space(
                action_mask_debug.get("allowed_actions")  # type: ignore[arg-type]
            )
            allowed_set = set(allowed_actions)
            fallback_action = allowed_actions[0]

            debug = {
                "sampling": "deterministic",
                "allowed_actions": [action_item.value for action_item in allowed_actions],
                "action_mask": action_mask_debug,
            }

            ranker_candidate = ranker_proposed_main_action
            if ranker_candidate is None and ranker_ordered_actions:
                ranker_candidate = ranker_ordered_actions[0]
            if ranker_candidate == MainAction.REFLECT:
                ranker_candidate = MainAction.REFLECT_COMPLEX

            ranker_confidence: Optional[float] = None
            if isinstance(ranker_debug, Mapping):
                conf_raw = ranker_debug.get("confidence")
                if conf_raw not in (None, ""):
                    parsed_conf = _clamp01(conf_raw, -1.0)
                    if parsed_conf >= 0.0:
                        ranker_confidence = parsed_conf
                        debug["ranker_confidence"] = round(parsed_conf, 3)

            if ranker_proposed_main_action is not None:
                debug["ranker_proposed_main_action"] = ranker_proposed_main_action.value
            if ranker_candidate is not None:
                debug["ranker_directive_main_action"] = ranker_candidate.value

            if ranker_candidate is None:
                action = fallback_action
                debug["action_source"] = "mask_rule_fallback"
                debug["ranker_proposal_applied"] = "no_ranker_action_fallback"
            elif ranker_candidate not in allowed_set:
                action = fallback_action
                debug["action_source"] = "allowed_action_mask_fallback"
                debug["invalid_action"] = True
                debug["invalid_ranker_action"] = ranker_candidate.value
                debug["ranker_proposal_applied"] = "fallback_to_allowed_head"
            else:
                action = ranker_candidate
                debug["action_source"] = "ranker_masked_directive"
                debug["ranker_proposal_applied"] = ranker_candidate.value.lower()

            focus_choice_override, focus_choice_override_debug = _should_force_clarify_preference_for_focus_choice(
                state=self.state,
                action=action,
                allowed_actions=allowed_actions,
                focus_choice_context=focus_choice_context,
            )
            debug["focus_choice_override"] = focus_choice_override_debug
            if focus_choice_override:
                action = MainAction.CLARIFY_PREFERENCE
                debug["action_source"] = "focus_choice_override"
                debug["ranker_proposal_applied"] = "focus_choice_override_from_question"

            if review_first_turn and action != MainAction.QUESTION:
                debug["review_first_turn_main_action_forced"] = {
                    "from_action": action.value,
                    "to_action": MainAction.QUESTION.value,
                    "reason": "reflect_summary_question_bundle",
                }
                action = MainAction.QUESTION
            if review_second_turn and action != MainAction.REFLECT_COMPLEX:
                if not review_info_request_priority:
                    debug["review_second_turn_main_action_forced"] = {
                        "from_action": action.value,
                        "to_action": MainAction.REFLECT_COMPLEX.value,
                        "reason": "review_second_turn_reflect_complex_fixed",
                    }
                    action = MainAction.REFLECT_COMPLEX

            if self.state.phase == Phase.REVIEW_REFLECTION:
                debug["review_phase_override"] = {
                    "forced_action": action.value,
                    "reason": (
                        "review_first_turn_reflect_summary_question_bundle"
                        if review_first_turn
                        else (
                            "review_info_request_priority"
                            if review_info_request_priority
                            else (
                                "review_second_turn_reflect_complex_fixed"
                                if review_second_turn
                                else "feedback_after_user_review_response"
                            )
                        )
                    ),
                }
            if closing_first_turn:
                debug["closing_phase_override"] = {
                    "forced_affirm_mode": AffirmationMode.NONE.value,
                    "forced_output_shape": "reflect_then_closing_confirmation_question",
                    "main_action_policy": (
                        "ranker_select_reflect_simple_or_complex"
                        if should_run_action_ranker
                        else "mask_select_reflect_simple_or_complex"
                    ),
                }

            layer1_target_slot_miss = _detect_layer1_target_slot_miss(phase_debug)
            scaling_slot_key = _scaling_slot_key_for_phase(self.state.phase)
            target_slot_keys = {
                _slot_key_from_focus_text(item)
                for item in (layer1_target_slot_miss.get("target_slot_keys") or [])
                if _slot_key_from_focus_text(item)
            }
            should_force_scaling_by_layer1 = bool(
                not crisis_override
                and _is_scaling_question_phase(self.state.phase)
                and action != MainAction.SCALING_QUESTION
                and scaling_slot_key
                and scaling_slot_key in target_slot_keys
                and bool(layer1_target_slot_miss.get("detected"))
                and not _is_scale_followup_incomplete(self.state)
            )
            if should_force_scaling_by_layer1:
                debug["layer1_target_slot_miss_scaling_override"] = {
                    "applied": True,
                    "from_action": action.value,
                    "to_action": MainAction.SCALING_QUESTION.value,
                    "phase": self.state.phase.value,
                    "scaling_slot_key": scaling_slot_key,
                    "reason": layer1_target_slot_miss.get("reason"),
                    "target_slot_keys": list(layer1_target_slot_miss.get("target_slot_keys") or []),
                    "returned_slot_keys": list(layer1_target_slot_miss.get("returned_slot_keys") or []),
                    "matched_slot_keys": list(layer1_target_slot_miss.get("matched_slot_keys") or []),
                }
                action = MainAction.SCALING_QUESTION
            else:
                debug["layer1_target_slot_miss_scaling_override"] = {
                    "applied": False,
                    "phase": self.state.phase.value,
                    "scaling_slot_key": scaling_slot_key,
                    "reason": layer1_target_slot_miss.get("reason"),
                    "target_slot_keys": list(layer1_target_slot_miss.get("target_slot_keys") or []),
                    "returned_slot_keys": list(layer1_target_slot_miss.get("returned_slot_keys") or []),
                    "matched_slot_keys": list(layer1_target_slot_miss.get("matched_slot_keys") or []),
                    "scale_followup_incomplete": _is_scale_followup_incomplete(self.state),
                }
            if (
                not bool(action_mask_debug.get("info_share_phase_gate_open"))
                and self.state.info_mode != InfoMode.NONE
            ):
                self.state.info_mode = InfoMode.NONE
                debug["info_mode_reset_by_phase_gate"] = True

        # 5) Add affirmation (suppressed in crisis mode)
        if not crisis_override and affirm_mode is None:
            # Final fallback only when the parallel decision result is missing.
            try:
                if self.affirmation_decider is not None:
                    affirm_mode, affirm_mode_decider_debug = self.affirmation_decider.decide(
                        user_text=user_text,
                        history=self.history,
                        state=self.state,
                        features=features,
                    )
                else:
                    affirm_mode, affirm_mode_decider_debug = _decide_affirm_mode(
                        user_text=user_text,
                        features=features,
                        state=self.state,
                    )
                affirm_debug["fallback"] = "result_missing -> sequential_decider"
                affirm_debug["decider_debug"] = affirm_mode_decider_debug
                affirm_debug["rule_debug"] = affirm_mode_decider_debug
            except Exception as e:
                affirm_mode = AffirmationMode.NONE
                affirm_debug["fallback"] = "result_missing_error -> false"
                affirm_debug["fallback_error"] = str(e)

        add_affirm_layer2 = AffirmationMode.NONE if crisis_override else _normalize_affirmation_mode(affirm_mode)
        add_affirm = add_affirm_layer2
        if not crisis_override and closing_first_turn:
            add_affirm = AffirmationMode.NONE
            affirm_debug["forced_mode"] = "closing_first_turn_no_affirm"
            affirm_debug["forced_bundle"] = [
                "no_affirm",
                "reflect_simple_or_complex_by_ranker",
                "closing_confirmation_question",
            ]
        elif not crisis_override and self.state.phase == Phase.REVIEW_REFLECTION:
            if review_first_turn:
                add_affirm = AffirmationMode.NONE
                affirm_debug["forced_mode"] = "review_phase_question_turn"
                affirm_debug["forced_mode_detail"] = "review_phase_first_turn_no_affirm"
                affirm_debug["forced_bundle"] = [
                    "no_affirm",
                    "reflect_complex",
                    "summary",
                    "session_review_question",
                ]
            elif review_info_request_priority:
                affirm_debug["forced_mode"] = "review_phase_info_request_priority_no_forced_affirm"
            elif review_second_turn:
                add_affirm = AffirmationMode.COMPLEX
                affirm_debug["forced_mode"] = "review_phase_feedback_turn"
                affirm_debug["forced_mode_detail"] = "review_phase_second_turn_complex_affirm"
                affirm_debug["forced_bundle"] = [
                    "complex_affirm",
                    "reflect_complex",
                ]
            else:
                add_affirm = AffirmationMode.COMPLEX
                affirm_debug["forced_mode"] = "review_phase_feedback_turn"
        # 6) Next state (projected updates for this turn)
        reflection_style: Optional[ReflectionStyle] = None
        reflection_style_debug: Optional[Dict[str, Any]] = None
        if _is_reflect_action(action):
            reflection_style = _reflection_style_from_action(action)
            reflection_style_debug = {
                "selected_style": reflection_style.value,
                "selected_by": "ranker_main_action_or_mask_fallback",
            }

        last_action = self.state.last_actions[-1] if self.state.last_actions else None
        is_current_question_like = action in (
            MainAction.QUESTION,
            MainAction.SCALING_QUESTION,
            MainAction.ASK_PERMISSION_TO_SHARE_INFO,
        )
        consecutive_question_like = (
            is_current_question_like
            and last_action in _QUESTION_LIKE_ACTIONS
        )
        allow_question_without_preface = (
            is_current_question_like
            and features.is_short_reply
            and not consecutive_question_like
        )
        if consecutive_question_like:
            debug["question_preface_forced_by_consecutive_question"] = True
        if allow_question_without_preface and add_affirm != AffirmationMode.NONE:
            add_affirm = AffirmationMode.NONE
            affirm_debug["forced_mode"] = "short_reply_question_turn"
            affirm_debug["suppressed_by_short_reply_question"] = True
        affirm_debug["final_affirm_mode"] = add_affirm.value
        affirm_debug["final_add_affirm"] = bool(add_affirm)
        if crisis_override:
            affirm_debug["suppressed_by_risk_override"] = True

        if not isinstance(phase_debug, dict):
            phase_debug = dict(phase_debug or {})
        target_behavior_focus = _get_confirmed_target_behavior_for_change_talk(state=self.state)
        selected_focus_candidates = _select_change_talk_candidates_for_action(
            candidates=change_talk_candidates,
            max_items=2,
            current_phase=self.state.phase,
            target_behavior_focus=target_behavior_focus,
            prioritize_target_behavior=_should_prioritize_target_behavior_for_phase(self.state.phase),
            importance_estimate=features.importance_estimate,
        )
        slot_target = _derive_slot_target_from_phase_debug(
            phase_debug,
            focus_candidates=selected_focus_candidates,
        )
        slot_hint_source_debug: Optional[Mapping[str, Any]] = phase_debug
        if isinstance(phase_debug, Mapping) and not _did_phase_transition_for_current_turn(phase_debug):
            phase_debug_for_hint_collection = dict(phase_debug)
            removed_empty_root_hint = False
            for key in ("slot_quality_target_examples", "slot_repair_hints"):
                value = phase_debug_for_hint_collection.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 0:
                    phase_debug_for_hint_collection.pop(key, None)
                    removed_empty_root_hint = True
            if removed_empty_root_hint:
                # stay override can leave an empty root hint list behind even when Layer2 kept usable hints
                slot_hint_source_debug = phase_debug_for_hint_collection
        slot_quality_target_examples_for_turn = _collect_slot_quality_target_examples(
            phase_prediction_debug=slot_hint_source_debug,
            slot_target=slot_target,
            current_phase=self.state.phase,
        )
        phase_debug["slot_quality_target_examples"] = [
            dict(item) for item in slot_quality_target_examples_for_turn
        ]
        phase_debug["slot_repair_hints"] = [dict(item) for item in slot_quality_target_examples_for_turn]
        change_talk_inference = _build_change_talk_hint_from_candidates(
            candidates=selected_focus_candidates,
            current_phase=self.state.phase,
        )
        if not change_talk_inference:
            change_talk_inference = ""
        change_talk_debug["selected_focus_candidates"] = [
            dataclasses.asdict(candidate) for candidate in selected_focus_candidates
        ]
        change_talk_debug["target_behavior_focus"] = target_behavior_focus
        change_talk_debug["slot_target"] = slot_target
        change_talk_debug["selected_focus_hint"] = change_talk_inference

        next_state = apply_action_to_state(
            state=self.state,
            features=features,
            action=action,
            add_affirm=add_affirm,
            reflection_style=reflection_style,
        )

        phase_slots_snapshot = _copy_phase_slot_memory(self.state.phase_slots)
        current_phase_slots_snapshot = dict(phase_slots_snapshot.get(self.state.phase.name, {}))
        phase_slots_pending_snapshot = _copy_phase_slot_memory(self.state.phase_slot_pending)
        current_phase_slots_pending_snapshot = dict(phase_slots_pending_snapshot.get(self.state.phase.name, {}))

        decision = Decision(
            phase=self.state.phase,
            main_action=action,
            add_affirm=add_affirm,
            next_state=next_state,
            reflection_style=reflection_style,
            risk_mode=risk_assessment.level if risk_assessment else RiskLevel.NONE,
            focus_candidates=list(change_talk_candidates),
            slot_target=slot_target,
            debug={
                "features": dataclasses.asdict(features),
                "feature_debug": feature_debug,
                "change_talk_inference": change_talk_inference,
                "change_talk_focus_candidates": [
                    dataclasses.asdict(candidate) for candidate in change_talk_candidates
                ],
                "selected_focus_candidates": [
                    dataclasses.asdict(candidate) for candidate in selected_focus_candidates
                ],
                "change_talk_debug": change_talk_debug,
                "phase_debug": phase_debug,
                "phase_slots": phase_slots_snapshot,
                "current_phase_slots": current_phase_slots_snapshot,
                "phase_slots_pending": phase_slots_pending_snapshot,
                "current_phase_slots_pending": current_phase_slots_pending_snapshot,
                "ranker_debug": ranker_debug,
                "affirm_debug": affirm_debug,
                "first_turn_hint": first_turn_hint.value if first_turn_hint else None,
                "first_turn_hint_debug": first_turn_hint_debug,
                "scale_followup_debug": scale_followup_debug,
                "allow_question_without_preface": allow_question_without_preface,
                "risk_assessment": (
                    {
                        "level": risk_assessment.level.value,
                        "reason": risk_assessment.reason,
                        "raw_output": risk_assessment.raw_output,
                    }
                    if risk_assessment
                    else None
                ),
                "reflection_style_debug": reflection_style_debug,
                **debug,
            },
        )

        slot_gate_debug_raw = (
            phase_debug.get("slot_gate")
            if isinstance(phase_debug, Mapping)
            else None
        )
        slot_gate_debug = (
            dict(slot_gate_debug_raw)
            if isinstance(slot_gate_debug_raw, Mapping)
            else {}
        )
        phase_intent_effective = str(
            phase_debug.get("phase_intent_effective") or phase_debug.get("phase_intent") or ""
        ).strip().lower()
        closing_phase_complete = False
        if self.state.phase == Phase.CLOSING:
            slot_gate_phase = _parse_phase_from_any(slot_gate_debug.get("current_phase"))
            fill_rate_threshold = _clamp01(
                slot_gate_debug.get("fill_rate_threshold"),
                self.cfg.phase_slot_fill_rate_threshold,
            )
            fill_rate = _clamp01(slot_gate_debug.get("fill_rate"), 0.0)
            if "pass_fill" in slot_gate_debug:
                pass_fill = bool(slot_gate_debug.get("pass_fill"))
            else:
                pass_fill = fill_rate >= fill_rate_threshold

            min_turns_required_raw = slot_gate_debug.get("min_turns_required")
            try:
                min_turns_required = max(0, int(min_turns_required_raw))
            except Exception:
                min_turns_required = _min_turns_required_for_exit(self.state.phase)
            phase_turns_raw = slot_gate_debug.get("phase_turns")
            try:
                phase_turns = max(0, int(phase_turns_raw))
            except Exception:
                phase_turns = max(0, int(self.state.phase_turns))
            if "turn_gate_pass" in slot_gate_debug:
                turn_gate_pass = bool(slot_gate_debug.get("turn_gate_pass"))
            else:
                turn_gate_pass = phase_turns >= min_turns_required
            closing_max_turns_raw = getattr(self.cfg, "closing_max_turns", 3)
            try:
                closing_max_turns = max(1, int(closing_max_turns_raw))
            except Exception:
                closing_max_turns = 3
            force_close_by_max_turns = phase_turns >= closing_max_turns

            # In CLOSING completion checks, do not block ending only because of relationship signals such as high resistance or discord.
            # Keep those observations for debugging only.
            resistance_value = _clamp01(features.resistance, 0.0)
            discord_value = _clamp01(features.discord, 0.0)
            resistance_threshold = _clamp01(self.cfg.phase_hold_resistance_threshold, 0.6)
            discord_threshold = _clamp01(self.cfg.phase_hold_discord_threshold, 0.6)
            hold_reasons_raw = slot_gate_debug.get("relational_hold_reasons")
            hold_reasons: List[str] = []
            if isinstance(hold_reasons_raw, list):
                hold_reasons = [str(item) for item in hold_reasons_raw if str(item).strip()]
            if not hold_reasons:
                if resistance_value >= resistance_threshold:
                    hold_reasons.append("high_resistance")
                if discord_value >= discord_threshold:
                    hold_reasons.append("high_discord")
            pass_relational = True
            relational_hold_ignored = len(hold_reasons) > 0

            closing_phase_complete = bool(
                slot_gate_phase == Phase.CLOSING
                and (force_close_by_max_turns or (pass_fill and turn_gate_pass))
            )
            decision.debug["closing_completion_decision"] = {
                "decision": "advance" if closing_phase_complete else "stay",
                "by": "slot_fill_min_turns_or_closing_max_turns_gate",
                "slot_gate_phase": slot_gate_phase.value if isinstance(slot_gate_phase, Phase) else "",
                "phase_intent_effective": phase_intent_effective,
                "fill_rate": fill_rate,
                "fill_rate_threshold": fill_rate_threshold,
                "pass_fill": pass_fill,
                "phase_turns": phase_turns,
                "min_turns_required": min_turns_required,
                "turn_gate_pass": turn_gate_pass,
                "closing_max_turns": closing_max_turns,
                "force_close_by_max_turns": force_close_by_max_turns,
                "resistance": resistance_value,
                "resistance_threshold": resistance_threshold,
                "discord": discord_value,
                "discord_threshold": discord_threshold,
                "pass_relational": pass_relational,
                "relational_hold_ignored_for_closing": relational_hold_ignored,
                "relational_hold_reasons": hold_reasons,
            }

        if closing_phase_complete:
            closing_reflect_action = action
            if closing_reflect_action not in {MainAction.REFLECT_SIMPLE, MainAction.REFLECT_COMPLEX}:
                ranker_reflect_candidate = next(
                    (
                        candidate
                        for candidate in ranker_ordered_actions
                        if candidate in {MainAction.REFLECT_SIMPLE, MainAction.REFLECT_COMPLEX}
                    ),
                    None,
                )
                closing_reflect_action = ranker_reflect_candidate or MainAction.REFLECT_COMPLEX
                decision.debug["closing_final_reflect_action_override"] = {
                    "from_action": action.value,
                    "to_action": closing_reflect_action.value,
                    "reason": "closing_final_turn_requires_reflect_simple_or_complex",
                    "selected_by": (
                        "ranker_ordered_actions" if ranker_reflect_candidate is not None else "fallback_reflect_complex"
                    ),
                }

            final_add_affirm = add_affirm_layer2 if not crisis_override else AffirmationMode.NONE
            if final_add_affirm != add_affirm:
                decision.debug["closing_final_affirm_override"] = {
                    "from_affirm": add_affirm.value,
                    "to_affirm": final_add_affirm.value,
                    "reason": "closing_final_turn_uses_layer2_affirm_decision",
                }

            # Even final closing turns still pass through Layer3 / Layer4.
            # However, constrain the main action on the termination turn to reflection (simple/complex) only.
            action = closing_reflect_action
            add_affirm = final_add_affirm
            reflection_style = _reflection_style_from_action(action)
            next_state = apply_action_to_state(
                state=self.state,
                features=features,
                action=action,
                add_affirm=add_affirm,
                reflection_style=reflection_style,
            )
            is_current_question_like = action in (
                MainAction.QUESTION,
                MainAction.SCALING_QUESTION,
                MainAction.ASK_PERMISSION_TO_SHARE_INFO,
            )
            consecutive_question_like = (
                is_current_question_like
                and last_action in _QUESTION_LIKE_ACTIONS
            )
            allow_question_without_preface = (
                is_current_question_like
                and features.is_short_reply
                and not consecutive_question_like
            )
            decision.main_action = action
            decision.add_affirm = add_affirm
            decision.reflection_style = reflection_style
            decision.next_state = next_state
            decision.debug["allow_question_without_preface"] = allow_question_without_preface
            decision.debug["closing_final_generation_path"] = "layer3_layer4"

        def _refresh_phase_slot_debug_snapshots() -> None:
            phase_slots_snapshot_local = _copy_phase_slot_memory(self.state.phase_slots)
            phase_slots_pending_snapshot_local = _copy_phase_slot_memory(self.state.phase_slot_pending)
            decision.debug["phase_slots"] = phase_slots_snapshot_local
            decision.debug["current_phase_slots"] = dict(
                phase_slots_snapshot_local.get(self.state.phase.name, {})
            )
            decision.debug["phase_slots_pending"] = phase_slots_pending_snapshot_local
            decision.debug["current_phase_slots_pending"] = dict(
                phase_slots_pending_snapshot_local.get(self.state.phase.name, {})
            )

        def _refresh_non_current_phase_debug() -> None:
            phase_debug["staged_non_current_phase_slots"] = staged_non_current_phase_slots_debug
            phase_debug["non_current_slot_review"] = non_current_slot_review_payload
            phase_debug["non_current_slot_review_debug"] = non_current_slot_review_debug
            phase_debug["reviewed_phase_slots"] = [
                dict(item)
                for item in [*reviewed_updates, *non_current_reviewed_updates]
                if isinstance(item, Mapping)
            ]
            phase_debug["reviewed_non_current_phase_slots"] = [
                dict(item) for item in non_current_reviewed_updates if isinstance(item, Mapping)
            ]
            if non_current_review_schema_issues:
                phase_debug["non_current_slot_review_schema_issues"] = list(non_current_review_schema_issues)
            else:
                phase_debug.pop("non_current_slot_review_schema_issues", None)
            phase_debug["non_current_slot_review_timing"] = str(
                non_current_slot_review_debug.get("execution_timing", "") or ""
            )
            phase_debug["non_current_slot_review_waited_before_generation"] = bool(
                non_current_slot_review_debug.get("waited_before_generation")
            )
            phase_debug["non_current_slot_review_waited_after_generation"] = bool(
                non_current_slot_review_debug.get("waited_after_generation")
            )
            if isinstance(layer1_slot_fill_debug, dict):
                layer1_slot_fill_debug["staged_non_current_phase_slots"] = staged_non_current_phase_slots_debug
                layer1_slot_fill_debug["promoted_non_current_phase_slots"] = pending_promotion_debug

        def _rebuild_next_state_after_slot_updates() -> None:
            nonlocal next_state
            next_state = apply_action_to_state(
                state=self.state,
                features=features,
                action=action,
                add_affirm=add_affirm,
                reflection_style=reflection_style,
            )
            decision.next_state = next_state

        def _apply_non_current_slot_review_result(
            *,
            raw_result: Optional[Tuple[Dict[str, Any], Dict[str, Any]]],
            error: Optional[str],
            execution_timing: str,
        ) -> None:
            nonlocal non_current_slot_review_payload
            nonlocal non_current_slot_review_debug
            nonlocal non_current_review_schema_issues
            nonlocal non_current_reviewed_updates
            nonlocal staged_non_current_phase_slots_debug
            nonlocal pending_promotion_debug
            (
                non_current_slot_review_payload,
                non_current_slot_review_debug,
                non_current_review_schema_issues,
                non_current_reviewed_updates,
                staged_non_current_phase_slots_debug,
            ) = _resolve_non_current_slot_review_result(
                raw_result=raw_result,
                error=error,
                execution_timing=execution_timing,
            )
            if self.state.phase != phase_before_step:
                late_promotion_debug = _promote_pending_phase_slot_updates_on_phase_entry(
                    state=self.state,
                    entered_phase=self.state.phase,
                )
                late_promotion_debug["phase_before"] = phase_before_step.value
                late_promotion_debug["phase_after"] = self.state.phase.value
                existing_promoted_slots = (
                    dict(pending_promotion_debug.get("promoted_slots") or {})
                    if isinstance(pending_promotion_debug.get("promoted_slots"), Mapping)
                    else {}
                )
                late_promoted_slots = (
                    dict(late_promotion_debug.get("promoted_slots") or {})
                    if isinstance(late_promotion_debug.get("promoted_slots"), Mapping)
                    else {}
                )
                if existing_promoted_slots and late_promoted_slots:
                    merged_promoted_slots = dict(existing_promoted_slots)
                    merged_promoted_slots.update(late_promoted_slots)
                    pending_promotion_debug = dict(late_promotion_debug)
                    pending_promotion_debug["promoted"] = True
                    pending_promotion_debug["promoted_count"] = len(merged_promoted_slots)
                    pending_promotion_debug["promoted_slots"] = merged_promoted_slots
                    pending_promotion_debug["reason"] = "merged_with_deferred_non_current_promotion"
                elif existing_promoted_slots:
                    pending_promotion_debug["phase_before"] = phase_before_step.value
                    pending_promotion_debug["phase_after"] = self.state.phase.value
                else:
                    pending_promotion_debug = late_promotion_debug
            _refresh_non_current_phase_debug()
            _refresh_phase_slot_debug_snapshots()
            _rebuild_next_state_after_slot_updates()

        _refresh_non_current_phase_debug()
        if should_run_non_current_slot_reviewer:
            if self.state.phase != phase_before_step:
                try:
                    non_current_slot_review_result = _run_non_current_slot_reviewer()
                    non_current_slot_review_error = None
                except Exception as e:
                    non_current_slot_review_result = None
                    non_current_slot_review_error = str(e)
                _apply_non_current_slot_review_result(
                    raw_result=non_current_slot_review_result,
                    error=non_current_slot_review_error,
                    execution_timing="awaited_before_layer3_layer4_on_phase_transition",
                )
            else:
                non_current_slot_review_executor = ThreadPoolExecutor(max_workers=1)
                non_current_slot_review_future = non_current_slot_review_executor.submit(
                    _run_non_current_slot_reviewer
                )
                non_current_slot_review_executor.shutdown(wait=False)
                non_current_slot_review_executor = None
                non_current_slot_review_debug["execution_timing"] = "deferred_parallel_with_layer3_layer4"
                non_current_slot_review_debug["waited_before_generation"] = False
                non_current_slot_review_debug["waited_after_generation"] = True
                _refresh_non_current_phase_debug()

        # 7) Generation (Layer3 integration -> Layer4 writing)
        try:
            configured_writer_history_turns = max(1, int(self.writer_history_max_turns))
        except Exception:
            configured_writer_history_turns = 120
        try:
            configured_layer4_temperature = float(self.layer4_temperature)
        except Exception:
            configured_layer4_temperature = 0.2
        configured_layer4_temperature = max(0.0, min(1.5, configured_layer4_temperature))

        if self.response_integrator is not None:
            response_integrator = self.response_integrator
            response_integrator_owner = "injected"
        else:
            response_integrator = LLMResponseIntegrator(
                llm=self.llm,
                max_history_turns=configured_writer_history_turns,
            )
            response_integrator_owner = "default_llm_response_integrator"
        if hasattr(response_integrator, "json_mode"):
            try:
                setattr(response_integrator, "json_mode", layer3_json_mode)
            except Exception:
                pass

        try:
            response_brief, response_integrator_debug = response_integrator.integrate(
                history=self.history,
                state=self.state,
                action=action,
                add_affirm=add_affirm,
                reflection_style=reflection_style,
                risk_assessment=risk_assessment,
                focus_candidates=change_talk_candidates,
                slot_target=slot_target,
                first_turn_hint=first_turn_hint,
                allow_question_without_preface=allow_question_without_preface,
                current_user_is_short=features.is_short_reply,
                phase_prediction_debug=phase_debug,
                action_ranking_debug=ranker_debug,
                features=features,
                change_talk_hint=change_talk_inference,
                focus_choice_context=focus_choice_context,
            )
        except Exception as e:
            response_brief = _default_response_brief(
                state=self.state,
                action=action,
                add_affirm=add_affirm,
                reflection_style=reflection_style,
                risk_assessment=risk_assessment,
                focus_candidates=change_talk_candidates,
                change_talk_hint=change_talk_inference,
                history=self.history,
                slot_target=slot_target,
                slot_quality_target_examples=slot_quality_target_examples_for_turn,
                focus_choice_context=focus_choice_context,
            )
            response_integrator_debug = {
                "method": "response_integrator_error_fallback",
                "error": str(e),
                "fallback": "default_response_brief",
                "request_label": "layer3:response_integrator",
            }

        decision.debug["response_integrator"] = {
            "owner": response_integrator_owner,
            "type": type(response_integrator).__name__,
        }
        decision.debug["response_integrator_debug"] = response_integrator_debug
        writer_history_turns = configured_writer_history_turns
        layer4_enabled = bool(self.layer4_enabled)
        layer4_repair_schema_issues: List[str] = []
        raw_schema_issues = (
            response_integrator_debug.get("schema_issues")
            if isinstance(response_integrator_debug, Mapping)
            else None
        )
        if isinstance(raw_schema_issues, Sequence) and not isinstance(raw_schema_issues, (str, bytes)):
            layer4_repair_schema_issues = _merge_issue_codes(raw_schema_issues)
        layer3_draft_text = _sanitize_human_draft_text(response_brief.draft_response_text)
        layer3_draft_ending_family_guidance = _analyze_reflect_ending_family_bias(
            text=layer3_draft_text,
            history=self.history,
            action=action,
        )
        _apply_reflect_ending_family_guidance_to_brief(
            brief=response_brief,
            guidance=layer3_draft_ending_family_guidance,
        )
        response_brief_payload = dataclasses.asdict(response_brief)
        response_brief_payload["slot_repair_hints"] = list(
            response_brief_payload.get("slot_quality_target_examples") or []
        )
        decision.debug["response_brief"] = response_brief_payload
        layer3_draft_validation_ok, layer3_draft_validation_reason = validate_output(
            action,
            layer3_draft_text,
            add_affirm=add_affirm,
            state=self.state,
            first_turn_hint=first_turn_hint,
            change_talk_inference=change_talk_inference,
            brief=response_brief,
            closing_phase_complete=closing_phase_complete,
        )
        layer3_draft_soft_warnings = collect_soft_validation_warnings(
            action,
            layer3_draft_text,
            add_affirm=add_affirm,
            change_talk_inference=change_talk_inference,
            history=self.history,
            state=self.state,
            brief=response_brief,
            closing_phase_complete=closing_phase_complete,
        )
        decision.debug["layer3_draft_validation"] = {
            "ok": layer3_draft_validation_ok,
            "reason": layer3_draft_validation_reason,
            "soft_warnings": list(layer3_draft_soft_warnings),
            "ending_family_guidance": dict(layer3_draft_ending_family_guidance),
        }
        initial_layer4_repair_issue_codes = _build_layer4_repair_issue_codes_for_draft(
            schema_issues=layer4_repair_schema_issues,
            validation_ok=layer3_draft_validation_ok,
            validation_reason=layer3_draft_validation_reason,
            soft_warnings=layer3_draft_soft_warnings,
        )
        effective_layer4_repair_issue_codes = _copy_layer4_repair_issue_codes(
            initial_layer4_repair_issue_codes
        )
        decision.debug["layer4_writer"] = {
            "enabled": layer4_enabled,
            "mode": "llm_writer" if layer4_enabled else "layer3_draft_passthrough",
            "scope": "all_actions",
            "final_closing_bypass": False,
            "max_history_turns": writer_history_turns,
            "temperature": configured_layer4_temperature if layer4_enabled else None,
            "request_label": "layer4:response_writer" if layer4_enabled else None,
            "response_integrator_schema_issues": list(layer4_repair_schema_issues),
            "initial_layer4_repair_issue_codes": _copy_layer4_repair_issue_codes(
                initial_layer4_repair_issue_codes
            ),
            "effective_layer4_repair_issue_codes": _copy_layer4_repair_issue_codes(
                effective_layer4_repair_issue_codes
            ),
            "layer4_repair_issue_codes": _copy_layer4_repair_issue_codes(
                effective_layer4_repair_issue_codes
            ),
            "fallback_used": False,
            "fallback_stage": "",
            "timeout": False,
            "error_type": "",
            "error_reason": "",
            "fallback_source": "",
        }
        layer4_writer_debug = decision.debug["layer4_writer"]
        normalization_events: List[Dict[str, Any]] = []
        generation_attempts: List[Dict[str, Any]] = []
        assistant_output_fallback_events: List[Dict[str, Any]] = []
        last_raw_output: str = ""
        messages: List[Dict[str, str]] = []
        layer4_generation_failed = False
        last_successful_assistant_text = ""

        def _record_layer4_fallback(
            *,
            stage: str,
            exc: Exception,
            fallback_source: str,
        ) -> None:
            error_info = _summarize_assistant_generation_exception(exc)
            fallback_reason = "layer4_writer_timeout" if error_info["timeout"] else "layer4_writer_exception"
            event = {
                "stage": stage,
                "reason": fallback_reason,
                "action": action.value,
                "timeout": bool(error_info["timeout"]),
                "error_type": error_info["error_type"],
                "error_reason": error_info["error_reason"],
                "fallback_source": fallback_source,
            }
            assistant_output_fallback_events.append(event)
            if generation_attempts:
                last_attempt = generation_attempts[-1]
                if last_attempt.get("stage") == stage and not bool(last_attempt.get("ok", True)):
                    last_attempt["fallback_used"] = True
                    last_attempt["fallback_source"] = fallback_source
            if isinstance(layer4_writer_debug, dict):
                layer4_writer_debug["fallback_used"] = True
                layer4_writer_debug["fallback_stage"] = stage
                layer4_writer_debug["timeout"] = bool(error_info["timeout"])
                layer4_writer_debug["error_type"] = error_info["error_type"]
                layer4_writer_debug["error_reason"] = error_info["error_reason"]
                layer4_writer_debug["fallback_source"] = fallback_source

        if layer4_enabled:
            messages = build_writer_messages(
                history=self.history,
                state=self.state,
                action=action,
                add_affirm=add_affirm,
                reflection_style=reflection_style,
                risk_assessment=risk_assessment,
                first_turn_hint=first_turn_hint,
                brief=response_brief,
                layer4_repair_issue_codes=effective_layer4_repair_issue_codes,
                max_history_turns=writer_history_turns,
                closing_phase_complete=closing_phase_complete,
                focus_choice_context=focus_choice_context,
            )

            def _generate_assistant_text(
                temp: float,
                *,
                messages_in: Optional[List[Dict[str, str]]] = None,
                stage: str = "initial",
            ) -> str:
                nonlocal last_raw_output
                request_label = "layer4:response_writer"
                try:
                    raw_output = self.llm.generate(
                        messages_in or messages,
                        temperature=temp,
                        request_label=request_label,
                    )
                except Exception as exc:
                    error_info = _summarize_assistant_generation_exception(exc)
                    generation_attempts.append(
                        {
                            "stage": stage,
                            "request_label": request_label,
                            "temperature": temp,
                            "raw_output": "",
                            "normalized_text": "",
                            "ok": False,
                            "timeout": bool(error_info["timeout"]),
                            "error_type": error_info["error_type"],
                            "error_reason": error_info["error_reason"],
                            "fallback_used": False,
                            "fallback_source": "",
                        }
                    )
                    raise
                last_raw_output = raw_output
                normalized_text, norm_debug = _normalize_assistant_output_text(raw_output)
                generation_attempts.append(
                    {
                        "stage": stage,
                        "request_label": request_label,
                        "temperature": temp,
                        "raw_output": raw_output,
                        "normalized_text": normalized_text,
                        "ok": True,
                        "timeout": False,
                        "error_type": "",
                        "error_reason": "",
                        "fallback_used": False,
                        "fallback_source": "",
                    }
                )
                if norm_debug is not None:
                    event = dict(norm_debug)
                    event["temperature"] = temp
                    event["stage"] = stage
                    normalization_events.append(event)
                return normalized_text

            try:
                assistant_text = _generate_assistant_text(configured_layer4_temperature, stage="initial")
                last_successful_assistant_text = assistant_text
            except Exception as exc:
                assistant_text = layer3_draft_text
                layer4_generation_failed = True
                _record_layer4_fallback(
                    stage="initial",
                    exc=exc,
                    fallback_source="layer3_draft",
                )
        else:
            assistant_text = layer3_draft_text
            last_raw_output = assistant_text
            generation_attempts.append(
                {
                    "stage": "layer3_draft_passthrough",
                    "temperature": None,
                    "raw_output": response_brief.draft_response_text,
                    "normalized_text": assistant_text,
                    "ok": True,
                    "timeout": False,
                    "error_type": "",
                    "error_reason": "",
                    "fallback_used": False,
                    "fallback_source": "",
                }
            )

        requested_validation_mode = _normalize_output_validation_mode(
            getattr(self.cfg, "output_validation_mode", "warn")
        )
        validation_mode = "warn" if layer4_enabled else requested_validation_mode
        validation_ok, validation_reason = validate_output(
            action,
            assistant_text,
            add_affirm=add_affirm,
            state=self.state,
            first_turn_hint=first_turn_hint,
            change_talk_inference=change_talk_inference,
            brief=response_brief,
            closing_phase_complete=closing_phase_complete,
        )
        initial_soft_warnings = collect_soft_validation_warnings(
            action,
            assistant_text,
            add_affirm=add_affirm,
            change_talk_inference=change_talk_inference,
            history=self.history,
            state=self.state,
            brief=response_brief,
            enforce_draft_length_cap=layer4_enabled,
            closing_phase_complete=closing_phase_complete,
        )
        validation_attempts: List[Dict[str, Any]] = [
            {
                "stage": "initial",
                "ok": validation_ok,
                "reason": validation_reason,
                "text": assistant_text,
                "soft_warnings": initial_soft_warnings,
            }
        ]
        current_soft_warnings: List[str] = list(initial_soft_warnings)
        rewrite_trigger_soft_hints: List[str] = []
        rewrite_trigger_warnings: List[str] = []
        rewrite_generation_failed = False
        rewrite_fallback_source = ""
        decision.debug["layer4_soft_warning_rewrite"] = {
            "applied": False,
            "failed": rewrite_generation_failed,
            "fallback_source": rewrite_fallback_source,
            "trigger_warnings": rewrite_trigger_warnings,
            "added_soft_hints": rewrite_trigger_soft_hints,
            "added_issue_codes": rewrite_trigger_soft_hints,
            "effective_layer4_repair_issue_codes": _copy_layer4_repair_issue_codes(
                effective_layer4_repair_issue_codes
            ),
            "layer4_repair_issue_codes": _copy_layer4_repair_issue_codes(
                effective_layer4_repair_issue_codes
            ),
        }
        layer4_writer_debug = decision.debug.get("layer4_writer")
        if isinstance(layer4_writer_debug, Mapping):
            # NOTE: This assumes a dict-like object is being stored, so update it explicitly after the Mapping check.
            try:
                layer4_writer_debug["effective_layer4_repair_issue_codes"] = _copy_layer4_repair_issue_codes(  # type: ignore[index]
                    effective_layer4_repair_issue_codes
                )
                layer4_writer_debug["layer4_repair_issue_codes"] = _copy_layer4_repair_issue_codes(  # type: ignore[index]
                    effective_layer4_repair_issue_codes
                )
            except Exception:
                pass

        if assistant_output_fallback_events:
            last_fallback_event = dict(assistant_output_fallback_events[-1])
            decision.debug["assistant_output_fallback"] = {
                "used": True,
                "stage": str(last_fallback_event.get("stage", "") or ""),
                "reason": str(last_fallback_event.get("reason", "") or ""),
                "action": str(last_fallback_event.get("action", "") or ""),
                "timeout": bool(last_fallback_event.get("timeout")),
                "error_type": str(last_fallback_event.get("error_type", "") or ""),
                "error_reason": str(last_fallback_event.get("error_reason", "") or ""),
                "fallback_source": str(last_fallback_event.get("fallback_source", "") or ""),
                "events": [dict(event) for event in assistant_output_fallback_events],
            }

        if closing_phase_complete and not validation_ok:
            initial_failure_reason = validation_reason
            # Do not regenerate on the final closing turn. Keep the first output and monitor in log-only mode.
            validation_attempts.append(
                {
                    "stage": "closing_final_log_only",
                    "ok": validation_ok,
                    "reason": validation_reason,
                    "text": assistant_text,
                    "soft_warnings": current_soft_warnings,
                }
            )
            if not validation_ok:
                decision.debug["closing_final_validation_unresolved"] = {
                    "used_llm_retry": False,
                    "initial_reason": initial_failure_reason,
                    "final_reason": validation_reason,
                    "fixed_fallback_used": False,
                }
        elif not validation_ok:
            validation_attempts.append(
                {
                    "stage": "layer4_validation_log_only" if layer4_enabled else "layer3_validation_log_only",
                    "ok": validation_ok,
                    "reason": validation_reason,
                    "text": assistant_text,
                    "soft_warnings": current_soft_warnings,
                }
            )

        validation_rule_violations: List[str] = []
        validation_soft_warnings: List[str] = []
        for attempt in validation_attempts:
            if not bool(attempt.get("ok", False)):
                reason = str(attempt.get("reason", "") or "")
                if reason and reason != "ok" and reason not in validation_rule_violations:
                    validation_rule_violations.append(reason)
            for warning in attempt.get("soft_warnings") or []:
                if warning not in validation_soft_warnings:
                    validation_soft_warnings.append(warning)

        decision.debug["assistant_output_validation"] = {
            "ok": validation_ok,
            "reason": validation_reason,
            "mode": validation_mode,
            "requested_mode": requested_validation_mode,
            "enforced": False,
            "accepted_with_warnings": (not validation_ok),
            "rule_violations": validation_rule_violations,
            "attempts": validation_attempts,
            "soft_warnings": validation_soft_warnings,
            "log_only": True,
        }
        decision.debug["layer4_edit_audit"] = _audit_layer4_edit(
            enabled=layer4_enabled,
            draft_response_text=layer3_draft_text,
            assistant_text=assistant_text,
            brief=response_brief,
        )

        # 8) MI-adherence self-check (log only)
        evaluation: Optional[OutputEvaluation] = None
        if self.output_evaluator is not None:
            try:
                evaluation = self.output_evaluator.evaluate(
                    action=action,
                    assistant_text=assistant_text,
                    history=self.history,
                    state=self.state,
                )
            except Exception as e:
                evaluation = OutputEvaluation(score=0.0, feedback=f"evaluation_error:{e}", raw_output=None)

        if non_current_slot_review_future is not None:
            try:
                non_current_slot_review_result = non_current_slot_review_future.result()
                non_current_slot_review_error = None
            except Exception as e:
                non_current_slot_review_result = None
                non_current_slot_review_error = str(e)
            finally:
                if non_current_slot_review_executor is not None:
                    non_current_slot_review_executor.shutdown(wait=False)
                    non_current_slot_review_executor = None
            _apply_non_current_slot_review_result(
                raw_result=non_current_slot_review_result,
                error=non_current_slot_review_error,
                execution_timing="deferred_parallel_with_layer3_layer4",
            )
            non_current_slot_review_future = None

        # 9) Update history (assistant) and refresh state
        self.history.append(("assistant", assistant_text))
        prev_phase = self.state.phase
        self.state = next_state
        # Reset phase_turns when the phase changes; actual transition control lives in enforce_phase_progression.
        if self.state.phase != prev_phase:
            self.state.phase_turns = 0
        self.state.last_user_text = user_text

        if evaluation:
            decision.debug["evaluation"] = {
                "score": evaluation.score,
                "feedback": evaluation.feedback,
                "raw_output": evaluation.raw_output,
            }

        if normalization_events:
            decision.debug["assistant_output_normalization"] = normalization_events
        if generation_attempts:
            decision.debug["assistant_generation_attempts"] = generation_attempts
        decision.debug["assistant_raw_output"] = last_raw_output
        decision.debug["assistant_response_text"] = assistant_text

        if closing_phase_complete:
            decision.debug["closing_final_message_appended"] = False
            decision.debug["session_end_triggered"] = True
            decision.debug["session_end_reason"] = "closing_final_advance"
            decision.debug["session_end_condition"] = {
                "phase": self.state.phase.value,
                "phase_intent_effective": phase_intent_effective,
                "closing_completion_decision": decision.debug.get("closing_completion_decision"),
            }
            decision.debug["response_generation_skipped"] = False

        return assistant_text, decision


# ----------------------------
# Demo dummy LLM (replace before production use)
# ----------------------------
@dataclass
class DummyLLM:
    def generate(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
        sys = messages[0]["content"]
        if "Current main action: REFLECT" in sys:
            return "It sounds like you have been trying to sort through what has been happening lately."
        if "Current main action: QUESTION" in sys:
            return "In the current situation, what feels most important to change first?"
        if "Current main action: SUMMARY" in sys:
            return "So far, it sounds like things feel difficult right now, and at the same time some wish for change is starting to show up. Does that fit?"
        if "Current main action: CLARIFY_PREFERENCE" in sys:
            return "It sounds like information may feel a bit early right now. At this moment, would it help more to sort through your feelings or to hear a few options?"
        if "Current main action: ASK_PERMISSION_TO_SHARE_INFO" in sys:
            return "Would it be okay if I shared a few options?"
        if "Current main action: PROVIDE_INFO" in sys:
            return "For example, options might include keeping a brief record, making the goal smaller, or asking someone around you for support. Which of those feels most doable?"
        return "I understand."


def _run_demo() -> None:
    # Minimal no-API demo that always uses DummyLLM.
    llm = DummyLLM()

    bot = MIRhythmBot(
        llm=llm,
        cfg=PlannerConfig(stochastic=False, seed=1),
    )
    inputs = [
        "Lately I have been staying up too late more often, and that is becoming a problem.",
        "Hmm, but work has been really busy...",
        "Yeah, that's true.",
        "If possible I want to get to sleep earlier.",
        "What do you think I could do?",
        "Yes, please tell me.",
        "I see.",
    ]
    for u in inputs:
        a, d = bot.step(u)
        print("U:", u)
        print("A:", a)
        print(
            "DBG phase:", d.phase.value,
            "action:", d.main_action.value,
            "reflect_streak:", d.next_state.reflect_streak,
            "r_since_q:", d.next_state.r_since_q,
        )
        print("----")

if __name__ == "__main__":
    _run_demo()
