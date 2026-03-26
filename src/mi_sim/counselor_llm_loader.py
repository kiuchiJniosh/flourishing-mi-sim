from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .env_utils import build_llm_from_config, get_model_config
from .mi_counselor_agent import (
    LLMMIEvaluator,
    LLMActionRanker,
    LLMAffirmationDecider,
    LLMChangeTalkInferer,
    LLMFeatureExtractor,
    LLMNonCurrentSlotReviewer,
    LLMSlotReviewer,
    LLMPhaseSlotFiller,
    LLMResponseIntegrator,
    LLMRiskDetector,
    MIRhythmBot,
    PlannerConfig,
    RuleBasedFeatureExtractor,
)


def _to_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _to_opt_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _build_planner_config_from_env() -> PlannerConfig:
    cfg = PlannerConfig()
    threshold_raw = os.getenv("PHASE_SLOT_QUALITY_MIN_THRESHOLD")
    threshold = _to_opt_float(threshold_raw)
    if threshold is not None and 0.0 <= threshold <= 1.0:
        cfg.phase_slot_quality_min_threshold = threshold
    return cfg


def _build_optional_phase_slot_filler(api_key: str, cfg: Dict[str, Any]) -> Optional[LLMPhaseSlotFiller]:
    """Create an LLMPhaseSlotFiller only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 8) or 8)
    return LLMPhaseSlotFiller(llm=llm, temperature=temperature, max_history_turns=max_history_turns)


def _build_optional_action_ranker(api_key: str, cfg: Dict[str, Any]) -> Optional[LLMActionRanker]:
    """Create an LLMActionRanker only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    return LLMActionRanker(llm=llm)


def _build_optional_slot_reviewer(api_key: str, cfg: Dict[str, Any]) -> Optional[LLMSlotReviewer]:
    """Create an LLMSlotReviewer only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 8) or 8)
    quality_pass_threshold = float(cfg.get("quality_pass_threshold", 0.62) or 0.62)
    return LLMSlotReviewer(
        llm=llm,
        temperature=temperature,
        max_history_turns=max_history_turns,
        quality_pass_threshold=quality_pass_threshold,
    )


def _build_optional_non_current_slot_reviewer(
    api_key: str,
    cfg: Dict[str, Any],
) -> Optional[LLMNonCurrentSlotReviewer]:
    """Create an LLMNonCurrentSlotReviewer only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 8) or 8)
    quality_pass_threshold = float(cfg.get("quality_pass_threshold", 0.62) or 0.62)
    return LLMNonCurrentSlotReviewer(
        llm=llm,
        temperature=temperature,
        max_history_turns=max_history_turns,
        quality_pass_threshold=quality_pass_threshold,
    )


def _build_optional_change_talk_inferer(api_key: str, cfg: Dict[str, Any]) -> Optional[LLMChangeTalkInferer]:
    """Create an LLMChangeTalkInferer only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 6) or 6)
    return LLMChangeTalkInferer(llm=llm, temperature=temperature, max_history_turns=max_history_turns)


def _build_optional_affirmation_decider(
    api_key: str,
    cfg: Dict[str, Any],
) -> Optional[LLMAffirmationDecider]:
    """Create an LLMAffirmationDecider only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 6) or 6)
    return LLMAffirmationDecider(llm=llm, temperature=temperature, max_history_turns=max_history_turns)


def _build_feature_extractor(api_key: str, cfg: Dict[str, Any]) -> LLMFeatureExtractor | RuleBasedFeatureExtractor:
    """
    Layer 1 feature extractor.
    Falls back to RuleBasedFeatureExtractor when `enabled` is false.
    """
    if not cfg or cfg.get("enabled") is False:
        return RuleBasedFeatureExtractor()
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 6) or 6)
    return LLMFeatureExtractor(llm=llm, temperature=temperature, max_history_turns=max_history_turns)


def _build_optional_risk_detector(api_key: str, cfg: Dict[str, Any]) -> Optional[LLMRiskDetector]:
    """Create an LLMRiskDetector only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 8) or 8)
    return LLMRiskDetector(llm=llm, temperature=temperature, max_history_turns=max_history_turns)


def _build_optional_mi_evaluator(api_key: str, cfg: Dict[str, Any]) -> Optional[LLMMIEvaluator]:
    """Create an LLMMIEvaluator only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    max_history_turns = int(cfg.get("max_history_turns", 6) or 6)
    return LLMMIEvaluator(llm=llm, temperature=temperature, max_history_turns=max_history_turns)


def _build_optional_response_integrator(api_key: str, cfg: Dict[str, Any]) -> Optional[LLMResponseIntegrator]:
    """Create an LLMResponseIntegrator only when `enabled` is truthy."""
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None
    llm = build_llm_from_config(cfg, api_key)
    temperature = float(cfg.get("temperature", 0.1) or 0.1)
    max_history_turns = int(cfg.get("max_history_turns", 120) or 120)
    retry_once_on_schema_error = _to_bool(cfg.get("retry_once_on_schema_error"), default=False)
    return LLMResponseIntegrator(
        llm=llm,
        temperature=temperature,
        max_history_turns=max_history_turns,
        retry_once_on_schema_error=retry_once_on_schema_error,
    )


def _extract_rewrite_threshold(cfg: Dict[str, Any]) -> Optional[float]:
    """Return `rewrite_threshold` as a float, or `None` if unavailable."""
    threshold = cfg.get("rewrite_threshold")
    if threshold is None:
        return None
    try:
        return float(threshold)
    except Exception:
        return None


def build_counselor_stack(
    *,
    api_key: str,
) -> Dict[str, Any]:
    """
    Build the counselor bot and its surrounding classifiers/evaluators in one place.
    """
    counselor_writer_cfg = get_model_config(
        "counselor_response_writer",
        role="counselor",
        fallback_modes=("counselor_llm",),
    )
    counselor_slot_fill_cfg = get_model_config(
        "counselor_phase_slot_filler",
        role="counselor",
    )
    counselor_action_cfg = get_model_config("counselor_action", role="counselor")
    counselor_slot_reviewer_cfg = get_model_config(
        "counselor_slot_reviewer",
        role="counselor",
    )
    counselor_non_current_slot_reviewer_cfg = get_model_config(
        "counselor_slot_reviewer_non_current",
        role="counselor",
        fallback_modes=("counselor_slot_reviewer",),
    )
    counselor_change_talk_cfg = get_model_config(
        "counselor_change_talk_inferer",
        role="counselor",
    )
    counselor_affirmation_cfg = get_model_config(
        "counselor_affirmation_decider",
        role="counselor",
    )
    counselor_feature_cfg = get_model_config(
        "counselor_feature_extractor",
        role="counselor",
    )
    risk_detector_cfg = get_model_config("counselor_risk_detector", role="counselor")
    mi_evaluator_cfg = get_model_config("counselor_mi_evaluator", role="counselor")
    response_integrator_cfg = get_model_config(
        "counselor_response_integrator",
        role="counselor",
        fallback_modes=("counselor_llm",),
    )

    llm = build_llm_from_config(counselor_writer_cfg, api_key)
    feature_extractor = _build_feature_extractor(api_key, counselor_feature_cfg)
    phase_slot_filler = _build_optional_phase_slot_filler(api_key, counselor_slot_fill_cfg)
    action_ranker = _build_optional_action_ranker(api_key, counselor_action_cfg)
    slot_reviewer = _build_optional_slot_reviewer(api_key, counselor_slot_reviewer_cfg)
    slot_reviewer_non_current = _build_optional_non_current_slot_reviewer(
        api_key,
        counselor_non_current_slot_reviewer_cfg,
    )
    change_talk_inferer = _build_optional_change_talk_inferer(api_key, counselor_change_talk_cfg)
    affirmation_decider = _build_optional_affirmation_decider(api_key, counselor_affirmation_cfg)
    response_integrator = _build_optional_response_integrator(api_key, response_integrator_cfg)
    risk_detector = _build_optional_risk_detector(api_key, risk_detector_cfg)
    mi_evaluator = _build_optional_mi_evaluator(api_key, mi_evaluator_cfg)
    rewrite_threshold = _extract_rewrite_threshold(mi_evaluator_cfg)
    writer_history_max_turns = int(counselor_writer_cfg.get("max_history_turns", 120) or 120)
    try:
        layer4_temperature = float(counselor_writer_cfg.get("temperature", 0.2) or 0.2)
    except Exception:
        layer4_temperature = 0.2
    layer4_enabled = _to_bool(counselor_writer_cfg.get("enabled"), default=True)

    planner_cfg = _build_planner_config_from_env()
    counselor = MIRhythmBot(
        llm=llm,
        cfg=planner_cfg,
        feature_extractor=feature_extractor,
        phase_slot_filler=phase_slot_filler,
        action_ranker=action_ranker,
        slot_reviewer=slot_reviewer,
        slot_reviewer_non_current=slot_reviewer_non_current,
        change_talk_inferer=change_talk_inferer,
        affirmation_decider=affirmation_decider,
        response_integrator=response_integrator,
        risk_detector=risk_detector,
        output_evaluator=mi_evaluator,
        evaluation_rewrite_threshold=rewrite_threshold,
        writer_history_max_turns=writer_history_max_turns,
        layer4_temperature=layer4_temperature,
        layer4_enabled=layer4_enabled,
    )

    return {
        "counselor": counselor,
        "llm": llm,
        "counselor_cfg": counselor_writer_cfg,
        "writer_cfg": counselor_writer_cfg,
        "slot_fill_cfg": counselor_slot_fill_cfg,
        "action_cfg": counselor_action_cfg,
        "slot_reviewer_cfg": counselor_slot_reviewer_cfg,
        "non_current_slot_reviewer_cfg": counselor_non_current_slot_reviewer_cfg,
        "change_talk_cfg": counselor_change_talk_cfg,
        "affirmation_cfg": counselor_affirmation_cfg,
        "feature_cfg": counselor_feature_cfg,
        "response_integrator_cfg": response_integrator_cfg,
        "risk_cfg": risk_detector_cfg,
        "mi_eval_cfg": mi_evaluator_cfg,
    }
