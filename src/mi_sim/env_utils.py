from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

import yaml
from dotenv import load_dotenv

from .paths import resolve_config_path, resolve_env_path


DEFAULT_ENV_PATH = resolve_env_path()
DEFAULT_MODEL_CONFIG_PATH = resolve_config_path("model_settings.yaml")

# Built-in defaults per mode
DEFAULT_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    # ---- Counselor (shared) ----
    "counselor_llm": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "never",
        "temperature_policy": "auto",
        "system_handling": "as_input",
    },
    "counselor_phase_slot_filler": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.0,
        "max_history_turns": 8,
    },
    "counselor_feature_extractor": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.0,
        "max_history_turns": 6,
    },
    "counselor_action": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
    },
    "counselor_slot_reviewer": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.0,
        "max_history_turns": 8,
        "quality_pass_threshold": 0.62,
    },
    "counselor_slot_reviewer_non_current": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.0,
        "max_history_turns": 8,
        "quality_pass_threshold": 0.62,
    },
    "counselor_change_talk_inferer": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.0,
        "max_history_turns": 6,
    },
    "counselor_affirmation_decider": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.0,
        "max_history_turns": 6,
    },
    "counselor_response_integrator": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.1,
        "max_history_turns": 120,
        "retry_once_on_schema_error": False,
    },
    "counselor_response_writer": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "never",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.2,
        "max_history_turns": 120,
    },
    "counselor_session_eval_ratings": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.1,
        "max_history_turns": 120,
    },
    "counselor_session_eval_client_feedback": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "never",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.2,
        "max_history_turns": 120,
    },
    "counselor_session_eval_supervisor_feedback": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "never",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": True,
        "temperature": 0.1,
        "max_history_turns": 120,
    },
    # ---- Client (shared) ----
    "client_profile_llm": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
    },
    "client_state_llm": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
    },
    "client_reply_llm": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "never",
        "temperature_policy": "auto",
        "system_handling": "as_input",
    },
    # ---- Safety checks and response evaluation (optional) ----
    "counselor_risk_detector": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": False,
        "temperature": 0.0,
        "max_history_turns": 8,
    },
    "counselor_mi_evaluator": {
        "api": "responses",
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "verbosity": "low",
        "json_mode": "auto",
        "temperature_policy": "auto",
        "system_handling": "as_input",
        "enabled": False,
        "temperature": 0.0,
        "max_history_turns": 6,
        "rewrite_threshold": None,
    },
}

_MODEL_CONFIG_NESTED_PATHS: Dict[str, Tuple[str, ...]] = {
    # client
    "client_profile_llm": ("client", "profile_llm"),
    "client_state_llm": ("client", "state_llm"),
    "client_reply_llm": ("client", "reply_llm"),
    # counselor
    "counselor_feature_extractor": ("counselor", "layer1", "feature_extractor"),
    "counselor_phase_slot_filler": ("counselor", "layer1", "phase_slot_filler"),
    "counselor_action": ("counselor", "layer2", "action_ranker"),
    "counselor_slot_reviewer": ("counselor", "layer2", "slot_reviewer"),
    "counselor_slot_reviewer_non_current": ("counselor", "layer2", "slot_reviewer_non_current"),
    "counselor_change_talk_inferer": ("counselor", "layer2", "change_talk_inferer"),
    "counselor_affirmation_decider": ("counselor", "layer2", "affirmation_decider"),
    "counselor_response_integrator": ("counselor", "layer3", "response_integrator"),
    "counselor_response_writer": ("counselor", "layer4", "response_writer"),
    "counselor_session_eval_ratings": ("counselor", "session_evaluation", "ratings"),
    "counselor_session_eval_client_feedback": ("counselor", "session_evaluation", "client_feedback"),
    "counselor_session_eval_supervisor_feedback": ("counselor", "session_evaluation", "supervisor_feedback"),
    "counselor_llm": ("counselor", "layer4", "response_writer"),
    "counselor_risk_detector": ("counselor", "safety", "risk_detector"),
    "counselor_mi_evaluator": ("counselor", "safety", "mi_evaluator"),
}

_MODEL_CFG_LEAF_HINT_KEYS = {
    "api",
    "model",
    "reasoning_effort",
    "verbosity",
    "json_mode",
    "temperature_policy",
    "system_handling",
    "enabled",
    "temperature",
    "max_history_turns",
    "quality_pass_threshold",
    "rewrite_threshold",
    "retry_once_on_schema_error",
    "timeout_seconds",
    "max_retries",
    "retry_base_seconds",
    "retry_max_seconds",
    "retry_log",
}


def load_openai_api_key(env_path: Optional[Path] = None) -> str:
    """
    Load the API key from `.env` (`OPENAI_API_KEY`) or from the environment.
    """
    load_dotenv(dotenv_path=env_path or resolve_env_path())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in `.env` or the environment.")
    return api_key


def _is_model_cfg_leaf(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in _MODEL_CFG_LEAF_HINT_KEYS)


def _get_nested_mapping(data: Mapping[str, Any], path: Sequence[str]) -> Optional[Dict[str, Any]]:
    cursor: Any = data
    for part in path:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(part)
    if isinstance(cursor, Mapping) and _is_model_cfg_leaf(cursor):
        return dict(cursor)
    return None


def _load_model_settings(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load model settings from YAML and normalize them into a mode -> config mapping."""
    resolved_path = path or resolve_config_path("model_settings.yaml")
    if not resolved_path.exists():
        return {}
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}

    # Legacy format: accept top-level mode keys as-is
    for mode, cfg in data.items():
        if not isinstance(mode, str):
            continue
        if isinstance(cfg, Mapping) and _is_model_cfg_leaf(cfg):
            normalized[mode] = dict(cfg)

    # New format: resolve the layer structure into mode keys
    for mode, nested_path in _MODEL_CONFIG_NESTED_PATHS.items():
        if mode in normalized:
            continue
        resolved = _get_nested_mapping(data, nested_path)
        if resolved is not None:
            normalized[mode] = resolved

    return normalized


def _merge_default_and_yaml(mode: str, cfg_from_yaml: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(DEFAULT_MODEL_CONFIG.get(mode, {}))
    override = cfg_from_yaml.get(mode)
    if isinstance(override, dict):
        base.update(override)
    return base


def get_model_config(
    mode: str,
    *,
    config_path: Optional[Path] = None,
    role: Literal["counselor", "client", "none"] = "counselor",
    fallback_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Centralize model configuration by mode.
    Priority: YAML > built-in defaults.
    If `fallback_modes` is provided and `mode` is missing from YAML,
    use the first fallback mode that resolves to a non-empty config.
    """
    cfg_from_yaml = _load_model_settings(config_path)
    base = _merge_default_and_yaml(mode, cfg_from_yaml)

    if mode not in cfg_from_yaml and fallback_modes:
        for alt in fallback_modes:
            candidate = _merge_default_and_yaml(alt, cfg_from_yaml)
            if candidate:
                base = candidate
                break

    # Keep `role` for backward compatibility; it is not used in config resolution yet.
    _ = role
    return base


def build_llm_from_config(model_cfg: Dict[str, Any], api_key: str) -> Any:
    """
    Build an LLM client from the dictionary defined in `model_settings.yaml`.
    """
    from .openai_llm import OpenAIChatCompletionsLLM, OpenAIResponsesLLM

    def _as_opt_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _as_opt_int(v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _as_opt_bool(v: Any) -> Optional[bool]:
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return v
        text = str(v).strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
        return None

    api_type = str(model_cfg.get("api", "responses")).strip().lower()
    if api_type == "chat":
        return OpenAIChatCompletionsLLM(
            api_key=api_key,
            model=model_cfg.get("model", ""),
            timeout_seconds=_as_opt_float(model_cfg.get("timeout_seconds")),
            max_retries=_as_opt_int(model_cfg.get("max_retries")),
            retry_base_seconds=_as_opt_float(model_cfg.get("retry_base_seconds")),
            retry_max_seconds=_as_opt_float(model_cfg.get("retry_max_seconds")),
            retry_log=_as_opt_bool(model_cfg.get("retry_log")),
        )

    return OpenAIResponsesLLM(
        api_key=api_key,
        model=model_cfg.get("model", ""),
        reasoning_effort=model_cfg.get("reasoning_effort"),
        verbosity=model_cfg.get("verbosity"),
        json_mode=model_cfg.get("json_mode", "auto"),
        temperature_policy=model_cfg.get("temperature_policy", "auto"),
        system_handling=model_cfg.get("system_handling", "as_input"),
        timeout_seconds=_as_opt_float(model_cfg.get("timeout_seconds")),
        max_retries=_as_opt_int(model_cfg.get("max_retries")),
        retry_base_seconds=_as_opt_float(model_cfg.get("retry_base_seconds")),
        retry_max_seconds=_as_opt_float(model_cfg.get("retry_max_seconds")),
        retry_log=_as_opt_bool(model_cfg.get("retry_log")),
    )
