from __future__ import annotations

from typing import Any, Dict

from .env_utils import build_llm_from_config, get_model_config


def build_client_llms(
    api_key: str,
) -> Dict[str, Any]:
    """
    クライアント用の profile/state/reply LLM と設定をまとめて構築する（統一キー）。
    """
    client_cfg = get_model_config("client_profile_llm", role="client")
    state_cfg = get_model_config("client_state_llm", role="client")
    llm_state = build_llm_from_config(state_cfg, api_key)

    reply_cfg = get_model_config("client_reply_llm", role="client")
    llm_reply = build_llm_from_config(reply_cfg, api_key)

    return {
        "client_cfg": client_cfg,
        "client_state_cfg": state_cfg,
        "client_reply_cfg": reply_cfg,
        "client_llm_state": llm_state,
        "client_llm_reply": llm_reply,
    }
