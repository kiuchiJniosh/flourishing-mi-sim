from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Dict, List, Literal, Optional

from openai import OpenAI


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "on", "y"}:
        return True
    if v in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _resolve_float_option(value: Optional[float], *, env_key: str, default: float) -> float:
    if value is not None:
        return _to_float(value, default)
    env_raw = os.getenv(env_key)
    return _to_float(env_raw, default)


def _resolve_int_option(value: Optional[int], *, env_key: str, default: int) -> int:
    if value is not None:
        return _to_int(value, default)
    env_raw = os.getenv(env_key)
    return _to_int(env_raw, default)


def _resolve_bool_option(value: Optional[bool], *, env_key: str, default: bool) -> bool:
    if value is not None:
        return _to_bool(value, default)
    env_raw = os.getenv(env_key)
    return _to_bool(env_raw, default)


def _extract_status_code(exc: Exception) -> Optional[int]:
    # Be APIStatusError-compatible and keep the lookup loose across SDK versions.
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    val = _to_float(retry_after, -1.0)
    if val <= 0:
        return None
    return val


def _is_retryable_exception(exc: Exception) -> bool:
    status = _extract_status_code(exc)
    if status is not None:
        return status in {408, 409, 429, 500, 502, 503, 504}

    name = exc.__class__.__name__.lower()
    if "ratelimit" in name:
        return True
    if "timeout" in name:
        return True
    if "connection" in name:
        return True
    return False


def _call_with_retry(
    *,
    request_fn: Callable[[], Any],
    request_label: str,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    retry_log: bool,
) -> Any:
    retries = max(0, int(max_retries))
    total_attempts = retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            result = request_fn()
            if attempt > 1 and retry_log:
                print(f"✅ OpenAI {request_label} recovered ({attempt}/{total_attempts}).")
            return result
        except Exception as e:
            should_retry = _is_retryable_exception(e)
            if attempt >= total_attempts or not should_retry:
                raise

            retry_after = _extract_retry_after_seconds(e)
            backoff = retry_base_seconds * (2 ** (attempt - 1))
            base_wait = retry_after if retry_after is not None else backoff
            wait_seconds = min(retry_max_seconds, max(0.1, float(base_wait)))
            jitter = random.uniform(0.0, min(1.0, wait_seconds * 0.1))
            wait_seconds = wait_seconds + jitter

            if retry_log:
                short_error = str(e).replace("\n", " ").strip()
                if len(short_error) > 160:
                    short_error = short_error[:157] + "..."
                print(
                    f"⚠️ OpenAI {request_label} failed ({attempt}/{total_attempts}): "
                    f"{e.__class__.__name__}: {short_error}"
                )
                print(f"   Retrying in {wait_seconds:.1f} seconds.")
            time.sleep(wait_seconds)


def _extract_output_text_from_response(response: Any) -> str:
    """
    Extract text from a `responses.create` result.
    Prefer `output_text` when the SDK provides it; otherwise walk `output` and join chunks.
    """
    out_text = getattr(response, "output_text", None)
    if isinstance(out_text, str):
        return out_text

    chunks: List[str] = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    chunks.append(t)
    return "".join(chunks)


def _should_use_json_mode(messages: List[Dict[str, str]]) -> bool:
    """Use a lightweight check to prefer JSON mode when the body mentions `json`."""
    try:
        for m in messages:
            content = str(m.get("content", "") or "").lower()
            if "json" in content:
                return True
    except Exception:
        return False
    return False


def _model_disallows_temperature(model: str) -> bool:
    """
    For GPT-5 family models (`gpt-5`, `gpt-5.1`, `gpt-5-mini`, `gpt-5-nano`, etc.),
    it is safer not to send `temperature`.
    Provider-prefixed forms such as `openai/gpt-5.1` are also accepted.
    """
    m = (model or "").strip().lower()
    if "/" in m:
        m = m.rsplit("/", 1)[-1].strip()
    return m == "gpt-5" or m.startswith("gpt-5-") or m.startswith("gpt-5.")


class OpenAIChatCompletionsLLM:
    """A simple LLM client that uses `chat.completions` from the v1 OpenAI Python SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_base_seconds: Optional[float] = None,
        retry_max_seconds: Optional[float] = None,
        retry_log: Optional[bool] = None,
    ):
        self.timeout_seconds = max(
            1.0,
            _resolve_float_option(timeout_seconds, env_key="OPENAI_TIMEOUT_SECONDS", default=45.0),
        )
        self.max_retries = max(
            0,
            _resolve_int_option(max_retries, env_key="OPENAI_MAX_RETRIES", default=2),
        )
        self.retry_base_seconds = max(
            0.1,
            _resolve_float_option(retry_base_seconds, env_key="OPENAI_RETRY_BASE_SECONDS", default=1.0),
        )
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            _resolve_float_option(retry_max_seconds, env_key="OPENAI_RETRY_MAX_SECONDS", default=20.0),
        )
        self.retry_log = _resolve_bool_option(retry_log, env_key="OPENAI_RETRY_LOG", default=True)
        # Disable the SDK's implicit retries and surface wait times in our own retry loop.
        self.client = OpenAI(api_key=api_key, timeout=self.timeout_seconds, max_retries=0)
        self.model = model

    def generate(self, messages: List[Dict[str, str]], *, temperature: float = 0.2, **kwargs: Any) -> str:
        extra: Dict[str, Any] = dict(kwargs) if kwargs else {}
        extra.pop("messages", None)
        extra.pop("response_format", None)
        request_label = str(extra.pop("request_label", "") or "chat.completions.create")

        response = _call_with_retry(
            request_label=request_label,
            max_retries=self.max_retries,
            retry_base_seconds=self.retry_base_seconds,
            retry_max_seconds=self.retry_max_seconds,
            retry_log=self.retry_log,
            request_fn=lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **extra,
            ),
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        try:
            return "".join(getattr(part, "text", "") for part in content or [])
        except Exception:
            return ""


class OpenAIResponsesLLM:
    """
    LLM client for the OpenAI Responses API.
    - `system_handling="as_input"` (default): include the system role in `input` for compatibility
    - `system_handling="instructions"`: aggregate system role messages into `instructions`
    - `json_mode`: `auto` / `always` / `never`
    - `temperature_policy`: `auto` (omit for GPT-5 family) / `always` / `never`
    - Drop extra kwargs conservatively
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        reasoning_effort: Optional[str] = "medium",
        verbosity: Optional[str] = "low",
        store: bool = False,
        safety_identifier: Optional[str] = None,
        system_handling: Literal["instructions", "as_input"] = "as_input",
        json_mode: Literal["auto", "always", "never"] = "auto",
        temperature_policy: Literal["auto", "always", "never"] = "auto",
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_base_seconds: Optional[float] = None,
        retry_max_seconds: Optional[float] = None,
        retry_log: Optional[bool] = None,
    ):
        self.timeout_seconds = max(
            1.0,
            _resolve_float_option(timeout_seconds, env_key="OPENAI_TIMEOUT_SECONDS", default=45.0),
        )
        self.max_retries = max(
            0,
            _resolve_int_option(max_retries, env_key="OPENAI_MAX_RETRIES", default=2),
        )
        self.retry_base_seconds = max(
            0.1,
            _resolve_float_option(retry_base_seconds, env_key="OPENAI_RETRY_BASE_SECONDS", default=1.0),
        )
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            _resolve_float_option(retry_max_seconds, env_key="OPENAI_RETRY_MAX_SECONDS", default=20.0),
        )
        self.retry_log = _resolve_bool_option(retry_log, env_key="OPENAI_RETRY_LOG", default=True)
        # Disable the SDK's implicit retries and surface wait times in our own retry loop.
        self.client = OpenAI(api_key=api_key, timeout=self.timeout_seconds, max_retries=0)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.store = store
        self.safety_identifier = safety_identifier
        self.system_handling = system_handling
        self.json_mode = json_mode
        self.temperature_policy = temperature_policy

    def _split_messages(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        instruction_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        if self.system_handling == "instructions":
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    instruction_parts.append(str(content))
                else:
                    input_items.append({"role": role, "content": content})
            return {
                "instructions": "\n\n".join(instruction_parts) if instruction_parts else None,
                "input": input_items if input_items else "",
            }

        # as_input: pass through as-is
        return {"instructions": None, "input": messages}

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        extra: Dict[str, Any] = dict(kwargs) if kwargs else {}
        request_label = str(extra.pop("request_label", "") or "responses.create")
        # Remove arguments that are unnecessary or risky for Responses API compatibility
        extra.pop("seed", None)
        extra.pop("messages", None)
        extra.pop("response_format", None)

        if "max_tokens" in extra and "max_output_tokens" not in extra:
            extra["max_output_tokens"] = extra.pop("max_tokens")

        req: Dict[str, Any] = {
            "model": self.model,
            "store": self.store,
        }
        req.update(self._split_messages(messages))

        # JSON mode detection
        text_format: Dict[str, Any] = {"type": "text"}
        if self.json_mode == "always" or (self.json_mode == "auto" and _should_use_json_mode(messages)):
            text_format = {"type": "json_object"}
        req["text"] = {"format": text_format}
        if self.verbosity:
            req["text"]["verbosity"] = self.verbosity

        # reasoning effort
        if self.reasoning_effort:
            req["reasoning"] = {"effort": self.reasoning_effort}

        # safety_identifier
        if self.safety_identifier:
            req["safety_identifier"] = self.safety_identifier

        # temperature
        send_temperature = False
        if self.temperature_policy == "always":
            send_temperature = True
        elif self.temperature_policy == "auto" and not _model_disallows_temperature(self.model):
            send_temperature = True

        if send_temperature:
            req["temperature"] = float(temperature)

        # Merge the remaining options without overriding `model`, `input`, `store`, or `text`
        for k, v in extra.items():
            if k in ("model", "input", "store", "text"):
                continue
            req[k] = v

        response = _call_with_retry(
            request_label=request_label,
            max_retries=self.max_retries,
            retry_base_seconds=self.retry_base_seconds,
            retry_max_seconds=self.retry_max_seconds,
            retry_log=self.retry_log,
            request_fn=lambda: self.client.responses.create(**req),
        )
        return _extract_output_text_from_response(response)
