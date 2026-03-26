from __future__ import annotations

"""
Client-side agent for simulation.

- Standalone module for cases where we want to split it out from conversation_environment.py.
- Shared by human counselor x LLM client setups for labeling.
- Shared by LLM counselor x LLM client self-play simulations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Literal, Optional, Tuple
from pathlib import Path
import os
import yaml
import json
import re

from .paths import APP_DIR, PROJECT_ROOT, resolve_config_path, resolve_project_path
from .mi_counselor_agent import LLMClient

# Default client settings (also referenced by the CLI)
DEFAULT_CLIENT_CODE: str = "LANG_MGR"
DEFAULT_FIRST_CLIENT_UTTERANCE: str = "Recently, my daily routine has fallen apart, and I have been feeling down."
REPLY_SENTENCE_MIN: int = 1
REPLY_SENTENCE_MAX: int = 4


class ClientAgent(Protocol):
    """
    Interface for a client-side agent.

    respond:
      - counselor_text: the most recent counselor utterance (current input)
      - history: a list of ConversationTurn-like objects (speaker/text attributes are enough)
      - return: the next client utterance (str)
    """

    def respond(self, counselor_text: str, history: List[Any]) -> str:
        ...


# ==============================
# Client internal state
# ==============================

@dataclass
class ClientInternalState:
    """
    Six metrics kept as the client's internal log (all continuous values from 0 to 10).

    - pos_affect: strength of positive affect
    - neg_affect: strength of negative affect
    - importance_change: perceived importance of change / goal attainment
    - confidence_change: perceived confidence in change / goal attainment
    - like_counselor: affinity toward the counselor
    - tension_counselor: dissonance / tension toward the counselor

    trait_* values indicate how easy it is for each score to surface.
    -1: hard to surface (tends to stay lower)
     0: neutral
    +1: easy to surface (tends to stay higher)
    """

    # ------------------------------
    # State (0-10)
    # ------------------------------
    pos_affect: float = 5.0
    neg_affect: float = 5.0
    importance_change: float = 3.0
    confidence_change: float = 2.0
    like_counselor: float = 5.0
    tension_counselor: float = 0.0

    # ------------------------------
    # Traits: how easily each score surfaces
    # -1 = hard to surface
    #  0 = neutral
    # +1 = easy to surface
    # ------------------------------
    trait_expression_pos: float = 0.0
    trait_expression_neg: float = 0.0
    trait_expression_importance: float = 0.0
    trait_expression_confidence: float = 0.0
    trait_expression_like: float = 0.0
    trait_expression_tension: float = 0.0

    STATE_KEYS = (
        "pos_affect",
        "neg_affect",
        "importance_change",
        "confidence_change",
        "like_counselor",
        "tension_counselor",
    )

    TRAIT_KEYS = (
        "trait_expression_pos",
        "trait_expression_neg",
        "trait_expression_importance",
        "trait_expression_confidence",
        "trait_expression_like",
        "trait_expression_tension",
    )

    def to_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in self.STATE_KEYS}

    def to_traits_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in self.TRAIT_KEYS}

    def to_full_dict(self) -> Dict[str, float]:
        data = self.to_dict()
        data.update(self.to_traits_dict())
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base: Optional["ClientInternalState"] = None) -> "ClientInternalState":
        """
        Helper for restoring from JSON and similar sources.
        Unknown keys and values are ignored, and defaults are used as the base.
        """
        base_state = base if isinstance(base, cls) else cls()
        # Initialize with the base values first (state + traits).
        base_kwargs = {k: getattr(base_state, k) for k in cls.STATE_KEYS + cls.TRAIT_KEYS}
        base = cls(**base_kwargs)  # type: ignore[arg-type]
        if not isinstance(data, dict):
            return base

        for key in cls.STATE_KEYS + cls.TRAIT_KEYS:
            v = data.get(key)
            if v is None:
                continue
            try:
                setattr(base, key, float(v))
            except (TypeError, ValueError):
                # Ignore values that cannot be converted to numbers.
                continue

        return base

    # ------------------------------
    # Retrieve expression tendency (trait_expression_*)
    # ------------------------------
    def get_expression(self, state_key: str) -> float:
        """
        Return the expression tendency for each score.
        Values are clipped to the range -1.0 to +1.0.
        """
        mapping = {
            "pos_affect": "trait_expression_pos",
            "neg_affect": "trait_expression_neg",
            "importance_change": "trait_expression_importance",
            "confidence_change": "trait_expression_confidence",
            "like_counselor": "trait_expression_like",
            "tension_counselor": "trait_expression_tension",
        }
        trait_key = mapping.get(state_key)
        if trait_key is None:
            return 0.0

        try:
            val = float(getattr(self, trait_key, 0.0))
        except (TypeError, ValueError):
            val = 0.0

        # Clip to -1.0 to +1.0.
        if val < -1.0:
            val = -1.0
        elif val > 1.0:
            val = 1.0
        return val

    def adjust_by_expression(self, state_key: str, before: float, target: float) -> float:
        """
        Distort an LLM-suggested target according to the score's expression tendency.

        - expr = -1 (hard to surface)
            upward change (target > before) tends to shrink (0.5x)
            downward change (target < before) tends to grow (1.5x)

        - expr = +1 (easy to surface)
            upward change tends to grow (1.5x)
            downward change tends to shrink (0.5x)

        expr = 0 (neutral) leaves the target unchanged.
        """
        expr = self.get_expression(state_key)
        delta = target - before

        # Leave unchanged when there is no delta or the tendency is neutral.
        if delta == 0.0 or expr == 0.0:
            return target

        # Tendency strength (0-1). Around 0.5 gives a moderate effect.
        alpha = 0.5

        if delta > 0:
            # Upward change: expr > 0 amplifies, expr < 0 suppresses.
            mult = 1.0 + alpha * expr
        else:
            # Downward change: expr > 0 suppresses, expr < 0 amplifies.
            mult = 1.0 - alpha * expr

        return before + delta * mult


@dataclass
class SimpleClientLLM(ClientAgent):
    """
    Simple client agent powered by an LLM.

    - Responds naturally as a client with concerns
    - Intended for experiments and simulations
    """
    llm: LLMClient  # LLM for replies
    llm_state: Optional[LLMClient] = None  # LLM for state estimation (used in two_stage)
    style: Literal["cooperative", "ambivalent", "resistant"] = "cooperative"
    persona: Optional[str] = None
    scenario: Optional[str] = None
    temperature: float = 0.4
    seed: Optional[int] = None
    # Per-turn ceiling for state changes (None means no limit).
    # Improvement: set a default ceiling to reduce abrupt jumps and saturation.
    max_state_step: Optional[float] = 0.8
    max_history_turns: int = 20

    # Overall scaling factor for change magnitude (values > 1.0 increase change).
    # Improvement: default to 1.0 to avoid double amplification and keep changes natural.
    delta_scale: float = 1.0
    # Post-utterance reassessment (right after speaking) should move less than after listening.
    post_reply_delta_scale: float = 0.6
    post_reply_max_state_step: Optional[float] = 0.5

    # Ease-of-change per metric (1.0 = baseline, 0.0 = no change).
    #   - pos/neg: easier to change
    #   - importance/confidence/tension: harder to change
    #   - like: medium
    sensitivity_pos: float = 1.0
    sensitivity_neg: float = 1.0
    sensitivity_importance: float = 0.8
    sensitivity_confidence: float = 0.8
    sensitivity_like: float = 1.0
    sensitivity_tension: float = 0.8

    # Number of decimal places used to store internal state (up to 2 decimal places).
    state_decimal_places: int = 2

    # Safety net for when like/tension do not move (small adjustment only when unchanged).
    relationship_heuristic: bool = True
    relationship_heuristic_only_if_unchanged: bool = True

    # Client internal state (updated each turn)
    internal_state: ClientInternalState = field(default_factory=ClientInternalState)
    prompt_profile: Optional[Dict[str, Any]] = None
    _last_debug_info: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.persona is None:
            self.persona = self._build_persona(self.style, self.scenario)
        else:
            self.persona = str(self.persona)

        if self.scenario is not None:
            self.scenario = str(self.scenario)

        # Apply the style-specific initial state baseline (cooperative / ambivalent / resistant).
        try:
            baseline = self._get_style_state_baseline(self.style)
            for k, v in baseline.items():
                setattr(self.internal_state, k, float(v))
        except Exception:
            pass

        # Allow the initial importance/confidence values to be overridden via environment variables (soft defaults: 3 and 2).
        try:
            imp0 = os.getenv("CLIENT_IMPORTANCE_BASELINE")
            conf0 = os.getenv("CLIENT_CONFIDENCE_BASELINE")
            if imp0 is not None:
                self.internal_state.importance_change = self._clip_0_10(float(imp0))
            if conf0 is not None:
                self.internal_state.confidence_change = self._clip_0_10(float(conf0))
        except Exception:
            # Keep the defaults (3, 2) even if reading fails.
            pass

    def reset(self) -> None:
        """
        Intended to be called when resetting a session.
        Restores the internal state to its initial values while keeping traits and applying the style baseline.
        """
        # Keep the existing traits.
        traits = self.internal_state.to_traits_dict()
        # Style-specific baseline.
        baseline = self._get_style_state_baseline(self.style)
        st = ClientInternalState()
        for k, v in baseline.items():
            setattr(st, k, float(v))
        # Restore traits.
        for k, v in traits.items():
            try:
                setattr(st, k, float(v))
            except Exception:
                pass
        # Override initial importance/confidence via .env values, if present.
        try:
            imp0 = os.getenv("CLIENT_IMPORTANCE_BASELINE")
            conf0 = os.getenv("CLIENT_CONFIDENCE_BASELINE")
            if imp0 is not None:
                st.importance_change = self._clip_0_10(float(imp0))
            if conf0 is not None:
                st.confidence_change = self._clip_0_10(float(conf0))
        except Exception:
            pass
        self.internal_state = st

    def get_internal_state(self) -> Dict[str, float]:
        """
        Accessor used by ConversationEnvironment for logging.
        """
        return self.internal_state.to_dict()

    def get_last_debug_info(self) -> Dict[str, Any]:
        """
        Return debug information for the most recent LLM response.
        - raw: raw response
        - reply: the reply actually used
        - old_state / new_state: state before and after the update (including traits)
        - internal_state_reason: explanation for the state change, if the LLM provided one
        - meta: additional self-labels and similar metadata
        """
        return dict(self._last_debug_info)

    @staticmethod
    def _get_prompt_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
        raw = profile.get("prompt_profile")
        return dict(raw) if isinstance(raw, Mapping) else {}

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _mapping_list(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                out.append(dict(item))
        return out

    @classmethod
    def _render_prompt_profile_context(cls, prompt_profile: Mapping[str, Any]) -> str:
        lines: List[str] = []

        facts = cls._mapping_list(prompt_profile.get("surface_facts_early"))
        scenes = cls._mapping_list(prompt_profile.get("problem_scenes"))
        wording_bank = prompt_profile.get("wording_bank")
        hidden = cls._string_list(prompt_profile.get("hidden_formulation"))

        if facts:
            lines.append("[Concrete facts easy to surface early]")
            for item in facts[:3]:
                text = str(item.get("text") or "").strip()
                if text:
                    lines.append(f"- {text}")

        if scenes:
            lines.append("[Concrete scenes that often cause difficulty]")
            for item in scenes[:3]:
                scene = str(item.get("scene") or "").strip()
                if scene:
                    lines.append(f"- {scene}")

        if isinstance(wording_bank, Mapping):
            lines.append("[Typical wording used by the person]")
            for key in ("complaint_openers", "self_view_phrases", "hedges"):
                values = cls._string_list(wording_bank.get(key))
                for text in values[:2]:
                    lines.append(f"- {text}")

        if hidden:
            lines.append("[Background notes]")
            lines.append("- For background understanding only. Do not use technical or analytical language in the body.")
            for text in hidden[:3]:
                lines.append(f"- {text}")

        return "\n".join(lines).strip()

    @classmethod
    def _resolve_prompt_profile_stage(
        cls,
        prompt_profile: Mapping[str, Any],
        history: List[Any],
    ) -> str:
        _ = prompt_profile
        client_turns = 0
        for turn in history:
            if getattr(turn, "speaker", "") == "client":
                client_turns += 1
        if client_turns <= 2:
            return "early"
        if client_turns <= 5:
            return "middle"
        return "late"

    @classmethod
    def _collect_reveal_ids_for_stage(
        cls,
        prompt_profile: Mapping[str, Any],
        stage: str,
    ) -> List[str]:
        reveal_plan = prompt_profile.get("reveal_plan")
        if not isinstance(reveal_plan, Mapping):
            return []
        ids = reveal_plan.get(stage)
        return cls._string_list(ids)

    @classmethod
    def _build_turn_specific_prompt_hint(
        cls,
        prompt_profile: Mapping[str, Any],
        history: List[Any],
    ) -> str:
        lines: List[str] = []
        stage = cls._resolve_prompt_profile_stage(prompt_profile, history)
        target_ids = set(cls._collect_reveal_ids_for_stage(prompt_profile, stage))

        facts = cls._mapping_list(prompt_profile.get("surface_facts_early"))
        scenes = cls._mapping_list(prompt_profile.get("problem_scenes"))

        if not target_ids:
            for item in facts[:2]:
                text = str(item.get("text") or "").strip()
                if text:
                    lines.append(f"- Candidate life fact: {text}")
            for item in scenes[:2]:
                scene = str(item.get("scene") or "").strip()
                if scene:
                    lines.append(f"- Candidate scene: {scene}")
                    likely_words = cls._string_list(item.get("likely_words"))
                    if likely_words:
                        lines.append(f"- Example wording: {' / '.join(likely_words[:2])}")
            return "\n".join(lines).strip()

        for item in facts:
            item_id = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            if item_id in target_ids and text:
                lines.append(f"- Candidate life fact: {text}")

        for item in scenes:
            item_id = str(item.get("id") or "").strip()
            scene = str(item.get("scene") or "").strip()
            if item_id in target_ids and scene:
                lines.append(f"- Candidate scene: {scene}")
                likely_words = cls._string_list(item.get("likely_words"))
                if likely_words:
                    lines.append(f"- Example wording: {' / '.join(likely_words[:2])}")

        return "\n".join(lines[:8]).strip()

    @staticmethod
    def _build_persona(style: str, scenario: Optional[str] = None) -> str:
        presets = {
            "cooperative": (
                "You are a client who is dealing with some difficulties in daily life or behavior.\n"
                "Speak honestly about your feelings and situation, as much as you can, in a polite tone.\n"
                "This is a place to express your real thoughts and uncertainty, not to argue with the counselor.\n"
            ),
            "ambivalent": (
                "You are a client whose desire to change is mixed with the feeling that it might not work anyway.\n"
                "Put both hopeful feelings and anxiety or hesitation into words honestly and politely.\n"
                "Respond naturally to reflections, and it is okay to sometimes show hesitation or indecision.\n"
            ),
            "resistant": (
                "You are a client who is somewhat guarded and has doubts or distrust about change.\n"
                "Stay polite, but mix in responses that show resistance or skepticism, such as 'but,' 'anyway,' or 'I failed before.'\n"
                "Do not become aggressive; keep your responses grounded in your real feelings.\n"
            ),
        }
        base_persona = presets.get(style, presets["cooperative"])
        if scenario:
            base_persona += "\n[Scenario]\n" + str(scenario).strip() + "\n"

        # Append additional rules from an external file.
        # - If the file exists, append its full contents as-is.
        # - Default: config/client_prompt_rules.md
        # - Can be overridden with CLIENT_PROMPT_RULES_MD_PATH
        rules_path_env = (os.getenv("CLIENT_PROMPT_RULES_MD_PATH") or "").strip()
        if rules_path_env:
            rules_path = resolve_project_path(rules_path_env)
        else:
            rules_path = resolve_config_path("client_prompt_rules.md")
        if rules_path.exists():
            try:
                rules_text = rules_path.read_text(encoding="utf-8").strip()
            except OSError as e:
                raise RuntimeError(f"Failed to load client_prompt_rules.md: {rules_path}") from e
            if rules_text:
                base_persona += "\n\n" + rules_text + "\n"
        return base_persona

    # ------------------------------
    # Profile loading (CLI-compatible)
    # ------------------------------
    @staticmethod
    def _find_client_profiles_path() -> Path:
        env_path = (
            os.getenv("CLIENT_PROFILES_PATH")
            or os.getenv("CLIENTS_YAML_PATH")
            or os.getenv("CLIENTS_YAML")
        )
        if env_path:
            p = resolve_project_path(env_path).resolve()
            if p.is_file():
                return p
            raise FileNotFoundError(f"client_profiles.yaml not found (specified via environment variable): {p}")

        candidates = [
            APP_DIR / "client_profiles.yaml",
            resolve_config_path("client_profiles.yaml"),
            PROJECT_ROOT / "client_profiles.yaml",
            Path.cwd() / "client_profiles.yaml",
            # Compatibility (old names)
            APP_DIR / "clients.yaml",
            PROJECT_ROOT / "clients.yaml",
            Path.cwd() / "clients.yaml",
        ]
        for p in candidates:
            if p.is_file():
                return p
        tried = "\n".join([f"- {c}" for c in candidates])
        raise FileNotFoundError(
            "client_profiles.yaml was not found. Searched the following locations:\n" + tried
        )

    @staticmethod
    def _load_client_profiles_yaml(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _get_client_profile(cfg: Dict[str, Any], client_code: str) -> Dict[str, Any]:
        clients = cfg.get("clients")
        if not isinstance(clients, dict):
            raise KeyError("client_profiles.yaml does not contain a clients section.")
        code = (client_code or "").strip()
        if code not in clients:
            available = ", ".join(sorted([str(k) for k in clients.keys()]))
            raise KeyError(f"client_code={code} was not found in client_profiles.yaml. Available: {available}")
        profile = clients.get(code)
        if not isinstance(profile, dict):
            raise TypeError(f"client_profiles.yaml clients.{code} is not a dict.")
        return profile

    @staticmethod
    def _derive_client_meta(profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, str]:
        def _s(x: Any) -> str:
            return "" if x is None else str(x).strip()

        defs = cfg.get("definitions") or {}
        perma_defs = defs.get("perma") or {}
        style_defs = defs.get("interpersonal_styles") or {}
        # Prefer the new perma_patterns format and accept the old perma_focus_patterns format for compatibility.
        pattern_defs = defs.get("perma_patterns") or defs.get("perma_focus_patterns") or {}

        # pattern may include both a code and a label, such as "M1 Meaning (meaning) decline",
        # so interpret the first token as the code.
        raw_pattern = _s(profile.get("pattern"))
        pattern_code = raw_pattern.split()[0] if raw_pattern else ""
        pattern_info = pattern_defs.get(pattern_code) or {}
        if not isinstance(pattern_info, dict):
            pattern_info = {}
        # Use the defined label if present; otherwise fall back to the raw value.
        pattern_label = _s(pattern_info.get("label")) or (raw_pattern or pattern_code)

        primary_focus_code = _s(pattern_info.get("primary_focus"))
        primary_focus_label = _s(perma_defs.get(primary_focus_code)) or primary_focus_code
        # perma_patterns may omit primary_focus, so infer the primary focus from the lowest-scoring axis.
        if not primary_focus_code:
            perma_score_items: List[Tuple[str, float]] = []
            for key in ("P", "E", "R", "M", "A"):
                raw_v = pattern_info.get(key)
                if raw_v is None:
                    continue
                try:
                    perma_score_items.append((key, float(raw_v)))
                except (TypeError, ValueError):
                    continue
            if perma_score_items:
                min_score = min(v for _, v in perma_score_items)
                score_map = {k: v for k, v in perma_score_items}
                for key in ("P", "E", "R", "M", "A"):
                    if key in score_map and score_map[key] == min_score:
                        primary_focus_code = key
                        primary_focus_label = _s(perma_defs.get(key)) or key
                        break
        # Fallback when no definition exists (infer from the first code letter: P/E/R/M/A).
        if not primary_focus_code and pattern_code:
            head = (pattern_code[0] if pattern_code else "").upper()
            if head in perma_defs:
                primary_focus_code = head
                primary_focus_label = _s(perma_defs.get(head)) or head

        interpersonal_style_code = _s(profile.get("interpersonal_style"))
        style_info = style_defs.get(interpersonal_style_code) or {}
        interpersonal_label = _s(style_info.get("label")) or interpersonal_style_code

        # Collect background and attributes.
        bg = profile.get("background") or {}
        age_range = _s(bg.get("age_range")) if isinstance(bg, dict) else ""
        sex = _s(profile.get("sex"))
        marital_status = _s(profile.get("marital_status"))

        return {
            "pattern_code": pattern_code,
            "pattern_label": pattern_label,
            "primary_focus_code": primary_focus_code,
            "primary_focus_label": primary_focus_label,
            "interpersonal_style_code": interpersonal_style_code,
            "interpersonal_style_label": interpersonal_label,
            "age_range": age_range,
            "sex": sex,
            "marital_status": marital_status,
        }

    @staticmethod
    def _resolve_client_llm_style(env_style: str, interpersonal_style_code: str) -> str:
        env_style = (env_style or "").strip().lower()
        if env_style in ("cooperative", "ambivalent", "resistant"):
            return env_style
        m = {
            "s1": "cooperative",
            "s2": "resistant",
            "s3": "ambivalent",
        }
        return m.get((interpersonal_style_code or "").strip().lower(), "cooperative")

    @staticmethod
    def _build_client_scenario_text(
        client_code: str,
        profile: Dict[str, Any],
        cfg: Dict[str, Any],
        derived_meta: Dict[str, str],
    ) -> str:
        prompt_profile = SimpleClientLLM._get_prompt_profile(profile)
        if prompt_profile:
            rendered = SimpleClientLLM._render_prompt_profile_context(prompt_profile)
            if rendered:
                return rendered

        # Prefer case_overview if it has already been prepared.
        overview = profile.get("case_overview")
        if isinstance(overview, str) and overview.strip():
            return overview.strip()

        # Compose text with the minimum required structure.
        b = []
        pc = str(profile.get("presenting_concern") or "").strip()
        if pc:
            b.append(f"Presenting concern: {pc}")
        bg = profile.get("background") or {}
        if isinstance(bg, dict):
            age = str(bg.get("age_range") or "").strip()
            occ = str(bg.get("occupation_context") or "").strip()
            living = str(bg.get("living_situation") or "").strip()
            info = " / ".join([x for x in (age, occ, living) if x])
            if info:
                b.append(f"Background: {info}")
        baseline = profile.get("baseline_perma") or {}
        if isinstance(baseline, dict):
            keys = ["P", "E", "R", "M", "A"]
            vals = [f"{k}:{baseline.get(k)}" for k in keys if k in baseline]
            if vals:
                b.append("Pre-session PERMA: " + ", ".join(vals))
        strengths = profile.get("strengths_resources") or []
        if isinstance(strengths, list) and strengths:
            joined = "; ".join([str(s) for s in strengths])
            b.append(f"Strengths / resources: {joined}")
        mc = str(profile.get("maintaining_cycle") or "").strip()
        if mc:
            b.append(f"Maintaining cycle: {mc}")
        sg = str(profile.get("session_goal") or "").strip()
        if sg:
            b.append(f"Interview focus: {sg}")
        if not b:
            return f"Basic information for client {client_code}."
        return "\n".join(b)

    @staticmethod
    def _apply_trait_expression_to_client(client: "SimpleClientLLM", profile: Dict[str, Any]) -> None:
        te = profile.get("trait_expression") or {}
        if not isinstance(te, dict):
            return
        for k in ClientInternalState.TRAIT_KEYS:
            v = te.get(k)
            if v is None:
                continue
            try:
                setattr(client.internal_state, k, float(v))
            except (TypeError, ValueError):
                pass

    @dataclass
    class ClientBundle:
        style: str
        first_utterance: str
        profiles_path: Path
        derived_meta: Dict[str, str]

    @classmethod
    def from_profile(
        cls,
        *,
        client_code: str,
        llm: LLMClient,
        llm_state: Any,
        llm_reply: Any,
        env_style: str = "auto",
        first_client_utterance_env: Optional[str] = None,
        max_state_step_env: Optional[str] = None,
        default_first_utterance: str = DEFAULT_FIRST_CLIENT_UTTERANCE,
    ) -> Tuple["SimpleClientLLM", "SimpleClientLLM.ClientBundle"]:
        # Load the profile from YAML.
        profiles_path = cls._find_client_profiles_path()
        cfg = cls._load_client_profiles_yaml(profiles_path)
        profile = cls._get_client_profile(cfg, client_code)
        derived_meta = cls._derive_client_meta(profile, cfg)
        prompt_profile = cls._get_prompt_profile(profile)

        scenario = cls._build_client_scenario_text(client_code, profile, cfg, derived_meta)

        # Determine the client's response style.
        style = cls._resolve_client_llm_style(env_style, derived_meta.get("interpersonal_style_code", ""))

        # Limit the magnitude of state changes.
        # Improvement: apply a default ceiling of 0.8. Only an explicit "none" makes it unlimited.
        DEFAULT_MAX_STEP = 0.8
        max_state_step: Optional[float]
        try:
            if max_state_step_env is None or str(max_state_step_env).strip() == "":
                max_state_step = DEFAULT_MAX_STEP
            elif str(max_state_step_env).strip().lower() == "none":
                max_state_step = None
            else:
                max_state_step = float(max_state_step_env)
        except (TypeError, ValueError):
            max_state_step = DEFAULT_MAX_STEP

        # Instantiate the client.
        client = cls(
            llm=llm_reply,
            style=style,
            scenario=scenario,
            max_state_step=max_state_step,
        )
        # Keep the state LLM for the two-stage pipeline.
        try:
            client.llm_state = llm_state  # type: ignore[attr-defined]
        except Exception:
            pass
        client.prompt_profile = prompt_profile or None
        # Apply trait tendencies.
        cls._apply_trait_expression_to_client(client, profile)

        # Determine the initial client utterance.
        first_env = (first_client_utterance_env or "").strip()
        presenting = str(profile.get("presenting_concern") or "").strip()

        # Generation policy: env > auto (try LLM generation, then fall back to presenting concern).
        mode = str(os.getenv("CLIENT_FIRST_UTTERANCE_MODE", "auto")).strip().lower()
        generate = True if mode in ("llm", "on", "true") else False if mode in ("presenting", "off", "false") else True

        if first_env:
            first = first_env
        elif generate and llm_reply is not None:
            first = cls._generate_first_utterance(
                llm=llm_reply,
                persona=client.persona or "",
                presenting_concern=presenting,
                style=style,
                first_utterance_seed=(
                    prompt_profile.get("first_utterance_seed")
                    if isinstance(prompt_profile, Mapping)
                    else None
                ),
            ) or (presenting or default_first_utterance)
        else:
            first = presenting or default_first_utterance

        bundle = cls.ClientBundle(
            style=style,
            first_utterance=first,
            profiles_path=profiles_path,
            derived_meta=derived_meta,
        )
        return client, bundle

    # ------------------------------
    # LLM generation for the first utterance
    # ------------------------------
    @staticmethod
    def _generate_first_utterance(
        *,
        llm: Any,
        persona: str,
        presenting_concern: str,
        style: str,
        first_utterance_seed: Optional[Mapping[str, Any]] = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a natural first utterance for the initial intake, using presenting_concern as input.
        Returns an empty string on failure so the caller can fall back.
        """
        try:
            sys = (
                (persona or "").strip()
                + "\n\n[Task]\n"
                "You are the client in an initial intake session. Based on the difficulties below,\n"
                "generate a short 1-3 sentence opening utterance in natural English.\n"
                "- Do not give evaluations, advice, summaries, or self-solved solutions\n"
                "- Do not praise or instruct the therapist\n"
                "- It is okay to include hesitation or ambiguity\n"
                "- Do not stop at abstract concerns; if possible, naturally include one concrete life fact or scene\n"
                "- Output text only (no JSON or explanation)\n"
                "- Write only natural English in the body. Do not mix in slang, placeholders, or untranslated tokens except proper nouns or common abbreviations\n"
                "- Even if the input contains mixed or broken wording, do not repeat it verbatim; rephrase naturally in English when possible\n"
            )
            seed_lines: List[str] = []
            if isinstance(first_utterance_seed, Mapping):
                concern_core = SimpleClientLLM._string_list(first_utterance_seed.get("concern_core"))
                concrete_facts = SimpleClientLLM._string_list(
                    first_utterance_seed.get("concrete_fact_candidates")
                )
                scenes = SimpleClientLLM._string_list(first_utterance_seed.get("scene_candidates"))
                if concern_core:
                    seed_lines.append("Core concern: " + " / ".join(concern_core[:2]))
                if concrete_facts:
                    seed_lines.append(
                        "At most one life fact to include: " + " / ".join(concrete_facts[:3])
                    )
                if scenes:
                    seed_lines.append(
                        "At most one concrete scene to include: " + " / ".join(scenes[:2])
                    )
                seed_lines.append("Do not cram too many facts into one utterance")
            user_parts = [
                "Presenting concern: " + (presenting_concern or ""),
            ]
            if seed_lines:
                user_parts.append("[First-utterance hints]\n" + "\n".join(seed_lines))
            user = "\n".join(user_parts).strip()
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ]
            text = llm.generate(messages, temperature=temperature)
            text = (text or "").strip()
            # Remove stray quotes and markdown wrappers (minor cleanup).
            if text.startswith("\"") and text.endswith("\""):
                text = text[1:-1].strip()
            if text.startswith("`") and text.endswith("`"):
                text = text.strip("`")
            # Lightly trim to 1-3 sentences (simple splitting).
            parts = [p.strip() for p in re.split(r"[\n]+", text) if p.strip()]
            if len(parts) >= 1:
                text = parts[0]
            return text
        except Exception:
            return ""

    @staticmethod
    def _clip_0_10(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 10.0:
            return 10.0
        return x

    def _round_state(self, x: float) -> float:
        try:
            nd = int(self.state_decimal_places)
        except (TypeError, ValueError):
            nd = 2
        return round(float(x), nd)

    def _get_sensitivity(self, state_key: str) -> float:
        """
        Return the per-metric ease-of-change coefficient.
        - 1.0: baseline
        - 0.0: no change
        - >1.0: easier to change

        Negative values are clipped to 0.0 because they would invert the direction of change.
        """
        mapping = {
            "pos_affect": "sensitivity_pos",
            "neg_affect": "sensitivity_neg",
            "importance_change": "sensitivity_importance",
            "confidence_change": "sensitivity_confidence",
            "like_counselor": "sensitivity_like",
            "tension_counselor": "sensitivity_tension",
        }
        attr = mapping.get(state_key)
        if not attr:
            return 1.0
        try:
            val = float(getattr(self, attr))
        except (TypeError, ValueError):
            val = 1.0
        if val < 0.0:
            val = 0.0
        return val

    @staticmethod
    def _calc_relationship_deltas(counselor_text: str) -> Tuple[float, float, List[str]]:
        """
        Estimate small like/tension changes from the tone and content of counselor_text.
        This is a fallback when the LLM does not move like/tension.

        Returns:
          (delta_like, delta_tension, hits)
        """
        t = str(counselor_text or "").strip()
        if not t:
            return 0.0, 0.0, []

        rules: List[Tuple[str, float, float, str]] = [
            # Empathy / acceptance (like↑ tension↓)
            ("that's tough", +0.60, -0.30, "empathy"),
            ("that's hard", +0.60, -0.30, "empathy"),
            ("that sounds hard", +0.60, -0.30, "empathy"),
            ("i get it", +0.70, -0.30, "empathy"),
            ("i understand", +0.70, -0.30, "empathy"),
            ("thank you", +0.40, -0.20, "respect"),
            ("that's good", +0.70, -0.30, "affirm"),
            ("great", +0.90, -0.40, "affirm"),
            ("effort", +0.50, -0.20, "affirm"),
            ("trying hard", +0.50, -0.20, "affirm"),
            # Denial / criticism / contempt (like↓ tension↑)
            ("not trying hard enough", -1.40, +1.00, "blame"),
            ("excuse", -1.10, +0.80, "blame"),
            ("trouble", -1.60, +1.20, "hostile"),
            ("messy", -1.20, +0.90, "dismiss"),
            ("impossible", -0.80, +0.60, "dismiss"),
            ("i don't understand", -1.30, +1.00, "reject"),
            ("can't understand", -1.30, +1.00, "reject"),
            ("idiot", -2.00, +2.00, "insult"),
            ("stop being weak", -1.60, +1.40, "insult"),
            # Commands / pressure (light tension increase)
            ("should", -0.60, +0.40, "directive"),
            ("have no choice", -0.40, +0.30, "directive"),
            ("must", -0.80, +0.60, "directive"),
        ]

        dl = 0.0
        dt = 0.0
        hits: List[str] = []
        for phrase, r_dl, r_dt, tag in rules:
            if phrase in t:
                dl += r_dl
                dt += r_dt
                hits.append(tag + ":" + phrase)

        # Slight punctuation-based adjustment (tends to intensify wording)
        if "!" in t or "！" in t:
            dt += 0.20
            hits.append("punct:!")
        # "?" can still carry a little pressure, so apply only a tiny adjustment.
        if "?" in t or "？" in t:
            dt += 0.05
            hits.append("punct:?")

        return dl, dt, hits

    def _apply_relationship_heuristic(
        self,
        new_state_dict: Dict[str, float],
        counselor_text: str,
        meta: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Safety net for cases where like/tension do not move at all.
        Respect changes made by the LLM and only apply a small correction when unchanged.
        """
        if not self.relationship_heuristic:
            return new_state_dict

        # Current values (before the update).
        try:
            old_like = float(self.internal_state.like_counselor)
        except (TypeError, ValueError):
            old_like = 5.0
        try:
            old_tension = float(self.internal_state.tension_counselor)
        except (TypeError, ValueError):
            old_tension = 0.0

        # Values after LLM parsing.
        like_val = new_state_dict.get("like_counselor", old_like)
        tension_val = new_state_dict.get("tension_counselor", old_tension)
        try:
            like_val = float(like_val)
        except (TypeError, ValueError):
            like_val = old_like
        try:
            tension_val = float(tension_val)
        except (TypeError, ValueError):
            tension_val = old_tension

        # Change detection (respect changes made by the LLM).
        eps = 1e-9
        like_changed = abs(like_val - old_like) > eps
        tension_changed = abs(tension_val - old_tension) > eps

        # If only_if_unchanged is set, do not correct fields that already moved.
        if self.relationship_heuristic_only_if_unchanged:
            apply_like = not like_changed
            apply_tension = not tension_changed
        else:
            apply_like = True
            apply_tension = True

        if not apply_like and not apply_tension:
            return new_state_dict

        dl_raw, dt_raw, hits = self._calc_relationship_deltas(counselor_text)
        scale = float(getattr(self, "delta_scale", 1.0))

        dl = 0.0
        dt = 0.0

        if apply_like:
            target_like = like_val + dl_raw
            adjusted_like = self.internal_state.adjust_by_expression("like_counselor", like_val, target_like)
            sens_like = self._get_sensitivity("like_counselor")
            dl = (adjusted_like - like_val) * scale * sens_like

        if apply_tension:
            target_tension = tension_val + dt_raw
            adjusted_tension = self.internal_state.adjust_by_expression("tension_counselor", tension_val, target_tension)
            sens_tension = self._get_sensitivity("tension_counselor")
            dt = (adjusted_tension - tension_val) * scale * sens_tension

        # Respect the per-turn ceiling.
        max_step = self.max_state_step
        if max_step is not None:
            try:
                ms = float(max_step)
            except (TypeError, ValueError):
                ms = None
            if ms is not None:
                if apply_like:
                    if dl > ms:
                        dl = ms
                    elif dl < -ms:
                        dl = -ms
                if apply_tension:
                    if dt > ms:
                        dt = ms
                    elif dt < -ms:
                        dt = -ms

        if apply_like:
            like_val = self._clip_0_10(like_val + dl)
            like_val = self._round_state(like_val)
            new_state_dict["like_counselor"] = like_val

        if apply_tension:
            tension_val = self._clip_0_10(tension_val + dt)
            tension_val = self._round_state(tension_val)
            new_state_dict["tension_counselor"] = tension_val

        meta["relationship_heuristic"] = {
            "applied_like": bool(apply_like),
            "applied_tension": bool(apply_tension),
            "sensitivity_like": self._round_state(self._get_sensitivity("like_counselor")),
            "sensitivity_tension": self._round_state(self._get_sensitivity("tension_counselor")),
            "delta_like": self._round_state(dl),
            "delta_tension": self._round_state(dt),
            "hits": hits,
        }
        return new_state_dict

    # ------------------------------
    # Two-stage pipeline: state -> reply
    # ------------------------------
    def _infer_state_from_trigger(
        self,
        *,
        trigger_text: str,
        history: List[Any],
        state_dict_before: Dict[str, float],
        trait_dict: Dict[str, float],
        transition: Literal["after_listen", "after_speak"],
        delta_scale_override: Optional[float] = None,
        max_state_step_override: Optional[float] = None,
    ) -> tuple[Dict[str, float], Dict[str, str], Dict[str, Any], str]:
        """
        Update internal_state once in response to the specified trigger
        (counselor utterance or self utterance).
        """
        raw_state = "{}"
        if self.llm_state is None:
            return (
                dict(state_dict_before),
                {},
                {"parse_status": "state_llm_missing", "transition": transition},
                raw_state,
            )

        if transition == "after_listen":
            transition_instruction = (
                "Transition type: after_listen (right after hearing the counselor).\n"
                "Infer the updated internal_state based on the most recent counselor utterance."
            )
            trigger_role = "user"
        else:
            transition_instruction = (
                "Transition type: after_speak (right after verbalizing the client's own utterance).\n"
                "Infer an internal_state that reflects small changes in self-perception and emotion based on the utterance you just produced."
            )
            trigger_role = "assistant"

        state_system = (
            (self.persona or "")
            + "\n\n"
            "[About your internal state scores]\n"
            "You have the following six internal state scores (0-10).\n"
            "- pos_affect / neg_affect / importance_change / confidence_change / like_counselor / tension_counselor\n"
            "- The current scores, traits, and transition type are provided in the user message.\n\n"
            "[Output (state only)]\n"
            "Return the updated internal_state, and a reason if needed, as JSON.\n"
            "Output only the following format: {\n"
            "  \"internal_state\": {six numeric fields},\n"
            "  \"internal_state_reason\": {...(optional)}\n"
            "}\n"
            "Do not output any other fields such as reply.\n"
        )
        state_context_user = (
            "[Runtime context (updated every turn)]\n"
            f"Current scores: {json.dumps(state_dict_before, ensure_ascii=False)}\n"
            "[Score expression tendencies (-1: hard to surface, 0: neutral, +1: easy to surface)]\n"
            f"{json.dumps(trait_dict, ensure_ascii=False)}\n"
            f"{transition_instruction}\n"
            f"[Current trigger utterance] {trigger_text}\n"
            "Based on the above, return only the internal_state as JSON."
        )

        messages_state: List[Dict[str, str]] = [{"role": "system", "content": state_system}]
        for turn in history[-self.max_history_turns :]:
            speaker = getattr(turn, "speaker", "")
            text = getattr(turn, "text", "")
            if not isinstance(text, str):
                continue
            role = "user" if speaker == "counselor" else "assistant"
            messages_state.append({"role": role, "content": text})

        expected_last_speaker = "counselor" if trigger_role == "user" else "client"
        if not (
            history
            and getattr(history[-1], "speaker", None) == expected_last_speaker
            and str(getattr(history[-1], "text", "")).strip() == str(trigger_text).strip()
        ):
            messages_state.append({"role": trigger_role, "content": trigger_text})
        messages_state.append({"role": "user", "content": state_context_user})

        try:
            raw_state = self.llm_state.generate(messages_state, temperature=float(self.temperature))  # type: ignore[union-attr]
        except (TypeError, ValueError):
            raw_state = self.llm_state.generate(messages_state)  # type: ignore[union-attr]
        except Exception as e:
            return (
                dict(state_dict_before),
                {},
                {"parse_status": "state_llm_error", "error": str(e), "transition": transition},
                "{}",
            )

        parsed_state, state_reason, meta = self._parse_state_only(
            str(raw_state),
            delta_scale_override=delta_scale_override,
            max_state_step_override=max_state_step_override,
        )
        meta["transition"] = transition
        return parsed_state, state_reason, meta, str(raw_state)

    def _respond_two_stage(self, counselor_text: str, history: List[Any]) -> str:
        # 1) State update right after listening (state-only JSON)
        state_dict_before = self.internal_state.to_dict()
        trait_dict = self.internal_state.to_traits_dict()
        listen_state_dict, state_reason_listen, meta_listen, raw_state_listen = self._infer_state_from_trigger(
            trigger_text=counselor_text,
            history=history,
            state_dict_before=state_dict_before,
            trait_dict=trait_dict,
            transition="after_listen",
        )
        listen_state_dict = self._apply_relationship_heuristic(listen_state_dict, counselor_text, meta_listen)
        self.internal_state = ClientInternalState.from_dict(listen_state_dict, base=self.internal_state)
        state_after_listen = self.internal_state.to_full_dict()

        # 2) Reply generation (reply only, plain text)
        state_dict_after = self.internal_state.to_dict()
        # Build a soft guide based on the internal state.
        soft_guide = self._build_motivation_soft_guide(state_dict_after)
        turn_hint = ""
        if isinstance(self.prompt_profile, Mapping):
            turn_hint = self._build_turn_specific_prompt_hint(self.prompt_profile, history)

        reply_system = (
            (self.persona or "")
            + "\n\n"
            "[Your current state]\n"
            "- The current scores and the utterance tone guide are provided in the user message.\n"
            "[Soft utterance tone guide (applied automatically)]\n"
            "- Prioritize and reflect the guide in the user message.\n"
            "[Request]\n"
            f"- Return the client's next utterance in {REPLY_SENTENCE_MIN}-{REPLY_SENTENCE_MAX} sentences, with content and tone consistent with the updated internal_state.\n"
            "- Do not use JSON; return text only.\n"
            "- Before outputting, check that the reply does not conflict with internal_state and match the strength of the statement to the state.\n"
            "- Write only natural English in the reply body. Do not mix in slang, romanization, placeholders, or internal labels except proper nouns or common abbreviations.\n"
            "- If the immediate prior utterance contains mixed or broken wording, do not echo it verbatim; rewrite it naturally in English when possible, and if it is unclear, focus only on the surrounding experience.\n"
        )
        reply_context_user = (
            "[Runtime context (updated every turn)]\n"
            f"Current scores: {json.dumps(state_dict_after, ensure_ascii=False)}\n"
            "[Soft utterance tone guide (applied automatically)]\n"
            f"{soft_guide}\n"
            "[Concrete hints allowed in this turn]\n"
            f"{turn_hint}\n"
            "- Do not end with only abstract feelings; if possible, include one concrete scene from work, life, another person, or a time of day.\n"
            "- Limit yourself to one concrete fact and one concrete scene.\n"
            "- Use the person's own wording rather than technical or analytical language.\n"
            f"[Counselor utterance received this turn] {counselor_text}\n"
            "Return only the body of a natural client response that is consistent with the above."
        )

        messages_reply: List[Dict[str, str]] = [{"role": "system", "content": reply_system}]
        for turn in history[-self.max_history_turns :]:
            speaker = getattr(turn, "speaker", "")
            text = getattr(turn, "text", "")
            if not isinstance(text, str):
                continue
            role = "user" if speaker == "counselor" else "assistant"
            messages_reply.append({"role": role, "content": text})
        if not (
            history and getattr(history[-1], "speaker", None) == "counselor" and getattr(history[-1], "text", "").strip() == counselor_text.strip()
        ):
            messages_reply.append({"role": "user", "content": counselor_text})
        messages_reply.append({"role": "user", "content": reply_context_user})

        try:
            raw_reply = self.llm.generate(messages_reply, temperature=float(self.temperature))
        except (TypeError, ValueError):
            raw_reply = self.llm.generate(messages_reply)
        except Exception as e:
            # Fall back to a safe short response if the reply LLM fails.
            raw_reply = ""
            fallback = self._fallback_reply(counselor_text, state_dict_after)
            reply_text = fallback
        else:
            reply_text = str(raw_reply or "").strip().strip("`\"")

        # Sanitize any JSON fragments that slip into the reply.
        reply_text = self._strip_json_fragments(reply_text)

        # Lightly post-process the reply for consistency with internal_state (minimal rewording, e.g. strong -> softer).
        reply_text, consistency_meta = self._postprocess_reply_consistency(reply_text, state_dict_after)

        # 3) State update right after speaking (self-talk)
        speak_state_before = self.internal_state.to_dict()
        speak_trait_dict = self.internal_state.to_traits_dict()
        speak_state_dict, state_reason_speak, meta_speak, raw_state_speak = self._infer_state_from_trigger(
            trigger_text=reply_text,
            history=history,
            state_dict_before=speak_state_before,
            trait_dict=speak_trait_dict,
            transition="after_speak",
            delta_scale_override=self.post_reply_delta_scale,
            max_state_step_override=self.post_reply_max_state_step,
        )
        self.internal_state = ClientInternalState.from_dict(speak_state_dict, base=self.internal_state)
        state_after_speak = self.internal_state.to_full_dict()

        final_reason = state_reason_speak if state_reason_speak else state_reason_listen
        final_meta = dict(meta_speak or {})
        final_meta["pipeline"] = "two_stage"
        final_meta["consistency"] = consistency_meta

        # Store debug information.
        self._last_debug_info = {
            "raw_state": str(raw_state_speak or raw_state_listen),
            "raw_state_after_listen": str(raw_state_listen),
            "raw_state_after_speak": str(raw_state_speak),
            "raw_reply": str(raw_reply),
            "reply": reply_text,
            "old_state": state_dict_before,
            "state_after_listen": state_after_listen,
            "state_after_speak": state_after_speak,
            "new_state": self.internal_state.to_full_dict(),
            "internal_state_reason_after_listen": state_reason_listen,
            "internal_state_reason_after_speak": state_reason_speak,
            "internal_state_reason": final_reason,
            "meta_after_listen": meta_listen,
            "meta_after_speak": meta_speak,
            "meta": final_meta,
        }

        return reply_text

    # ------------------------------
    # Soft rules: speaking hints based on motivation and confidence
    # ------------------------------
    def _build_motivation_soft_guide(self, state: Dict[str, float]) -> str:
        """
        Generate lightweight tone-adjustment hints based on importance and confidence thresholds.
        The rules prioritize not being too directive and avoid forcing fixed vocabulary.

        Default thresholds:
          - importance: low<=3, mid=4-7, high>=8
          - confidence: low<=3, mid=4-7, high>=8
        """
        try:
            imp = float(state.get("importance_change", 3.0))
        except Exception:
            imp = 3.0
        try:
            conf = float(state.get("confidence_change", 2.0))
        except Exception:
            conf = 2.0

        # Thresholds (can be adjusted via .env in the future if needed).
        IMP_LOW, IMP_HIGH = 3.0, 8.0
        CONF_LOW, CONF_HIGH = 3.0, 8.0

        def _band(x: float, low: float, high: float) -> str:
            if x <= low:
                return "low"
            if x >= high:
                return "high"
            return "mid"

        imp_b = _band(imp, IMP_LOW, IMP_HIGH)
        conf_b = _band(conf, CONF_LOW, CONF_HIGH)

        hints: List[str] = []

        # Hints based on confidence.
        if conf_b == "low":
            hints.append("Low confidence (≤3): reduce definitive statements and naturally include hesitation, conditional language, and hold-back expressions.")
            hints.append("Do not repeat the same hesitation words; vary endings and openings a little each time.")
        elif conf_b == "mid":
            hints.append("Medium confidence (4-7): emphasize trying something small and moving forward conditionally.")
        else:  # high
            hints.append("High confidence (≥8): it is fine to express intent to act and concretize things more clearly.")

        # Hints based on importance.
        if imp_b == "low":
            hints.append("Low importance (≤3): include low priority and hesitation, and soften urgent or duty-heavy wording.")
        elif imp_b == "mid":
            hints.append("Medium importance (4-7): show a small reason or value while keeping the scale modest.")
        else:  # high
            hints.append("High importance (≥8): it is okay to state the value or meaning of the effort more clearly.")

        return "\n".join("- " + h for h in hints)

    @staticmethod
    def _pick_variant(candidates: List[str], anchor: str) -> str:
        """
        Deterministically choose a candidate based on the text.
        No randomness: the same input always yields the same output.
        """
        if not candidates:
            return ""
        idx = sum(ord(ch) for ch in str(anchor)) % len(candidates)
        return candidates[idx]

    def _get_style_state_baseline(self, style: str) -> Dict[str, float]:
        """
        Return the default initial internal state for the given CLIENT_STYLE
        (cooperative / ambivalent / resistant). Unknown values fall back to cooperative.
        """
        s = (style or "cooperative").strip().lower()
        if s == "cooperative":
            return {
                "pos_affect": 5.0,
                "neg_affect": 3.0,
                "importance_change": 3.0,
                "confidence_change": 2.0,
                "like_counselor": 5.0,
                "tension_counselor": 0.0,
            }
        if s == "ambivalent":
            return {
                "pos_affect": 5.0,
                "neg_affect": 5.0,
                "importance_change": 3.0,
                "confidence_change": 1.0,
                "like_counselor": 5.0,
                "tension_counselor": 3.0,
            }
        if s == "resistant":
            return {
                "pos_affect": 1.0,
                "neg_affect": 8.0,
                "importance_change": 1.0,
                "confidence_change": 1.0,
                "like_counselor": 1.0,
                "tension_counselor": 5.0,
            }
        # Fallback
        return {
            "pos_affect": 5.0,
            "neg_affect": 3.0,
            "importance_change": 3.0,
            "confidence_change": 2.0,
            "like_counselor": 5.0,
            "tension_counselor": 0.0,
        }

    def _strip_json_fragments(self, text: str) -> str:
        """
        Simple filter that removes accidental JSON objects/arrays from the reply.
        Drop lines that are entirely {…} / […] and join the rest back together.
        """
        lines = [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]
        cleaned: List[str] = []
        for ln in lines:
            t = ln.strip()
            if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
                continue
            cleaned.append(t)
        return "\n".join(cleaned).strip()

    def _fallback_reply(self, counselor_text: str, state: Dict[str, float]) -> str:
        """
        Safe short fallback when the reply LLM fails.
        Uses a modest 1-2 sentence phrasing based on confidence and importance ranges.
        """
        imp = float(state.get("importance_change", 3.0) or 3.0)
        conf = float(state.get("confidence_change", 2.0) or 2.0)
        if conf <= 3.0 and imp <= 3.0:
            return "Hmm, I still cannot decide clearly. I want to think while watching things on a small scale."
        if conf <= 3.0 and imp > 3.0:
            return "I do have some desire to try, but I cannot quite decide. I want to start by looking for something I can try without pushing too hard."
        if conf > 7.5 and imp > 7.5:
            return "I will try just one small thing first. I want to see whether I can keep it up."
        return "I do feel a little more positive, but I want to look for something I can try within a reasonable range."

    def _shorten_to_max_sentences(self, text: str, max_sentences: int = REPLY_SENTENCE_MAX) -> str:
        # Roughly split on sentence endings and trim to the maximum number of sentences (simple processing).
        s = re.split(r"[。！？\n]+", text)
        s = [x.strip() for x in s if x.strip()]
        if not s:
            return text.strip()
        cut = s[:max_sentences]
        res = "。".join(cut)
        if not res.endswith("。"):
            res += "。"
        return res

    def _postprocess_reply_consistency(self, reply: str, state: Dict[str, float]) -> tuple[str, Dict[str, Any]]:
        """
        Post-filter that lightly rephrases expressions that strongly conflict with internal state.
        - confidence<=3 with strong commitment (e.g. "I'll try it", "I'll do it", "I've decided") -> soften.
        - importance<=3 with urgent / definitive / duty-heavy wording (e.g. "must", "immediately", "absolutely") -> soften.
        Keep replacements minimal and trim to at most four sentences.
        """
        meta: Dict[str, Any] = {"applied": False, "rules": [], "original": reply}
        text = reply
        try:
            imp = float(state.get("importance_change", 3.0))
        except Exception:
            imp = 3.0
        try:
            conf = float(state.get("confidence_change", 2.0))
        except Exception:
            conf = 2.0

        # Soften strong commitment language when confidence is low.
        if conf <= 3.0:
            patterns: Dict[str, List[str]] = {
                "i'll try": [
                    "I want to think about whether I can really try it a little more.",
                    "I want to see whether I can test it within a range that feels manageable.",
                    "I want to think about trying it in a very small way first.",
                ],
                "i'll do it": [
                    "I am still unsure whether I can really do it.",
                    "If I do it, I want to think about it within a range that does not push too hard.",
                    "I want to hold off on deciding for now.",
                ],
                "i'll incorporate it": [
                    "I have not fully decided whether I can bring it in yet.",
                    "If I bring it in, I want to try only part of it first.",
                    "I want to watch and see a little longer before deciding.",
                ],
                "i've decided": [
                    "I have not fully decided yet.",
                    "I have a direction in mind, but I am still uncertain about the final choice.",
                    "I need a little more time before deciding.",
                ],
                "i'll continue": [
                    "I am not yet confident I can keep it up.",
                    "If I continue, I want to do it in a lighter way.",
                    "I want to watch how it goes a little before deciding whether to continue.",
                ],
                "i'll move forward": [
                    "I want to think a little more about whether I can really move forward with it.",
                    "If I move forward, I want to do it at a pace that does not feel too heavy.",
                    "I am still wavering about whether to move forward.",
                ],
                "i'll work on it": [
                    "There are still parts I do not fully understand yet.",
                    "If I work on it, I want to start very small.",
                    "I want to leave the decision open for now.",
                ],
            }
            applied_local = False
            for k, variants in patterns.items():
                if k in text:
                    replacement = self._pick_variant(variants, f"{text}|{k}|{conf:.2f}")
                    text = text.replace(k, replacement)
                    applied_local = True
            if applied_local:
                meta["applied"] = True
                meta["rules"].append("confidence_low_soften_commit")

        # Soften strong duty or urgency language when importance is low.
        if imp <= 3.0:
            replacements: Dict[str, List[str]] = {
                "must": ["if possible", "if it fits", "within a range that feels manageable"],
                "absolutely": ["as much as possible", "if I can", "if it fits"],
                "right away": ["first", "for now", "after watching a little"],
                "soon": ["if I can do it early", "as early as possible", "without rushing"],
                "should": ["might be worth trying", "I would want to consider it if needed", "I could try it if it feels possible"],
            }
            applied_local = False
            for k, variants in replacements.items():
                if k in text:
                    replacement = self._pick_variant(variants, f"{text}|{k}|{imp:.2f}")
                    text = text.replace(k, replacement)
                    applied_local = True
            if applied_local:
                meta["applied"] = True
                meta["rules"].append("importance_low_soften_obligation")

        # Client-like voice (suppress counselor-like phrasing).
        text, voice_meta = self._enforce_client_voice(text)

        # Trim to the maximum number of sentences.
        new_text = self._shorten_to_max_sentences(text, REPLY_SENTENCE_MAX)
        if new_text != reply:
            meta["applied"] = True
            meta["rules"].append("shorten_sentences")
        if voice_meta.get("applied"):
            meta["applied"] = True
            meta["rules"].append("client_voice_enforced")
            meta["voice"] = voice_meta

        meta["result"] = new_text
        return new_text, meta

    def _enforce_client_voice(self, text: str) -> tuple[str, Dict[str, Any]]:
        """
        If counselor-like phrasing (permission-seeking, suggestions, summary bullets, meta explanations, etc.) slips in,
        lightly rewrite it into a short first-person client utterance.
        - Keep replacements minimal. Avoid over-transforming the meaning.
        - Only apply strong normalization when counselor-like phrasing is clearly detected.
        """
        original = text
        applied = False
        notes: List[str] = []
        counselor_like_score = 0

        # Remove obvious meta lines and role labels.
        lines = [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]
        list_marker_count = 0
        clean_lines: List[str] = []
        for ln in lines:
            if re.search(r"^(meta:|T:|therapist:|counselor:)", ln.strip(), flags=re.IGNORECASE):
                applied = True
                counselor_like_score += 2
                notes.append("remove_meta_role_lines")
                continue
            if re.search(r"^\s*(?:[-・*]|\d+[\.)）]?)\s*", ln):
                list_marker_count += 1
            clean_lines.append(ln)

        # Remove bullets only when there are multiple bullet lines (keep isolated markers).
        if list_marker_count >= 2:
            counselor_like_score += 1
            stripped_lines: List[str] = []
            for ln in clean_lines:
                new_ln = re.sub(r"^\s*(?:[-・*]|\d+[\.)）]?)\s*", "", ln)
                stripped_lines.append(new_ln)
            clean_lines = stripped_lines
            applied = True
            notes.append("strip_list_markers")

        if clean_lines:
            text = "。".join([s.strip("。 ") for s in clean_lines])

        # Detect counselor-style stock phrases.
        counselor_markers = [
            "may i",
            "shall we proceed",
            "let's sort it out together",
            "i can help organize it",
            "i suggest",
            "to summarize",
            "in short",
        ]
        counselor_like_score += sum(1 for marker in counselor_markers if marker in text)

        # Apply stronger normalization only when multiple counselor-like signals are present.
        if counselor_like_score >= 2:
            replacements: Dict[str, List[str]] = {
                "may i": [
                    "If it is okay, I would like to talk about it a little.",
                    "I am a little unsure about whether I should talk about it.",
                ],
                "shall we proceed": [
                    "I am a little unsure right now.",
                    "I have not fully decided how to move forward yet.",
                ],
                "let's sort it out together": [
                    "I still cannot sort it out clearly.",
                    "I am having trouble organizing it well.",
                ],
                "i can help organize it": [
                    "I am not organizing it very well yet.",
                    "I do want to sort it out, but I am still wavering.",
                ],
                "i suggest": [
                    "I have not decided yet.",
                    "I have some thoughts, but I cannot say them clearly yet.",
                ],
                "to summarize": ["for now", "what I am feeling right now is"],
                "in short": ["for now", "what I am feeling right now is"],
            }
            local_applied = False
            for k, variants in replacements.items():
                if k in text:
                    replacement = self._pick_variant(variants, f"{original}|{k}|voice")
                    text = text.replace(k, replacement)
                    local_applied = True
            if local_applied:
                applied = True
                notes.append("replace_counselor_phrases")

            # Typical coaching directive "let's do it" -> a client's hesitation.
            if re.search(r"\blet(?:'s| us)\b[.\s]*$", text, flags=re.IGNORECASE) and not re.search(r"[?؟?]$", text):
                tail_variants = [
                    "I want to watch and see a little longer before deciding.",
                    "I am not ready to say that with confidence yet.",
                ]
                replacement = self._pick_variant(tail_variants, f"{original}|let_us")
                text = re.sub(r"\blet(?:'s| us)\b[.\s]*$", replacement, text, flags=re.IGNORECASE)
                applied = True
                notes.append("soften_let_us_do")

        return text, {"applied": applied, "notes": notes, "original": original, "score": counselor_like_score}

    def _parse_state_only(
        self,
        raw: str,
        *,
        delta_scale_override: Optional[float] = None,
        max_state_step_override: Optional[float] = None,
    ) -> tuple[Dict[str, float], Dict[str, str], Dict[str, Any]]:
        """
        Parse state-only JSON and return a new state after applying conversion, limits, and clipping.
        On error, return the current state unchanged.
        """
        text = str(raw).strip()
        meta: Dict[str, Any] = {"parse_status": "ok_state"}

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return self.internal_state.to_full_dict(), {}, {"parse_status": "state_fallback_no_brace"}
            try:
                data = json.loads(m.group(0))
                meta["parse_status"] = "state_fallback_brace"
            except json.JSONDecodeError:
                return self.internal_state.to_full_dict(), {}, {"parse_status": "state_fallback_json_error"}

        if not isinstance(data, dict):
            return self.internal_state.to_full_dict(), {}, {"parse_status": "state_fallback_non_dict"}

        state_in = data.get("internal_state") or {}
        merged_state = self.internal_state.to_dict()
        provided_keys: List[str] = []

        if isinstance(state_in, dict):
            for key in merged_state.keys():
                v = state_in.get(key)
                if v is None:
                    continue
                try:
                    merged_state[key] = float(v)
                    provided_keys.append(key)
                except (TypeError, ValueError):
                    continue

        meta["state_keys_provided"] = provided_keys

        # From here on, shared state-update calculations (clipping, distortion, limits, etc.).
        max_step = self.max_state_step if max_state_step_override is None else max_state_step_override
        current = self.internal_state.to_dict()

        try:
            nd = int(self.state_decimal_places)
        except (TypeError, ValueError):
            nd = 2

        scale_src = delta_scale_override if delta_scale_override is not None else getattr(self, "delta_scale", 1.0)
        try:
            scale = float(scale_src)
        except (TypeError, ValueError):
            scale = 1.0
        if scale < 0.0:
            scale = 0.0
        meta["delta_scale_used"] = self._round_state(scale)
        meta["max_state_step_used"] = max_step
        meta["sensitivity"] = {
            "pos_affect": self._round_state(self._get_sensitivity("pos_affect")),
            "neg_affect": self._round_state(self._get_sensitivity("neg_affect")),
            "importance_change": self._round_state(self._get_sensitivity("importance_change")),
            "confidence_change": self._round_state(self._get_sensitivity("confidence_change")),
            "like_counselor": self._round_state(self._get_sensitivity("like_counselor")),
            "tension_counselor": self._round_state(self._get_sensitivity("tension_counselor")),
        }

        for k, v in list(merged_state.items()):
            try:
                target = float(v)
            except (TypeError, ValueError):
                continue

            target = self._clip_0_10(target)
            try:
                before = float(current.get(k, target))
            except (TypeError, ValueError):
                before = target

            adjusted = self.internal_state.adjust_by_expression(k, before, target)
            sensitivity = self._get_sensitivity(k)
            delta = (adjusted - before) * scale * sensitivity

            if max_step is not None:
                try:
                    ms = float(max_step)
                except (TypeError, ValueError):
                    ms = None
                if ms is not None:
                    if delta > ms:
                        delta = ms
                    elif delta < -ms:
                        delta = -ms

            new_value = before + delta
            new_value = self._clip_0_10(new_value)
            new_value = round(float(new_value), nd)
            merged_state[k] = new_value

        merged_full = self.internal_state.to_full_dict()
        merged_full.update(merged_state)

        state_reason_in = data.get("internal_state_reason") or {}
        state_reason: Dict[str, str] = {}
        if isinstance(state_reason_in, dict):
            for k, v in state_reason_in.items():
                if v is None:
                    continue
                state_reason[str(k)] = str(v)

        return merged_full, state_reason, meta

    # ------------------------------
    # Reply + internal state update
    # ------------------------------
    def respond(self, counselor_text: str, history: List[Any]) -> str:
        """
        Always respond using the two-stage pipeline (state -> reply).
        """
        return self._respond_two_stage(counselor_text, history)
