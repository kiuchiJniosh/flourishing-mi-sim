from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from .paths import resolve_config_path, resolve_project_path

logger = logging.getLogger(__name__)

DEFAULT_MI_KNOWLEDGE_PATH = resolve_config_path("mi_knowledge.md")

_SECTION_COMMON = "Common Knowledge"
_SECTION_PHASE = "Phase-Related Knowledge"
_SECTION_REFLECT = "REFLECT_SIMPLE / REFLECT_COMPLEX / REFLECT_DOUBLE (reflection)"
_SECTION_AFFIRM = "AFFIRM (affirmation)"
_SECTION_QUESTION = "QUESTION (questioning)"
_SECTION_SUMMARY = "SUMMARY (summary)"
_SECTION_INFO = "ASK_PERMISSION_TO_SHARE_INFO / PROVIDE_INFO (information sharing)"
_COMMON_SUBSECTION_CHANGE_TALK = "Understanding Change Talk"

_SECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
    _SECTION_COMMON: ("\u5171\u901a\u306e\u77e5\u8b58",),
    _SECTION_PHASE: ("\u5404\u30d5\u30a7\u30fc\u30ba\u306b\u95a2\u308f\u308b\u77e5\u8b58",),
    _SECTION_REFLECT: ("REFLECT_SIMPLE / REFLECT_COMPLEX / REFLECT_DOUBLE\uff08\u805e\u304d\u8fd4\u3057\uff09",),
    _SECTION_AFFIRM: ("AFFIRM\uff08\u662f\u8a8d\uff09",),
    _SECTION_QUESTION: ("QUESTION\uff08\u8cea\u554f\uff09",),
    _SECTION_SUMMARY: ("SUMMARY\uff08\u8981\u7d04\uff09",),
    _SECTION_INFO: ("ASK_PERMISSION_TO_SHARE_INFO / PROVIDE_INFO\uff08\u60c5\u5831\u63d0\u4f9b\uff09",),
}

_COMMON_SUBSECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
    _COMMON_SUBSECTION_CHANGE_TALK: ("\u30c1\u30a7\u30f3\u30b8\u30c8\u30fc\u30af\u3092\u7406\u89e3\u3059\u308b",),
}

_AGENT_BASE_SECTIONS: Dict[str, Tuple[str, ...]] = {
    # Phase slot filling: phase knowledge only (lighter prompts)
    "phase_slot_filler": (_SECTION_PHASE,),
    "slot_filler": (_SECTION_PHASE,),
    # Ranker: common knowledge only
    "action_ranker": (_SECTION_COMMON,),
    # Change-talk inference: common knowledge only
    "change_talk_inferer": (_SECTION_COMMON,),
    # Affirmation judgment: common knowledge + affirmation
    # (also absorbs future LLM judge agent names)
    "affirmation_decider": (_SECTION_COMMON, _SECTION_AFFIRM),
    "affirmation_classifier": (_SECTION_COMMON, _SECTION_AFFIRM),
    "affirm_mode_decider": (_SECTION_COMMON, _SECTION_AFFIRM),
    # Orchestrator (= response generator): common + phase + affirmation + main action
    "response_generator": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    "orchestrator": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    # The separated response path (Layer 3 / Layer 4) uses the same knowledge scope
    "response_integrator": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    "response_writer": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    # Existing auxiliary agents use common knowledge only
    "risk_detector": (_SECTION_COMMON,),
    "mi_evaluator": (_SECTION_COMMON,),
    # Human counselor action classification uses common + action knowledge
    "human_counselor_action_classifier": (
        _SECTION_COMMON,
        _SECTION_REFLECT,
        _SECTION_AFFIRM,
        _SECTION_QUESTION,
        _SECTION_SUMMARY,
        _SECTION_INFO,
    ),
}

_AGENT_COMMON_SUBSECTIONS: Dict[str, Tuple[str, ...]] = {
    # Change-talk inference only references the change-talk subsection within common knowledge
    "change_talk_inferer": (_COMMON_SUBSECTION_CHANGE_TALK,),
}

_REFLECT_ACTIONS = {"REFLECT", "REFLECT_SIMPLE", "REFLECT_COMPLEX", "REFLECT_DOUBLE"}


def _resolve_knowledge_path(path: str | None = None) -> Path:
    raw = (path or os.getenv("MI_KNOWLEDGE_MD_PATH") or "").strip()
    if not raw:
        return DEFAULT_MI_KNOWLEDGE_PATH
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return resolve_project_path(candidate)


@lru_cache(maxsize=8)
def _load_knowledge_cached(path_str: str, mtime_ns: int, size: int) -> str:
    path = Path(path_str)
    if mtime_ns < 0 or size < 0 or not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("MI knowledge read failed: path=%s error=%s", path, exc)
        return ""

    normalized = text.strip()
    if not normalized:
        return ""
    return normalized


def _knowledge_cache_key(path: Path) -> Tuple[str, int, int]:
    try:
        stat = path.stat()
    except Exception:
        return (str(path), -1, -1)
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _split_top_level_sections(markdown: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Split Markdown into top-level (`##`) sections.
    Returns:
      - prefix: text before the first `##` (such as a title)
      - sections: [(heading, full block text), ...]
    """
    lines = markdown.splitlines()
    prefix_lines: List[str] = []
    sections: List[Tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = [line]
            continue
        if current_heading is None:
            prefix_lines.append(line)
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    prefix = "\n".join(prefix_lines).strip()
    return prefix, sections


def _normalize_action_name(main_action: str | None) -> str:
    if not main_action:
        return ""
    token = str(main_action).strip().upper()
    if "." in token:
        token = token.split(".")[-1]
    return token


def _action_sections(main_action: str | None) -> Tuple[str, ...]:
    action = _normalize_action_name(main_action)
    if action in _REFLECT_ACTIONS:
        return (_SECTION_REFLECT,)
    if action in {"QUESTION", "SCALING_QUESTION", "CLARIFY_PREFERENCE"}:
        return (_SECTION_QUESTION,)
    if action == "SUMMARY":
        return (_SECTION_SUMMARY,)
    if action in {"ASK_PERMISSION_TO_SHARE_INFO", "PROVIDE_INFO"}:
        return (_SECTION_INFO,)
    return tuple()


def _section_names_for_agent(*, agent_name: str, main_action: str | None) -> Tuple[str, ...]:
    agent_key = (agent_name or "").strip().lower()
    base = _AGENT_BASE_SECTIONS.get(agent_key, (_SECTION_COMMON,))
    if agent_key in {"response_generator", "orchestrator", "response_integrator", "response_writer"}:
        merged: List[str] = list(base)
        for sec in _action_sections(main_action):
            if sec not in merged:
                merged.append(sec)
        return tuple(merged)
    return base


def _expand_aliases(names: Tuple[str, ...], *, alias_map: Dict[str, Tuple[str, ...]]) -> set[str]:
    expanded: set[str] = set()
    for name in names:
        if not name:
            continue
        expanded.add(name)
        for alias in alias_map.get(name, tuple()):
            if alias:
                expanded.add(alias)
    return expanded


def _select_knowledge_for_agent(
    *,
    knowledge: str,
    agent_name: str,
    main_action: str | None,
) -> str:
    prefix, sections = _split_top_level_sections(knowledge)
    if not sections:
        return knowledge

    section_names = set(_section_names_for_agent(agent_name=agent_name, main_action=main_action))
    allowed_headings = _expand_aliases(tuple(section_names), alias_map=_SECTION_ALIASES)
    selected_blocks: List[str] = [block for heading, block in sections if heading in allowed_headings]
    if not selected_blocks:
        # If headings change and nothing matches, fall back to the full text for safety.
        return knowledge

    def _filter_common_subsections(block: str, *, allowed_subsections: Tuple[str, ...]) -> str:
        allowed = _expand_aliases(allowed_subsections, alias_map=_COMMON_SUBSECTION_ALIASES)
        if not allowed:
            return block

        lines = block.splitlines()
        if not lines:
            return block

        top_heading = lines[0]
        if not top_heading.startswith("## "):
            return block

        current_heading: str | None = None
        current_lines: List[str] = []
        collected: List[str] = []

        for line in lines[1:]:
            if line.startswith("### "):
                if current_heading is not None and current_heading in allowed:
                    collected.append("\n".join(current_lines).strip())
                current_heading = line[4:].strip()
                current_lines = [line]
            elif current_heading is not None:
                current_lines.append(line)

        if current_heading is not None and current_heading in allowed:
            collected.append("\n".join(current_lines).strip())

        if not collected:
            # If no matching subsection exists, return the original block for safety.
            return block

        return "\n\n".join([top_heading] + collected).strip()

    agent_key = (agent_name or "").strip().lower()
    common_subsections = _AGENT_COMMON_SUBSECTIONS.get(agent_key, tuple())
    if common_subsections:
        common_headings = _expand_aliases((_SECTION_COMMON,), alias_map=_SECTION_ALIASES)
        filtered_blocks: List[str] = []
        for heading, block in sections:
            if heading in common_headings:
                filtered_blocks.append(
                    _filter_common_subsections(
                        block,
                        allowed_subsections=common_subsections,
                    )
                )
            elif heading in allowed_headings:
                filtered_blocks.append(block)
        selected_blocks = filtered_blocks

    parts: List[str] = []
    if prefix:
        parts.append(prefix)
    parts.extend(selected_blocks)
    return "\n\n".join(parts).strip()


def inject_mi_knowledge(
    system_prompt: str,
    *,
    agent_name: str,
    path: str | None = None,
    main_action: str | None = None,
) -> str:
    """
    Append MI knowledge loaded from Markdown to the system prompt.
    - Set `MI_KNOWLEDGE_MD_PATH` to change the source file.
    - The default is `config/mi_knowledge.md`.
    - Depending on `agent_name` / `main_action`, only the relevant sections are passed through.
    - If the file is missing or empty, the original `system_prompt` is returned unchanged.
    """

    resolved = _resolve_knowledge_path(path)
    if not resolved.exists() or not resolved.is_file():
        logger.warning("MI knowledge file not found: %s", resolved)
    knowledge_raw = _load_knowledge_cached(*_knowledge_cache_key(resolved))
    knowledge = _select_knowledge_for_agent(
        knowledge=knowledge_raw,
        agent_name=agent_name,
        main_action=main_action,
    )
    if not knowledge:
        return system_prompt

    return (
        f"{system_prompt}\n\n"
        "[Additional Reference Knowledge (Markdown)]\n"
        f"- Target: {agent_name}\n"
        "- Use the knowledge below as a reference when making decisions. Prioritize the conversation context, "
        "safety, and existing output constraints, and do not adopt contradictory content as-is.\n"
        f"{knowledge}"
    )
