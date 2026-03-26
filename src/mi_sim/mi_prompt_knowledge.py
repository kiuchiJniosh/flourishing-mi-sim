from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from .paths import resolve_config_path, resolve_project_path

logger = logging.getLogger(__name__)

DEFAULT_MI_KNOWLEDGE_PATH = resolve_config_path("mi_knowledge.md")

_SECTION_COMMON = "共通の知識"
_SECTION_PHASE = "各フェーズに関わる知識"
_SECTION_REFLECT = "REFLECT_SIMPLE / REFLECT_COMPLEX / REFLECT_DOUBLE（聞き返し）"
_SECTION_AFFIRM = "AFFIRM（是認）"
_SECTION_QUESTION = "QUESTION（質問）"
_SECTION_SUMMARY = "SUMMARY（要約）"
_SECTION_INFO = "ASK_PERMISSION_TO_SHARE_INFO / PROVIDE_INFO（情報提供）"
_COMMON_SUBSECTION_CHANGE_TALK = "チェンジトークを理解する"

_AGENT_BASE_SECTIONS: Dict[str, Tuple[str, ...]] = {
    # フェーズスロット埋め: フェーズ知識のみ（プロンプト軽量化）
    "phase_slot_filler": (_SECTION_PHASE,),
    "slot_filler": (_SECTION_PHASE,),
    # ランカー: 共通のみ
    "action_ranker": (_SECTION_COMMON,),
    # チェンジトーク推論: 共通のみ
    "change_talk_inferer": (_SECTION_COMMON,),
    # 是認判定: 共通 + 是認（将来のLLM判定エージェント名も吸収）
    "affirmation_decider": (_SECTION_COMMON, _SECTION_AFFIRM),
    "affirmation_classifier": (_SECTION_COMMON, _SECTION_AFFIRM),
    "affirm_mode_decider": (_SECTION_COMMON, _SECTION_AFFIRM),
    # オーケストレータ（= response generator）: 共通 + フェーズ + 是認 + 主動作
    "response_generator": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    "orchestrator": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    # 分離後の応答系（Layer3/Layer4）も同等の知識スコープを使う
    "response_integrator": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    "response_writer": (_SECTION_COMMON, _SECTION_PHASE, _SECTION_AFFIRM),
    # 既存の補助エージェントは共通知識のみ
    "risk_detector": (_SECTION_COMMON,),
    "mi_evaluator": (_SECTION_COMMON,),
    # 人手カウンセラーのアクション分類は、共通 + 行動系知識
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
    # チェンジトーク推論は、共通知識の中でもチェンジトーク理解部分だけを参照する
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
    Markdown をトップレベル（##）単位で分割する。
    返り値:
      - prefix: 最初の ## 以前（タイトルなど）
      - sections: [(見出し名, ブロック全文), ...]
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
    selected_blocks: List[str] = [block for heading, block in sections if heading in section_names]
    if not selected_blocks:
        # 見出し名変更などで一致しない場合は、従来どおり全文を渡して安全側に倒す
        return knowledge

    def _filter_common_subsections(block: str, *, allowed_subsections: Tuple[str, ...]) -> str:
        allowed = {name.strip() for name in allowed_subsections if str(name).strip()}
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
            # 該当小見出しが存在しない場合は元ブロックを返し、安全側に倒す
            return block

        return "\n\n".join([top_heading] + collected).strip()

    agent_key = (agent_name or "").strip().lower()
    common_subsections = _AGENT_COMMON_SUBSECTIONS.get(agent_key, tuple())
    if common_subsections:
        filtered_blocks: List[str] = []
        for heading, block in sections:
            if heading == _SECTION_COMMON:
                filtered_blocks.append(
                    _filter_common_subsections(
                        block,
                        allowed_subsections=common_subsections,
                    )
                )
            elif heading in section_names:
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
    システムプロンプトへ、Markdownから読み込んだMI知識を追記する。
    - `MI_KNOWLEDGE_MD_PATH` を設定すると読み込み先を変更できる。
    - 既定は `config/mi_knowledge.md`。
    - agent_name / main_action に応じて、関連セクションのみを抽出して渡す。
    - ファイルが無い/空なら元の `system_prompt` をそのまま返す。
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
        "【追加参照知識（Markdown）】\n"
        f"- 対象: {agent_name}\n"
        "- 以下の知識を参照して判断してください。会話文脈・安全性・既存の出力制約を優先し、"
        "矛盾する内容はそのまま採用しないでください。\n"
        f"{knowledge}"
    )
