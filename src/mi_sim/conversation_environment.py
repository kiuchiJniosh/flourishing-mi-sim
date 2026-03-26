from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .mi_counselor_agent import DummyLLM, LLMClient, MIRhythmBot, Phase
from .perma_client_agent import ClientAgent, SimpleClientLLM


Speaker = Literal["client", "counselor"]


@dataclass
class ConversationTurn:
    """
    対話の1発話分のログ。
    - speaker: "client" または "counselor"
    - text: 実際の発話テキスト
    - meta: カウンセラー側のときだけ、フェーズやアクション等のメタ情報を入れる（クライアント側は通常 None）
    """
    speaker: Speaker
    text: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ConversationEnvironment:
    """
    カウンセラーエージェント（MIRhythmBot）とクライアントエージェントの
    両者のやり取りと履歴を一元管理する「環境」クラス。

    - counselor: MIRhythmBot（フェーズ判定・リズム制御を内蔵）
    - client: ClientAgent（シミュレーション用途。人間クライアントなら None でもOK）
    - log: ConversationTurn のリストとして、時系列の全発話とメタ情報を保持
    - session_meta: セッション共通のメタデータ（session_id / mode / model など）を格納
    """
    counselor: MIRhythmBot
    client: Optional[ClientAgent] = None
    log: List[ConversationTurn] = field(default_factory=list)
    session_meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "session_id" not in self.session_meta:
            self.session_meta["session_id"] = str(uuid.uuid4())

    def _mark_session_end(self, decision_debug: Dict[str, Any]) -> None:
        self.session_meta["session_ended"] = True
        self.session_meta["session_end_reason"] = decision_debug.get("session_end_reason")
        self.session_meta["session_end_condition"] = decision_debug.get("session_end_condition")

    def _apply_max_turn_phase_policy(self, *, turn_no: int, max_turns: int) -> Optional[Dict[str, Any]]:
        """
        `--max-turns` 到達時のフェーズ運用ルール。
        - CLOSING 中ならそのまま維持
        - REVIEW_REFLECTION で 2 ターン以上進んでいれば CLOSING へ
        - それ以外は REVIEW_REFLECTION を開始
        """
        if int(turn_no) < int(max_turns):
            return None

        state = getattr(self.counselor, "state", None)
        if state is None:
            return None

        current_phase = getattr(state, "phase", None)
        phase_turns_raw = getattr(state, "phase_turns", 0)
        try:
            phase_turns = max(0, int(phase_turns_raw))
        except Exception:
            phase_turns = 0

        if current_phase == Phase.CLOSING:
            return {
                "max_turn_phase_override_applied": True,
                "max_turn_phase_override_reason": "already_closing_keep",
                "max_turn_phase_override_phase_before": Phase.CLOSING.value,
                "max_turn_phase_override_phase_after": Phase.CLOSING.value,
                "max_turn_phase_override_phase_turns_before": phase_turns,
            }

        if current_phase == Phase.REVIEW_REFLECTION:
            if phase_turns >= 2:
                setattr(state, "phase", Phase.CLOSING)
                setattr(state, "phase_turns", 0)
                return {
                    "max_turn_phase_override_applied": True,
                    "max_turn_phase_override_reason": "review_completed_go_closing",
                    "max_turn_phase_override_phase_before": Phase.REVIEW_REFLECTION.value,
                    "max_turn_phase_override_phase_after": Phase.CLOSING.value,
                    "max_turn_phase_override_phase_turns_before": phase_turns,
                }
            return {
                "max_turn_phase_override_applied": True,
                "max_turn_phase_override_reason": "review_not_completed_keep",
                "max_turn_phase_override_phase_before": Phase.REVIEW_REFLECTION.value,
                "max_turn_phase_override_phase_after": Phase.REVIEW_REFLECTION.value,
                "max_turn_phase_override_phase_turns_before": phase_turns,
            }

        if isinstance(current_phase, Phase):
            phase_before_text = current_phase.value
        else:
            phase_before_text = str(current_phase or "")

        setattr(state, "phase", Phase.REVIEW_REFLECTION)
        setattr(state, "phase_turns", 0)
        return {
            "max_turn_phase_override_applied": True,
            "max_turn_phase_override_reason": "start_review_reflection",
            "max_turn_phase_override_phase_before": phase_before_text,
            "max_turn_phase_override_phase_after": Phase.REVIEW_REFLECTION.value,
            "max_turn_phase_override_phase_turns_before": phase_turns,
        }

    def reset(self, *, new_session_id: bool = True) -> None:
        """環境とログをリセット。counselor/clientの内部状態も初期化し、必要なら新しいsession_idを振る。"""
        self.log.clear()
        if hasattr(self.counselor, "reset"):
            self.counselor.reset()
        if self.client is not None and hasattr(self.client, "reset"):
            self.client.reset()  # type: ignore[call-arg]
        if new_session_id:
            self.session_meta["session_id"] = str(uuid.uuid4())

    @staticmethod
    def _build_latency_meta(
        *,
        start_perf: float,
        end_perf: float,
        start_ns: int,
        end_ns: int,
    ) -> Dict[str, Any]:
        """入力受領から出力確定までの遅延情報を、CSV化しやすい単純な形で返す。"""
        latency_ms = max(0.0, (end_perf - start_perf) * 1000.0)
        end_unix_ms = int(end_ns // 1_000_000)
        start_unix_ms_direct = int(start_ns // 1_000_000)
        start_unix_ms_from_latency = int(max(0, end_unix_ms - round(latency_ms)))
        if abs(start_unix_ms_direct - start_unix_ms_from_latency) <= 3000:
            start_unix_ms = start_unix_ms_direct
        else:
            # 壁時計の急補正が入った場合でも、duration と整合する時刻を優先する。
            start_unix_ms = start_unix_ms_from_latency
        return {
            "latency_ms": round(latency_ms, 3),
            "latency_start_unix_ms": start_unix_ms,
            "latency_end_unix_ms": end_unix_ms,
        }

    # ===== 人間クライアント用：1ターン分のやり取り =====
    def step_with_human(self, user_text: str) -> str:
        """
        人間クライアントとの対話に使う入口。
        - 入力: クライアント（人間）の発話
        - 出力: カウンセラーの返答
        """
        # クライアント発話をログに追加
        self.log.append(ConversationTurn(speaker="client", text=user_text, meta=None))

        # カウンセラーの応答（MIRhythmBot が MI ロジック＋LLM で生成）
        counselor_start_ns = time.time_ns()
        counselor_start_perf = time.perf_counter()
        reply, decision = self.counselor.step(user_text)
        counselor_end_perf = time.perf_counter()
        counselor_end_ns = time.time_ns()
        counselor_latency = self._build_latency_meta(
            start_perf=counselor_start_perf,
            end_perf=counselor_end_perf,
            start_ns=counselor_start_ns,
            end_ns=counselor_end_ns,
        )
        decision_debug = decision.debug if isinstance(decision.debug, dict) else {}
        session_end_triggered = bool(decision_debug.get("session_end_triggered"))
        skip_counselor_log = session_end_triggered and not str(reply).strip()
        if not skip_counselor_log:
            # カウンセラー発話＋MIの決定ログを追加
            self.log.append(
                ConversationTurn(
                    speaker="counselor",
                    text=reply,
                    meta={
                        "phase": decision.phase.value,
                        "main_action": decision.main_action.value,
                        "add_affirm": decision.add_affirm.value,
                        "assistant_raw_output": decision.debug.get("assistant_raw_output", reply),
                        "assistant_response_text": decision.debug.get("assistant_response_text", reply),
                        **counselor_latency,
                        "debug": decision.debug,
                    },
                )
            )
        if session_end_triggered:
            self._mark_session_end(decision_debug)
        return reply

    # ===== LLMクライアントとの自動対話（シミュレーション） =====
    def simulate(
        self,
        first_client_utterance: str,
        max_turns: int = 10,
        *,
        progress: bool = False,
        max_turns_completion: str = "hard_stop",
        max_total_turns: Optional[int] = None,
    ) -> List[ConversationTurn]:
        """
        クライアント側も LLM で自動生成する「自己対話シミュレーション」用。
        - first_client_utterance: 最初のクライアント発話
        - max_turns: 目安ターン数。`phase_to_closing` 時はこの到達を終結シーケンス開始トリガーとして扱う
        - progress: True にすると進行状況を簡易表示
        - max_turns_completion:
          - "hard_stop": `max_turns` で終了（従来挙動）
          - "phase_to_closing": `max_turns` 到達後に REVIEW_REFLECTION -> CLOSING を進めて終了を待つ
        - max_total_turns: "phase_to_closing" 時の安全上限（未指定なら max_turns + 7）
        """
        if self.client is None:
            raise ValueError("simulate() を使うには client エージェントが必要です。")

        completion_mode = str(max_turns_completion or "hard_stop").strip().lower()
        if completion_mode not in {"hard_stop", "phase_to_closing"}:
            completion_mode = "hard_stop"

        try:
            soft_turn_limit = max(0, int(max_turns))
        except Exception:
            soft_turn_limit = 0

        if completion_mode == "hard_stop":
            hard_turn_limit = soft_turn_limit
        else:
            if max_total_turns is None:
                hard_turn_limit = soft_turn_limit + 7 if soft_turn_limit > 0 else 0
            else:
                try:
                    hard_turn_limit = max(soft_turn_limit, int(max_total_turns))
                except Exception:
                    hard_turn_limit = soft_turn_limit + 7 if soft_turn_limit > 0 else 0

        self.session_meta["max_turns"] = soft_turn_limit
        self.session_meta["max_turns_completion"] = completion_mode
        if completion_mode == "phase_to_closing":
            self.session_meta["max_total_turns"] = hard_turn_limit

        user_text = first_client_utterance
        self.log.append(ConversationTurn(speaker="client", text=user_text, meta=None))

        for turn_no in range(1, hard_turn_limit + 1):
            if progress:
                if completion_mode == "phase_to_closing":
                    print(
                        "[simulate] counselor generating... "
                        f"(turn {turn_no}/{hard_turn_limit}; soft={soft_turn_limit})"
                    )
                else:
                    print(f"[simulate] counselor generating... (turn {turn_no}/{hard_turn_limit})")
            phase_override_debug = self._apply_max_turn_phase_policy(
                turn_no=turn_no,
                max_turns=soft_turn_limit,
            )
            # 1) カウンセラーが応答
            counselor_start_ns = time.time_ns()
            counselor_start_perf = time.perf_counter()
            reply, decision = self.counselor.step(user_text)
            counselor_end_perf = time.perf_counter()
            counselor_end_ns = time.time_ns()
            counselor_latency = self._build_latency_meta(
                start_perf=counselor_start_perf,
                end_perf=counselor_end_perf,
                start_ns=counselor_start_ns,
                end_ns=counselor_end_ns,
            )
            decision_debug = decision.debug if isinstance(decision.debug, dict) else {}
            if phase_override_debug:
                decision_debug.update(phase_override_debug)
            session_end_triggered = bool(decision_debug.get("session_end_triggered"))
            skip_counselor_log = session_end_triggered and not str(reply).strip()
            if not skip_counselor_log:
                self.log.append(
                    ConversationTurn(
                        speaker="counselor",
                        text=reply,
                        meta={
                            "phase": decision.phase.value,
                            "main_action": decision.main_action.value,
                            "add_affirm": decision.add_affirm.value,
                            "assistant_raw_output": decision.debug.get("assistant_raw_output", reply),
                            "assistant_response_text": decision.debug.get("assistant_response_text", reply),
                            **counselor_latency,
                            "debug": decision.debug,
                        },
                    )
                )
            if session_end_triggered:
                self._mark_session_end(decision_debug)
                if progress:
                    reason = str(decision_debug.get("session_end_reason") or "session_end_triggered")
                    print(f"  counselor -> session end ({reason})")
                    print("----")
                break
            if progress:
                print(
                    f"  counselor -> action={decision.main_action.value}, "
                    f"phase={decision.phase.value}"
                )
                if completion_mode == "phase_to_closing":
                    print(
                        "[simulate] client generating... "
                        f"(turn {turn_no}/{hard_turn_limit}; soft={soft_turn_limit})"
                    )
                else:
                    print(f"[simulate] client generating... (turn {turn_no}/{hard_turn_limit})")

            # 2) クライアントが応答
            client_start_ns = time.time_ns()
            client_start_perf = time.perf_counter()
            user_text = self.client.respond(reply, self.log)
            client_end_perf = time.perf_counter()
            client_end_ns = time.time_ns()
            client_latency = self._build_latency_meta(
                start_perf=client_start_perf,
                end_perf=client_end_perf,
                start_ns=client_start_ns,
                end_ns=client_end_ns,
            )

            def _strip_trait_keys(state_obj: Any) -> Optional[Dict[str, Any]]:
                if not isinstance(state_obj, dict):
                    return None
                clean: Dict[str, Any] = {}
                for k, v in state_obj.items():
                    key = str(k)
                    if key.startswith("trait_"):
                        continue
                    clean[key] = v
                return clean or None

            client_debug: Dict[str, Any] = {}
            if hasattr(self.client, "get_last_debug_info"):
                try:
                    dbg = self.client.get_last_debug_info()  # type: ignore[call-arg]
                    if isinstance(dbg, dict):
                        client_debug = dbg
                except Exception:
                    client_debug = {}

            # 2-a) 「聞いた直後」の状態は、直前の counselor 行に載せる
            state_after_listen = _strip_trait_keys(client_debug.get("state_after_listen"))
            if state_after_listen is not None and self.log and self.log[-1].speaker == "counselor":
                counselor_meta = dict(self.log[-1].meta or {})
                counselor_meta["client_internal_state"] = state_after_listen

                reason_after_listen = client_debug.get("internal_state_reason_after_listen") or None
                if reason_after_listen is not None:
                    counselor_meta["client_internal_state_reason"] = reason_after_listen

                meta_after_listen = client_debug.get("meta_after_listen") or None
                if meta_after_listen is not None:
                    counselor_meta["client_meta"] = meta_after_listen
                    parse_status = meta_after_listen.get("parse_status") if isinstance(meta_after_listen, dict) else None
                    if parse_status:
                        counselor_meta["parse_status"] = parse_status

                raw_after_listen = client_debug.get("raw_state_after_listen")
                if raw_after_listen:
                    counselor_meta["client_raw"] = raw_after_listen

                self.log[-1].meta = counselor_meta

            # 2-b) 「話した直後」の状態は、client 行に載せる
            client_meta: Dict[str, Any] = dict(client_latency)
            state_after_speak = _strip_trait_keys(client_debug.get("state_after_speak"))
            if state_after_speak is None and hasattr(self.client, "get_internal_state"):
                try:
                    state_after_speak = _strip_trait_keys(self.client.get_internal_state())  # type: ignore[call-arg]
                except Exception:
                    state_after_speak = None
            if state_after_speak is not None:
                client_meta["client_internal_state"] = state_after_speak

            reason_after_speak = (
                client_debug.get("internal_state_reason_after_speak")
                or client_debug.get("internal_state_reason")
                or None
            )
            if reason_after_speak is not None:
                client_meta["client_internal_state_reason"] = reason_after_speak

            meta_after_speak = client_debug.get("meta_after_speak") or client_debug.get("meta") or None
            if meta_after_speak is not None:
                client_meta["client_meta"] = meta_after_speak
                parse_status = meta_after_speak.get("parse_status") if isinstance(meta_after_speak, dict) else None
                if parse_status:
                    client_meta["parse_status"] = parse_status

            raw_client = (
                client_debug.get("raw_reply")
                or client_debug.get("raw_state_after_speak")
                or client_debug.get("raw_state")
            )
            if raw_client:
                client_meta["client_raw"] = raw_client

            self.log.append(
                ConversationTurn(
                    speaker="client",
                    text=user_text,
                    meta=(client_meta or None),
                )
            )
            if progress:
                print("  client ->", user_text)
                print("----")

        if (
            completion_mode == "phase_to_closing"
            and hard_turn_limit > 0
            and not bool(self.session_meta.get("session_ended"))
        ):
            self.session_meta["session_truncated"] = True
            self.session_meta["session_truncated_reason"] = "max_total_turns_reached_before_session_end"

        return self.log

    # ===== ログのエクスポート =====
    def to_json_serializable(self) -> List[Dict[str, Any]]:
        """ログを JSON シリアライズしやすい形に変換します。"""
        session_meta = dict(self.session_meta)
        return [
            {"speaker": t.speaker, "text": t.text, "meta": t.meta, "session_meta": session_meta}
            for t in self.log
        ]

    def print_log(self) -> None:
        """コンソールでざっとログを確認したいとき用。"""
        for i, turn in enumerate(self.log):
            head = "C" if turn.speaker == "client" else "T"  # C=Client, T=Therapist
            print(f"[{i:02d}] {head}: {turn.text}")
            if turn.meta:
                phase = turn.meta.get("phase")
                action = turn.meta.get("main_action")
                print(f"      meta: phase={phase}, action={action}")


@dataclass
class SessionLogEnvironment:
    """
    人手入力など、エージェントを伴わないセッションログを扱うための軽量コンテナ。
    - log: ConversationTurn の配列（手動で append していく想定）
    - session_meta: セッション共通メタデータ
    - to_json_serializable(): 保存用レコードを返す（SupportsSessionLog 互換）
    """
    log: List[ConversationTurn] = field(default_factory=list)
    session_meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "session_id" not in self.session_meta:
            self.session_meta["session_id"] = str(uuid.uuid4())

    def to_json_serializable(self) -> List[Dict[str, Any]]:
        session_meta = dict(self.session_meta)
        return [
            {"speaker": t.speaker, "text": t.text, "meta": t.meta, "session_meta": session_meta}
            for t in self.log
        ]


# 後方互換のための別名（旧 ManualConversationEnvironment 用）
ManualConversationEnvironment = SessionLogEnvironment


# ===== デモ用スクリプト =====

def demo_human_like() -> None:
    """
    人間クライアントを想定した簡易デモ。
    """
    counselor = MIRhythmBot(llm=DummyLLM())
    env = ConversationEnvironment(counselor=counselor)

    inputs = [
        "最近、夜更かしが増えてしまって困っています。",
        "でも仕事が忙しくて、ついだらだらしてしまいます。",
        "そうですね…本当は、もう少し自分の時間も大事にしたい気持ちがあります。",
    ]

    for u in inputs:
        reply = env.step_with_human(u)
        print("Client   :", u)
        print("Counselor:", reply)
        print("---")

    print("==== LOG ====")
    env.print_log()


def demo_two_agents() -> None:
    """
    カウンセラーとクライアントの両方を LLM で回すシミュレーションのデモ（DummyLLM）。
    """
    counselor = MIRhythmBot(llm=DummyLLM())
    client = SimpleClientLLM(llm=DummyLLM())
    env = ConversationEnvironment(counselor=counselor, client=client)

    env.simulate(
        first_client_utterance="最近、生活リズムが崩れてしまって、気持ちも落ち込んでいます。",
        max_turns=5,
    )

    print("==== SIMULATION LOG ====")
    env.print_log()


if __name__ == "__main__":
    demo_human_like()
    print("\n\n")
    demo_two_agents()
