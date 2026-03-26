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
    A log entry for one utterance in the dialogue.
    - speaker: either "client" or "counselor"
    - text: the actual utterance text
    - meta: metadata such as phase or action, used for counselor turns
      (usually `None` for client turns)
    """
    speaker: Speaker
    text: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ConversationEnvironment:
    """
    An environment that centrally manages the interaction history between
    the counselor agent (`MIRhythmBot`) and the client agent.

    - counselor: `MIRhythmBot` with phase and rhythm control built in
    - client: `ClientAgent` (for simulation; `None` is fine for human-client use)
    - log: keeps all utterances and metadata in chronological order as `ConversationTurn` entries
    - session_meta: stores session-wide metadata such as `session_id`, `mode`, and `model`
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
        Phase policy when `--max-turns` is reached.
        - Stay in CLOSING if already there
        - Move to CLOSING after at least two turns in REVIEW_REFLECTION
        - Otherwise start REVIEW_REFLECTION
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
        """Reset the environment and log, and reinitialize agent state if available."""
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
        """Return simple latency metadata that is easy to serialize to CSV."""
        latency_ms = max(0.0, (end_perf - start_perf) * 1000.0)
        end_unix_ms = int(end_ns // 1_000_000)
        start_unix_ms_direct = int(start_ns // 1_000_000)
        start_unix_ms_from_latency = int(max(0, end_unix_ms - round(latency_ms)))
        if abs(start_unix_ms_direct - start_unix_ms_from_latency) <= 3000:
            start_unix_ms = start_unix_ms_direct
        else:
            # Prefer a timestamp that stays consistent with the duration even if the wall clock jumps.
            start_unix_ms = start_unix_ms_from_latency
        return {
            "latency_ms": round(latency_ms, 3),
            "latency_start_unix_ms": start_unix_ms,
            "latency_end_unix_ms": end_unix_ms,
        }

    # ===== Human-client interaction: one turn =====
    def step_with_human(self, user_text: str) -> str:
        """
        Entry point for human-client conversations.
        - input: the client's utterance
        - output: the counselor's reply
        """
        # Log the client utterance
        self.log.append(ConversationTurn(speaker="client", text=user_text, meta=None))

        # Counselor response generated by MIRhythmBot using MI logic plus the LLM
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
            # Add the counselor utterance and MI decision metadata
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

    # ===== Automated dialogue with an LLM client (simulation) =====
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
        For self-play simulations where the client side is also generated by an LLM.
        - first_client_utterance: the initial client utterance
        - max_turns: target number of turns. In `phase_to_closing`, reaching this
          count triggers the closing sequence
        - progress: print a lightweight progress indicator when `True`
        - max_turns_completion:
          - "hard_stop": stop at `max_turns` (legacy behavior)
          - "phase_to_closing": after `max_turns`, advance REVIEW_REFLECTION -> CLOSING and wait for completion
        - max_total_turns: safety cap for `phase_to_closing` (defaults to `max_turns + 7`)
        """
        if self.client is None:
            raise ValueError("simulate() requires a client agent.")

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
            # 1) Counselor responds
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

            # 2) Client responds
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

            # 2-a) The post-listen state belongs on the preceding counselor row
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

            # 2-b) The post-speak state belongs on the client row
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

    # ===== Log export =====
    def to_json_serializable(self) -> List[Dict[str, Any]]:
        """Convert the log into a JSON-serializable structure."""
        session_meta = dict(self.session_meta)
        return [
            {"speaker": t.speaker, "text": t.text, "meta": t.meta, "session_meta": session_meta}
            for t in self.log
        ]

    def print_log(self) -> None:
        """Print a quick log view in the console."""
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
    A lightweight container for session logs that do not involve an agent,
    such as manually entered transcripts.
    - log: an array of `ConversationTurn` entries, intended to be appended manually
    - session_meta: session-wide metadata
    - to_json_serializable(): returns records suitable for saving (`SupportsSessionLog` compatible)
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


# Backward-compatible alias for the old `ManualConversationEnvironment` name
ManualConversationEnvironment = SessionLogEnvironment


# ===== Demo script =====

def demo_human_like() -> None:
    """
    A simple demo that simulates a human client.
    """
    counselor = MIRhythmBot(llm=DummyLLM())
    env = ConversationEnvironment(counselor=counselor)

    inputs = [
        "I have been staying up late more often lately, and it's becoming a problem.",
        "But work has been busy, so I end up procrastinating.",
        "Yes... honestly, I want to value my own time a bit more.",
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
    A demo that runs both the counselor and client with LLMs (`DummyLLM`).
    """
    counselor = MIRhythmBot(llm=DummyLLM())
    client = SimpleClientLLM(llm=DummyLLM())
    env = ConversationEnvironment(counselor=counselor, client=client)

    env.simulate(
        first_client_utterance="My daily routine has been falling apart lately, and my mood has been low.",
        max_turns=5,
    )

    print("==== SIMULATION LOG ====")
    env.print_log()


if __name__ == "__main__":
    demo_human_like()
    print("\n\n")
    demo_two_agents()
