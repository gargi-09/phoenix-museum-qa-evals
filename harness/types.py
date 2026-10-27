"""Typed state for the harness loop.

    retrieve -> answer -> verify -> accept | retry (bounded, with backoff) | abstain -> trace

Every run ends in exactly one Outcome. An ABSTAINED run always carries the
AbstainReason that stopped it, so "no answer" is never silent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class Outcome(str, Enum):
    ACCEPTED = "accepted"
    ABSTAINED = "abstained"


class AbstainReason(str, Enum):
    RETRIEVAL_GATE = "retrieval_gate"                    # the abstention gate fired; no answer step
    RETRIEVAL_FAILED = "retrieval_failed"                # retrieve timed out or raised
    NO_FURTHER_RECORDED_ATTEMPT = "no_further_recorded_attempt"  # replay: nothing left to replay
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"    # max_attempts used, none verified


class AttemptStatus(str, Enum):
    VERIFIED_PASS = "verified_pass"
    VERIFIED_FAIL = "verified_fail"
    TIMEOUT = "timeout"        # the answer or verify step exceeded step_timeout_s
    ERROR = "error"            # the answer step raised


class Step(str, Enum):
    RETRIEVE = "retrieve"
    ANSWER = "answer"
    VERIFY = "verify"
    BACKOFF = "backoff"
    DECIDE = "decide"


@dataclass(frozen=True)
class HarnessConfig:
    max_attempts: int = 3          # total answer attempts, first one included
    step_timeout_s: float = 10.0   # per retrieve / answer / verify step
    backoff_base_s: float = 0.25   # sleep before attempt n (n >= 2): base * 2**(n-2)
    backoff_max_s: float = 2.0

    def backoff_for(self, attempt: int) -> float:
        return min(self.backoff_base_s * 2 ** (attempt - 2), self.backoff_max_s)


@dataclass
class Retrieval:
    candidate_ids: list[str]
    abstained: bool
    source: str                    # "replayed" on the public service: recorded candidates
    detail: str = ""


@dataclass
class AnswerAttempt:
    raw_output: str
    system_prompt_delivered: bool = True


@dataclass
class Attempt:
    number: int
    status: AttemptStatus
    raw_output: str | None = None
    verdict: str | None = None                 # deterministic_verdict label
    failed_evaluators: list[str] = field(default_factory=list)
    evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    detail: str = ""


@dataclass
class TraceEvent:
    step: Step
    attempt: int | None
    elapsed_ms: float
    detail: str = ""


@dataclass
class HarnessState:
    sample_id: str
    question: str
    retrieval: Retrieval | None = None
    attempts: list[Attempt] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    outcome: Outcome | None = None
    abstain_reason: AbstainReason | None = None
    accepted_attempt: int | None = None
    total_ms: float = 0.0

    @property
    def retried(self) -> bool:
        return len(self.attempts) > 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["retried"] = self.retried
        return d


class NoFurtherAttempt(Exception):
    """Raised by an Answerer that has no answer for this attempt number."""


class Retriever(Protocol):
    async def retrieve(self, sample: dict[str, Any]) -> Retrieval: ...


class Answerer(Protocol):
    def has_attempt(self, sample: dict[str, Any], attempt: int) -> bool: ...

    async def answer(self, sample: dict[str, Any], attempt: int,
                     feedback: list[str]) -> AnswerAttempt: ...
