"""REPLAY adapters: what the public service runs.

Retrieval is replayed, not live: each sample carries the candidates the real
hybrid retriever returned when it was recorded. The answer is the recorded
model output. Each sample has ONE recorded attempt (none were recorded twice),
so in replay a failed verification ends in "abstained: no further recorded
attempt", never in a retry. The retry path is exercised by tests with
scripted answers.
"""
from __future__ import annotations

import json
from typing import Any

from harness.types import AnswerAttempt, NoFurtherAttempt, Retrieval


def load_samples(path: str) -> list[dict[str, Any]]:
    return json.load(open(path, encoding="utf-8"))["scenarios"]


def recorded_attempts(sample: dict[str, Any]) -> list[str]:
    """A sample's recorded answers, in order. Today always 0 or 1."""
    if "attempts" in sample:
        return list(sample["attempts"])
    raw = ((sample.get("payload") or {}).get("llm") or {}).get("raw_content")
    return [raw] if raw is not None else []


class ReplayRetriever:
    async def retrieve(self, sample: dict[str, Any]) -> Retrieval:
        payload = sample.get("payload") or {}
        return Retrieval(candidate_ids=list(payload.get("candidate_ids") or []),
                         abstained=bool((payload.get("result") or {}).get("abstained")),
                         source="replayed",
                         detail="recorded candidates from the real hybrid retriever")


class ReplayAnswerer:
    def has_attempt(self, sample: dict[str, Any], attempt: int) -> bool:
        return attempt <= len(recorded_attempts(sample))

    async def answer(self, sample: dict[str, Any], attempt: int,
                     feedback: list[str]) -> AnswerAttempt:
        attempts = recorded_attempts(sample)
        if attempt > len(attempts):
            raise NoFurtherAttempt(f"{sample.get('id')}: {len(attempts)} recorded attempt(s)")
        return AnswerAttempt(attempts[attempt - 1],
                             system_prompt_delivered=not sample.get("dropped_by_script", False))
