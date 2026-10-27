"""Identity regression, in the code: the system prompt is never silently
dropped between reasoning.answer_question and the wire.

The data-side half (a dropped prompt is DETECTED by prompt_integrity) is in
the release gate. This half checks it is SENT: the pipeline passes it, and
every wire format carries it. No model is called; the Anthropic SDK is faked.
"""
import sys
import types

import pytest

pytest.importorskip("sklearn", reason="fixture retriever needs scikit-learn")

import api_client  # noqa: E402
import reasoning  # noqa: E402

INSTRUCTIONS_START = reasoning.SYSTEM_PROMPT[:60]


def test_answer_question_passes_the_instruction_block(monkeypatch):
    from fixture_mode import ScriptedModel, load_fixture_retriever

    model, seen = ScriptedModel(), []
    model.scenario = "clean"

    def capture(messages, **kw):
        seen.append((messages, kw))
        return model(messages, **kw)

    monkeypatch.setattr(reasoning, "call_haiku", capture)
    reasoning.answer_question("What year was Coney Island Beach by Reginald Marsh made?",
                              load_fixture_retriever())
    assert len(seen) == 1                                     # one model call per answer
    messages, kw = seen[0]
    assert kw.get("system") and INSTRUCTIONS_START in kw["system"]
    assert all(m["role"] != "system" for m in messages)


@pytest.mark.parametrize("fmt", ["anthropic", "openai", "prepend"])
def test_every_wire_format_carries_the_system_prompt(fmt):
    system = reasoning.SYSTEM_PROMPT
    payload = api_client.build_payload([{"role": "user", "content": "Q?"}], 100,
                                       system=system, message_format=fmt)
    wire = payload.get("system", "") + "".join(
        m["content"] if isinstance(m["content"], str) else str(m["content"])
        for m in payload["messages"])
    assert INSTRUCTIONS_START in wire


def test_a_system_role_in_messages_is_rejected_not_forwarded():
    with pytest.raises(ValueError):
        api_client.build_payload([{"role": "system", "content": "x"},
                                  {"role": "user", "content": "Q?"}], 100)


def test_anthropic_request_sets_the_top_level_system_field(monkeypatch):
    sent = {}

    class Stop(Exception):
        pass

    fake = types.ModuleType("anthropic")
    for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                 "BadRequestError", "RateLimitError", "APIStatusError"):
        setattr(fake, name, type(name, (Exception,), {}))
    fake.APIConnectionError = Stop

    class Messages:
        def create(self, **kwargs):
            sent.update(kwargs)
            raise Stop("captured; no request is made")

    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(api_client, "_get_anthropic_client",
                        lambda: types.SimpleNamespace(messages=Messages()))
    res = api_client._call_anthropic([{"role": "user", "content": "Q?"}], 100, "test", True,
                                     reasoning.SYSTEM_PROMPT)
    assert res["error_type"] == "network_error"               # the fake stopped the call
    assert sent["system"] == reasoning.SYSTEM_PROMPT
