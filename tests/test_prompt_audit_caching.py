"""The prompt-token audit with prompt caching on.

Anthropic's usage.input_tokens counts only the uncached part of the prompt;
cache reads and writes come back in cache_read_input_tokens and
cache_creation_input_tokens. With a cache breakpoint after the system prompt,
a healthy call reports an input_tokens close to the "without instructions"
estimate, so reading input_tokens alone would flag a prompt that arrived.
api_client sums the three before the audit sees them. No model is called;
the SDK is faked.
"""
import sys
import types

import pytest

import api_client
import tracing

WITH, WITHOUT = 4288, 3838            # count_tokens estimates: ~450-token instruction block


def _call(monkeypatch, **usage):
    fake = types.ModuleType("anthropic")
    for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                 "BadRequestError", "RateLimitError", "APIStatusError", "APIConnectionError"):
        setattr(fake, name, type(name, (Exception,), {}))

    class Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text='{"answer": "1935."}')],
                stop_reason="end_turn", stop_details=None, id="msg_test",
                usage=types.SimpleNamespace(output_tokens=40, **usage))

    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(api_client, "_get_anthropic_client",
                        lambda: types.SimpleNamespace(messages=Messages()))
    res = api_client._call_anthropic([{"role": "user", "content": "Q?"}], 100, "test", True, "SYSTEM")
    assert res.get("error_type") is None, res
    return res["usage"]


def _audit(usage):
    return tracing.audit_prompt_tokens(usage["prompt_tokens"], WITH, WITHOUT)["status"]


def test_a_cached_system_prompt_is_not_flagged_as_dropped(monkeypatch):
    # The system prompt was read from the cache: input_tokens alone sits
    # below the midpoint, which used to fail the audit.
    usage = _call(monkeypatch, input_tokens=3830, cache_read_input_tokens=458,
                  cache_creation_input_tokens=0)
    assert tracing.audit_prompt_tokens(3830, WITH, WITHOUT)["status"] == "dropped_suspected"
    assert usage["prompt_tokens"] == 4288
    assert _audit(usage) == "ok"


def test_the_first_call_that_writes_the_cache_is_not_flagged(monkeypatch):
    usage = _call(monkeypatch, input_tokens=3830, cache_read_input_tokens=0,
                  cache_creation_input_tokens=458)
    assert usage["prompt_tokens"] == 4288
    assert _audit(usage) == "ok"


def test_a_dropped_prompt_is_still_caught_with_caching_on(monkeypatch):
    # Nothing cached and the instructions missing: the sum is still short.
    usage = _call(monkeypatch, input_tokens=3830, cache_read_input_tokens=0,
                  cache_creation_input_tokens=0)
    assert _audit(usage) == "dropped_suspected"


@pytest.mark.parametrize("cache_fields", [{}, {"cache_read_input_tokens": None,
                                                "cache_creation_input_tokens": None}])
def test_a_response_without_cache_fields_counts_input_tokens(monkeypatch, cache_fields):
    usage = _call(monkeypatch, input_tokens=4288, **cache_fields)
    assert usage["prompt_tokens"] == 4288 and usage["total_tokens"] == 4328
    assert _audit(usage) == "ok"
