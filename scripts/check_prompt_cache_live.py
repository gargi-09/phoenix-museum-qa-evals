"""
check_prompt_cache_live.py -- OPT-IN, LOCAL-ONLY, PAID (two short calls, well
under $0.001). Checks that prompt caching really engages for the pipeline's
system prompt, and that the prompt-token audit stays correct when it does.

    python scripts/check_prompt_cache_live.py          # prints the plan, calls nothing
    python scripts/check_prompt_cache_live.py --yes    # makes the 2 real calls

tests/test_prompt_audit_caching.py proves the accounting with a faked SDK, but
it cannot prove caching ever happens: below a model's minimum cacheable prompt
length the breakpoint is ignored with no error, and the cache fields come back
0. This makes two identical calls with a breakpoint after the system prompt.
The first must write the cache and the second must read it, and the audit
must say "ok" on both. (Suggested by a reader of the write-up.)

Needs ANTHROPIC_API_KEY in .env. Not run in CI, not in the Docker image, and
it refuses to run in the serving environment.
"""
import argparse
import os
import sys
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

MAX_TOKENS = 200
QUESTION = ("What year was Coney Island Beach by Reginald Marsh made? "
            "No records are attached to this message.")


def refuse(msg):
    print(f"[cache-check] REFUSING: {msg}")
    sys.exit(2)


def guard():
    if os.getenv("K_SERVICE") or os.getenv("LLM_PROVIDER", "").lower() == "disabled":
        refuse("this is the serving environment. This check is local-only.")
    import api_client
    if api_client.PROVIDER != "anthropic" or not api_client.check_credentials(quiet=True):
        refuse("needs ANTHROPIC_API_KEY in .env (Anthropic provider). Nothing was called.")
    return api_client


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--yes", action="store_true", help="make the 2 paid calls")
    args = ap.parse_args()

    import reasoning
    import tracing

    print("[cache-check] plan: 2 identical calls to the model with a cache breakpoint after "
          "the system prompt, plus 2 free count_tokens calls.")
    if not args.yes:
        print("[cache-check] dry run: nothing was called. Re-run with --yes.")
        return 0

    api_client = guard()
    client = api_client._get_anthropic_client()
    model = api_client.ANTHROPIC_MODEL

    # A per-run nonce at the start of the cached prefix, so call 1 always
    # writes a fresh entry instead of reading one left by an earlier run.
    system = [{"type": "text", "text": f"Cache check run {uuid.uuid4()}."},
              {"type": "text", "text": reasoning.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": QUESTION}]

    with_i = client.messages.count_tokens(model=model, messages=messages, system=system).input_tokens
    without_i = client.messages.count_tokens(model=model, messages=messages).input_tokens
    print(f"[cache-check] {model}: prompt {with_i} tokens with the system prompt, {without_i} "
          f"without (system prompt ~{with_i - without_i})")

    ok = True
    for n in (1, 2):
        resp = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=system, messages=messages)
        usage = api_client.usage_from_response(resp)
        summed = tracing.audit_prompt_tokens(usage["prompt_tokens"], with_i, without_i)["status"]
        alone = tracing.audit_prompt_tokens(resp.usage.input_tokens, with_i, without_i)["status"]
        print(f"[cache-check] call {n}: input_tokens={resp.usage.input_tokens} "
              f"cache_creation={usage['cache_creation_input_tokens']} "
              f"cache_read={usage['cache_read_input_tokens']} -> prompt_tokens={usage['prompt_tokens']}; "
              f"audit {summed} (input_tokens alone would say {alone})")
        expected = "cache_creation_input_tokens" if n == 1 else "cache_read_input_tokens"
        if not usage[expected]:
            ok = False
            print(f"[cache-check] FAIL: call {n} reported no {expected}. The breakpoint was "
                  f"ignored: the prefix is probably below {model}'s minimum cacheable length.")
        if summed != "ok":
            ok = False
            print(f"[cache-check] FAIL: the audit flagged call {n} although the prompt was sent.")

    print("[cache-check] PASS: caching engaged and the audit stayed correct." if ok
          else "[cache-check] FAILED (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
