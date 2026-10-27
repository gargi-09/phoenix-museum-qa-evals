"""scripts/record_llm_judge.py with a FAKE model: no key, no network.

Checks the script writes down what the judge returned (PASS, FAIL or a
failed call) without assuming any verdict, and only judges s01-s03.
"""
import importlib.util
import json
import os

import pytest

from conftest import REPO_ROOT

RECORDED = os.path.join(REPO_ROOT, "samples", "recorded_outputs.json")
pytestmark = pytest.mark.skipif(not os.path.exists(RECORDED), reason="needs recorded samples")


def load_script():
    spec = importlib.util.spec_from_file_location(
        "record_llm_judge", os.path.join(REPO_ROOT, "scripts", "record_llm_judge.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_records_each_verdict_as_returned(tmp_path, monkeypatch):
    import api_client
    mod = load_script()
    replies = iter([
        {"content": '{"label": "PASS", "explanation": "Looks right."}', "usage": {"input_tokens": 1}},
        {"content": '{"label": "FAIL", "explanation": "Not cited."}', "usage": {"input_tokens": 2}},
        {"content": None, "usage": None, "error_type": "server_error"},
    ])
    seen = []

    def fake(messages, **kw):
        seen.append(messages[0]["content"])
        return {"error_type": None, **next(replies)}

    monkeypatch.setattr(api_client, "PROVIDER", "anthropic")
    monkeypatch.setattr(api_client, "check_credentials", lambda quiet=False: True)
    monkeypatch.setattr(api_client, "call_haiku", fake)
    monkeypatch.setattr(mod, "OUT_PATH", str(tmp_path / "judge.json"))
    monkeypatch.setattr("sys.argv", ["record_llm_judge.py", "--yes"])
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)

    assert mod.main() == 0
    doc = json.load(open(tmp_path / "judge.json", encoding="utf-8"))
    assert doc["label"] == "recorded LLM-judge outputs"
    assert [r["id"] for r in doc["results"]] == ["s01", "s02", "s03"]
    assert [r["judge"]["label"] for r in doc["results"]] == ["PASS", "FAIL", "INCONCLUSIVE"]
    assert doc["summary"] == {"judged": 3, "judge_passed": ["s01"], "judge_passed_count": 1}
    assert len(seen) == 3
    # The judge sees only the question and answer: no record IDs, no token counts.
    recorded = {e["id"]: e for e in json.load(open(RECORDED, encoding="utf-8"))["scenarios"]}
    for prompt, sid in zip(seen, ("s01", "s02", "s03")):
        assert recorded[sid]["payload"]["question"] in prompt
        assert "prompt_tokens" not in prompt and "record_ids_used" not in prompt
