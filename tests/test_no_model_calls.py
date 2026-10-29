"""The serving path must never touch a model: no reading of model credentials,
no import of the Anthropic SDK or an HTTP client, no outbound connection.

Runs the server in a FRESH interpreter (nothing pre-imported) with tripwires
installed before server.app is imported, a decoy ANTHROPIC_API_KEY present in
the environment, then exercises every endpoint. Any tripwire fails the test.
"""
import os
import re
import subprocess
import sys
import textwrap

from conftest import REPO_ROOT

GUARD_SCRIPT = textwrap.dedent(r"""
    import builtins, json, os, socket, sys
    sys.path.insert(0, REPO_ROOT); sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

    tripped = []
    FORBIDDEN_ENV = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "API_KEY", "MODEL_ENDPOINT"}
    # With live answers on, the SDK is imported at startup (import only: no
    # client, no key read, no connection); everything else stays forbidden.
    SDK_ALLOWED = os.environ.get("GUARD_ALLOW_SDK_IMPORT") == "1"
    FORBIDDEN_MODULES = (("requests", "dotenv", "urllib3") if SDK_ALLOWED
                         else ("anthropic", "requests", "dotenv", "urllib3"))
    # Local-only recorders: the serving process must never import them.
    LOCAL_ONLY_SCRIPTS = ("record_samples", "record_llm_judge")

    # Decoy: if anything reads it, the tripwire (not the value) is what we see.
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-decoy-must-never-be-read"

    EnvBase = type(os.environ)
    class GuardedEnv(EnvBase):
        def __getitem__(self, key):
            if key in FORBIDDEN_ENV:
                tripped.append(f"env read: {key}")
                raise KeyError(key)
            return EnvBase.__getitem__(self, key)
    os.environ.__class__ = GuardedEnv

    real_import = builtins.__import__
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # level > 0 is a relative import inside a package (fastapi's own
        # `.requests` module), not the third-party library.
        if level == 0 and name.split(".")[0] in FORBIDDEN_MODULES + LOCAL_ONLY_SCRIPTS:
            tripped.append(f"import: {name}")
            raise ImportError(f"blocked by test: {name}")
        return real_import(name, globals, locals, fromlist, level)
    builtins.__import__ = guarded_import

    # Loopback stays allowed: on Windows the asyncio event loop builds its
    # internal wake-up socketpair by connecting to 127.0.0.1. Any other
    # destination, and every DNS lookup, trips.
    LOOPBACK = {"127.0.0.1", "::1", "localhost"}
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def outbound(address):
        host = address[0] if isinstance(address, tuple) and address else address
        return host not in LOOPBACK

    def guarded_connect(self, address):
        if outbound(address):
            tripped.append(f"network connect: {address!r}")
            raise OSError("outbound network blocked by test")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if outbound(address):
            tripped.append(f"network connect_ex: {address!r}")
            raise OSError("outbound network blocked by test")
        return real_connect_ex(self, address)

    def no_dns(*a, **kw):
        tripped.append(f"dns: {a[:1]!r}")
        raise OSError("DNS blocked by test")

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = no_dns

    # High enough that every request below reaches its handler: a 429 would
    # skip the code under test and pass the guard vacuously.
    os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"
    import server.app as appmod                  # module-level app + warm-up run here
    from fastapi.testclient import TestClient    # in-process ASGI: no sockets
    c = TestClient(appmod.app)

    out = {"health": c.get("/health").status_code}
    samples = c.get("/samples").json()["samples"]
    codes = []
    for s in samples:
        codes.append(c.get(f"/samples/{s['id']}").status_code)
        codes.append(c.post("/evaluate", json={"sample_id": s["id"]}).status_code)
        codes.append(c.post("/evaluate", json={"sample_id": s["id"], "system_prompt_delivered": False,
                                               "answer": "Free prose with no citations."}).status_code)
    out["sample_codes"] = sorted(set(codes))
    out["ask"] = c.get("/ask?q=anything").status_code
    out["retrieve"] = [c.post("/retrieve", json={"question": q}).status_code
                       for q in ("What is the capital of Mongolia?", "Coney Island Beach Reginald Marsh")]
    out["gate"] = [c.post("/gate/run", json={"break": b}).status_code
                   for b in ("none", "approve_seeded", "drop_known_good", "hide_dropped_prompt")]
    out["live_status"] = c.get("/live/status").json().get("enabled")

    # Belt and braces: even called directly, the client refuses.
    import api_client
    direct = api_client.call_haiku([{"role": "user", "content": "hi"}])

    print(json.dumps({"tripped": tripped, "samples": len(samples), "out": out,
                      "provider": api_client.PROVIDER, "direct_error": direct["error_type"],
                      "loaded": [m for m in FORBIDDEN_MODULES + LOCAL_ONLY_SCRIPTS
                                 if m in sys.modules]}))
""")


def run_guard(**extra_env):
    env = {k: v for k, v in os.environ.items()
           if k not in ("LLM_PROVIDER", "LIVE_ANSWERS", "COUNTER_BUCKET", "LIVE_ALLOW_MEMORY_COUNTER")}
    env.update(extra_env)
    proc = subprocess.run([sys.executable, "-c", f"REPO_ROOT = {REPO_ROOT!r}\n" + GUARD_SCRIPT],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_serving_path_never_reads_keys_imports_clients_or_opens_sockets():
    r = run_guard()
    assert r["tripped"] == [], r["tripped"]
    assert r["loaded"] == []
    assert r["samples"] > 0 and r["out"]["health"] == 200
    assert r["out"]["sample_codes"] == [200]                  # every call reached its handler
    assert r["out"]["ask"] == 400
    assert r["out"]["retrieve"] == [200, 200]                 # live BM25 ran, under the tripwires
    assert r["out"]["gate"] == [200, 200, 200, 200]           # the live release gate, too
    assert r["out"]["live_status"] is False                   # live answers are off by default


def test_with_live_answers_on_the_other_endpoints_still_never_touch_the_model():
    """Production runs with LIVE_ANSWERS=on and the key mounted. Every endpoint
    except POST /live/answer must still never read the key, import the SDK or
    open a socket. (The live path itself is tested with a fake client in
    tests/test_live_answer.py.)"""
    r = run_guard(LIVE_ANSWERS="on", LIVE_ALLOW_MEMORY_COUNTER="1", GUARD_ALLOW_SDK_IMPORT="1")
    assert r["tripped"] == [], r["tripped"]           # no key read, no socket, no DNS
    assert r["loaded"] == []                          # requests / dotenv / urllib3 still absent
    assert r["out"]["sample_codes"] == [200] and r["out"]["live_status"] is True


def test_model_client_is_disabled_in_the_serving_process():
    r = run_guard()
    assert r["provider"] == "disabled"
    assert r["direct_error"] == "model_calls_disabled"


def test_recording_scripts_refuse_in_serving_mode():
    for script in ("record_samples.py", os.path.join("scripts", "record_llm_judge.py")):
        env = dict(os.environ, LLM_PROVIDER="disabled")
        proc = subprocess.run([sys.executable, script, "--yes"], cwd=REPO_ROOT,
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 2 and "REFUSING" in proc.stdout, script
        env = dict(os.environ, K_SERVICE="museum-qa-api")
        env.pop("LLM_PROVIDER", None)
        proc = subprocess.run([sys.executable, script, "--yes"], cwd=REPO_ROOT,
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 2 and "REFUSING" in proc.stdout, script


def test_image_never_contains_the_recorder_or_secrets():
    dockerfile = open(os.path.join(REPO_ROOT, "Dockerfile"), encoding="utf-8").read()
    assert "record_samples.py" not in dockerfile.replace("the recorders", "")
    assert "record_llm_judge" not in dockerfile and "ANTHROPIC" not in dockerfile
    # The only script in the image is the release gate.
    assert [l for l in dockerfile.splitlines() if "scripts/" in l] == [
        "COPY scripts/release_gate.py ./scripts/release_gate.py"]
    for ignore in (".dockerignore", ".gcloudignore"):
        lines = [l.strip() for l in open(os.path.join(REPO_ROOT, ignore), encoding="utf-8")]
        assert "*" in lines and "scripts/*" in lines, f"{ignore} must start from an allow-list"
        allowed = [l for l in lines if l.startswith("!")]
        assert not any(".env" in l or "record_" in l for l in allowed)
        assert [l for l in allowed if "scripts" in l] == ["!scripts/", "!scripts/release_gate.py"]
    reqs = open(os.path.join(REPO_ROOT, "requirements-server.txt"), encoding="utf-8").read().lower()
    for pkg in ("requests", "python-dotenv", "torch", "sentence-transformers", "arize-phoenix"):
        assert f"\n{pkg}=" not in reqs and not reqs.startswith(f"{pkg}=")
    assert "\nanthropic==" in reqs              # pinned, for POST /live/answer only


def test_deploy_script_pins_safe_cloud_run_settings():
    sh = open(os.path.join(REPO_ROOT, "deploy.sh"), encoding="utf-8").read()
    assert 'REQUIRED_ACCOUNT="gargivipat@gmail.com"' in sh
    assert "LLM_PROVIDER=disabled" in sh and "--update-env-vars" not in sh
    # The key reaches the service only as a Secret Manager reference, never a value.
    assert "--set-secrets ANTHROPIC_API_KEY=$SECRET:latest" in sh
    assert not re.search(r"ANTHROPIC_API_KEY=(?!\$SECRET:latest)", sh)
    assert not re.search(r"sk-ant-[A-Za-z0-9_-]{20,}", sh)   # a scan pattern is fine; a key is not
    assert "LIVE_DAILY_CAP:-50" in sh and "LIVE_PER_IP_PER_HOUR:-5" in sh   # the approved caps
    for flag in ("--max-instances 2", "--timeout 60", "--memory 512Mi", "--concurrency 20"):
        assert flag in sh
    assert '[ "${1:-}" != "--yes" ]' in sh                    # dry run unless --yes
    assert "--service-account $RUNTIME_SA" in sh                # not the Editor default account
    assert "add-iam-policy-binding" not in sh and "--role" not in sh   # the account gets no roles


def test_live_setup_script_grants_only_scoped_access_and_never_sees_the_key():
    sh = open(os.path.join(REPO_ROOT, "setup_live.sh"), encoding="utf-8").read()
    assert 'REQUIRED_ACCOUNT="gargivipat@gmail.com"' in sh and '[ "${1:-}" != "--yes" ]' in sh
    roles = re.findall(r"--role (roles/[\w.]+)", sh)
    assert sorted(roles) == ["roles/secretmanager.secretAccessor", "roles/storage.objectUser"]
    assert "projects add-iam-policy-binding" not in sh        # nothing project-wide
    assert "--public-access-prevention" in sh
    assert not re.search(r"sk-ant-[A-Za-z0-9_-]{20,}", sh)
    assert "versions add" in sh.split("Done. Now add the key YOURSELF")[1]
