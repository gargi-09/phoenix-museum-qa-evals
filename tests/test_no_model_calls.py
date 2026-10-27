"""The serving path must never touch a model: no reading of model credentials,
no import of the Anthropic SDK or an HTTP client, no outbound connection.

Runs the server in a FRESH interpreter (nothing pre-imported) with tripwires
installed before server.app is imported, a decoy ANTHROPIC_API_KEY present in
the environment, then exercises every endpoint. Any tripwire fails the test.
"""
import os
import subprocess
import sys
import textwrap

from conftest import REPO_ROOT

GUARD_SCRIPT = textwrap.dedent(r"""
    import builtins, json, os, socket, sys
    sys.path.insert(0, REPO_ROOT); sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

    tripped = []
    FORBIDDEN_ENV = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "API_KEY", "MODEL_ENDPOINT"}
    FORBIDDEN_MODULES = ("anthropic", "requests", "dotenv", "urllib3")
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

    import server.app as appmod                  # module-level app + warm-up run here
    from fastapi.testclient import TestClient    # in-process ASGI: no sockets
    c = TestClient(appmod.app)

    out = {"health": c.get("/health").status_code}
    samples = c.get("/samples").json()["samples"]
    for s in samples:
        c.get(f"/samples/{s['id']}")
        c.post("/evaluate", json={"sample_id": s["id"]})
        c.post("/evaluate", json={"sample_id": s["id"], "system_prompt_delivered": False,
                                  "answer": "Free prose with no citations."})
    c.get("/ask?q=anything")

    # Belt and braces: even called directly, the client refuses.
    import api_client
    direct = api_client.call_haiku([{"role": "user", "content": "hi"}])

    print(json.dumps({"tripped": tripped, "samples": len(samples), "out": out,
                      "provider": api_client.PROVIDER, "direct_error": direct["error_type"],
                      "loaded": [m for m in FORBIDDEN_MODULES + LOCAL_ONLY_SCRIPTS
                                 if m in sys.modules]}))
""")


def run_guard():
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_PROVIDER",)}
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
    assert "record_samples" not in dockerfile.replace("record_samples.py, any", "")
    assert "record_llm_judge" not in dockerfile and "scripts" not in dockerfile
    assert "ANTHROPIC" not in dockerfile
    for ignore in (".dockerignore", ".gcloudignore"):
        lines = [l.strip() for l in open(os.path.join(REPO_ROOT, ignore), encoding="utf-8")]
        assert "*" in lines, f"{ignore} must start from an allow-list"
        assert not any(l.startswith("!") and (".env" in l or "record_" in l or "scripts" in l)
                       for l in lines)
    reqs = open(os.path.join(REPO_ROOT, "requirements-server.txt"), encoding="utf-8").read().lower()
    for pkg in ("anthropic", "requests", "python-dotenv", "torch", "sentence-transformers"):
        assert f"\n{pkg}=" not in reqs and not reqs.startswith(f"{pkg}=")


def test_deploy_script_pins_safe_cloud_run_settings():
    sh = open(os.path.join(REPO_ROOT, "deploy.sh"), encoding="utf-8").read()
    assert 'REQUIRED_ACCOUNT="gargivipat@gmail.com"' in sh
    assert "--set-env-vars LLM_PROVIDER=disabled" in sh      # replaces all other env vars
    assert "ANTHROPIC" not in sh.replace("an Anthropic key", "")
    assert "--set-secrets" not in sh and "--update-env-vars" not in sh
    for flag in ("--max-instances 2", "--timeout 60", "--memory 512Mi", "--concurrency 20"):
        assert flag in sh
    assert '[ "${1:-}" != "--yes" ]' in sh                    # dry run unless --yes
    assert "--service-account $RUNTIME_SA" in sh                # not the Editor default account
    assert "add-iam-policy-binding" not in sh and "--role" not in sh   # the account gets no roles
