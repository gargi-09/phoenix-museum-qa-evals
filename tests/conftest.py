import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

RECORDED = os.path.join(REPO_ROOT, "samples", "recorded_outputs.json")
FIXTURE = os.path.join(REPO_ROOT, "docs", "data", "sandbox.json")

# Tests run against the recorded file when it exists, and the scripted
# fixture file otherwise -- same per-sample shape. Set before server.app is
# imported, because importing it builds the module-level app.
os.environ.setdefault("SAMPLES_PATH", RECORDED if os.path.exists(RECORDED) else FIXTURE)


def sample_files():
    return [p for p in (RECORDED, FIXTURE) if os.path.exists(p)]
