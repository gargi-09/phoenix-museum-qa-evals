"""The public web app (web/): pinned dependencies, strict TypeScript with no
`any`, and a deploy that ships the files it reads. No build needed here; CI
builds it in .github/workflows/pages.yml.
"""
import json
import os
import re

from conftest import REPO_ROOT

WEB = os.path.join(REPO_ROOT, "web")


def read(*p):
    return open(os.path.join(REPO_ROOT, *p), encoding="utf-8").read()


def test_dependencies_are_pinned_exactly():
    pkg = json.loads(read("web", "package.json"))
    for name, version in {**pkg["dependencies"], **pkg["devDependencies"]}.items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"{name}@{version} is not an exact version"
    assert os.path.exists(os.path.join(WEB, "package-lock.json"))


def test_typescript_is_strict_and_has_no_any():
    cfg = json.loads(read("web", "tsconfig.json"))["compilerOptions"]
    assert cfg["strict"] and cfg["noImplicitAny"] and cfg["noUncheckedIndexedAccess"]
    offenders = []
    for root, _, files in os.walk(os.path.join(WEB, "src")):
        for f in files:
            if f.endswith((".ts", ".tsx")):
                text = open(os.path.join(root, f), encoding="utf-8").read()
                if re.search(r"(:\s*any\b|\bas any\b|<any>)", text):
                    offenders.append(f)
    assert offenders == []


def test_the_app_reads_only_generated_files_that_exist():
    sync = read("web", "scripts", "sync-data.mjs")
    for src in re.findall(r"\['([^']+)',", sync):
        assert os.path.exists(os.path.join(REPO_ROOT, src)), src
    assert "docs/sandbox.js" in sync                     # the parity-checked engine, not a copy


def test_api_base_is_the_deployed_service():
    assert "https://museum-qa-api-o7gdkkg7ja-uc.a.run.app" in read("web", "src", "lib", "api.ts")


def test_pages_workflow_type_checks_builds_and_keeps_the_classic_page():
    wf = read(".github", "workflows", "pages.yml")
    assert "npm run --prefix web build" in wf             # build = sync + tsc --noEmit + vite build
    assert "classic.html" in wf
    assert "secrets." not in wf
    assert json.loads(read("web", "package.json"))["scripts"]["build"].count("tsc --noEmit") == 1
