#!/usr/bin/env bash
# deploy.sh -- Deploy the sample-queries-only API to Cloud Run.
#
#   ./deploy.sh            # checks everything, prints the exact commands, runs nothing
#   ./deploy.sh --yes      # same checks, then runs them
#
# Env overrides: PROJECT_ID (default: gcloud's configured project),
#                REGION (default us-central1), SERVICE (default museum-qa-api).
#
# Refuses to run unless: the active gcloud account is gargivipat@gmail.com,
# a project is set, and samples/recorded_outputs.json exists and is labelled
# "recorded model outputs". The service gets exactly one env var,
# LLM_PROVIDER=disabled (--set-env-vars replaces any others), and no secrets.
set -euo pipefail

REQUIRED_ACCOUNT="gargivipat@gmail.com"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-museum-qa-api}"
SAMPLES="samples/recorded_outputs.json"

cd "$(dirname "$0")"

fail() { echo "deploy.sh: REFUSING: $*" >&2; exit 1; }

ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
[ "$ACCOUNT" = "$REQUIRED_ACCOUNT" ] || fail "active gcloud account is '${ACCOUNT:-none}', expected $REQUIRED_ACCOUNT.
  Run: gcloud auth login $REQUIRED_ACCOUNT && gcloud config set account $REQUIRED_ACCOUNT"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT_ID" ] || fail "no project. Run: gcloud config set project YOUR_PROJECT_ID"

[ -f "$SAMPLES" ] || fail "$SAMPLES is missing. Record it first: python record_samples.py --yes"
grep -q '"label": "recorded model outputs"' "$SAMPLES" \
  || fail "$SAMPLES is not labelled \"recorded model outputs\""
if grep -q -E 'sk-ant-[A-Za-z0-9_-]{8,}' "$SAMPLES"; then
  fail "$SAMPLES contains something that looks like an Anthropic key"
fi

# Exactly what will run, in order.
CMDS=(
  "gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $PROJECT_ID"
  "gcloud run deploy $SERVICE --source . --project $PROJECT_ID --region $REGION --allow-unauthenticated --port 8080 --memory 512Mi --cpu 1 --concurrency 20 --max-instances 2 --min-instances 0 --timeout 60 --set-env-vars LLM_PROVIDER=disabled --quiet"
)

echo "Account : $ACCOUNT"
echo "Project : $PROJECT_ID"
echo "Region  : $REGION"
echo "Service : $SERVICE"
echo
echo "These commands create or use billable resources (Cloud Build, Artifact Registry, Cloud Run):"
for c in "${CMDS[@]}"; do echo "  \$ $c"; done
echo

if [ "${1:-}" != "--yes" ]; then
  echo "Nothing was run. Re-run with --yes to execute exactly the commands above."
  exit 0
fi

for c in "${CMDS[@]}"; do
  echo "+ $c"
  eval "$c"
done

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')"
echo
echo "Service URL: $URL"
echo "Smoke test:"
curl -sS -f "$URL/health"; echo
curl -sS -f -o /dev/null -w "GET /samples/q01 -> HTTP %{http_code} in %{time_total}s\n" "$URL/samples/q01"
echo
echo "Next: set API_BASE = '$URL' in docs/index.html, then measure latency:"
echo "  python scripts/measure_latency.py $URL"
