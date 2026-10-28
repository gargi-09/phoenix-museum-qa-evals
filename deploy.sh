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
# It runs as a dedicated service account with NO project roles (RUNTIME_SA),
# not the default compute account, which has Editor on the project.
#
# Live answers (POST /live/answer) are ON by default (LIVE=on): one real
# claude-haiku-5-5 call per answer, capped at LIVE_DAILY_CAP per UTC day
# across all instances and LIVE_PER_IP_PER_HOUR per visitor. The key is
# mounted from Secret Manager (--set-secrets); it is never in this script,
# the image or an env var literal. One-time setup: bash setup_live.sh.
# LIVE=off deploys with live answers off and mounts no secret.
set -euo pipefail

REQUIRED_ACCOUNT="gargivipat@gmail.com"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-museum-qa-api}"
SAMPLES="samples/recorded_outputs.json"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-museum-qa-api-runtime}"
LIVE="${LIVE:-on}"
LIVE_DAILY_CAP="${LIVE_DAILY_CAP:-50}"
LIVE_PER_IP_PER_HOUR="${LIVE_PER_IP_PER_HOUR:-5}"
SECRET="anthropic-api-key"

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

RUNTIME_SA="$RUNTIME_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Exactly what will run, in order. The service account is created only if it
# does not exist yet, and is granted no roles.
CMDS=(
  "gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $PROJECT_ID"
)
if ! gcloud iam service-accounts describe "$RUNTIME_SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
  CMDS+=("gcloud iam service-accounts create $RUNTIME_SA_NAME --project $PROJECT_ID --display-name 'museum-qa-api runtime (no roles)'")
fi
if [ "$LIVE" = "on" ]; then
  BUCKET="$PROJECT_ID-live-quota"
  gcloud secrets versions describe latest --secret "$SECRET" --project "$PROJECT_ID" >/dev/null 2>&1 \
    || fail "secret $SECRET has no version. Run bash setup_live.sh, add the key, or deploy with LIVE=off"
  ENV_VARS="LLM_PROVIDER=disabled,LIVE_ANSWERS=on,LIVE_DAILY_CAP=$LIVE_DAILY_CAP,LIVE_PER_IP_PER_HOUR=$LIVE_PER_IP_PER_HOUR,COUNTER_BUCKET=$BUCKET"
  LIVE_FLAGS="--set-env-vars $ENV_VARS --set-secrets ANTHROPIC_API_KEY=$SECRET:latest"
else
  LIVE_FLAGS="--set-env-vars LLM_PROVIDER=disabled,LIVE_ANSWERS=off --clear-secrets"
fi
CMDS+=("gcloud run deploy $SERVICE --source . --project $PROJECT_ID --region $REGION --allow-unauthenticated --service-account $RUNTIME_SA --port 8080 --memory 512Mi --cpu 1 --concurrency 20 --max-instances 2 --min-instances 0 --timeout 60 $LIVE_FLAGS --quiet")

echo "Account : $ACCOUNT"
echo "Project : $PROJECT_ID"
echo "Region  : $REGION"
echo "Service : $SERVICE"
echo "Runs as : $RUNTIME_SA (no project roles)"
echo "Live    : $LIVE (cap $LIVE_DAILY_CAP/day, $LIVE_PER_IP_PER_HOUR per visitor per hour)"
echo
echo "These commands create or use billable resources (Cloud Build, Artifact Registry, Cloud Run)"
echo "or change IAM (a service account with no roles):"
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
