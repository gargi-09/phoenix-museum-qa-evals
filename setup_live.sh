#!/usr/bin/env bash
# setup_live.sh -- ONE-TIME setup for live answers (POST /live/answer).
#
#   ./setup_live.sh          # prints the exact commands, runs nothing
#   ./setup_live.sh --yes    # runs them
#
# Creates, with least privilege for the service's own account (which has NO
# project roles):
#   - a private Cloud Storage bucket for the durable daily quota counter, and
#     objectUser on THAT bucket only;
#   - an EMPTY Secret Manager secret for the Anthropic key, and
#     secretAccessor on THAT secret only.
# It never sees the key. Add it yourself afterwards (see the last lines).
set -euo pipefail

REQUIRED_ACCOUNT="gargivipat@gmail.com"
REGION="${REGION:-us-central1}"
cd "$(dirname "$0")"
fail() { echo "setup_live.sh: REFUSING: $*" >&2; exit 1; }

ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
[ "$ACCOUNT" = "$REQUIRED_ACCOUNT" ] || fail "active gcloud account is '${ACCOUNT:-none}', expected $REQUIRED_ACCOUNT"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT_ID" ] || fail "no project. Run: gcloud config set project YOUR_PROJECT_ID"

SA="museum-qa-api-runtime@$PROJECT_ID.iam.gserviceaccount.com"
BUCKET="$PROJECT_ID-live-quota"
SECRET="anthropic-api-key"

CMDS=(
  "gcloud services enable secretmanager.googleapis.com storage.googleapis.com --project $PROJECT_ID"
  "gcloud storage buckets create gs://$BUCKET --project $PROJECT_ID --location $REGION --uniform-bucket-level-access --public-access-prevention"
  "gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member serviceAccount:$SA --role roles/storage.objectUser"
  "gcloud secrets create $SECRET --project $PROJECT_ID --replication-policy automatic"
  "gcloud secrets add-iam-policy-binding $SECRET --project $PROJECT_ID --member serviceAccount:$SA --role roles/secretmanager.secretAccessor"
)

echo "Account : $ACCOUNT"
echo "Project : $PROJECT_ID"
echo "Runs as : $SA"
echo "Bucket  : gs://$BUCKET (quota counter, private)"
echo "Secret  : $SECRET (created empty)"
echo
echo "These commands enable APIs, create a bucket and a secret (billable, within"
echo "free-tier sizes) and grant the service account access to those two only:"
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

echo
echo "Done. Now add the key YOURSELF (it is never passed through this script):"
echo "  grep '^ANTHROPIC_API_KEY=' .env | cut -d= -f2- | tr -d '\\r\\n' | \\"
echo "    gcloud secrets versions add $SECRET --project $PROJECT_ID --data-file=-"
echo "Then deploy with: bash deploy.sh --yes"
