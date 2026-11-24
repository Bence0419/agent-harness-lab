#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"

if [[ -z "$INSTANCE_ID" ]]; then
  echo "Usage: $0 <instance-id>"
  exit 64
fi

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK_REPO="${SWEBENCH_TASK_REPO:-$HOME/projects/swe-bench/swe-bench-multilingual-tasks}"
EXPECTED_TASK_REPO_COMMIT="6e08cbcd763c869abd7aae4bcb9e1b3991180777"

OPENCODE_BIN="$(readlink -f "$HARNESS_DIR/node_modules/.bin/opencode")"
OPENCODE_POLICY="$HARNESS_DIR/configs/opencode-swebench-policy-v1.json"
SQUID_CONFIG="$HARNESS_DIR/configs/swebench-openai-only-squid-v1.conf"
SQUID_IMAGE="ubuntu/squid@sha256:6a097f68bae708cedbabd6188d68c7e2e7a38cedd05a176e1cc0ba29e3bbe029"

PROVIDER="openai"
MODEL="gpt-5.4-mini"
VARIANT="medium"
AGENT="build"
OPENCODE_VERSION="2.2.0"
REFERENCE_ID="opencode-reference-v1"
NETWORK_POLICY="openai-only-squid-v1"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set"
  exit 1
fi

if [[ ! -d "$TASK_REPO/.git" ]]; then
  echo "Task repository not found: $TASK_REPO"
  exit 1
fi

ACTUAL_TASK_REPO_COMMIT="$(git -C "$TASK_REPO" rev-parse HEAD)"

if [[ "$ACTUAL_TASK_REPO_COMMIT" != "$EXPECTED_TASK_REPO_COMMIT" ]]; then
  echo "Unexpected SWE-bench task repo commit:"
  echo "Expected: $EXPECTED_TASK_REPO_COMMIT"
  echo "Actual:   $ACTUAL_TASK_REPO_COMMIT"
  exit 1
fi

TASK_DIR="$TASK_REPO/tasks/$INSTANCE_ID"

if [[ ! -d "$TASK_DIR" ]]; then
  echo "Task not found: $INSTANCE_ID"
  exit 1
fi

read_yaml_value() {
  local key="$1"

  sed -n "s/^${key}:[[:space:]]*//p" "$TASK_DIR/task.yaml" \
    | head -1 \
    | sed -e "s/^['\"]//" -e "s/['\"]$//"
}

BASE_COMMIT="$(read_yaml_value base_commit)"
IMAGE="$(read_yaml_value image)"

if [[ -z "$BASE_COMMIT" || -z "$IMAGE" ]]; then
  echo "Could not read task metadata"
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Docker image not found locally:"
  echo "$IMAGE"
  exit 1
fi

if ! docker image inspect "$SQUID_IMAGE" >/dev/null 2>&1; then
  echo "Pinned Squid image not found locally:"
  echo "$SQUID_IMAGE"
  exit 1
fi

if [[ ! -x "$OPENCODE_BIN" ]]; then
  echo "OpenCode executable not found: $OPENCODE_BIN"
  exit 1
fi

ACTUAL_OPENCODE_VERSION="$("$OPENCODE_BIN" --version)"

if [[ "$ACTUAL_OPENCODE_VERSION" != "opencode v$OPENCODE_VERSION" ]]; then
  echo "Unexpected OpenCode version:"
  echo "Expected: opencode v$OPENCODE_VERSION"
  echo "Actual:   $ACTUAL_OPENCODE_VERSION"
  exit 1
fi

if [[ ! -f "$OPENCODE_POLICY" ]]; then
  echo "OpenCode benchmark policy not found: $OPENCODE_POLICY"
  exit 1
fi

if [[ ! -f "$SQUID_CONFIG" ]]; then
  echo "Squid benchmark policy not found: $SQUID_CONFIG"
  exit 1
fi

RUN_ID="${INSTANCE_ID}-${REFERENCE_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$HARNESS_DIR/results/raw/swebench/$INSTANCE_ID/$RUN_ID"

mkdir -p "$RUN_DIR"

PROMPT_FILE="$RUN_DIR/prompt.txt"

{
  echo "Fix the issue described below in this repository."
  echo "Run relevant tests to verify your changes."
  echo
  cat "$TASK_DIR/problem_statement.md"
} > "$PROMPT_FILE"

SAFE_RUN_ID="${RUN_ID//[^a-zA-Z0-9_.-]/-}"

AGENT_CONTAINER="oc-${SAFE_RUN_ID}"
PROXY_CONTAINER="proxy-${SAFE_RUN_ID}"
NETWORK_NAME="net-${SAFE_RUN_ID}"

cleanup() {
  docker logs "$PROXY_CONTAINER" \
    >"$RUN_DIR/proxy.log" 2>&1 || true

  docker rm -f \
    "$AGENT_CONTAINER" \
    "$PROXY_CONTAINER" \
    >/dev/null 2>&1 || true

  docker network rm "$NETWORK_NAME" \
    >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "=== SWE-bench OpenCode Run ==="
echo "Run:        $RUN_ID"
echo "Instance:   $INSTANCE_ID"
echo "Base:       $BASE_COMMIT"
echo "Image:      $IMAGE"
echo "OpenCode:   $OPENCODE_VERSION"
echo "Agent:      $AGENT"
echo "Model:      $PROVIDER/$MODEL#$VARIANT"
echo "Network:    $NETWORK_POLICY"
echo

echo "[1/7] Creating isolated benchmark network..."

docker network create \
  --internal \
  "$NETWORK_NAME" >/dev/null

echo "[2/7] Starting OpenAI-only egress proxy..."

docker run -d \
  --name "$PROXY_CONTAINER" \
  --network "$NETWORK_NAME" \
  --network-alias squid \
  -v "$SQUID_CONFIG:/etc/squid/squid.conf:ro" \
  "$SQUID_IMAGE" >/dev/null

docker network connect bridge "$PROXY_CONTAINER"

sleep 2

echo "[3/7] Creating isolated agent container..."

docker create \
  --name "$AGENT_CONTAINER" \
  --network "$NETWORK_NAME" \
  --entrypoint sleep \
  -e OPENAI_API_KEY \
  -e HTTP_PROXY="http://squid:3128" \
  -e HTTPS_PROXY="http://squid:3128" \
  -e http_proxy="http://squid:3128" \
  -e https_proxy="http://squid:3128" \
  -e NO_PROXY="localhost,127.0.0.1" \
  -e no_proxy="localhost,127.0.0.1" \
  -e OPENCODE_CONFIG="/etc/opencode/opencode.json" \
  -v "$OPENCODE_BIN:/usr/local/bin/opencode:ro" \
  -v "$OPENCODE_POLICY:/etc/opencode/opencode.json:ro" \
  "$IMAGE" \
  infinity >/dev/null

docker start "$AGENT_CONTAINER" >/dev/null

echo "[4/7] Preparing clean repository..."

docker cp \
  "$PROMPT_FILE" \
  "$AGENT_CONTAINER:/tmp/problem_statement.txt"

docker exec "$AGENT_CONTAINER" bash -c "
  set -e
  cd /testbed

  git config core.fileMode false

  ACTUAL_HEAD=\"\$(git rev-parse HEAD)\"

  if [[ \"\$ACTUAL_HEAD\" != \"$BASE_COMMIT\" ]]; then
    echo \"Unexpected repository HEAD: \$ACTUAL_HEAD\" >&2
    exit 1
  fi

  git reset --hard \"$BASE_COMMIT\" >/dev/null
  git clean -fd >/dev/null

  if [[ -n \"\$(git status --porcelain)\" ]]; then
    echo 'Repository is not clean before agent execution' >&2
    git status --short >&2
    exit 1
  fi
"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[5/7] Running OpenCode..."

set +e

docker exec "$AGENT_CONTAINER" bash -c '
  cd /testbed

  PROMPT="$(cat /tmp/problem_statement.txt)"

  exec /usr/local/bin/opencode run \
    --standalone \
    --auto \
    --agent build \
    --model "openai/gpt-5.4-mini#medium" \
    --format json \
    "$PROMPT"
' >"$RUN_DIR/trace.jsonl" 2>"$RUN_DIR/opencode.stderr.log"

OPENCODE_EXIT=$?

set -e

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[6/7] Capturing model patch..."

docker exec "$AGENT_CONTAINER" bash -c "
  cd /testbed
  git -c core.fileMode=false status --short
" >"$RUN_DIR/git-status.txt"

# Intent-to-add makes newly created non-ignored files appear in git diff.
docker exec "$AGENT_CONTAINER" bash -c "
  cd /testbed
  git add -N -- . >/dev/null 2>&1 || true
  git -c core.fileMode=false diff --binary \"$BASE_COMMIT\"
" >"$RUN_DIR/model.patch"

echo "[7/7] Writing SWE-bench prediction and metadata..."

node - "$INSTANCE_ID" "$RUN_DIR/model.patch" <<'NODE' \
  >"$RUN_DIR/prediction.jsonl"
const fs = require("fs");

const [instanceId, patchPath] = process.argv.slice(2);
const modelPatch = fs.readFileSync(patchPath, "utf8");

const prediction = {
  instance_id: instanceId,
  model_name_or_path: "opencode-reference-v1-openai-gpt-5.4-mini-medium",
  model_patch: modelPatch
};

process.stdout.write(JSON.stringify(prediction) + "\n");
NODE

HARNESS_COMMIT="$(git -C "$HARNESS_DIR" rev-parse HEAD)"

if [[ -n "$(git -C "$HARNESS_DIR" status --porcelain)" ]]; then
  HARNESS_DIRTY=true
else
  HARNESS_DIRTY=false
fi

cat >"$RUN_DIR/metadata.json" <<META
{
  "runId": "$RUN_ID",
  "instanceId": "$INSTANCE_ID",
  "reference": "$REFERENCE_ID",
  "taskRepoCommit": "$ACTUAL_TASK_REPO_COMMIT",
  "baseCommit": "$BASE_COMMIT",
  "image": "$IMAGE",
  "harnessCommit": "$HARNESS_COMMIT",
  "harnessDirty": $HARNESS_DIRTY,
  "opencodeVersion": "$OPENCODE_VERSION",
  "agent": "$AGENT",
  "provider": "$PROVIDER",
  "model": "$MODEL",
  "variant": "$VARIANT",
  "nativeHarness": true,
  "networkPolicy": "$NETWORK_POLICY",
  "squidImage": "$SQUID_IMAGE",
  "startedAt": "$STARTED_AT",
  "finishedAt": "$FINISHED_AT",
  "opencodeExitCode": $OPENCODE_EXIT
}
META

echo
echo "=== Result ==="
echo "OpenCode exit: $OPENCODE_EXIT"
echo "Artifacts:     $RUN_DIR"
echo

echo "=== Patch stat ==="

if [[ -s "$RUN_DIR/model.patch" ]]; then
  git apply --stat "$RUN_DIR/model.patch" || true
else
  echo "EMPTY PATCH"
fi

if [[ "$OPENCODE_EXIT" -ne 0 ]]; then
  echo
  echo "OpenCode execution failed."
  exit "$OPENCODE_EXIT"
fi
