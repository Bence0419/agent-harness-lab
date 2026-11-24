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

HOST_NODE="${HOST_NODE:-$HOME/.nvm/versions/node/v24.19.0}"
PI_MODULES="$HARNESS_DIR/node_modules"

PROVIDER="bme-lab"
MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
THINKING="provider-default"
TOOLS="read,bash,edit,write"

SQUID_CONFIG="$HARNESS_DIR/configs/swebench-bme-only-squid-v1.conf"
SQUID_IMAGE="ubuntu/squid@sha256:6a097f68bae708cedbabd6188d68c7e2e7a38cedd05a176e1cc0ba29e3bbe029"
PI_BME_CONFIG="$HARNESS_DIR/configs/pi-bme-qwen-v1-models.json"
NETWORK_POLICY="bme-only-squid-v1"

if [[ -z "${BME_LAB_API_KEY:-}" ]]; then
  echo "BME_LAB_API_KEY is not set"
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

if [[ ! -x "$HOST_NODE/bin/node" ]]; then
  echo "Pinned Node runtime not found: $HOST_NODE"
  exit 1
fi

if [[ ! -x "$PI_MODULES/.bin/pi" ]]; then
  echo "Pi executable not found: $PI_MODULES/.bin/pi"
  exit 1
fi

RUN_ID="${INSTANCE_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$HARNESS_DIR/results/raw/swebench/$INSTANCE_ID/$RUN_ID"

mkdir -p "$RUN_DIR"

PROMPT_FILE="$RUN_DIR/prompt.txt"

{
  echo "Fix the issue described below in this repository."
  echo "Run relevant tests to verify your changes."
  echo
  cat "$TASK_DIR/problem_statement.md"
} > "$PROMPT_FILE"

CONTAINER_NAME="pi-${RUN_ID//[^a-zA-Z0-9_.-]/-}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

trap cleanup EXIT

SAFE_RUN_ID="$(echo "$RUN_ID" | tr -c '[:alnum:]_.-' '-')"
NETWORK_NAME="net-${SAFE_RUN_ID}"
PROXY_CONTAINER="proxy-${SAFE_RUN_ID}"

cleanup_bme() {
  docker rm -f "$PROXY_CONTAINER" >/dev/null 2>&1 || true
  docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
  cleanup
}

trap cleanup_bme EXIT

echo "=== SWE-bench Pi Run ==="
echo "Run:        $RUN_ID"
echo "Instance:   $INSTANCE_ID"
echo "Base:       $BASE_COMMIT"
echo "Image:      $IMAGE"
echo "Model:      $PROVIDER/$MODEL"
echo "Thinking:   $THINKING"
echo

echo "[0/5] Creating BME-only benchmark network..."

docker network create --internal "$NETWORK_NAME" >/dev/null

docker run -d   --name "$PROXY_CONTAINER"   --network "$NETWORK_NAME"   --network-alias squid   -v "$SQUID_CONFIG:/etc/squid/squid.conf:ro"   "$SQUID_IMAGE" >/dev/null

docker network connect bridge "$PROXY_CONTAINER"

echo "[1/5] Creating isolated agent container..."

docker create \
  --name "$CONTAINER_NAME" \
  --entrypoint sleep \
  --network "$NETWORK_NAME" \
  -e BME_LAB_API_KEY \
  -e HTTP_PROXY="http://squid:3128" \
  -e HTTPS_PROXY="http://squid:3128" \
  -e http_proxy="http://squid:3128" \
  -e https_proxy="http://squid:3128" \
  -e NO_PROXY="localhost,127.0.0.1" \
  -e no_proxy="localhost,127.0.0.1" \
  -e PATH="/opt/node24/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -v "$HOST_NODE:/opt/node24:ro" \
  -v "$PI_MODULES:/opt/pi-node_modules:ro" \
  -v "$PI_BME_CONFIG:/tmp/pi-models.json:ro" \
  "$IMAGE" \
  infinity >/dev/null

docker start "$CONTAINER_NAME" >/dev/null

echo "[2/5] Preparing clean repository..."

docker cp \
  "$PROMPT_FILE" \
  "$CONTAINER_NAME:/tmp/problem_statement.txt"

docker exec "$CONTAINER_NAME" bash -c "
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

echo "[3/5] Running Pi..."

set +e

AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS:-3600}"

timeout --signal=TERM --kill-after=30s "$AGENT_TIMEOUT_SECONDS" \
docker exec "$CONTAINER_NAME" bash -c '
  cd /testbed

  PROMPT="$(cat /tmp/problem_statement.txt)"

  mkdir -p /root/.pi/agent
  cp /tmp/pi-models.json /root/.pi/agent/models.json

  exec /opt/pi-node_modules/.bin/pi \
    --provider bme-lab \
    --model 'Qwen/Qwen3.6-35B-A3B-FP8' \
    --print \
    --mode json \
    --no-session \
    --no-extensions \
    --no-skills \
    --no-prompt-templates \
    --no-context-files \
    --no-approve \
    --tools read,bash,edit,write \
    "$PROMPT"
' >"$RUN_DIR/trace.jsonl" 2>"$RUN_DIR/pi.stderr.log"

PI_EXIT=$?

set -e

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[4/5] Capturing model patch..."

docker exec "$CONTAINER_NAME" bash -c "
  cd /testbed
  git -c core.fileMode=false status --short
" >"$RUN_DIR/git-status.txt"

# Intent-to-add makes newly created non-ignored files appear in git diff.
docker exec "$CONTAINER_NAME" bash -c "
  cd /testbed
  git add -N -- . >/dev/null 2>&1 || true
  git -c core.fileMode=false diff --binary \"$BASE_COMMIT\"
" >"$RUN_DIR/model.patch"

echo "[5/5] Writing SWE-bench prediction..."

node - "$INSTANCE_ID" "$RUN_DIR/model.patch" <<'NODE' \
  >"$RUN_DIR/prediction.jsonl"
const fs = require("fs");

const [instanceId, patchPath] = process.argv.slice(2);

const modelPatch = fs.readFileSync(patchPath, "utf8");

const prediction = {
  instance_id: instanceId,
  model_name_or_path: "pi-baseline-v1-bme-qwen3.6-35b-a3b-fp8",
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
  "taskRepoCommit": "$ACTUAL_TASK_REPO_COMMIT",
  "baseCommit": "$BASE_COMMIT",
  "image": "$IMAGE",
  "harnessCommit": "$HARNESS_COMMIT",
  "harnessDirty": $HARNESS_DIRTY,
  "piVersion": "1.0.2",
  "provider": "$PROVIDER",
  "model": "$MODEL",
  "thinking": "$THINKING",
  "tools": "$TOOLS",
  "startedAt": "$STARTED_AT",
  "finishedAt": "$FINISHED_AT",
  "piExitCode": $PI_EXIT
}
META

echo
echo "=== Result ==="
echo "Pi exit:    $PI_EXIT"
echo "Artifacts:  $RUN_DIR"
echo

echo "=== Patch stat ==="

if [[ -s "$RUN_DIR/model.patch" ]]; then
  git apply --stat "$RUN_DIR/model.patch" || true
else
  echo "EMPTY PATCH"
fi

if [[ "$PI_EXIT" -ne 0 ]]; then
  echo
  echo "Pi execution failed."
  exit "$PI_EXIT"
fi
