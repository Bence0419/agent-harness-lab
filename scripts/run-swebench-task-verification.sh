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

TREATMENT_ID="verification-v1"

PROVIDER="openai"
MODEL="gpt-5.4-mini"
THINKING="medium"
TOOLS="read,bash,edit,write"

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

if [[ ! -x "$HOST_NODE/bin/node" ]]; then
  echo "Pinned Node runtime not found: $HOST_NODE"
  exit 1
fi

if [[ ! -x "$PI_MODULES/.bin/pi" ]]; then
  echo "Pi executable not found: $PI_MODULES/.bin/pi"
  exit 1
fi

RUN_ID="${INSTANCE_ID}-${TREATMENT_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$HARNESS_DIR/results/raw/swebench/$INSTANCE_ID/$RUN_ID"

mkdir -p "$RUN_DIR"

SOLVE_PROMPT_FILE="$RUN_DIR/prompt.txt"
VERIFICATION_PROMPT_FILE="$RUN_DIR/verification-prompt.txt"

# Keep the initial solve prompt identical to baseline-v1.
{
  echo "Fix the issue described below in this repository."
  echo "Run relevant tests to verify your changes."
  echo
  cat "$TASK_DIR/problem_statement.md"
} > "$SOLVE_PROMPT_FILE"

# The second pass sees the attempted fix but no evaluator-only information.
{
  echo "The repository already contains an attempted fix for the issue described below."
  echo
  echo "Act as a verification and repair pass."
  echo "Do not assume the current implementation is correct."
  echo
  echo "Inspect the current git diff and relevant surrounding code."
  echo "Look specifically for:"
  echo "- incomplete fixes,"
  echo "- related state or invariants that may also need updating,"
  echo "- behavioral regressions,"
  echo "- edge cases not covered by the current change."
  echo
  echo "Run relevant tests."
  echo "If you find a problem, repair it and verify the result."
  echo "If the current implementation is already correct, do not make unnecessary changes."
  echo
  echo "Original issue:"
  echo
  cat "$TASK_DIR/problem_statement.md"
} > "$VERIFICATION_PROMPT_FILE"

CONTAINER_NAME="pi-${RUN_ID//[^a-zA-Z0-9_.-]/-}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "=== SWE-bench Pi Verification Run ==="
echo "Run:        $RUN_ID"
echo "Treatment:  $TREATMENT_ID"
echo "Instance:   $INSTANCE_ID"
echo "Base:       $BASE_COMMIT"
echo "Image:      $IMAGE"
echo "Model:      $PROVIDER/$MODEL"
echo "Thinking:   $THINKING"
echo

echo "[1/6] Creating isolated agent container..."

docker create \
  --name "$CONTAINER_NAME" \
  --entrypoint sleep \
  -e OPENAI_API_KEY \
  -e PATH="/opt/node24/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -v "$HOST_NODE:/opt/node24:ro" \
  -v "$PI_MODULES:/opt/pi-node_modules:ro" \
  "$IMAGE" \
  infinity >/dev/null

docker start "$CONTAINER_NAME" >/dev/null

echo "[2/6] Preparing clean repository..."

docker cp \
  "$SOLVE_PROMPT_FILE" \
  "$CONTAINER_NAME:/tmp/problem_statement.txt"

docker cp \
  "$VERIFICATION_PROMPT_FILE" \
  "$CONTAINER_NAME:/tmp/verification_prompt.txt"

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
SOLVE_STARTED_AT="$STARTED_AT"

echo "[3/6] Running baseline solve pass..."

set +e

docker exec "$CONTAINER_NAME" bash -c '
  cd /testbed

  PROMPT="$(cat /tmp/problem_statement.txt)"

  exec /opt/pi-node_modules/.bin/pi \
    --provider openai \
    --model gpt-5.4-mini \
    --thinking medium \
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

SOLVE_EXIT=$?

set -e

SOLVE_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Read-only snapshot of the state before verification.
docker exec "$CONTAINER_NAME" bash -c "
  cd /testbed
  git -c core.fileMode=false status --short
" >"$RUN_DIR/pre-verification-git-status.txt"

docker exec "$CONTAINER_NAME" bash -c "
  cd /testbed
  git -c core.fileMode=false diff --binary \"$BASE_COMMIT\"
" >"$RUN_DIR/pre-verification.patch"

VERIFICATION_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[4/6] Running fresh verification pass..."

set +e

docker exec "$CONTAINER_NAME" bash -c '
  cd /testbed

  PROMPT="$(cat /tmp/verification_prompt.txt)"

  exec /opt/pi-node_modules/.bin/pi \
    --provider openai \
    --model gpt-5.4-mini \
    --thinking medium \
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
' >"$RUN_DIR/verification-trace.jsonl" \
  2>"$RUN_DIR/verification.stderr.log"

VERIFICATION_EXIT=$?

set -e

VERIFICATION_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FINISHED_AT="$VERIFICATION_FINISHED_AT"

echo "[5/6] Capturing final model patch..."

docker exec "$CONTAINER_NAME" bash -c "
  cd /testbed
  git -c core.fileMode=false status --short
" >"$RUN_DIR/git-status.txt"

# Intent-to-add is only performed after both agent passes are complete.
docker exec "$CONTAINER_NAME" bash -c "
  cd /testbed
  git add -N -- . >/dev/null 2>&1 || true
  git -c core.fileMode=false diff --binary \"$BASE_COMMIT\"
" >"$RUN_DIR/model.patch"

echo "[6/6] Writing SWE-bench prediction and metadata..."

node - "$INSTANCE_ID" "$RUN_DIR/model.patch" <<'NODE' \
  >"$RUN_DIR/prediction.jsonl"
const fs = require("fs");

const [instanceId, patchPath] = process.argv.slice(2);

const modelPatch = fs.readFileSync(patchPath, "utf8");

const prediction = {
  instance_id: instanceId,
  model_name_or_path: "pi-verification-v1-openai-gpt-5.4-mini",
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
  "experiment": "$TREATMENT_ID",
  "base": "baseline-v1",
  "verificationPasses": 1,
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
  "solveStartedAt": "$SOLVE_STARTED_AT",
  "solveFinishedAt": "$SOLVE_FINISHED_AT",
  "verificationStartedAt": "$VERIFICATION_STARTED_AT",
  "verificationFinishedAt": "$VERIFICATION_FINISHED_AT",
  "piExitCode": $SOLVE_EXIT,
  "solveExitCode": $SOLVE_EXIT,
  "verificationExitCode": $VERIFICATION_EXIT
}
META

echo
echo "=== Result ==="
echo "Solve exit:        $SOLVE_EXIT"
echo "Verification exit: $VERIFICATION_EXIT"
echo "Artifacts:         $RUN_DIR"
echo

echo "=== Final patch stat ==="

if [[ -s "$RUN_DIR/model.patch" ]]; then
  git apply --stat "$RUN_DIR/model.patch" || true
else
  echo "(empty patch)"
fi

exit "$VERIFICATION_EXIT"
