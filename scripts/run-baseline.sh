#!/usr/bin/env bash
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ROOT="${AGENT_HARNESS_TARGET_ROOT:-$HOME/projects/agent-harness-targets}"

TASK_ID="${1:-}"

if [[ -z "$TASK_ID" ]]; then
  echo "Usage: $0 <task-id>" >&2
  exit 64
fi

CONFIG_FILE="$HARNESS_DIR/configs/baseline-v1.json"
TASK_FILE="$HARNESS_DIR/experiments/tasks/$TASK_ID.json"
PI_BIN="$HARNESS_DIR/node_modules/.bin/pi"

read_json() {
  node -e '
    const fs = require("fs");
    const data = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    const value = process.argv[2]
      .split(".")
      .reduce((current, key) => current[key], data);

    process.stdout.write(
      Array.isArray(value) ? value.join(",") : String(value)
    );
  ' "$1" "$2"
}

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config: $CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Missing task: $TASK_FILE" >&2
  exit 1
fi

if [[ ! -x "$PI_BIN" ]]; then
  echo "Pi executable not found: $PI_BIN" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

PROVIDER="$(read_json "$CONFIG_FILE" provider)"
MODEL="$(read_json "$CONFIG_FILE" model)"
THINKING="$(read_json "$CONFIG_FILE" thinking)"
TOOLS="$(read_json "$CONFIG_FILE" tools)"

REPOSITORY="$(read_json "$TASK_FILE" repository)"
START_COMMIT="$(read_json "$TASK_FILE" startCommit)"
PROMPT="$(read_json "$TASK_FILE" prompt)"
VERIFY_COMMAND="$(read_json "$TASK_FILE" verification.command)"
PRE_VERIFY_EXPECTED="$(read_json "$TASK_FILE" verification.preExpectedExitCode)"
POST_VERIFY_EXPECTED="$(read_json "$TASK_FILE" verification.postExpectedExitCode)"
HAS_HELD_OUT="$(node -e '
  const fs = require("fs");
  const task = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  process.stdout.write(task.heldOutVerification ? "true" : "false");
' "$TASK_FILE")"

if [[ "$HAS_HELD_OUT" == "true" ]]; then
  HELD_OUT_SCRIPT="$(read_json "$TASK_FILE" heldOutVerification.script)"
  PRE_HELD_OUT_EXPECTED="$(read_json "$TASK_FILE" heldOutVerification.preExpectedExitCode)"
  POST_HELD_OUT_EXPECTED="$(read_json "$TASK_FILE" heldOutVerification.postExpectedExitCode)"
else
  HELD_OUT_SCRIPT=""
  PRE_HELD_OUT_EXPECTED=0
  POST_HELD_OUT_EXPECTED=0
fi

PI_VERSION="$("$PI_BIN" --version)"

TARGET_DIR="$TARGET_ROOT/$REPOSITORY"

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  echo "Target repository not found: $TARGET_DIR" >&2
  exit 1
fi

if ! git -C "$TARGET_DIR" cat-file -e "$START_COMMIT^{commit}" 2>/dev/null; then
  echo "Start commit does not exist: $START_COMMIT" >&2
  exit 1
fi

RUN_TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_ID="${TASK_ID}-${RUN_TIMESTAMP}"
RUN_DIR="$HARNESS_DIR/results/raw/$TASK_ID/$RUN_ID"

mkdir -p "$RUN_DIR"

HARNESS_COMMIT="$(git -C "$HARNESS_DIR" rev-parse HEAD)"
STARTED_AT="$(date -u --iso-8601=seconds)"

echo "=== Agent Harness Baseline Run ==="
echo "Run:      $RUN_ID"
echo "Task:     $TASK_ID"
echo "Model:    $PROVIDER/$MODEL"
echo "Thinking: $THINKING"
echo "Target:   $TARGET_DIR"
echo

echo "[1/7] Resetting target repository..."
git -C "$TARGET_DIR" reset --hard "$START_COMMIT" >/dev/null
git -C "$TARGET_DIR" clean -fd >/dev/null

echo "[2/7] Running pre-verification..."
if (
  cd "$TARGET_DIR"
  bash -lc "$VERIFY_COMMAND"
) >"$RUN_DIR/pre-verification.log" 2>&1; then
  PRE_VERIFY_EXIT=0
else
  PRE_VERIFY_EXIT=$?
fi

echo "[3/7] Running pre held-out verification..."
if [[ "$HAS_HELD_OUT" == "true" ]]; then
  if (
    cd "$HARNESS_DIR"
    node "$HELD_OUT_SCRIPT" "$TARGET_DIR"
  ) >"$RUN_DIR/pre-held-out-verification.log" 2>&1; then
    PRE_HELD_OUT_EXIT=0
  else
    PRE_HELD_OUT_EXIT=$?
  fi
else
  PRE_HELD_OUT_EXIT=0
  echo "No held-out verifier configured" >"$RUN_DIR/pre-held-out-verification.log"
fi

echo "[4/7] Running Pi..."
if (
  cd "$TARGET_DIR"

  "$PI_BIN" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --thinking "$THINKING" \
    --print \
    --mode json \
    --no-session \
    --no-extensions \
    --no-skills \
    --no-prompt-templates \
    --no-context-files \
    --no-approve \
    --tools "$TOOLS" \
    "$PROMPT"
) >"$RUN_DIR/trace.jsonl" 2>"$RUN_DIR/pi.stderr.log"; then
  PI_EXIT=0
else
  PI_EXIT=$?
fi

echo "[5/7] Running independent verification..."
if (
  cd "$TARGET_DIR"
  bash -lc "$VERIFY_COMMAND"
) >"$RUN_DIR/post-verification.log" 2>&1; then
  POST_VERIFY_EXIT=0
else
  POST_VERIFY_EXIT=$?
fi

echo "[6/7] Running post held-out verification..."
if [[ "$HAS_HELD_OUT" == "true" ]]; then
  if (
    cd "$HARNESS_DIR"
    node "$HELD_OUT_SCRIPT" "$TARGET_DIR"
  ) >"$RUN_DIR/post-held-out-verification.log" 2>&1; then
    POST_HELD_OUT_EXIT=0
  else
    POST_HELD_OUT_EXIT=$?
  fi
else
  POST_HELD_OUT_EXIT=0
  echo "No held-out verifier configured" >"$RUN_DIR/post-held-out-verification.log"
fi

echo "[7/7] Capturing repository changes..."
git -C "$TARGET_DIR" diff --no-ext-diff "$START_COMMIT" >"$RUN_DIR/changes.diff"
git -C "$TARGET_DIR" status --short >"$RUN_DIR/git-status.txt"

FINISHED_AT="$(date -u --iso-8601=seconds)"

cat >"$RUN_DIR/metadata.json" <<META
{
  "runId": "$RUN_ID",
  "taskId": "$TASK_ID",
  "harnessCommit": "$HARNESS_COMMIT",
  "targetStartCommit": "$START_COMMIT",
  "piVersion": "$PI_VERSION",
  "provider": "$PROVIDER",
  "model": "$MODEL",
  "thinking": "$THINKING",
  "tools": "$TOOLS",
  "startedAt": "$STARTED_AT",
  "finishedAt": "$FINISHED_AT",
  "preVerificationExitCode": $PRE_VERIFY_EXIT,
  "expectedPreVerificationExitCode": $PRE_VERIFY_EXPECTED,
  "preHeldOutVerificationExitCode": $PRE_HELD_OUT_EXIT,
  "expectedPreHeldOutVerificationExitCode": $PRE_HELD_OUT_EXPECTED,
  "piExitCode": $PI_EXIT,
  "postVerificationExitCode": $POST_VERIFY_EXIT,
  "expectedPostVerificationExitCode": $POST_VERIFY_EXPECTED,
  "postHeldOutVerificationExitCode": $POST_HELD_OUT_EXIT,
  "expectedPostHeldOutVerificationExitCode": $POST_HELD_OUT_EXPECTED
}
META

echo
echo "=== Result ==="
echo "Pre-verification exit:  $PRE_VERIFY_EXIT"
echo "Pi exit:                $PI_EXIT"
echo "Pre held-out exit:      $PRE_HELD_OUT_EXIT"
echo "Post-verification exit: $POST_VERIFY_EXIT"
echo "Post held-out exit:     $POST_HELD_OUT_EXIT"
echo "Artifacts:              $RUN_DIR"

if [[ "$PRE_VERIFY_EXIT" -eq "$PRE_VERIFY_EXPECTED" && \
      "$PRE_HELD_OUT_EXIT" -eq "$PRE_HELD_OUT_EXPECTED" && \
      "$PI_EXIT" -eq 0 && \
      "$POST_VERIFY_EXIT" -eq "$POST_VERIFY_EXPECTED" && \
      "$POST_HELD_OUT_EXIT" -eq "$POST_HELD_OUT_EXPECTED" ]]; then
  echo "RESULT: SUCCESS"
  exit 0
else
  echo "RESULT: FAILURE"
  exit 1
fi
