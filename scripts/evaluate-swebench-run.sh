#!/usr/bin/env bash
set -euo pipefail

RUN_DIR_ARG="${1:-}"

if [[ -z "$RUN_DIR_ARG" ]]; then
  echo "Usage: $0 <run-directory>"
  exit 64
fi

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(realpath "$RUN_DIR_ARG")"

SWEBENCH_ROOT="${SWEBENCH_ROOT:-$HOME/projects/swe-bench}"
TASK_REPO="${SWEBENCH_TASK_REPO:-$SWEBENCH_ROOT/swe-bench-multilingual-tasks}"
SWEBENCH_BIN="${SWEBENCH_BIN:-$SWEBENCH_ROOT/.venv/bin/swebench}"
SWEBENCH_PYTHON="${SWEBENCH_PYTHON:-$SWEBENCH_ROOT/.venv/bin/python}"

PREDICTION="$RUN_DIR/prediction.jsonl"
METADATA="$RUN_DIR/metadata.json"

if [[ ! -f "$PREDICTION" ]]; then
  echo "Prediction not found: $PREDICTION"
  exit 1
fi

if [[ ! -f "$METADATA" ]]; then
  echo "Metadata not found: $METADATA"
  exit 1
fi

if [[ ! -x "$SWEBENCH_BIN" ]]; then
  echo "SWE-bench executable not found: $SWEBENCH_BIN"
  exit 1
fi

INSTANCE_ID="$(node -e '
const fs = require("fs");
const m = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
process.stdout.write(m.instanceId);
' "$METADATA")"

GENERATION_RUN_ID="$(node -e '
const fs = require("fs");
const m = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
process.stdout.write(m.runId);
' "$METADATA")"

MODEL_NAME="$(node -e '
const fs = require("fs");
const line = fs.readFileSync(process.argv[1], "utf8").trim().split(/\r?\n/)[0];
const p = JSON.parse(line);
process.stdout.write(p.model_name_or_path);
' "$PREDICTION")"

PREDICTION_INSTANCE="$(node -e '
const fs = require("fs");
const line = fs.readFileSync(process.argv[1], "utf8").trim().split(/\r?\n/)[0];
const p = JSON.parse(line);
process.stdout.write(p.instance_id);
' "$PREDICTION")"

if [[ "$PREDICTION_INSTANCE" != "$INSTANCE_ID" ]]; then
  echo "Prediction/metadata instance mismatch:"
  echo "metadata:   $INSTANCE_ID"
  echo "prediction: $PREDICTION_INSTANCE"
  exit 1
fi

EVAL_RUN_ID="${EVAL_RUN_ID:-eval-$GENERATION_RUN_ID}"

SWEBENCH_VERSION="$(
  "$SWEBENCH_PYTHON" - <<'PY'
import importlib.metadata
print(importlib.metadata.version("swebench"))
PY
)"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== SWE-bench Evaluation ==="
echo "Generation run: $GENERATION_RUN_ID"
echo "Evaluation run: $EVAL_RUN_ID"
echo "Instance:       $INSTANCE_ID"
echo "Model:          $MODEL_NAME"
echo "SWE-bench:      $SWEBENCH_VERSION"
echo

set +e

(
  cd "$SWEBENCH_ROOT"

  "$SWEBENCH_BIN" eval multilingual \
    -p "$PREDICTION" \
    -i "$INSTANCE_ID" \
    --run-id "$EVAL_RUN_ID" \
    --task-repo "$TASK_REPO" \
    -j 1 \
    -t 1800
) \
  > >(tee "$RUN_DIR/swebench.stdout.log") \
  2> >(tee "$RUN_DIR/swebench.stderr.log" >&2)

EVAL_EXIT=$?

set -e

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

REPORT_SOURCE="$SWEBENCH_ROOT/${MODEL_NAME}.${EVAL_RUN_ID}.json"
REPORT_COPY="$RUN_DIR/swebench-report.json"

if [[ ! -f "$REPORT_SOURCE" ]]; then
  echo
  echo "Expected SWE-bench report not found:"
  echo "$REPORT_SOURCE"
  echo "Evaluator exit code: $EVAL_EXIT"
  exit 1
fi

cp "$REPORT_SOURCE" "$REPORT_COPY"

node - \
  "$REPORT_COPY" \
  "$INSTANCE_ID" \
  "$GENERATION_RUN_ID" \
  "$EVAL_RUN_ID" \
  "$MODEL_NAME" \
  "$SWEBENCH_VERSION" \
  "$EVAL_EXIT" \
  "$STARTED_AT" \
  "$FINISHED_AT" \
  > "$RUN_DIR/evaluation.json" <<'NODE'
const fs = require("fs");

const [
  reportPath,
  instanceId,
  generationRunId,
  evaluationRunId,
  modelName,
  swebenchVersion,
  evaluatorExitCode,
  startedAt,
  finishedAt
] = process.argv.slice(2);

const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));

const has = (key) =>
  Array.isArray(report[key]) && report[key].includes(instanceId);

const out = {
  schemaVersion: 1,
  instanceId,
  generationRunId,
  evaluationRunId,
  modelName,
  swebenchVersion,
  evaluatorExitCode: Number(evaluatorExitCode),
  startedAt,
  finishedAt,

  submitted: has("submitted_ids"),
  completed: has("completed_ids"),
  incomplete: has("incomplete_ids"),
  resolved: has("resolved_ids"),
  unresolved: has("unresolved_ids"),
  infraFailure: has("infra_failure_ids"),
  ambiguousFailure: has("ambiguous_failure_ids"),
  emptyPatch: has("empty_patch_ids"),
  error: has("error_ids"),

  reportSchemaVersion: report.schema_version ?? null
};

process.stdout.write(JSON.stringify(out, null, 2) + "\n");
NODE

echo
echo "=== Normalized result ==="
cat "$RUN_DIR/evaluation.json"

if [[ "$EVAL_EXIT" -ne 0 ]]; then
  echo
  echo "SWE-bench evaluator exited with code $EVAL_EXIT"
  exit "$EVAL_EXIT"
fi
