#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const runDirArg = process.argv[2];

if (!runDirArg) {
  console.error("Usage: node scripts/summarize-run.js <run-directory>");
  process.exit(64);
}

const runDir = path.resolve(runDirArg);
const harnessDir = path.resolve(__dirname, "..");

const metadataPath = path.join(runDir, "metadata.json");
const tracePath = path.join(runDir, "trace.jsonl");
const gitStatusPath = path.join(runDir, "git-status.txt");

for (const required of [metadataPath, tracePath]) {
  if (!fs.existsSync(required)) {
    console.error(`Missing required run artifact: ${required}`);
    process.exit(1);
  }
}

const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));

const taskPath = path.join(
  harnessDir,
  "experiments",
  "tasks",
  `${metadata.taskId}.json`
);

if (!fs.existsSync(taskPath)) {
  console.error(`Task definition not found: ${taskPath}`);
  process.exit(1);
}

const task = JSON.parse(fs.readFileSync(taskPath, "utf8"));

const lines = fs
  .readFileSync(tracePath, "utf8")
  .split("\n")
  .filter(Boolean);

const events = lines.map((line, index) => {
  try {
    return JSON.parse(line);
  } catch (error) {
    throw new Error(`Invalid JSON on trace line ${index + 1}`);
  }
});

const eventCounts = {};
const toolUsage = {};
const toolSequence = [];

let turns = 0;
let modelCalls = 0;

let inputTokens = 0;
let outputTokens = 0;
let reasoningTokens = 0;
let cacheReadTokens = 0;
let cacheWriteTokens = 0;
let totalTokens = 0;
let costUSD = 0;

let finalResponse = null;

for (const event of events) {
  eventCounts[event.type] = (eventCounts[event.type] || 0) + 1;

  if (event.type === "turn_start") {
    turns++;
  }

  if (event.type === "tool_execution_start") {
    const name = event.toolName;

    toolUsage[name] = (toolUsage[name] || 0) + 1;

    toolSequence.push({
      tool: name,
      args: event.args || {}
    });
  }

  if (
    event.type === "message_end" &&
    event.message?.role === "assistant"
  ) {
    const usage = event.message.usage;

    if (usage) {
      modelCalls++;

      inputTokens += usage.input || 0;
      outputTokens += usage.output || 0;
      reasoningTokens += usage.reasoning || 0;
      cacheReadTokens += usage.cacheRead || 0;
      cacheWriteTokens += usage.cacheWrite || 0;
      totalTokens += usage.totalTokens || 0;
      costUSD += usage.cost?.total || 0;
    }

    const text = event.message.content
      ?.filter((item) => item.type === "text")
      .map((item) => item.text)
      .join("\n")
      .trim();

    if (text) {
      finalResponse = text;
    }
  }
}

let changedFiles = [];

if (fs.existsSync(gitStatusPath)) {
  changedFiles = fs
    .readFileSync(gitStatusPath, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => ({
      status: line.slice(0, 2),
      path: line.slice(3)
    }));
}

const expectedPreVerificationExitCode =
  task.verification?.preExpectedExitCode ?? 1;

const expectedPostVerificationExitCode =
  task.verification?.postExpectedExitCode ?? 0;

const heldOutConfigured = Boolean(task.heldOutVerification);

const expectedPreHeldOutVerificationExitCode =
  task.heldOutVerification?.preExpectedExitCode ?? 0;

const expectedPostHeldOutVerificationExitCode =
  task.heldOutVerification?.postExpectedExitCode ?? 0;

const startedAt = new Date(metadata.startedAt);
const finishedAt = new Date(metadata.finishedAt);

const durationSeconds =
  Number.isNaN(startedAt.getTime()) || Number.isNaN(finishedAt.getTime())
    ? null
    : (finishedAt - startedAt) / 1000;

function isCompatibleVerificationCommand(actualCommand, expectedCommand) {
  const actual = actualCommand.trim();
  const expected = expectedCommand.trim();

  if (actual === expected) {
    return true;
  }

  // Allow output-only variants that do not change which tests are executed.
  if (expected === "npm test") {
    return (
      actual === "npm test --silent" ||
      actual === "npm --silent test" ||
      /^npm test -- --test-reporter=\S+$/.test(actual)
    );
  }

  return false;
}

let lastExplicitMutationIndex = -1;

for (let i = 0; i < toolSequence.length; i += 1) {
  if (toolSequence[i].tool === "edit" || toolSequence[i].tool === "write") {
    lastExplicitMutationIndex = i;
  }
}

const agentSelfVerified = toolSequence.some(
  (call, index) =>
    index > lastExplicitMutationIndex &&
    call.tool === "bash" &&
    typeof call.args?.command === "string" &&
    isCompatibleVerificationCommand(
      call.args.command,
      task.verification.command
    )
);

const taskValid =
  metadata.preVerificationExitCode === expectedPreVerificationExitCode;

const heldOutValid =
  !heldOutConfigured ||
  metadata.preHeldOutVerificationExitCode ===
    expectedPreHeldOutVerificationExitCode;

const visibleSuccess =
  metadata.postVerificationExitCode === expectedPostVerificationExitCode;

const heldOutSuccess =
  !heldOutConfigured ||
  metadata.postHeldOutVerificationExitCode ===
    expectedPostHeldOutVerificationExitCode;

const success =
  taskValid &&
  heldOutValid &&
  metadata.piExitCode === 0 &&
  visibleSuccess &&
  heldOutSuccess;

const summary = {
  runId: metadata.runId,
  taskId: metadata.taskId,

  harness: {
    commit: metadata.harnessCommit,
    piVersion: metadata.piVersion || null,
    provider: metadata.provider,
    model: metadata.model,
    thinking: metadata.thinking,
    tools: metadata.tools
  },

  result: {
    taskValid,
    heldOutConfigured,
    heldOutValid,
    visibleSuccess,
    heldOutSuccess,
    success,

    preVerificationExitCode: metadata.preVerificationExitCode,
    expectedPreVerificationExitCode,

    preHeldOutVerificationExitCode:
      metadata.preHeldOutVerificationExitCode ?? null,
    expectedPreHeldOutVerificationExitCode:
      heldOutConfigured ? expectedPreHeldOutVerificationExitCode : null,

    piExitCode: metadata.piExitCode,

    postVerificationExitCode: metadata.postVerificationExitCode,
    expectedPostVerificationExitCode,

    postHeldOutVerificationExitCode:
      metadata.postHeldOutVerificationExitCode ?? null,
    expectedPostHeldOutVerificationExitCode:
      heldOutConfigured ? expectedPostHeldOutVerificationExitCode : null
  },

  execution: {
    startedAt: metadata.startedAt,
    finishedAt: metadata.finishedAt,
    durationSeconds,
    turns,
    modelCalls,
    toolCalls: toolSequence.length
  },

  toolUsage,
  toolSequence,

  usage: {
    inputTokens,
    outputTokens,
    reasoningTokens,
    cacheReadTokens,
    cacheWriteTokens,
    totalTokens,
    costUSD
  },

  repository: {
    targetStartCommit: metadata.targetStartCommit,
    filesChanged: changedFiles.length,
    changedFiles
  },

  verification: {
    agentSelfVerified
  },

  finalResponse,

  trace: {
    eventCount: events.length,
    eventTypes: eventCounts
  }
};

const outputPath = path.join(runDir, "summary.json");

fs.writeFileSync(
  outputPath,
  JSON.stringify(summary, null, 2) + "\n"
);

console.log(JSON.stringify(summary, null, 2));
console.error(`\nSummary written to: ${outputPath}`);
