#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const manifestPath = process.argv[2];

if (!manifestPath) {
  console.error(
    "Usage: node scripts/summarize-evaluation.js <evaluation-manifest.json>"
  );
  process.exit(64);
}

const harnessRoot = path.resolve(__dirname, "..");
const absoluteManifestPath = path.resolve(manifestPath);

const manifest = JSON.parse(
  fs.readFileSync(absoluteManifestPath, "utf8")
);

function mean(values) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  if (values.length === 0) return null;

  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 1) {
    return sorted[middle];
  }

  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function min(values) {
  return values.length ? Math.min(...values) : null;
}

function max(values) {
  return values.length ? Math.max(...values) : null;
}

function stddev(values) {
  if (values.length === 0) return null;

  const avg = mean(values);
  const variance =
    values.reduce((sum, value) => sum + (value - avg) ** 2, 0) /
    values.length;

  return Math.sqrt(variance);
}

function statistics(values) {
  return {
    mean: mean(values),
    median: median(values),
    min: min(values),
    max: max(values),
    stddev: stddev(values)
  };
}

function rate(count, total) {
  return total === 0 ? null : count / total;
}

function round(value, digits = 4) {
  if (value === null || value === undefined) return value;

  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function roundStats(stats) {
  return {
    mean: round(stats.mean),
    median: round(stats.median),
    min: round(stats.min),
    max: round(stats.max),
    stddev: round(stats.stddev)
  };
}

function trajectorySignature(summary) {
  return summary.toolSequence
    .map((call) => call.tool)
    .join(">");
}

function aggregateRuns(runs) {
  const successes = runs.filter((run) => run.result.success).length;

  const heldOutRuns = runs.filter(
    (run) => run.result.heldOutConfigured
  );

  const heldOutSuccesses = heldOutRuns.filter(
    (run) => run.result.heldOutSuccess
  ).length;

  const selfVerified = runs.filter(
    (run) => run.verification.agentSelfVerified
  ).length;

  const durations = runs.map(
    (run) => run.execution.durationSeconds
  );

  const toolCalls = runs.map(
    (run) => run.execution.toolCalls
  );

  const tokens = runs.map(
    (run) => run.usage.totalTokens
  );

  const costs = runs.map(
    (run) => run.usage.costUSD
  );

  const signatures = runs.map(trajectorySignature);
  const signatureCounts = {};

  for (const signature of signatures) {
    signatureCounts[signature] =
      (signatureCounts[signature] ?? 0) + 1;
  }

  const dominantTrajectoryCount =
    Math.max(...Object.values(signatureCounts));

  return {
    runs: runs.length,

    success: {
      count: successes,
      rate: round(rate(successes, runs.length))
    },

    heldOut: {
      configuredRuns: heldOutRuns.length,
      successCount: heldOutSuccesses,
      successRate: round(
        rate(heldOutSuccesses, heldOutRuns.length)
      )
    },

    selfVerification: {
      count: selfVerified,
      rate: round(rate(selfVerified, runs.length))
    },

    durationSeconds: roundStats(statistics(durations)),
    toolCalls: roundStats(statistics(toolCalls)),
    totalTokens: roundStats(statistics(tokens)),

    costUSD: {
      ...roundStats(statistics(costs)),
      total: round(
        costs.reduce((sum, value) => sum + value, 0),
        6
      )
    },

    trajectory: {
      uniqueToolSequences:
        Object.keys(signatureCounts).length,
      dominantToolSequenceCount:
        dominantTrajectoryCount,
      dominantToolSequenceRate:
        round(rate(dominantTrajectoryCount, runs.length)),
      toolSequences: signatureCounts
    }
  };
}

const allRuns = [];
const perTask = {};

for (const taskId of manifest.tasks) {
  const runIds = manifest.runs[taskId];

  if (!Array.isArray(runIds)) {
    throw new Error(`Missing runs for task: ${taskId}`);
  }

  const summaries = [];

  for (const runId of runIds) {
    const runDir = path.join(
      harnessRoot,
      "results",
      "raw",
      taskId,
      runId
    );

    const summaryPath = path.join(
      runDir,
      "summary.json"
    );

    if (!fs.existsSync(summaryPath)) {
      throw new Error(
        `Missing summary: ${summaryPath}`
      );
    }

    const summary = JSON.parse(
      fs.readFileSync(summaryPath, "utf8")
    );

    if (summary.taskId !== taskId) {
      throw new Error(
        `Task mismatch for ${runId}: ` +
        `${summary.taskId} !== ${taskId}`
      );
    }

    summaries.push(summary);
    allRuns.push(summary);
  }

  perTask[taskId] = aggregateRuns(summaries);
}

if (allRuns.length !== manifest.runCount) {
  throw new Error(
    `Manifest says ${manifest.runCount} runs, ` +
    `but ${allRuns.length} were loaded`
  );
}

const report = {
  evaluation: {
    id: manifest.id,
    description: manifest.description,
    agentBaseline: manifest.agentBaseline,
    executionSuite: manifest.executionSuite,
    analysisSuite: manifest.analysisSuite,
    taskCount: manifest.taskCount,
    repetitionsPerTask: manifest.repetitionsPerTask,
    runCount: manifest.runCount
  },

  overall: aggregateRuns(allRuns),

  perTask
};

const outputDir = path.join(
  harnessRoot,
  "results",
  "evaluations",
  manifest.id
);

fs.mkdirSync(outputDir, { recursive: true });

const outputPath = path.join(
  outputDir,
  "summary.json"
);

fs.writeFileSync(
  outputPath,
  JSON.stringify(report, null, 2) + "\n"
);

console.log(JSON.stringify(report, null, 2));
console.error(`\nEvaluation summary written to: ${outputPath}`);
