#!/usr/bin/env node

const assert = require("node:assert/strict");
const path = require("node:path");

const targetDir = process.argv[2];

if (!targetDir) {
  console.error("Usage: node js-bugfix-005.js <target-directory>");
  process.exit(64);
}

const { isFeatureEnabled } = require(
  path.join(path.resolve(targetDir), "src", "feature-flags.js")
);

const cases = [
  {
    feature: "darkMode",
    userConfig: {},
    organizationConfig: { darkMode: false },
    expected: false
  },
  {
    feature: "auditLog",
    userConfig: { auditLog: false },
    organizationConfig: {},
    expected: false
  },
  {
    feature: "betaDashboard",
    userConfig: {},
    organizationConfig: { betaDashboard: false },
    expected: false
  },
  {
    feature: "betaDashboard",
    userConfig: { betaDashboard: true },
    organizationConfig: { betaDashboard: false },
    expected: true
  },
  {
    feature: "darkMode",
    userConfig: {},
    organizationConfig: {},
    expected: true
  },
  {
    feature: "unknownFeature",
    userConfig: {},
    organizationConfig: {},
    expected: false
  }
];

for (const {
  feature,
  userConfig,
  organizationConfig,
  expected
} of cases) {
  assert.equal(
    isFeatureEnabled(feature, userConfig, organizationConfig),
    expected,
    `incorrect resolution for feature ${feature}`
  );
}

console.log(`Held-out verification passed: ${cases.length}/${cases.length} cases`);
