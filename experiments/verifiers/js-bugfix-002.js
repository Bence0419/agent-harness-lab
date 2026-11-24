#!/usr/bin/env node

const assert = require("node:assert/strict");
const path = require("node:path");

const targetDir = process.argv[2];

if (!targetDir) {
  console.error("Usage: node js-bugfix-002.js <target-directory>");
  process.exit(64);
}

const { calculateShipping } = require(
  path.join(path.resolve(targetDir), "src", "shipping.js")
);

const cases = [
  { weight: 0.999, expected: 5 },
  { weight: 1.001, expected: 10 },
  { weight: 4.999, expected: 10 },
  { weight: 5, expected: 10 },
  { weight: 5.001, expected: 20 }
];

for (const { weight, expected } of cases) {
  assert.equal(
    calculateShipping(weight),
    expected,
    `incorrect shipping cost for weight ${weight}`
  );
}

console.log(`Held-out verification passed: ${cases.length}/${cases.length} cases`);
