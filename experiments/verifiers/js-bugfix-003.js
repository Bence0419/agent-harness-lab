#!/usr/bin/env node

const assert = require("node:assert/strict");
const path = require("node:path");

const targetDir = process.argv[2];

if (!targetDir) {
  console.error("Usage: node js-bugfix-003.js <target-directory>");
  process.exit(64);
}

const { calculateInvoiceTotal } = require(
  path.join(path.resolve(targetDir), "src", "invoice.js")
);

const cases = [
  { subtotal: 250, country: "AT", expected: 300 },
  { subtotal: 50, country: "AT", expected: 60 },
  { subtotal: 200, country: "HU", expected: 254 },
  { subtotal: 200, country: "DE", expected: 238 },
  { subtotal: 250, country: "US", expected: 250 }
];

for (const { subtotal, country, expected } of cases) {
  assert.equal(
    calculateInvoiceTotal(subtotal, country),
    expected,
    `incorrect invoice total for ${country} with subtotal ${subtotal}`
  );
}

console.log(`Held-out verification passed: ${cases.length}/${cases.length} cases`);
