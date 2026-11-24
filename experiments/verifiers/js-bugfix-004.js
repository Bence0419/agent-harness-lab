#!/usr/bin/env node

const assert = require("node:assert/strict");
const path = require("node:path");

const targetDir = process.argv[2];

if (!targetDir) {
  console.error("Usage: node js-bugfix-004.js <target-directory>");
  process.exit(64);
}

const { calculateCheckoutTotal } = require(
  path.join(path.resolve(targetDir), "src", "checkout.js")
);

const cases = [
  { unitPrice: 50, quantity: 3, coupon: "SAVE10", expected: 135 },
  { unitPrice: 25, quantity: 4, coupon: "SAVE20", expected: 80 },
  { unitPrice: 12.5, quantity: 2, coupon: "SAVE10", expected: 22.5 },
  { unitPrice: 40, quantity: 5, coupon: "SAVE20", expected: 160 },
  { unitPrice: 75, quantity: 2, coupon: "NONE", expected: 150 }
];

for (const { unitPrice, quantity, coupon, expected } of cases) {
  assert.equal(
    calculateCheckoutTotal(unitPrice, quantity, coupon),
    expected,
    `incorrect checkout total for ${coupon}, unitPrice=${unitPrice}, quantity=${quantity}`
  );
}

console.log(`Held-out verification passed: ${cases.length}/${cases.length} cases`);
