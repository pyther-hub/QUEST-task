// node --test web/   (no npm dependencies)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { solveDarcy } from "./solver.js";

const cases = JSON.parse(readFileSync(new URL("./fixtures/solver_cases.json", import.meta.url)));

const relL2 = (x, ref) => {
  let d = 0, s = 0;
  for (let i = 0; i < ref.length; i++) { d += (x[i] - ref[i]) ** 2; s += ref[i] ** 2; }
  return Math.sqrt(d / s);
};

test("five Python-solved fixtures are present", () => {
  assert.equal(cases.length, 5);
});

cases.forEach((c, k) => {
  test(`matches solve_darcy (case ${k}, ${c.n}x${c.n})`, () => {
    const r = solveDarcy(Float64Array.from(c.a), c.n);
    assert.ok(r.converged, `not converged after ${r.iters} iterations`);
    assert.ok(r.relres < 1e-8);
    const err = relL2(r.u, c.u);
    assert.ok(err < 1e-4, `relL2 ${err}`);
  });
});

test("returns no partial answer when it does not converge", () => {
  const c = cases[cases.length - 1];
  const r = solveDarcy(Float64Array.from(c.a), c.n, { maxIter: 3 });
  assert.equal(r.converged, false);
  assert.equal(r.u, null);
  assert.equal(r.iters, 3);
  assert.ok(r.relres >= 1e-8);
});
