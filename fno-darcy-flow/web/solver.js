// Darcy solver for the browser: -div(a grad u) = f on the unit square, u = 0 on the boundary.
// Same discretisation as src/data/darcy.py solve_darcy(a, f, h=1/n, face="arithmetic"):
// 5-point finite-volume stencil on cell centres, face permeability = mean of the two cells,
// and a boundary face uses the boundary cell's own a. Solved matrix-free with Jacobi-preconditioned CG.

// a: Float32Array/Float64Array of n*n, row-major (index r * n + c). Returns
// { u: Float64Array | null, iters, relres, converged }; u is null unless converged (no partial answers).
export function solveDarcy(a, n, { f = 1, tol = 1e-8, maxIter = 20000 } = {}) {
  const N = n * n, h = 1 / n;
  // Face coefficients: east (c, c+1) and south (r, r+1) between interior neighbours.
  const ae = new Float64Array(N), as = new Float64Array(N), diag = new Float64Array(N);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const i = r * n + c, ai = a[i];
    const e = c < n - 1 ? 0.5 * (ai + a[i + 1]) : ai;
    const w = c > 0 ? 0.5 * (ai + a[i - 1]) : ai;
    const s = r < n - 1 ? 0.5 * (ai + a[i + n]) : ai;
    const nn = r > 0 ? 0.5 * (ai + a[i - n]) : ai;
    if (c < n - 1) ae[i] = e;
    if (r < n - 1) as[i] = s;
    diag[i] = e + w + s + nn;
  }
  const apply = (x, y) => {
    for (let i = 0; i < N; i++) y[i] = diag[i] * x[i];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const i = r * n + c;
      if (c < n - 1) { y[i] -= ae[i] * x[i + 1]; y[i + 1] -= ae[i] * x[i]; }
      if (r < n - 1) { y[i] -= as[i] * x[i + n]; y[i + n] -= as[i] * x[i]; }
    }
  };

  const b = f * h * h, bnorm = b * Math.sqrt(N);
  const u = new Float64Array(N), res = new Float64Array(N).fill(b);
  const z = new Float64Array(N), p = new Float64Array(N), q = new Float64Array(N);
  for (let i = 0; i < N; i++) p[i] = z[i] = res[i] / diag[i];
  let rz = 0;
  for (let i = 0; i < N; i++) rz += res[i] * z[i];
  let relres = 1, iters = 0;
  while (iters < maxIter) {
    apply(p, q);
    let pq = 0;
    for (let i = 0; i < N; i++) pq += p[i] * q[i];
    const alpha = rz / pq;
    let rr = 0;
    for (let i = 0; i < N; i++) { u[i] += alpha * p[i]; res[i] -= alpha * q[i]; rr += res[i] * res[i]; }
    iters++;
    relres = Math.sqrt(rr) / bnorm;
    if (relres < tol) break;
    let rzNew = 0;
    for (let i = 0; i < N; i++) { z[i] = res[i] / diag[i]; rzNew += res[i] * z[i]; }
    const beta = rzNew / rz;
    rz = rzNew;
    for (let i = 0; i < N; i++) p[i] = z[i] + beta * p[i];
  }
  const converged = relres < tol;
  return { u: converged ? u : null, iters, relres, converged };
}
