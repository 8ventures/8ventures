export type Vector = number[];
export type Matrix = number[][];

export function zeros(length: number): Vector {
  return new Array<number>(length).fill(0);
}

export function cloneVector(v: Vector): Vector {
  return v.slice();
}

export function add(a: Vector, b: Vector): Vector {
  return a.map((ai, idx) => ai + b[idx]);
}

export function subtract(a: Vector, b: Vector): Vector {
  return a.map((ai, idx) => ai - b[idx]);
}

export function scale(v: Vector, scalar: number): Vector {
  return v.map((vi) => vi * scalar);
}

export function dot(a: Vector, b: Vector): number {
  return a.reduce((acc, ai, idx) => acc + ai * b[idx], 0);
}

export function norm(a: Vector): number {
  return Math.sqrt(dot(a, a));
}

export function normalize(a: Vector, epsilon = 1e-12): Vector {
  const n = norm(a);
  if (n < epsilon) {
    return zeros(a.length);
  }
  return scale(a, 1 / n);
}

export function quadraticForm(v: Vector, Q: Matrix): number {
  return dot(matVec(Q, v), v);
}

export function matVec(M: Matrix, v: Vector): Vector {
  return M.map((row) => dot(row, v));
}

export function outer(a: Vector, b: Vector): Matrix {
  return a.map((ai) => b.map((bj) => ai * bj));
}

export function addInPlace(target: Vector, increment: Vector): void {
  for (let i = 0; i < target.length; i += 1) {
    target[i] += increment[i];
  }
}

export function subtractInPlace(target: Vector, decrement: Vector): void {
  for (let i = 0; i < target.length; i += 1) {
    target[i] -= decrement[i];
  }
}

export function ensureMetric(metric: Matrix | undefined, dimension: number): Matrix {
  if (metric) {
    return metric;
  }
  const identity: Matrix = [];
  for (let i = 0; i < dimension; i += 1) {
    const row = new Array<number>(dimension).fill(0);
    row[i] = 1;
    identity.push(row);
  }
  return identity;
}

export function isZeroVector(v: Vector, tolerance = 1e-12): boolean {
  return v.every((value) => Math.abs(value) < tolerance);
}

export function projectToSphere(v: Vector, radius: number, epsilon = 1e-12): Vector {
  const n = norm(v);
  if (n < epsilon) {
    return scale(zeros(v.length), radius);
  }
  return scale(v, radius / n);
}
