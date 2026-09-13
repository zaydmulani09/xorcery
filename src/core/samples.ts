/**
 * Sample inputs used to screen candidate programs.
 *
 * The kernel rejects a program at the first sample that disagrees with the
 * spec, so the first samples should be the ones most likely to disagree:
 * "generic" 32-bit values with a mix of set and clear bits. Edge cases
 * (0, 1, -1, INT_MIN, ...) follow, because those are where an almost-right
 * program goes wrong. Candidates that survive all 32 samples are then
 * verified for real (`gpu/verify.ts`), and a counterexample from that stage
 * replaces the last sample, so the screen tightens as the search goes on.
 */
import { NS } from '../gpu/kernel';
import { hash32 } from './u32';

const EDGE = [
  0, 1, 0xffffffff, 0x80000000, 0x7fffffff, 2, 3, 0xfffffffe, 0x80000001, 0x40000000,
  0x0000ffff, 0xffff0000, 0x55555555, 0xaaaaaaaa, 0x00010000, 0x0000000f, 0xf0000000, 255, 256, 7,
];

const GENERIC = [0x9e3779b9, 0x2545f491, 0xdeadbeef, 0x12345678];

export interface SampleSet {
  nInputs: number;
  /** inputs[i][s] = value of input i for sample s */
  inputs: Uint32Array[];
}

/** Deterministic sample set: NS tuples of nInputs values. */
export function makeSamples(nInputs: number, seed = 1): SampleSet {
  const inputs: Uint32Array[] = [];
  for (let i = 0; i < nInputs; i++) inputs.push(new Uint32Array(NS));
  const seen = new Set<string>();
  let s = 0;
  const push = (tuple: number[]): boolean => {
    if (s >= NS) return false;
    const key = tuple.map((v) => v >>> 0).join(',');
    if (seen.has(key)) return false;
    seen.add(key);
    for (let i = 0; i < nInputs; i++) inputs[i][s] = tuple[i] >>> 0;
    s++;
    return true;
  };
  // generic values first: they reject almost every wrong program on sample 0
  for (let g = 0; g < 4; g++) push(Array.from({ length: nInputs }, (_, i) => hash32(GENERIC[g] + i * 0x1234567 + seed)));
  if (nInputs === 1) {
    for (const e of EDGE) push([e]);
  } else if (nInputs === 2) {
    const pairs = [
      [0, 0], [0, 1], [1, 0], [1, 1], [0xffffffff, 0], [0, 0xffffffff], [0xffffffff, 0xffffffff], [0xffffffff, 1], [1, 0xffffffff],
      [0x80000000, 0], [0, 0x80000000], [0x80000000, 0x80000000], [0x80000000, 1], [1, 0x80000000], [0x7fffffff, 1], [1, 0x7fffffff],
      [0x7fffffff, 0x80000000], [0x80000000, 0x7fffffff], [5, 3], [3, 5], [2, 31], [31, 2], [0xfffffffe, 0xffffffff], [255, 256],
    ];
    for (const p of pairs) push(p);
  } else {
    const trip = [
      [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0xffffffff, 0, 1], [0, 0xffffffff, 1], [1, 1, 0xffffffff], [0x80000000, 0x7fffffff, 1],
      [0x7fffffff, 0x80000000, 0xffffffff], [1, 2, 3], [3, 2, 1], [5, 3, 0], [0xffffffff, 0xffffffff, 0xffffffff], [0x80000000, 0x80000000, 0x80000000],
    ];
    for (const t of trip) push(t);
  }
  // fill with hashed values
  for (let i = 0; s < NS; i++) push(Array.from({ length: nInputs }, (_, j) => hash32(0x51ed27 + i * 977 + j * 7919 + seed * 31)));
  return { nInputs, inputs };
}

export function sampleTuple(set: SampleSet, s: number): number[] {
  return set.inputs.map((arr) => arr[s]);
}

/** Targets = spec applied to every sample. */
export function computeTargets(set: SampleSet, fn: (x: number, y: number, z: number) => number): Uint32Array {
  const t = new Uint32Array(NS);
  for (let s = 0; s < NS; s++) {
    const x = set.inputs[0][s], y = set.inputs[1]?.[s] ?? 0, z = set.inputs[2]?.[s] ?? 0;
    t[s] = fn(x, y, z) >>> 0;
  }
  return t;
}
