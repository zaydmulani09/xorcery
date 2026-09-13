/**
 * CPU fallback verification, in a Web Worker, for when the GPU cannot be
 * trusted (its spot check disagreed with the CPU) or is absent.
 *
 * One input: all 2^32 values, exhaustively (tens of seconds in JIT'd JS).
 * Two/three inputs: the special-value grid plus 2^26 hashed tuples.
 */
import { Program, SpaceConfig } from './core/space';
import { emitFunction } from './core/program';
import { CompiledSpec } from './spec/compile';
import { VerifyReport } from './gpu/verify';

export interface CpuVerifyOptions {
  cfg: SpaceConfig;
  program: Program;
  spec: CompiledSpec;
  signal?: AbortSignal;
  onProgress?: (done: bigint, total: bigint, pass: string) => void;
  /** hashed tuples for multi-input specs, as a power of two (default 26) */
  randomLog2?: number;
}

export const WORKER_SOURCE = `
self.onmessage = (e) => {
  const { progSrc, specSrc, nInputs, randomLog2, specials } = e.data;
  const prog = new Function(progSrc + '\\nreturn f;')();
  const spec = new Function(specSrc + '\\nreturn spec;')();
  const hash32 = (x) => { x >>>= 0; x ^= x >>> 16; x = Math.imul(x, 0x7feb352d) >>> 0; x ^= x >>> 15; x = Math.imul(x, 0x846ca68b) >>> 0; x ^= x >>> 16; return x >>> 0; };
  const ce = [];
  let mismatches = 0;
  let checked = 0;
  const report = (done, total, pass) => self.postMessage({ type: 'progress', done: String(done), total: String(total), pass });
  if (nInputs === 1) {
    const total = 4294967296;
    for (let hi = 0; hi < 65536; hi++) {
      const base = hi * 65536;
      for (let lo = 0; lo < 65536; lo++) {
        const x = (base + lo) >>> 0;
        if (prog(x) !== spec(x, 0, 0)) { mismatches++; if (ce.length < 8) ce.push([x]); }
      }
      checked = base + 65536;
      if ((hi & 255) === 255) report(checked, total, 'all 2^32 inputs (CPU)');
      if (mismatches && hi > 16) break;
    }
    self.postMessage({ type: 'done', mismatches, ce, checked: String(checked), passes: [{ name: 'all 2^32 inputs (CPU)', count: String(checked) }] });
    return;
  }
  const n = specials.length;
  const grid = nInputs === 2 ? n * n : n * n * n;
  const rnd = 2 ** randomLog2;
  const total = grid + rnd;
  for (let i = 0; i < grid; i++) {
    const x = specials[i % n], y = specials[Math.floor(i / n) % n], z = nInputs === 3 ? specials[Math.floor(i / (n * n)) % n] : 0;
    const a = nInputs === 2 ? prog(x, y) : prog(x, y, z);
    if (a !== spec(x, y, z)) { mismatches++; if (ce.length < 8) ce.push([x, y, z].slice(0, nInputs)); }
  }
  checked = grid;
  for (let i = 0; i < rnd; i++) {
    const x = hash32(i * 3 + 1), y = hash32(i * 3 + 2), z = nInputs === 3 ? hash32(i * 3 + 3) : 0;
    const a = nInputs === 2 ? prog(x, y) : prog(x, y, z);
    if (a !== spec(x, y, z)) { mismatches++; if (ce.length < 8) ce.push([x, y, z].slice(0, nInputs)); }
    if ((i & 0x3fffff) === 0x3fffff) { checked = grid + i + 1; report(checked, total, 'hashed inputs (CPU)'); if (mismatches) break; }
  }
  checked = grid + rnd;
  self.postMessage({ type: 'done', mismatches, ce, checked: String(checked), passes: [{ name: 'special-value grid (CPU)', count: String(grid) }, { name: '2^' + randomLog2 + ' hashed inputs (CPU)', count: String(rnd) }] });
};
`;

const SPECIALS = [
  0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 127, 128, 255, 256, 1000, 1023, 1024, 4095, 4096, 65535, 65536,
  0x7fffffff, 0x80000000, 0x80000001, 0xffffffff, 0xfffffffe, 0xfffffff0, 0xffffff00, 0xffff0000, 0xff000000, 0xf0000000,
  0x40000000, 0x3fffffff, 0xc0000000, 0x55555555, 0xaaaaaaaa, 0x33333333, 0xcccccccc, 0x0f0f0f0f, 0xf0f0f0f0, 0x00ff00ff,
  0xff00ff00, 0x0000ffff, 0x12345678, 0x9e3779b9, 0xdeadbeef, 0xcafebabe, 0x01010101, 0x80808080, 0x7f7f7f7f, 0x00010000,
];

export function verifyOnCpu(opts: CpuVerifyOptions): Promise<VerifyReport> {
  const { cfg, program, spec } = opts;
  const nInputs = cfg.nInputs;
  const t0 = performance.now();
  return new Promise((resolve, reject) => {
    const blob = new Blob([WORKER_SOURCE], { type: 'text/javascript' });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url);
    const finish = () => { worker.terminate(); URL.revokeObjectURL(url); };
    const onAbort = () => { finish(); resolve({ mode: nInputs === 1 ? 'exhaustive' : 'sampled', nInputs, checked: 0n, passes: [], mismatches: 0, counterexamples: [], elapsedMs: performance.now() - t0, complete: false, backend: 'cpu' }); };
    opts.signal?.addEventListener('abort', onAbort, { once: true });
    worker.onerror = (e) => { finish(); reject(new Error('CPU verification worker failed: ' + e.message)); };
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') { opts.onProgress?.(BigInt(m.done), BigInt(m.total), m.pass); return; }
      finish();
      opts.signal?.removeEventListener('abort', onAbort);
      resolve({
        mode: nInputs === 1 ? 'exhaustive' : 'sampled', nInputs, checked: BigInt(m.checked),
        passes: m.passes.map((p: { name: string; count: string }) => ({ name: p.name, count: BigInt(p.count) })),
        mismatches: m.mismatches, counterexamples: m.ce, elapsedMs: performance.now() - t0, complete: true, backend: 'cpu',
      });
    };
    worker.postMessage({
      progSrc: emitFunction(cfg, program, 'js', 'f'),
      specSrc: spec.jsSource,
      nInputs,
      randomLog2: opts.randomLog2 ?? 26,
      specials: SPECIALS.map((v) => v >>> 0),
    });
  });
}
