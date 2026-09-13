/**
 * Verification of a found program against the spec.
 *
 * One input:   every one of the 2^32 inputs is evaluated on the GPU. That is
 *              a proof of equivalence (for the spec as written), not a test.
 * Two inputs:  2^64 pairs cannot be enumerated. The GPU checks all 2^32 pairs
 *              of 16-bit values, the same again with the high halves set, all
 *              pairs from a grid of special values, and 2^30 hashed pairs.
 *              That is labelled "verified on N inputs", never "proved".
 * Three inputs: special-value grid plus 2^30 hashed triples.
 *
 * The program and the spec are both compiled to WGSL and compared inside one
 * kernel, so the check costs one dispatch per chunk and no readback except
 * the mismatch counter. The first few counterexamples are captured so the
 * search can learn from them.
 */
import { Gpu } from './device';
import { Program, SpaceConfig } from '../core/space';
import { emitFunction } from '../core/program';
import { CompiledSpec } from '../spec/compile';
import { hash32 } from '../core/u32';

export type VerifyMode = 'exhaustive' | 'sampled';

export interface VerifyPass {
  name: string;
  /** inputs checked in this pass */
  count: bigint;
}

export interface VerifyReport {
  mode: VerifyMode;
  nInputs: number;
  /** total inputs checked */
  checked: bigint;
  passes: VerifyPass[];
  mismatches: number;
  counterexamples: number[][];
  elapsedMs: number;
  /** false if the run was aborted before completing */
  complete: boolean;
}

export interface VerifyOptions {
  gpu: Gpu;
  cfg: SpaceConfig;
  program: Program;
  spec: CompiledSpec;
  signal?: AbortSignal;
  onProgress?: (done: bigint, total: bigint, pass: string) => void;
  targetMs?: number;
  /** size of the hashed (random) pass for 2/3 inputs, as a power of two (default 30) */
  randomLog2?: number;
}

const PER_THREAD = 64;
const WG = 256;
const PER_WG = PER_THREAD * WG; // 16384 inputs per workgroup

/** Special values used for the multi-input grid passes. */
const SPECIAL = [
  0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 127, 128, 255, 256, 1000, 1023, 1024, 4095, 4096, 65535, 65536,
  0x7fffffff, 0x80000000, 0x80000001, 0xffffffff, 0xfffffffe, 0xfffffff0, 0xffffff00, 0xffff0000, 0xff000000, 0xf0000000,
  0x40000000, 0x3fffffff, 0xc0000000, 0x55555555, 0xaaaaaaaa, 0x33333333, 0xcccccccc, 0x0f0f0f0f, 0xf0f0f0f0, 0x00ff00ff,
  0xff00ff00, 0x0000ffff, 0x12345678, 0x9e3779b9, 0xdeadbeef, 0xcafebabe, 0x01010101, 0x80808080, 0x7f7f7f7f, 0x00010000,
  0x00000100, 0x10000000, 0x08000000, 0x00800000, 0x00008000, 0x00000080, 0x0000000f, 0x000000ff,
];

function kernelSource(cfg: SpaceConfig, program: Program, spec: CompiledSpec, nInputs: number): string {
  const prog = emitFunction(cfg, program, 'wgsl', 'prog');
  // prog helpers and spec helpers may overlap: strip duplicate function definitions
  const seen = new Set<string>();
  const dedupe = (src: string): string =>
    src
      .split('\n')
      .filter((line) => {
        const m = /^fn (\w+)\(/.exec(line);
        if (!m) return true;
        if (m[1] === 'prog' || m[1] === 'spec') return true;
        if (seen.has(m[1])) return false;
        seen.add(m[1]);
        return true;
      })
      .join('\n');
  const args = ['x', 'y', 'z'].slice(0, nInputs).join(', ');
  return `// xorcery verification kernel
${dedupe(spec.wgsl)}
${dedupe(prog)}

struct Params { base: u32, mode: u32, specialCount: u32, pad: u32 }
struct Out { count: atomic<u32>, pad0: u32, pad1: u32, pad2: u32, ce: array<u32, 24> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: Out;
@group(0) @binding(2) var<storage, read> special: array<u32>;

fn hash(v: u32) -> u32 {
  var x = v;
  x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return x;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let start = params.base + gid.x * ${PER_THREAD}u;
  for (var j = 0u; j < ${PER_THREAD}u; j++) {
    let i = start + j;
    var x = 0u; var y = 0u; var z = 0u;
    switch (params.mode) {
      case 0u: { x = i; }                                           // exhaustive unary
      case 1u: { x = i >> 16u; y = i & 0xffffu; }                    // 16-bit x 16-bit
      case 2u: { x = (i >> 16u) | 0xffff0000u; y = (i & 0xffffu) | 0xffff0000u; } // high halves set
      case 3u: { x = i >> 16u; y = (i & 0xffffu) | 0xffff0000u; }   // mixed
      case 4u: { x = (i >> 16u) | 0xffff0000u; y = i & 0xffffu; }   // mixed
      case 5u: { x = hash(i * 2u + 1u); y = hash(i * 2u + 2u); }    // hashed pairs
      case 6u: { let n = params.specialCount; let a = i % n; let b = (i / n) % n; x = special[a]; y = special[b]; }
      case 7u: { x = hash(i * 3u + 1u); y = hash(i * 3u + 2u); z = hash(i * 3u + 3u); }
      case 8u: { let n = params.specialCount; x = special[i % n]; y = special[(i / n) % n]; z = special[(i / (n * n)) % n]; }
      case 9u: { x = (i >> 16u) << 16u; y = (i & 0xffffu) << 16u; } // low halves clear
      default: { x = i; }
    }
    if (prog(${args}) != spec(x, y, z)) {
      let k = atomicAdd(&out.count, 1u);
      if (k < 8u) { out.ce[k * 3u] = x; out.ce[k * 3u + 1u] = y; out.ce[k * 3u + 2u] = z; }
    }
  }
}
`;
}

interface Pass { name: string; mode: number; count: bigint }

function plan(nInputs: number, randomLog2: number): Pass[] {
  const sp = BigInt(SPECIAL.length);
  const rnd = 1n << BigInt(randomLog2);
  if (nInputs === 1) return [{ name: 'all 2^32 inputs', mode: 0, count: 1n << 32n }];
  if (nInputs === 2) {
    return [
      { name: 'special-value grid', mode: 6, count: sp * sp },
      { name: 'all 16-bit x 16-bit pairs', mode: 1, count: 1n << 32n },
      { name: 'high halves set', mode: 2, count: 1n << 32n },
      { name: 'mixed halves (a)', mode: 3, count: 1n << 32n },
      { name: 'mixed halves (b)', mode: 4, count: 1n << 32n },
      { name: 'low halves clear', mode: 9, count: 1n << 32n },
      { name: `2^${randomLog2} hashed pairs`, mode: 5, count: rnd },
    ];
  }
  return [
    { name: 'special-value grid', mode: 8, count: sp * sp * sp },
    { name: `2^${randomLog2} hashed triples`, mode: 7, count: rnd },
  ];
}

export async function verifyProgram(opts: VerifyOptions): Promise<VerifyReport> {
  const { gpu, cfg, program, spec } = opts;
  const device = gpu.device;
  const nInputs = cfg.nInputs;
  const passes = plan(nInputs, opts.randomLog2 ?? 30);
  const total = passes.reduce((a, p) => a + p.count, 0n);
  const targetMs = opts.targetMs ?? 45;
  const t0 = performance.now();

  const code = kernelSource(cfg, program, spec, nInputs);
  const module = device.createShaderModule({ code });
  const ci = await module.getCompilationInfo();
  const errs = ci.messages.filter((m) => m.type === 'error');
  if (errs.length) throw new Error('verify shader compile error: ' + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n') + '\n' + code);
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });

  const paramsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const outSize = 16 + 24 * 4;
  const outBuf = device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const stagingBuf = device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const specialBuf = device.createBuffer({ size: SPECIAL.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(specialBuf, 0, new Uint32Array(new ArrayBuffer(SPECIAL.length * 4)).map((_, i) => SPECIAL[i] >>> 0));
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: outBuf } },
      { binding: 2, resource: { buffer: specialBuf } },
    ],
  });
  const cleanup = () => { paramsBuf.destroy(); outBuf.destroy(); stagingBuf.destroy(); specialBuf.destroy(); };

  const report: VerifyReport = {
    mode: nInputs === 1 ? 'exhaustive' : 'sampled', nInputs, checked: 0n, passes: [], mismatches: 0, counterexamples: [], elapsedMs: 0, complete: false,
  };
  device.queue.writeBuffer(outBuf, 0, new Uint32Array(new ArrayBuffer(outSize)));

  let workgroups = 256;
  const maxWg = Math.min(gpu.info.maxWorkgroupsPerDimension, 65535);
  const params = new Uint32Array(new ArrayBuffer(16));
  let done = 0n;
  try {
    for (const pass of passes) {
      let base = 0n;
      while (base < pass.count) {
        if (opts.signal?.aborted) { report.elapsedMs = performance.now() - t0; return report; }
        const remainingWg = (pass.count - base + BigInt(PER_WG) - 1n) / BigInt(PER_WG);
        const wg = remainingWg < BigInt(workgroups) ? Number(remainingWg) : workgroups;
        params[0] = Number(base & 0xffffffffn);
        params[1] = pass.mode;
        params[2] = SPECIAL.length;
        device.queue.writeBuffer(paramsBuf, 0, params);
        const enc = device.createCommandEncoder();
        const cp = enc.beginComputePass();
        cp.setPipeline(pipeline);
        cp.setBindGroup(0, bindGroup);
        cp.dispatchWorkgroups(wg);
        cp.end();
        enc.copyBufferToBuffer(outBuf, 0, stagingBuf, 0, outSize);
        const td = performance.now();
        device.queue.submit([enc.finish()]);
        await stagingBuf.mapAsync(GPUMapMode.READ);
        const ms = performance.now() - td;
        const view = new Uint32Array(stagingBuf.getMappedRange().slice(0));
        stagingBuf.unmap();
        const covered = BigInt(wg) * BigInt(PER_WG);
        const step = covered > pass.count - base ? pass.count - base : covered;
        base += covered;
        done += step;
        if (wg === workgroups) {
          const scale = Math.min(2.5, Math.max(0.3, targetMs / Math.max(ms, 1)));
          workgroups = Math.max(16, Math.min(maxWg, Math.round(workgroups * scale)));
        }
        const count = view[0];
        if (count > 0) {
          report.mismatches = count;
          for (let k = 0; k < Math.min(count, 8); k++) report.counterexamples.push([view[4 + k * 3], view[5 + k * 3], view[6 + k * 3]].slice(0, nInputs));
          report.checked = done;
          report.elapsedMs = performance.now() - t0;
          report.passes.push({ name: pass.name, count: base > pass.count ? pass.count : base });
          report.complete = true;
          return report;
        }
        opts.onProgress?.(done, total, pass.name);
      }
      report.passes.push({ name: pass.name, count: pass.count });
    }
  } finally {
    cleanup();
  }
  report.checked = total;
  report.elapsedMs = performance.now() - t0;
  report.complete = true;
  return report;
}

/** Cheap CPU screen: special-value grid + a few thousand hashed inputs. Returns a counterexample or null. */
export function quickCheck(cfg: SpaceConfig, program: Program, spec: CompiledSpec, evalProg: (inputs: number[]) => number): number[] | null {
  const n = cfg.nInputs;
  const check = (t: number[]): boolean => {
    const want = spec.fn(t[0], t[1] ?? 0, t[2] ?? 0) >>> 0;
    return evalProg(t) === want;
  };
  const sp = SPECIAL;
  if (n === 1) {
    for (const x of sp) if (!check([x])) return [x];
    for (let i = 0; i < 4096; i++) { const x = hash32(i * 7 + 3); if (!check([x])) return [x]; }
  } else if (n === 2) {
    for (const x of sp) for (const y of sp) if (!check([x, y])) return [x, y];
    for (let i = 0; i < 4096; i++) { const x = hash32(i * 2 + 1), y = hash32(i * 2 + 2); if (!check([x, y])) return [x, y]; }
  } else {
    const sub = sp.filter((_, i) => i % 3 === 0);
    for (const x of sub) for (const y of sub) for (const z of sub) if (!check([x, y, z])) return [x, y, z];
    for (let i = 0; i < 4096; i++) { const t = [hash32(i * 3 + 1), hash32(i * 3 + 2), hash32(i * 3 + 3)]; if (!check(t)) return t; }
  }
  return null;
}
