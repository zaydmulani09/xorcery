/**
 * GPU self-test: shader compilers have bugs (the Intel D3D12 driver
 * miscompiled `countOneBits(~x & (x - 1))` into `x` while this was being
 * built), and a superoptimizer that silently loses programs to a
 * miscompiled op is worse than one that refuses to run. So before trusting a
 * device, every op's WGSL template is evaluated on the GPU for a few hundred
 * operand pairs and compared with the CPU implementation, and one small
 * search is cross-checked against the CPU enumerator.
 */
import { OPS, Op } from '../core/isa';
import { Space, SpaceConfig, enumeratePrograms, evalProgram, programToString } from '../core/space';
import { makeSamples, computeTargets, sampleTuple } from '../core/samples';
import { hash32 } from '../core/u32';
import { Gpu } from './device';
import { runSearch, SearchSession } from './search';
import { HELPERS, HELPER_DEPS } from '../core/isa';

export interface SelfTestReport {
  ok: boolean;
  /** ops whose GPU results disagreed with the CPU */
  badOps: { op: string; a: number; b: number; gpu: number; cpu: number }[];
  /** the search cross-check: null if skipped */
  search: { ok: boolean; gpuCount: number; cpuCount: number; detail?: string } | null;
  ms: number;
}

const PAIRS = 256;

function helperBlock(ops: Op[]): string {
  const need = new Set<string>();
  for (const op of ops) if (op.wgsl.helper) need.add(op.wgsl.helper);
  for (const h of Array.from(need)) for (const d of HELPER_DEPS[h] ?? []) need.add(d);
  return Object.keys(HELPERS.wgsl).filter((h) => need.has(h)).map((h) => HELPERS.wgsl[h]).join('\n');
}

export async function opSelfTest(gpu: Gpu, ops: Op[] = OPS): Promise<SelfTestReport['badOps']> {
  const device = gpu.device;
  const cases = ops.map((op, i) => `    case ${i}u: { v = ${op.wgsl.t.replace('$a', 'a').replace('$b', 'b')}; }`).join('\n');
  const code = `
${helperBlock(ops)}
@group(0) @binding(0) var<storage, read> ins: array<u32>;
@group(0) @binding(1) var<storage, read_write> outs: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${ops.length * PAIRS}u) { return; }
  let op = i / ${PAIRS}u;
  let k = i % ${PAIRS}u;
  // operands derive from the thread id, like a real kernel, not from a plain load
  let a = ins[k * 2u] ^ (i * 0u);
  let b = ins[k * 2u + 1u];
  var v = 0u;
  switch (op) {
${cases}
    default: { v = 0u; }
  }
  outs[i] = v;
}`;
  const module = device.createShaderModule({ code });
  const ci = await module.getCompilationInfo();
  const errs = ci.messages.filter((m) => m.type === 'error');
  if (errs.length) throw new Error('self-test shader failed to compile: ' + errs.map((e) => e.message).join('; '));
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });

  // operand pairs: specials, single bits, shift-ish amounts, hashed
  const vals: number[] = [0, 1, 2, 3, 31, 32, 33, 0xffffffff, 0x80000000, 0x7fffffff, 0x0000ffff, 0xffff0000, 0x55555555, 0xaaaaaaaa, 0x12345678];
  for (let i = 0; i < 32; i++) vals.push((1 << i) >>> 0);
  const ins = new Uint32Array(new ArrayBuffer(PAIRS * 2 * 4));
  for (let k = 0; k < PAIRS; k++) {
    const a = k < vals.length ? vals[k] : hash32(k * 3 + 1);
    const b = k < vals.length ? vals[(k * 7 + 3) % vals.length] : k < 2 * vals.length ? (k % 33) : hash32(k * 5 + 2);
    ins[k * 2] = a >>> 0;
    ins[k * 2 + 1] = b >>> 0;
  }
  const inBuf = device.createBuffer({ size: ins.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, ins);
  const n = ops.length * PAIRS;
  const outBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, staging, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  inBuf.destroy(); outBuf.destroy(); staging.destroy();

  const bad: SelfTestReport['badOps'] = [];
  ops.forEach((op, oi) => {
    for (let k = 0; k < PAIRS; k++) {
      const a = ins[k * 2], b = ins[k * 2 + 1];
      const cpu = op.fn(a, b) >>> 0;
      const g = out[oi * PAIRS + k];
      if (g !== cpu) { bad.push({ op: op.id, a, b, gpu: g, cpu }); break; }
    }
  });
  return bad;
}

/** Cross-check one small search against the CPU enumerator (base ops, one input, length 3). */
export async function searchSelfTest(gpu: Gpu): Promise<NonNullable<SelfTestReport['search']>> {
  const cfg: SpaceConfig = { nInputs: 1, consts: [1, 31], ops: OPS.filter((o) => o.group === 'base') };
  const samples = makeSamples(1);
  // target: abs(x) = (x ^ (x >> 31)) - (x >> 31), which has several 3-instruction forms
  const targets = computeTargets(samples, (x) => ((x | 0) < 0 ? -x : x) >>> 0);
  const space = new Space(cfg, 3);
  const cpuSet = new Set<string>();
  for (const p of enumeratePrograms(space, 3)) {
    let ok = true;
    for (let s = 0; s < targets.length; s++) if (evalProgram(cfg, p, sampleTuple(samples, s)) !== targets[s]) { ok = false; break; }
    if (ok) cpuSet.add(programToString(cfg, p));
  }
  const gpuSet = new Set<string>();
  let gpuEvaluated = 0n;
  const session = new SearchSession();
  for await (const ev of runSearch({ cfg, samples, targets, minLen: 3, maxLen: 3, gpu, stopAtFirstLength: false }, session)) {
    if (ev.type === 'candidate') gpuSet.add(programToString(cfg, ev.program));
    if (ev.type === 'length-done') gpuEvaluated = ev.evaluated;
    if (ev.type === 'error') return { ok: false, gpuCount: gpuSet.size, cpuCount: cpuSet.size, detail: ev.message };
  }
  let cpuEvaluated = 0;
  for (const _ of enumeratePrograms(space, 3)) cpuEvaluated++;
  const missing = Array.from(cpuSet).filter((p) => !gpuSet.has(p));
  const extra = Array.from(gpuSet).filter((p) => !cpuSet.has(p));
  const ok = missing.length === 0 && extra.length === 0 && Number(gpuEvaluated) === cpuEvaluated;
  return {
    ok, gpuCount: gpuSet.size, cpuCount: cpuSet.size,
    detail: ok ? undefined : `evaluated gpu=${gpuEvaluated} cpu=${cpuEvaluated}; missing=[${missing.join(' | ')}] extra=[${extra.join(' | ')}]`,
  };
}

export async function selfTest(gpu: Gpu, withSearch = true): Promise<SelfTestReport> {
  const t0 = performance.now();
  const badOps = await opSelfTest(gpu);
  const search = withSearch ? await searchSelfTest(gpu) : null;
  return { ok: badOps.length === 0 && (search?.ok ?? true), badOps, search, ms: performance.now() - t0 };
}
