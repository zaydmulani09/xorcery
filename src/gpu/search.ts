/**
 * The search driver: iterative deepening over program length, CPU for the
 * tiny lengths and the WebGPU kernel for everything else.
 *
 * `runSearch` is an async generator so the caller stays in charge: it can
 * verify a candidate, decide the search is over, or feed a counterexample
 * back as a new sample and let the search continue where it left off.
 */
import {
  NS, MAX_RESULTS, ENTRY_WORDS, WG, PARAMS_WORDS, P_START, P_RADIX, P_LIMIT, P_S, P_MID, P_FIRSTR, P_NINPUTS, P_NCONSTS, P_MIDCOUNT, P_NOPS,
  buildLayout, generateKernel, kernelKey, packSearchData,
} from './kernel';
import { Gpu } from './device';
import { Program, Space, SpaceConfig, enumeratePrograms, evalProgram } from '../core/space';
import { Op } from '../core/isa';
import { SampleSet, sampleTuple } from '../core/samples';

export interface SearchProgress {
  L: number;
  prefixDone: bigint;
  prefixTotal: bigint;
  /** dead-code-free programs evaluated so far (all lengths) */
  evaluated: bigint;
  evaluatedThisLength: bigint;
  elapsedMs: number;
  /** programs per second over the last few dispatches */
  rate: number;
  dispatches: number;
  workgroupsPerDispatch: number;
  msPerDispatch: number;
  backend: 'cpu' | 'gpu';
}

export type SearchEvent =
  | { type: 'progress'; progress: SearchProgress }
  | { type: 'candidate'; program: Program; L: number }
  | { type: 'length-start'; L: number; prefixTotal: bigint; rawSpace: bigint; backend: 'cpu' | 'gpu' }
  | { type: 'length-done'; L: number; evaluated: bigint; elapsedMs: number }
  | { type: 'compiled'; ms: number }
  | { type: 'error'; message: string };

export interface SearchOptions {
  cfg: SpaceConfig;
  samples: SampleSet;
  targets: Uint32Array;
  minLen: number;
  maxLen: number;
  gpu: Gpu | null;
  signal?: AbortSignal;
  /** target milliseconds per GPU dispatch (TDR safety + UI responsiveness) */
  targetMs?: number;
  /** stop as soon as candidates were reported at some length (default true) */
  stopAtFirstLength?: boolean;
}

/** Mutable search state the caller may poke (e.g. to add a counterexample sample). */
export class SearchSession {
  /** replaced samples/targets take effect at the next dispatch */
  pendingSamples: { samples: SampleSet; targets: Uint32Array } | null = null;
  /** set to stop the search entirely */
  stop = false;
  /** incremented by the caller for every candidate it accepts; the search stops at the end of a length that produced one */
  accepted = 0;
}

// one compiled pipeline per (device, op list)
const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();

export function getPipeline(device: GPUDevice, ops: Op[]): Promise<GPUComputePipeline> {
  let m = pipelineCache.get(device);
  if (!m) { m = new Map(); pipelineCache.set(device, m); }
  const key = kernelKey(ops);
  let p = m.get(key);
  if (!p) {
    p = (async () => {
      const t0 = performance.now();
      const code = generateKernel(ops);
      const module = device.createShaderModule({ code });
      const ci = await module.getCompilationInfo();
      const errs = ci.messages.filter((x) => x.type === 'error');
      if (errs.length) throw new Error('shader compile error: ' + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      console.log(`[xorcery] search kernel [${key}] compiled in ${(performance.now() - t0).toFixed(0)} ms`);
      return pipeline;
    })();
    m.set(key, p);
    p.catch(() => m!.delete(key));
    if (m.size > 16) m.delete(m.keys().next().value!);
  }
  return p;
}

/** Start compiling the kernel for an op list without waiting (warm-up). */
export function precompile(gpu: Gpu, ops: Op[]): void {
  getPipeline(gpu.device, ops).catch(() => {});
}

function decodeEntry(words: Uint32Array, L: number): Program {
  const prog: Program = [];
  for (let i = 0; i < L; i++) {
    const w = words[i];
    prog.push({ op: w >>> 16, a: (w >>> 8) & 0xff, b: w & 0xff });
  }
  return prog;
}

function matchesSamples(cfg: SpaceConfig, prog: Program, samples: SampleSet, targets: Uint32Array): boolean {
  for (let s = 0; s < NS; s++) {
    if (evalProgram(cfg, prog, sampleTuple(samples, s)) !== targets[s]) return false;
  }
  return true;
}

export async function* runSearch(opts: SearchOptions, session: SearchSession): AsyncGenerator<SearchEvent> {
  const { cfg, minLen, maxLen, gpu } = opts;
  let samples = opts.samples;
  let targets = opts.targets;
  const targetMs = opts.targetMs ?? 45;
  const space = new Space(cfg, maxLen);
  const t0 = performance.now();
  let evaluated = 0n;
  const aborted = () => opts.signal?.aborted || session.stop;

  // start the (slow) shader compile now; it overlaps with the CPU lengths
  const tCompile = performance.now();
  const pipelinePromise = gpu && maxLen >= 3 ? getPipeline(gpu.device, cfg.ops) : null;
  pipelinePromise?.catch(() => {});
  let compiledReported = false;

  // GPU resources shared by every length of this search
  let res: {
    paramsBuf: GPUBuffer; pairsBuf: GPUBuffer; layoutBuf: GPUBuffer; resultsBuf: GPUBuffer; stagingBuf: GPUBuffer; dataBuf: GPUBuffer; bindGroup: GPUBindGroup;
  } | null = null;
  const cleanup = () => {
    if (!res) return;
    for (const b of [res.paramsBuf, res.pairsBuf, res.layoutBuf, res.resultsBuf, res.stagingBuf, res.dataBuf]) b.destroy();
    res = null;
  };

  try {
    for (let L = Math.max(1, minLen); L <= maxLen; L++) {
      if (aborted()) return;
      const raw = space.rawCount(L);
      const acceptedBefore = session.accepted;
      let evaluatedThisLength = 0n;
      const tL = performance.now();

      if (L <= 2 || !gpu) {
        // ---------------------------------------------------------- CPU path
        yield { type: 'length-start', L, prefixTotal: 1n, rawSpace: raw, backend: 'cpu' };
        let n = 0;
        let lastYield = performance.now();
        for (const prog of enumeratePrograms(space, L)) {
          n++;
          if (session.pendingSamples) { samples = session.pendingSamples.samples; targets = session.pendingSamples.targets; session.pendingSamples = null; }
          if (matchesSamples(cfg, prog, samples, targets)) {
            yield { type: 'candidate', program: prog, L };
            if (aborted()) return;
          }
          if ((n & 0xffff) === 0) {
            const now = performance.now();
            if (now - lastYield > 40) {
              lastYield = now;
              evaluatedThisLength = BigInt(n);
              yield { type: 'progress', progress: { L, prefixDone: 0n, prefixTotal: 1n, evaluated: evaluated + evaluatedThisLength, evaluatedThisLength, elapsedMs: now - t0, rate: 0, dispatches: 0, workgroupsPerDispatch: 0, msPerDispatch: 0, backend: 'cpu' } };
              await new Promise((r) => setTimeout(r, 0));
              if (aborted()) return;
            }
          }
          if (!gpu && L >= 4 && n > 50_000_000) {
            yield { type: 'error', message: 'CPU fallback stopped after 50M programs; WebGPU is needed for longer programs' };
            return;
          }
        }
        evaluatedThisLength = BigInt(n);
        evaluated += evaluatedThisLength;
        yield { type: 'length-done', L, evaluated: evaluatedThisLength, elapsedMs: performance.now() - tL };
        if (session.accepted > acceptedBefore && (opts.stopAtFirstLength ?? true)) return;
        continue;
      }

      // -------------------------------------------------------------- GPU path
      const device = gpu.device;
      const S = L - 2;
      const prefixRadix: number[] = [];
      let prefixTotal = 1n;
      for (let i = 0; i < S; i++) { prefixRadix.push(space.slotCount(i)); prefixTotal *= BigInt(space.slotCount(i)); }
      const midCount = space.slotCount(S);
      yield { type: 'length-start', L, prefixTotal, rawSpace: raw, backend: 'gpu' };

      let pipeline: GPUComputePipeline;
      try {
        pipeline = await pipelinePromise!;
      } catch (e) {
        yield { type: 'error', message: String((e as Error).message ?? e) };
        return;
      }
      if (!compiledReported) { compiledReported = true; yield { type: 'compiled', ms: performance.now() - tCompile }; }
      if (aborted()) return;

      if (!res) {
        const lay = buildLayout(space, maxLen);
        const paramsBuf = device.createBuffer({ size: PARAMS_WORDS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const pairsBuf = device.createBuffer({ size: lay.pairs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(pairsBuf, 0, lay.pairs);
        const layoutBuf = device.createBuffer({ size: lay.layout.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(layoutBuf, 0, lay.layout);
        const resultsSize = 16 + MAX_RESULTS * ENTRY_WORDS * 4;
        const resultsBuf = device.createBuffer({ size: resultsSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const stagingBuf = device.createBuffer({ size: resultsSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const data = packSearchData(cfg.nInputs, samples.inputs, targets, cfg.consts);
        const dataBuf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(dataBuf, 0, data);
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: pairsBuf } },
            { binding: 2, resource: { buffer: resultsBuf } },
            { binding: 3, resource: { buffer: dataBuf } },
            { binding: 4, resource: { buffer: layoutBuf } },
          ],
        });
        res = { paramsBuf, pairsBuf, layoutBuf, resultsBuf, stagingBuf, dataBuf, bindGroup };
      }
      const resultsSize = 16 + MAX_RESULTS * ENTRY_WORDS * 4;

      // candidates per workgroup upper bound, to keep the u32 'evaluated' counter from wrapping
      const n = cfg.nInputs, k = cfg.consts.length;
      const MID = n + k + S;
      let maxInner = 0;
      for (const op of cfg.ops) maxInner += op.kind === 'unary' ? 1 : op.kind === 'comm' ? MID + 1 : 2 * MID + 1;
      const perWorkgroup = BigInt(midCount) * BigInt(maxInner);
      const wgCap = Number(3_000_000_000n / (perWorkgroup === 0n ? 1n : perWorkgroup));
      const maxWg = Math.max(16, Math.min(gpu.info.maxWorkgroupsPerDimension, 65535, wgCap));

      let workgroups = Math.min(maxWg, 256);
      let prefixDone = 0n;
      let dispatches = 0;
      let msPerDispatch = 0;
      const rateWindow: { t: number; n: bigint }[] = [];
      const params = new Uint32Array(new ArrayBuffer(PARAMS_WORDS * 4));
      for (let i = 0; i < S; i++) params[P_RADIX + i] = prefixRadix[i];
      params[P_S] = S; params[P_MID] = MID; params[P_FIRSTR] = n + k; params[P_NINPUTS] = n; params[P_NCONSTS] = k; params[P_MIDCOUNT] = midCount; params[P_NOPS] = cfg.ops.length;
      let lastProgress = performance.now();

      while (prefixDone < prefixTotal) {
        if (aborted()) return;
        if (session.pendingSamples) {
          samples = session.pendingSamples.samples; targets = session.pendingSamples.targets; session.pendingSamples = null;
          device.queue.writeBuffer(res.dataBuf, 0, packSearchData(cfg.nInputs, samples.inputs, targets, cfg.consts));
        }
        const remaining = prefixTotal - prefixDone;
        const limit = remaining < BigInt(workgroups) ? Number(remaining) : workgroups;
        let rest = prefixDone;
        for (let i = S - 1; i >= 0; i--) {
          const r = BigInt(prefixRadix[i]);
          params[P_START + i] = Number(rest % r);
          rest /= r;
        }
        params[P_LIMIT] = limit;
        device.queue.writeBuffer(res.paramsBuf, 0, params);
        const enc = device.createCommandEncoder();
        enc.clearBuffer(res.resultsBuf, 0, 16);
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, res.bindGroup);
        pass.dispatchWorkgroups(limit);
        pass.end();
        enc.copyBufferToBuffer(res.resultsBuf, 0, res.stagingBuf, 0, resultsSize);
        const td = performance.now();
        device.queue.submit([enc.finish()]);
        await res.stagingBuf.mapAsync(GPUMapMode.READ);
        const ms = performance.now() - td;
        const view = new Uint32Array(res.stagingBuf.getMappedRange().slice(0));
        res.stagingBuf.unmap();
        const count = view[0];
        const evalCount = BigInt(view[1]);
        prefixDone += BigInt(limit);
        evaluated += evalCount;
        evaluatedThisLength += evalCount;
        dispatches++;
        msPerDispatch = ms;
        if (limit === workgroups) {
          const scale = Math.min(2.5, Math.max(0.3, targetMs / Math.max(ms, 1)));
          workgroups = Math.max(16, Math.min(maxWg, Math.round(workgroups * scale)));
        }
        const now = performance.now();
        rateWindow.push({ t: now, n: evaluated });
        while (rateWindow.length > 12) rateWindow.shift();

        if (count > 0) {
          const nEntries = Math.min(count, MAX_RESULTS);
          const seen = new Set<string>();
          for (let e = 0; e < nEntries; e++) {
            const prog = decodeEntry(view.subarray(4 + e * ENTRY_WORDS, 4 + e * ENTRY_WORDS + L), L);
            const key = JSON.stringify(prog);
            if (seen.has(key)) continue;
            seen.add(key);
            // tri-check: the CPU interpreter must agree with the GPU on the samples
            if (!matchesSamples(cfg, prog, samples, targets)) {
              yield { type: 'error', message: `GPU/CPU disagreement on candidate ${key} — please report this` };
              continue;
            }
            yield { type: 'candidate', program: prog, L };
            if (aborted()) return;
          }
        }
        if (now - lastProgress > 60 || prefixDone >= prefixTotal) {
          lastProgress = now;
          const first = rateWindow[0], last = rateWindow[rateWindow.length - 1];
          const rate = rateWindow.length > 1 && last.t > first.t ? Number(last.n - first.n) / ((last.t - first.t) / 1000) : 0;
          yield {
            type: 'progress',
            progress: { L, prefixDone, prefixTotal, evaluated, evaluatedThisLength, elapsedMs: now - t0, rate, dispatches, workgroupsPerDispatch: workgroups, msPerDispatch, backend: 'gpu' },
          };
        }
      }
      yield { type: 'length-done', L, evaluated: evaluatedThisLength, elapsedMs: performance.now() - tL };
      if (session.accepted > acceptedBefore && (opts.stopAtFirstLength ?? true)) return;
    }
  } finally {
    cleanup();
  }
}

export { WG };
