/**
 * Orchestration: spec -> samples -> search -> screen -> verify.
 *
 * The search only sees 32 samples; every candidate it reports is screened on
 * the CPU (special values + hashed inputs) and, once the search is over,
 * verified on the GPU (exhaustively for one-input specs). A counterexample at
 * either stage replaces a sample and the search continues, so wrong answers
 * are never shown, only used to sharpen the screen.
 */
import { OpGroup, opsForIds } from './core/isa';
import { Program, SpaceConfig, Space, evalProgram, isDeadCodeFree } from './core/space';
import { SampleSet, computeTargets, makeSamples } from './core/samples';
import { compileToJs, exprString } from './core/program';
import { compileSpec, CompiledSpec } from './spec/compile';
import { Gpu } from './gpu/device';
import { runSearch, SearchProgress, SearchSession } from './gpu/search';
import { quickCheck, verifyProgram, VerifyReport } from './gpu/verify';
import { NS } from './gpu/kernel';

export interface EngineConfig {
  spec: string;
  opIds: string[];
  consts: number[];
  maxLen: number;
  /** collect alternative minimal programs for up to this many ms after the first hit */
  graceMs?: number;
  verifyRandomLog2?: number;
}

export interface Solution {
  program: Program;
  L: number;
  expr: string;
  verify?: VerifyReport;
  /** counterexample from the CPU screen or GPU verification, if any */
  rejected?: number[];
}

export interface EngineResult {
  spec: CompiledSpec;
  cfg: SpaceConfig;
  solutions: Solution[];
  /** the length at which solutions were found (0 if none) */
  L: number;
  evaluated: bigint;
  searchMs: number;
  verifyMs: number;
  /** true if every length up to maxLen was exhausted without a solution */
  exhausted: boolean;
  aborted: boolean;
  /** raw program count per length (for display) */
  rawCounts: bigint[];
  screenedOut: number;
  error?: string;
}

export interface EngineEvents {
  onStatus?: (msg: string) => void;
  onProgress?: (p: SearchProgress) => void;
  onLength?: (ev: { L: number; prefixTotal: bigint; rawSpace: bigint; backend: 'cpu' | 'gpu' }) => void;
  onLengthDone?: (ev: { L: number; evaluated: bigint; elapsedMs: number }) => void;
  onCandidate?: (sol: Solution) => void;
  onVerifyProgress?: (sol: Solution, done: bigint, total: bigint, pass: string) => void;
  onVerified?: (sol: Solution) => void;
  onError?: (msg: string) => void;
}

export function buildConfig(spec: CompiledSpec, opIds: string[], consts: number[]): SpaceConfig {
  const ops = opsForIds(opIds);
  if (ops.length === 0) throw new Error('enable at least one op');
  const uniq = Array.from(new Set(consts.map((c) => c >>> 0)));
  if (uniq.length > 12) throw new Error('at most 12 constants');
  return { nInputs: spec.nInputs, consts: uniq, ops };
}

/** Replace the last sample with a counterexample (keeps NS fixed). */
function withCounterexample(samples: SampleSet, spec: CompiledSpec, ce: number[]): { samples: SampleSet; targets: Uint32Array } {
  const inputs = samples.inputs.map((a) => Uint32Array.from(a));
  for (let i = 0; i < samples.nInputs; i++) inputs[i][NS - 1] = (ce[i] ?? 0) >>> 0;
  const next: SampleSet = { nInputs: samples.nInputs, inputs };
  return { samples: next, targets: computeTargets(next, spec.fn) };
}

export async function runEngine(config: EngineConfig, gpu: Gpu | null, events: EngineEvents, signal?: AbortSignal): Promise<EngineResult> {
  const spec = compileSpec(config.spec);
  const cfg = buildConfig(spec, config.opIds, config.consts);
  const maxLen = Math.max(1, Math.min(8, config.maxLen | 0));
  const space = new Space(cfg, maxLen);
  const rawCounts: bigint[] = [];
  for (let L = 1; L <= maxLen; L++) rawCounts.push(space.rawCount(L));

  let samples = makeSamples(spec.nInputs);
  let targets = computeTargets(samples, spec.fn);
  const result: EngineResult = {
    spec, cfg, solutions: [], L: 0, evaluated: 0n, searchMs: 0, verifyMs: 0, exhausted: false, aborted: false, rawCounts, screenedOut: 0,
  };
  const t0 = performance.now();
  const graceMs = config.graceMs ?? 1500;

  let minLen = 1;
  for (let round = 0; round < 4; round++) {
    const session = new SearchSession();
    const seen = new Set<string>();
    const accepted: Solution[] = [];
    let firstHitAt = 0;
    let foundL = 0;
    let lengthStartedAt = performance.now();

    const iter = runSearch({ cfg, samples, targets, minLen, maxLen, gpu, signal, stopAtFirstLength: true }, session);
    for await (const ev of iter) {
      if (signal?.aborted) { result.aborted = true; break; }
      switch (ev.type) {
        case 'length-start':
          lengthStartedAt = performance.now();
          events.onLength?.({ L: ev.L, prefixTotal: ev.prefixTotal, rawSpace: ev.rawSpace, backend: ev.backend });
          break;
        case 'progress':
          result.evaluated = ev.progress.evaluated;
          events.onProgress?.(ev.progress);
          if (foundL && performance.now() - firstHitAt > Math.min(graceMs, 0.5 * (performance.now() - lengthStartedAt) + 300)) session.stop = true;
          break;
        case 'length-done':
          events.onLengthDone?.(ev);
          break;
        case 'compiled':
          events.onStatus?.(`kernel compiled in ${ev.ms.toFixed(0)} ms`);
          break;
        case 'error':
          events.onError?.(ev.message);
          if (!ev.message.startsWith('GPU/CPU')) { result.error = ev.message; session.stop = true; }
          break;
        case 'candidate': {
          const key = JSON.stringify(ev.program);
          if (seen.has(key)) break;
          seen.add(key);
          if (!isDeadCodeFree(cfg, ev.program)) break;
          const evalProg = (inputs: number[]) => evalProgram(cfg, ev.program, inputs);
          const ce = quickCheck(cfg, ev.program, spec, evalProg);
          if (ce) {
            result.screenedOut++;
            const next = withCounterexample(samples, spec, ce);
            samples = next.samples; targets = next.targets;
            session.pendingSamples = next;
            break;
          }
          // the printed code must mean the same thing as the interpreter
          const js = compileToJs(cfg, ev.program);
          let printerOk = true;
          for (let s = 0; s < NS; s++) {
            const t = samples.inputs.map((a) => a[s]);
            if (js(...t) !== evalProg(t)) { printerOk = false; break; }
          }
          if (!printerOk) { events.onError?.('printer/interpreter disagreement on ' + key); break; }
          const sol: Solution = { program: ev.program, L: ev.L, expr: exprString(cfg, ev.program) };
          accepted.push(sol);
          if (!foundL) { foundL = ev.L; firstHitAt = performance.now(); }
          events.onCandidate?.(sol);
          break;
        }
      }
      if (result.error) break;
    }
    result.searchMs = performance.now() - t0;
    if (result.aborted || result.error) break;
    if (accepted.length === 0) { result.exhausted = true; break; }

    // ---- full verification -------------------------------------------------
    const tv = performance.now();
    let anyGood = false;
    let newCe: number[] | null = null;
    const toVerify = accepted.slice(0, 6);
    for (const sol of toVerify) {
      if (signal?.aborted) { result.aborted = true; break; }
      if (!gpu) { anyGood = true; continue; }
      events.onStatus?.('verifying');
      try {
        const rep = await verifyProgram({
          gpu, cfg, program: sol.program, spec, signal,
          randomLog2: config.verifyRandomLog2,
          onProgress: (d, t, p) => events.onVerifyProgress?.(sol, d, t, p),
        });
        sol.verify = rep;
        if (rep.mismatches > 0) { sol.rejected = rep.counterexamples[0]; newCe = newCe ?? rep.counterexamples[0]; }
        else if (rep.complete) anyGood = true;
        events.onVerified?.(sol);
      } catch (e) {
        events.onError?.(String((e as Error).message ?? e));
        result.error = String((e as Error).message ?? e);
        break;
      }
    }
    result.verifyMs += performance.now() - tv;
    const good = toVerify.filter((s) => !s.rejected);
    if (anyGood || !gpu) {
      result.solutions = good.length ? good : toVerify;
      result.L = foundL;
      break;
    }
    if (result.error || result.aborted) break;
    // every candidate was wrong: learn the counterexample and search this length again
    if (newCe) {
      const next = withCounterexample(samples, spec, newCe);
      samples = next.samples; targets = next.targets;
      result.screenedOut += toVerify.length;
      minLen = foundL;
      events.onStatus?.('counterexample found; searching again');
      continue;
    }
    break;
  }
  return result;
}

export const DEFAULT_GROUPS: OpGroup[] = ['base', 'mul', 'bits', 'cmp'];
