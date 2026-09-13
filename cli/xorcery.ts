#!/usr/bin/env -S deno run --allow-all
/**
 * xorcery CLI — the same engine as the web app, on your GPU, from a terminal.
 * Deno ships WebGPU, so nothing else is needed:
 *
 *   deno task cli "x & (x - 1)"
 *   deno task cli "(int)x < 0 ? -x : x" --len 5 --json
 *   deno task cli --preset p18
 *   deno task cli --bench            # every preset, markdown table
 *
 * (deno.json enables extension-less imports; without it: deno run -A --sloppy-imports cli/xorcery.ts ...)
 *
 * Options:
 *   --ops base,cmp,bits,...   op groups (default: base) or --op add,sub,...
 *   --consts 0,1,31,-1        constant pool (default: 0,1,31,-1)
 *   --len N                   maximum program length (default 5)
 *   --lang c|rust|js|wgsl     code to print (default c)
 *   --json                    machine-readable output
 *   --cpu                     ignore the GPU (CPU search; short programs only)
 */
import { runEngine, EngineResult } from '../src/engine';
import { GROUPS, OPS, OpGroup, Lang } from '../src/core/isa';
import { emitFunction } from '../src/core/program';
import { getGpu } from '../src/gpu/device';
import { PRESETS, DEFAULT_CONSTS, presetById } from '../src/presets';
import { selfTest } from '../src/gpu/selftest';

const args = Deno.args;
const flag = (name: string): string | undefined => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : undefined;
};
const has = (name: string): boolean => args.includes('--' + name);
const positional = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--') && !['json', 'bench', 'cpu', 'help', 'selftest'].includes(args[i - 1].slice(2))));

if (has('help') || (args.length === 0)) {
  console.log(`xorcery — your GPU rediscovers bit hacks

usage: xorcery "<spec>" [--ops base,cmp] [--consts 0,1,31,-1] [--len 5] [--lang c] [--json]
       xorcery --preset p18
       xorcery --bench
       xorcery --selftest

spec: a C expression over 32-bit words, e.g. "(int)x < 0 ? -x : x"
op groups: ${GROUPS.map((g) => `${g.id} (${g.hint})`).join('; ')}`);
  Deno.exit(0);
}

const parseConst = (s: string): number => {
  s = s.trim();
  if (s.startsWith('-')) return -Number(s.slice(1)) >>> 0;
  return Number(s) >>> 0;
};

const fmt = (n: bigint | number) => n.toLocaleString('en-US');
const ms = (x: number) => (x < 1000 ? `${x.toFixed(0)} ms` : `${(x / 1000).toFixed(2)} s`);

const gpu = has('cpu') ? null : await getGpu();
if (!has('json')) console.error(gpu ? `gpu: ${gpu.info.name || gpu.info.vendor || 'WebGPU device'}` : 'gpu: none (CPU fallback)');

if (has('selftest')) {
  if (!gpu) { console.error('no GPU'); Deno.exit(1); }
  const rep = await selfTest(gpu, true);
  console.log(JSON.stringify(rep, (_, v) => (typeof v === 'bigint' ? v.toString() : v), 2));
  Deno.exit(rep.ok ? 0 : 1);
}

interface Job { name: string; spec: string; opIds: string[]; consts: number[]; maxLen: number; textbook?: number }

function jobFromArgs(): Job {
  const presetId = flag('preset');
  if (presetId) {
    const p = presetById(presetId);
    if (!p) { console.error(`unknown preset ${presetId}; known: ${PRESETS.map((x) => x.id).join(' ')}`); Deno.exit(2); }
    const ids = p.ops ?? OPS.filter((o) => (p.groups ?? ['base']).includes(o.group)).map((o) => o.id);
    return { name: p.name, spec: p.spec, opIds: ids, consts: p.consts ?? DEFAULT_CONSTS, maxLen: Number(flag('len') ?? p.maxLen ?? 5), textbook: p.size };
  }
  const spec = positional[0];
  if (!spec) { console.error('missing spec'); Deno.exit(2); }
  let opIds: string[];
  if (flag('op')) opIds = flag('op')!.split(',').map((s) => s.trim());
  else {
    const groups = (flag('ops') ?? 'base').split(',').map((s) => s.trim()) as OpGroup[];
    opIds = OPS.filter((o) => groups.includes(o.group)).map((o) => o.id);
  }
  const consts = (flag('consts') ?? '0,1,31,-1').split(',').filter(Boolean).map(parseConst);
  return { name: spec, spec, opIds, consts, maxLen: Number(flag('len') ?? 5) };
}

async function runJob(job: Job, quiet: boolean): Promise<EngineResult> {
  let last = 0;
  const enc = new TextEncoder();
  const res = await runEngine({ spec: job.spec, opIds: job.opIds, consts: job.consts, maxLen: job.maxLen }, gpu, {
    onProgress: (p) => {
      if (quiet) return;
      const now = performance.now();
      if (now - last < 200) return;
      last = now;
      const frac = p.prefixTotal > 0n ? (Number(p.prefixDone) / Number(p.prefixTotal)) * 100 : 0;
      Deno.stderr.writeSync(enc.encode(`\r  L=${p.L} ${frac.toFixed(1).padStart(5)}%  ${fmt(p.evaluated).padStart(16)} programs  ${(p.rate / 1e9).toFixed(2)} G/s   `));
    },
    onLengthDone: (e) => { if (!quiet) Deno.stderr.writeSync(enc.encode(`\r  L=${e.L} done: ${fmt(e.evaluated)} programs in ${ms(e.elapsedMs)}                    \n`)); },
    onVerifyProgress: (_s, d, t, pass) => {
      if (quiet) return;
      const now = performance.now();
      if (now - last < 200 && d < t) return;
      last = now;
      Deno.stderr.writeSync(enc.encode(`\r  verifying: ${pass} ${fmt(d)}/${fmt(t)}   `));
    },
    onStatus: (m) => { if (!quiet && m.startsWith('kernel compiled')) Deno.stderr.writeSync(enc.encode(`  ${m}\n`)); },
    onError: (m) => console.error('\n  ! ' + m),
  });
  if (!quiet) Deno.stderr.writeSync(enc.encode('\r' + ' '.repeat(70) + '\r'));
  return res;
}

function verifyText(res: EngineResult): string {
  const s = res.solutions[0];
  if (!s?.verify) return 'matches all samples (not verified)';
  const v = s.verify;
  if (v.mismatches) return 'REJECTED';
  return v.mode === 'exhaustive' ? `proved on all ${fmt(v.checked)} inputs (${v.backend})` : `verified on ${fmt(v.checked)} inputs (not exhaustive, ${v.backend})`;
}

if (has('bench')) {
  const rows = ['| preset | textbook | found | program | programs tried | search | check |', '|---|---:|---:|---|---:|---:|---|'];
  for (const p of PRESETS) {
    const ids = p.ops ?? OPS.filter((o) => (p.groups ?? ['base']).includes(o.group)).map((o) => o.id);
    const job: Job = { name: p.name, spec: p.spec, opIds: ids, consts: p.consts ?? DEFAULT_CONSTS, maxLen: Math.min(Number(flag('len') ?? p.maxLen ?? 5), p.hard ? 4 : 8), textbook: p.size };
    console.error(`\n${p.name}`);
    const res = await runJob(job, false);
    const s = res.solutions[0];
    rows.push(`| ${p.name} | ${p.size ?? ''} | ${s ? res.L : `none ≤ ${job.maxLen}`} | ${s ? '`' + s.expr + '`' : ''} | ${fmt(res.evaluated)} | ${ms(res.searchMs)} | ${s ? verifyText(res) : ''} |`);
  }
  console.log('\n' + rows.join('\n'));
  console.log(`\nxorcery on ${gpu?.info.name || 'CPU'}`);
  Deno.exit(0);
}

const job = jobFromArgs();
if (!has('json')) console.error(`spec: ${job.spec}\nops: ${job.opIds.join(' ')}\nconsts: ${job.consts.map((c) => (c >= 0xfffff000 ? String(c - 0x100000000) : c >= 1024 ? '0x' + c.toString(16) : String(c))).join(', ')}\nmax length: ${job.maxLen}`);
const res = await runJob(job, has('json'));

if (has('json')) {
  console.log(JSON.stringify({
    spec: job.spec, ops: job.opIds, consts: job.consts, maxLen: job.maxLen,
    gpu: gpu?.info ?? null,
    length: res.L, evaluated: res.evaluated.toString(), searchMs: res.searchMs, verifyMs: res.verifyMs, exhausted: res.exhausted, error: res.error ?? null,
    solutions: res.solutions.map((s) => ({
      expr: s.expr, program: s.program,
      consts: s.cfg.consts, synthesized: s.synthesized,
      c: emitFunction(s.cfg, s.program, 'c'), rust: emitFunction(s.cfg, s.program, 'rust'), js: emitFunction(s.cfg, s.program, 'js'), wgsl: emitFunction(s.cfg, s.program, 'wgsl'),
      verify: s.verify ? { ...s.verify, checked: s.verify.checked.toString(), passes: s.verify.passes.map((p) => ({ name: p.name, count: p.count.toString() })) } : null,
    })),
  }, null, 2));
} else if (res.solutions.length) {
  const s = res.solutions[0];
  console.log(`\n${res.L} instruction${res.L > 1 ? 's' : ''}:  ${s.expr}`);
  console.log(`${verifyText(res)} · ${fmt(res.evaluated)} programs searched in ${ms(res.searchMs)}`);
  if (res.solutions.length > 1) console.log('alternatives: ' + res.solutions.slice(1).map((a) => a.expr).join(' | '));
  console.log('\n' + emitFunction(s.cfg, s.program, (flag('lang') ?? 'c') as Lang));
} else {
  console.log(res.error ? `error: ${res.error}` : `no program of length ≤ ${job.maxLen} computes this with the chosen ops and constants (${fmt(res.evaluated)} tried in ${ms(res.searchMs)})`);
  Deno.exit(1);
}
