import { getGpu } from './gpu/device';
import { runEngine } from './engine';
import { OPS } from './core/isa';

const log = (s: string) => { const el = document.getElementById('log')!; el.textContent += s + '\n'; };
const q = new URLSearchParams(location.search);
const spec = q.get('spec') ?? '(int)x < 0 ? -x : x';
const maxLen = Number(q.get('len') ?? 4);
const groups = (q.get('groups') ?? 'base').split(',');
const consts = (q.get('consts') ?? '0,1,31,-1').split(',').map((s) => Number(s) >>> 0);

(async () => {
  const gpu = await getGpu((r) => log('DEVICE LOST ' + r));
  log('gpu: ' + (gpu ? gpu.info.name + ' ' + JSON.stringify(gpu.info) : 'none'));
  const opIds = OPS.filter((o) => groups.includes(o.group)).map((o) => o.id);
  log(`spec: ${spec}  ops: ${opIds.join(' ')}  consts: ${consts.join(',')}  maxLen: ${maxLen}`);
  const t0 = performance.now();
  let lastP = 0;
  const res = await runEngine({ spec, opIds, consts, maxLen }, gpu, {
    onLength: (e) => log(`L=${e.L} backend=${e.backend} prefixes=${e.prefixTotal} raw=${e.rawSpace}`),
    onProgress: (p) => { if (performance.now() - lastP > 500) { lastP = performance.now(); log(`  L=${p.L} ${p.prefixDone}/${p.prefixTotal} evaluated=${p.evaluated} rate=${(p.rate / 1e6).toFixed(1)}M/s wg=${p.workgroupsPerDispatch} ms=${p.msPerDispatch.toFixed(1)}`); } },
    onLengthDone: (e) => log(`L=${e.L} done: evaluated=${e.evaluated} in ${e.elapsedMs.toFixed(0)}ms`),
    onCandidate: (s) => log(`CANDIDATE L=${s.L}: ${s.expr}`),
    onVerifyProgress: (s, d, t, p) => { if (performance.now() - lastP > 500) { lastP = performance.now(); log(`  verify ${p}: ${d}/${t}`); } },
    onVerified: (s) => log(`VERIFIED ${s.expr}: ${JSON.stringify({ ...s.verify, checked: String(s.verify?.checked), passes: s.verify?.passes.map((p) => p.name + '=' + p.count) })}`),
    onError: (m) => log('ERROR ' + m),
    onStatus: (m) => log('status: ' + m),
  });
  log(`DONE in ${(performance.now() - t0).toFixed(0)}ms: L=${res.L} solutions=${res.solutions.length} evaluated=${res.evaluated} exhausted=${res.exhausted} screened=${res.screenedOut} err=${res.error ?? ''}`);
  for (const s of res.solutions) log('  ' + s.expr + (s.verify ? ` [${s.verify.mode}, checked ${s.verify.checked}, ${s.verify.elapsedMs.toFixed(0)}ms]` : ''));
  (window as any).__result = res;
})();

// debug: ?verifyTest=1 verifies a deliberately wrong program (x | 1 vs x + 1)
if (q.get('verifyTest')) {
  (async () => {
    const { verifyProgram } = await import('./gpu/verify');
    const { compileSpec } = await import('./spec/compile');
    const gpu = await getGpu();
    if (!gpu) return;
    const spec = compileSpec(q.get('vspec') ?? 'x + 1');
    const ops = OPS.filter((o) => o.group === 'base');
    const cfg = { nInputs: spec.nInputs, consts: [1], ops };
    const orIdx = ops.findIndex((o) => o.id === 'or');
    const prog = [{ op: orIdx, a: 0, b: spec.nInputs }];
    const t0 = performance.now();
    const rep = await verifyProgram({ gpu, cfg, program: prog, spec, onProgress: (d, t, p) => log(`  vt ${p} ${d}/${t}`) });
    log(`VERIFYTEST ${(performance.now() - t0).toFixed(0)}ms: ` + JSON.stringify({ ...rep, checked: String(rep.checked), passes: rep.passes.map((p) => p.name + '=' + p.count) }));
  })();
}
