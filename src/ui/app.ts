/**
 * The xorcery web app: spec editor, ISA/constant controls, live search
 * telemetry, result + proof card, code emitters, preset gallery and the
 * benchmark runner. Plain DOM, no framework.
 */
import { GROUPS, OPS, OP_BY_ID, Lang } from '../core/isa';
import { Space, SpaceConfig, deadCodeFreeCount } from '../core/space';
import { emitFunction, listing } from '../core/program';
import { compileSpec } from '../spec/compile';
import { SpecError } from '../spec/parser';
import { getGpu, Gpu, hasWebGpu } from '../gpu/device';
import { precompile } from '../gpu/search';
import { selfTest } from '../gpu/selftest';
import { runEngine, EngineResult, Solution, EngineConfig } from '../engine';
import { PRESETS, Preset, DEFAULT_CONSTS, presetById } from '../presets';
import { fmtInt, fmtShort, fmtRate, fmtMs, fmtEta, fmtConstIn, parseConst, escapeHtml, colorExpr } from './format';

interface UiState {
  spec: string;
  ops: Set<string>;
  consts: number[];
  maxLen: number;
  presetId: string | null;
}

interface PresetOutcome {
  status: 'ok' | 'none' | 'error' | 'running';
  L?: number;
  expr?: string;
  evaluated?: bigint;
  ms?: number;
  verified?: string;
  maxLen?: number;
}

const $ = <T extends HTMLElement>(sel: string, root: ParentNode = document): T => root.querySelector(sel) as T;
const el = (tag: string, cls?: string, html?: string): HTMLElement => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};

export class App {
  state: UiState = { spec: '', ops: new Set(), consts: [...DEFAULT_CONSTS], maxLen: 5, presetId: null };
  gpu: Gpu | null = null;
  running: AbortController | null = null;
  result: EngineResult | null = null;
  selected: Solution | null = null;
  lang: Lang | 'listing' = 'c';
  outcomes = new Map<string, PresetOutcome>();
  benchmarking = false;
  private lengthTimes = new Map<number, number>();
  private lengthStart = 0;
  private rate = 0;
  private compileNote = '';

  constructor(readonly root: HTMLElement) {}

  async init(): Promise<void> {
    this.root.innerHTML = TEMPLATE;
    this.loadOutcomes();
    this.bindControls();
    const fromUrl = this.readUrl();
    if (!fromUrl) this.applyPreset(presetById('p9')!);
    this.renderControls();
    this.renderGallery();
    this.refreshSpecInfo();
    this.renderResult();

    const badge = $('#gpu');
    if (!hasWebGpu()) {
      badge.classList.add('off');
      badge.querySelector('span')!.textContent = 'no WebGPU';
      $('#nowebgpu').hidden = false;
    } else {
      badge.querySelector('span')!.textContent = 'GPU…';
      this.gpu = await getGpu((reason) => this.setStatus(`GPU device lost: ${reason}`, true));
      if (this.gpu) {
        badge.querySelector('span')!.textContent = this.gpu.info.name;
        // warm the most common kernel while the user reads the page
        precompile(this.gpu, OPS.filter((o) => o.group === 'base'));
        this.runSelfTest();
      } else {
        badge.classList.add('off');
        badge.querySelector('span')!.textContent = 'WebGPU unavailable';
        $('#nowebgpu').hidden = false;
      }
    }
    // the page is the demo: run the loaded spec straight away
    this.search();
  }

  /** Check every op's GPU arithmetic against the CPU and cross-check one search; warn about broken ops. */
  private async runSelfTest(): Promise<void> {
    if (!this.gpu) return;
    try {
      const rep = await selfTest(this.gpu, false);
      const badge = $('#gpu');
      if (rep.ok) {
        badge.title = `GPU self-test passed: all ${OPS.length} ops agree with the CPU (${rep.ms.toFixed(0)} ms)`;
      } else {
        badge.classList.add('off');
        const list = rep.badOps.map((b) => `${b.op}(${fmtConstIn(b.a)}, ${fmtConstIn(b.b)}) = ${b.gpu} on the GPU, ${b.cpu} on the CPU`).join('; ');
        this.setStatus(`GPU self-test FAILED — this driver miscompiles: ${list}. Those ops were disabled; results that use them could not be trusted.`, true);
        for (const b of rep.badOps) this.state.ops.delete(b.op);
        this.renderControls();
        this.refreshSpecInfo();
      }
      $('#selftest').textContent = rep.ok ? `self-test: ${OPS.length}/${OPS.length} ops OK` : `self-test: ${rep.badOps.length} op(s) broken`;
      $('#selftest').className = 'spec-info ' + (rep.ok ? '' : 'err');
    } catch (e) {
      $('#selftest').textContent = 'self-test failed to run: ' + String((e as Error).message ?? e);
    }
  }

  // ------------------------------------------------------------ state helpers
  private applyPreset(p: Preset): void {
    this.state.spec = p.spec;
    const ids = p.ops ?? OPS.filter((o) => (p.groups ?? ['base']).includes(o.group)).map((o) => o.id);
    this.state.ops = new Set(ids);
    this.state.consts = p.consts ? [...p.consts] : [...DEFAULT_CONSTS];
    this.state.maxLen = p.maxLen ?? 5;
    this.state.presetId = p.id;
  }

  private opIds(): string[] {
    return OPS.filter((o) => this.state.ops.has(o.id)).map((o) => o.id);
  }

  private engineConfig(): EngineConfig {
    return { spec: this.state.spec, opIds: this.opIds(), consts: this.state.consts, maxLen: this.state.maxLen };
  }

  private writeUrl(): void {
    const s = this.state;
    const payload = { s: s.spec, o: this.opIds(), c: s.consts.map((c) => c >>> 0), l: s.maxLen, p: s.presetId ?? undefined };
    const json = JSON.stringify(payload);
    const b64 = btoa(unescape(encodeURIComponent(json))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    history.replaceState(null, '', '#' + b64);
  }

  private readUrl(): boolean {
    const h = location.hash.slice(1);
    if (!h) return false;
    try {
      const json = decodeURIComponent(escape(atob(h.replace(/-/g, '+').replace(/_/g, '/'))));
      const p = JSON.parse(json);
      if (typeof p.s !== 'string') return false;
      this.state.spec = p.s;
      this.state.ops = new Set((p.o as string[]).filter((id) => OP_BY_ID.has(id)));
      this.state.consts = (p.c as number[]).map((c) => c >>> 0).slice(0, 12);
      this.state.maxLen = Math.max(1, Math.min(8, p.l | 0));
      this.state.presetId = typeof p.p === 'string' && presetById(p.p) ? p.p : null;
      return true;
    } catch {
      return false;
    }
  }

  shareUrl(): string {
    this.writeUrl();
    return location.href;
  }

  // ------------------------------------------------------------ controls
  private bindControls(): void {
    const ta = $<HTMLTextAreaElement>('#spec');
    ta.addEventListener('input', () => {
      this.state.spec = ta.value;
      this.state.presetId = null;
      this.refreshSpecInfo();
      this.renderGallery();
    });
    ta.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); this.search(); }
    });
    $('#run').addEventListener('click', () => (this.running ? this.stop() : this.search()));
    $<HTMLSelectElement>('#len').addEventListener('change', (e) => {
      this.state.maxLen = Number((e.target as HTMLSelectElement).value);
      this.refreshSpecInfo();
    });
    $('#runall').addEventListener('click', () => (this.benchmarking ? this.stop() : this.benchmark()));
    $('#copymd').addEventListener('click', () => this.copyMarkdown());
    $('#share').addEventListener('click', () => this.copy(this.shareUrl(), 'link copied'));
    $('#copyexpr').addEventListener('click', () => { if (this.selected) this.copy(this.selected.expr, 'expression copied'); });
    $('#copycode').addEventListener('click', () => this.copy($('#code').textContent ?? '', 'code copied'));
    $('#copytable').addEventListener('click', () => this.copyTable());
    $('#clearres').addEventListener('click', () => { this.outcomes.clear(); this.saveOutcomes(); this.renderGallery(); });
    $('.tabs').addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (!b) return;
      this.lang = b.dataset.lang as Lang | 'listing';
      this.renderCode();
    });
  }

  private renderControls(): void {
    const s = this.state;
    $<HTMLTextAreaElement>('#spec').value = s.spec;
    $<HTMLSelectElement>('#len').value = String(s.maxLen);

    // op groups
    const gc = $('#groups');
    gc.innerHTML = '';
    for (const g of GROUPS) {
      const ids = OPS.filter((o) => o.group === g.id).map((o) => o.id);
      const on = ids.every((id) => s.ops.has(id));
      const some = ids.some((id) => s.ops.has(id));
      const c = el('span', 'chip' + (on ? ' on' : ''), `${g.label}${some && !on ? ' <small>partial</small>' : ''}`);
      c.title = g.hint;
      c.addEventListener('click', () => {
        for (const id of ids) on ? s.ops.delete(id) : s.ops.add(id);
        s.presetId = null;
        this.renderControls();
        this.refreshSpecInfo();
      });
      gc.appendChild(c);
    }
    // individual ops
    const oc = $('#ops');
    oc.innerHTML = '';
    for (const o of OPS) {
      const c = el('span', 'chip' + (s.ops.has(o.id) ? ' on' : ''), o.id);
      c.title = o.desc;
      c.addEventListener('click', () => {
        s.ops.has(o.id) ? s.ops.delete(o.id) : s.ops.add(o.id);
        s.presetId = null;
        this.renderControls();
        this.refreshSpecInfo();
      });
      oc.appendChild(c);
    }
    // constants
    const cc = $('#consts');
    cc.innerHTML = '';
    s.consts.forEach((v, i) => {
      const c = el('span', 'chip const', `${escapeHtml(fmtConstIn(v))}<span class="x" title="remove">×</span>`);
      c.querySelector('.x')!.addEventListener('click', () => {
        s.consts.splice(i, 1);
        s.presetId = null;
        this.renderControls();
        this.refreshSpecInfo();
      });
      cc.appendChild(c);
    });
    const add = el('span', 'chip add', `<input placeholder="add: 7, 0xff, -8" spellcheck="false" />`);
    const inp = add.querySelector('input')!;
    const commit = () => {
      const v = parseConst(inp.value);
      if (v === null) { if (inp.value.trim()) this.toast('not a 32-bit constant'); return; }
      if (s.consts.length >= 12) { this.toast('at most 12 constants'); return; }
      if (!s.consts.includes(v)) s.consts.push(v);
      s.presetId = null;
      this.renderControls();
      this.refreshSpecInfo();
      $<HTMLInputElement>('#consts input').focus();
    };
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); commit(); } });
    inp.addEventListener('blur', commit);
    cc.appendChild(add);
    $('#run').textContent = this.running ? 'Stop' : 'Search';
    $('#run').className = 'btn ' + (this.running ? 'danger' : 'primary');
  }

  /** Parse the spec and show input count + search-space sizes. */
  private refreshSpecInfo(): { cfg: SpaceConfig; ok: boolean } {
    const ta = $<HTMLTextAreaElement>('#spec');
    const err = $('#specerr');
    const info = $('#specinfo');
    try {
      const spec = compileSpec(this.state.spec);
      const ops = OPS.filter((o) => this.state.ops.has(o.id));
      if (!ops.length) throw new Error('enable at least one op');
      const cfg: SpaceConfig = { nInputs: spec.nInputs, consts: this.state.consts, ops };
      ta.classList.remove('bad');
      err.textContent = '';
      const space = new Space(cfg, this.state.maxLen);
      const parts: string[] = [];
      let total = 0n;
      for (let L = 1; L <= this.state.maxLen; L++) {
        const c = deadCodeFreeCount(space, L);
        total += c;
        parts.push(`<span title="${fmtInt(c)} dead-code-free programs of length ${L}">L${L} <b>${fmtShort(c)}</b></span>`);
      }
      const eta = this.rate > 0 ? ` · worst case ${fmtEta(Number(total) / this.rate)} at ${fmtRate(this.rate)}` : '';
      info.innerHTML = `${spec.nInputs} input${spec.nInputs > 1 ? 's' : ''} (${['x', 'y', 'z'].slice(0, spec.nInputs).join(', ')}) · ${ops.length} ops · ${cfg.consts.length} constants · programs to try: ${parts.join(' · ')}${eta}`;
      return { cfg, ok: true };
    } catch (e) {
      ta.classList.add('bad');
      const msg = e instanceof SpecError ? `${e.message} (at ${e.pos})` : String((e as Error).message ?? e);
      err.textContent = msg;
      info.innerHTML = '';
      return { cfg: { nInputs: 1, consts: [], ops: [] }, ok: false };
    }
  }

  // ------------------------------------------------------------ search
  async search(): Promise<void> {
    if (this.running) return;
    const { ok } = this.refreshSpecInfo();
    if (!ok) return;
    this.writeUrl();
    const ac = new AbortController();
    this.running = ac;
    this.result = null;
    this.selected = null;
    this.lengthTimes.clear();
    this.compileNote = '';
    this.renderControls();
    this.renderResult();
    this.resetLadder();
    this.setStatus(this.gpu ? 'starting…' : 'no WebGPU: searching on the CPU (short programs only)');
    const cfg = this.engineConfig();
    const presetId = this.state.presetId;
    if (presetId) { this.outcomes.set(presetId, { status: 'running' }); this.renderGallery(); }
    const t0 = performance.now();
    try {
      const res = await runEngine(cfg, this.gpu, {
        onStatus: (m) => { this.setStatus(m); if (m.startsWith('kernel compiled')) this.compileNote = m; },
        onLength: (e) => {
          this.lengthStart = performance.now();
          this.updateLadder(e.L, 0n, e.prefixTotal, 'active');
          this.setStatus(`length ${e.L}: ${e.backend === 'gpu' ? 'GPU' : 'CPU'} search over ${fmtInt(deadCodeFreeCount(new Space(this.spaceCfg(), e.L), e.L))} programs`);
        },
        onProgress: (p) => {
          if (p.rate > 0) this.rate = p.rate;
          this.updateLadder(p.L, p.prefixDone, p.prefixTotal, 'active');
          $('#st-eval').textContent = fmtShort(p.evaluated);
          $('#st-eval').title = fmtInt(p.evaluated) + ' programs';
          $('#st-rate').textContent = p.backend === 'gpu' ? fmtRate(p.rate) : 'CPU';
          $('#st-time').textContent = fmtMs(performance.now() - t0);
          $('#st-disp').textContent = p.backend === 'gpu' ? `${p.workgroupsPerDispatch} wg · ${p.msPerDispatch.toFixed(0)} ms` : '—';
          if (p.backend === 'gpu' && p.prefixTotal > 0n) {
            const frac = Number(p.prefixDone) / Number(p.prefixTotal);
            const elapsed = performance.now() - this.lengthStart;
            const eta = frac > 0.02 ? (elapsed / frac - elapsed) / 1000 : NaN;
            this.setStatus(`length ${p.L}: ${(frac * 100).toFixed(1)}% of prefixes · ${fmtRate(p.rate)}${isFinite(eta) ? ` · ${fmtEta(eta)} left` : ''}`);
          }
        },
        onLengthDone: (e) => {
          this.lengthTimes.set(e.L, e.elapsedMs);
          this.updateLadder(e.L, 1n, 1n, 'done', e.elapsedMs, e.evaluated);
        },
        onCandidate: (sol) => {
          this.updateLadder(sol.L, 1n, 1n, 'found');
          this.showCandidate(sol);
        },
        onVerifyProgress: (sol, done, total, pass) => this.showVerifyProgress(sol, done, total, pass),
        onVerified: (sol) => { this.renderResult(); void sol; },
        onError: (m) => this.setStatus(m, true),
      }, ac.signal);
      this.result = res;
      $('#st-time').textContent = fmtMs(performance.now() - t0);
      $('#st-eval').textContent = fmtShort(res.evaluated);
      $('#st-eval').title = fmtInt(res.evaluated) + ' programs';
      if (this.rate > 0 && $('#st-rate').textContent === '—') $('#st-rate').textContent = fmtRate(this.rate);
      if (res.error) this.setStatus(res.error, true);
      else if (res.aborted) this.setStatus('stopped');
      else if (res.solutions.length) {
        this.setStatus(`found ${res.solutions.length} program${res.solutions.length > 1 ? 's' : ''} of length ${res.L} in ${fmtMs(res.searchMs)}${this.compileNote ? ' (incl. ' + this.compileNote.replace('kernel compiled in ', '') + ' of one-time shader compile)' : ''} · verified in ${fmtMs(res.verifyMs)}`);
      } else this.setStatus(`no program of length ≤ ${cfg.maxLen} computes this with the chosen ops and constants (${fmtInt(res.evaluated)} tried)`);
      if (!res.solutions.length) for (let L = 1; L <= cfg.maxLen; L++) if (!this.lengthTimes.has(L)) this.updateLadder(L, 0n, 1n, 'skip');
      this.selected = res.solutions[0] ?? null;
      this.renderResult();
      if (presetId) {
        const sol = res.solutions[0];
        this.outcomes.set(presetId, res.error
          ? { status: 'error' }
          : sol
            ? { status: 'ok', L: res.L, expr: sol.expr, evaluated: res.evaluated, ms: res.searchMs, verified: verifyLabel(sol), maxLen: cfg.maxLen }
            : { status: res.aborted ? 'error' : 'none', evaluated: res.evaluated, ms: res.searchMs, maxLen: cfg.maxLen });
        this.saveOutcomes();
        this.renderGallery();
      }
    } catch (e) {
      this.setStatus(String((e as Error).message ?? e), true);
      if (presetId) { this.outcomes.set(presetId, { status: 'error' }); this.renderGallery(); }
    } finally {
      this.running = null;
      this.renderControls();
    }
  }

  stop(): void {
    this.running?.abort();
    this.benchmarking = false;
    $('#runall').textContent = 'Run all (benchmark)';
  }

  private spaceCfg(): SpaceConfig {
    const spec = compileSpec(this.state.spec);
    return { nInputs: spec.nInputs, consts: this.state.consts, ops: OPS.filter((o) => this.state.ops.has(o.id)) };
  }

  // ------------------------------------------------------------ ladder + stats
  private resetLadder(): void {
    const lad = $('#ladder');
    lad.innerHTML = '';
    let cfg: SpaceConfig | null = null;
    try { cfg = this.spaceCfg(); } catch { /* spec errors are shown elsewhere */ }
    const space = cfg ? new Space(cfg, this.state.maxLen) : null;
    for (let L = 1; L <= this.state.maxLen; L++) {
      const n = space ? deadCodeFreeCount(space, L) : 0n;
      const row = el('div', 'lrow', '');
      row.style.display = 'contents';
      row.dataset.l = String(L);
      row.innerHTML = `<span class="L">L${L}</span><span class="bar"><i></i></span><span class="cnt num" title="${fmtInt(n)} programs">${fmtShort(n)}</span><span class="t num"></span>`;
      lad.appendChild(row);
    }
    $('#st-eval').textContent = '0';
    $('#st-rate').textContent = '—';
    $('#st-time').textContent = '0 ms';
    $('#st-disp').textContent = '—';
  }

  private updateLadder(L: number, done: bigint, total: bigint, state: 'active' | 'done' | 'found' | 'skip', ms?: number, evaluated?: bigint): void {
    const row = $(`#ladder .lrow[data-l="${L}"]`);
    if (!row) return;
    const cur = row.className;
    if (cur.includes('row-found') && state !== 'found') return;
    row.className = 'lrow row-' + state;
    const bar = row.querySelector('.bar i') as HTMLElement;
    const frac = total > 0n ? Number(done) / Number(total) : 0;
    bar.style.width = `${Math.max(state === 'done' || state === 'found' ? 100 : frac * 100, state === 'active' ? 1.5 : 0)}%`;
    if (ms !== undefined) (row.querySelector('.t') as HTMLElement).textContent = fmtMs(ms);
    if (evaluated !== undefined) { const c = row.querySelector('.cnt') as HTMLElement; c.textContent = fmtShort(evaluated); c.title = fmtInt(evaluated) + ' programs evaluated'; }
    if (state === 'skip') (row.querySelector('.t') as HTMLElement).textContent = 'not reached';
  }

  private setStatus(msg: string, err = false): void {
    const s = $('#status');
    s.textContent = msg;
    s.className = 'status-line' + (err ? ' err' : '');
  }

  // ------------------------------------------------------------ result card
  private showCandidate(sol: Solution): void {
    if (!this.selected) this.selected = sol;
    this.renderResult(true);
  }

  private showVerifyProgress(sol: Solution, done: bigint, total: bigint, pass: string): void {
    if (sol !== this.selected) return;
    const p = $('#proof');
    p.className = 'proof run';
    const frac = total > 0n ? Number(done) / Number(total) : 0;
    p.innerHTML = `<span class="ico">⏳</span><div style="flex:1"><b>verifying on the GPU</b> — ${escapeHtml(pass)}<div class="sub num">${fmtInt(done)} / ${fmtInt(total)} inputs</div><div class="pbar"><i style="width:${(frac * 100).toFixed(1)}%"></i></div></div>`;
  }

  private renderResult(candidateOnly = false): void {
    const card = $('#result');
    const sol = this.selected;
    if (!sol) {
      card.hidden = true;
      $('#noresult').hidden = false;
      $('#noresult').textContent = this.running ? 'searching…' : this.result ? (this.result.error ? 'search failed' : 'nothing found in range — enable more ops, add a constant, or raise the length') : '';
      return;
    }
    $('#noresult').hidden = true;
    card.hidden = false;
    if (candidateOnly) card.classList.add('pop');
    const res = this.result;
    $('#r-len').innerHTML = `${sol.L}<small>instruction${sol.L > 1 ? 's' : ''}</small>`;
    $('#r-expr').innerHTML = colorExpr(sol.expr);
    // minimality
    const cfg = res?.cfg ?? this.safeCfg();
    const min = $('#minimal');
    if (cfg && sol.L > 1) {
      const space = new Space(cfg, sol.L);
      let below = 0n;
      for (let L = 1; L < sol.L; L++) below += deadCodeFreeCount(space, L);
      min.innerHTML = `<b>shortest possible</b> with these ${cfg.ops.length} ops and ${cfg.consts.length} constants: all <b class="num">${fmtInt(below)}</b> programs of length &lt; ${sol.L} were tried and none match.`;
    } else if (cfg) {
      min.innerHTML = `a single instruction — nothing to beat.`;
    }
    // proof
    const p = $('#proof');
    const v = sol.verify;
    if (!v) {
      if (this.running) { p.className = 'proof run'; p.innerHTML = `<span class="ico">⏳</span><div><b>candidate</b> — matches the spec on all samples; full verification pending…</div>`; }
      else { p.className = 'proof'; p.innerHTML = `<span class="ico">·</span><div><b>matches the spec on all samples</b><div class="sub">WebGPU is needed for the full verification pass.</div></div>`; }
    } else if (v.mismatches > 0) {
      p.className = 'proof bad';
      p.innerHTML = `<span class="ico">✗</span><div><b>rejected</b> — differs from the spec at ${v.counterexamples.map((c) => 'x=' + fmtConstIn(c[0]) + (c.length > 1 ? ', y=' + fmtConstIn(c[1]) : '') + (c.length > 2 ? ', z=' + fmtConstIn(c[2]) : '')).slice(0, 3).join('; ')}<div class="sub">the search learned this counterexample and continued</div></div>`;
    } else if (v.mode === 'exhaustive') {
      p.className = 'proof ok';
      p.innerHTML = `<span class="ico">✓</span><div><b>proved</b> — identical to the spec on all <span class="num">${fmtInt(v.checked)}</span> possible inputs (every 32-bit x), checked on the ${v.backend === 'cpu' ? 'CPU (the GPU failed its spot check)' : 'GPU'} in ${fmtMs(v.elapsedMs)}.</div>`;
    } else {
      p.className = 'proof ok';
      p.innerHTML = `<span class="ico">✓</span><div><b>verified</b> on <span class="num">${fmtInt(v.checked)}</span> inputs in ${fmtMs(v.elapsedMs)} — ${v.passes.map((x) => escapeHtml(x.name)).join(', ')}.<div class="sub">not a proof: with ${v.nInputs} inputs there are 2<sup>${32 * v.nInputs}</sup> cases, too many to enumerate.</div></div>`;
    }
    this.renderCode();
    // alternatives
    const alts = $('#alts');
    alts.innerHTML = '';
    const list = res?.solutions ?? [];
    $('#altwrap').hidden = list.length <= 1;
    for (const s of list) {
      const a = el('div', 'alt' + (s === sol ? ' on' : ''), `<span>${colorExpr(s.expr)}</span><span class="st ${s.verify ? (s.verify.mismatches ? 'bad' : 'ok') : ''}">${verifyLabel(s)}</span>`);
      a.addEventListener('click', () => { this.selected = s; this.renderResult(); });
      alts.appendChild(a);
    }
  }

  private safeCfg(): SpaceConfig | null {
    try { return this.spaceCfg(); } catch { return null; }
  }

  private renderCode(): void {
    const sol = this.selected;
    const cfg = this.result?.cfg ?? this.safeCfg();
    if (!sol || !cfg) return;
    for (const b of Array.from($('.tabs').querySelectorAll('button'))) b.classList.toggle('on', b.dataset.lang === this.lang);
    const code = this.lang === 'listing' ? listing(cfg, sol.program) : emitFunction(cfg, sol.program, this.lang);
    $('#code').textContent = code;
  }

  // ------------------------------------------------------------ gallery / benchmark
  private renderGallery(): void {
    const tb = $('#gallery tbody');
    tb.innerHTML = '';
    let lastGroup = '';
    for (const p of PRESETS) {
      if (p.group !== lastGroup) {
        lastGroup = p.group;
        tb.appendChild(el('tr', 'grp', `<td colspan="5">${p.group === 'hd' ? "Hacker's Delight benchmark (Gulwani et al., PLDI 2011)" : 'more classics'}</td>`));
      }
      const o = this.outcomes.get(p.id);
      let res = '<span class="no">—</span>';
      let numc = '';
      if (o?.status === 'running') res = '<span class="run blink">searching…</span>';
      else if (o?.status === 'ok') { res = `<span class="ok">${o.L} instr</span> <span title="${escapeHtml(o.expr ?? '')}">${escapeHtml(o.expr ?? '')}</span>`; numc = `${fmtShort(o.evaluated ?? 0n)} · ${fmtMs(o.ms ?? 0)}${o.verified ? ' · ' + o.verified : ''}`; }
      else if (o?.status === 'none') { res = `<span class="no">none ≤ ${o.maxLen}</span>`; numc = `${fmtShort(o.evaluated ?? 0n)} · ${fmtMs(o.ms ?? 0)}`; }
      else if (o?.status === 'error') res = '<span class="bad">error</span>';
      const tr = el('tr', 'p' + (p.id === this.state.presetId ? ' sel' : ''), `<td class="name">${escapeHtml(p.name)}${p.hard ? ' <span class="no" title="textbook solution is long; likely beyond the horizon in this ISA">†</span>' : ''}</td><td class="spec" title="${escapeHtml(p.spec)}">${escapeHtml(p.spec.replace(/\s+/g, ' '))}</td><td class="num">${p.size ?? ''}</td><td class="res">${res}</td><td class="num">${numc}</td>`);
      tr.addEventListener('click', () => {
        if (this.running) return;
        this.applyPreset(p);
        this.renderControls();
        this.refreshSpecInfo();
        this.renderGallery();
        this.search();
      });
      tb.appendChild(tr);
    }
  }

  async benchmark(): Promise<void> {
    if (this.running || this.benchmarking) return;
    this.benchmarking = true;
    $('#runall').textContent = 'Stop benchmark';
    for (const p of PRESETS) {
      if (!this.benchmarking) break;
      this.applyPreset(p);
      if (p.hard) this.state.maxLen = Math.min(this.state.maxLen, 4);
      this.renderControls();
      this.refreshSpecInfo();
      this.renderGallery();
      await this.search();
      await new Promise((r) => setTimeout(r, 50));
    }
    this.benchmarking = false;
    $('#runall').textContent = 'Run all (benchmark)';
    this.renderGallery();
  }

  private saveOutcomes(): void {
    try {
      const obj: Record<string, unknown> = {};
      for (const [k, v] of this.outcomes) if (v.status !== 'running') obj[k] = { ...v, evaluated: v.evaluated?.toString() };
      localStorage.setItem('xorcery.outcomes', JSON.stringify(obj));
    } catch { /* storage may be unavailable */ }
  }

  private loadOutcomes(): void {
    try {
      const raw = localStorage.getItem('xorcery.outcomes');
      if (!raw) return;
      const obj = JSON.parse(raw) as Record<string, any>;
      for (const k of Object.keys(obj)) this.outcomes.set(k, { ...obj[k], evaluated: obj[k].evaluated ? BigInt(obj[k].evaluated) : undefined });
    } catch { /* ignore */ }
  }

  // ------------------------------------------------------------ clipboard
  private copy(text: string, msg: string): void {
    navigator.clipboard?.writeText(text).then(() => this.toast(msg), () => this.toast('copy failed'));
  }

  private toast(msg: string): void {
    const t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1600);
  }

  private copyMarkdown(): void {
    const res = this.result, sol = this.selected;
    if (!res || !sol) { this.toast('nothing to copy yet'); return; }
    const gpuName = this.gpu?.info.name ?? 'CPU';
    const lines = [
      `**spec** \`${this.state.spec.replace(/\s+/g, ' ')}\``,
      `**shortest program (${sol.L} instruction${sol.L > 1 ? 's' : ''})** \`${sol.expr}\``,
      `${verifyLabel(sol, true)} · searched ${fmtInt(res.evaluated)} programs in ${fmtMs(res.searchMs)} on ${gpuName}`,
      `found with xorcery — ${this.shareUrl()}`,
    ];
    this.copy(lines.join('\n'), 'markdown copied');
  }

  private copyTable(): void {
    const gpuName = this.gpu?.info.name ?? 'CPU';
    const rows = ['| preset | textbook | found | program | programs tried | time | check |', '|---|---:|---:|---|---:|---:|---|'];
    for (const p of PRESETS) {
      const o = this.outcomes.get(p.id);
      if (!o || o.status === 'running') continue;
      const found = o.status === 'ok' ? String(o.L) : o.status === 'none' ? `none ≤ ${o.maxLen}` : 'error';
      rows.push(`| ${p.name} | ${p.size ?? ''} | ${found} | ${o.expr ? '`' + o.expr + '`' : ''} | ${o.evaluated !== undefined ? fmtInt(o.evaluated) : ''} | ${o.ms !== undefined ? fmtMs(o.ms) : ''} | ${o.verified ?? ''} |`);
    }
    rows.push('', `xorcery on ${gpuName} — ${location.origin}${location.pathname}`);
    this.copy(rows.join('\n'), 'table copied as markdown');
  }
}

function verifyLabel(s: Solution, long = false): string {
  const v = s.verify;
  if (!v) return long ? 'matches all samples (no GPU verification)' : 'samples';
  if (v.mismatches) return long ? `rejected (counterexample x=${fmtConstIn(v.counterexamples[0]?.[0] ?? 0)})` : 'rejected';
  const cpu = v.backend === 'cpu' ? ' (CPU)' : '';
  if (v.mode === 'exhaustive') return long ? `proved on all ${fmtInt(v.checked)} inputs${cpu}` : 'proved 2^32' + cpu;
  return long ? `verified on ${fmtInt(v.checked)} inputs (not exhaustive)${cpu}` : `verified ${fmtShort(v.checked)}${cpu}`;
}

const TEMPLATE = `
<header class="top">
  <h1><span>xor</span>cery</h1>
  <div class="tag">your GPU rediscovers bit hacks</div>
  <div class="right">
    <span class="gpu-badge" id="gpu"><i></i><span>…</span></span>
    <span id="selftest" class="spec-info"></span>
    <a href="https://github.com/zaydmulani09/xorcery" target="_blank" rel="noopener">source</a>
  </div>
</header>
<p class="intro">Write what a function of 32-bit words should compute. Your GPU tries <em>every</em> straight-line program over the ops you allow, shortest first, hands you the winner as C/Rust/JS/WGSL — and proves it on all 4,294,967,296 inputs.</p>
<div id="nowebgpu" class="nowebgpu" hidden>
  <b>This browser has no WebGPU.</b> xorcery searches billions of programs per second on your GPU; without it the CPU fallback only reaches length 3.
  Chrome 113+, Edge 113+, Safari 26 and Firefox 141+ (Windows) ship WebGPU.
</div>
<div class="grid">
  <div class="stack">
    <section class="panel stack">
      <h2>spec <span class="hint">C syntax · 32-bit words · <span class="kbd">Ctrl</span>+<span class="kbd">Enter</span> to search</span></h2>
      <textarea id="spec" class="spec" spellcheck="false" autocapitalize="off"></textarea>
      <div id="specerr" class="spec-error"></div>
      <div id="specinfo" class="spec-info"></div>
      <div class="row"><span class="label">ops</span><div id="groups" class="chips"></div></div>
      <details class="ops"><summary>individual ops</summary><div id="ops" class="chips"></div></details>
      <div class="row"><span class="label">consts</span><div id="consts" class="chips"></div></div>
      <div class="row"><span class="label">max len</span>
        <select id="len" class="len"><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option></select>
        <span class="spec-info">iterative deepening: the first hit is the shortest</span>
      </div>
      <div class="actions"><button id="run" class="btn primary">Search</button><button id="share" class="btn">Share link</button></div>
    </section>
    <section class="panel stack">
      <h2>search <span class="hint">dead-code-free straight-line programs</span></h2>
      <div class="stats">
        <div class="stat"><div class="k">programs tried</div><div class="v num" id="st-eval">0</div></div>
        <div class="stat"><div class="k">throughput</div><div class="v num" id="st-rate">—</div></div>
        <div class="stat"><div class="k">elapsed</div><div class="v num" id="st-time">0 ms</div></div>
        <div class="stat"><div class="k">dispatch</div><div class="v num" id="st-disp" style="font-size:14px;line-height:2">—</div></div>
      </div>
      <div id="ladder" class="ladder"></div>
      <div id="status" class="status-line"></div>
    </section>
  </div>
  <div class="stack">
    <section class="panel stack result" id="resultpanel">
      <h2>result</h2>
      <div id="noresult" class="empty"></div>
      <div id="result" class="stack" hidden>
        <div class="headline"><div class="len" id="r-len"></div></div>
        <div class="expr" id="r-expr"></div>
        <div class="proof" id="proof"></div>
        <div class="minimal" id="minimal"></div>
        <div id="altwrap" hidden><h2>alternatives at the same length</h2><div id="alts" class="alts"></div></div>
        <div class="codewrap">
          <div class="tabs"><button data-lang="c" class="on">C</button><button data-lang="rust">Rust</button><button data-lang="js">JavaScript</button><button data-lang="wgsl">WGSL</button><button data-lang="listing">listing</button></div>
          <pre class="code" id="code"></pre>
          <button id="copycode" class="btn small ghost copy">copy</button>
        </div>
        <div class="actions"><button id="copyexpr" class="btn small">copy expression</button><button id="copymd" class="btn small">copy result as markdown</button></div>
      </div>
    </section>
  </div>
</div>
<section class="panel gallery">
  <div class="head"><h2>gallery</h2><span class="spec-info">click a row to load and search it · † textbook answer is long, likely beyond the horizon in this ISA</span><span class="sp"></span>
    <button id="runall" class="btn small">Run all (benchmark)</button><button id="copytable" class="btn small">copy table</button><button id="clearres" class="btn small ghost">clear</button></div>
  <div class="tablewrap"><table class="gal" id="gallery"><thead><tr><th>function</th><th>spec</th><th>textbook</th><th>found</th><th>search</th></tr></thead><tbody></tbody></table></div>
</section>
<section class="about">
  <h2>what this is</h2>
  <p>xorcery is a <b>superoptimizer</b>: you write what a function should compute, and it finds the <em>shortest</em> straight-line program that computes it — no branches, no loops, just the ops you allow. It does this the honest way, by trying every program. Your GPU evaluates a few billion candidates per second, iterative deepening guarantees the first hit is the shortest, and for one-input functions the answer is then <b>proved</b> by evaluating all 4,294,967,296 inputs.</p>
  <p>The specs are C expressions over 32-bit words (<code>(int)x &gt;&gt; 31</code> is an arithmetic shift, <code>x &gt;&gt; 31</code> a logical one; <code>popcnt</code>, <code>clz</code>, <code>ctz</code>, <code>mulhi</code>, <code>rotl</code>, <code>select</code> and friends are built in). The classic answers in the gallery come from Henry Warren's <em>Hacker's Delight</em>; the first 25 are the standard synthesis benchmark from Gulwani et al. (PLDI 2011).</p>
  <p>Read the <a href="https://github.com/zaydmulani09/xorcery#readme" target="_blank" rel="noopener">README</a> for how the search kernel works, the pruning rules, the honest limits (2^64 pairs cannot be enumerated), and benchmark numbers.</p>
</section>
<footer>
  <span>xorcery · <a href="https://github.com/zaydmulani09/xorcery">github.com/zaydmulani09/xorcery</a></span>
  <span>by <a href="https://zayd.dpdns.org">Zayd Mulani</a></span>
  <span>MIT · zero runtime dependencies · everything runs in your browser</span>
</footer>
<div id="toast" class="toast"></div>
`;

