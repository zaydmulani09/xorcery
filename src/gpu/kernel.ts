/**
 * WGSL code generation for the search kernel.
 *
 * One workgroup owns one *prefix* (the first S = L-2 instructions). It
 * evaluates the prefix once, cooperatively, into workgroup memory: thread s
 * computes the whole prefix chain for sample s. Then every thread takes a
 * share of the *middle* slot's candidates; for each one it computes the middle
 * result vector into registers and runs the *last* slot's inner loop, which is
 * unrolled per op by this generator so the hot loop has no switch and no table
 * lookups: just operand loads, one ALU op and a compare, with early exit on
 * the first mismatching sample.
 *
 * Dead code is pruned structurally rather than checked after the fact. The
 * last instruction must read the middle result (otherwise the middle
 * instruction is dead); prefix results that no prefix instruction reads must
 * be read by the middle or the last instruction, and there are only three
 * operand slots left to do it with. Together these rules make the kernel
 * evaluate exactly the dead-code-free programs of length L — the CPU
 * enumerator in `core/space.ts` is the reference for that claim and the tests
 * compare the two counts.
 *
 * The kernel depends only on the list of enabled ops. Program length, input
 * count, constants and the per-slot digit layout all arrive through buffers,
 * so one (slow, ~2 s on some drivers) shader compile serves every length of a
 * search and every preset that uses the same ISA.
 */
import { HELPERS, HELPER_DEPS, Op } from '../core/isa';
import { PAIR_KINDS, PairKind, Space, pairKind } from '../core/space';

export const NS = 32; // samples per search
export const WG = 256; // workgroup size
export const MAX_RESULTS = 64; // result entries
export const ENTRY_WORDS = 8; // u32 per result entry (one packed instruction per slot, L <= 8)
export const MAX_LEN = 8;
export const MAX_VALUES = 3 + 12 + (MAX_LEN - 2); // inputs + consts + prefix results held in workgroup memory
export const LAYOUT_STRIDE = 4; // u32 per (slot, op) layout entry: base, count, pairOffset, pad

/** Uniform block layout (u32 indices). */
export const P_START = 0; // 8 words: prefix start digits
export const P_RADIX = 8; // 8 words: prefix slot radices
export const P_LIMIT = 16;
export const P_S = 17;
export const P_MID = 18;
export const P_FIRSTR = 19;
export const P_NINPUTS = 20;
export const P_NCONSTS = 21;
export const P_MIDCOUNT = 22;
export const P_NOPS = 23;
export const PARAMS_WORDS = 24;

export interface SearchLayout {
  /** flat pair tables */
  pairs: Uint32Array<ArrayBuffer>;
  /** per (slot, op) entries: base, count, pairOffset, 0 */
  layout: Uint32Array<ArrayBuffer>;
}

/** Build the digit layout tables for every slot < maxLen. */
export function buildLayout(space: Space, maxLen: number): SearchLayout {
  const cfg = space.cfg;
  const tables: number[] = [];
  const off: Record<number, Record<PairKind, number>> = {};
  for (let slot = 0; slot < maxLen; slot++) {
    off[slot] = {} as Record<PairKind, number>;
    for (const kind of PAIR_KINDS) {
      const t = space.layouts[slot].tables[kind];
      off[slot][kind] = tables.length;
      for (const v of t) tables.push(v);
    }
  }
  const nops = cfg.ops.length;
  const layout = new Uint32Array(new ArrayBuffer(Math.max(16, maxLen * nops * LAYOUT_STRIDE * 4)));
  for (let slot = 0; slot < maxLen; slot++) {
    const lay = space.layouts[slot];
    for (let oi = 0; oi < nops; oi++) {
      const e = (slot * nops + oi) * LAYOUT_STRIDE;
      layout[e] = lay.opBase[oi];
      layout[e + 1] = lay.opCount[oi];
      layout[e + 2] = off[slot][pairKind(cfg.ops[oi])];
    }
  }
  const pairs = new Uint32Array(new ArrayBuffer(Math.max(16, tables.length * 4)));
  pairs.set(tables);
  return { pairs, layout };
}

function wgslExpr(op: Op, a: string, b: string): string {
  return op.wgsl.t.replace('$a', a).replace('$b', b);
}

function helperSource(ops: Op[]): string {
  const need = new Set<string>();
  for (const op of ops) if (op.wgsl.helper) need.add(op.wgsl.helper);
  for (const h of Array.from(need)) for (const d of HELPER_DEPS[h] ?? []) need.add(d);
  return Object.keys(HELPERS.wgsl).filter((h) => need.has(h)).map((h) => HELPERS.wgsl[h]).join('\n');
}

/** Cache key: the kernel text depends only on the op list. */
export function kernelKey(ops: Op[]): string {
  return ops.map((o) => o.id).join(',');
}

/**
 * Generate the search kernel for a list of ops. Everything else is dynamic.
 */
export function generateKernel(ops: Op[]): string {
  const NOPS = ops.length;

  // ---- generic apply (prefix and middle evaluation) ----------------------
  const applyLines = ['fn apply(op: u32, a: u32, b: u32) -> u32 {', '  switch (op) {'];
  ops.forEach((op, i) => applyLines.push(`    case ${i}u: { return ${wgslExpr(op, 'a', 'b')}; }`));
  applyLines.push('    default: { return 0u; }', '  }', '}');

  // ---- inner loop (last slot), unrolled per op ---------------------------
  const sampleLoop = (opExpr: string, lop: number, la: string, lb: string): string => `
        {
          var ok = true;
          for (var s = 0u; s < ${NS}u; s++) {
            if ((${opExpr}) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, ${lop}u, ${la}, ${lb}); }
        }`;

  const inner: string[] = [];
  ops.forEach((op, oi) => {
    const A = 'mv[s]';
    const B = (v: string) => `pv[${v} * ${NS}u + s]`;
    const block: string[] = [];
    block.push(`      // ---- ${op.id} ----`);
    if (op.kind === 'unary') {
      block.push(`      if (rem == NONE) {`);
      block.push(sampleLoop(wgslExpr(op, A, '0u'), oi, 'mid', '0u'));
      block.push(`      }`);
    } else if (op.kind === 'comm') {
      block.push(`      if (rem == NONE) {`);
      block.push(`        for (var b = 0u; b < mid; b++) {`);
      block.push(sampleLoop(wgslExpr(op, A, B('b')), oi, 'mid', 'b'));
      block.push(`        }`);
      if (op.self) block.push(sampleLoop(wgslExpr(op, A, A), oi, 'mid', 'mid'));
      block.push(`      } else {`);
      block.push(sampleLoop(wgslExpr(op, A, B('rem')), oi, 'mid', 'rem'));
      block.push(`      }`);
    } else {
      block.push(`      if (rem == NONE) {`);
      block.push(`        for (var b = 0u; b < mid; b++) {`);
      block.push(sampleLoop(wgslExpr(op, A, B('b')), oi, 'mid', 'b'));
      block.push(`        }`);
      block.push(`        for (var a = 0u; a < mid; a++) {`);
      block.push(sampleLoop(wgslExpr(op, B('a'), A), oi, 'a', 'mid'));
      block.push(`        }`);
      if (op.self) block.push(sampleLoop(wgslExpr(op, A, A), oi, 'mid', 'mid'));
      block.push(`      } else {`);
      block.push(sampleLoop(wgslExpr(op, A, B('rem')), oi, 'mid', 'rem'));
      block.push(sampleLoop(wgslExpr(op, B('rem'), A), oi, 'rem', 'mid'));
      block.push(`      }`);
    }
    inner.push(block.join('\n'));
  });

  const unaryCases = ops.map((op, i) => (op.kind === 'unary' ? `    case ${i}u: { return true; }` : '')).filter(Boolean).join('\n');

  return `// xorcery search kernel — ops=[${ops.map((o) => o.id).join(',')}]
const NS: u32 = ${NS}u;
const NONE: u32 = 0xffffffffu;
const NOPS: u32 = ${NOPS}u;

struct Params {
  start: array<vec4<u32>, 2>,   // prefix start digits d[0..S) (odometer origin)
  radix: array<vec4<u32>, 2>,   // slot counts for prefix slots
  limit: u32,                   // workgroups that carry a valid prefix in this dispatch
  S: u32,                       // prefix length (L - 2)
  mid: u32,                     // value index of the middle result (= nInputs + nConsts + S)
  firstR: u32,                  // value index of r0 (= nInputs + nConsts)
  nInputs: u32,
  nConsts: u32,
  midCount: u32,                // digits in the middle slot
  nops: u32,
}
struct Results {
  count: atomic<u32>,
  evaluated: atomic<u32>,
  pad0: u32,
  pad1: u32,
  entries: array<u32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: Results;
@group(0) @binding(3) var<storage, read> data: array<u32>;    // inputs [n*NS], targets [NS], consts [k]
@group(0) @binding(4) var<storage, read> slotmap: array<u32>;  // per (slot, op): base, count, pairOffset, 0

var<workgroup> pv: array<u32, ${MAX_VALUES * NS}>;
var<workgroup> tg: array<u32, ${NS}>;

${helperSource(ops)}
${applyLines.join('\n')}

fn isUnary(op: u32) -> bool {
  switch (op) {
${unaryCases}
    default: { return false; }
  }
}

// digit -> (op, a, b) for a slot, using the layout table
fn decode(slot: u32, d: u32) -> vec3<u32> {
  var e = slot * NOPS * ${LAYOUT_STRIDE}u;
  for (var o = 0u; o < NOPS; o++) {
    let base = slotmap[e];
    let cnt = slotmap[e + 1u];
    if (d < base + cnt) {
      let p = pairs[slotmap[e + 2u] + (d - base)];
      return vec3<u32>(o, p >> 8u, p & 0xffu);
    }
    e += ${LAYOUT_STRIDE}u;
  }
  let p = pairs[0u];
  return vec3<u32>(0u, p >> 8u, p & 0xffu);
}

fn resultBit(v: u32) -> u32 {
  return select(0u, 1u << (v - params.firstR), v >= params.firstR);
}

fn pack(op: u32, a: u32, b: u32) -> u32 { return (op << 16u) | (a << 8u) | b; }

fn claim(pp: ptr<function, array<u32, ${MAX_LEN}>>, S: u32, mop: u32, ma: u32, mb: u32, lop: u32, la: u32, lb: u32) {
  let slot = atomicAdd(&results.count, 1u);
  if (slot < ${MAX_RESULTS}u) {
    let base = slot * ${ENTRY_WORDS}u;
    for (var i = 0u; i < S; i++) { results.entries[base + i] = (*pp)[i]; }
    results.entries[base + S] = pack(mop, ma, mb);
    results.entries[base + S + 1u] = pack(lop, la, lb);
  }
}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let wgi = wg.x;
  let S = params.S;
  let mid = params.mid;
  let n = params.nInputs;
  let k = params.nConsts;

  // ---- prefix digits: odometer add of the workgroup index -----------------
  var d: array<u32, ${MAX_LEN}>;
  var pop: array<u32, ${MAX_LEN}>;
  var pa: array<u32, ${MAX_LEN}>;
  var pb: array<u32, ${MAX_LEN}>;
  var pp: array<u32, ${MAX_LEN}>;
  var carry = wgi;
  for (var i = i32(S) - 1; i >= 0; i--) {
    let ui = u32(i);
    let r = params.radix[ui >> 2u][ui & 3u];
    let v = params.start[ui >> 2u][ui & 3u] + carry;
    d[ui] = v % r;
    carry = v / r;
  }
  var skip = wgi >= params.limit;

  // ---- decode the prefix and find results it leaves unread ----------------
  var used = 0u;
  for (var i = 0u; i < S; i++) {
    let ins = decode(i, d[i]);
    pop[i] = ins.x; pa[i] = ins.y; pb[i] = ins.z;
    pp[i] = pack(ins.x, ins.y, ins.z);
    used |= resultBit(ins.y);
    if (!isUnary(ins.x)) { used |= resultBit(ins.z); }
  }
  let needed = ((1u << S) - 1u) & ~used;
  // the middle instruction can absorb two unread results and the last one more
  if (countOneBits(needed) > 3u) { skip = true; }

  // ---- cooperative prefix evaluation: thread s computes sample s ----------
  if (lid < ${NS}u && !skip) {
    let s = lid;
    for (var i = 0u; i < n; i++) { pv[i * ${NS}u + s] = data[i * ${NS}u + s]; }
    for (var i = 0u; i < k; i++) { pv[(n + i) * ${NS}u + s] = data[n * ${NS}u + ${NS}u + i]; }
    tg[s] = data[n * ${NS}u + s];
    for (var i = 0u; i < S; i++) {
      let a = pv[pa[i] * ${NS}u + s];
      let b = pv[pb[i] * ${NS}u + s];
      pv[(params.firstR + i) * ${NS}u + s] = apply(pop[i], a, b);
    }
  }
  workgroupBarrier();
  if (skip) { return; }

  var evaluated = 0u;
  // ---- middle slot: each thread takes a share of the candidates -----------
  for (var m = lid; m < params.midCount; m += ${WG}u) {
    let mins = decode(S, m);
    let mop = mins.x; let ma = mins.y; let mb = mins.z;
    var needed2 = needed & ~resultBit(ma);
    if (!isUnary(mop)) { needed2 &= ~resultBit(mb); }
    if (countOneBits(needed2) > 1u) { continue; }
    let rem = select(NONE, params.firstR + countTrailingZeros(needed2), needed2 != 0u);
    var mv: array<u32, ${NS}>;
    for (var s = 0u; s < ${NS}u; s++) {
      mv[s] = apply(mop, pv[ma * ${NS}u + s], pv[mb * ${NS}u + s]);
    }
    // ---- last slot: unrolled per op; must read the middle result ----------
${inner.join('\n')}
  }
  atomicAdd(&results.evaluated, evaluated);
}
`;
}

/** Pack the search data buffer: inputs, targets, constants. */
export function packSearchData(nInputs: number, inputs: Uint32Array[], targets: Uint32Array, consts: number[]): Uint32Array<ArrayBuffer> {
  const out = new Uint32Array(new ArrayBuffer((nInputs * NS + NS + Math.max(1, consts.length)) * 4));
  for (let i = 0; i < nInputs; i++) out.set(inputs[i], i * NS);
  out.set(targets, nInputs * NS);
  consts.forEach((c, i) => { out[nInputs * NS + NS + i] = c >>> 0; });
  return out;
}
