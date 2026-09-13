/**
 * The program space: how straight-line programs are laid out, counted,
 * enumerated and decoded.
 *
 * A program of length L is a list of L instructions. Instruction i reads its
 * operands from the *value space* available at slot i:
 *
 *     index        0 .. n-1        the inputs (x, y, z)
 *     index        n .. n+k-1      the constant pool
 *     index    n+k .. n+k+i-1      the results r0 .. r(i-1) of earlier slots
 *
 * The value of the last instruction is the program's output.
 *
 * Each slot has a *digit*: an integer in [0, slotCount(i)) that identifies
 * (op, a, b). Digits are op-major, and inside an op they index a *pair table*
 * (one table per slot per op-kind) which lists the operand pairs that are
 * worth trying — see `pairTable()` for the pruning rules. A whole program is
 * the mixed-radix number formed by its digits, with the *last* slot as the
 * least-significant digit so that neighbouring program ids share a prefix.
 *
 * The GPU kernel (`src/gpu/kernel.ts`) decodes digits with exactly these
 * tables, which is why the tables are plain data rather than closed-form
 * arithmetic: one lookup instead of a square root, and no chance of the CPU
 * and GPU disagreeing about what digit 137 means.
 *
 * A program is *dead-code-free* when every result except the last is read by
 * a later instruction. Only dead-code-free programs are ever evaluated: a
 * program with an unused instruction is just a shorter program in disguise,
 * and that shorter program was already tried at the previous length.
 */
import { Op, OpKind } from './isa';

export interface SpaceConfig {
  nInputs: number;
  consts: number[];
  ops: Op[];
}

export interface Instr {
  /** index into cfg.ops */
  op: number;
  a: number;
  b: number;
}
export type Program = Instr[];

export type PairKind = 'unary' | 'comm0' | 'comm1' | 'nc0' | 'nc1';
export const PAIR_KINDS: PairKind[] = ['unary', 'comm0', 'comm1', 'nc0', 'nc1'];

export function pairKind(op: Op): PairKind {
  if (op.kind === 'unary') return 'unary';
  if (op.kind === 'comm') return op.self ? 'comm1' : 'comm0';
  return op.self ? 'nc1' : 'nc0';
}

export const valueCount = (cfg: SpaceConfig, slot: number): number => cfg.nInputs + cfg.consts.length + slot;
export const isConst = (cfg: SpaceConfig, v: number): boolean => v >= cfg.nInputs && v < cfg.nInputs + cfg.consts.length;
export const isResult = (cfg: SpaceConfig, v: number): boolean => v >= cfg.nInputs + cfg.consts.length;
export const resultIndex = (cfg: SpaceConfig, v: number): number => v - cfg.nInputs - cfg.consts.length;
export const firstResult = (cfg: SpaceConfig): number => cfg.nInputs + cfg.consts.length;

/**
 * Operand pairs enumerated for a given slot and op kind, packed as (a << 8 | b).
 * Rules:
 *  - both operands constant is never tried (that is constant folding; add the
 *    folded constant to the pool instead);
 *  - commutative ops only try a >= b;
 *  - a == b only for ops where op(v, v) is neither v nor a constant.
 * Unary ops try every non-constant value (packed as (a << 8)).
 */
export function pairTable(cfg: SpaceConfig, slot: number, kind: PairKind): Uint32Array {
  const V = valueCount(cfg, slot);
  const out: number[] = [];
  if (kind === 'unary') {
    for (let a = 0; a < V; a++) if (!isConst(cfg, a)) out.push(a << 8);
    return Uint32Array.from(out);
  }
  const comm = kind === 'comm0' || kind === 'comm1';
  const self = kind === 'comm1' || kind === 'nc1';
  for (let a = 0; a < V; a++) {
    const bMax = comm ? a : V - 1;
    for (let b = 0; b <= bMax; b++) {
      if (a === b && !self) continue;
      if (isConst(cfg, a) && isConst(cfg, b)) continue;
      out.push((a << 8) | b);
    }
  }
  return Uint32Array.from(out);
}

export interface SlotLayout {
  slot: number;
  /** digit base (op-major offset) per op index */
  opBase: number[];
  /** pair count per op index */
  opCount: number[];
  /** total digits in this slot */
  count: number;
  /** pair tables for this slot, keyed by kind */
  tables: Record<PairKind, Uint32Array>;
}

export function slotLayout(cfg: SpaceConfig, slot: number): SlotLayout {
  const tables = {} as Record<PairKind, Uint32Array>;
  for (const k of PAIR_KINDS) tables[k] = pairTable(cfg, slot, k);
  const opBase: number[] = [];
  const opCount: number[] = [];
  let acc = 0;
  for (const op of cfg.ops) {
    const n = tables[pairKind(op)].length;
    opBase.push(acc);
    opCount.push(n);
    acc += n;
  }
  return { slot, opBase, opCount, count: acc, tables };
}

export class Space {
  readonly layouts: SlotLayout[] = [];
  constructor(readonly cfg: SpaceConfig, readonly maxLen: number) {
    for (let i = 0; i < maxLen; i++) this.layouts.push(slotLayout(cfg, i));
  }
  slotCount(slot: number): number {
    return this.layouts[slot].count;
  }
  /** Raw number of programs of exactly length L (before dead-code pruning). */
  rawCount(L: number): bigint {
    let n = 1n;
    for (let i = 0; i < L; i++) n *= BigInt(this.slotCount(i));
    return n;
  }
  decodeDigit(slot: number, digit: number): Instr {
    const lay = this.layouts[slot];
    // ops are few (<= 30): linear scan is fine and mirrors the GPU
    let op = lay.opBase.length - 1;
    for (let i = 0; i < lay.opBase.length; i++) {
      if (digit < lay.opBase[i] + lay.opCount[i]) { op = i; break; }
    }
    const idx = digit - lay.opBase[op];
    const packed = lay.tables[pairKind(this.cfg.ops[op])][idx];
    return { op, a: packed >>> 8, b: packed & 0xff };
  }
  encodeDigit(slot: number, ins: Instr): number {
    const lay = this.layouts[slot];
    const table = lay.tables[pairKind(this.cfg.ops[ins.op])];
    const packed = (ins.a << 8) | (this.cfg.ops[ins.op].kind === 'unary' ? 0 : ins.b);
    const idx = table.indexOf(packed);
    if (idx < 0) throw new Error(`instruction not in enumeration: slot ${slot} ${JSON.stringify(ins)}`);
    return lay.opBase[ins.op] + idx;
  }
  /** Program id -> program (last slot is the least-significant digit). */
  decode(L: number, id: bigint): Program {
    const digits: number[] = new Array(L);
    for (let i = L - 1; i >= 0; i--) {
      const c = BigInt(this.slotCount(i));
      digits[i] = Number(id % c);
      id /= c;
    }
    return digits.map((d, i) => this.decodeDigit(i, d));
  }
  encode(prog: Program): bigint {
    let id = 0n;
    for (let i = 0; i < prog.length; i++) id = id * BigInt(this.slotCount(i)) + BigInt(this.encodeDigit(i, prog[i]));
    return id;
  }
}

/** Bitmask over result indices r0..r(L-2) that are read by some instruction. */
export function usedResults(cfg: SpaceConfig, prog: Program): number {
  const r0 = firstResult(cfg);
  let used = 0;
  for (const ins of prog) {
    if (ins.a >= r0) used |= 1 << (ins.a - r0);
    if (cfg.ops[ins.op].kind !== 'unary' && ins.b >= r0) used |= 1 << (ins.b - r0);
  }
  return used;
}

export function isDeadCodeFree(cfg: SpaceConfig, prog: Program): boolean {
  const L = prog.length;
  if (L <= 1) return true;
  const need = (1 << (L - 1)) - 1;
  return (usedResults(cfg, prog) & need) === need;
}

/** Evaluate a program on one input vector. Returns the u32 output. */
export function evalProgram(cfg: SpaceConfig, prog: Program, inputs: number[]): number {
  const n = cfg.nInputs, k = cfg.consts.length;
  const vals: number[] = new Array(n + k + prog.length);
  for (let i = 0; i < n; i++) vals[i] = inputs[i] >>> 0;
  for (let i = 0; i < k; i++) vals[n + i] = cfg.consts[i] >>> 0;
  let out = 0;
  for (let i = 0; i < prog.length; i++) {
    const ins = prog[i];
    out = cfg.ops[ins.op].fn(vals[ins.a], vals[ins.b] ?? 0) >>> 0;
    vals[n + k + i] = out;
  }
  return out;
}

/**
 * Enumerate every dead-code-free program of exactly length L (CPU, for tests
 * and the no-WebGPU fallback). Order matches the mixed-radix id order.
 */
export function* enumeratePrograms(space: Space, L: number): Generator<Program> {
  const cfg = space.cfg;
  const prog: Instr[] = new Array(L);
  function* rec(slot: number): Generator<Program> {
    const c = space.slotCount(slot);
    for (let d = 0; d < c; d++) {
      prog[slot] = space.decodeDigit(slot, d);
      if (slot === L - 1) {
        if (isDeadCodeFree(cfg, prog)) yield prog.slice();
      } else {
        yield* rec(slot + 1);
      }
    }
  }
  if (L === 0) return;
  yield* rec(0);
}

export function programToString(cfg: SpaceConfig, prog: Program): string {
  const name = (v: number): string => {
    if (v < cfg.nInputs) return 'xyz'[v];
    if (isConst(cfg, v)) return String(cfg.consts[v - cfg.nInputs] >>> 0);
    return 'r' + resultIndex(cfg, v);
  };
  return prog
    .map((ins, i) => {
      const op = cfg.ops[ins.op];
      return `r${i} = ${op.id} ${name(ins.a)}${op.kind === 'unary' ? '' : ' ' + name(ins.b)}`;
    })
    .join('; ');
}
