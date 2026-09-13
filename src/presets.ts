/**
 * Preset specs.
 *
 * The first 25 are the "Hacker's Delight" benchmark from Gulwani, Jha,
 * Tiwari & Venkatesan, "Synthesis of Loop-Free Programs" (PLDI 2011), the
 * standard suite for bit-vector program synthesis. They are written here as
 * *specifications* (what the function computes), not as the known tricks, so
 * the search has to rediscover the trick. Where the classic trick is the only
 * sensible way to state the function, the spec is the trick and the search
 * must prove it minimal — or find something shorter.
 *
 * `size` is the instruction count of the textbook solution in this ISA, for
 * reference. It is not fed to the search.
 */
import { OpGroup } from './core/isa';

export interface Preset {
  id: string;
  name: string;
  /** hd = Hacker's Delight benchmark P1..P25; extra = additional classics */
  group: 'hd' | 'extra';
  spec: string;
  /** what the textbook says, for the reader */
  note: string;
  /** textbook solution length in this ISA, if known */
  size?: number;
  groups?: OpGroup[];
  /** explicit op ids (overrides groups) */
  ops?: string[];
  consts?: number[];
  maxLen?: number;
  /** likely beyond an integrated GPU's horizon in the given ISA */
  hard?: boolean;
}

export const DEFAULT_CONSTS = [0, 1, 31, 0xffffffff];
export const BASE: OpGroup[] = ['base'];
export const BASE_CMP: OpGroup[] = ['base', 'cmp'];
export const BASE_BITS: OpGroup[] = ['base', 'bits'];

export const PRESETS: Preset[] = [
  // ---- Hacker's Delight benchmark ---------------------------------------
  { id: 'p1', group: 'hd', name: 'P1 · turn off the rightmost 1-bit', spec: 'x & (x - 1)', note: 'HD 2-1. The search must show nothing shorter exists.', size: 2, groups: BASE },
  { id: 'p2', group: 'hd', name: 'P2 · test for 2^n − 1 form', spec: 'x & (x + 1)', note: 'HD 2-1: zero iff x is 0 or of the form 2^n - 1.', size: 2, groups: BASE },
  { id: 'p3', group: 'hd', name: 'P3 · isolate the rightmost 1-bit', spec: 'x & -x', note: 'HD 2-1.', size: 2, groups: BASE },
  { id: 'p4', group: 'hd', name: 'P4 · mask from rightmost 1 to bit 0', spec: 'x ^ (x - 1)', note: 'HD 2-1: ones from the rightmost 1-bit down.', size: 2, groups: BASE },
  { id: 'p5', group: 'hd', name: 'P5 · right-propagate the rightmost 1', spec: 'x | (x - 1)', note: 'HD 2-1.', size: 2, groups: BASE },
  { id: 'p6', group: 'hd', name: 'P6 · turn on the rightmost 0-bit', spec: 'x | (x + 1)', note: 'HD 2-1.', size: 2, groups: BASE },
  { id: 'p7', group: 'hd', name: 'P7 · isolate the rightmost 0-bit', spec: '~x & (x + 1)', note: 'HD 2-1.', size: 3, groups: BASE },
  { id: 'p8', group: 'hd', name: 'P8 · mask of the trailing 0s', spec: '~x & (x - 1)', note: 'HD 2-1.', size: 3, groups: BASE },
  { id: 'p9', group: 'hd', name: 'P9 · absolute value, no branches', spec: '(int)x < 0 ? -x : x', note: 'HD 2-4: (x ^ (x >> 31)) - (x >> 31) with an arithmetic shift.', size: 3, groups: BASE },
  { id: 'p10', group: 'hd', name: 'P10 · same number of leading zeros?', spec: 'clz(x) == clz(y)', note: 'HD 5-3: nlz(x) == nlz(y) iff (x ^ y) <= (x & y), unsigned.', size: 3, groups: BASE_CMP },
  { id: 'p11', group: 'hd', name: 'P11 · fewer leading zeros than y?', spec: 'clz(x) < clz(y)', note: 'HD 5-3: nlz(x) < nlz(y) iff (x & ~y) > y, unsigned.', size: 3, groups: BASE_CMP },
  { id: 'p12', group: 'hd', name: 'P12 · leading zeros ≤ y’s?', spec: 'clz(x) <= clz(y)', note: 'HD 5-3.', size: 3, groups: BASE_CMP },
  { id: 'p13', group: 'hd', name: 'P13 · sign function', spec: '(int)x < 0 ? -1 : (x == 0 ? 0 : 1)', note: 'HD 2-7: (x >> 31) | (-x >>> 31).', size: 4, groups: BASE },
  { id: 'p14', group: 'hd', name: 'P14 · floor average, no overflow', spec: '(x >> 1) + (y >> 1) + (x & y & 1)', note: 'HD 2-5: (x & y) + ((x ^ y) >> 1).', size: 4, groups: BASE },
  { id: 'p15', group: 'hd', name: 'P15 · ceiling average, no overflow', spec: '(x >> 1) + (y >> 1) + ((x | y) & 1)', note: 'HD 2-5: (x | y) - ((x ^ y) >> 1).', size: 4, groups: BASE },
  { id: 'p16', group: 'hd', name: 'P16 · signed max, no branches', spec: '(int)x < (int)y ? y : x', note: 'HD 2-19: x ^ ((x ^ y) & -(x < y)).', size: 5, groups: BASE_CMP },
  { id: 'p17', group: 'hd', name: 'P17 · turn off rightmost run of 1s', spec: '((x | (x - 1)) + 1) & x', note: 'HD 2-1.', size: 4, groups: BASE },
  { id: 'p18', group: 'hd', name: 'P18 · is x a power of two?', spec: 'x != 0 && (x & (x - 1)) == 0', note: 'HD 2-1 — 1 if x is a power of two, else 0.', size: 4, groups: BASE_CMP },
  { id: 'p19', group: 'hd', name: 'P19 · exchange two bit fields', spec: 'uint32_t t = ((x >> z) ^ x) & y;\nx ^ t ^ (t << z)', note: 'HD 2-20, fields selected by mask y and distance z. Three inputs, six instructions — at the horizon.', size: 6, groups: BASE, consts: [], maxLen: 6, hard: true },
  { id: 'p20', group: 'hd', name: 'P20 · next number with the same popcount', spec: 'uint32_t s = x & -x;\nuint32_t r = s + x;\nr | (((x ^ r) >> 2) / s)', note: "Gosper's hack (HD 2-1). Needs division or ctz; well beyond the brute-force horizon.", size: 9, groups: ['base', 'bits'], hard: true },
  { id: 'p21', group: 'hd', name: 'P21 · cycle through three values', spec: 'x == 0 ? y : (x == y ? z : 0)', note: 'Cycle a -> b -> c -> a for a = 0, b = y, c = z. Three inputs; the textbook answer uses 8 ops.', size: 8, groups: BASE_CMP, consts: [0], hard: true },
  { id: 'p22', group: 'hd', name: 'P22 · parity', spec: 'popcnt(x) & 1', note: 'HD 5-2: an xor-shift cascade (11 ops) without popcount; 2 with it.', size: 2, groups: BASE_BITS },
  { id: 'p23', group: 'hd', name: 'P23 · population count', spec: 'popcnt(x)', note: 'HD 5-1: 12 ops of masks and shifts in the base ISA. Enable "bit count" to see it collapse to one.', size: 12, groups: BASE, hard: true, consts: [0, 1, 2, 4, 0x55555555, 0x33333333, 0x0f0f0f0f, 0x01010101] },
  { id: 'p24', group: 'hd', name: 'P24 · round up to a power of two', spec: 'uint32_t t = x - 1;\nt = t | (t >> 1);\nt = t | (t >> 2);\nt = t | (t >> 4);\nt = t | (t >> 8);\nt = t | (t >> 16);\nt + 1', note: 'HD 3-2 clp2, 12 ops in the base ISA. With clz enabled a much shorter program exists — but 1 << (32 - clz(x - 1)) is wrong at x = 0, and the search will not accept it.', size: 12, groups: BASE_BITS, consts: [0, 1, 31, 32, 0xffffffff], hard: true },
  { id: 'p25', group: 'hd', name: 'P25 · high half of a 32×32 product', spec: 'mulhi(x, y)', note: 'HD 8-2: 16 ops with 16-bit halves in the base ISA; one op with mulhi enabled.', size: 16, groups: ['base', 'mul'], consts: [0xffff, 16], hard: true },

  // ---- extras -----------------------------------------------------------
  { id: 'sext8', group: 'extra', name: 'sign-extend a byte', spec: '(x & 0x80) != 0 ? (x | 0xffffff00) : (x & 0xff)', note: 'Classic answer: (x << 24) >> 24 with an arithmetic shift. Or ((x + 128) & 255) - 128.', size: 2, groups: BASE, consts: [24, 0xff, 128] },
  { id: 'half', group: 'extra', name: 'signed halve, round toward zero', spec: '(int)x / 2', note: 'HD 10-1: (x + (x >>> 31)) >> 1 with an arithmetic shift.', size: 3, groups: BASE },
  { id: 'ctz', group: 'extra', name: 'trailing zeros without ctz', spec: 'ctz(x)', note: 'HD 5-4: popcnt(~x & (x - 1)), or 31 - clz(x & -x) which is wrong at x = 0.', size: 3, ops: ['add', 'sub', 'and', 'or', 'xor', 'shl', 'shr', 'sar', 'not', 'neg', 'popcnt', 'clz'], consts: [0, 1, 31, 32, 0xffffffff] },
  { id: 'hibit', group: 'extra', name: 'isolate the highest 1-bit', spec: 'x == 0 ? 0 : (1 << (31 - clz(x)))', note: 'With clz: 0x80000000 >> clz(x) — but the shift amount is masked, so x = 0 needs care.', size: 3, ops: ['add', 'sub', 'and', 'or', 'xor', 'shl', 'shr', 'sar', 'not', 'neg', 'clz'], consts: [0, 1, 31, 0x80000000, 0xffffffff] },
  { id: 'umin', group: 'extra', name: 'unsigned min, no branches', spec: 'x < y ? x : y', note: 'HD 2-19: y ^ ((x ^ y) & -(x < y)) needs a compare; y + ((x - y) & ((x - y) >> 31)) is a famous *wrong* answer (overflow).', size: 5, groups: BASE_CMP },
  { id: 'zbyte', group: 'extra', name: 'does the word contain a zero byte?', spec: '(x & 0xff) == 0 || (x & 0xff00) == 0 || (x & 0xff0000) == 0 || (x & 0xff000000) == 0', note: 'HD 6-1: (x - 0x01010101) & ~x & 0x80808080 is nonzero iff some byte is zero. Here the spec wants a 0/1 answer.', size: 5, groups: BASE_CMP, consts: [0, 0x01010101, 0x80808080] },
  { id: 'absdiff', group: 'extra', name: 'absolute difference (signed)', spec: '(int)x < (int)y ? y - x : x - y', note: 'Textbook: t = x - y; (t ^ (t >> 31)) - (t >> 31). Note it is only correct where x - y does not overflow — the search will tell you.', size: 4, groups: BASE },
  { id: 'round8', group: 'extra', name: 'round up to a multiple of 8', spec: '(x + 7) & ~7', note: 'Two ops with the constant -8 in the pool; three without.', size: 2, groups: BASE, consts: [0, 1, 7, 8, 0xfffffff8] },
  { id: 'rev2', group: 'extra', name: 'swap adjacent bit pairs', spec: '((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2)', note: 'One step of a bit reversal. The search knows the mask trick or finds another.', size: 5, groups: BASE, consts: [2, 0x33333333, 0xcccccccc] },
  { id: 'sqrtish', group: 'extra', name: 'is x a multiple of 3? (no division)', spec: 'x % 3 == 0', note: 'Multiply by the modular inverse: x * 0xaaaaaaab <= 0x55555555. Needs multiply and compare.', size: 2, groups: ['base', 'mul', 'cmp'], consts: [0, 0xaaaaaaab, 0x55555555] },
];

export function presetById(id: string): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}
