/**
 * The instruction set the superoptimizer searches over.
 *
 * Every op is a total function over 32-bit words (see `u32.ts` for the exact
 * semantics). Ops are grouped so the UI can toggle whole families; a smaller
 * ISA means a smaller search space and a faster (but weaker) search.
 *
 * `kind` drives the enumeration:
 *  - `comm`    : commutative binary op — only ordered pairs a >= b are enumerated;
 *  - `noncomm` : binary op where operand order matters — all pairs;
 *  - `unary`   : one operand.
 * `self` says whether `op(v, v)` is worth enumerating. It is false when the
 * result is either the operand itself (x & x) or a constant (x ^ x, x - x,
 * x < x), because a strictly shorter program already computes that.
 *
 * Each op carries *templates* for every output language. `$a`/`$b` are the
 * operands; `prec` is the precedence of the produced expression (15 = atomic,
 * used for calls) so the printer can parenthesise correctly.
 */
import * as U from './u32';

export type OpKind = 'comm' | 'noncomm' | 'unary';
export type OpGroup = 'base' | 'mul' | 'bits' | 'cmp' | 'cmpx' | 'rot' | 'minmax' | 'div';
export type Lang = 'c' | 'rust' | 'js' | 'wgsl' | 'spec';

export interface Tmpl {
  /** template with $a / $b placeholders */
  t: string;
  /** precedence of the resulting expression (C-like table; 15 = atomic) */
  prec: number;
  /** helper function this template relies on (emitted once in a preamble) */
  helper?: string;
}

export interface Op {
  id: string;
  /** short human description */
  desc: string;
  group: OpGroup;
  kind: OpKind;
  self: boolean;
  fn: (a: number, b: number) => number;
  c: Tmpl;
  rust: Tmpl;
  js: Tmpl;
  wgsl: Tmpl;
  /** display form: the spec language itself, so a result can be pasted back as a spec */
  spec: Tmpl;
}

export const P_UN = 14, P_MUL = 13, P_ADD = 12, P_SHIFT = 11, P_CMP = 10, P_EQ = 9, P_AND = 8, P_XOR = 7, P_OR = 6, P_CALL = 15;
/** Precedence used by the printer for atoms (names, literals). */
export const P_ATOM = 15;

const call = (name: string, helper?: string): Tmpl => ({ t: `${name}($a, $b)`, prec: P_CALL, helper });
const call1 = (name: string, helper?: string): Tmpl => ({ t: `${name}($a)`, prec: P_CALL, helper });
const inf = (sym: string, prec: number): Tmpl => ({ t: `$a ${sym} $b`, prec });
const T = (t: string, prec = P_CALL, helper?: string): Tmpl => ({ t, prec, helper });

/** Display templates (spec-language syntax) where the C template is too noisy. */
const SPEC_TMPL: Record<string, Tmpl> = {
  sar: T('(int)$a >> $b', P_SHIFT),
  eq: T('($a == $b)'), ne: T('($a != $b)'),
  ult: T('($a < $b)'), ule: T('($a <= $b)'),
  slt: T('((int)$a < (int)$b)'), sle: T('((int)$a <= (int)$b)'),
  clz: call1('clz'), ctz: call1('ctz'), popcnt: call1('popcnt'),
  rotl: call('rotl'), rotr: call('rotr'), bswap: call1('bswap'), brev: call1('brev'),
  umin: call('min'), umax: call('max'), smin: call('smin'), smax: call('smax'),
  udiv: inf('/', P_MUL), urem: inf('%', P_MUL), mulhi: call('mulhi'),
};

function op(
  id: string, desc: string, group: OpGroup, kind: OpKind, self: boolean,
  fn: (a: number, b: number) => number,
  c: Tmpl, rust: Tmpl, js: Tmpl, wgsl: Tmpl,
): Op {
  return { id, desc, group, kind, self, fn, c, rust, js, wgsl, spec: SPEC_TMPL[id] ?? c };
}

export const OPS: Op[] = [
  // ---- base -------------------------------------------------------------
  op('add', 'a + b', 'base', 'comm', true, U.add,
    inf('+', P_ADD), T('$a.wrapping_add($b)'), T('(($a + $b) >>> 0)'), inf('+', P_ADD)),
  op('sub', 'a - b', 'base', 'noncomm', false, U.sub,
    inf('-', P_ADD), T('$a.wrapping_sub($b)'), T('(($a - $b) >>> 0)'), inf('-', P_ADD)),
  op('and', 'a & b', 'base', 'comm', false, U.and,
    inf('&', P_AND), inf('&', P_AND), T('(($a & $b) >>> 0)'), inf('&', P_AND)),
  op('or', 'a | b', 'base', 'comm', false, U.or,
    inf('|', P_OR), inf('|', P_OR), T('(($a | $b) >>> 0)'), inf('|', P_OR)),
  op('xor', 'a ^ b', 'base', 'comm', false, U.xor,
    inf('^', P_XOR), inf('^', P_XOR), T('(($a ^ $b) >>> 0)'), inf('^', P_XOR)),
  op('shl', 'a << b', 'base', 'noncomm', true, U.shl,
    inf('<<', P_SHIFT), T('$a.wrapping_shl($b)'), T('(($a << ($b & 31)) >>> 0)'), T('$a << ($b & 31u)', P_SHIFT)),
  op('shr', 'a >> b (logical)', 'base', 'noncomm', true, U.shr,
    inf('>>', P_SHIFT), T('$a.wrapping_shr($b)'), T('($a >>> ($b & 31))'), T('$a >> ($b & 31u)', P_SHIFT)),
  op('sar', 'a >> b (arithmetic)', 'base', 'noncomm', true, U.sar,
    T('(uint32_t)((int32_t)$a >> $b)', P_UN), T('(($a as i32).wrapping_shr($b)) as u32', P_UN),
    T('(($a >> ($b & 31)) >>> 0)'), T('bitcast<u32>(bitcast<i32>($a) >> ($b & 31u))')),
  op('not', '~a', 'base', 'unary', true, U.not,
    T('~$a', P_UN), T('!$a', P_UN), T('(~$a >>> 0)'), T('~$a', P_UN)),
  op('neg', '-a', 'base', 'unary', true, U.neg,
    T('-$a', P_UN), T('$a.wrapping_neg()'), T('(-$a >>> 0)'), T('(0u - $a)')),
  // ---- multiply ---------------------------------------------------------
  op('mul', 'a * b (low 32 bits)', 'mul', 'comm', true, U.mul,
    inf('*', P_MUL), T('$a.wrapping_mul($b)'), T('(Math.imul($a, $b) >>> 0)'), inf('*', P_MUL)),
  op('mulhi', 'high 32 bits of a * b (unsigned)', 'mul', 'comm', true, U.mulhi,
    call('mulhi32', 'mulhi'), T('(($a as u64 * $b as u64) >> 32) as u32', P_UN),
    call('mulhi32', 'mulhi'), call('mulhi32', 'mulhi')),
  // ---- bit counting -----------------------------------------------------
  op('clz', 'count leading zeros (clz(0)=32)', 'bits', 'unary', true, U.clz,
    call1('clz32', 'clz'), T('$a.leading_zeros()'), call1('Math.clz32'), call1('countLeadingZeros')),
  op('ctz', 'count trailing zeros (ctz(0)=32)', 'bits', 'unary', true, U.ctz,
    call1('ctz32', 'ctz'), T('$a.trailing_zeros()'), call1('ctz32', 'ctz'), call1('countTrailingZeros')),
  op('popcnt', 'population count', 'bits', 'unary', true, U.popcnt,
    call1('popcnt32', 'popcnt'), T('$a.count_ones()'), call1('popcnt32', 'popcnt'), call1('countOneBits')),
  // ---- compare ----------------------------------------------------------
  op('eq', 'a == b ? 1 : 0', 'cmp', 'comm', false, U.eq,
    T('(uint32_t)($a == $b)', P_UN), T('($a == $b) as u32', P_UN), T('(+($a === $b))'), T('u32($a == $b)')),
  op('ult', 'a < b (unsigned) ? 1 : 0', 'cmp', 'noncomm', false, U.ult,
    T('(uint32_t)($a < $b)', P_UN), T('($a < $b) as u32', P_UN), T('(+($a < $b))'), T('u32($a < $b)')),
  op('slt', 'a < b (signed) ? 1 : 0', 'cmp', 'noncomm', false, U.slt,
    T('(uint32_t)((int32_t)$a < (int32_t)$b)', P_UN), T('(($a as i32) < ($b as i32)) as u32', P_UN),
    T('(+(($a | 0) < ($b | 0)))'), T('u32(bitcast<i32>($a) < bitcast<i32>($b))')),
  op('ne', 'a != b ? 1 : 0', 'cmpx', 'comm', false, U.ne,
    T('(uint32_t)($a != $b)', P_UN), T('($a != $b) as u32', P_UN), T('(+($a !== $b))'), T('u32($a != $b)')),
  op('ule', 'a <= b (unsigned) ? 1 : 0', 'cmpx', 'noncomm', false, U.ule,
    T('(uint32_t)($a <= $b)', P_UN), T('($a <= $b) as u32', P_UN), T('(+($a <= $b))'), T('u32($a <= $b)')),
  op('sle', 'a <= b (signed) ? 1 : 0', 'cmpx', 'noncomm', false, U.sle,
    T('(uint32_t)((int32_t)$a <= (int32_t)$b)', P_UN), T('(($a as i32) <= ($b as i32)) as u32', P_UN),
    T('(+(($a | 0) <= ($b | 0)))'), T('u32(bitcast<i32>($a) <= bitcast<i32>($b))')),
  // ---- rotate / byte ops ------------------------------------------------
  op('rotl', 'rotate left', 'rot', 'noncomm', true, U.rotl,
    call('rotl32', 'rotl'), T('$a.rotate_left($b & 31)'), call('rotl32', 'rotl'), call('rotl32', 'rotl')),
  op('rotr', 'rotate right', 'rot', 'noncomm', true, U.rotr,
    call('rotr32', 'rotr'), T('$a.rotate_right($b & 31)'), call('rotr32', 'rotr'), call('rotr32', 'rotr')),
  op('bswap', 'byte swap', 'rot', 'unary', true, U.bswap,
    call1('bswap32', 'bswap'), T('$a.swap_bytes()'), call1('bswap32', 'bswap'), call1('bswap32', 'bswap')),
  op('brev', 'bit reverse', 'rot', 'unary', true, U.brev,
    call1('brev32', 'brev'), T('$a.reverse_bits()'), call1('brev32', 'brev'), call1('reverseBits')),
  // ---- min / max --------------------------------------------------------
  op('umin', 'min (unsigned)', 'minmax', 'comm', false, U.umin,
    call('umin32', 'umin'), T('$a.min($b)'), call('Math.min'), call('min')),
  op('umax', 'max (unsigned)', 'minmax', 'comm', false, U.umax,
    call('umax32', 'umax'), T('$a.max($b)'), call('Math.max'), call('max')),
  op('smin', 'min (signed)', 'minmax', 'comm', false, U.smin,
    call('smin32', 'smin'), T('(($a as i32).min($b as i32)) as u32', P_UN), call('smin32', 'smin'), T('bitcast<u32>(min(bitcast<i32>($a), bitcast<i32>($b)))')),
  op('smax', 'max (signed)', 'minmax', 'comm', false, U.smax,
    call('smax32', 'smax'), T('(($a as i32).max($b as i32)) as u32', P_UN), call('smax32', 'smax'), T('bitcast<u32>(max(bitcast<i32>($a), bitcast<i32>($b)))')),
  // ---- division ---------------------------------------------------------
  op('udiv', 'a / b (unsigned; x/0 = 0xFFFFFFFF)', 'div', 'noncomm', true, U.udiv,
    call('udiv32', 'udiv'), call('udiv32', 'udiv'), call('udiv32', 'udiv'), call('udiv32', 'udiv')),
  op('urem', 'a % b (unsigned; x%0 = x)', 'div', 'noncomm', false, U.urem,
    call('urem32', 'urem'), call('urem32', 'urem'), call('urem32', 'urem'), call('urem32', 'urem')),
];

export const OP_BY_ID: Map<string, Op> = new Map(OPS.map((o) => [o.id, o]));

export const GROUPS: { id: OpGroup; label: string; hint: string }[] = [
  { id: 'base', label: 'base', hint: 'add sub and or xor shl shr sar not neg' },
  { id: 'mul', label: 'multiply', hint: 'mul mulhi' },
  { id: 'bits', label: 'bit count', hint: 'clz ctz popcnt' },
  { id: 'cmp', label: 'compare', hint: 'eq ult slt (0/1 results)' },
  { id: 'cmpx', label: 'compare+', hint: 'ne ule sle' },
  { id: 'rot', label: 'rotate/bytes', hint: 'rotl rotr bswap brev' },
  { id: 'minmax', label: 'min/max', hint: 'umin umax smin smax' },
  { id: 'div', label: 'divide', hint: 'udiv urem' },
];

export function opsForGroups(groups: Iterable<OpGroup>): Op[] {
  const set = new Set(groups);
  return OPS.filter((o) => set.has(o.group));
}

export function opsForIds(ids: Iterable<string>): Op[] {
  const set = new Set(ids);
  return OPS.filter((o) => set.has(o.id));
}

/** Helper routines referenced by templates, per language. */
export const HELPERS: Record<Lang, Record<string, string>> = {
  spec: {},
  c: {
    clz: 'static inline uint32_t clz32(uint32_t x) { return x ? (uint32_t)__builtin_clz(x) : 32u; }',
    ctz: 'static inline uint32_t ctz32(uint32_t x) { return x ? (uint32_t)__builtin_ctz(x) : 32u; }',
    popcnt: 'static inline uint32_t popcnt32(uint32_t x) { return (uint32_t)__builtin_popcount(x); }',
    rotl: 'static inline uint32_t rotl32(uint32_t x, uint32_t n) { n &= 31; return (x << n) | (x >> ((32 - n) & 31)); }',
    rotr: 'static inline uint32_t rotr32(uint32_t x, uint32_t n) { n &= 31; return (x >> n) | (x << ((32 - n) & 31)); }',
    bswap: 'static inline uint32_t bswap32(uint32_t x) { return __builtin_bswap32(x); }',
    brev: 'static inline uint32_t brev32(uint32_t x) { x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1); x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2); x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4); return __builtin_bswap32(x); }',
    umin: 'static inline uint32_t umin32(uint32_t a, uint32_t b) { return a < b ? a : b; }',
    umax: 'static inline uint32_t umax32(uint32_t a, uint32_t b) { return a > b ? a : b; }',
    smin: 'static inline uint32_t smin32(uint32_t a, uint32_t b) { return (int32_t)a < (int32_t)b ? a : b; }',
    smax: 'static inline uint32_t smax32(uint32_t a, uint32_t b) { return (int32_t)a > (int32_t)b ? a : b; }',
    udiv: 'static inline uint32_t udiv32(uint32_t a, uint32_t b) { return b ? a / b : 0xFFFFFFFFu; }',
    urem: 'static inline uint32_t urem32(uint32_t a, uint32_t b) { return b ? a % b : a; }',
    mulhi: 'static inline uint32_t mulhi32(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }',
  },
  rust: {
    udiv: 'fn udiv32(a: u32, b: u32) -> u32 { if b == 0 { u32::MAX } else { a / b } }',
    urem: 'fn urem32(a: u32, b: u32) -> u32 { if b == 0 { a } else { a % b } }',
  },
  js: {
    ctz: 'const ctz32 = (x) => x === 0 ? 32 : 31 - Math.clz32(x & -x);',
    popcnt: 'const popcnt32 = (x) => { x = x - ((x >>> 1) & 0x55555555); x = (x & 0x33333333) + ((x >>> 2) & 0x33333333); return Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24; };',
    rotl: 'const rotl32 = (x, n) => { n &= 31; return ((x << n) | (x >>> ((32 - n) & 31))) >>> 0; };',
    rotr: 'const rotr32 = (x, n) => { n &= 31; return ((x >>> n) | (x << ((32 - n) & 31))) >>> 0; };',
    bswap: 'const bswap32 = (x) => ((x >>> 24) | ((x >>> 8) & 0xff00) | ((x << 8) & 0xff0000) | (x << 24)) >>> 0;',
    brev: 'const brev32 = (x) => { x = ((x >>> 1) & 0x55555555) | ((x & 0x55555555) << 1); x = ((x >>> 2) & 0x33333333) | ((x & 0x33333333) << 2); x = ((x >>> 4) & 0x0f0f0f0f) | ((x & 0x0f0f0f0f) << 4); return bswap32(x >>> 0); };',
    smin: 'const smin32 = (a, b) => ((a | 0) < (b | 0) ? a : b) >>> 0;',
    smax: 'const smax32 = (a, b) => ((a | 0) > (b | 0) ? a : b) >>> 0;',
    udiv: 'const udiv32 = (a, b) => b === 0 ? 0xffffffff : Math.floor(a / b) >>> 0;',
    urem: 'const urem32 = (a, b) => b === 0 ? a : (a % b) >>> 0;',
    mulhi: 'const mulhi32 = (a, b) => Number((BigInt(a) * BigInt(b)) >> 32n) >>> 0;',
  },
  wgsl: {
    rotl: 'fn rotl32(x: u32, n: u32) -> u32 { let s = n & 31u; return (x << s) | (x >> ((32u - s) & 31u)); }',
    rotr: 'fn rotr32(x: u32, n: u32) -> u32 { let s = n & 31u; return (x >> s) | (x << ((32u - s) & 31u)); }',
    bswap: 'fn bswap32(x: u32) -> u32 { return (x >> 24u) | ((x >> 8u) & 0xff00u) | ((x << 8u) & 0xff0000u) | (x << 24u); }',
    udiv: 'fn udiv32(a: u32, b: u32) -> u32 { return select(a / b, 0xffffffffu, b == 0u); }',
    urem: 'fn urem32(a: u32, b: u32) -> u32 { return select(a % b, a, b == 0u); }',
    mulhi: 'fn mulhi32(a: u32, b: u32) -> u32 { let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u; let lo = al * bl; let m1 = ah * bl; let m2 = al * bh; let hi = ah * bh; let c = ((lo >> 16u) + (m1 & 0xffffu) + (m2 & 0xffffu)) >> 16u; return hi + (m1 >> 16u) + (m2 >> 16u) + c; }',
  },
};

/** The JS-side dependency order for helpers (brev needs bswap). */
export const HELPER_DEPS: Record<string, string[]> = { brev: ['bswap'] };
