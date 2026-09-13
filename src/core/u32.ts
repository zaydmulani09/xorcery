/**
 * 32-bit word helpers with exact, documented semantics.
 *
 * Every arithmetic operation in xorcery is defined over 32-bit words. JavaScript
 * numbers are doubles, so each helper normalises its result back into the
 * unsigned range [0, 2^32) with `>>> 0`. The same semantics are implemented in
 * WGSL by `src/gpu/wgsl.ts`; the two must agree bit-for-bit, and the test-suite
 * plus the runtime "tri-check" (JS spec vs JS program vs GPU) enforce that.
 *
 * Deliberate choices (documented because they are the kind of thing people
 * argue about):
 *  - shifts mask the amount by 31 (like x86, WGSL and JavaScript);
 *  - clz(0) = ctz(0) = 32 (like WGSL and Math.clz32; unlike x86 `bsr`);
 *  - unsigned division by zero returns 0xFFFFFFFF and remainder returns the
 *    dividend (the RISC-V convention), so every op is total.
 */

export const MASK = 0xffffffff;
export const TWO32 = 0x100000000;

export const u32 = (x: number): number => x >>> 0;
export const i32 = (x: number): number => x | 0;

export const add = (a: number, b: number): number => (a + b) >>> 0;
export const sub = (a: number, b: number): number => (a - b) >>> 0;
export const mul = (a: number, b: number): number => Math.imul(a, b) >>> 0;
export const and = (a: number, b: number): number => (a & b) >>> 0;
export const or = (a: number, b: number): number => (a | b) >>> 0;
export const xor = (a: number, b: number): number => (a ^ b) >>> 0;
export const shl = (a: number, b: number): number => (a << (b & 31)) >>> 0;
export const shr = (a: number, b: number): number => a >>> (b & 31);
export const sar = (a: number, b: number): number => (a >> (b & 31)) >>> 0;
export const not = (a: number): number => ~a >>> 0;
export const neg = (a: number): number => -a >>> 0;
export const clz = (a: number): number => Math.clz32(a);
export const ctz = (a: number): number => (a === 0 ? 32 : 31 - Math.clz32(a & -a));
export const popcnt = (a: number): number => {
  let x = a >>> 0;
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
};
export const bswap = (a: number): number =>
  ((a >>> 24) | ((a >>> 8) & 0xff00) | ((a << 8) & 0xff0000) | (a << 24)) >>> 0;
export const brev = (a: number): number => {
  let x = a >>> 0;
  x = ((x >>> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  x = ((x >>> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  x = ((x >>> 4) & 0x0f0f0f0f) | ((x & 0x0f0f0f0f) << 4);
  return bswap(x >>> 0);
};
export const rotl = (a: number, b: number): number => {
  const s = b & 31;
  return ((a << s) | (a >>> ((32 - s) & 31))) >>> 0;
};
export const rotr = (a: number, b: number): number => {
  const s = b & 31;
  return ((a >>> s) | (a << ((32 - s) & 31))) >>> 0;
};
export const ult = (a: number, b: number): number => (a >>> 0 < b >>> 0 ? 1 : 0);
export const ule = (a: number, b: number): number => (a >>> 0 <= b >>> 0 ? 1 : 0);
export const slt = (a: number, b: number): number => ((a | 0) < (b | 0) ? 1 : 0);
export const sle = (a: number, b: number): number => ((a | 0) <= (b | 0) ? 1 : 0);
export const eq = (a: number, b: number): number => (a >>> 0 === b >>> 0 ? 1 : 0);
export const ne = (a: number, b: number): number => (a >>> 0 !== b >>> 0 ? 1 : 0);
export const umin = (a: number, b: number): number => (a >>> 0 < b >>> 0 ? a >>> 0 : b >>> 0);
export const umax = (a: number, b: number): number => (a >>> 0 > b >>> 0 ? a >>> 0 : b >>> 0);
export const smin = (a: number, b: number): number => ((a | 0) < (b | 0) ? a >>> 0 : b >>> 0);
export const smax = (a: number, b: number): number => ((a | 0) > (b | 0) ? a >>> 0 : b >>> 0);
export const udiv = (a: number, b: number): number => (b >>> 0 === 0 ? 0xffffffff : Math.floor((a >>> 0) / (b >>> 0)) >>> 0);
export const urem = (a: number, b: number): number => (b >>> 0 === 0 ? a >>> 0 : (a >>> 0) % (b >>> 0) >>> 0);
/** High 32 bits of the 64-bit unsigned product. */
export const mulhi = (a: number, b: number): number => {
  const aLo = a & 0xffff, aHi = a >>> 16;
  const bLo = b & 0xffff, bHi = b >>> 16;
  const lo = aLo * bLo;
  const mid1 = aHi * bLo;
  const mid2 = aLo * bHi;
  const hi = aHi * bHi;
  // carry out of the low 32 bits: (lo >>> 16) + (mid1 & 0xffff) + (mid2 & 0xffff) can exceed 2^16
  const carry = Math.floor(((lo >>> 16) + (mid1 & 0xffff) + (mid2 & 0xffff)) / 0x10000);
  return (hi + (mid1 >>> 16) + (mid2 >>> 16) + carry) >>> 0;
};
/** High 32 bits of the 64-bit signed product. */
export const mulhs = (a: number, b: number): number => {
  // mulhs = mulhi - (a<0 ? b : 0) - (b<0 ? a : 0)
  let h = mulhi(a, b);
  if ((a | 0) < 0) h = (h - b) >>> 0;
  if ((b | 0) < 0) h = (h - a) >>> 0;
  return h >>> 0;
};
export const abs = (a: number): number => ((a | 0) < 0 ? -a >>> 0 : a >>> 0);
export const sign = (a: number): number => ((a | 0) < 0 ? 0xffffffff : a === 0 ? 0 : 1);

export const hex = (x: number): string => '0x' + (x >>> 0).toString(16).padStart(8, '0');

/** Format a u32 the way a C programmer would want to read it. */
export function fmtConst(x: number): string {
  x >>>= 0;
  if (x < 256) return String(x);
  if (x === 0xffffffff) return '-1';
  return hex(x);
}

/** Deterministic 32-bit hash (lowbias32 by Chris Wellons) — used for sample generation. */
export function hash32(x: number): number {
  x = x >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}
