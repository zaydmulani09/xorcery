/**
 * Synthesized ("magic") constants.
 *
 * The constant pool is small on purpose — every constant widens every slot —
 * but the *last* instruction of a program can use any 32-bit constant at no
 * enumeration cost: for `r = v op c` with v known on every sample and the
 * target known, c is determined (or constrained bit-by-bit) by the samples.
 * Solve for c from the samples, then check it on all of them.
 *
 *   add:  c = t - v            sub:  c = v - t  (v - c)   |  c = t + v  (c - v)
 *   xor:  c = t ^ v            mul:  c = t * v^-1 (mod 2^32), needs an odd v
 *   and:  c has a 1 where t has a 1 (v must too), a 0 where v has a 1 and t a 0
 *   or:   c has a 1 where t has a 1 and v a 0, a 0 where t has a 0 (v must too)
 *   shr:  c = clz(t) - clz(v) on a sample with t != 0      shl: c = ctz(t) - ctz(v)
 *   sar:  like shr, counting leading ones when v is negative
 *   rotl/rotr:  the amount is one of 32 values — try them all
 *
 * The same rules run on the GPU (kernel.ts) and here on the CPU (for lengths
 * 1–2 and for re-checking GPU claims).
 */
import { Op } from './isa';
import { clz, ctz } from './u32';

export type FreeKind = 'b' | 'a' | 'shift';

/** Which ops support a synthesized constant, and in which operand position. */
export function freeKind(op: Op): FreeKind | null {
  switch (op.id) {
    case 'add': case 'xor': case 'and': case 'or': case 'mul': return 'b';
    case 'sub': return 'b'; // both v - c and c - v are handled (see solve)
    case 'shl': case 'shr': case 'sar': case 'rotl': case 'rotr': return 'shift';
    default: return null;
  }
}

/** Modular inverse of an odd 32-bit number (Newton iteration). */
export function inv32(v: number): number {
  let x = v; // correct to 3 bits (v*v ≡ 1 mod 8 for odd v)
  for (let i = 0; i < 5; i++) x = Math.imul(x, 2 - Math.imul(v, x)) >>> 0;
  return x >>> 0;
}

export interface FreeSolution {
  /** the constant */
  c: number;
  /** 'b': op(v, c); 'a': op(c, v) */
  pos: 'a' | 'b';
}

/**
 * Find a constant such that op(v[s], c) (or op(c, v[s])) equals t[s] on every
 * sample. Returns null when none exists.
 */
export function solveFreeConstant(op: Op, v: ArrayLike<number>, t: ArrayLike<number>): FreeSolution | null {
  const n = v.length;
  const check = (c: number, pos: 'a' | 'b'): FreeSolution | null => {
    for (let s = 0; s < n; s++) {
      const r = pos === 'b' ? op.fn(v[s], c) : op.fn(c, v[s]);
      if ((r >>> 0) !== (t[s] >>> 0)) return null;
    }
    return { c: c >>> 0, pos };
  };
  switch (op.id) {
    case 'add': return check((t[0] - v[0]) >>> 0, 'b');
    case 'xor': return check((t[0] ^ v[0]) >>> 0, 'b');
    case 'sub': return check((v[0] - t[0]) >>> 0, 'b') ?? check((t[0] + v[0]) >>> 0, 'a');
    case 'mul': {
      let s0 = -1;
      for (let s = 0; s < n; s++) if (v[s] & 1) { s0 = s; break; }
      if (s0 < 0) return null;
      return check(Math.imul(t[s0], inv32(v[s0])) >>> 0, 'b');
    }
    case 'and': {
      let must1 = 0, must0 = 0;
      for (let s = 0; s < n; s++) {
        if ((t[s] & ~v[s]) !== 0) return null; // a 1 in t where v has a 0
        must1 |= v[s] & t[s];
        must0 |= v[s] & ~t[s];
      }
      if ((must1 & must0) !== 0) return null;
      return check(must1 >>> 0, 'b');
    }
    case 'or': {
      let must1 = 0, must0 = 0;
      for (let s = 0; s < n; s++) {
        if ((v[s] & ~t[s]) !== 0) return null; // a 1 in v where t has a 0
        must1 |= t[s] & ~v[s];
        must0 |= ~t[s];
      }
      if ((must1 & must0) !== 0) return null;
      return check(must1 >>> 0, 'b');
    }
    case 'shr': {
      // t = v >> c: for a sample with t != 0 the amount is the change in leading zeros
      for (let s = 0; s < n; s++) if (t[s] !== 0) { const c = clz(t[s]) - clz(v[s]); return c >= 0 && c < 32 ? check(c, 'b') : null; }
      return check(31, 'b');
    }
    case 'shl': {
      for (let s = 0; s < n; s++) if (t[s] !== 0) { const c = ctz(t[s]) - ctz(v[s]); return c >= 0 && c < 32 ? check(c, 'b') : null; }
      return check(31, 'b');
    }
    case 'sar': {
      for (let s = 0; s < n; s++) {
        if (t[s] === 0 || (t[s] >>> 0) === 0xffffffff) continue;
        const neg = (v[s] & 0x80000000) !== 0;
        const c = neg ? clz(~t[s] >>> 0) - clz(~v[s] >>> 0) : clz(t[s]) - clz(v[s]);
        return c >= 0 && c < 32 ? check(c, 'b') : null;
      }
      return check(31, 'b');
    }
    case 'rotl': case 'rotr': {
      for (let c = 0; c < 32; c++) { const r = check(c, 'b'); if (r) return r; }
      return null;
    }
    default: return null;
  }
}

/** Number of "programs" a free-constant attempt counts as (for the evaluated counter). */
export function freeAttempts(op: Op): number {
  switch (freeKind(op)) {
    case 'shift': return op.id === 'rotl' || op.id === 'rotr' ? 32 : 1;
    case 'b': case 'a': return op.id === 'sub' ? 2 : 1;
    default: return 0;
  }
}
