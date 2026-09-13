import { describe, expect, it } from 'vitest';
import * as U from './u32';
import { OPS, opsForGroups } from './isa';
import { Space, SpaceConfig, deadCodeFreeCount, enumeratePrograms, evalProgram, isDeadCodeFree, Program } from './space';
import { compileToJs, emitFunction, exprString, printProgram } from './program';

describe('u32 semantics', () => {
  it('shifts mask the amount by 31', () => {
    expect(U.shl(1, 33)).toBe(2);
    expect(U.shr(0x80000000, 32)).toBe(0x80000000);
    expect(U.sar(0x80000000, 31)).toBe(0xffffffff);
    expect(U.sar(0x80000000, 63)).toBe(0xffffffff);
  });
  it('clz/ctz/popcnt agree with the reference definitions', () => {
    expect(U.clz(0)).toBe(32);
    expect(U.ctz(0)).toBe(32);
    expect(U.clz(1)).toBe(31);
    expect(U.ctz(0x80000000)).toBe(31);
    expect(U.popcnt(0xffffffff)).toBe(32);
    expect(U.popcnt(0x80000001)).toBe(2);
    for (let i = 0; i < 2000; i++) {
      const x = U.hash32(i);
      let pc = 0, t = x;
      while (t) { pc += t & 1; t >>>= 1; }
      expect(U.popcnt(x)).toBe(pc);
      expect(U.ctz(x)).toBe(x === 0 ? 32 : x.toString(2).length - x.toString(2).replace(/0+$/, '').length);
    }
  });
  it('mulhi matches BigInt', () => {
    for (let i = 0; i < 5000; i++) {
      const a = U.hash32(i), b = U.hash32(i * 7 + 1);
      const ref = Number((BigInt(a) * BigInt(b)) >> 32n);
      expect(U.mulhi(a, b)).toBe(ref);
      const sref = Number(((BigInt(a | 0) * BigInt(b | 0)) >> 32n) & 0xffffffffn);
      expect(U.mulhs(a, b)).toBe(sref);
    }
    expect(U.mulhi(0xffffffff, 0xffffffff)).toBe(0xfffffffe);
  });
  it('rotates, bswap and brev', () => {
    expect(U.rotl(0x80000001, 1)).toBe(3);
    expect(U.rotr(3, 1)).toBe(0x80000001);
    expect(U.rotl(5, 32)).toBe(5);
    expect(U.bswap(0x12345678)).toBe(0x78563412);
    expect(U.brev(1)).toBe(0x80000000);
    expect(U.brev(0x0000ffff)).toBe(0xffff0000);
    for (let i = 0; i < 200; i++) {
      const x = U.hash32(i);
      expect(U.brev(U.brev(x))).toBe(x);
      expect(U.bswap(U.bswap(x))).toBe(x);
    }
  });
  it('division is total (RISC-V convention)', () => {
    expect(U.udiv(7, 0)).toBe(0xffffffff);
    expect(U.urem(7, 0)).toBe(7);
    expect(U.udiv(0xffffffff, 2)).toBe(0x7fffffff);
    expect(U.urem(0xffffffff, 10)).toBe(5);
  });
  it('every op returns a canonical u32', () => {
    for (const op of OPS) {
      for (let i = 0; i < 50; i++) {
        const v = op.fn(U.hash32(i), U.hash32(i + 1000));
        expect(v).toBe(v >>> 0);
        expect(Number.isInteger(v)).toBe(true);
      }
    }
  });
});

function cfgOf(nInputs: number, consts: number[], groups: string[]): SpaceConfig {
  return { nInputs, consts, ops: opsForGroups(groups as any) };
}

describe('program space', () => {
  const cfg = cfgOf(1, [0, 1, 31], ['base']);
  const space = new Space(cfg, 4);

  it('digits round-trip through decode/encode', () => {
    for (let slot = 0; slot < 4; slot++) {
      const c = space.slotCount(slot);
      for (let d = 0; d < c; d++) {
        const ins = space.decodeDigit(slot, d);
        expect(space.encodeDigit(slot, ins)).toBe(d);
      }
    }
  });
  it('program ids round-trip', () => {
    for (let L = 1; L <= 4; L++) {
      const total = space.rawCount(L);
      for (let i = 0; i < 200; i++) {
        const id = BigInt(U.hash32(i)) % total;
        const prog = space.decode(L, id);
        expect(space.encode(prog)).toBe(id);
      }
    }
  });
  it('pair tables never contain constant-constant pairs or excluded self pairs', () => {
    for (let slot = 0; slot < 4; slot++) {
      const lay = space.layouts[slot];
      for (let d = 0; d < lay.count; d++) {
        const ins = space.decodeDigit(slot, d);
        const op = cfg.ops[ins.op];
        const isC = (v: number) => v >= 1 && v < 4;
        if (op.kind === 'unary') {
          expect(isC(ins.a)).toBe(false);
        } else {
          expect(isC(ins.a) && isC(ins.b)).toBe(false);
          if (!op.self) expect(ins.a).not.toBe(ins.b);
          if (op.kind === 'comm') expect(ins.a).toBeGreaterThanOrEqual(ins.b);
        }
      }
    }
  });
  it('enumeratePrograms yields exactly the dead-code-free programs, in id order', () => {
    for (let L = 1; L <= 3; L++) {
      let count = 0;
      let lastId = -1n;
      let bad = 0;
      for (const p of enumeratePrograms(space, L)) {
        if (!isDeadCodeFree(cfg, p)) bad++;
        const id = space.encode(p);
        if (!(id > lastId)) bad++;
        lastId = id;
        count++;
      }
      expect(bad).toBe(0);
      expect(count).toBeGreaterThan(0);
      if (L <= 2) {
        // brute-force count over all raw ids
        let brute = 0;
        const total = Number(space.rawCount(L));
        for (let id = 0; id < total; id++) if (isDeadCodeFree(cfg, space.decode(L, BigInt(id)))) brute++;
        expect(count).toBe(brute);
      }
    }
    // known count for this configuration (1 input, consts [0,1,31], base ops), length 3
    let n3 = 0;
    for (const _ of enumeratePrograms(space, 3)) n3++;
    expect(n3).toBe(170856);
  }, 20000);
  it('finds the classic x & (x - 1) by CPU enumeration at length 2', () => {
    const target = (x: number) => U.and(x, U.sub(x, 1));
    const samples = [0, 1, 2, 3, 0x80000000, 0xffffffff, 12345678, 0x7fffffff];
    let found: Program | null = null;
    for (const p of enumeratePrograms(space, 2)) {
      if (samples.every((x) => evalProgram(cfg, p, [x]) === target(x))) { found = p; break; }
    }
    expect(found).not.toBeNull();
    expect(exprString(cfg, found!)).toBe('x & (x - 1)');
  });
});

describe('printer / emitters', () => {
  const allCfg: SpaceConfig = { nInputs: 2, consts: [0, 1, 31, 0xffffffff, 0x80000000], ops: OPS };
  const space = new Space(allCfg, 5);

  it('emitted JavaScript computes the same values as the interpreter (all ops, random programs)', () => {
    let checked = 0;
    for (let L = 1; L <= 5; L++) {
      const total = space.rawCount(L);
      for (let i = 0; i < 120; i++) {
        const id = (BigInt(U.hash32(i + L * 100000)) * BigInt(U.hash32(i))) % total;
        const prog = space.decode(L, id);
        const f = compileToJs(allCfg, prog);
        for (let j = 0; j < 8; j++) {
          const x = U.hash32(j + 77), y = U.hash32(j + 991);
          expect(f(x, y)).toBe(evalProgram(allCfg, prog, [x, y]));
          checked++;
        }
      }
    }
    expect(checked).toBeGreaterThan(1000);
  });
  it('inlines single-use results and names multi-use ones', () => {
    const cfg = cfgOf(1, [31], ['base']);
    // abs: r0 = x sar 31; r1 = x xor r0; r2 = r1 - r0
    const sar = cfg.ops.findIndex((o) => o.id === 'sar');
    const xor = cfg.ops.findIndex((o) => o.id === 'xor');
    const sub = cfg.ops.findIndex((o) => o.id === 'sub');
    const prog: Program = [{ op: sar, a: 0, b: 1 }, { op: xor, a: 2, b: 0 }, { op: sub, a: 3, b: 2 }];
    expect(exprString(cfg, prog)).toBe('t0 = (int)x >> 31; (x ^ t0) - t0');
    const p = printProgram(cfg, prog, 'rust');
    expect(p.temps.length).toBe(1);
    expect(p.expr).toBe('(x ^ t0).wrapping_sub(t0)');
    const c = emitFunction(cfg, prog, 'c');
    expect(c).toContain('uint32_t t0 = (uint32_t)((int32_t)x >> 31u);');
    expect(c).toContain('return (x ^ t0) - t0;');
  });
  it('parenthesises mixed operators but not associative chains', () => {
    const cfg = cfgOf(3, [], ['base']);
    const add = cfg.ops.findIndex((o) => o.id === 'add');
    const and = cfg.ops.findIndex((o) => o.id === 'and');
    const shl = cfg.ops.findIndex((o) => o.id === 'shl');
    // ((x + y) + z) & (x << y)
    const prog: Program = [{ op: add, a: 1, b: 0 }, { op: add, a: 3, b: 2 }, { op: shl, a: 0, b: 1 }, { op: and, a: 5, b: 4 }];
    expect(exprString(cfg, prog)).toBe('(x << y) & (z + y + x)');
  });
  it('WGSL and C output mention the helpers they use', () => {
    const cfg = cfgOf(1, [1], ['rot', 'bits']);
    const rotl = cfg.ops.findIndex((o) => o.id === 'rotl');
    const prog: Program = [{ op: rotl, a: 0, b: 1 }];
    expect(emitFunction(cfg, prog, 'wgsl')).toContain('fn rotl32');
    expect(emitFunction(cfg, prog, 'c')).toContain('static inline uint32_t rotl32');
    expect(emitFunction(cfg, prog, 'rust')).toContain('x.rotate_left(1 & 31)');
  });
});

describe('dead-code-free counting', () => {
  it('matches enumeration for several configurations', () => {
    const configs: [SpaceConfig, number, number[]][] = [
      [{ nInputs: 1, consts: [0, 1, 31], ops: opsForGroups(['base']) }, 3, [42, 2268, 170856]],
      [{ nInputs: 1, consts: [0, 1], ops: opsForGroups(['base']) }, 4, [30, 1260, 78840, 7199280]],
      [{ nInputs: 1, consts: [0, 1, 3, 31, 0xffffffff], ops: opsForGroups(['base']) }, 3, [66, 5148, 515592]],
      [{ nInputs: 2, consts: [1], ops: opsForGroups(['base', 'cmp']) }, 4, [63, 3591, 333207, 45339399]],
    ];
    for (const [cfg, maxL, expected] of configs) {
      const space = new Space(cfg, maxL);
      for (let L = 1; L <= maxL; L++) expect(Number(deadCodeFreeCount(space, L))).toBe(expected[L - 1]);
    }
    // and against a fresh enumeration for a config not listed above
    const cfg: SpaceConfig = { nInputs: 2, consts: [0], ops: opsForGroups(['base', 'bits']) };
    const space = new Space(cfg, 3);
    for (let L = 1; L <= 3; L++) {
      let n = 0;
      for (const _ of enumeratePrograms(space, L)) n++;
      expect(Number(deadCodeFreeCount(space, L))).toBe(n);
    }
  });
});
