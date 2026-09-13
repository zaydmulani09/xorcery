import { describe, expect, it } from 'vitest';
import { OPS, OP_BY_ID } from './isa';
import { freeKind, inv32, solveFreeConstant } from './freeconst';
import { hash32 } from './u32';

describe('synthesized constants', () => {
  it('inv32 inverts every odd number tried', () => {
    for (let i = 0; i < 2000; i++) {
      const v = (hash32(i) | 1) >>> 0;
      expect(Math.imul(v, inv32(v)) >>> 0).toBe(1);
    }
  });
  it('recovers a hidden constant for every solvable op', () => {
    const vs = Array.from({ length: 32 }, (_, s) => hash32(s + 11));
    let checked = 0;
    for (const op of OPS) {
      const kind = freeKind(op);
      if (!kind) continue;
      for (let trial = 0; trial < 40; trial++) {
        const hidden = kind === 'shift' ? trial % 32 : hash32(trial * 31 + 7);
        const pos: 'a' | 'b' = op.id === 'sub' && trial % 2 ? 'a' : 'b';
        const ts = vs.map((v) => (pos === 'b' ? op.fn(v, hidden) : op.fn(hidden, v)) >>> 0);
        const sol = solveFreeConstant(op, vs, ts);
        expect(sol).not.toBeNull();
        // the solution need not equal the hidden constant (and/or leave bits free) but must reproduce the targets
        for (let s = 0; s < vs.length; s++) {
          const r = sol!.pos === 'b' ? op.fn(vs[s], sol!.c) : op.fn(sol!.c, vs[s]);
          expect(r >>> 0).toBe(ts[s]);
        }
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(300);
  });
  it('rejects impossible targets', () => {
    const vs = [1, 2, 3, 4, 5, 6, 7, 8];
    const and = OP_BY_ID.get('and')!, or = OP_BY_ID.get('or')!, add = OP_BY_ID.get('add')!;
    expect(solveFreeConstant(and, vs, [1, 2, 3, 4, 5, 6, 7, 9])).toBeNull(); // 9 has a bit 8 lacks
    expect(solveFreeConstant(or, vs, [0, 2, 3, 4, 5, 6, 7, 8])).toBeNull(); // 0 lacks a bit 1 has
    expect(solveFreeConstant(add, vs, [2, 3, 4, 5, 6, 7, 8, 10])).toBeNull();
  });
});
