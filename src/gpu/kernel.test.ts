import { describe, expect, it } from 'vitest';
import { OPS, opsForGroups } from '../core/isa';
import { Space, evalProgram } from '../core/space';
import { FREE, generateKernel, buildLayout, kernelKey } from './kernel';
import { resolveFree } from './search';

describe('kernel generator', () => {
  it('emits one unrolled block per op plus synthesized-constant blocks, deterministically', () => {
    const ops = opsForGroups(['base', 'bits']);
    const code = generateKernel(ops);
    for (const op of ops) expect(code).toContain(`// ---- ${op.id} ----`);
    for (const id of ['add', 'sub', 'xor', 'and', 'or', 'shl', 'shr', 'sar']) expect(code).toContain(`// ---- ${id} with a synthesized constant ----`);
    expect(code).not.toContain('countOneBits(~'); // the driver-bug pattern must never be emitted
    expect(code).toContain('fn popcnt32');
    expect(generateKernel(ops)).toBe(code);
    expect(kernelKey(ops)).toBe(ops.map((o) => o.id).join(','));
  });
  it('builds layout tables whose per-slot bases and counts cover every digit', () => {
    const cfg = { nInputs: 2, consts: [0, 1], ops: OPS };
    const space = new Space(cfg, 4);
    const lay = buildLayout(space, 4);
    for (let slot = 0; slot < 4; slot++) {
      let total = 0;
      for (let oi = 0; oi < OPS.length; oi++) {
        const e = (slot * OPS.length + oi) * 4;
        expect(lay.layout[e]).toBe(total);
        total += lay.layout[e + 1];
      }
      expect(total).toBe(space.slotCount(slot));
    }
  });
});

describe('resolveFree', () => {
  const cfg = { nInputs: 1, consts: [0, 1], ops: opsForGroups(['base']) };
  const and = cfg.ops.findIndex((o) => o.id === 'and');
  const shr = cfg.ops.findIndex((o) => o.id === 'shr');
  const xor = cfg.ops.findIndex((o) => o.id === 'xor');
  it('appends a new constant and shifts result indices', () => {
    // r0 = x >> 1; r1 = x ^ r0; r2 = r1 & FREE   (value indices: x=0, c0=1, c1=2, r0=3, r1=4)
    const prog = [{ op: shr, a: 0, b: 2 }, { op: xor, a: 3, b: 0 }, { op: and, a: 4, b: FREE }];
    const { program, cfg: ext } = resolveFree(cfg, prog, 0x55555555);
    expect(ext.consts).toEqual([0, 1, 0x55555555]);
    expect(program).toEqual([{ op: shr, a: 0, b: 2 }, { op: xor, a: 4, b: 0 }, { op: and, a: 5, b: 3 }]);
    expect(evalProgram(ext, program, [0xaaaaaaaa])).toBe(0x55555555);
  });
  it('reuses a pool constant when the synthesized value is already there', () => {
    const prog = [{ op: shr, a: 0, b: 2 }, { op: and, a: 3, b: FREE }];
    const { program, cfg: same } = resolveFree(cfg, prog, 1);
    expect(same).toBe(cfg);
    expect(program).toEqual([{ op: shr, a: 0, b: 2 }, { op: and, a: 3, b: 2 }]);
  });
});
