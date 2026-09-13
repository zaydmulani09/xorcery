import { describe, expect, it } from 'vitest';
import { compileSpec } from './compile';
import { parseSpec } from './parser';
import * as U from '../core/u32';

const run = (src: string, x = 0, y = 0, z = 0) => compileSpec(src).fn(x, y, z);

describe('spec language', () => {
  it('parses C-like expressions with the right precedence', () => {
    expect(run('x & (x - 1)', 12)).toBe(8);
    expect(run('x + y * 2', 1, 3)).toBe(7);
    expect(run('1 << 31 | x')).toBe(0x80000000);
    expect(run('x | y ^ z & 4', 1, 6, 5)).toBe(1 | (6 ^ (5 & 4)));
    expect(run('x << 1 + 1', 1)).toBe(4);
  });
  it('signedness follows C conversions', () => {
    expect(run('x >> 31', 0x80000000)).toBe(1);
    expect(run('(int)x >> 31', 0x80000000)).toBe(0xffffffff);
    expect(run('(int32_t)x >> 31', 0x80000000)).toBe(0xffffffff);
    expect(run('x >>> 31', 0x80000000)).toBe(1);
    expect(run('(int)x < 0', 0x80000000)).toBe(1);
    expect(run('x < 0', 0x80000000)).toBe(0);
    expect(run('(int)x < 0 ? -x : x', 0xfffffffe)).toBe(2);
    expect(run('(unsigned)((int)x >> 1)', 0x80000000)).toBe(0xc0000000);
    // signed + unsigned -> unsigned
    expect(run('((int)x + y) >> 31', 0x80000000, 0)).toBe(1);
    // literal adopts the signedness of the other operand
    expect(run('(int)x / 2', 0xffffffff)).toBe(0);
    expect(run('x / 2', 0xffffffff)).toBe(0x7fffffff);
    expect(run('(int)x % 3', (-7) >>> 0)).toBe((-1) >>> 0);
  });
  it('statements and return', () => {
    expect(run('t = x >> 31; (x ^ t) - t', 0xfffffffb)).toBe(U.sub(U.xor(0xfffffffb, 1), 1));
    expect(run('int s = (int)x >> 31; return (x ^ s) - s;', 0xfffffffb)).toBe(5);
    expect(run('uint32_t a = x + 1;\n uint32_t b = a * 2;\n b - 1', 3)).toBe(7);
  });
  it('intrinsics', () => {
    expect(run('popcnt(x)', 0xf0f0)).toBe(8);
    expect(run('popcount(x)', 0xf0f0)).toBe(8);
    expect(run('clz(x)', 1)).toBe(31);
    expect(run('ctz(x)', 8)).toBe(3);
    expect(run('rotl(x, 4)', 0x12345678)).toBe(0x23456781);
    expect(run('bswap(x)', 0x11223344)).toBe(0x44332211);
    expect(run('abs(x)', 0xffffff00)).toBe(256);
    expect(run('sign(x)', 0xffffff00)).toBe(0xffffffff);
    expect(run('sign(x)', 0)).toBe(0);
    expect(run('mulhi(x, y)', 0xffffffff, 0xffffffff)).toBe(0xfffffffe);
    expect(run('mulhs(x, y)', 0xffffffff, 0xffffffff)).toBe(0);
    expect(run('select(x, y, z)', 0, 5, 9)).toBe(9);
    expect(run('select(x, y, z)', 2, 5, 9)).toBe(5);
    expect(run('umax(x, y)', 3, 0xffffffff)).toBe(0xffffffff);
    expect(run('smax(x, y)', 3, 0xffffffff)).toBe(3);
    expect(run('sar(x, 4)', 0x80000000)).toBe(0xf8000000);
    expect(run('ugt(x, y)', 5, 3)).toBe(1);
    expect(run('sgt(x, y)', 0xffffffff, 3)).toBe(0);
    expect(run('x == y', 7, 7)).toBe(1);
    expect(run('x != y', 7, 7)).toBe(0);
    expect(run('!x', 0)).toBe(1);
    expect(run('x && y', 2, 0)).toBe(0);
    expect(run('x || y', 2, 0)).toBe(1);
  });
  it('detects the inputs used', () => {
    expect(compileSpec('x + 1').nInputs).toBe(1);
    expect(compileSpec('y + 1').nInputs).toBe(2);
    expect(compileSpec('x + z').nInputs).toBe(3);
    expect(() => compileSpec('1 + 2')).toThrow(/input/);
  });
  it('reports errors with positions', () => {
    expect(() => parseSpec('x +')).toThrow(/unexpected end/);
    expect(() => parseSpec('foo(x)')).toThrow(/unknown function/);
    expect(() => parseSpec('x + q')).toThrow(/unknown variable/);
    expect(() => parseSpec('popcnt(x, y)')).toThrow(/takes 1 argument/);
    expect(() => parseSpec('0x1ffffffff')).toThrow(/32 bits/);
    expect(() => parseSpec('x $ 1')).toThrow(/unexpected character/);
  });
  it('hex, binary, underscores and suffixes', () => {
    expect(run('0xFFFF_FFFF ^ x')).toBe(0xffffffff);
    expect(run('0b1010 + x')).toBe(10);
    expect(run('4294967295u - x')).toBe(0xffffffff);
    expect(run('-1 | x')).toBe(0xffffffff);
  });
  it('WGSL output references the helpers it needs', () => {
    const c = compileSpec('rotl(x, 3) + sdiv(x, 2) + abs(x)');
    expect(c.wgsl).toContain('fn rotl32');
    expect(c.wgsl).toContain('fn sdiv32');
    expect(c.wgsl).toContain('fn abs32');
    expect(c.wgsl).toContain('fn spec(x: u32, y: u32, z: u32) -> u32');
    const m = compileSpec('mulhs(x, y)');
    expect(m.wgsl.indexOf('fn mulhi32')).toBeLessThan(m.wgsl.indexOf('fn mulhs32'));
  });
});

describe('spec statements', () => {
  it('allows reassignment of a temporary', () => {
    const c = compileSpec('t = x - 1; t = t | (t >> 1); t = t | (t >> 2); t + 1');
    expect(c.fn(5, 0, 0)).toBe(8);
    expect(c.wgsl).toContain('var v_t: u32 =');
    expect((c.wgsl.match(/var v_t/g) ?? []).length).toBe(1);
  });
  it('a temporary may shadow an input name', () => {
    const c = compileSpec('x = x + 1; x * 2');
    expect(c.fn(3, 0, 0)).toBe(8);
  });
});
