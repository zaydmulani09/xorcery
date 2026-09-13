/**
 * The spec language: a small C-flavoured expression language over 32-bit
 * words, used to describe the function to be synthesised.
 *
 *   t = x & (x - 1);            // statements bind temporaries
 *   (int)x < 0 ? -x : x         // the last expression is the result
 *
 * Every value is a 32-bit word. Signedness is a *static* tag that only
 * changes the meaning of `>>`, `<`, `<=`, `>`, `>=`, `/` and `%`, exactly as
 * in C: `(int)x >> 31` is an arithmetic shift, `x >> 31` is logical. Literals
 * are untyped and adopt the signedness of the other operand; mixing signed and
 * unsigned yields unsigned (C's usual arithmetic conversions). `>>>` is always
 * logical, `sar()` always arithmetic, and every ISA op is also callable as a
 * function (`slt(a, b)`, `popcnt(a)`, `mulhi(a, b)`, ...), so nothing depends
 * on remembering the conversion rules.
 *
 * The parser produces an AST; `compile.ts` turns it into JavaScript (for the
 * CPU) and WGSL (for the GPU). Both compilers share the same semantics table,
 * and the runtime cross-checks them against each other.
 */

export type Sign = 'S' | 'U' | 'N';

export type Node =
  | { k: 'num'; v: number; sign: Sign }
  | { k: 'var'; name: string; sign: Sign }
  | { k: 'un'; op: '~' | '-' | '!'; e: Node; sign: Sign }
  | { k: 'bin'; op: string; a: Node; b: Node; sign: Sign }
  | { k: 'call'; fn: string; args: Node[]; sign: Sign }
  | { k: 'cond'; c: Node; a: Node; b: Node; sign: Sign }
  | { k: 'cast'; to: 'S' | 'U'; e: Node; sign: Sign };

export interface Stmt {
  name: string;
  e: Node;
  declSign?: 'S' | 'U';
}

export interface SpecAst {
  stmts: Stmt[];
  result: Node;
  /** which of x, y, z are referenced */
  inputs: boolean[];
}

export class SpecError extends Error {
  constructor(msg: string, readonly pos: number) {
    super(msg);
  }
}

/** Intrinsic functions: name -> arity. Every ISA op id is included. */
export const INTRINSICS: Record<string, number> = {
  add: 2, sub: 2, and: 2, or: 2, xor: 2, shl: 2, shr: 2, sar: 2, not: 1, neg: 1,
  mul: 2, mulhi: 2, mulhs: 2, clz: 1, ctz: 1, popcnt: 1, popcount: 1,
  eq: 2, ne: 2, ult: 2, ule: 2, ugt: 2, uge: 2, slt: 2, sle: 2, sgt: 2, sge: 2,
  rotl: 2, rotr: 2, bswap: 1, brev: 1,
  umin: 2, umax: 2, smin: 2, smax: 2, min: 2, max: 2,
  udiv: 2, urem: 2, sdiv: 2, srem: 2,
  abs: 1, sign: 1, select: 3, bool: 1,
};

// ---------------------------------------------------------------- lexer ---

type Tok = { t: 'num'; v: number; pos: number } | { t: 'id'; v: string; pos: number } | { t: 'op'; v: string; pos: number } | { t: 'eof'; pos: number };

const OPS3 = ['>>>'];
const OPS2 = ['<<', '>>', '<=', '>=', '==', '!=', '&&', '||'];
const OPS1 = '+-*/%&|^~!<>?:()=;,';

export function lex(src: string): Tok[] {
  const toks: Tok[] = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') { const e = src.indexOf('*/', i + 2); i = e < 0 ? n : e + 2; continue; }
    if (/[0-9]/.test(c)) {
      const start = i;
      let v: number;
      if (c === '0' && (src[i + 1] === 'x' || src[i + 1] === 'X')) {
        i += 2; const s = i;
        while (i < n && /[0-9a-fA-F_]/.test(src[i])) i++;
        const digits = src.slice(s, i).replace(/_/g, '');
        if (!digits) throw new SpecError('bad hex literal', start);
        v = Number(BigInt('0x' + digits) & 0xffffffffn);
        if (BigInt('0x' + digits) > 0xffffffffn) throw new SpecError('literal does not fit in 32 bits', start);
      } else if (c === '0' && (src[i + 1] === 'b' || src[i + 1] === 'B')) {
        i += 2; const s = i;
        while (i < n && /[01_]/.test(src[i])) i++;
        const digits = src.slice(s, i).replace(/_/g, '');
        if (!digits) throw new SpecError('bad binary literal', start);
        if (BigInt('0b' + digits) > 0xffffffffn) throw new SpecError('literal does not fit in 32 bits', start);
        v = Number(BigInt('0b' + digits));
      } else {
        const s = i;
        while (i < n && /[0-9_]/.test(src[i])) i++;
        const digits = src.slice(s, i).replace(/_/g, '');
        if (BigInt(digits) > 0xffffffffn) throw new SpecError('literal does not fit in 32 bits', start);
        v = Number(digits);
      }
      // C-style suffixes
      while (i < n && /[uUlL]/.test(src[i])) i++;
      toks.push({ t: 'num', v: v >>> 0, pos: start });
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      const s = i;
      while (i < n && /[A-Za-z0-9_]/.test(src[i])) i++;
      toks.push({ t: 'id', v: src.slice(s, i), pos: s });
      continue;
    }
    const three = src.slice(i, i + 3), two = src.slice(i, i + 2);
    if (OPS3.includes(three)) { toks.push({ t: 'op', v: three, pos: i }); i += 3; continue; }
    if (OPS2.includes(two)) { toks.push({ t: 'op', v: two, pos: i }); i += 2; continue; }
    if (OPS1.includes(c)) { toks.push({ t: 'op', v: c, pos: i }); i++; continue; }
    throw new SpecError(`unexpected character '${c}'`, i);
  }
  toks.push({ t: 'eof', pos: n });
  return toks;
}

// --------------------------------------------------------------- parser ---

const TYPE_WORDS: Record<string, 'S' | 'U' | null> = {
  int: 'S', int32_t: 'S', i32: 'S', signed: 'S', long: 'S',
  uint: 'U', uint32_t: 'U', u32: 'U', unsigned: 'U', uint32: 'U',
  let: null, const: null, var: null, auto: null,
};

function joinSign(a: Sign, b: Sign): Sign {
  if (a === 'U' || b === 'U') return 'U';
  if (a === 'S' || b === 'S') return 'S';
  return 'N';
}

export function parseSpec(src: string): SpecAst {
  const toks = lex(src);
  let p = 0;
  const peek = () => toks[p];
  const next = () => toks[p++];
  const isOp = (v: string) => { const t = peek(); return t.t === 'op' && t.v === v; };
  const expect = (v: string) => { if (!isOp(v)) throw new SpecError(`expected '${v}'`, peek().pos); p++; };

  const scope = new Map<string, Sign>();
  const inputs = [false, false, false];
  const stmts: Stmt[] = [];

  function primary(): Node {
    const t = next();
    if (t.t === 'num') return { k: 'num', v: t.v, sign: 'N' };
    if (t.t === 'id') {
      if (isOp('(')) {
        p++;
        const args: Node[] = [];
        if (!isOp(')')) {
          for (;;) { args.push(expr()); if (isOp(',')) { p++; continue; } break; }
        }
        expect(')');
        const arity = INTRINSICS[t.v];
        if (arity === undefined) throw new SpecError(`unknown function '${t.v}'`, t.pos);
        if (args.length !== arity) throw new SpecError(`'${t.v}' takes ${arity} argument${arity === 1 ? '' : 's'}`, t.pos);
        const signed = /^(s|abs$|sign$|mulhs)/.test(t.v) && !/^(sub|shl|shr|select|sar)$/.test(t.v);
        return { k: 'call', fn: t.v, args, sign: signed ? 'S' : 'U' };
      }
      const idx = ['x', 'y', 'z'].indexOf(t.v);
      if (idx >= 0 && !scope.has(t.v)) { inputs[idx] = true; return { k: 'var', name: t.v, sign: 'U' }; }
      const s = scope.get(t.v);
      if (s === undefined) throw new SpecError(`unknown variable '${t.v}' (inputs are x, y, z)`, t.pos);
      return { k: 'var', name: t.v, sign: s };
    }
    if (t.t === 'op' && t.v === '(') {
      // cast?
      const t2 = peek();
      if (t2.t === 'id' && TYPE_WORDS[t2.v] !== undefined && TYPE_WORDS[t2.v] !== null) {
        const to = TYPE_WORDS[t2.v] as 'S' | 'U';
        p++;
        // allow "(unsigned int)"
        while (peek().t === 'id' && TYPE_WORDS[(peek() as any).v] !== undefined) p++;
        expect(')');
        const e = unary();
        return { k: 'cast', to, e, sign: to };
      }
      const e = expr();
      expect(')');
      return e;
    }
    throw new SpecError(t.t === 'eof' ? 'unexpected end of input' : `unexpected '${(t as any).v}'`, t.pos);
  }

  function unary(): Node {
    const t = peek();
    if (t.t === 'op' && (t.v === '~' || t.v === '-' || t.v === '!')) {
      p++;
      const e = unary();
      return { k: 'un', op: t.v as '~' | '-' | '!', e, sign: t.v === '!' ? 'U' : e.sign };
    }
    return primary();
  }

  const levels: string[][] = [
    ['*', '/', '%'], ['+', '-'], ['<<', '>>', '>>>'], ['<', '<=', '>', '>='], ['==', '!='], ['&'], ['^'], ['|'], ['&&'], ['||'],
  ];
  function binLevel(i: number): Node {
    if (i < 0) return unary();
    let a = binLevel(i - 1);
    for (;;) {
      const t = peek();
      if (t.t === 'op' && levels[i].includes(t.v)) {
        p++;
        const b = binLevel(i - 1);
        let sign: Sign;
        if (['<', '<=', '>', '>=', '==', '!=', '&&', '||'].includes(t.v)) sign = 'U';
        else if (t.v === '<<' || t.v === '>>' || t.v === '>>>') sign = t.v === '>>>' ? 'U' : a.sign;
        else sign = joinSign(a.sign, b.sign);
        a = { k: 'bin', op: t.v, a, b, sign };
      } else return a;
    }
  }
  function expr(): Node {
    const c = binLevel(levels.length - 1);
    if (isOp('?')) {
      p++;
      const a = expr();
      expect(':');
      const b = expr();
      return { k: 'cond', c, a, b, sign: joinSign(a.sign, b.sign) };
    }
    return c;
  }

  // statements: [type] name = expr ;   |  return expr ;  |  expr
  let result: Node | null = null;
  while (peek().t !== 'eof') {
    if (result) throw new SpecError('unexpected code after the result expression', peek().pos);
    const t = peek();
    if (t.t === 'id' && t.v === 'return') { p++; result = expr(); if (isOp(';')) p++; continue; }
    // declaration?
    let declSign: 'S' | 'U' | undefined;
    let q = p;
    while (toks[q].t === 'id' && TYPE_WORDS[(toks[q] as any).v] !== undefined) {
      const s = TYPE_WORDS[(toks[q] as any).v];
      if (s) declSign = s;
      q++;
    }
    const nameTok = toks[q];
    const eqTok = toks[q + 1];
    if (nameTok.t === 'id' && eqTok.t === 'op' && eqTok.v === '=') {
      p = q + 2;
      const e = expr();
      if (isOp(';')) p++; else if (peek().t !== 'eof') throw new SpecError("expected ';'", peek().pos);
      const name = nameTok.v;
      if (INTRINSICS[name] !== undefined) throw new SpecError(`'${name}' is a built-in function`, nameTok.pos);
      const sign: Sign = declSign ?? (e.sign === 'N' ? 'U' : e.sign);
      scope.set(name, sign);
      stmts.push({ name, e, declSign });
      continue;
    }
    result = expr();
    if (isOp(';')) p++;
  }
  if (!result) throw new SpecError('the spec needs a result expression', toks[toks.length - 1].pos);
  return { stmts, result, inputs };
}
