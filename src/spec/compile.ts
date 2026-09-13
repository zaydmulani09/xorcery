/**
 * Compile a spec AST to JavaScript (CPU reference) and WGSL (GPU reference).
 *
 * Both back-ends are driven by the same table of intrinsic templates, most of
 * which are the ISA templates from `core/isa.ts`, so the CPU and the GPU
 * cannot quietly disagree about what `>>` or `/` means.
 */
import { HELPERS, HELPER_DEPS, OP_BY_ID } from '../core/isa';
import * as U from '../core/u32';
import { Node, SpecAst, SpecError, parseSpec } from './parser';

type Lang = 'js' | 'wgsl';

interface Tpl { t: string; helper?: string }

/** Extra helpers beyond the ISA ones (signed division, abs, sign, mulhs). */
const EXTRA_HELPERS: Record<Lang, Record<string, string>> = {
  js: {
    sdiv: 'const sdiv32 = (a, b) => { a |= 0; b |= 0; if (b === 0) return 0xffffffff; if (a === -2147483648 && b === -1) return 0x80000000; return ((a / b) | 0) >>> 0; };',
    srem: 'const srem32 = (a, b) => { a |= 0; b |= 0; if (b === 0) return a >>> 0; if (a === -2147483648 && b === -1) return 0; return (a % b) >>> 0; };',
    abs: 'const abs32 = (a) => ((a | 0) < 0 ? -a : a) >>> 0;',
    sign: 'const sign32 = (a) => ((a | 0) < 0 ? 0xffffffff : a === 0 ? 0 : 1);',
    mulhs: 'const mulhs32 = (a, b) => Number(((BigInt(a | 0) * BigInt(b | 0)) >> 32n) & 0xffffffffn);',
  },
  wgsl: {
    sdiv: 'fn sdiv32(a: u32, b: u32) -> u32 { let ia = bitcast<i32>(a); let ib = bitcast<i32>(b); if (ib == 0) { return 0xffffffffu; } if (a == 0x80000000u && ib == -1) { return a; } return bitcast<u32>(ia / ib); }',
    srem: 'fn srem32(a: u32, b: u32) -> u32 { let ia = bitcast<i32>(a); let ib = bitcast<i32>(b); if (ib == 0) { return a; } if (a == 0x80000000u && ib == -1) { return 0u; } return bitcast<u32>(ia % ib); }',
    abs: 'fn abs32(a: u32) -> u32 { return bitcast<u32>(abs(bitcast<i32>(a))); }',
    sign: 'fn sign32(a: u32) -> u32 { return bitcast<u32>(sign(bitcast<i32>(a))); }',
    mulhs: 'fn mulhs32(a: u32, b: u32) -> u32 { var h = mulhi32(a, b); if ((a & 0x80000000u) != 0u) { h = h - b; } if ((b & 0x80000000u) != 0u) { h = h - a; } return h; }',
  },
};
const EXTRA_DEPS: Record<string, string[]> = { mulhs: ['mulhi'] };

function isaTpl(id: string, lang: Lang): Tpl {
  const op = OP_BY_ID.get(id)!;
  const t = op[lang];
  return { t: t.t, helper: t.helper };
}

const CALL_ALIASES: Record<string, string> = {
  popcount: 'popcnt', min: 'umin', max: 'umax',
};

function callTpl(fn: string, lang: Lang): Tpl {
  fn = CALL_ALIASES[fn] ?? fn;
  if (OP_BY_ID.has(fn)) return isaTpl(fn, lang);
  switch (fn) {
    case 'ugt': return swap(isaTpl('ult', lang));
    case 'uge': return swap(isaTpl('ule', lang));
    case 'sgt': return swap(isaTpl('slt', lang));
    case 'sge': return swap(isaTpl('sle', lang));
    case 'sdiv': return { t: 'sdiv32($a, $b)', helper: 'sdiv' };
    case 'srem': return { t: 'srem32($a, $b)', helper: 'srem' };
    case 'abs': return { t: 'abs32($a)', helper: 'abs' };
    case 'sign': return { t: 'sign32($a)', helper: 'sign' };
    case 'mulhs': return { t: 'mulhs32($a, $b)', helper: 'mulhs' };
    case 'bool': return lang === 'js' ? { t: '(+($a !== 0))' } : { t: 'u32($a != 0u)' };
    case 'select': return lang === 'js' ? { t: '(($c !== 0) ? $a : $b)' } : { t: 'select($b, $a, $c != 0u)' };
  }
  throw new Error(`no template for ${fn}`);
}
function swap(t: Tpl): Tpl {
  return { t: t.t.replace('$a', '$TMP').replace('$b', '$a').replace('$TMP', '$b'), helper: t.helper };
}

function lit(v: number, lang: Lang): string {
  v >>>= 0;
  return lang === 'js' ? (v < 256 ? String(v) : '0x' + v.toString(16)) : (v < 256 ? `${v}u` : `0x${v.toString(16)}u`);
}

export class Emitter {
  helpers = new Set<string>();
  /** user variables (statement names) — emitted with a v_ prefix to dodge keywords and inputs */
  locals = new Set<string>();
  constructor(readonly lang: Lang) {}

  /**
   * Substitute operands into a template. Every non-atomic operand is wrapped
   * in parentheses: WGSL refuses to mix `&` with `-` (or `<<` with `+`)
   * without them, and the output is for machines, not for reading.
   */
  private use(t: Tpl, a: string, b?: string, c?: string): string {
    if (t.helper) this.helpers.add(t.helper);
    const paren = (x: string) => (/^[A-Za-z_][A-Za-z0-9_]*$|^(0x[0-9a-fA-F]+|\d+)u?$/.test(x) ? x : `(${x})`);
    let s = t.t.replace('$a', paren(a));
    if (b !== undefined) s = s.replace('$b', paren(b));
    if (c !== undefined) s = s.replace('$c', paren(c));
    return s;
  }

  expr(n: Node): string {
    const L = this.lang;
    switch (n.k) {
      case 'num': return lit(n.v, L);
      case 'var': return ['x', 'y', 'z'].includes(n.name) && !this.locals.has(n.name) ? n.name : 'v_' + n.name;
      case 'cast': return this.expr(n.e);
      case 'un': {
        const e = this.expr(n.e);
        if (n.op === '~') return this.use(isaTpl('not', L), e);
        if (n.op === '-') return this.use(isaTpl('neg', L), e);
        return this.use(isaTpl('eq', L), e, lit(0, L));
      }
      case 'cond': {
        return this.use(callTpl('select', L), this.expr(n.a), this.expr(n.b), this.expr(n.c));
      }
      case 'call': {
        const args = n.args.map((a) => this.expr(a));
        if (n.fn === 'select') return this.use(callTpl('select', L), args[1], args[2], args[0]);
        return this.use(callTpl(n.fn, L), args[0], args[1]);
      }
      case 'bin': {
        const a = this.expr(n.a), b = this.expr(n.b);
        const signed = n.sign === 'S';
        const cmpSigned = joinIsSigned(n.a.sign, n.b.sign);
        switch (n.op) {
          case '+': return this.use(isaTpl('add', L), a, b);
          case '-': return this.use(isaTpl('sub', L), a, b);
          case '*': return this.use(isaTpl('mul', L), a, b);
          case '&': return this.use(isaTpl('and', L), a, b);
          case '|': return this.use(isaTpl('or', L), a, b);
          case '^': return this.use(isaTpl('xor', L), a, b);
          case '<<': return this.use(isaTpl('shl', L), a, b);
          case '>>>': return this.use(isaTpl('shr', L), a, b);
          case '>>': return this.use(isaTpl(n.a.sign === 'S' ? 'sar' : 'shr', L), a, b);
          case '/': return this.use(signed ? callTpl('sdiv', L) : isaTpl('udiv', L), a, b);
          case '%': return this.use(signed ? callTpl('srem', L) : isaTpl('urem', L), a, b);
          case '<': return this.use(isaTpl(cmpSigned ? 'slt' : 'ult', L), a, b);
          case '<=': return this.use(isaTpl(cmpSigned ? 'sle' : 'ule', L), a, b);
          case '>': return this.use(isaTpl(cmpSigned ? 'slt' : 'ult', L), b, a);
          case '>=': return this.use(isaTpl(cmpSigned ? 'sle' : 'ule', L), b, a);
          case '==': return this.use(isaTpl('eq', L), a, b);
          case '!=': return this.use(isaTpl('ne', L), a, b);
          case '&&': return this.use(isaTpl('and', L), this.use(callTpl('bool', L), a), this.use(callTpl('bool', L), b));
          case '||': return this.use(isaTpl('or', L), this.use(callTpl('bool', L), a), this.use(callTpl('bool', L), b));
        }
        throw new Error(`unknown operator ${n.op}`);
      }
    }
  }

  helperSource(): string[] {
    const all = new Set(this.helpers);
    for (const h of Array.from(all)) {
      for (const d of HELPER_DEPS[h] ?? []) all.add(d);
      for (const d of EXTRA_DEPS[h] ?? []) all.add(d);
    }
    const src: string[] = [];
    for (const h of Object.keys(HELPERS[this.lang])) if (all.has(h)) src.push(HELPERS[this.lang][h]);
    for (const h of Object.keys(EXTRA_HELPERS[this.lang])) if (all.has(h)) src.push(EXTRA_HELPERS[this.lang][h]);
    return src;
  }
}

function joinIsSigned(a: string, b: string): boolean {
  if (a === 'U' || b === 'U') return false;
  return a === 'S' || b === 'S';
}

export interface CompiledSpec {
  ast: SpecAst;
  nInputs: number;
  /** the reference function (u32 in, u32 out) */
  fn: (x: number, y: number, z: number) => number;
  /** JavaScript source of the reference function */
  jsSource: string;
  /** WGSL source: helpers + `fn spec(x: u32, y: u32, z: u32) -> u32` */
  wgsl: string;
}

export function emitJs(ast: SpecAst, fnName = 'spec'): string {
  const em = new Emitter('js');
  const body: string[] = [];
  for (const s of ast.stmts) { const e = em.expr(s.e); const first = !em.locals.has(s.name); em.locals.add(s.name); body.push(`  ${first ? 'let ' : ''}v_${s.name} = ${e};`); }
  const result = em.expr(ast.result);
  return [...em.helperSource(), `function ${fnName}(x, y, z) {`, '  x >>>= 0; y >>>= 0; z >>>= 0;', ...body, `  return ${result};`, '}'].join('\n');
}

export function emitWgsl(ast: SpecAst, fnName = 'spec'): string {
  const em = new Emitter('wgsl');
  const body: string[] = [];
  for (const s of ast.stmts) { const e = em.expr(s.e); const first = !em.locals.has(s.name); em.locals.add(s.name); body.push(`  ${first ? 'var ' : ''}v_${s.name}${first ? ': u32' : ''} = ${e};`); }
  const result = em.expr(ast.result);
  return [...em.helperSource(), `fn ${fnName}(x: u32, y: u32, z: u32) -> u32 {`, ...body, `  return ${result};`, '}'].join('\n');
}

export function compileSpec(src: string): CompiledSpec {
  const ast = parseSpec(src);
  let nInputs = 0;
  ast.inputs.forEach((used, i) => { if (used) nInputs = i + 1; });
  if (nInputs === 0) throw new SpecError('the spec must read at least one input (x, y or z)', 0);
  const jsSource = emitJs(ast);
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const fn = new Function(`${jsSource}\nreturn spec;`)() as CompiledSpec['fn'];
  // smoke-test the function so a broken template fails here, not in a worker
  fn(1, 2, 3);
  void U;
  return { ast, nInputs, fn, jsSource, wgsl: emitWgsl(ast) };
}
