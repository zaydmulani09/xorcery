/**
 * Printing programs: as an inlined expression, as a listing, and as a complete
 * function in C, Rust, JavaScript or WGSL.
 *
 * Results that are read exactly once are inlined into their consumer; results
 * read more than once become named temporaries (`t0`, `t1`, ...). That is how
 *     r0 = x >> 31; r1 = x ^ r0; r2 = r1 - r0
 * becomes
 *     t0 = x >> 31;  (x ^ t0) - t0
 */
import { HELPERS, HELPER_DEPS, Lang, Op, P_ATOM, P_CALL, P_UN, Tmpl } from './isa';
import { Program, SpaceConfig, firstResult, isConst } from './space';

const ASSOC = new Set(['add', 'mul', 'and', 'or', 'xor']);
const INPUT_NAMES = ['x', 'y', 'z'];

interface Piece {
  s: string;
  prec: number;
  /** op id when this piece is an infix expression (for the associativity rule) */
  infixOp?: string;
}

export function fmtLit(v: number, lang: Lang): string {
  v >>>= 0;
  const hex = '0x' + v.toString(16).padStart(8, '0');
  const isSmall = v < 256;
  switch (lang) {
    case 'c': return isSmall ? `${v}u` : `${hex}u`;
    case 'wgsl': return isSmall ? `${v}u` : `${hex}u`;
    case 'rust': return isSmall ? `${v}` : hex;
    case 'js': return isSmall ? `${v}` : hex;
    case 'spec': return v < 1024 ? `${v}` : v >= 0xfffff000 ? `${v - 0x100000000}` : hex;
  }
}

function tmplOf(op: Op, lang: Lang): Tmpl {
  return op[lang];
}

/** Readability rank for commutative operand ordering: inputs, temporaries, expressions, constants. */
function rank(p: Piece): number {
  if (p.prec === P_ATOM) {
    if (/^[xyz]$/.test(p.s)) return 'xyz'.indexOf(p.s) * 0.1;
    if (/^t\d+$/.test(p.s)) return 1;
    return 3;
  }
  return 2;
}

function fill(t: Tmpl, op: Op, a: Piece, b: Piece | null, lang: Lang): Piece {
  const isInfix = t.t.startsWith('$a ') && t.prec < P_CALL;
  const isMethod = t.t.startsWith('$a.');
  if (b && op.kind === 'comm' && rank(a) > rank(b)) [a, b] = [b, a];
  const wrap = (p: Piece, slot: 'a' | 'b'): string => {
    if (p.prec >= P_CALL) return p.s;
    if (isMethod && slot === 'a') return `(${p.s})`;
    if (p.prec === P_UN) {
      // "a - -b" and "a + +b" read badly; "x & -x" is fine.
      const first = p.s[0];
      if (isInfix && (first === '-' || first === '+') && (op.id === 'add' || op.id === 'sub')) return `(${p.s})`;
      return p.s;
    }
    // p is an infix expression
    if (!isInfix) return `(${p.s})`;
    if (p.infixOp === op.id && ASSOC.has(op.id)) return p.s;
    return `(${p.s})`;
  };
  let s = t.t.replace('$a', wrap(a, 'a'));
  if (b) s = s.replace('$b', wrap(b, 'b'));
  void lang;
  return { s, prec: t.prec, infixOp: isInfix ? op.id : undefined };
}

export interface Printed {
  /** temporaries in definition order: [name, expression] */
  temps: [string, string][];
  /** the final expression */
  expr: string;
  /** helper names needed by the templates used */
  helpers: string[];
}

/** Build the inlined expression form for a language. */
export function printProgram(cfg: SpaceConfig, prog: Program, lang: Lang): Printed {
  const r0 = firstResult(cfg);
  const L = prog.length;
  const uses = new Array<number>(L).fill(0);
  for (const ins of prog) {
    if (ins.a >= r0) uses[ins.a - r0]++;
    if (cfg.ops[ins.op].kind !== 'unary' && ins.b >= r0) uses[ins.b - r0]++;
  }
  const helpers = new Set<string>();
  const temps: [string, string][] = [];
  const tempName = new Map<number, string>();
  const pieces: Piece[] = new Array(L);

  const valuePiece = (v: number): Piece => {
    if (v < cfg.nInputs) return { s: INPUT_NAMES[v], prec: P_ATOM };
    if (isConst(cfg, v)) return { s: fmtLit(cfg.consts[v - cfg.nInputs], lang), prec: P_ATOM };
    const ri = v - r0;
    const tn = tempName.get(ri);
    if (tn) return { s: tn, prec: P_ATOM };
    return pieces[ri];
  };

  for (let i = 0; i < L; i++) {
    const ins = prog[i];
    const op = cfg.ops[ins.op];
    const t = tmplOf(op, lang);
    if (t.helper) helpers.add(t.helper);
    const a = valuePiece(ins.a);
    const b = op.kind === 'unary' ? null : valuePiece(ins.b);
    const p = fill(t, op, a, b, lang);
    pieces[i] = p;
    if (i < L - 1 && uses[i] > 1) {
      const name = `t${temps.length}`;
      temps.push([name, p.s]);
      tempName.set(i, name);
    }
  }
  // helper dependency closure
  for (const h of Array.from(helpers)) for (const d of HELPER_DEPS[h] ?? []) helpers.add(d);
  const ordered = Object.keys(HELPERS[lang]).filter((h) => helpers.has(h));
  return { temps, expr: pieces[L - 1].s, helpers: ordered };
}

/**
 * Human-readable one-liner in spec-language syntax (temporaries shown inline
 * as "t0 = ...;"), so a result can be pasted straight back in as a spec.
 */
export function exprString(cfg: SpaceConfig, prog: Program): string {
  const p = printProgram(cfg, prog, 'spec');
  const temps = p.temps.map(([n, e]) => `${n} = ${e}; `).join('');
  return temps + p.expr;
}

export function listing(cfg: SpaceConfig, prog: Program): string {
  const name = (v: number): string => {
    if (v < cfg.nInputs) return INPUT_NAMES[v];
    if (isConst(cfg, v)) return fmtLit(cfg.consts[v - cfg.nInputs], 'js');
    return 'r' + (v - firstResult(cfg));
  };
  return prog
    .map((ins, i) => {
      const op = cfg.ops[ins.op];
      return `r${i} = ${op.id.padEnd(6)} ${name(ins.a)}${op.kind === 'unary' ? '' : ', ' + name(ins.b)}`;
    })
    .join('\n');
}

/** A complete, compilable function in the requested language. */
export function emitFunction(cfg: SpaceConfig, prog: Program, lang: Lang, fnName = 'f'): string {
  const p = printProgram(cfg, prog, lang);
  const args = INPUT_NAMES.slice(0, cfg.nInputs);
  const helperSrc = p.helpers.map((h) => HELPERS[lang][h]).filter(Boolean);
  const lines: string[] = [];
  switch (lang) {
    case 'c': {
      lines.push('#include <stdint.h>');
      if (helperSrc.length) lines.push('', ...helperSrc);
      lines.push('', `static inline uint32_t ${fnName}(${args.map((a) => `uint32_t ${a}`).join(', ')}) {`);
      for (const [n, e] of p.temps) lines.push(`    uint32_t ${n} = ${e};`);
      lines.push(`    return ${p.expr};`, '}');
      break;
    }
    case 'rust': {
      if (helperSrc.length) lines.push(...helperSrc, '');
      lines.push(`pub fn ${fnName}(${args.map((a) => `${a}: u32`).join(', ')}) -> u32 {`);
      for (const [n, e] of p.temps) lines.push(`    let ${n} = ${e};`);
      lines.push(`    ${p.expr}`, '}');
      break;
    }
    case 'js': {
      if (helperSrc.length) lines.push(...helperSrc, '');
      lines.push(`function ${fnName}(${args.join(', ')}) {`);
      for (const a of args) lines.push(`    ${a} >>>= 0;`);
      for (const [n, e] of p.temps) lines.push(`    const ${n} = ${e};`);
      lines.push(`    return ${p.expr};`, '}');
      break;
    }
    case 'wgsl': {
      if (helperSrc.length) lines.push(...helperSrc, '');
      lines.push(`fn ${fnName}(${args.map((a) => `${a}: u32`).join(', ')}) -> u32 {`);
      for (const [n, e] of p.temps) lines.push(`    let ${n} = ${e};`);
      lines.push(`    return ${p.expr};`, '}');
      break;
    }
  }
  return lines.join('\n');
}

/**
 * Compile the program to a JavaScript function via the emitted source. Used
 * by the test-suite and the runtime cross-check to make sure the printed code
 * means what the interpreter computed.
 */
export function compileToJs(cfg: SpaceConfig, prog: Program): (...args: number[]) => number {
  const src = emitFunction(cfg, prog, 'js', 'f');
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  return new Function(`${src}\nreturn f;`)() as (...args: number[]) => number;
}
