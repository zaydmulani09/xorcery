/** Number and text formatting helpers for the UI. */

export function fmtInt(n: bigint | number): string {
  return (typeof n === 'bigint' ? n : Math.round(n)).toLocaleString('en-US');
}

/** 6200652384 -> "6.20 B"; 1234 -> "1,234" */
export function fmtShort(n: bigint | number): string {
  const v = typeof n === 'bigint' ? Number(n) : n;
  if (v < 100_000) return fmtInt(v);
  const units: [number, string][] = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'k']];
  for (const [u, s] of units) if (v >= u) return `${(v / u).toFixed(v / u >= 100 ? 0 : v / u >= 10 ? 1 : 2)} ${s}`;
  return fmtInt(v);
}

export function fmtRate(perSec: number): string {
  if (!perSec || !isFinite(perSec)) return '—';
  if (perSec >= 1e9) return `${(perSec / 1e9).toFixed(2)} G/s`;
  if (perSec >= 1e6) return `${(perSec / 1e6).toFixed(1)} M/s`;
  if (perSec >= 1e3) return `${(perSec / 1e3).toFixed(1)} k/s`;
  return `${perSec.toFixed(0)}/s`;
}

export function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
  const m = Math.floor(ms / 60_000), s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

export function fmtEta(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '?';
  if (seconds < 1) return '<1 s';
  if (seconds < 90) return `~${Math.ceil(seconds)} s`;
  if (seconds < 3600) return `~${Math.ceil(seconds / 60)} min`;
  if (seconds < 86400) return `~${(seconds / 3600).toFixed(1)} h`;
  return `~${(seconds / 86400).toFixed(1)} days`;
}

export function fmtConstIn(v: number): string {
  v >>>= 0;
  if (v < 1024) return String(v);
  if (v >= 0xfffff000) return String(v - 0x100000000);
  return '0x' + v.toString(16);
}

/** Parse a user-typed constant: decimal, negative, hex, binary. */
export function parseConst(s: string): number | null {
  s = s.trim().replace(/_/g, '').replace(/[uUlL]+$/, '');
  if (!s) return null;
  let neg = false;
  if (s.startsWith('-')) { neg = true; s = s.slice(1); }
  if (s.startsWith('~')) {
    const v = parseConst(s.slice(1));
    return v === null ? null : ~v >>> 0;
  }
  let v: bigint;
  try {
    if (/^0x[0-9a-f]+$/i.test(s)) v = BigInt(s);
    else if (/^0b[01]+$/i.test(s)) v = BigInt(s);
    else if (/^\d+$/.test(s)) v = BigInt(s);
    else return null;
  } catch { return null; }
  if (v > 0xffffffffn) return null;
  let n = Number(v);
  if (neg) n = -n >>> 0;
  return n >>> 0;
}

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** Light syntax colouring for the expression form. */
export function colorExpr(expr: string): string {
  return escapeHtml(expr)
    .replace(/\b(t\d+)\b/g, '<span class="t">$1</span>')
    .replace(/\b(0x[0-9a-f]+|\d+)\b/g, '<span class="c">$1</span>')
    .replace(/\b([xyz])\b/g, '<span class="x">$1</span>')
    .replace(/\b(clz32|ctz32|popcnt32|rotl32|rotr32|bswap32|brev32|umin32|umax32|smin32|smax32|udiv32|urem32|mulhi32|uint32_t|int32_t)\b/g, '<span class="k">$1</span>');
}
