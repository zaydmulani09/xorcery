// xorcery search kernel — ops=[add,sub,and,or,xor,shl,shr,sar,not,neg]
const NS: u32 = 32u;
const NONE: u32 = 0xffffffffu;
const NOPS: u32 = 10u;

struct Params {
  start: array<vec4<u32>, 2>,   // prefix start digits d[0..S) (odometer origin)
  radix: array<vec4<u32>, 2>,   // slot counts for prefix slots
  limit: u32,                   // workgroups that carry a valid prefix in this dispatch
  S: u32,                       // prefix length (L - 2)
  mid: u32,                     // value index of the middle result (= nInputs + nConsts + S)
  firstR: u32,                  // value index of r0 (= nInputs + nConsts)
  nInputs: u32,
  nConsts: u32,
  midCount: u32,                // digits in the middle slot
  nops: u32,
}
struct Results {
  count: atomic<u32>,
  evaluated: atomic<u32>,
  pad0: u32,
  pad1: u32,
  entries: array<u32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: Results;
@group(0) @binding(3) var<storage, read> data: array<u32>;    // inputs [n*NS], targets [NS], consts [k]
@group(0) @binding(4) var<storage, read> slotmap: array<u32>;  // per (slot, op): base, count, pairOffset, 0

var<workgroup> pv: array<u32, 672>;
var<workgroup> tg: array<u32, 32>;


fn apply(op: u32, a: u32, b: u32) -> u32 {
  switch (op) {
    case 0u: { return a + b; }
    case 1u: { return a - b; }
    case 2u: { return a & b; }
    case 3u: { return a | b; }
    case 4u: { return a ^ b; }
    case 5u: { return a << (b & 31u); }
    case 6u: { return a >> (b & 31u); }
    case 7u: { return bitcast<u32>(bitcast<i32>(a) >> (b & 31u)); }
    case 8u: { return ~a; }
    case 9u: { return (0u - a); }
    default: { return 0u; }
  }
}

fn isUnary(op: u32) -> bool {
  switch (op) {
    case 8u: { return true; }
    case 9u: { return true; }
    default: { return false; }
  }
}

// digit -> (op, a, b) for a slot, using the layout table
fn decode(slot: u32, d: u32) -> vec3<u32> {
  var e = slot * NOPS * 4u;
  for (var o = 0u; o < NOPS; o++) {
    let base = slotmap[e];
    let cnt = slotmap[e + 1u];
    if (d < base + cnt) {
      let p = pairs[slotmap[e + 2u] + (d - base)];
      return vec3<u32>(o, p >> 8u, p & 0xffu);
    }
    e += 4u;
  }
  let p = pairs[0u];
  return vec3<u32>(0u, p >> 8u, p & 0xffu);
}

fn resultBit(v: u32) -> u32 {
  return select(0u, 1u << (v - params.firstR), v >= params.firstR);
}

fn pack(op: u32, a: u32, b: u32) -> u32 { return (op << 16u) | (a << 8u) | b; }

fn claim(pp: ptr<function, array<u32, 8>>, S: u32, mop: u32, ma: u32, mb: u32, lop: u32, la: u32, lb: u32) {
  let slot = atomicAdd(&results.count, 1u);
  if (slot < 64u) {
    let base = slot * 8u;
    for (var i = 0u; i < S; i++) { results.entries[base + i] = (*pp)[i]; }
    results.entries[base + S] = pack(mop, ma, mb);
    results.entries[base + S + 1u] = pack(lop, la, lb);
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let wgi = wg.x;
  let S = params.S;
  let mid = params.mid;
  let n = params.nInputs;
  let k = params.nConsts;

  // ---- prefix digits: odometer add of the workgroup index -----------------
  var d: array<u32, 8>;
  var pop: array<u32, 8>;
  var pa: array<u32, 8>;
  var pb: array<u32, 8>;
  var pp: array<u32, 8>;
  var carry = wgi;
  for (var i = i32(S) - 1; i >= 0; i--) {
    let ui = u32(i);
    let r = params.radix[ui >> 2u][ui & 3u];
    let v = params.start[ui >> 2u][ui & 3u] + carry;
    d[ui] = v % r;
    carry = v / r;
  }
  var skip = wgi >= params.limit;

  // ---- decode the prefix and find results it leaves unread ----------------
  var used = 0u;
  for (var i = 0u; i < S; i++) {
    let ins = decode(i, d[i]);
    pop[i] = ins.x; pa[i] = ins.y; pb[i] = ins.z;
    pp[i] = pack(ins.x, ins.y, ins.z);
    used |= resultBit(ins.y);
    if (!isUnary(ins.x)) { used |= resultBit(ins.z); }
  }
  let needed = ((1u << S) - 1u) & ~used;
  // the middle instruction can absorb two unread results and the last one more
  if (countOneBits(needed) > 3u) { skip = true; }
  // canonical order: an instruction that does not read the previous result must have a larger key
  for (var i = 1u; i < S; i++) {
    let prev = params.firstR + i - 1u;
    let dep = (pa[i] == prev) || (!isUnary(pop[i]) && pb[i] == prev);
    if (!dep && pp[i] <= pp[i - 1u]) { skip = true; }
  }

  // ---- cooperative prefix evaluation: thread s computes sample s ----------
  if (lid < 32u && !skip) {
    let s = lid;
    for (var i = 0u; i < n; i++) { pv[i * 32u + s] = data[i * 32u + s]; }
    for (var i = 0u; i < k; i++) { pv[(n + i) * 32u + s] = data[n * 32u + 32u + i]; }
    tg[s] = data[n * 32u + s];
    for (var i = 0u; i < S; i++) {
      let a = pv[pa[i] * 32u + s];
      let b = pv[pb[i] * 32u + s];
      pv[(params.firstR + i) * 32u + s] = apply(pop[i], a, b);
    }
  }
  workgroupBarrier();
  if (skip) { return; }

  var evaluated = 0u;
  // ---- middle slot: each thread takes a share of the candidates -----------
  for (var m = lid; m < params.midCount; m += 256u) {
    let mins = decode(S, m);
    let mop = mins.x; let ma = mins.y; let mb = mins.z;
    var needed2 = needed & ~resultBit(ma);
    if (!isUnary(mop)) { needed2 &= ~resultBit(mb); }
    if (countOneBits(needed2) > 1u) { continue; }
    if (S > 0u) {
      // canonical order against the last prefix instruction
      let prev = mid - 1u;
      let dep = (ma == prev) || (!isUnary(mop) && mb == prev);
      if (!dep && pack(mop, ma, mb) <= pp[S - 1u]) { continue; }
    }
    let rem = select(NONE, params.firstR + countTrailingZeros(needed2), needed2 != 0u);
    var mv: array<u32, 32>;
    for (var s = 0u; s < 32u; s++) {
      mv[s] = apply(mop, pv[ma * 32u + s], pv[mb * 32u + s]);
    }
    // ---- last slot: unrolled per op; must read the middle result ----------
      // ---- add ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] + pv[b * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 0u, mid, b); }
        }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] + mv[s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 0u, mid, mid); }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] + pv[rem * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 0u, mid, rem); }
        }
      }
      // ---- sub ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] - pv[b * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 1u, mid, b); }
        }
        }
        for (var a = 0u; a < mid; a++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((pv[a * 32u + s] - mv[s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 1u, a, mid); }
        }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] - pv[rem * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 1u, mid, rem); }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((pv[rem * 32u + s] - mv[s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 1u, rem, mid); }
        }
      }
      // ---- and ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] & pv[b * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 2u, mid, b); }
        }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] & pv[rem * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 2u, mid, rem); }
        }
      }
      // ---- or ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] | pv[b * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 3u, mid, b); }
        }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] | pv[rem * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 3u, mid, rem); }
        }
      }
      // ---- xor ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] ^ pv[b * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 4u, mid, b); }
        }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] ^ pv[rem * 32u + s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 4u, mid, rem); }
        }
      }
      // ---- shl ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] << (pv[b * 32u + s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 5u, mid, b); }
        }
        }
        for (var a = 0u; a < mid; a++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((pv[a * 32u + s] << (mv[s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 5u, a, mid); }
        }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] << (mv[s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 5u, mid, mid); }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] << (pv[rem * 32u + s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 5u, mid, rem); }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((pv[rem * 32u + s] << (mv[s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 5u, rem, mid); }
        }
      }
      // ---- shr ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] >> (pv[b * 32u + s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 6u, mid, b); }
        }
        }
        for (var a = 0u; a < mid; a++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((pv[a * 32u + s] >> (mv[s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 6u, a, mid); }
        }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] >> (mv[s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 6u, mid, mid); }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((mv[s] >> (pv[rem * 32u + s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 6u, mid, rem); }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((pv[rem * 32u + s] >> (mv[s] & 31u)) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 6u, rem, mid); }
        }
      }
      // ---- sar ----
      if (rem == NONE) {
        for (var b = 0u; b < mid; b++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((bitcast<u32>(bitcast<i32>(mv[s]) >> (pv[b * 32u + s] & 31u))) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 7u, mid, b); }
        }
        }
        for (var a = 0u; a < mid; a++) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((bitcast<u32>(bitcast<i32>(pv[a * 32u + s]) >> (mv[s] & 31u))) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 7u, a, mid); }
        }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((bitcast<u32>(bitcast<i32>(mv[s]) >> (mv[s] & 31u))) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 7u, mid, mid); }
        }
      } else {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((bitcast<u32>(bitcast<i32>(mv[s]) >> (pv[rem * 32u + s] & 31u))) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 7u, mid, rem); }
        }

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((bitcast<u32>(bitcast<i32>(pv[rem * 32u + s]) >> (mv[s] & 31u))) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 7u, rem, mid); }
        }
      }
      // ---- not ----
      if (rem == NONE) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if ((~mv[s]) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 8u, mid, 0u); }
        }
      }
      // ---- neg ----
      if (rem == NONE) {

        {
          var ok = true;
          for (var s = 0u; s < 32u; s++) {
            if (((0u - mv[s])) != tg[s]) { ok = false; break; }
          }
          evaluated += 1u;
          if (ok) { claim(&pp, S, mop, ma, mb, 9u, mid, 0u); }
        }
      }
  }
  atomicAdd(&results.evaluated, evaluated);
}

