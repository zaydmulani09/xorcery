# xorcery

**Your GPU rediscovers bit hacks.**

Type what a function of 32-bit words should compute. xorcery brute-forces *every* straight-line program over the ops you allow — billions per second, on your GPU, in the browser — returns the shortest one, and then **proves** it by evaluating all 4,294,967,296 possible inputs.

**Live: [xorcery.vercel.app](https://xorcery.vercel.app)** · zero runtime dependencies · everything runs client-side

```
spec        (int)x < 0 ? -x : x          // absolute value
result      t0 = (int)x >> 31; (x ^ t0) - t0     3 instructions
proof       identical to the spec on all 4,294,967,296 inputs (190 ms on an Intel Iris Xe)
minimality  all 3,618 dead-code-free programs of length < 3 were tried and none match
```

It is a *superoptimizer* in the original sense (Massalin, 1987): no heuristics, no SMT solver, no LLM — exhaustive enumeration, made practical by a search kernel that evaluates a few billion candidate programs per second on an integrated laptop GPU, and made trustworthy by exhaustive verification and a startup self-test that catches miscompiling shader compilers.

## What it found

Running the standard [Hacker's Delight benchmark](#benchmark) (Gulwani et al., PLDI 2011) it reproduces the textbook answers, and in a few places does better than the textbook:

| function | textbook | xorcery | notes |
|---|---:|---|---|
| is `x` a power of two? (0/1) | 4 ops | **3 ops**: `t = -x; (x ^ t) < t` | proved on all 2^32 inputs |
| does the word contain a zero byte? (0/1) | 5 ops | **4 ops**: `x < (x \| (0x80808080 & (x - 0x01010101)))` | proved on all 2^32 inputs |
| turn off the rightmost run of 1s | `((x \| (x-1)) + 1) & x` | also `x & (x + (x & -x))` and `x & (x - (x \| -x))` | same length, new forms |
| mask of trailing zeros | `~x & (x - 1)` | also `~(x \| -x)` | same length |
| isolate the highest set bit (with `clz`) | — | `x & (0x80000000 >> clz(x))` | correct at x = 0, where the obvious version fails |
| absolute difference (signed) | `t = x - y; (t ^ (t>>31)) - (t>>31)` | **no correct program of ≤ 5 ops exists** in the base ISA | the textbook trick is wrong under overflow; xorcery refuses it |

"Textbook" counts instructions in xorcery's ISA (a compare that yields 0/1 is one op).

## How it works

### 1. The spec

Specs are C expressions over 32-bit words — the same thing you'd paste into a compiler:

```c
(int)x < 0 ? -x : x                 // signedness is a static tag, exactly as in C
uint32_t t = ((x >> z) ^ x) & y;    // statements bind temporaries
x ^ t ^ (t << z)
popcnt(x) & 1                       // popcnt, clz, ctz, mulhi, rotl, bswap, select... are built in
```

The spec is compiled twice: to JavaScript for the CPU and to WGSL for the GPU, from the same table of operator semantics as the instruction set (`src/core/isa.ts`). Shifts mask their amount, `clz(0) = 32`, division by zero is total (RISC-V convention), and the two compilations are cross-checked at runtime.

### 2. The program space

A program is a list of instructions `r_i = op(a, b)` where `a`, `b` are inputs, pool constants, or earlier results. Each slot has a *digit* (an op and an operand pair), and a whole program is a mixed-radix number. Only *dead-code-free* programs are ever evaluated — every result except the last must be read later — because a program with an unused instruction is a shorter program in disguise, and the shorter one was already tried.

The exact number of dead-code-free programs of each length is computed by a small dynamic program over "which results are still unread" (`deadCodeFreeCount` in `src/core/space.ts`). With 10 base ops, one input and four constants: 54 · 3,564 · 312,984 · 37,919,664 · 6,200,652,384 programs for lengths 1–5.

### 3. The GPU kernel

`src/gpu/kernel.ts` generates a WGSL compute shader, specialised for the enabled ops, that enumerates every dead-code-free program of length L:

- **One workgroup owns one prefix** — the first L−2 instructions. Thread *s* evaluates the whole prefix on sample *s* into workgroup memory (32 samples), once.
- **Each thread takes a share of the middle slot's candidates**, computes that instruction's 32-sample vector into registers, and
- **runs the last slot's inner loop**, unrolled per op by the generator: for every operand pair that reads the middle result, one ALU op and one compare per sample, exiting at the first sample that disagrees with the spec. Almost every wrong program dies on sample 0.
- Dead code is pruned structurally: the last instruction must read the middle result, and unread prefix results must be consumed by the two remaining instructions (`needed`/`rem` in the kernel), so the kernel never decodes a program it will not evaluate.
- The prefix index is a mixed-radix odometer with carry, so lengths whose prefix count exceeds 2^32 still work without 64-bit integers.
- Matches are claimed with an atomic and written as packed instructions; the CPU re-evaluates every claimed program (the GPU/CPU "tri-check") before believing it.

Dispatches are sized adaptively to ~45 ms each (Windows kills GPU contexts that hog the device), so the page stays responsive and progress is live. The kernel depends only on the op list — length, constants and slot layout arrive through buffers — so one shader compile (1–3 s on some drivers) serves every length of a search.

Measured on an **Intel Iris Xe** (integrated, 96 EU): **~1.7 billion programs/s** at length 5 (6.2 B programs in 3.6 s). Discrete GPUs should be 5–20× faster.

### 4. Screening and verification (CEGIS, the honest way)

The search only sees 32 samples — generic values first (they reject almost everything), edge cases after. A candidate that survives them is screened on the CPU against a few thousand structured inputs; a counterexample replaces one of the sample slots and the search continues. When the search finishes, the winner is verified on the GPU:

- **one input**: every one of the 2^32 inputs. This is a proof of equivalence with the spec, not a test. ~100 ms.
- **two inputs**: all 2^32 pairs of 16-bit values, the same with high halves set, mixed halves, a grid of special values, and 2^30 hashed pairs — 22.5 billion inputs, labelled *verified*, never *proved*: 2^64 pairs cannot be enumerated.
- **three inputs**: the special-value grid and 2^30 hashed triples.

Before any verdict is accepted, the verification kernel dumps `prog(x)` and `spec(x)` for 512 inputs and the CPU checks that the GPU computed what JavaScript computes. If it did not, the GPU is declared unreliable and the verification reruns on the CPU in a worker (2^32 inputs in ~30 s).

That spot check exists because it was needed: while xorcery was being built, the Intel Iris Xe D3D12 driver compiled `countOneBits(~x & (x - 1))` to *`x`*. The GPU now uses bit-trick implementations of `popcnt`, `clz`, `ctz` and `brev` instead of the WGSL builtins, and every op is self-tested against the CPU on startup (the badge in the header).

### 5. Minimality

Lengths are searched in increasing order and a length is exhausted before the next begins, so the first hit is the shortest program **in the chosen ISA with the chosen constant pool** — a claim the result card states precisely, with the number of shorter programs that were tried. It says nothing about programs using other ops, other constants, branches, or loops.

## Benchmark

The 25 programs from *Synthesis of Loop-Free Programs* (Gulwani, Jha, Tiwari, Venkatesan, PLDI 2011), written as specifications, plus ten classics. Numbers below are one run on an Intel Iris Xe (13th-gen i7 laptop, Chrome 140, Windows 11); "search" excludes the ~2 s one-time shader compile per ISA. Run **Run all (benchmark)** in the gallery for your own GPU and copy the table.

| preset | textbook | found | search | verified |
|---|---:|---:|---:|---|
| P1 turn off rightmost 1 | 2 | 2 | 13 ms | 2^32 |
| P2 test for 2^n − 1 form | 2 | 2 | 11 ms | 2^32 |
| P3 isolate rightmost 1 | 2 | 2 | 9 ms | 2^32 |
| P4 mask from rightmost 1 down | 2 | 2 | 7 ms | 2^32 |
| P5 right-propagate rightmost 1 | 2 | 2 | 11 ms | 2^32 |
| P6 turn on rightmost 0 | 2 | 2 | 10 ms | 2^32 |
| P7 isolate rightmost 0 | 3 | 3 | 74 ms | 2^32 |
| P8 mask of trailing 0s | 3 | 3 | 73 ms | 2^32 |
| P9 abs | 3 | 3 | 20 ms | 2^32 |
| P10 nlz(x) == nlz(y) | 3 | 3 | 2.3 s | 22.5 B pairs |
| P11 nlz(x) < nlz(y) | 3 | 3 | 87 ms | 22.5 B pairs |
| P12 nlz(x) ≤ nlz(y) | 3 | 3 | 510 ms | 22.5 B pairs |
| P13 sign | 4 | 4 | 109 ms | 2^32 |
| P14 floor average | 4 | 4 | 88 ms | 22.5 B pairs |
| P15 ceiling average | 4 | 4 | 90 ms | 22.5 B pairs |
| P16 signed max | 5 | 5 | 9.4 s (15.5 B programs) | 22.5 B pairs |
| P17 turn off rightmost run of 1s | 4 | 4 | 94 ms | 2^32 |
| P18 is power of two | 4 | **3** | 42 ms | 2^32 |
| P19 exchange bit fields (3 inputs) | 6 | none ≤ 4 | 650 ms | — |
| P20 next with same popcount | 9 | none ≤ 4 | 2.0 s | — |
| P21 cycle three values (3 inputs) | 8 | none ≤ 4 | 223 ms | — |
| P22 parity (with popcnt) | 2 | 2 | 6 ms | 2^32 |
| P23 popcount (base ISA) | 12 | none ≤ 4 | 603 ms | — |
| P24 round up to power of two | 12 | none ≤ 4 | 61 ms | — |
| P25 mulhi (with mulhi) | 1 | 1 | 6 ms | 22.5 B pairs |

P19–P21 and P23–P24 have textbook solutions of 6–12 instructions; brute force at length 6 is ~2 × 10^12 programs (about 20 minutes on this GPU, a couple of minutes on a large discrete one), and length 7+ is out of reach. That is the honest horizon of exhaustive search; the point of the tool is what happens below it.

## Spec language reference

```
values        32-bit words. Literals: 42, 0xff, 0b1010, 4294967295u, -1 (= 0xffffffff)
inputs        x, y, z (up to three; the search space depends on how many you use)
statements    name = expr;   uint32_t name = expr;   int name = expr;   (reassignment allowed)
result        the final expression, or `return expr;`
operators     + - * / % & | ^ ~ << >> >>> ! && || == != < <= > >= ?:
signedness    (int)e / (unsigned)e tag an expression; >> < <= > >= / % follow C's rules
              (signed if both operands are signed; a literal adopts the other operand's tag)
              >>> is always logical, sar(a, b) always arithmetic
functions     add sub and or xor shl shr sar not neg mul mulhi mulhs
              clz ctz popcnt (popcount) bswap brev rotl rotr
              eq ne ult ule ugt uge slt sle sgt sge  (0/1 results)
              umin umax smin smax (min/max = unsigned)  udiv urem sdiv srem
              abs sign select(c, a, b) bool
semantics     shifts mask the amount by 31; clz(0) = ctz(0) = 32;
              x / 0 = 0xFFFFFFFF, x % 0 = x (RISC-V); sdiv(INT_MIN, -1) = INT_MIN
```

## Using the engine without the UI

The engine has no DOM dependencies. From the browser console on the live page: `xorcery` is the app; `xorcery.result` holds the last `EngineResult` (programs, verification reports, counts).

```ts
import { runEngine } from './src/engine';
const res = await runEngine(
  { spec: 'x & (x - 1)', opIds: ['add','sub','and','or','xor','shl','shr','sar','not','neg'], consts: [0, 1, 31, 0xffffffff], maxLen: 5 },
  gpu, // from getGpu() in src/gpu/device.ts, or null for the CPU fallback
  { onProgress: (p) => console.log(p.evaluated, p.rate) },
);
console.log(res.solutions[0].expr, res.solutions[0].verify);
```

## Development

```bash
npm install
npm run dev        # http://localhost:5211
npm test           # vitest: semantics, enumeration counts, DP counting, emitters, spec language
npm run build      # tsc --noEmit + vite build -> dist/
```

```
src/core/u32.ts        exact 32-bit semantics for every op (the single source of truth)
src/core/isa.ts        the instruction set: kinds, pruning flags, per-language templates
src/core/space.ts      program space: digits, pair tables, mixed-radix ids, dead-code rule, DP counting
src/core/program.ts    printer (inlined expressions) and C/Rust/JS/WGSL emitters
src/core/samples.ts    the 32-sample screen
src/spec/parser.ts     spec language lexer + parser with C signedness rules
src/spec/compile.ts    spec -> JavaScript and WGSL
src/gpu/kernel.ts      search kernel generator (the interesting file)
src/gpu/search.ts      iterative deepening driver, adaptive dispatch, odometer prefixes
src/gpu/verify.ts      exhaustive / structured verification kernel + spot check
src/gpu/selftest.ts    per-op GPU-vs-CPU self-test, search cross-check
src/cpuverify.ts       worker fallback verification
src/engine.ts          spec -> samples -> search -> screen -> verify orchestration
src/presets.ts         Hacker's Delight P1–P25 and extras, as specifications
src/ui/app.ts          the page
```

The GPU kernel is validated against the CPU enumerator: for several configurations the number of programs the kernel evaluates equals the CPU's dead-code-free count exactly (`npm test` checks the counting; the browser self-test checks the kernel).

## Limitations

- Minimality is relative to the ISA and constant pool you chose. Constants must be in the pool — the search does not synthesise them (both-constant operands are never tried; add the folded constant instead).
- Straight-line code only: no branches, no loops, no memory. Comparisons produce 0/1.
- Length 6 is minutes on an integrated GPU; length 7 is not reachable by brute force. Textbook programs longer than that (popcount, Gosper's hack, clp2) are beyond the horizon unless you add the op that shortens them.
- Two- and three-input results are verified on billions of inputs but not proved.
- WebGPU is required for anything beyond length 3; without it the CPU fallback is 1000× slower.
- 32-bit words only.

## Related work

- Henry S. Warren, *Hacker's Delight* (2002, 2012) and his superoptimizer **Aha!** ("A Hacker's Assistant", C, CPU): the direct ancestor. Aha! reached 4-instruction programs in hours; xorcery reaches 5 in seconds and proves the result.
- Massalin, *Superoptimizer: A Look at the Smallest Program* (1987).
- Gulwani, Jha, Tiwari, Venkatesan, *Synthesis of Loop-Free Programs* (PLDI 2011) — the benchmark, solved there with an SMT solver (Brahma).
- Schkufza, Sharma, Aiken, *Stochastic Superoptimization* (STOKE, 2013) — stochastic search over x86-64.
- Sasnauskas et al., *Souper* — an LLVM superoptimizer built on SMT.
- Valizadeh & Berger (2023, 2024) and *Simba* (arXiv 2605.08243, 2026) — GPU-accelerated enumerative synthesis on CUDA/H100s, the closest recent research; xorcery is the browser-native, verify-everything, interactive cousin.

## Author

Zayd Mulani — [zayd.dpdns.org](https://zayd.dpdns.org) · [github.com/zaydmulani09](https://github.com/zaydmulani09)

MIT License.
