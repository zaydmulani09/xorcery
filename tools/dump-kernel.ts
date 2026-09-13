/**
 * Print the generated search kernel for an op-group list, so the WGSL can be
 * read without running anything:  deno run --sloppy-imports tools/dump-kernel.ts base > docs/kernel-base.wgsl
 */
import { OPS, OpGroup } from '../src/core/isa';
import { generateKernel } from '../src/gpu/kernel';

const groups = ((globalThis as any).Deno?.args?.[0] ?? 'base').split(',') as OpGroup[];
const ops = OPS.filter((o) => groups.includes(o.group));
console.log(generateKernel(ops));
