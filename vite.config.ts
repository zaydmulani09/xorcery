import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'es2022',
    sourcemap: false,
    minify: 'esbuild',
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
} as any);
