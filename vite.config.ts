import { defineConfig, Plugin } from 'vite';
import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

/** Dev-only: `POST /__save?name=og.png` with a PNG body writes public/<name> (used by tools/og.html). */
function savePlugin(): Plugin {
  return {
    name: 'xorcery-save',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__save', (req, res) => {
        const url = new URL(req.url ?? '', 'http://x');
        const name = (url.searchParams.get('name') ?? 'og.png').replace(/[^a-z0-9._-]/gi, '');
        const chunks: Buffer[] = [];
        req.on('data', (c: Buffer) => chunks.push(c));
        req.on('end', () => {
          mkdirSync(resolve('public'), { recursive: true });
          writeFileSync(resolve('public', name), Buffer.concat(chunks));
          res.statusCode = 200;
          res.end('saved ' + name);
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [savePlugin()],
  build: {
    target: 'es2022',
    sourcemap: false,
    minify: 'esbuild',
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
} as any);
