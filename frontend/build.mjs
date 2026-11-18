import {copyFile, cp, mkdir, rm} from 'node:fs/promises'
import {join} from 'node:path'
import {build, context} from 'esbuild'

const watch = process.argv.includes('--watch')
const root = import.meta.dirname
const output = join(root, 'dist')
await rm(output, {recursive: true, force: true})
await mkdir(output, {recursive: true})
await Promise.all([
  cp(join(root, 'public'), output, {recursive: true}),
  copyFile(join(root, 'index.html'), join(output, 'index.html')),
])

const options = {
  entryPoints: {'assets/check': 'src/main.tsx'},
  absWorkingDir: root,
  outdir: output,
  bundle: true,
  format: 'esm',
  target: 'es2022',
  minify: !watch,
  sourcemap: watch,
  logLevel: 'info',
}

if (watch) {
  const app = await context(options)
  await app.watch()
  await app.serve({host: '127.0.0.1', port: 5173, servedir: output})
} else {
  await build(options)
}
