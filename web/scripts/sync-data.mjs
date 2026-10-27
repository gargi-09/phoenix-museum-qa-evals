// Copies the files the app reads at runtime into web/public/, from where they
// already live in the repo. Nothing here is hand-written data: the engine is
// the compiled docs/sandbox.js, and every JSON file is produced by a script.
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = dirname(web);
const pub = join(web, 'public');

const files = [
  ['docs/sandbox.js', 'sandbox.js'],
  ['docs/data/recorded_samples.json', 'data/recorded_samples.json'],
  ['docs/data/metrics.json', 'data/metrics.json'],
  ['samples/recorded_llm_judge.json', 'data/recorded_llm_judge.json'],
];

rmSync(pub, { recursive: true, force: true });
for (const [from, to] of files) {
  mkdirSync(dirname(join(pub, to)), { recursive: true });
  cpSync(join(repo, from), join(pub, to));
}
cpSync(join(repo, 'docs/assets'), join(pub, 'assets'), { recursive: true });
console.log(`[sync] ${files.length} files + docs/assets -> web/public/`);
