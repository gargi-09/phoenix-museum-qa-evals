// The few Node globals sandbox.ts touches when it runs as a CLI
// (`node docs/sandbox.js ...`) or is loaded with require() by the parity
// test. Declared here instead of pulling in @types/node for three names.
declare var module: { exports: unknown } | undefined;
declare var require: {
  (id: string): unknown;
  main: unknown;
};
declare var process: {
  argv: string[];
  exit(code: number): never;
};
