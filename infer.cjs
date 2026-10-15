'use strict';
// Local CPU/WASM inference. No remote loader, keys, sockets or document uploads.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const resources = path.resolve(process.argv[2]);
const networkAttempts = [];
globalThis.fetch = async (url) => {
  networkAttempts.push(String(url));
  throw new Error('Fetch disabled during local inference');
};
const ort = require(path.join(resources, 'wasm', 'ort.wasm.min.js'));
ort.env.logLevel = 'error';
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = pathToFileURL(path.join(resources, 'wasm') + path.sep).href;
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
  input += chunk;
  if (input.length > 2000000) throw new Error('Bounded batch input exceeded');
});
process.stdin.on('end', async () => {
  try {
    const batch = JSON.parse(input);
    const rows = batch.ids.length;
    const width = batch.ids[0]?.length;
    if (!rows || rows > 128 || !width || width > 256) throw new Error('Invalid batch dimensions');
    for (const list of [batch.ids, batch.mask]) {
      if (list.length !== rows || list.some(row => row.length !== width || row.some(x => !Number.isSafeInteger(x)))) throw new Error('Malformed token tensors');
    }
    const started = performance.now();
    const model = new Uint8Array(fs.readFileSync(path.join(resources, 'model', 'onnx', 'model_quantized.onnx')));
    const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
    const tensor = (values) => new ort.Tensor('int64', BigInt64Array.from(values.flat(), BigInt), [rows, width]);
    const feeds = { input_ids: tensor(batch.ids), attention_mask: tensor(batch.mask) };
    if (session.inputNames.includes('token_type_ids')) feeds.token_type_ids = tensor(batch.ids.map(row => row.map(() => 0)));
    const output = await session.run(feeds);
    const hidden = output[session.outputNames[0]];
    if (hidden.dims.length !== 3 || hidden.dims[0] !== rows || hidden.dims[1] !== width) throw new Error('Unexpected embedding shape');
    const channels = hidden.dims[2];
    const vectors = [];
    for (let i = 0; i < rows; i++) {
      const vector = new Array(channels).fill(0);
      let count = 0;
      for (let j = 0; j < width; j++) {
        if (!batch.mask[i][j]) continue;
        count++;
        const offset = (i * width + j) * channels;
        for (let k = 0; k < channels; k++) vector[k] += hidden.data[offset + k];
      }
      for (let k = 0; k < channels; k++) vector[k] /= count;
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
      vectors.push(vector.map(value => value / Math.max(norm, 1e-12)));
    }
    await session.release();
    process.stdout.write(JSON.stringify({ vectors, backend: 'onnxruntime-web 1.30.0 / CPU WASM, one thread', seconds: (performance.now() - started) / 1000, fetch_attempts: networkAttempts }));
  } catch (error) {
    process.stderr.write(String(error.stack || error));
    process.exitCode = 1;
  }
});
