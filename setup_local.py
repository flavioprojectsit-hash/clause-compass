"""Explicit local preparation, only when run by the user; no automatic startup."""
import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / 'resources'
REVISION = '751bff37182d3f1213fa05d7196b954e230abad9'
WASM_VERSION = '1.30.0'
WASM_INTEGRITY = 'sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=='


def get_bytes(url, maximum):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise RuntimeError('Download exceeds prototype size limit')
    return data


def main():
    RESOURCES.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--only-binary=:all:', '--target', str(RESOURCES / 'vendor'), '--report', str(RESOURCES / 'python-packages.json'), 'numpy==2.5.4', 'tokenizers==0.23.3'], check=True)
    model_manifest = {'source': 'Xenova/all-MiniLM-L6-v2', 'revision': REVISION, 'files': []}
    for name in ['tokenizer.json', 'config.json', 'onnx/model_quantized.onnx']:
        url = f'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/{REVISION}/{name}'
        data = get_bytes(url, 80 * 1024 * 1024)
        target = RESOURCES / 'model' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        model_manifest['files'].append({'name': name, 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    (RESOURCES / 'model-manifest.json').write_text(json.dumps(model_manifest, indent=2) + '\n', encoding='utf-8')
    tarball = get_bytes(f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{WASM_VERSION}.tgz', 120 * 1024 * 1024)
    if 'sha512-' + base64.b64encode(hashlib.sha512(tarball).digest()).decode('ascii') != WASM_INTEGRITY:
        raise RuntimeError('WASM package integrity mismatch')
    destination = RESOURCES / 'wasm'
    destination.mkdir(exist_ok=True)
    required = {'ort.wasm.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'}
    saved = set()
    with tarfile.open(fileobj=io.BytesIO(tarball), mode='r:gz') as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in required:
                content = archive.extractfile(member).read()
                (destination / name).write_bytes(content)
                saved.add(name)
    if saved != required:
        raise RuntimeError('Required runtime file missing')
    print('Local resources prepared. No server, account or paid service started.')


if __name__ == '__main__':
    main()
