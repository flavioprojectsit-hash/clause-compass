"""Local semantic evidence retrieval. AI-created prototype; no eligibility verdicts."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUESTIONS = {
    'payment': 'When and under which conditions is money paid to the contributor?',
    'deadline': 'What is the deadline for completing or submitting the work?',
    'ai': 'May AI-generated or AI-assisted work be submitted?',
    'publication': 'What work must be published, shared or licensed?',
    'eligibility': 'Who can participate, and are there hardware, location or account requirements?',
}
LABELS = {'payment': 'Pagamento', 'deadline': 'Consegna', 'ai': 'Uso di AI', 'publication': 'Pubblicazione e diritti', 'eligibility': 'Requisiti'}


class LocalRetriever:
    def __init__(self, resources: Path, node_executable=None):
        self.resources = resources.resolve()
        vendor = self.resources / 'vendor'
        if vendor.is_dir():
            sys.path.insert(0, str(vendor))
        import numpy as np
        from tokenizers import Tokenizer
        self.np = np
        self.ort_version = 'onnxruntime-web 1.30.0 / CPU WASM'
        self.node_executable = node_executable or os.environ.get('CLAUSE_COMPASS_NODE') or shutil.which('node')
        if not self.node_executable:
            raise ValueError('Node.js executable missing; supply --node or CLAUSE_COMPASS_NODE.')
        self.network_attempts = []
        self.cache = {}
        model_path = self.resources / 'model' / 'onnx' / 'model_quantized.onnx'
        tokenizer_path = self.resources / 'model' / 'tokenizer.json'
        manifest_path = self.resources / 'model-manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Model manifest missing; prepare the fixed public model first.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for item in manifest['files']:
            path = self.resources / 'model' / item['name']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Model file hash mismatch: ' + item['name'])
        self.revision = manifest['revision']
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        # No remote/pretrained loader: both model and tokenizer are local files.

    def embed(self, texts: list[str]):
        if not texts or len(texts) > 128:
            raise ValueError('Use between one and 128 passages per batch.')
        if all(text in self.cache for text in texts):
            return self.np.asarray([self.cache[text] for text in texts], dtype=self.np.float32)
        tokens = self.tokenizer.encode_batch(texts)
        lengths = [len(item.ids) for item in tokens]
        if max(lengths) > 256:
            raise ValueError('A passage exceeds 256 tokens. Split it into shorter paragraphs; nothing was silently truncated.')
        width = max(lengths)
        np = self.np
        ids = np.zeros((len(tokens), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, item in enumerate(tokens):
            ids[i, :len(item.ids)] = item.ids
            mask[i, :len(item.ids)] = item.attention_mask
        run = subprocess.run([self.node_executable, str(Path(__file__).with_name('infer.cjs')), str(self.resources)], input=json.dumps({'ids': ids.tolist(), 'mask': mask.tolist()}), text=True, encoding='utf-8', capture_output=True, timeout=120)
        if run.returncode:
            raise RuntimeError('Local WASM inference failed: ' + run.stderr[:1500])
        result = json.loads(run.stdout)
        self.network_attempts.extend(result['fetch_attempts'])
        vectors = np.asarray(result['vectors'], dtype=np.float32)
        for text, vector in zip(texts, vectors, strict=True):
            self.cache[text] = vector
        return vectors

    def retrieve(self, document: dict, query_vectors=None):
        paragraphs = document.get('paragraphs', [])
        if not paragraphs or len(paragraphs) > 100 or not all(isinstance(x, str) and x.strip() for x in paragraphs):
            raise ValueError('Each document needs 1–100 nonempty plain-text paragraphs.')
        if query_vectors is None:
            query_vectors = self.embed(list(QUESTIONS.values()))
        vectors = self.embed(paragraphs)
        similarities = query_vectors @ vectors.T
        result = {'id': document['id'], 'title': document['title'], 'source': document['source'], 'questions': {}}
        for row, (key, question) in enumerate(QUESTIONS.items()):
            order = self.np.argsort(-similarities[row], kind='stable')[:3]
            result['questions'][key] = {
                'question': question,
                'answer_presence': 'not_determined_by_retrieval',
                'passages': [{'paragraph_index': int(index), 'text': paragraphs[index], 'similarity': float(similarities[row, index])} for index in order],
            }
        return result


def render_report(results: list[dict], output: Path):
    esc = html.escape
    nav = ''.join(f'<a href="#document-{i}">{esc(doc["title"])}</a>' for i, doc in enumerate(results))
    sections = []
    for i, doc in enumerate(results):
        cards = []
        for key, item in doc['questions'].items():
            quotes = ''.join(
                f'<div class="passage"><span class="rank">{rank}</span><blockquote>{esc(p["text"])}</blockquote>'
                f'<div class="reference">Paragrafo {p["paragraph_index"] + 1} · affinità {p["similarity"]:.3f} · <a href="#source-{i}-{p["paragraph_index"]}">Testo completo</a></div></div>'
                for rank, p in enumerate(item['passages'], 1)
            )
            cards.append(f'<article><h3>{LABELS[key]}</h3><p class="question">{esc(item["question"])}</p>{quotes}</article>')
        original = ''.join(f'<p id="source-{i}-{n}"><strong>{n + 1}.</strong> {esc(text)}</p>' for n, text in enumerate(doc['original_paragraphs']))
        sections.append(f'<section id="document-{i}"><div class="document-head"><span class="eyebrow">TESTO ORIGINALE, ORDINATO PER PERTINENZA</span><h2>{esc(doc["title"])}</h2><p>Fonte: {esc(doc["source"])}</p></div><div class="cards">{"".join(cards)}</div><article class="full-source"><h3>Testo completo della fonte</h3><p>Controlla anche i passaggi che non compaiono nei risultati.</p>{original}</article></section>')
    page = '''<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Clause Compass · passaggi da verificare</title><style>
:root{color-scheme:light;font-family:Segoe UI,Arial,sans-serif;color:#182d39;background:#f2f5f2;font-synthesis:none}*{box-sizing:border-box}body{margin:0}header{background:#173e3b;color:#fff;padding:38px max(24px,calc((100vw - 1260px)/2)) 32px}header .tag{color:#bfe4c6;font-size:12px;letter-spacing:.15em}h1{font-size:42px;letter-spacing:-.04em;margin:12px 0 9px}header p{line-height:1.6;max-width:800px;color:#e2eeea;margin:0}.layout{display:grid;grid-template-columns:225px minmax(0,1fr);gap:30px;max-width:1310px;margin:auto;padding:30px 24px}nav{position:sticky;top:20px;align-self:start}nav h2{font-size:12px;color:#526c67;text-transform:uppercase;letter-spacing:.13em}nav a{display:block;color:#234d45;text-decoration:none;padding:13px 10px;border-left:2px solid #c5d8ce;font-size:14px}nav a:hover,nav a:focus{background:#dbe9e0;border-color:#235d4c}.note{background:#fff8e2;border:1px solid #e6d7a4;border-radius:10px;padding:17px 20px;font-size:14px;line-height:1.6}section{scroll-margin-top:20px;margin:30px 0 60px}.eyebrow{font-size:10px;letter-spacing:.1em;color:#477365}h2{font-size:25px;letter-spacing:-.02em;margin:9px 0}.document-head p{color:#60736d;font-size:13px;overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:22px}article{background:white;border:1px solid #d8e2dc;border-radius:13px;padding:23px}article:last-child{grid-column:1/-1}h3{font-size:19px;margin:0 0 8px}.question{color:#63766e;font-size:12px;line-height:1.5;margin-bottom:20px}.passage{position:relative;padding:0 0 15px 29px;margin-bottom:16px;border-bottom:1px solid #e8ede9}.passage:last-child{border:0;margin:0;padding-bottom:0}.rank{position:absolute;left:0;top:0;background:#edf4ee;color:#396753;font-size:11px;padding:3px 6px;border-radius:4px}blockquote{margin:0;font-size:14px;line-height:1.65;overflow-wrap:anywhere}.reference{color:#6b7d72;font-size:11px;margin-top:8px}footer{padding:25px;color:#65776f;font-size:12px;border-top:1px solid #d8e2dc;line-height:1.6}@media(max-width:800px){.layout{grid-template-columns:1fr;padding:20px 16px}nav{position:static}nav a{display:inline-block;font-size:12px;padding:9px}.cards{grid-template-columns:1fr}article:last-child{grid-column:auto}h1{font-size:34px}}@media print{header{padding:20px}nav{display:none}.layout{display:block;padding:0}.cards{display:block}article{break-inside:avoid;margin:12px 0}footer{padding:10px}}
</style></head><body><header><span class="tag">FLAVIO STUDIO AI · PROTOTIPO LOCALE 0.1</span><h1>Clause Compass</h1><p>Trova i passaggi da rileggere prima di impegnarti in un progetto. Ricerca semantica su CPU, senza chat o invio del testo a un'API.</p></header><div class="layout"><nav aria-label="Documenti"><h2>Documenti</h2>NAV</nav><main><div class="note"><strong>La pertinenza non è una risposta verificata.</strong> Il modello propone tre passaggi per domanda anche quando la risposta manca. Affinità non significa probabilità, ammissibilità o promessa di pagamento. Rileggi il testo e la fonte. In questa demo tutti i documenti sono inventati; è verificata soltanto la ricerca in inglese.</div>SECTIONS</main></div><footer>Creato con AI. Nessun cliente, ordine o compenso dimostrato da questa demo. Modello aperto all-MiniLM-L6-v2, Apache 2.0; inferenza locale con ONNX Runtime. Questa pagina non esegue script o richieste di rete.</footer></body></html>'''.replace('NAV</nav>', nav + '</nav>').replace('SECTIONS</main>', ''.join(sections) + '</main>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')


def evaluate(model: LocalRetriever, cases_path: Path, output: Path):
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    if cases['questions'] != QUESTIONS:
        raise ValueError('Evaluation questions differ from the frozen plan.')
    started = time.perf_counter()
    model.embed(list(QUESTIONS.values()) + [paragraph for doc in cases['cases'] for paragraph in doc['paragraphs']])
    query_vectors = model.embed(list(QUESTIONS.values()))
    results, checks = [], []
    for doc in cases['cases']:
        result = model.retrieve(doc, query_vectors)
        result['original_paragraphs'] = doc['paragraphs']
        results.append(result)
        for key, expected in doc['expected'].items():
            got = [x['paragraph_index'] for x in result['questions'][key]['passages']]
            texts_exact = all(p['text'] == doc['paragraphs'][p['paragraph_index']] for p in result['questions'][key]['passages'])
            checks.append({'document': doc['id'], 'question': key, 'expected': expected, 'retrieved': got, 'top_three_hit': None if expected is None else bool(set(expected) & set(got)), 'text_exact': texts_exact, 'answer_presence_not_invented': result['questions'][key]['answer_presence'] == 'not_determined_by_retrieval'})
    scored = [x for x in checks if x['expected'] is not None]
    correct = sum(x['top_three_hit'] for x in scored)
    summary = {'model': 'Xenova/all-MiniLM-L6-v2', 'revision': model.revision, 'runtime_version': model.ort_version, 'providers': ['CPU WebAssembly one thread'], 'seconds_for_queries_and_six_documents': time.perf_counter() - started, 'documents': len(results), 'questions': len(checks), 'answerable_questions': len(scored), 'top_three_hits': correct, 'top_three_fraction': correct / len(scored), 'required_fraction_fixed_before_run': cases['acceptance']['answerable_top_three_minimum_fraction'], 'missing_answer_cases': sum(x['expected'] is None for x in checks), 'source_text_exact_all': all(x['text_exact'] for x in checks), 'answer_presence_not_claimed_all': all(x['answer_presence_not_invented'] for x in checks), 'wasm_fetch_disabled': True, 'wasm_fetch_attempts': model.network_attempts, 'independent_playtest_or_customer_validation': False, 'checks': checks}
    summary['retrieval_feasibility_passed'] = summary['top_three_fraction'] >= summary['required_fraction_fixed_before_run'] and summary['source_text_exact_all'] and summary['answer_presence_not_claimed_all']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    output.with_name(output.stem + '-passages.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    render_report(results, output.with_suffix('.html'))
    return summary


def main():
    parser = argparse.ArgumentParser(description='Local semantic passage finder; source review remains necessary.')
    parser.add_argument('command', choices=['demo', 'search'])
    parser.add_argument('--resources', type=Path, required=True, help='Local directory containing model, vendor and model-manifest.json')
    parser.add_argument('--input', type=Path, required=True, help='Frozen synthetic JSON for demo, or a plain-text brief for search')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', help='Optional local Node.js executable')
    args = parser.parse_args()
    if args.input.stat().st_size > 1024 * 1024:
        parser.error('Input is limited to one megabyte in this prototype.')
    model = LocalRetriever(args.resources, args.node)
    if args.command == 'demo':
        result = evaluate(model, args.input, args.output)
        print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=True))
    else:
        content = args.input.read_text(encoding='utf-8-sig')
        doc = {'id': 'local-brief', 'title': args.input.stem, 'source': 'local:' + args.input.name, 'paragraphs': [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]}
        result = model.retrieve(doc)
        result['original_paragraphs'] = doc['paragraphs']
        render_report([result], args.output)
        print(json.dumps({'documents': 1, 'questions': 5, 'output': str(args.output)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
