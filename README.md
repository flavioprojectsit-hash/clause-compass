# Clause Compass — private prototype 0.1

Find source passages about payment, delivery deadlines, AI rules, publication rights and participation requirements in an English project brief. The output is an HTML evidence report with the original paragraph references and the complete source text.

This prototype was created with AI by Flavio, an AI assistant acting for a human owner under the trade name Flavio Studio AI. No client use, sales, human review, contest entry or public release is claimed.

## What was actually tested

Six synthetic English briefs, thirty questions, twenty-six with a predeclared answer passage. The correct passage appeared among the first three in 25/26 answerable questions. Four questions have no answer in the supplied text: the application does **not** detect that absence and still shows suggestions. This is a retrieval tool, not a contractual verdict or an eligibility decision.

The missed payment passage is in the brief containing a hostile instruction. The hostile text was displayed as source data and was not executed. Similarity scores are not confidence percentages. The full original source is available because ranked results can omit an important clause.

The six-document inference took approximately two seconds on one Intel N150 CPU thread using CPU WebAssembly. Python network calls and the JavaScript fetch interface were blocked during the initial successful controlled test, with no attempted calls recorded. An OS firewall audit or packet capture was not performed. Local source inspection uses file-based loaders; the application contains no document-upload API. Avoid using confidential real documents until an independent review of the complete local environment.

The rendered desktop report and a source-paragraph anchor were checked in a browser. No horizontal overflow was measured at the observed 1280-pixel viewport. Mobile layout, long real documents, languages other than English, installation on another PC, and customer usefulness have not been verified.

On 15 October 2026, the reviewed ZIP was also extracted into a separate directory with no resources. Its preparation completed from the public dependencies, and the supplied example search ran successfully with Python 3.12.14 in isolated mode and Node.js 24.19.0. All seven source paragraphs and fifteen working paragraph references were present in the generated report. This used the same Windows PC; it is not an independent user test, a clean operating-system test, or a new measurement of retrieval accuracy. The application code and model were unchanged.

## Local preparation

Python 3.12 and Node.js 24 were used. No API key or GPU is required. `setup_local.py` downloads the pinned public model and runtime and installs Python packages only into this directory's `resources/vendor`. It does not create an account, change system settings or start a service. An internet connection is needed for that preparation, not for inference afterward.

```text
python setup_local.py
python clause_compass.py demo --resources resources --input evaluation-cases.json --output demo-result.json
python clause_compass.py search --resources resources --input example-brief.txt --output evidence.html
```

If Node is not on the path, add `--node` followed by its local executable path. Use UTF-8 text with blank lines between paragraphs. Limits: one megabyte per input file, one hundred paragraphs per brief, 256 tokens per paragraph. Overlong paragraphs are rejected instead of silently shortened. The current version produces a file; it has no web upload form or background monitor.

Open `demo-result.html` directly to inspect the supplied synthetic demonstration. It contains no scripts or external assets. The precomputed report alone does not analyze a newly edited text: run the local command for a new input.

## Components and rights

- Open model: [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2), Apache 2.0, revision `751bff37182d3f1213fa05d7196b954e230abad9`.
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), MIT, version 1.30.0, local CPU WebAssembly.
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers), Apache 2.0, version 0.23.3.
- [NumPy](https://numpy.org/about/), BSD, version 2.5.4.

Downloads and package integrity hashes are recorded by preparation. No proprietary source images, client files or models requiring an account are bundled. The original application source is supplied for private review. Any public release, license of this source or contest submission remains subject to the human owner's decision; this README grants no new publication consent.

## Proposed next test

Give this specific prototype to an independent freelancer or small studio through a permitted channel. Compare whether it helps locate clauses faster than ordinary document search, record missed clauses, and ask whether a usable local package would have value. No price or paid tier is established. Do not add PDF ingestion, multilingual support, an installer or more cases merely to grow the feature list.

The DEV Side Hustle challenge is a possible distribution/prize channel, not a validated customer or recurring revenue. Its rules require a public demo/video, code repository, English entry and an eligible human account holder. No such entry has been made. A policy clarification about public attribution and payment methods was sent to the organizer before requesting account creation.
