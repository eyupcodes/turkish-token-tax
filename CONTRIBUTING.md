# Contributing

Install `.[dev,tokenizers]`, then run `ruff check .`, `ruff format --check .`,
`python -m pytest` and `python -m build`. Tests must remain offline; the suite blocks
socket connections. Keep new fixtures small, original and deterministic. Coverage
must remain at least 90% with branch coverage enabled.

## Add a tokenizer

1. Check the original model card's license and accessibility. Tokenizer files may
   have terms independent of this repository's code. Do not commit credentials.
2. Resolve the model's exact commit SHA and download its `tokenizer.json`. Record
   `name`, `kind: hf-json`, `model`, `revision`, `license`, `artifact.url` and its
   SHA-256 in `configs/tokenizers.lock.json`. The resolve URL must contain the SHA.
   `unk_token` is optional; set it only if that actual vocabulary contains it.
3. For a cased BERT WordPiece vocabulary, use `kind: bert-vocab` and `vocab.txt`.
   Other normalizer settings require an explicit new adapter and fixture tests;
   do not pretend all BERT-family tokenizers share BERTurk's defaults.
4. Keep adapters weight-free and turn off added special tokens, truncation and
   padding. Add offline tests for any new behavior using a tiny fixture vocabulary.
5. Run `ttt run --config configs/default.yaml` online once to populate the verified
   cache. Repeat with `--offline`, compare report hashes, review absolute counts
   and unknown rates, then refresh the README table with `ttt leaderboard`.

New data requires provenance, a compatible explicit license, integrity hashes and
line-alignment checks. Keep corpus and code licensing distinct. Never mix synthetic
fixtures into a published benchmark, silently relabel domains or change locks just
to obtain a more attractive ranking. Lock updates and refreshed results need review.
