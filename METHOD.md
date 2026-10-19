# Measurement method

## Corpus and counting

Use all 1,012 aligned FLORES-200 devtest Turkish/English pairs (see DATA.md).
Read UTF-8; discard only line terminators with `splitlines()`. No Unicode
normalization, lowercasing, deduplication, sampling or domain assignment is added.
Each sentence is tokenized independently. No chat templates, BOS/EOS, padding or
truncation are applied. Native tokenizer normalization still applies (e.g. BERT's
basic normalization). Empty lines and unequal alignment are errors, not silently
dropped pairs. Hash mismatches, unpinned revisions and duplicate tokenizer names fail.

The reported token tax is **sum(TR tokens) / sum(EN tokens)**, not the mean of
sentence ratios. It describes this corpus and this encoding; translations can differ
in length and register. A ratio below one is possible. It is not proof that Turkish
is cheaper for every task.

- Tokens per word: total sentence tokens / total whitespace-delimited words (`\S+`).
  Punctuation remains attached; this is a transparent operational definition, not
  a claim about linguistic words.
- Characters per token: original Unicode code points, including whitespace and
  punctuation, divided by total tokens. This is neither bytes nor grapheme clusters.
- Fragmentation: count tokens for each whitespace word **in isolation**. Counts
  form a histogram; boundary-sensitive tokenizers may differ in full sentences.
  This does not identify morphemes or prove that particular suffixes are split.
- Unknown rate: unknown-token count / tokens where an unknown ID exists. Tiktoken
  covers bytes and has no unknown tokens, so its unknown rate is zero.
- Byte-fallback rate: token IDs spelled `<0xHH>` / tokens when those IDs exist.
  Otherwise `null` (unavailable). Byte-level BPE is not SentencePiece byte fallback;
  its raw byte vocabulary is not misleadingly counted as fallback.
- Context capacity: a fixed comparison budget (default 8,192 tokens) multiplied
  by observed Turkish characters/token. This estimates text density. It is not
  an actual model window, prompt-inclusive capacity, or measured document packing.

## Tokenizers and reproducibility

`configs/tokenizers.lock.json` pins HF model IDs, exact 40-character revisions,
artifact hashes, and licenses. Tokenizer JSON disables any saved padding/truncation.
BERTurk is reconstructed from its pinned WordPiece vocabulary with cased defaults.
No weights or HF credentials are needed. Tiktoken 0.14.0 is pinned, along with its
rank-file SHA-256 and regex copied from that version's MIT-licensed
[encoding definitions](https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py).
Only ordinary encoding is used; literal special-token strings are ordinary text.

Downloads are content-addressed and reverified before use. `--offline` fails on a
missing cache entry. Paths in YAML resolve relative to that YAML, independent of
the shell working directory. Pinned dependencies are in `pyproject.toml`.
Reports record runtime tokenizer versions and lock digests, and include neither
timestamps nor machine paths. Repeated runs with the same environment/inputs
produce identical JSON, Markdown and HTML bytes; the unit suite verifies this on
a fixture, and the real default run is checked separately.

## Domain subsets

`ttt domains` applies the same counting to each subset in
`configs/domains.lock.json`: verified download, blank and duplicate pairs
removed, then the first `sample_size` pairs by SHA-256 of `tr + TAB + en`.
The per-domain ratio is again sum(TR tokens) / sum(EN tokens). Subsets differ in
size, sentence length and source, so a ratio is comparable across tokenizers
within one domain, not across domains or against the FLORES-200 table.

## Interpretation and limits

BERTurk and XLM-R are encoder references. A Turkish-specific tokenizer can obtain
a low TR/EN ratio partly by fragmenting **English** heavily; do not rank it as a
better generative LLM. Inspect absolute Turkish tokens, characters/token and unknown
rate together. Among measured generative encodings, this initial corpus favors
o200k_base. Token counts do not measure generation quality, model speed, translation
quality or API prices. Providers may apply templates and bill tokens differently.

Only FLORES-200 devtest is measured. No confidence intervals or domain-level claims
are made. Llama, Gemma, Mistral, Qwen3 and mT5 are not measured in this initial run.
Gated tokenizer access, domain subsets, expanded plots and publication drafts remain
future work; no account access or publication is implied.
