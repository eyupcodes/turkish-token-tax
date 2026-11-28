# Changelog

## Unreleased

- Dependabot proposes monthly, grouped updates for GitHub Actions and pinned Python dependencies; `tiktoken` and `tokenizers` are excluded because the published token counts depend on their exact versions (#31).
- `ttt chart`, `ttt drafts`, `ttt domains`, `ttt leaderboard` and `ttt domain-table` report an unwritable output as a one-line `error: cannot write <path>: <reason>` (exit 1) instead of a traceback (#29).
- `ttt drafts` reports an unreadable or malformed saved report as a one-line `error:` naming the file (exit 1) instead of a traceback, and writes no draft files in that case (#27).
- `ttt run` and `ttt domains` report a corpus archive that is not a readable tar.gz/zip, or a locked member missing from it, as a one-line `failed:` error (exit 1) instead of a traceback or a bare `KeyError` (#25).
- `ttt run` reports a config file with invalid YAML as a one-line `Benchmark failed: ... invalid YAML` error (exit 1) instead of a traceback (#23).
- CI runs `mypy` on `src` (`mypy` and `types-PyYAML` are in the `dev` extras); a tiktoken spec without a pinned regex now fails with a clear error instead of a failure inside tiktoken (#21).
- Verified FLORES-200 devtest loading with immutable content digests and license attribution.
- Five real tokenizer adapters, cached artifacts and immutable revision locks.
- Deterministic CLI, JSON/Markdown/HTML reports, ratio chart and generated README leaderboard.
- Explicit diagnostic availability, isolated-word fragmentation and context-density estimates.
- `ttt drafts`: local Kaggle dataset and HF Space drafts with a hashed provenance manifest; nothing is published.
- `ttt chart` draws the TR/EN ratio as an SVG; `ttt leaderboard --check` fails CI when the README table is stale (#4).
- Licensed domain subsets (news: Global Voices, technical: tldr-pages, conversation: Tatoeba) with `ttt domains` and a checked README table; no legal subset for lack of an openly licensed source (#5).
- Verified downloads are written to the cache atomically, so an interrupted write cannot leave a corrupt cache entry (#17).
- `ttt leaderboard`, `ttt domain-table` and `ttt chart` report an invalid or incomplete saved report as a one-line `error:` naming the file (exit 1) instead of a traceback (#19).
