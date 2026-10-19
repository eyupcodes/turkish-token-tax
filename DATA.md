# Data and attribution

Code is MIT licensed. Data retains its own license; MIT does not relicense it.

The default corpus is **FLORES-200**, devtest, `tur_Latn` and `eng_Latn`:
1,012 professionally translated, line-aligned sentence pairs from the original
Meta/Facebook research release. We use the complete two-language devtest split,
without filtering, rewriting, deduplicating or assigning domain labels.

- Authors: NLLB Team / Marta R. Costa-jussà et al. (2022).
- Paper: [No Language Left Behind](https://arxiv.org/abs/2207.04672).
- [Original source and download](https://github.com/facebookresearch/flores/tree/main/flores200).
- [CC BY-SA 4.0 license](https://github.com/facebookresearch/flores/blob/main/LICENSE_CC-BY-SA).
- Immutable archive and per-language SHA-256 digests: `configs/corpus.lock.json`.

The upstream archive is downloaded into an ignored, content-addressed cache.
Only the specified two members are read in memory; archive paths are never
extracted to the filesystem. Corpus text is not redistributed in this repository.
Any redistribution or adaptation must retain attribution and the applicable
share-alike terms. Aggregate measurements contain no sentence text.

The original repository is archived; FLORES+ is a separate evolving successor.
We intentionally benchmark the original locked split, not a moving dataset.

Offline unit tests use original, tiny synthetic fixture sentences under the
repository's MIT license. They are test fixtures, never leaderboard data.

News, technical, conversation and legal subsets remain roadmap issue #5.
No unsourced text or guessed domain labels are included in the benchmark.
