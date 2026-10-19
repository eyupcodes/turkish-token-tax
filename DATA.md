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

## Domain subsets

Pinned in `configs/domains.lock.json` (archive and per-language SHA-256, raw
pair count). Each corpus is labelled by what the whole source is; no sentence
is classified individually. Only aggregate counts are published.

| Domain | Corpus | License | Source |
|---|---|---|---|
| news | Global Voices (OPUS 2018Q4) | CC BY 3.0 | [Global Voices republishing](https://globalvoices.org/about/global-voices-attribution-policy/) |
| technical | tldr-pages (OPUS v2026-07-07) | CC BY 4.0 | [tldr-pages](https://github.com/tldr-pages/tldr) |
| conversation | Tatoeba (OPUS v2026-07-08) | CC BY 2.0 FR | [Tatoeba terms](https://tatoeba.org/eng/terms_of_use) |

All three are distributed by [OPUS](https://opus.nlpl.eu) (J. Tiedemann, 2012,
*Parallel Data, Tools and Interfaces in OPUS*, LREC). The license statements
were read from the README/LICENSE files inside the downloaded packages. Global
Voices' package defers to the source, whose content is CC BY.

Cleaning drops pairs with a blank side and exact duplicate pairs, then keeps up
to 2,000 pairs ordered by SHA-256 of `tr + TAB + en`. That is deterministic
and does not depend on file order or a random seed. tldr-pages has only 1,934
unique pairs (many lines repeat across pages), so all of them are used.

**No legal subset.** We found no Turkish–English legal parallel corpus with a
clear open license. EUbookshop on OPUS has no license file in its package, so
it is not used. A legal subset will be added only when such a source exists.
