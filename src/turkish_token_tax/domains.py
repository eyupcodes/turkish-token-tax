"""Domain subsets: licensed parallel corpora, verified, cleaned and sampled.

Each domain is pinned like the main corpus: archive digest, per-language
member digests and the raw line count. Loading then

1. drops pairs where either side is blank (OPUS keeps some),
2. drops exact duplicate pairs (Tatoeba repeats short sentences),
3. takes a deterministic sample: pairs ordered by SHA-256 of
   ``tr + "\\t" + en``, first ``sample_size`` kept. No random seed, no
   dependence on file order, and the same sample on every machine.

Domain labels come from the source corpus as a whole (Global Voices is news,
tldr-pages is command-line documentation, Tatoeba is short everyday
sentences); nothing is classified per sentence. No legal corpus is included:
we found no Turkish-English legal parallel text with a clear open license.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .data import Artifact, digest, fetch


class DomainCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    domain: Literal["news", "technical", "conversation", "legal"]
    license: str = Field(min_length=1)
    source: str = Field(pattern=r"^https://")
    attribution: str = Field(min_length=1)
    archive: Artifact
    tr_member: str
    en_member: str
    tr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    en_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pairs: int = Field(gt=0)
    sample_size: int = Field(gt=0)


def load_domain_specs(path: Path) -> list[DomainCorpus]:
    specs = [DomainCorpus.model_validate(i) for i in json.loads(path.read_text(encoding="utf-8"))]
    if not specs or len({s.name for s in specs}) != len(specs):
        raise ValueError("Domain lock must contain unique, nonempty names")
    return specs


def _read_member(bundle: zipfile.ZipFile, member: str, expected: str) -> list[str]:
    try:
        data = bundle.read(member)
    except KeyError:
        raise ValueError(f"Not a corpus file: {member}") from None
    if digest(data) != expected:
        raise ValueError(f"Corpus member integrity check failed: {member}")
    return data.decode("utf-8").splitlines()


def sample_key(pair: dict) -> str:
    return hashlib.sha256(f"{pair['tr']}\t{pair['en']}".encode()).hexdigest()


def clean_and_sample(tr_lines: list[str], en_lines: list[str], size: int) -> tuple[list, dict]:
    seen: set[tuple[str, str]] = set()
    pairs, blank, duplicate = [], 0, 0
    for tr, en in zip(tr_lines, en_lines, strict=True):
        tr, en = tr.strip(), en.strip()
        if not tr or not en:
            blank += 1
            continue
        if (tr, en) in seen:
            duplicate += 1
            continue
        seen.add((tr, en))
        pairs.append({"tr": tr, "en": en})
    sample = sorted(pairs, key=sample_key)[:size]
    for i, pair in enumerate(sample, start=1):
        pair["id"] = str(i)
    stats = {
        "raw_pairs": len(tr_lines),
        "blank_dropped": blank,
        "duplicates_dropped": duplicate,
        "clean_pairs": len(pairs),
        "sampled_pairs": len(sample),
    }
    return sample, stats


def domain_table(report: dict) -> str:
    """Markdown: one row per tokenizer, TR/EN ratio per domain, sorted by mean ratio."""
    domains = [d["domain"] for d in report["domains"]]
    ratios: dict[str, dict[str, float]] = {}
    for d in report["domains"]:
        for row in d["tokenizers"]:
            ratios.setdefault(row["name"], {})[d["domain"]] = row["tr_en_token_ratio"]
    order = sorted(ratios, key=lambda n: (sum(ratios[n].values()) / len(ratios[n]), n))
    lines = [
        "| Tokenizer | " + " | ".join(domains) + " |",
        "|---|" + "---:|" * len(domains),
    ]
    for name in order:
        cells = " | ".join(f"{ratios[name][d]:.4f}" for d in domains)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


def load_domain(spec: DomainCorpus, cache: Path, offline: bool = False) -> tuple[list, dict]:
    archive = fetch(spec.archive, cache, offline)
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as bundle:
        tr = _read_member(bundle, spec.tr_member, spec.tr_sha256)
        en = _read_member(bundle, spec.en_member, spec.en_sha256)
    if len(tr) != spec.pairs or len(en) != spec.pairs:
        raise ValueError(f"Misaligned corpus: {spec.name}")
    return clean_and_sample(tr, en, spec.sample_size)
