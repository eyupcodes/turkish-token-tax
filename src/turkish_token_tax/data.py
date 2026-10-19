"""Verified downloads and aligned corpus loading; never extract archives to disk."""

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(pattern=r"^https://")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Corpus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    license: str = Field(min_length=1)
    source: str = Field(pattern=r"^https://")
    archive: Artifact
    tr_member: str
    en_member: str
    tr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    en_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pairs: int = Field(gt=0)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(artifact: Artifact, cache: Path, offline: bool = False) -> Path:
    path = cache / artifact.sha256
    if path.exists():
        if digest(path.read_bytes()) != artifact.sha256:
            raise ValueError(f"Corrupt cache entry: {path.name}")
        return path
    if offline:
        raise FileNotFoundError(f"Missing offline cache entry: {path.name}")
    with urllib.request.urlopen(artifact.url, timeout=60) as response:
        data = response.read(100_000_001)
    if len(data) > 100_000_000 or digest(data) != artifact.sha256:
        raise ValueError(f"Download integrity check failed: {artifact.url}")
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def load_corpus(lock: Path, cache: Path, offline: bool = False) -> tuple[Corpus, list[dict]]:
    spec = Corpus.model_validate(json.loads(lock.read_text(encoding="utf-8")))
    archive = fetch(spec.archive, cache, offline)
    with tarfile.open(archive, mode="r:gz") as bundle:
        contents = []
        for member, expected in [
            (spec.tr_member, spec.tr_sha256),
            (spec.en_member, spec.en_sha256),
        ]:
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Not a corpus file: {member}")
            data = stream.read()
            if digest(data) != expected:
                raise ValueError(f"Corpus member integrity check failed: {member}")
            lines = data.decode("utf-8").splitlines()
            if len(lines) != spec.pairs or any(not line.strip() for line in lines):
                raise ValueError(f"Empty or misaligned corpus: {member}")
            contents.append(lines)
    pairs = [
        {"id": str(i + 1), "tr": tr, "en": en}
        for i, (tr, en) in enumerate(zip(*contents, strict=True))
    ]
    return spec, pairs
