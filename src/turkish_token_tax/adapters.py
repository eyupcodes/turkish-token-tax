"""Tokenizer-only adapters. No model weights, credentials or implicit downloads."""

import base64
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data import Artifact, fetch


class TokenizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    kind: Literal["hf-json", "bert-vocab", "tiktoken"]
    model: str
    revision: str
    license: str = Field(min_length=1)
    artifact: Artifact
    unk_token: str | None = None
    pattern: str | None = None

    @model_validator(mode="after")
    def immutable_revision(self):
        if self.kind == "tiktoken":
            if self.revision != "0.14.0" or not self.pattern:
                raise ValueError("tiktoken requires version 0.14.0 and its pinned regex")
        elif not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("HF revision must be an immutable commit SHA")
        elif f"/resolve/{self.revision}/" not in self.artifact.url:
            raise ValueError("HF artifact URL must contain its locked revision")
        return self


@dataclass(frozen=True)
class Encoded:
    count: int
    unknown: int | None
    byte_fallback: int | None


class Adapter(Protocol):
    def encode(self, text: str) -> Encoded: ...


class HFAdapter:
    def __init__(self, tokenizer, unk_token: str | None):
        self.tokenizer = tokenizer
        self.unk_id = tokenizer.token_to_id(unk_token) if unk_token else None
        self.fallback_ids = {
            index
            for token, index in tokenizer.get_vocab().items()
            if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token)
        }
        self.has_fallback = bool(self.fallback_ids)

    def encode(self, text: str) -> Encoded:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return Encoded(
            len(ids),
            ids.count(self.unk_id) if self.unk_id is not None else None,
            sum(i in self.fallback_ids for i in ids) if self.has_fallback else None,
        )


class TiktokenAdapter:
    def __init__(self, encoding):
        self.encoding = encoding

    def encode(self, text: str) -> Encoded:
        # Raw bytes are intrinsic to this vocabulary, not SentencePiece byte fallback.
        return Encoded(len(self.encoding.encode_ordinary(text)), 0, None)


def load_specs(path: Path) -> list[TokenizerSpec]:
    specs = [
        TokenizerSpec.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    if not specs or len({s.name for s in specs}) != len(specs):
        raise ValueError("Tokenizer lock must contain unique, nonempty names")
    return specs


def create_adapter(spec: TokenizerSpec, cache: Path, offline: bool = False) -> Adapter:
    artifact = fetch(spec.artifact, cache, offline)
    if spec.kind == "tiktoken":
        import tiktoken

        if importlib.metadata.version("tiktoken") != spec.revision:
            raise ValueError("Installed tiktoken version differs from the lock")
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in artifact.read_bytes().splitlines())
        }
        return TiktokenAdapter(
            tiktoken.Encoding(
                name=spec.name,
                pat_str=spec.pattern,
                mergeable_ranks=ranks,
                special_tokens={},
            )
        )
    from tokenizers import Tokenizer

    if spec.kind == "bert-vocab":
        from tokenizers import BertWordPieceTokenizer

        tokenizer = BertWordPieceTokenizer(str(artifact), lowercase=False)
    else:
        tokenizer = Tokenizer.from_file(str(artifact))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return HFAdapter(tokenizer, spec.unk_token)
