import base64
import json

import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers

from turkish_token_tax.adapters import HFAdapter, TokenizerSpec, create_adapter, load_specs
from turkish_token_tax.data import digest


def spec_for(tmp_path, kind="hf-json", data=None):
    if data is None:
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        data = tokenizer.to_str().encode()
    (tmp_path / digest(data)).write_bytes(data)
    return TokenizerSpec(
        name="fixture",
        kind=kind,
        model="test/fixture",
        revision="a" * 40,
        license="MIT",
        unk_token="[UNK]",
        artifact=dict(url=f"https://example.org/resolve/{'a' * 40}/vocab", sha256=digest(data)),
    )


def test_hf_json_without_special_tokens(tmp_path):
    spec = spec_for(tmp_path)
    adapter = create_adapter(spec, tmp_path, offline=True)
    assert adapter.encode("hello unknown").count == 2
    assert adapter.encode("unknown").unknown == 1
    assert adapter.encode("hello").byte_fallback is None
    assert adapter.encode("").count == 0


def test_wordpiece_case_and_no_special_tokens(tmp_path):
    spec = spec_for(tmp_path, "bert-vocab", b"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nHello\nworld\n")
    adapter = create_adapter(spec, tmp_path, offline=True)
    assert adapter.encode("Hello world").count == 2
    assert adapter.encode("hello").unknown == 1


def test_fallback_and_unknown_availability():
    tokenizer = Tokenizer(models.WordLevel({"<0xAA>": 0}, unk_token="<0xAA>"))
    adapter = HFAdapter(tokenizer, None)
    assert adapter.encode("unseen").byte_fallback == 1
    assert adapter.encode("unseen").unknown is None


def test_tiktoken_ordinary_bytes(tmp_path, monkeypatch):
    data = b"".join(base64.b64encode(bytes([i])) + f" {i}\n".encode() for i in range(256))
    (tmp_path / digest(data)).write_bytes(data)
    spec = TokenizerSpec(
        name="bytes",
        kind="tiktoken",
        model="fixture",
        revision="0.14.0",
        license="MIT",
        pattern=r"\S+|\s+",
        artifact=dict(url="https://example.org/vocab", sha256=digest(data)),
    )
    adapter = create_adapter(spec, tmp_path, offline=True)
    assert adapter.encode("ş").count == 2
    assert adapter.encode("a").unknown == 0
    assert adapter.encode("a").byte_fallback is None
    monkeypatch.setattr("importlib.metadata.version", lambda _: "wrong")
    with pytest.raises(ValueError, match="Installed tiktoken"):
        create_adapter(spec, tmp_path, offline=True)


@pytest.mark.parametrize(
    "changes",
    [
        dict(revision="main"),
        dict(artifact=dict(url="https://example.org/main/vocab", sha256="0" * 64)),
        dict(kind="tiktoken", revision="0.14.0"),
    ],
)
def test_unpinned_specs(tmp_path, changes):
    values = spec_for(tmp_path).model_dump()
    values.update(changes)
    with pytest.raises(ValidationError):
        TokenizerSpec.model_validate(values)


def test_unique_specs(tmp_path):
    spec = spec_for(tmp_path).model_dump()
    path = tmp_path / "lock.json"
    path.write_text(json.dumps([spec]))
    assert load_specs(path)[0].name == "fixture"
    for values in ([], [spec, spec]):
        path.write_text(json.dumps(values))
        with pytest.raises(ValueError, match="unique"):
            load_specs(path)
