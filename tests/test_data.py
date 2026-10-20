import io
import json
import tarfile
from contextlib import closing

import pytest
from pydantic import ValidationError

from turkish_token_tax.data import Artifact, digest, fetch, load_corpus


def corpus_fixture(tmp_path, tr=b"Merhaba dunya\n", en=b"Hello world\n", pairs=1):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for name, data in [("tr", tr), ("en", en)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    data = archive.getvalue()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / digest(data)).write_bytes(data)
    spec = dict(
        name="fixture",
        license="MIT",
        source="https://example.org/source",
        archive=dict(url="https://example.org/data", sha256=digest(data)),
        tr_member="tr",
        en_member="en",
        tr_sha256=digest(tr),
        en_sha256=digest(en),
        pairs=pairs,
    )
    lock = tmp_path / "corpus.json"
    lock.write_text(json.dumps(spec), encoding="utf-8")
    return lock, cache


def test_alignment(tmp_path):
    lock, cache = corpus_fixture(tmp_path)
    spec, pairs = load_corpus(lock, cache, offline=True)
    assert spec.license == "MIT"
    assert pairs == [{"id": "1", "tr": "Merhaba dunya", "en": "Hello world"}]


@pytest.mark.parametrize(
    "tr,en,count", [(b"\n", b"hi\n", 1), (b"hi\n", b"hi\nthere\n", 1), (b"hi\n", b"hi\n", 2)]
)
def test_bad_alignment(tmp_path, tr, en, count):
    with pytest.raises(ValueError, match="Empty or misaligned"):
        load_corpus(*corpus_fixture(tmp_path, tr, en, count), offline=True)


def test_member_integrity(tmp_path):
    lock, cache = corpus_fixture(tmp_path)
    spec = json.loads(lock.read_text())
    spec["tr_sha256"] = "0" * 64
    lock.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="member integrity"):
        load_corpus(lock, cache)


def test_directory_member(tmp_path):
    lock, cache = corpus_fixture(tmp_path)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as bundle:
        directory = tarfile.TarInfo("directory")
        directory.type = tarfile.DIRTYPE
        bundle.addfile(directory)
    raw = data.getvalue()
    (cache / digest(raw)).write_bytes(raw)
    spec = json.loads(lock.read_text())
    spec["archive"]["sha256"] = digest(raw)
    spec["tr_member"] = "directory"
    lock.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Not a corpus file"):
        load_corpus(lock, cache)


def test_cache_integrity_and_offline(tmp_path):
    artifact = Artifact(url="https://example.org/a", sha256="0" * 64)
    with pytest.raises(FileNotFoundError, match="Missing offline"):
        fetch(artifact, tmp_path, offline=True)
    (tmp_path / artifact.sha256).write_bytes(b"bad")
    with pytest.raises(ValueError, match="Corrupt cache"):
        fetch(artifact, tmp_path)


def test_verified_download(tmp_path, monkeypatch):
    data = b"fixture"
    artifact = Artifact(url="https://example.org/a", sha256=digest(data))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: closing(io.BytesIO(data)))
    assert fetch(artifact, tmp_path).read_bytes() == data
    with pytest.raises(ValueError, match="Download integrity"):
        fetch(Artifact(url=artifact.url, sha256="0" * 64), tmp_path)


def test_interrupted_write_leaves_no_cache_entry(tmp_path, monkeypatch):
    data = b"fixture"
    artifact = Artifact(url="https://example.org/a", sha256=digest(data))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: closing(io.BytesIO(data)))

    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        fetch(artifact, tmp_path)
    assert list(tmp_path.iterdir()) == []

    monkeypatch.undo()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: closing(io.BytesIO(data)))
    assert fetch(artifact, tmp_path).read_bytes() == data
    assert [p.name for p in tmp_path.iterdir()] == [artifact.sha256]


def test_schema_rejects_unpinned_source():
    with pytest.raises(ValidationError):
        Artifact(url="http://example.org", sha256="main")
