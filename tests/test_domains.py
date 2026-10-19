import io
import json
import zipfile

import pytest
from pydantic import ValidationError
from test_adapters import spec_for
from typer.testing import CliRunner

from turkish_token_tax.cli import app
from turkish_token_tax.data import digest
from turkish_token_tax.domains import (
    DomainCorpus,
    clean_and_sample,
    domain_table,
    load_domain,
    load_domain_specs,
    sample_key,
)

TR = "merhaba dunya\n\nmerhaba dunya\nhello hello\nbir iki\n"
EN = "hello world\nempty tr\nhello world\nhello\none two\n"


def domain_fixture(tmp_path, tr=TR, en=EN, pairs=5, sample_size=10, name="fixture-news"):
    tr_bytes, en_bytes = tr.encode(), en.encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("c.tr", tr_bytes)
        bundle.writestr("c.en", en_bytes)
    data = buffer.getvalue()
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / digest(data)).write_bytes(data)
    return (
        {
            "name": name,
            "domain": "news",
            "license": "CC-BY-4.0",
            "source": "https://example.org/source",
            "attribution": "Fixture authors (CC BY 4.0).",
            "archive": {"url": "https://example.org/c.zip", "sha256": digest(data)},
            "tr_member": "c.tr",
            "en_member": "c.en",
            "tr_sha256": digest(tr_bytes),
            "en_sha256": digest(en_bytes),
            "pairs": pairs,
            "sample_size": sample_size,
        },
        cache,
    )


def test_cleaning_drops_blanks_and_duplicates_and_samples_deterministically() -> None:
    sample, stats = clean_and_sample(TR.splitlines(), EN.splitlines(), size=2)

    assert stats == {
        "raw_pairs": 5,
        "blank_dropped": 1,
        "duplicates_dropped": 1,
        "clean_pairs": 3,
        "sampled_pairs": 2,
    }
    assert [p["id"] for p in sample] == ["1", "2"]
    keys = [sample_key(p) for p in sample]
    assert keys == sorted(keys)
    reordered, _ = clean_and_sample(TR.splitlines()[::-1], EN.splitlines()[::-1], size=2)
    assert [(p["tr"], p["en"]) for p in reordered] == [(p["tr"], p["en"]) for p in sample]


def test_load_domain_verifies_digests_and_alignment(tmp_path) -> None:
    spec, cache = domain_fixture(tmp_path)
    pairs, stats = load_domain(DomainCorpus.model_validate(spec), cache, offline=True)

    assert stats["clean_pairs"] == 3
    assert {p["tr"] for p in pairs} == {"merhaba dunya", "hello hello", "bir iki"}

    with pytest.raises(ValueError, match="Misaligned"):
        load_domain(DomainCorpus.model_validate({**spec, "pairs": 4}), cache, offline=True)
    with pytest.raises(ValueError, match="integrity"):
        load_domain(DomainCorpus.model_validate({**spec, "tr_sha256": "0" * 64}), cache, True)
    with pytest.raises(ValueError, match="Not a corpus file"):
        load_domain(DomainCorpus.model_validate({**spec, "tr_member": "x"}), cache, True)


def test_domain_lock_validation(tmp_path) -> None:
    spec, _ = domain_fixture(tmp_path)
    lock = tmp_path / "domains.json"
    lock.write_text(json.dumps([spec, spec]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_domain_specs(lock)
    with pytest.raises(ValidationError):
        DomainCorpus.model_validate({**spec, "domain": "sports"})
    with pytest.raises(ValidationError):
        DomainCorpus.model_validate({**spec, "source": "http://insecure"})


def row(name: str, ratio: float) -> dict:
    return {"name": name, "tr_en_token_ratio": ratio}


def test_domain_table_sorts_by_mean_ratio() -> None:
    report = {
        "domains": [
            {"domain": "news", "tokenizers": [row("b", 2.0), row("a", 1.0)]},
            {"domain": "technical", "tokenizers": [row("b", 1.0), row("a", 1.5)]},
        ]
    }

    lines = domain_table(report).splitlines()

    assert lines[0] == "| Tokenizer | news | technical |"
    assert lines[2] == "| a | 1.0000 | 1.5000 |"
    assert lines[3] == "| b | 2.0000 | 1.0000 |"


def test_cli_measures_domains_and_maintains_the_readme_table(tmp_path) -> None:
    spec, cache = domain_fixture(tmp_path, tr="hello\n", en="hello hello\n", pairs=1)
    domain_lock = tmp_path / "domains.json"
    domain_lock.write_text(json.dumps([spec]), encoding="utf-8")
    tokenizer = spec_for(cache).model_dump()
    tokenizer_lock = tmp_path / "tokenizers.json"
    tokenizer_lock.write_text(json.dumps([tokenizer]), encoding="utf-8")
    out = tmp_path / "out" / "domains.json"
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["domains", "--tokenizer-lock", str(tokenizer_lock), "--domain-lock", str(domain_lock)]
        + ["--cache", str(cache), "--out", str(out), "--offline"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["domains"][0]["tokenizers"][0]["tr_en_token_ratio"] == 0.5
    assert report["domains"][0]["license"] == "CC-BY-4.0"
    assert "hello" not in json.dumps(report["domains"][0]["stats"])

    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- domains:start -->\nold\n<!-- domains:end -->\n", encoding="utf-8")
    table_args = ["domain-table", "--report", str(out), "--readme", str(readme)]
    assert runner.invoke(app, [*table_args, "--check"]).exit_code == 1
    assert runner.invoke(app, table_args).exit_code == 0
    assert runner.invoke(app, [*table_args, "--check"]).exit_code == 0
    assert "| fixture | 0.5000 |" in readme.read_text(encoding="utf-8")


def test_cli_domains_reports_failures_and_bad_markers(tmp_path) -> None:
    spec, cache = domain_fixture(tmp_path)
    domain_lock = tmp_path / "domains.json"
    domain_lock.write_text(json.dumps([{**spec, "pairs": 99}]), encoding="utf-8")
    tokenizer_lock = tmp_path / "tokenizers.json"
    tokenizer_lock.write_text(json.dumps([spec_for(cache).model_dump()]), encoding="utf-8")
    runner = CliRunner()

    failed = runner.invoke(
        app,
        ["domains", "--tokenizer-lock", str(tokenizer_lock), "--domain-lock", str(domain_lock)]
        + ["--cache", str(cache), "--offline", "--out", str(tmp_path / "o.json")],
    )
    assert failed.exit_code == 1
    assert "Misaligned" in failed.output

    report = tmp_path / "r.json"
    report.write_text(json.dumps({"domains": []}), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n", encoding="utf-8")
    bad = runner.invoke(app, ["domain-table", "--report", str(report), "--readme", str(readme)])
    assert bad.exit_code == 2
