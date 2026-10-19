import csv
import hashlib
import io
import json

from typer.testing import CliRunner

from turkish_token_tax.cli import app
from turkish_token_tax.drafts import CSV_FIELDS, build_drafts


def side(tokens: int, words: int, characters: int) -> dict:
    return {
        "tokens": tokens,
        "words": words,
        "characters": characters,
        "tokens_per_word": tokens / words,
        "characters_per_token": characters / tokens,
    }


def row(name: str, tr: int, en: int) -> dict:
    return {
        "name": name,
        "lock": {"model": f"org/{name}", "revision": "a" * 40},
        "tr": side(tr, 10, 60),
        "en": side(en, 10, 55),
        "tr_en_token_ratio": tr / en,
    }


REPORT = {
    "pair_count": 3,
    "corpus": {"name": "fixture-corpus"},
    "corpus_lock_sha256": "c" * 64,
    "tokenizer_lock_sha256": "d" * 64,
    "tokenizers": [row("heavy<b>", 30, 12), row("light", 13, 12)],
}


def test_drafts_contain_only_aggregates_and_a_hashed_manifest(tmp_path) -> None:
    manifest = build_drafts(REPORT, tmp_path)

    assert manifest["status"] == "draft, not published"
    assert manifest["license"] == "CC-BY-SA-4.0"
    for relative, digest in manifest["files"].items():
        data = (tmp_path / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
    on_disk = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_csv_is_sorted_by_tax_and_has_stable_columns(tmp_path) -> None:
    build_drafts(REPORT, tmp_path)
    text = (tmp_path / "kaggle/leaderboard.csv").read_text(encoding="utf-8")

    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0]) == CSV_FIELDS
    assert [r["tokenizer"] for r in rows] == ["light", "heavy<b>"]
    assert rows[1]["tr_en_token_ratio"] == "2.500000"


def test_kaggle_metadata_is_private_and_attributed(tmp_path) -> None:
    build_drafts(REPORT, tmp_path)
    meta = json.loads((tmp_path / "kaggle/dataset-metadata.json").read_text(encoding="utf-8"))

    assert meta["isPrivate"] is True
    assert meta["licenses"] == [{"name": "CC-BY-SA-4.0"}]
    assert "FLORES-200" in meta["description"]
    assert {r["path"] for r in meta["resources"]} == {"leaderboard.csv", "benchmark.json"}


def test_space_files_are_static_and_escaped(tmp_path) -> None:
    build_drafts(REPORT, tmp_path)
    readme = (tmp_path / "hf-space/README.md").read_text(encoding="utf-8")
    index = (tmp_path / "hf-space/index.html").read_text(encoding="utf-8")

    assert readme.startswith("---\ntitle: Turkish Token Tax\n")
    assert "sdk: static" in readme
    # The Hub rejects emoji that are not a single Extended_Pictographic code point,
    # such as flags made of two regional indicators.
    emoji = next(line for line in readme.splitlines() if line.startswith("emoji: "))[7:]
    assert len(emoji) == 1 and not 0x1F1E6 <= ord(emoji) <= 0x1F1FF
    assert "heavy&lt;b&gt;" in index and "heavy<b>" not in index


def test_drafts_are_deterministic(tmp_path) -> None:
    first = build_drafts(REPORT, tmp_path / "a")
    second = build_drafts(REPORT, tmp_path / "b")

    assert first == second


def test_cli_writes_drafts_and_says_nothing_was_published(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(REPORT), encoding="utf-8")
    out = tmp_path / "drafts"

    result = CliRunner().invoke(app, ["drafts", "--report", str(report), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "Nothing was published" in result.output
    assert (out / "MANIFEST.json").exists()
