import hashlib
import json
import pathlib

import pytest
from test_adapters import spec_for
from test_data import corpus_fixture
from test_domains import domain_fixture
from typer.testing import CliRunner

from turkish_token_tax.cli import app, run_benchmark
from turkish_token_tax.data import digest
from turkish_token_tax.report import write_reports


def configuration(tmp_path):
    corpus, cache = corpus_fixture(tmp_path)
    spec = spec_for(cache)
    lock = tmp_path / "tokenizers.json"
    lock.write_text(json.dumps([spec.model_dump()]))
    config = tmp_path / "config.yaml"
    config.write_text(
        "corpus_lock: corpus.json\ntokenizer_lock: tokenizers.json\n"
        "cache: cache\noutput: results\ncontext_tokens: 100\n"
    )
    return config


def test_cli_determinism_and_paths(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    runner = CliRunner()
    first = runner.invoke(app, ["run", "--config", str(config), "--offline"])
    assert first.exit_code == 0, first.output
    output = tmp_path / "results"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    second = runner.invoke(app, ["run", "--config", str(config), "--offline"])
    assert second.exit_code == 0
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()} == hashes
    report = json.loads((output / "benchmark.json").read_text())
    assert report["pair_count"] == 1
    assert report["runtime"]["tokenizers"]
    assert report["tokenizers"][0]["lock"]["revision"] == "a" * 40
    assert "0.1.0" in runner.invoke(app, ["version"]).output
    assert runner.invoke(app, ["--help"]).exit_code == 0


def test_failure_and_unknown_config_key(tmp_path):
    config = configuration(tmp_path)
    config.write_text(config.read_text() + "unknown_option: true\n")
    result = CliRunner().invoke(app, ["run", "--config", str(config), "--offline"])
    assert result.exit_code == 1
    assert "Benchmark failed" in result.output


def test_report_escaping(tmp_path):
    config = configuration(tmp_path)
    report, output = run_benchmark(config, offline=True)
    report["corpus"]["name"] = "<script>alert(1)</script>"
    write_reports(report, output)
    document = (output / "benchmark.html").read_text()
    assert "<script>" not in document
    assert "&lt;script&gt;" in document


def test_generated_leaderboard_preserves_prose(tmp_path):
    config = configuration(tmp_path)
    report, output = run_benchmark(config, offline=True)
    write_reports(report, output)
    readme = tmp_path / "README.md"
    readme.write_text("Before\n<!-- leaderboard:start -->old<!-- leaderboard:end -->\nAfter\n")
    result = CliRunner().invoke(
        app, ["leaderboard", "--report", str(output / "benchmark.json"), "--readme", str(readme)]
    )
    assert result.exit_code == 0, result.output
    text = readme.read_text()
    assert text.startswith("Before\n") and text.endswith("\nAfter\n")
    assert "| fixture |" in text and "old" not in text
    for invalid in (
        "no markers",
        "<!-- leaderboard:end --><!-- leaderboard:start -->",
        "<!-- leaderboard:start --><!-- leaderboard:start --><!-- leaderboard:end -->",
    ):
        readme.write_text(invalid)
        result = CliRunner().invoke(
            app,
            ["leaderboard", "--report", str(output / "benchmark.json"), "--readme", str(readme)],
        )
        assert result.exit_code == 2
        assert readme.read_text() == invalid


@pytest.mark.parametrize("command", ["leaderboard", "domain-table", "chart"])
@pytest.mark.parametrize("content", ["{not json", "{}", "[]"])
def test_invalid_report_is_a_one_line_error(tmp_path, command, content):
    report = tmp_path / "report.json"
    report.write_text(content, encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("<!-- leaderboard:start --><!-- leaderboard:end -->\n")
    args = [command, "--report", str(report)]
    if command == "chart":
        args += ["--out", str(tmp_path / "chart.svg")]
    else:
        args += ["--readme", str(readme)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert f"error: {report}: not a usable report" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "chart.svg").exists()


def test_run_reports_invalid_yaml_config_without_a_traceback(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("corpus_lock: [unclosed\n")

    result = CliRunner().invoke(app, ["run", "--config", str(config), "--offline"])

    assert result.exit_code == 1
    assert "Benchmark failed" in result.output
    assert "invalid YAML" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "results").exists()


def test_run_reports_unreadable_corpus_archive_without_a_traceback(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    raw = b"definitely not a tar.gz"
    (cache / digest(raw)).write_bytes(raw)
    (tmp_path / "corpus.lock.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "license": "MIT",
                "source": "https://example.org/source",
                "archive": {"url": "https://example.org/data", "sha256": digest(raw)},
                "tr_member": "tr",
                "en_member": "en",
                "tr_sha256": "0" * 64,
                "en_sha256": "0" * 64,
                "pairs": 1,
            }
        )
    )
    (tmp_path / "tokenizers.lock.json").write_text("[]")
    config = tmp_path / "config.yaml"
    config.write_text(
        "corpus_lock: corpus.lock.json\ntokenizer_lock: tokenizers.lock.json\n"
        "cache: cache\noutput: results\ncontext_tokens: 8192\n"
    )

    result = CliRunner().invoke(app, ["run", "--config", str(config), "--offline"])

    assert result.exit_code == 1
    assert "not a readable tar.gz" in result.output
    assert "Traceback" not in result.output


BENCHMARKS = pathlib.Path(__file__).parents[1] / "benchmarks"


def blocked_path(tmp_path, name):
    """A path whose parent is a regular file, so it can never be written."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    return blocker / name


def assert_write_error(result, target):
    assert result.exit_code == 1, result.output
    assert f"error: cannot write {target}:" in result.output
    assert "Traceback" not in result.output


def test_chart_reports_an_unwritable_output(tmp_path):
    out = blocked_path(tmp_path, "chart.svg")
    report = BENCHMARKS / "flores200-devtest.json"
    result = CliRunner().invoke(app, ["chart", "--report", str(report), "--out", str(out)])
    assert_write_error(result, out)


def test_drafts_reports_an_unwritable_output(tmp_path):
    out = blocked_path(tmp_path, "drafts")
    report = BENCHMARKS / "flores200-devtest.json"
    result = CliRunner().invoke(app, ["drafts", "--report", str(report), "--out", str(out)])
    assert_write_error(result, out)


def test_domains_reports_an_unwritable_output(tmp_path):
    spec, cache = domain_fixture(tmp_path, tr="hello\n", en="hello hello\n", pairs=1)
    domain_lock = tmp_path / "domains.json"
    domain_lock.write_text(json.dumps([spec]), encoding="utf-8")
    tokenizer_lock = tmp_path / "tokenizers.json"
    tokenizer_lock.write_text(json.dumps([spec_for(cache).model_dump()]), encoding="utf-8")
    out = blocked_path(tmp_path, "domains.json")
    args = ["domains", "--tokenizer-lock", str(tokenizer_lock), "--domain-lock", str(domain_lock)]
    result = CliRunner().invoke(app, [*args, "--cache", str(cache), "--out", str(out), "--offline"])
    assert_write_error(result, out)


@pytest.mark.parametrize(
    ("command", "report"),
    [("leaderboard", "flores200-devtest.json"), ("domain-table", "domains.json")],
)
def test_readme_commands_report_an_unwritable_readme(tmp_path, monkeypatch, command, report):
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- leaderboard:start -->\nold\n<!-- leaderboard:end -->\n"
        "<!-- domains:start -->\nold\n<!-- domains:end -->\n",
        encoding="utf-8",
    )
    original = pathlib.Path.write_text

    def deny_readme(self, *args, **kwargs):
        if self == readme:
            raise PermissionError(13, "Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", deny_readme)
    result = CliRunner().invoke(
        app, [command, "--report", str(BENCHMARKS / report), "--readme", str(readme)]
    )
    assert_write_error(result, readme)
    assert "old" in readme.read_text(encoding="utf-8")
