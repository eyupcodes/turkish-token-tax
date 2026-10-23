import hashlib
import json

import pytest
from test_adapters import spec_for
from test_data import corpus_fixture
from typer.testing import CliRunner

from turkish_token_tax.cli import app, run_benchmark
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
