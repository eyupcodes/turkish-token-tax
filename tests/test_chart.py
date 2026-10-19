import json

from test_drafts import REPORT
from typer.testing import CliRunner

from turkish_token_tax.cli import app
from turkish_token_tax.report import chart_svg, leaderboard


def test_chart_is_sorted_escaped_and_deterministic() -> None:
    svg = chart_svg(REPORT)

    assert svg.startswith("<svg xmlns='http://www.w3.org/2000/svg'")
    assert svg.index(">light<") < svg.index(">heavy&lt;b&gt;<")
    assert "heavy<b>" not in svg
    assert "2.50×" in svg and "1.08×" in svg
    assert "parity (1.0)" in svg
    assert chart_svg(REPORT) == svg


def test_chart_colours_follow_the_ratio() -> None:
    cheap = {**REPORT, "tokenizers": [{**REPORT["tokenizers"][1], "tr_en_token_ratio": 0.6}]}

    assert "#3FB950" in chart_svg(cheap)
    assert "#F85149" in chart_svg(REPORT)


def write_readme(tmp_path, table: str):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# T\n\n<!-- leaderboard:start -->\n{table}<!-- leaderboard:end -->\n",
        encoding="utf-8",
        newline="\n",
    )
    report = tmp_path / "report.json"
    report.write_text(json.dumps(REPORT), encoding="utf-8")
    return readme, report


def check(readme, report):
    return CliRunner().invoke(
        app, ["leaderboard", "--report", str(report), "--readme", str(readme), "--check"]
    )


def test_leaderboard_check_passes_when_current(tmp_path) -> None:
    readme, report = write_readme(tmp_path, leaderboard(REPORT))
    before = readme.read_bytes()

    result = check(readme, report)

    assert result.exit_code == 0, result.output
    assert readme.read_bytes() == before


def test_leaderboard_check_ignores_crlf(tmp_path) -> None:
    readme, report = write_readme(tmp_path, leaderboard(REPORT))
    readme.write_bytes(readme.read_bytes().replace(b"\n", b"\r\n"))

    assert check(readme, report).exit_code == 0


def test_leaderboard_check_fails_when_stale_and_changes_nothing(tmp_path) -> None:
    readme, report = write_readme(tmp_path, "| stale |\n")
    before = readme.read_bytes()

    result = check(readme, report)

    assert result.exit_code == 1
    assert "out of date" in result.output
    assert readme.read_bytes() == before


def test_chart_command_writes_svg(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(REPORT), encoding="utf-8")
    out = tmp_path / "chart.svg"

    result = CliRunner().invoke(app, ["chart", "--report", str(report), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == chart_svg(REPORT)
