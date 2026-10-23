"""Run the locked benchmark; paths are relative to the YAML config, not the shell."""

import importlib.metadata
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .adapters import create_adapter, load_specs
from .data import digest, load_corpus
from .domains import domain_table, load_domain, load_domain_specs
from .drafts import build_drafts
from .metrics import compare
from .report import chart_svg, leaderboard, write_reports

app = typer.Typer(no_args_is_help=True)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus_lock: Path
    tokenizer_lock: Path
    cache: Path = Path("../.cache")
    output: Path = Path("../results")
    context_tokens: int = Field(default=8192, gt=0)


def run_benchmark(config: Path, offline: bool = False) -> tuple[dict, Path]:
    config = config.resolve()
    settings = RunConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    base = config.parent
    corpus_lock = base / settings.corpus_lock
    tokenizer_lock = base / settings.tokenizer_lock
    cache = base / settings.cache
    corpus, pairs = load_corpus(corpus_lock, cache, offline)
    rows = []
    for spec in load_specs(tokenizer_lock):
        tokenizer = create_adapter(spec, cache, offline)
        rows.append(
            {
                "name": spec.name,
                "lock": spec.model_dump(),
                **compare(pairs, tokenizer, settings.context_tokens),
            }
        )
    kinds = {row["lock"]["kind"] for row in rows}
    libraries = []
    if "tiktoken" in kinds:
        libraries.append("tiktoken")
    if kinds & {"hf-json", "bert-vocab"}:
        libraries.append("tokenizers")
    report = {
        "schema_version": 1,
        "version": __version__,
        "runtime": {name: importlib.metadata.version(name) for name in libraries},
        "corpus": corpus.model_dump(),
        "pair_count": len(pairs),
        "corpus_lock_sha256": digest(corpus_lock.read_bytes()),
        "tokenizer_lock_sha256": digest(tokenizer_lock.read_bytes()),
        "context_tokens": settings.context_tokens,
        "tokenizers": rows,
    }
    # Assert JSON serializability before any report files are written.
    json.dumps(report, allow_nan=False)
    return report, base / settings.output


@contextmanager
def report_errors(report: Path) -> Iterator[None]:
    """Turn an unreadable or malformed saved report into a one-line error, exit 1."""
    try:
        yield
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        detail = f"missing key {error}" if isinstance(error, KeyError) else str(error)
        typer.echo(f"error: {report}: not a usable report ({detail})", err=True)
        raise typer.Exit(1) from error


def render_report(report: Path, render: Callable[[dict], str]) -> str:
    with report_errors(report):
        return render(json.loads(report.read_text(encoding="utf-8")))


@app.command()
def run(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
    offline: Annotated[bool, typer.Option(help="Use verified cache only; never download.")] = False,
):
    """Measure all locked tokenizers on all aligned pairs and write three reports."""
    try:
        report, output = run_benchmark(config, offline)
        write_reports(report, output)
    except (ValueError, OSError, KeyError, ImportError) as error:
        typer.echo(f"Benchmark failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Measured {report['pair_count']} pairs with {len(report['tokenizers'])} tokenizers."
    )
    typer.echo(f"Reports: {output.resolve()}")


@app.command("leaderboard")
def refresh_leaderboard(
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    readme: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("README.md"),
    check: Annotated[
        bool, typer.Option(help="Fail if the README table is stale; change nothing.")
    ] = False,
):
    """Replace only the generated leaderboard block in a README."""
    begin = "<!-- leaderboard:start -->"
    end = "<!-- leaderboard:end -->"
    text = readme.read_text(encoding="utf-8")
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise typer.BadParameter("README must contain one ordered pair of leaderboard markers")
    table = render_report(report, leaderboard)
    before, rest = text.split(begin, 1)
    current, after = rest.split(end, 1)
    if check:
        # Compare without line-ending noise so a Windows checkout agrees with CI.
        if current.replace("\r\n", "\n").strip() != table.strip():
            typer.echo(f"{readme} leaderboard is out of date; run ttt leaderboard", err=True)
            raise typer.Exit(1)
        typer.echo(f"{readme} leaderboard matches {report}")
        return
    readme.write_text(before + begin + "\n" + table + end + after, encoding="utf-8")
    typer.echo(f"Updated {readme}")


def replace_block(text: str, name: str, block: str) -> tuple[str, str]:
    """Return (current block, text with the block replaced) between name markers."""
    begin, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise typer.BadParameter(f"README must contain one ordered pair of {name} markers")
    before, rest = text.split(begin, 1)
    current, after = rest.split(end, 1)
    return current, before + begin + "\n" + block + end + after


@app.command()
def domains(
    tokenizer_lock: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/tokenizers.lock.json"
    ),
    domain_lock: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/domains.lock.json"
    ),
    cache: Annotated[Path, typer.Option(file_okay=False)] = Path(".cache"),
    out: Annotated[Path, typer.Option(dir_okay=False)] = Path("benchmarks/domains.json"),
    context_tokens: Annotated[int, typer.Option(min=1)] = 8192,
    offline: Annotated[bool, typer.Option(help="Use verified cache only; never download.")] = False,
):
    """Measure every locked tokenizer on each licensed domain subset."""
    try:
        tokenizers = [
            (spec, create_adapter(spec, cache, offline)) for spec in load_specs(tokenizer_lock)
        ]
        results = []
        for spec in load_domain_specs(domain_lock):
            pairs, stats = load_domain(spec, cache, offline)
            results.append(
                {
                    **spec.model_dump(include={"name", "domain", "license", "source"}),
                    "attribution": spec.attribution,
                    "stats": stats,
                    "tokenizers": [
                        {"name": t.name, **compare(pairs, adapter, context_tokens)}
                        for t, adapter in tokenizers
                    ],
                }
            )
    except (ValueError, OSError, KeyError, ImportError) as error:
        typer.echo(f"Domain benchmark failed: {error}", err=True)
        raise typer.Exit(1) from error
    report = {
        "schema_version": 1,
        "version": __version__,
        "tokenizer_lock_sha256": digest(tokenizer_lock.read_bytes()),
        "domain_lock_sha256": digest(domain_lock.read_bytes()),
        "context_tokens": context_tokens,
        "domains": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    typer.echo(f"Measured {len(results)} domains with {len(tokenizers)} tokenizers: {out}")


@app.command("domain-table")
def refresh_domain_table(
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "benchmarks/domains.json"
    ),
    readme: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("README.md"),
    check: Annotated[
        bool, typer.Option(help="Fail if the README table is stale; change nothing.")
    ] = False,
):
    """Write (or check) the per-domain table between README domains markers."""
    table = render_report(report, domain_table)
    current, updated = replace_block(readme.read_text(encoding="utf-8"), "domains", table)
    if check:
        if current.replace("\r\n", "\n").strip() != table.strip():
            typer.echo(f"{readme} domain table is out of date; run ttt domain-table", err=True)
            raise typer.Exit(1)
        typer.echo(f"{readme} domain table matches {report}")
        return
    readme.write_text(updated, encoding="utf-8")
    typer.echo(f"Updated {readme}")


@app.command()
def chart(
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "benchmarks/flores200-devtest.json"
    ),
    out: Annotated[Path, typer.Option(dir_okay=False)] = Path("benchmarks/flores200-devtest.svg"),
):
    """Draw the TR/EN ratio chart (SVG) from a saved report."""
    out.write_text(render_report(report, chart_svg), encoding="utf-8", newline="\n")
    typer.echo(f"Wrote {out}")


@app.command()
def drafts(
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "benchmarks/flores200-devtest.json"
    ),
    out: Annotated[Path, typer.Option(file_okay=False)] = Path("drafts"),
):
    """Write local Kaggle dataset and Hugging Face Space drafts. Publishes nothing."""
    manifest = build_drafts(json.loads(report.read_text(encoding="utf-8")), out)
    typer.echo(f"Wrote {len(manifest['files'])} draft files and MANIFEST.json to {out}")
    typer.echo("Nothing was published; uploads need explicit approval.")


@app.command()
def version():
    """Show package version."""
    typer.echo(__version__)
