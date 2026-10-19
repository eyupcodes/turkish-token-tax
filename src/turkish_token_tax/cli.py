"""Run the locked benchmark; paths are relative to the YAML config, not the shell."""

import importlib.metadata
import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .adapters import create_adapter, load_specs
from .data import digest, load_corpus
from .drafts import build_drafts
from .metrics import compare
from .report import leaderboard, write_reports

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
):
    """Replace only the generated leaderboard block in a README."""
    begin = "<!-- leaderboard:start -->"
    end = "<!-- leaderboard:end -->"
    text = readme.read_text(encoding="utf-8")
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise typer.BadParameter("README must contain one ordered pair of leaderboard markers")
    data = json.loads(report.read_text(encoding="utf-8"))
    table = leaderboard(data)
    before, rest = text.split(begin, 1)
    _, after = rest.split(end, 1)
    readme.write_text(before + begin + "\n" + table + end + after, encoding="utf-8")
    typer.echo(f"Updated {readme}")


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
