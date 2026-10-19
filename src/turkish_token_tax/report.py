"""Deterministic, text-free JSON, Markdown and escaped HTML reports."""

import html
import json
from pathlib import Path


def leaderboard(report: dict) -> str:
    rows = ["| Tokenizer | TR / EN | TR tokens / word | TR chars / token |", "|---|---:|---:|---:|"]
    for row in sorted(report["tokenizers"], key=lambda r: (r["tr_en_token_ratio"], r["name"])):
        rows.append(
            f"| {row['name']} | {row['tr_en_token_ratio']:.4f} | "
            f"{row['tr']['tokens_per_word']:.4f} | "
            f"{row['tr']['characters_per_token']:.4f} |"
        )
    return "\n".join(rows) + "\n"


def write_reports(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    table = leaderboard(report)
    caption = (
        f"Corpus: {report['corpus']['name']}; {report['pair_count']} aligned pairs. "
        "Ratios compare total tokens, not API prices or model quality.\n\n"
    )
    (output / "benchmark.md").write_text(caption + table, encoding="utf-8")
    bars = []
    for row in report["tokenizers"]:
        ratio = row["tr_en_token_ratio"]
        bars.append(
            f"<li>{html.escape(row['name'])}: <meter min='0' max='4' "
            f"value='{ratio:.6f}'>{ratio:.4f}</meter> {ratio:.4f}</li>"
        )
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Turkish Token Tax</title><style>"
        "body{font:18px system-ui;max-width:70rem;margin:3rem auto;padding:1rem}"
        "pre{overflow:auto}li{margin:1rem 0}meter{width:15rem}</style>"
        "<h1>Turkish Token Tax</h1><p>"
        + html.escape(caption)
        + "</p><pre>"
        + html.escape(table)
        + "</pre><h2>TR / EN token ratio</h2><ul>"
        + "".join(bars)
        + "</ul><p>Byte fallback unavailable is reported as null; "
        "context capacity is a density estimate, not a fitted document count.</p></html>"
    )
    (output / "benchmark.html").write_text(document, encoding="utf-8")
