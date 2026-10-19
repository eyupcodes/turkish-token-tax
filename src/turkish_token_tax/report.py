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


def chart_svg(report: dict) -> str:
    """Horizontal bars of the TR/EN token ratio, with a parity line at 1.0.

    Hand-written SVG (no plotting dependency) so the output is byte-stable and
    renders on GitHub. Colours read on both light and dark backgrounds.
    """
    rows = sorted(report["tokenizers"], key=lambda r: (r["tr_en_token_ratio"], r["name"]))
    left, bar_h, gap, top, width = 130, 26, 12, 56, 640
    scale_max = max(2.0, max(r["tr_en_token_ratio"] for r in rows) * 1.1)
    plot_w = width - left - 70
    height = top + len(rows) * (bar_h + gap) + 40

    def x(value: float) -> float:
        return left + plot_w * value / scale_max

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' font-family='system-ui,sans-serif' "
        "role='img' aria-label='Turkish to English token ratio by tokenizer'>",
        "<title>Turkish / English token ratio by tokenizer</title>",
        "<text x='16' y='26' font-size='17' font-weight='700' fill='#58A6FF'>"
        "Turkish / English token ratio</text>",
        f"<text x='16' y='44' font-size='12' fill='#8B949E'>{html.escape(report['corpus']['name'])}"
        f", {report['pair_count']} pairs. Lower means cheaper Turkish.</text>",
    ]
    for i, row in enumerate(rows):
        ratio = row["tr_en_token_ratio"]
        y = top + i * (bar_h + gap)
        colour = "#3FB950" if ratio <= 1.0 else ("#D29922" if ratio < 1.6 else "#F85149")
        parts += [
            f"<text x='{left - 10}' y='{y + 18}' font-size='13' text-anchor='end' "
            f"fill='#8B949E'>{html.escape(row['name'])}</text>",
            f"<rect x='{left}' y='{y}' width='{x(ratio) - left:.1f}' height='{bar_h}' "
            f"rx='4' fill='{colour}'/>",
            f"<text x='{x(ratio) + 8:.1f}' y='{y + 18}' font-size='13' font-weight='600' "
            f"fill='#8B949E'>{ratio:.2f}×</text>",
        ]
    parity = x(1.0)
    parts += [
        f"<line x1='{parity:.1f}' y1='{top - 6}' x2='{parity:.1f}' y2='{height - 30}' "
        "stroke='#8B949E' stroke-dasharray='4 4'/>",
        f"<text x='{parity:.1f}' y='{height - 14}' font-size='11' text-anchor='middle' "
        "fill='#8B949E'>parity (1.0)</text>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


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
    (output / "benchmark.svg").write_text(chart_svg(report), encoding="utf-8", newline="\n")
