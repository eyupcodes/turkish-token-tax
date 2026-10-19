"""Local draft assets for a Kaggle dataset and a Hugging Face Space.

Nothing here publishes anything. The drafts are files on disk that a person
(or the GH OS panel, behind its approval window) reviews before any upload.

Only aggregate measurements leave the benchmark; no corpus sentence is copied.
Because the numbers are derived from FLORES-200 (CC-BY-SA-4.0), the drafts
carry the same license and attribution, and the manifest records the source
hashes so a reviewer can trace every published number back to its inputs.
"""

import csv
import hashlib
import html
import io
import json
from pathlib import Path

from .report import leaderboard

DATASET_SLUG = "eypfidan/turkish-token-tax"
SPACE_REPO = "eyupcodes/turkish-token-tax"
DATA_LICENSE = "CC-BY-SA-4.0"
ATTRIBUTION = (
    "Derived from FLORES-200 devtest (NLLB Team, 2022, "
    "https://github.com/facebookresearch/flores), licensed CC-BY-SA-4.0. "
    "Only aggregate token counts are included; no corpus text."
)
CSV_FIELDS = [
    "tokenizer",
    "model",
    "revision",
    "tr_en_token_ratio",
    "tr_tokens_per_word",
    "en_tokens_per_word",
    "tr_characters_per_token",
    "en_characters_per_token",
    "tr_tokens",
    "en_tokens",
]


def _rows(report: dict) -> list[dict]:
    rows = []
    for row in sorted(report["tokenizers"], key=lambda r: (r["tr_en_token_ratio"], r["name"])):
        rows.append(
            {
                "tokenizer": row["name"],
                "model": row["lock"]["model"],
                "revision": row["lock"]["revision"],
                "tr_en_token_ratio": f"{row['tr_en_token_ratio']:.6f}",
                "tr_tokens_per_word": f"{row['tr']['tokens_per_word']:.6f}",
                "en_tokens_per_word": f"{row['en']['tokens_per_word']:.6f}",
                "tr_characters_per_token": f"{row['tr']['characters_per_token']:.6f}",
                "en_characters_per_token": f"{row['en']['characters_per_token']:.6f}",
                "tr_tokens": str(row["tr"]["tokens"]),
                "en_tokens": str(row["en"]["tokens"]),
            }
        )
    return rows


def leaderboard_csv(report: dict) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_rows(report))
    return buffer.getvalue()


def kaggle_metadata(report: dict) -> dict:
    return {
        "title": "Turkish Token Tax: tokenizer efficiency TR vs EN",
        "id": DATASET_SLUG,
        "subtitle": "How many more tokens Turkish needs than English, per tokenizer",
        "description": (
            "Aggregate token counts for Turkish and English on the "
            f"{report['pair_count']}-pair FLORES-200 devtest split, measured "
            "with pinned tokenizer revisions. The TR/EN ratio is the 'token tax'. " + ATTRIBUTION
        ),
        "licenses": [{"name": DATA_LICENSE}],
        "keywords": ["nlp", "turkish", "tokenization", "benchmark", "llm"],
        "resources": [
            {"path": "leaderboard.csv", "description": "One row per tokenizer."},
            {"path": "benchmark.json", "description": "Full report with locks."},
        ],
        "isPrivate": True,
    }


def space_readme(report: dict) -> str:
    header = "\n".join(
        [
            "---",
            "title: Turkish Token Tax",
            "emoji: 🇹🇷",
            "colorFrom: red",
            "colorTo: gray",
            "sdk: static",
            "app_file: index.html",
            f"license: {DATA_LICENSE.lower()}",
            "pinned: false",
            "---",
        ]
    )
    return (
        f"{header}\n\n# Turkish Token Tax\n\n"
        "How many more tokens does Turkish need than English for the same "
        f"meaning? Measured on {report['pair_count']} FLORES-200 devtest pairs.\n\n"
        f"{leaderboard(report)}\n{ATTRIBUTION}\n\n"
        "Source code: https://github.com/eyupcodes/turkish-token-tax\n"
    )


def space_index(report: dict) -> str:
    cells = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(row[field])}</td>"
            for field in ("tokenizer", "tr_en_token_ratio", "tr_tokens_per_word")
        )
        + "</tr>"
        for row in _rows(report)
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Turkish Token Tax</title><style>"
        "body{font:17px system-ui;max-width:52rem;margin:2rem auto;padding:1rem}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.5rem;"
        "border-bottom:1px solid #ccc;text-align:left}</style>"
        "<h1>Turkish Token Tax</h1>"
        f"<p>Tokens needed for Turkish relative to English on {report['pair_count']} "
        "FLORES-200 devtest pairs. Lower is cheaper for Turkish.</p>"
        "<table><tr><th>Tokenizer</th><th>TR / EN</th><th>TR tokens / word</th></tr>"
        f"{cells}</table><p><small>{html.escape(ATTRIBUTION)}</small></p></html>\n"
    )


def build_drafts(report: dict, out: Path) -> dict:
    """Write every draft file and a manifest of their hashes; return the manifest."""
    benchmark = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    files = {
        "kaggle/dataset-metadata.json": json.dumps(
            kaggle_metadata(report), ensure_ascii=False, indent=2
        )
        + "\n",
        "kaggle/leaderboard.csv": leaderboard_csv(report),
        "kaggle/benchmark.json": benchmark,
        "hf-space/README.md": space_readme(report),
        "hf-space/index.html": space_index(report),
    }
    for relative, text in files.items():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "status": "draft, not published",
        "targets": {"kaggle_dataset": DATASET_SLUG, "hf_space": SPACE_REPO},
        "license": DATA_LICENSE,
        "attribution": ATTRIBUTION,
        "corpus": report["corpus"]["name"],
        "corpus_lock_sha256": report["corpus_lock_sha256"],
        "tokenizer_lock_sha256": report["tokenizer_lock_sha256"],
        "files": {
            relative: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for relative, text in sorted(files.items())
        },
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
