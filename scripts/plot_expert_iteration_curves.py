from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_points(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_points: list[dict[str, Any]] = []
    entropy_points: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        metadata = {
            "run_name": run_dir.name,
            "ei_batch_size": summary.get(
                "ei_batch_size", config.get("ei_batch_size", "")
            ),
            "rollouts_per_question": summary.get(
                "rollouts_per_question", config.get("rollouts_per_question", "")
            ),
            "sft_epochs_per_ei_step": summary.get(
                "sft_epochs_per_ei_step", config.get("sft_epochs_per_ei_step", "")
            ),
        }
        for row in read_metrics(run_dir / "metrics.jsonl"):
            if "eval/accuracy" in row:
                eval_points.append(
                    {
                        **metadata,
                        "ei_step": int(row.get("eval_step", row.get("ei_step", 0))),
                        "train_step": int(row.get("train_step", 0)),
                        "eval_accuracy": float(row["eval/accuracy"]),
                        "format_accuracy": float(row.get("eval/format_accuracy", 0.0)),
                    }
                )
            if "train/response_entropy" in row:
                entropy_points.append(
                    {
                        **metadata,
                        "ei_step": int(row.get("ei_step", 0)),
                        "train_step": int(row.get("train_step", 0)),
                        "response_entropy": float(row["train/response_entropy"]),
                    }
                )
    return eval_points, entropy_points


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_svg(
    points: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
) -> str:
    width = 960
    height = 560
    left = 80
    right = 24
    top = 32
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#9333ea",
        "#d97706",
        "#0891b2",
        "#4b5563",
        "#be123c",
    ]

    by_run: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_run.setdefault(point["run_name"], []).append(point)

    max_x = max((point[x_key] for point in points), default=1)
    max_y = max((point[y_key] for point in points), default=0.0)
    y_max = max(0.2, min(max(1.0, max_y * 1.2), max_y * 1.2 if max_y > 1 else 1.0))

    def x_of(value: int) -> float:
        return left + (value / max(1, max_x)) * plot_w

    def y_of(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_h

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="16">{html.escape(x_label)}</text>',
        (
            f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="16" '
            'transform="rotate(-90 20 '
            f'{height / 2})">{html.escape(y_label)}</text>'
        ),
    ]

    for i in range(6):
        value = y_max * i / 5
        y = y_of(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="12">{value:.2f}</text>'
        )

    legend_y = top
    for idx, (run_name, run_points) in enumerate(by_run.items()):
        run_points = sorted(run_points, key=lambda row: row[x_key])
        color = colors[idx % len(colors)]
        coords = " ".join(
            f'{x_of(point[x_key]):.2f},{y_of(point[y_key]):.2f}'
            for point in run_points
        )
        if coords:
            parts.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" '
                'stroke-width="2.5"/>'
            )
        for point in run_points:
            parts.append(
                f'<circle cx="{x_of(point[x_key]):.2f}" '
                f'cy="{y_of(point[y_key]):.2f}" r="4" fill="{color}"/>'
            )
        label = html.escape(run_name)
        parts.append(
            f'<rect x="{left + 18}" y="{legend_y - 11}" width="12" height="12" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + 36}" y="{legend_y}" font-size="13">{label}</text>'
        )
        legend_y += 18

    parts.append("</svg>")
    return "\n".join(parts)


def write_html(
    path: Path,
    eval_points: list[dict[str, Any]],
    entropy_points: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eval_svg = build_svg(
        eval_points,
        x_key="ei_step",
        y_key="eval_accuracy",
        x_label="EI step",
        y_label="validation accuracy",
    )
    entropy_svg = build_svg(
        entropy_points,
        x_key="train_step",
        y_key="response_entropy",
        x_label="optimizer step",
        y_label="response entropy",
    )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(point['run_name']))}</td>"
        f"<td>{point['ei_batch_size']}</td>"
        f"<td>{point['rollouts_per_question']}</td>"
        f"<td>{point['sft_epochs_per_ei_step']}</td>"
        f"<td>{point['ei_step']}</td>"
        f"<td>{point['eval_accuracy']:.4f}</td>"
        f"<td>{point['format_accuracy']:.4f}</td>"
        "</tr>"
        for point in eval_points
    )
    path.write_text(
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'><title>Expert Iteration Curves</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;margin-top:24px}"
        "td,th{border:1px solid #d1d5db;padding:6px 10px}"
        "th{background:#f3f4f6}</style></head><body>"
        "<h1>Expert Iteration Validation Accuracy</h1>"
        f"{eval_svg}"
        "<h1>Expert Iteration Response Entropy</h1>"
        f"{entropy_svg}"
        "<table><thead><tr><th>run</th><th>Db</th><th>G</th>"
        "<th>SFT epochs</th><th>EI step</th><th>accuracy</th>"
        "<th>format accuracy</th></tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
        "</body></html>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot expert iteration curves.")
    parser.add_argument("--output-dir", default="outputs/expert_iteration")
    parser.add_argument(
        "--eval-csv-out",
        default="outputs/expert_iteration/validation_accuracy_curves.csv",
    )
    parser.add_argument(
        "--entropy-csv-out",
        default="outputs/expert_iteration/response_entropy_curves.csv",
    )
    parser.add_argument(
        "--html-out",
        default="outputs/expert_iteration/expert_iteration_curves.html",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_points, entropy_points = collect_points(Path(args.output_dir))
    if not eval_points:
        raise SystemExit(f"No eval points found under {args.output_dir}")
    write_csv(
        Path(args.eval_csv_out),
        eval_points,
        [
            "run_name",
            "ei_batch_size",
            "rollouts_per_question",
            "sft_epochs_per_ei_step",
            "ei_step",
            "train_step",
            "eval_accuracy",
            "format_accuracy",
        ],
    )
    write_csv(
        Path(args.entropy_csv_out),
        entropy_points,
        [
            "run_name",
            "ei_batch_size",
            "rollouts_per_question",
            "sft_epochs_per_ei_step",
            "ei_step",
            "train_step",
            "response_entropy",
        ],
    )
    write_html(Path(args.html_out), eval_points, entropy_points)
    print(f"Wrote {args.eval_csv_out}")
    print(f"Wrote {args.entropy_csv_out}")
    print(f"Wrote {args.html_out}")
