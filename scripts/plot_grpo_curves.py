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


def collect_eval_points(output_dir: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        for row in read_metrics(run_dir / "metrics.jsonl"):
            if "eval/reward" not in row:
                continue
            points.append(
                {
                    "run_name": run_dir.name,
                    "loss_type": summary.get("loss_type", config.get("loss_type", "")),
                    "group_size": summary.get("group_size", config.get("group_size", "")),
                    "rollout_batch_size": summary.get(
                        "rollout_batch_size",
                        config.get("rollout_batch_size", ""),
                    ),
                    "grpo_step": int(row.get("grpo_step", row.get("eval_step", 0))),
                    "train_step": int(row.get("train_step", 0)),
                    "eval_reward": float(row["eval/reward"]),
                    "eval_accuracy": float(row.get("eval/accuracy", 0.0)),
                    "format_accuracy": float(row.get("eval/format_accuracy", 0.0)),
                }
            )
    return points


def write_csv(path: Path, points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "loss_type",
        "group_size",
        "rollout_batch_size",
        "grpo_step",
        "train_step",
        "eval_reward",
        "eval_accuracy",
        "format_accuracy",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def build_svg(points: list[dict[str, Any]]) -> str:
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
    ]

    by_run: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_run.setdefault(point["run_name"], []).append(point)

    max_step = max((point["grpo_step"] for point in points), default=1)
    max_reward = max((point["eval_reward"] for point in points), default=0.0)
    y_max = max(0.2, min(1.0, max_reward * 1.2))

    def x_of(step: int) -> float:
        return left + (step / max(1, max_step)) * plot_w

    def y_of(reward: float) -> float:
        return top + (1.0 - reward / y_max) * plot_h

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="16">GRPO step</text>',
        (
            f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="16" '
            'transform="rotate(-90 20 '
            f'{height / 2})">validation reward</text>'
        ),
    ]

    for i in range(6):
        reward = y_max * i / 5
        y = y_of(reward)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="12">{reward:.2f}</text>'
        )

    legend_y = top
    for idx, (run_name, run_points) in enumerate(by_run.items()):
        run_points = sorted(run_points, key=lambda row: row["grpo_step"])
        color = colors[idx % len(colors)]
        coords = " ".join(
            f'{x_of(point["grpo_step"]):.2f},{y_of(point["eval_reward"]):.2f}'
            for point in run_points
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            'stroke-width="2.5"/>'
        )
        for point in run_points:
            parts.append(
                f'<circle cx="{x_of(point["grpo_step"]):.2f}" '
                f'cy="{y_of(point["eval_reward"]):.2f}" r="4" fill="{color}"/>'
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


def write_html(path: Path, points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg(points)
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(point['run_name']))}</td>"
        f"<td>{html.escape(str(point['loss_type']))}</td>"
        f"<td>{html.escape(str(point['group_size']))}</td>"
        f"<td>{point['grpo_step']}</td>"
        f"<td>{point['train_step']}</td>"
        f"<td>{point['eval_reward']:.4f}</td>"
        f"<td>{point['eval_accuracy']:.4f}</td>"
        f"<td>{point['format_accuracy']:.4f}</td>"
        "</tr>"
        for point in points
    )
    path.write_text(
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'><title>GRPO Validation Curves</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;margin-top:24px}"
        "td,th{border:1px solid #d1d5db;padding:6px 10px}"
        "th{background:#f3f4f6}</style></head><body>"
        "<h1>GRPO Validation Reward Curves</h1>"
        f"{svg}"
        "<table><thead><tr><th>run</th><th>loss</th><th>group</th>"
        "<th>grpo step</th><th>train step</th><th>reward</th>"
        "<th>accuracy</th><th>format accuracy</th></tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
        "</body></html>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GRPO validation reward curves.")
    parser.add_argument("--output-dir", default="outputs/grpo_math")
    parser.add_argument(
        "--csv-out",
        default="outputs/grpo_math/validation_reward_curves.csv",
    )
    parser.add_argument(
        "--html-out",
        default="outputs/grpo_math/validation_reward_curves.html",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    points = collect_eval_points(Path(args.output_dir))
    if not points:
        raise SystemExit(f"No eval points found under {args.output_dir}")
    write_csv(Path(args.csv_out), points)
    write_html(Path(args.html_out), points)
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.html_out}")
