from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from statistics import mean


HTML_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 2rem; line-height: 1.45; }
.card { border: 1px solid #ddd; border-radius: 14px; padding: 1rem; margin: 1rem 0; box-shadow: 0 1px 8px rgba(0,0,0,.05); }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #eee; text-align: left; padding: .45rem; }
.badge { display: inline-block; padding: .2rem .5rem; border-radius: 999px; background: #eee; margin-right: .3rem; }
pre { overflow-x: auto; background: #f7f7f7; padding: .75rem; border-radius: 10px; }
"""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return chars[0] * min(width, len(values))
    sample = values[-width:]
    return "".join(chars[int((v - lo) / (hi - lo) * (len(chars) - 1))] for v in sample)


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    metrics = []
    for path in run_dir.rglob("metrics.jsonl"):
        for row in read_jsonl(path):
            row["source"] = str(path.relative_to(run_dir))
            metrics.append(row)
    pbt_events = read_jsonl(run_dir / "pbt_events.jsonl")
    returns = [float(m["return"]) for m in metrics if m.get("event") == "episode_end" and "return" in m]
    losses = [float(m["world_model_loss"]) for m in metrics if "world_model_loss" in m]

    html = ["<!doctype html><html><head><meta charset='utf-8'><title>ARC3 Training Report</title><style>", HTML_STYLE, "</style></head><body>"]
    html.append("<h1>ARC3 Training Report</h1>")
    html.append("<p>This report is generated from local logs only. It explains whether the agent is learning, how its world model is improving, and how the self-tuning controller changed training.</p>")
    html.append("<div class='card'><h2>At a glance</h2>")
    html.append(f"<span class='badge'>episodes: {len(returns)}</span><span class='badge'>metric rows: {len(metrics)}</span><span class='badge'>PBT events: {len(pbt_events)}</span>")
    if returns:
        html.append(f"<p><b>Episode return:</b> latest {returns[-1]:.3f}, best {max(returns):.3f}, mean {mean(returns):.3f}</p><pre>{sparkline(returns)}</pre>")
    if losses:
        html.append(f"<p><b>World-model loss:</b> latest {losses[-1]:.4f}, best {min(losses):.4f}</p><pre>{sparkline([-x for x in losses])}</pre>")
    html.append("</div>")

    html.append("<div class='card'><h2>Beginner explanation</h2>")
    html.append("<p>The player does not only react to the current grid. It first compresses the grid into a small latent state, predicts what each possible action may do next, and scores short imagined futures. The self-tuning platform then compares several training runs and keeps the settings that worked best.</p>")
    html.append("</div>")

    if pbt_events:
        html.append("<div class='card'><h2>Self-tuning decisions</h2><table><tr><th>generation</th><th>event</th><th>target/source</th><th>score/decision</th></tr>")
        for event in pbt_events[-50:]:
            html.append("<tr>" + "".join([
                f"<td>{escape(str(event.get('step','')))}</td>",
                f"<td>{escape(str(event.get('event','')))}</td>",
                f"<td>{escape(str(event.get('target_trial', event.get('trial_id',''))))} ← {escape(str(event.get('source_trial','')))}</td>",
                f"<td>{escape(str(event.get('score', event.get('decision',''))))}</td>",
            ]) + "</tr>")
        html.append("</table></div>")

    frame_files = sorted((run_dir / "frames").glob("*.npz")) if (run_dir / "frames").exists() else []
    html.append("<div class='card'><h2>Recorded frames</h2>")
    if frame_files:
        html.append("<p>Frame arrays are saved as compressed NumPy files for notebooks to animate. Latest files:</p><ul>")
        for f in frame_files[-10:]:
            html.append(f"<li>{escape(str(f.relative_to(run_dir)))}</li>")
        html.append("</ul>")
    else:
        html.append("<p>No frame files found yet. Run training for at least a few dozen steps.</p>")
    html.append("</div>")

    html.append("</body></html>")
    out = run_dir / "report.html"
    out.write_text("".join(html), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build an ARC3 training report")
    p.add_argument("run_dir")
    args = p.parse_args()
    path = build_report(args.run_dir)
    print(path)


if __name__ == "__main__":
    main()
