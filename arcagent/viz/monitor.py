from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from arcagent.viz.report import read_jsonl


MAX_RECENT_ROWS = 240
MAX_SERIES_POINTS = 420


def collect_run_state(run_dir: str | Path, max_recent_rows: int = MAX_RECENT_ROWS) -> dict[str, Any]:
    """Read a run directory and return bounded data for the live monitor."""

    run_dir = Path(run_dir).expanduser()
    metrics = _read_metrics(run_dir)
    pbt_events = _read_jsonl_if_present(run_dir / "pbt_events.jsonl")
    platform_events = _read_jsonl_if_present(run_dir / "platform_agent_events.jsonl")
    config = _read_json_if_present(run_dir / "config.json")

    train_steps = _nested_get(config, ["train", "steps"])
    latest = metrics[-1] if metrics else {}
    training_end = next((row for row in reversed(metrics) if row.get("event") == "training_end"), None)
    latest_mtime = _latest_mtime(run_dir)
    status = _status(metrics, training_end, latest_mtime)

    returns = _series(metrics, "return", event="episode_end")
    losses = _series(metrics, "world_model_loss")
    episode_returns = _series(metrics, "episode_return")
    buffer_sizes = _series(metrics, "buffer_size")
    epsilons = _series(metrics, "epsilon")

    latest_step = int(latest.get("step", 0) or 0)
    progress = None
    if isinstance(train_steps, int | float) and train_steps > 0:
        progress = min(1.0, latest_step / float(train_steps))

    summary = {
        "status": status,
        "latest_step": latest_step,
        "target_steps": train_steps,
        "progress": progress,
        "episodes": len(returns),
        "best_return": max((point["value"] for point in returns), default=None),
        "latest_return": returns[-1]["value"] if returns else None,
        "latest_episode_return": episode_returns[-1]["value"] if episode_returns else None,
        "latest_loss": losses[-1]["value"] if losses else None,
        "best_loss": min((point["value"] for point in losses), default=None),
        "latest_buffer_size": buffer_sizes[-1]["value"] if buffer_sizes else None,
        "latest_epsilon": epsilons[-1]["value"] if epsilons else None,
        "last_update_time": latest.get("time"),
        "latest_mtime": latest_mtime,
        "checkpoint_exists": (run_dir / "checkpoints" / "agent.pkl").exists(),
    }

    frames = _frame_files(run_dir)
    return {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "summary": summary,
        "metrics": {
            "row_count": len(metrics),
            "sources": sorted({str(row.get("source", "metrics.jsonl")) for row in metrics}),
            "latest": latest,
            "recent": _tail(metrics, max_recent_rows),
            "series": {
                "returns": returns[-MAX_SERIES_POINTS:],
                "losses": losses[-MAX_SERIES_POINTS:],
                "episode_returns": episode_returns[-MAX_SERIES_POINTS:],
                "buffer_sizes": buffer_sizes[-MAX_SERIES_POINTS:],
                "epsilons": epsilons[-MAX_SERIES_POINTS:],
            },
        },
        "events": {
            "pbt": _tail(pbt_events, max_recent_rows),
            "platform": _tail(platform_events, max_recent_rows),
        },
        "files": {
            "frames": frames[-80:],
            "pbt_summary": _read_json_if_present(run_dir / "pbt_summary.json"),
            "platform_summary": _read_json_if_present(run_dir / "platform_agent_summary.json"),
        },
        "config": config,
        "latest_frame": _load_latest_frame(run_dir),
    }


def discover_runs(root: str | Path = "runs") -> list[dict[str, Any]]:
    root = Path(root).expanduser()
    if not root.exists():
        return []
    candidates: dict[Path, dict[str, Any]] = {}
    markers = ["metrics.jsonl", "pbt_events.jsonl", "platform_agent_events.jsonl", "config.json"]
    for marker in markers:
        for path in root.rglob(marker):
            run_dir = path.parent
            stat = path.stat()
            current = candidates.get(run_dir)
            if current is None or stat.st_mtime > current["mtime"]:
                candidates[run_dir] = {"path": run_dir, "mtime": stat.st_mtime}
    return [
        {"run_dir": str(item["path"]), "mtime": item["mtime"]}
        for item in sorted(candidates.values(), key=lambda value: value["mtime"], reverse=True)
    ]


def serve(run_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, runs_root: str | Path = "runs") -> None:
    server = ThreadingHTTPServer((host, port), MonitorRequestHandler)
    server.run_dir = Path(run_dir).expanduser()  # type: ignore[attr-defined]
    server.runs_root = Path(runs_root).expanduser()  # type: ignore[attr-defined]
    url = f"http://{host}:{port}"
    print(f"ARC3 training monitor serving {server.run_dir} at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nmonitor stopped")
    finally:
        server.server_close()


class MonitorRequestHandler(BaseHTTPRequestHandler):
    server_version = "ARC3Monitor/0.1"

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/":
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if url.path == "/health":
            self._send_json({"ok": True, "service": "arc3-monitor"})
            return
        if url.path == "/api/state":
            params = parse_qs(url.query)
            run_dir = self._resolve_run_dir(params.get("run_dir", [""])[0])
            self._send_json(collect_run_state(run_dir))
            return
        if url.path == "/api/runs":
            self._send_json({"runs": discover_runs(self.server.runs_root)})  # type: ignore[attr-defined]
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _resolve_run_dir(self, value: str) -> Path:
        if not value:
            return self.server.run_dir  # type: ignore[attr-defined]
        path = Path(unquote(value)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _send_json(self, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("metrics.jsonl")):
        for row in read_jsonl(path):
            row["source"] = str(path.relative_to(run_dir))
            rows.append(row)
    return sorted(rows, key=lambda row: (float(row.get("time", 0.0) or 0.0), int(row.get("step", 0) or 0), str(row.get("source", ""))))


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    try:
        return read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return []


def _read_json_if_present(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _series(rows: list[dict[str, Any]], key: str, event: str | None = None) -> list[dict[str, float]]:
    points = []
    for row in rows:
        if event is not None and row.get("event") != event:
            continue
        value = row.get(key)
        if isinstance(value, int | float):
            points.append({"step": int(row.get("step", 0) or 0), "value": float(value)})
    return points


def _frame_files(run_dir: Path) -> list[dict[str, Any]]:
    frames_dir = run_dir / "frames"
    if not frames_dir.exists():
        return []
    files = []
    for path in sorted(frames_dir.glob("*.npz"), key=lambda item: item.stat().st_mtime):
        stat = path.stat()
        files.append({"path": str(path.relative_to(run_dir)), "mtime": stat.st_mtime, "size": stat.st_size})
    return files


def _load_latest_frame(run_dir: Path) -> dict[str, Any] | None:
    frames = _frame_files(run_dir)
    if not frames:
        return None
    path = run_dir / frames[-1]["path"]
    try:
        with np.load(path) as data:
            array = np.asarray(data["frames"])
        if array.ndim < 3:
            return None
        grid = np.asarray(array[-1], dtype=int)
        return {
            "source": frames[-1]["path"],
            "frame_count": int(array.shape[0]),
            "grid": grid.tolist(),
            "min": int(np.min(grid)),
            "max": int(np.max(grid)),
        }
    except (OSError, KeyError, ValueError):
        return None


def _latest_mtime(run_dir: Path) -> float | None:
    mtimes = []
    for relative in ["metrics.jsonl", "pbt_events.jsonl", "platform_agent_events.jsonl"]:
        path = run_dir / relative
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    if not mtimes and run_dir.exists():
        for path in run_dir.rglob("metrics.jsonl"):
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else None


def _status(metrics: list[dict[str, Any]], training_end: dict[str, Any] | None, latest_mtime: float | None) -> str:
    if training_end is not None:
        return "complete"
    if not metrics:
        return "waiting"
    if latest_mtime is not None and (time.time() - latest_mtime) < 20:
        return "active"
    return "idle"


def _nested_get(value: Any, path: list[str]) -> Any:
    cursor = value
    for part in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def _tail(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return rows[-max(0, count):]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a live ARC3 training monitor")
    parser.add_argument("run_dir", nargs="?", default="runs/default", help="Run directory to monitor")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--runs-root", default="runs", help="Directory used for recent-run discovery")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    serve(args.run_dir, host=args.host, port=args.port, runs_root=args.runs_root)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ARC3 Training Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-soft: #edf1f5;
      --text: #18202a;
      --muted: #667484;
      --line: #dbe2ea;
      --accent: #1f7a8c;
      --accent-2: #a24936;
      --good: #2d7d46;
      --warn: #a66a00;
      --shadow: 0 12px 32px rgba(26, 38, 52, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 249, 0.92);
      backdrop-filter: blur(12px);
    }
    .topbar {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(320px, 560px) auto;
      gap: 14px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .subtitle { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .run-picker {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
      min-height: 36px;
    }
    input { padding: 0 10px; width: 100%; }
    select { padding: 0 32px 0 10px; }
    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary {
      color: #ffffff;
      background: var(--accent);
      border-color: var(--accent);
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 20px 40px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.08fr 0.92fr;
      gap: 16px;
      align-items: start;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .panel, .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .metric {
      min-height: 86px;
      padding: 12px;
      display: grid;
      align-content: space-between;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .value {
      font-size: 23px;
      font-weight: 760;
      letter-spacing: 0;
      line-height: 1.1;
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .panel-title {
      font-weight: 760;
      font-size: 14px;
    }
    .panel-body { padding: 14px; }
    .stack { display: grid; gap: 16px; }
    .chart {
      width: 100%;
      height: 218px;
      display: block;
    }
    .progress-wrap {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .progress {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
    }
    .progress > div {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 180ms ease;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--muted);
    }
    .dot.active { background: var(--good); }
    .dot.complete { background: var(--accent); }
    .dot.waiting { background: var(--warn); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      padding: 8px 7px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 650;
      background: #fbfcfd;
      position: sticky;
      top: 0;
    }
    .scroll {
      max-height: 300px;
      overflow: auto;
    }
    .grid-view {
      display: grid;
      grid-template-columns: repeat(var(--cols), 1fr);
      gap: 2px;
      aspect-ratio: 1;
      max-height: 440px;
      margin: 0 auto;
      background: var(--line);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .cell {
      min-width: 0;
      min-height: 0;
      background: hsl(var(--hue), 52%, var(--lightness));
    }
    pre {
      margin: 0;
      max-height: 300px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      color: #26313e;
    }
    .empty {
      color: var(--muted);
      padding: 30px 12px;
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .recent-run {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }

    @media (max-width: 1100px) {
      .topbar { grid-template-columns: 1fr; }
      .toolbar { justify-content: flex-start; }
      .grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
    }
    @media (max-width: 620px) {
      main { padding: 12px; }
      .topbar { padding: 12px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .run-picker { grid-template-columns: 1fr; }
      .value { font-size: 19px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>ARC3 Training Monitor</h1>
        <div class="subtitle">Local JSONL monitor for Dreamer-lite, PBT, and platform-agent runs</div>
      </div>
      <div>
        <div class="run-picker">
          <input id="runDir" aria-label="Run directory" />
          <button id="loadRun" class="primary" type="button">Load</button>
        </div>
        <div id="recentRuns" class="recent-run"></div>
      </div>
      <div class="toolbar">
        <span class="status"><span id="statusDot" class="dot"></span><span id="statusText">connecting</span></span>
        <select id="refreshRate" aria-label="Refresh rate">
          <option value="1000">1s</option>
          <option value="2000" selected>2s</option>
          <option value="5000">5s</option>
          <option value="0">paused</option>
        </select>
        <button id="refreshNow" type="button">Refresh</button>
      </div>
    </div>
  </header>

  <main>
    <section class="metrics" aria-label="Training summary">
      <div class="metric"><div class="label">Step</div><div id="metricStep" class="value">-</div></div>
      <div class="metric"><div class="label">Episodes</div><div id="metricEpisodes" class="value">-</div></div>
      <div class="metric"><div class="label">Best return</div><div id="metricBestReturn" class="value">-</div></div>
      <div class="metric"><div class="label">Latest loss</div><div id="metricLoss" class="value">-</div></div>
      <div class="metric"><div class="label">Buffer</div><div id="metricBuffer" class="value">-</div></div>
      <div class="metric"><div class="label">Epsilon</div><div id="metricEpsilon" class="value">-</div></div>
    </section>

    <section class="panel" style="margin-bottom: 16px;">
      <div class="panel-header">
        <div class="panel-title">Progress</div>
        <div id="lastUpdate" class="label">No updates yet</div>
      </div>
      <div class="panel-body progress-wrap">
        <div class="progress" aria-label="Training progress"><div id="progressBar"></div></div>
        <div id="progressText" class="label">Waiting for metrics.</div>
      </div>
    </section>

    <section class="grid">
      <div class="stack">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Episode return</div>
            <div class="label">episode_end rows</div>
          </div>
          <div class="panel-body"><canvas id="returnChart" class="chart"></canvas></div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">World-model loss</div>
            <div class="label">lower is better</div>
          </div>
          <div class="panel-body"><canvas id="lossChart" class="chart"></canvas></div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Recent metrics</div>
            <div id="rowCount" class="label"></div>
          </div>
          <div class="scroll">
            <table>
              <thead><tr><th>step</th><th>event</th><th>return</th><th>loss</th><th>buffer</th><th>source</th></tr></thead>
              <tbody id="metricsRows"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Latest recorded grid</div>
            <div id="frameSource" class="label"></div>
          </div>
          <div class="panel-body"><div id="gridView" class="empty">No frame files found yet.</div></div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">PBT / platform events</div>
            <div class="label">latest first</div>
          </div>
          <div class="scroll">
            <table>
              <thead><tr><th>step</th><th>type</th><th>event</th><th>detail</th></tr></thead>
              <tbody id="eventRows"></tbody>
            </table>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Config</div>
            <div class="label">config.json</div>
          </div>
          <div class="panel-body"><pre id="configView">No config loaded.</pre></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const state = {
      timer: null,
      runDir: new URLSearchParams(window.location.search).get("run_dir") || "",
    };

    const $ = (id) => document.getElementById(id);

    function formatNumber(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      if (Math.abs(value) >= 1000) return String(Math.round(value));
      return Number(value).toFixed(digits).replace(/\.?0+$/, "");
    }

    function formatTime(seconds) {
      if (!seconds) return "No updates yet";
      return new Date(seconds * 1000).toLocaleString();
    }

    function setStatus(summary) {
      const status = summary?.status || "waiting";
      $("statusDot").className = "dot " + status;
      $("statusText").textContent = status;
    }

    function updateSummary(data) {
      const summary = data.summary || {};
      const step = summary.latest_step || 0;
      const target = summary.target_steps;
      $("metricStep").textContent = target ? `${step}/${target}` : String(step || "-");
      $("metricEpisodes").textContent = formatNumber(summary.episodes, 0);
      $("metricBestReturn").textContent = formatNumber(summary.best_return, 3);
      $("metricLoss").textContent = formatNumber(summary.latest_loss, 4);
      $("metricBuffer").textContent = formatNumber(summary.latest_buffer_size, 0);
      $("metricEpsilon").textContent = formatNumber(summary.latest_epsilon, 3);
      $("lastUpdate").textContent = `last update: ${formatTime(summary.last_update_time || summary.latest_mtime)}`;

      const progress = summary.progress;
      $("progressBar").style.width = progress === null || progress === undefined ? "0%" : `${Math.round(progress * 100)}%`;
      $("progressText").textContent = target
        ? `${step} of ${target} steps · checkpoint ${summary.checkpoint_exists ? "written" : "not written yet"}`
        : `${step || 0} steps observed · checkpoint ${summary.checkpoint_exists ? "written" : "not written yet"}`;
      setStatus(summary);
    }

    function drawChart(canvas, points, options) {
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const pad = { left: 46, right: 14, top: 16, bottom: 28 };
      ctx.strokeStyle = "#dbe2ea";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, rect.height - pad.bottom);
      ctx.lineTo(rect.width - pad.right, rect.height - pad.bottom);
      ctx.stroke();

      if (!points || points.length === 0) {
        ctx.fillStyle = "#667484";
        ctx.font = "13px system-ui";
        ctx.fillText("No data yet", pad.left + 12, pad.top + 24);
        return;
      }

      const xs = points.map((p) => p.step);
      const ys = points.map((p) => p.value);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      let minY = Math.min(...ys);
      let maxY = Math.max(...ys);
      if (minY === maxY) {
        const delta = Math.max(0.01, Math.abs(minY) * 0.1);
        minY -= delta;
        maxY += delta;
      }
      const plotW = rect.width - pad.left - pad.right;
      const plotH = rect.height - pad.top - pad.bottom;
      const xFor = (x) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * plotW;
      const yFor = (y) => pad.top + (1 - (y - minY) / (maxY - minY)) * plotH;

      ctx.strokeStyle = options.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = xFor(point.step);
        const y = yFor(point.value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      const last = points[points.length - 1];
      ctx.fillStyle = options.color;
      ctx.beginPath();
      ctx.arc(xFor(last.step), yFor(last.value), 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#667484";
      ctx.font = "12px system-ui";
      ctx.fillText(formatNumber(maxY, options.digits), 6, pad.top + 4);
      ctx.fillText(formatNumber(minY, options.digits), 6, rect.height - pad.bottom + 4);
      ctx.fillText(`step ${minX}`, pad.left, rect.height - 7);
      ctx.textAlign = "right";
      ctx.fillText(`step ${maxX}`, rect.width - pad.right, rect.height - 7);
      ctx.textAlign = "left";
    }

    function renderRows(data) {
      const recent = (data.metrics?.recent || []).slice(-42).reverse();
      $("rowCount").textContent = `${data.metrics?.row_count || 0} rows`;
      $("metricsRows").innerHTML = recent.map((row) => `
        <tr>
          <td>${row.step ?? ""}</td>
          <td>${row.event ?? ""}</td>
          <td>${formatNumber(row.return ?? row.episode_return, 3)}</td>
          <td>${formatNumber(row.world_model_loss, 4)}</td>
          <td>${row.buffer_size ?? ""}</td>
          <td>${row.source ?? ""}</td>
        </tr>
      `).join("");
    }

    function eventDetail(event) {
      if (event.decision) return event.decision;
      if (event.score !== undefined) return `score ${formatNumber(event.score, 3)}`;
      if (event.metrics) return JSON.stringify(event.metrics);
      if (event.changes) return JSON.stringify(event.changes);
      return JSON.stringify(event).slice(0, 180);
    }

    function renderEvents(data) {
      const events = [
        ...(data.events?.pbt || []).map((event) => ({...event, kind: "PBT"})),
        ...(data.events?.platform || []).map((event) => ({...event, kind: "platform"})),
      ].sort((a, b) => (b.time || b.step || 0) - (a.time || a.step || 0)).slice(0, 36);
      $("eventRows").innerHTML = events.map((event) => `
        <tr>
          <td>${event.step ?? ""}</td>
          <td>${event.kind}</td>
          <td>${event.event ?? event.action ?? ""}</td>
          <td>${eventDetail(event)}</td>
        </tr>
      `).join("");
    }

    function renderGrid(frame) {
      const container = $("gridView");
      if (!frame || !frame.grid || frame.grid.length === 0) {
        container.className = "empty";
        container.textContent = "No frame files found yet.";
        $("frameSource").textContent = "";
        return;
      }
      const grid = frame.grid;
      const rows = grid.length;
      const cols = grid[0].length;
      const min = frame.min ?? 0;
      const max = frame.max ?? 1;
      container.className = "grid-view";
      container.style.setProperty("--cols", cols);
      const span = Math.max(1, max - min);
      container.innerHTML = grid.flatMap((row) => row.map((value) => {
        const ratio = (value - min) / span;
        const hue = Math.round(195 + ratio * 130);
        const lightness = Math.round(92 - ratio * 54);
        return `<div class="cell" title="${value}" style="--hue:${hue};--lightness:${lightness}%"></div>`;
      })).join("");
      $("frameSource").textContent = `${frame.source} · ${frame.frame_count} frames · ${rows}x${cols}`;
    }

    async function refresh() {
      const runDir = $("runDir").value.trim();
      const params = new URLSearchParams();
      if (runDir) params.set("run_dir", runDir);
      const response = await fetch(`/api/state?${params.toString()}`, { cache: "no-store" });
      const data = await response.json();
      if (!state.runDir) {
        $("runDir").value = data.run_dir || runDir;
        state.runDir = $("runDir").value;
      }
      updateSummary(data);
      const series = data.metrics?.series || {};
      drawChart($("returnChart"), series.returns || [], { color: "#1f7a8c", digits: 3 });
      drawChart($("lossChart"), series.losses || [], { color: "#a24936", digits: 4 });
      renderRows(data);
      renderEvents(data);
      renderGrid(data.latest_frame);
      $("configView").textContent = data.config ? JSON.stringify(data.config, null, 2) : "No config loaded.";
    }

    async function loadRuns() {
      const response = await fetch("/api/runs", { cache: "no-store" });
      const data = await response.json();
      const runs = data.runs || [];
      if (runs.length === 0) {
        $("recentRuns").textContent = "No runs discovered under runs/.";
        return;
      }
      $("recentRuns").innerHTML = `Recent: ${runs.slice(0, 4).map((run) =>
        `<button type="button" data-run="${run.run_dir}">${run.run_dir}</button>`
      ).join(" ")}`;
      $("recentRuns").querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
          $("runDir").value = button.dataset.run;
          state.runDir = button.dataset.run;
          refresh();
        });
      });
    }

    function schedule() {
      if (state.timer) clearInterval(state.timer);
      const rate = Number($("refreshRate").value);
      if (rate > 0) state.timer = setInterval(refresh, rate);
    }

    $("runDir").value = state.runDir;
    $("loadRun").addEventListener("click", () => {
      state.runDir = $("runDir").value.trim();
      refresh();
    });
    $("refreshNow").addEventListener("click", refresh);
    $("refreshRate").addEventListener("change", schedule);
    window.addEventListener("resize", refresh);

    loadRuns();
    refresh();
    schedule();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
