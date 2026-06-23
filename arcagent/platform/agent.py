from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import time
from typing import Any, Protocol

import numpy as np

from arcagent.config import RunConfig
from arcagent.data import MetricLogger
from arcagent.train import Trainer
from arcagent.viz.report import build_report
from .llm_client import LiteLLMClient, LLMResult, extract_json_object


DEFAULT_DEEPSEEK_MODELS = [
    "opencode-go/deepseek-v4-flash",
    "opencode-go/deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], models: list[str], temperature: float = 0.2, max_tokens: int = 700) -> LLMResult: ...


@dataclass
class EvidenceBundle:
    docs_summary: str
    recent_metrics: list[dict[str, Any]]
    previous_decisions: list[dict[str, Any]]


@dataclass
class ExperimentProposal:
    rationale: str
    changes: dict[str, Any]
    expected_effect: str
    risk: str


@dataclass
class ExperimentOutcome:
    cycle: int
    proposal: ExperimentProposal
    run_dir: str
    score: float
    report_path: str
    metrics: dict[str, Any]
    llm_model: str
    llm_ok: bool
    llm_error: str | None = None


class ResearchCollector:
    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root)

    def collect(self, run_dir: str | Path) -> EvidenceBundle:
        docs = []
        for rel in ["README.md", "docs/algorithm_notes.md"]:
            path = self.repo_root / rel
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")
                docs.append(f"# {rel}\n{text[:3500]}")
        metrics = self._read_jsonl(Path(run_dir) / "metrics.jsonl")[-30:]
        decisions = self._read_jsonl(Path(run_dir) / "platform_agent_events.jsonl")[-20:]
        return EvidenceBundle("\n\n".join(docs), metrics, decisions)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows


class PlatformAgent:
    """Autonomous training-platform agent.

    The loop is intentionally concrete: gather local project evidence, ask a
    local DeepSeek model through LiteLLM for the next bounded experiment, execute
    that experiment, score it, log the decision, and generate a readable report.
    """

    def __init__(
        self,
        base_config: RunConfig,
        repo_root: str | Path = ".",
        llm_client: ChatClient | None = None,
        models: list[str] | None = None,
    ):
        self.base_config = base_config
        self.repo_root = Path(repo_root)
        self.run_dir = Path(base_config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.collector = ResearchCollector(self.repo_root)
        self.llm_client = llm_client or LiteLLMClient()
        self.models = models or DEFAULT_DEEPSEEK_MODELS
        self.logger = MetricLogger(self.run_dir, filename="platform_agent_events.jsonl")

    def run(self, cycles: int = 1, steps_per_cycle: int = 120) -> list[ExperimentOutcome]:
        outcomes = []
        current = copy.deepcopy(self.base_config)
        for cycle in range(cycles):
            evidence = self.collector.collect(self.run_dir)
            proposal, llm_result = self.propose_experiment(evidence, cycle)
            cfg = self.apply_proposal(current, proposal)
            cfg.train.steps = steps_per_cycle
            cfg.run_dir = str(self.run_dir / f"cycle_{cycle:02d}")
            trainer = Trainer(cfg)
            train_result = trainer.train()
            eval_metrics = trainer.evaluate(episodes=2)
            report = build_report(train_result.run_dir)
            score = float(eval_metrics["eval_return_mean"] + 2.0 * eval_metrics["eval_success_rate"])
            outcome = ExperimentOutcome(
                cycle=cycle,
                proposal=proposal,
                run_dir=str(train_result.run_dir),
                score=score,
                report_path=str(report),
                metrics=eval_metrics,
                llm_model=llm_result.model,
                llm_ok=llm_result.ok,
                llm_error=llm_result.error,
            )
            outcomes.append(outcome)
            self.logger.log(cycle, {"event": "platform_cycle", "outcome": asdict(outcome), "config": cfg.to_dict()})
            current = cfg
        self._write_summary(outcomes)
        return outcomes

    def smoke_deepseek(self) -> LLMResult:
        return self.llm_client.chat(
            messages=[{"role": "user", "content": "Reply exactly with JSON: {\"ok\":true,\"message\":\"ARC3_DEEPSEEK_SMOKE\"}"}],
            models=self.models,
            temperature=0.0,
            max_tokens=120,
        )

    def propose_experiment(self, evidence: EvidenceBundle, cycle: int) -> tuple[ExperimentProposal, LLMResult]:
        compact_docs = evidence.docs_summary[:2200]
        compact_metrics = json.dumps(evidence.recent_metrics[-8:], ensure_ascii=False)[:1600]
        compact_decisions = json.dumps(evidence.previous_decisions[-4:], ensure_ascii=False)[:1200]
        prompt = f"""
You are the autonomous training-platform agent for an ARC-AGI-3 world-model player.
Use the evidence to choose ONE small safe experiment for the next cycle.
Return only JSON with keys: rationale, changes, expected_effect, risk.
Allowed changes: train.learning_rate, train.epsilon, train.action_penalty, model.imagination_horizon, model.planning_rollouts.
Keep values conservative for a tiny local smoke run.
Example JSON:
{{"rationale":"increase exploration","changes":{{"train.epsilon":0.22}},"expected_effect":"collect more diverse transitions","risk":"short smoke run may be noisy"}}

Cycle: {cycle}
Docs excerpt:
{compact_docs}
Recent metrics JSON:
{compact_metrics}
Previous platform decisions JSON:
{compact_decisions}
""".strip()
        result = self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            models=self.models,
            temperature=0.2,
            max_tokens=1200,
        )
        if result.ok:
            try:
                data = extract_json_object(result.content)
                return self._sanitize_proposal(data), result
            except Exception as exc:
                result.error = f"parse_failed: {exc}; content={result.content[:500]}"
        return self._fallback_proposal(evidence, cycle), result

    def apply_proposal(self, cfg: RunConfig, proposal: ExperimentProposal) -> RunConfig:
        cfg = copy.deepcopy(cfg)
        for key, value in proposal.changes.items():
            if key == "train.learning_rate":
                cfg.train.learning_rate = float(np.clip(float(value), 1e-4, 2e-2))
            elif key == "train.epsilon":
                cfg.train.epsilon = float(np.clip(float(value), 0.02, 0.6))
            elif key == "train.action_penalty":
                cfg.train.action_penalty = float(np.clip(float(value), 0.001, 0.05))
            elif key == "model.imagination_horizon":
                cfg.model.imagination_horizon = int(np.clip(int(value), 3, 15))
            elif key == "model.planning_rollouts":
                cfg.model.planning_rollouts = int(np.clip(int(value), 2, 24))
        cfg.train.seed += 17
        cfg.env.seed += 17
        return cfg

    def _sanitize_proposal(self, data: dict[str, Any]) -> ExperimentProposal:
        changes = data.get("changes", {})
        if not isinstance(changes, dict):
            changes = {}
        allowed = {"train.learning_rate", "train.epsilon", "train.action_penalty", "model.imagination_horizon", "model.planning_rollouts"}
        clean = {k: v for k, v in changes.items() if k in allowed}
        if not clean:
            clean = {"train.epsilon": 0.2, "model.planning_rollouts": 8}
        return ExperimentProposal(
            rationale=str(data.get("rationale", "DeepSeek proposed a conservative smoke experiment.")),
            changes=clean,
            expected_effect=str(data.get("expected_effect", "Improve exploration or planning stability.")),
            risk=str(data.get("risk", "Tiny run may be too short to prove score improvement.")),
        )

    def _fallback_proposal(self, evidence: EvidenceBundle, cycle: int) -> ExperimentProposal:
        losses = [m.get("world_model_loss") for m in evidence.recent_metrics if isinstance(m.get("world_model_loss"), (int, float))]
        if losses and losses[-1] > min(losses):
            changes = {"train.learning_rate": 0.002, "model.planning_rollouts": 12}
            rationale = "Fallback: recent world-model loss is not improving; reduce learning rate and spend more planning rollouts."
        else:
            changes = {"train.epsilon": 0.25 if cycle == 0 else 0.18, "model.imagination_horizon": 8}
            rationale = "Fallback: gather more exploratory transitions while keeping imagination horizon moderate."
        return ExperimentProposal(rationale, changes, "Collect more useful transitions and stabilize imagined scoring.", "No LLM proposal was available or parseable.")

    def _write_summary(self, outcomes: list[ExperimentOutcome]) -> None:
        best = max(outcomes, key=lambda o: o.score) if outcomes else None
        summary = {
            "created_at": time.time(),
            "best_score": None if best is None else best.score,
            "best_run_dir": None if best is None else best.run_dir,
            "outcomes": [asdict(o) for o in outcomes],
        }
        (self.run_dir / "platform_agent_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
