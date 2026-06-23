from __future__ import annotations

from pathlib import Path
import re

FORBIDDEN_PATTERNS = [
    r"openai",
    r"anthropic",
    r"requests\.",
    r"urllib\.request",
    r"httpx",
    r"socket\.",
    r"subprocess",
]

SCAN_DIRS = ["submission", "arcagent/agent", "arcagent/models", "arcagent/envs"]


def scan(root: str | Path = ".") -> list[str]:
    root = Path(root)
    findings: list[str] = []
    for rel in SCAN_DIRS:
        path = root / rel
        if not path.exists():
            continue
        for file in path.rglob("*.py"):
            if file.name == "offline_check.py":
                continue
            text = file.read_text(encoding="utf-8")
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    findings.append(f"{file}: matched {pattern}")
    return findings


def main() -> None:
    findings = scan(Path(__file__).resolve().parents[1])
    if findings:
        print("Offline compliance check failed:")
        for finding in findings:
            print(" -", finding)
        raise SystemExit(1)
    print("Offline compliance check passed: no network/LLM/client patterns found in inference-critical code.")


if __name__ == "__main__":
    main()
