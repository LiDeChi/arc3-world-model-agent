__all__ = ["build_report", "collect_run_state"]


def __getattr__(name):
    if name == "build_report":
        from .report import build_report

        return build_report
    if name == "collect_run_state":
        from .monitor import collect_run_state

        return collect_run_state
    raise AttributeError(name)
