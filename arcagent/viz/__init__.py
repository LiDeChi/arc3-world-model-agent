__all__ = ["build_report"]


def __getattr__(name):
    if name == "build_report":
        from .report import build_report

        return build_report
    raise AttributeError(name)
