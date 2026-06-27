__all__ = ["run_benchmark"]


def __getattr__(name):
    if name == "run_benchmark":
        from .benchmark import run_benchmark

        return run_benchmark
    raise AttributeError(name)
