import abc
import warnings
from time import perf_counter
from dataclasses import dataclass


@dataclass
class Log(abc.ABC):
    elapsed_time: float
    warnings: list


class Logger:
    log: Log

    def __init__(self, log: Log):
        self.log = log

    def __enter__(self):
        self.warnings_catcher = warnings.catch_warnings(record=True)
        self.time_start = perf_counter()
        self.log.warnings = self.warnings_catcher.__enter__()
        return self.log

    def __exit__(self, exc_type, exc_val, traceback):
        self.log.elapsed_time = perf_counter() - self.time_start
        self.warnings_catcher.__exit__(exc_type, exc_val, traceback)


if __name__ == '__main__':
    pass
