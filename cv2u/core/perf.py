"""Performance helpers: a timer, a timing decorator, micro-benchmarks and image memory estimates."""

import statistics
import sys
import types
from collections.abc import Callable, Sequence
from functools import wraps
from time import perf_counter
from typing import Any, NamedTuple, ParamSpec, TextIO, TypeVar, overload

import numpy as np

__all__ = ["BenchmarkResult", "Timer", "benchmarkFunction", "formatBytes", "imageMemory", "timed"]

P = ParamSpec("P")
R = TypeVar("R")


class Timer:
    """A reusable wall-clock timer, as a context manager or started by hand.

    ``cv.getTickCount() / cv.getTickFrequency()`` works but is verbose and has no laps; ``Timer`` uses
    ``time.perf_counter`` and records named laps.

    Args:
        name: Label printed with ``verbose``.
        verbose: Print the elapsed time when the context exits.
        file: Stream for ``verbose`` output; ``None`` means ``sys.stdout`` at print time.

    Examples:
        >>> with Timer() as t:
        ...     pass
        >>> t.elapsed >= 0 and t.ms >= 0
        True
    """

    def __init__(self, name: str | None = None, *, verbose: bool = False, file: TextIO | None = None):
        self.name = name
        self.verbose = verbose
        self.file = file
        self._start: float | None = None
        self._end: float | None = None
        self.laps: list[tuple[str, float]] = []
        """``(name, seconds since the previous lap or start)`` pairs recorded by :meth:`lap`."""

    def start(self) -> "Timer":
        """Start (or restart) the timer, clearing the laps."""
        self._start = perf_counter()
        self._end = None
        self.laps = []
        return self

    def stop(self) -> float:
        """Stop the timer and return the elapsed seconds.

        Raises:
            RuntimeError: If the timer was never started.
        """
        if self._start is None:
            raise RuntimeError("Timer.stop() called before start()")
        self._end = perf_counter()
        return self.elapsed

    def lap(self, name: str | None = None) -> float:
        """Record a lap and return the seconds since the previous lap (or the start).

        Args:
            name: Lap label; ``None`` gives ``"lap1"``, ``"lap2"``, ...

        Raises:
            RuntimeError: If the timer is not running.
        """
        if self._start is None:
            raise RuntimeError("Timer.lap() called before start()")
        now = perf_counter()
        previous = self._start + sum(seconds for _, seconds in self.laps)
        seconds = now - previous
        self.laps.append((name if name is not None else f"lap{len(self.laps) + 1}", seconds))
        return seconds

    @property
    def elapsed(self) -> float:
        """Seconds between the start and the stop (or now, while running); ``0.0`` before the start."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else perf_counter()
        return end - self._start

    @property
    def ms(self) -> float:
        """:attr:`elapsed` in milliseconds."""
        return self.elapsed * 1000.0

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
        if self.verbose:
            print(str(self), file=self.file if self.file is not None else sys.stdout)

    def __str__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        text = f"{label}{self.ms:.3f} ms"
        if self.laps:
            text += " (" + ", ".join(f"{n} {s * 1000:.3f} ms" for n, s in self.laps) + ")"
        return text

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.6f}, laps={len(self.laps)})"


class _Timed:
    """The wrapper returned by :func:`timed`, with the time of the last call."""

    def __init__(self, fn: Callable[..., Any], repeats: int, name: str | None, file: TextIO | None, verbose: bool):
        self._fn = fn
        self._repeats = repeats
        self._name = name if name is not None else getattr(fn, "__name__", "function")
        self._file = file
        self._verbose = verbose
        self.lastElapsed: float = 0.0
        """Seconds taken by the last call (the mean of one call when ``repeats > 1``)."""
        self.history: list[float] = []
        """Seconds of every call, in order."""
        wraps(fn)(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        for _ in range(self._repeats - 1):
            self._fn(*args, **kwargs)
        result = self._fn(*args, **kwargs)
        self.lastElapsed = (perf_counter() - start) / self._repeats
        self.history.append(self.lastElapsed)
        if self._verbose:
            print(f"{self._name}: {self.lastElapsed * 1000:.3f} ms", file=self._file if self._file else sys.stdout)
        return result

    def __get__(self, instance: object, owner: type | None = None) -> Any:
        if instance is None:
            return self
        return types.MethodType(self, instance)


@overload
def timed(fn: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def timed(
    *, repeats: int = 1, name: str | None = None, file: TextIO | None = None, verbose: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def timed(
    fn: Callable[P, R] | None = None,
    /,
    *,
    repeats: int = 1,
    name: str | None = None,
    file: TextIO | None = None,
    verbose: bool = True,
) -> Any:
    """Decorator that times every call and prints or records the duration.

    Timing one function means a ``perf_counter`` pair around every call site; this wraps it once. The wrapper
    exposes ``lastElapsed`` (seconds of the last call) and ``history`` (seconds of every call).

    Args:
        fn: The function, when used as ``@timed`` without parentheses.
        repeats: Call the function this many times per call and report the mean; the last result is returned.
        name: Label for the printed line; the function name by default.
        file: Stream for the printed line; ``None`` means ``sys.stdout`` at print time.
        verbose: Print the time after every call; ``False`` only records it.

    Returns:
        The wrapped function.

    Raises:
        ValueError: If ``repeats`` is less than 1.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        return _Timed(func, repeats, name, file, verbose)

    return decorate if fn is None else decorate(fn)


class BenchmarkResult(NamedTuple):
    """Timings of :func:`benchmarkFunction`, in seconds per call."""

    mean: float
    median: float
    min: float
    max: float
    std: float
    repeats: int

    def __str__(self) -> str:
        return (
            f"mean {self.mean * 1000:.3f} ms, median {self.median * 1000:.3f} ms, "
            f"min {self.min * 1000:.3f} ms, max {self.max * 1000:.3f} ms, std {self.std * 1000:.3f} ms "
            f"({self.repeats} repeats)"
        )


def benchmarkFunction(
    fn: Callable[..., Any], *args: Any, repeats: int = 10, warmup: int = 1, **kwargs: Any
) -> BenchmarkResult:
    """Time ``fn(*args, **kwargs)`` several times and summarize the timings.

    Comparing two implementations (``cv.resize`` interpolations, a cv2u helper against the plain call) needs
    warm-up, repeats and robust statistics; ``timeit`` gives only a total.

    Args:
        fn: The function to time.
        *args: Positional arguments passed to ``fn``.
        repeats: Timed calls.
        warmup: Untimed calls before the measurement (first-call costs: caches, JIT, lazy imports).
        **kwargs: Keyword arguments passed to ``fn``.

    Returns:
        :class:`BenchmarkResult` with ``mean``, ``median``, ``min``, ``max`` and ``std`` in seconds per call and
        ``repeats``.

    Raises:
        ValueError: If ``repeats`` is less than 1 or ``warmup`` is negative.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    for _ in range(warmup):
        fn(*args, **kwargs)
    times: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        fn(*args, **kwargs)
        times.append(perf_counter() - start)
    std = statistics.pstdev(times) if repeats > 1 else 0.0
    return BenchmarkResult(statistics.fmean(times), statistics.median(times), min(times), max(times), std, repeats)


def imageMemory(src: np.ndarray | Sequence[int], dtype: Any = np.uint8) -> int:
    """Bytes an image (or an image of a given shape) occupies in memory.

    Deciding whether a batch, a pyramid or a video buffer fits in RAM needs ``prod(shape) * itemsize``, which
    OpenCV does not expose for a planned allocation.

    Args:
        src: An array (its own dtype is used) or a shape such as ``(height, width, channels)``.
        dtype: Element type when ``src`` is a shape.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If a shape dimension is negative.
    """
    if isinstance(src, np.ndarray):
        return int(src.nbytes)
    shape = [int(s) for s in src]
    if any(s < 0 for s in shape):
        raise ValueError(f"src shape must be non-negative, got {tuple(shape)}")
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)


_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def formatBytes(n: float, *, binary: bool = False, precision: int = 1) -> str:
    """Format a byte count for humans, e.g. ``"1.2 MB"``.

    Args:
        n: Number of bytes.
        binary: Use 1024-based units (``KiB``, ``MiB``, ...) instead of 1000-based ones.
        precision: Decimal places for values of 1 KB and above (bytes are printed as integers).

    Returns:
        The formatted string.

    Raises:
        ValueError: If ``n`` is negative or ``precision`` is negative.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if precision < 0:
        raise ValueError(f"precision must be >= 0, got {precision}")
    base = 1024.0 if binary else 1000.0
    value = float(n)
    index = 0
    while value >= base and index < len(_UNITS) - 1:
        value /= base
        index += 1
    if index == 0:
        return f"{int(value)} B"
    unit = _UNITS[index]
    if binary:
        unit = unit[0] + "iB"
    return f"{value:.{precision}f} {unit}"
