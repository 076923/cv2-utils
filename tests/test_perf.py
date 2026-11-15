import io
import time

import numpy as np
import pytest

from cv2u.core.perf import BenchmarkResult, Timer, benchmarkFunction, formatBytes, imageMemory, timed

# Timer


def test_timer_context_measures_sleep():
    with Timer() as t:
        time.sleep(0.01)
    assert 0.008 <= t.elapsed < 0.5
    assert t.ms == pytest.approx(t.elapsed * 1000)
    assert t.elapsed == t.elapsed  # frozen after exit
    assert repr(t).startswith("Timer(name=None, elapsed=")


def test_timer_laps_and_manual_use():
    t = Timer("work")
    assert t.elapsed == 0.0
    with pytest.raises(RuntimeError, match="start"):
        t.stop()
    with pytest.raises(RuntimeError, match="start"):
        t.lap()
    t.start()
    first = t.lap("read")
    time.sleep(0.005)
    second = t.lap()
    total = t.stop()
    assert [name for name, _ in t.laps] == ["read", "lap2"]
    assert second >= 0.004 and first >= 0.0
    assert total >= first + second - 1e-6
    assert str(t).startswith("work: ") and "read " in str(t) and "lap2 " in str(t)
    t.start()
    assert t.laps == [] and t.elapsed >= 0.0


def test_timer_verbose_prints_to_file():
    buffer = io.StringIO()
    with Timer("step", verbose=True, file=buffer):
        pass
    assert buffer.getvalue().startswith("step: ") and buffer.getvalue().endswith(" ms\n")


def test_timer_verbose_default_stdout(capsys):
    with Timer(verbose=True):
        pass
    assert capsys.readouterr().out.endswith(" ms\n")


# timed


def test_timed_decorator_records_and_prints():
    buffer = io.StringIO()

    @timed(name="add", file=buffer)
    def add(a, b):
        time.sleep(0.002)
        return a + b

    assert add(1, 2) == 3
    assert add.lastElapsed >= 0.001
    assert add.history == [add.lastElapsed]
    assert buffer.getvalue().startswith("add: ") and add.__name__ == "add"


def test_timed_bare_and_repeats(capsys):
    calls = []

    @timed
    def f(x):
        calls.append(x)
        return x

    assert f(5) == 5 and calls == [5]
    assert capsys.readouterr().out.startswith("f: ")

    @timed(repeats=3, verbose=False)
    def g(x):
        calls.append(x)
        return x * 2

    assert g(1) == 2 and calls == [5, 1, 1, 1]
    assert capsys.readouterr().out == ""
    assert g.lastElapsed >= 0.0
    with pytest.raises(ValueError, match="repeats"):
        timed(repeats=0)


def test_timed_on_method():
    class Box:
        def __init__(self):
            self.n = 3

        @timed(verbose=False)
        def value(self, k):
            return self.n * k

    box = Box()
    assert box.value(2) == 6
    assert box.value.lastElapsed >= 0.0 and box.value.history == [box.value.lastElapsed]  # bound access too
    assert box.value.__name__ == "value" and Box.value.lastElapsed >= 0.0


# benchmarkFunction


def test_benchmark_function_statistics():
    calls = []

    def work(a, scale=1):
        calls.append(a * scale)
        return a

    result = benchmarkFunction(work, 2, repeats=5, warmup=2, scale=3)
    assert isinstance(result, BenchmarkResult) and result.repeats == 5
    assert len(calls) == 7 and calls[0] == 6
    assert result.min <= result.median <= result.max
    assert result.min <= result.mean <= result.max and result.std >= 0.0
    assert "mean" in str(result) and "5 repeats" in str(result)
    single = benchmarkFunction(work, 1, repeats=1, warmup=0)
    assert single.std == 0.0 and single.mean == single.min == single.max
    with pytest.raises(ValueError, match="repeats"):
        benchmarkFunction(work, 1, repeats=0)
    with pytest.raises(ValueError, match="warmup"):
        benchmarkFunction(work, 1, warmup=-1)


def test_benchmark_function_sleep_timing():
    result = benchmarkFunction(time.sleep, 0.005, repeats=3, warmup=0)
    assert result.min >= 0.004


# imageMemory / formatBytes


def test_image_memory():
    assert imageMemory(np.zeros((4, 6, 3), np.uint8)) == 72
    assert imageMemory(np.zeros((4, 6), np.float32)) == 96
    assert imageMemory((480, 640, 3)) == 921_600
    assert imageMemory((480, 640), np.uint16) == 614_400
    assert imageMemory((4, 6, 3), dtype="float64") == 576
    assert imageMemory(()) == 1
    with pytest.raises(ValueError, match="src"):
        imageMemory((4, -1))


@pytest.mark.parametrize(
    ("n", "kwargs", "expected"),
    [
        (0, {}, "0 B"),
        (999, {}, "999 B"),
        (1000, {}, "1.0 KB"),
        (1_234_567, {}, "1.2 MB"),
        (1_234_567, {"precision": 2}, "1.23 MB"),
        (5 * 10**9, {}, "5.0 GB"),
        (10**12, {}, "1.0 TB"),
        (10**18, {}, "1000.0 PB"),
        (1024, {"binary": True}, "1.0 KiB"),
        (1023, {"binary": True}, "1023 B"),
        (3 * 1024**2, {"binary": True}, "3.0 MiB"),
    ],
)
def test_format_bytes(n, kwargs, expected):
    assert formatBytes(n, **kwargs) == expected


def test_format_bytes_errors():
    with pytest.raises(ValueError, match="n must"):
        formatBytes(-1)
    with pytest.raises(ValueError, match="precision"):
        formatBytes(10, precision=-1)
