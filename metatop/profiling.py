import cProfile
import functools
import io
import os
import pstats
import statistics
import time
import tracemalloc
import warnings
from contextlib import contextmanager

# Try importing optional dependencies
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Configuration


class ProfileConfig:
    """Configuration for the profiling module."""
    enabled = True  # Can be globally disabled
    output_format = 'text'  # 'text', 'csv', or 'html'
    sort_by = 'cumulative'  # 'cumulative', 'time', 'calls', etc.
    top_n = 20  # Number of functions to display in profiling results


def profile_function(output_file=None, sort_by=None, top_n=None, include_memory=False, track_lines=False):
    """
    Profile a function using cProfile with enhanced options.

    Parameters:
    -----------
    output_file : str, optional
        File to save profiling stats to
    sort_by : str, optional
        How to sort results ('cumulative', 'time', 'calls', etc.)
    top_n : int, optional
        Number of functions to display in results
    include_memory : bool, default=False
        Whether to track memory usage (requires memory_profiler)
    track_lines : bool, default=False
        Whether to use line_profiler (if available)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ProfileConfig.enabled:
                return func(*args, **kwargs)

            # Use line profiler if requested and available
            if track_lines and LINE_PROFILER_AVAILABLE:
                lp = LineProfiler()
                lp_wrapper = lp(func)
                result = lp_wrapper(*args, **kwargs)
                lp.print_stats()
                return result

            # Memory profiling
            if include_memory and MEMORY_PROFILER_AVAILABLE:
                tracemalloc.start()
                start_mem = tracemalloc.get_traced_memory()[
                    0] / (1024 * 1024)  # MB

            # CPU profiling
            profiler = cProfile.Profile()
            try:
                result = profiler.runcall(func, *args, **kwargs)

                # Memory profiling results
                if include_memory and MEMORY_PROFILER_AVAILABLE:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    current_mb = current / (1024 * 1024)  # MB
                    peak_mb = peak / (1024 * 1024)  # MB
                    mem_used = current_mb - start_mem
                    print(
                        f"Memory: Current: {current_mb:.2f}MB | Peak: {peak_mb:.2f}MB | Diff: {mem_used:.2f}MB")

                return result
            finally:
                s_by = sort_by or ProfileConfig.sort_by
                n = top_n or ProfileConfig.top_n

                if output_file:
                    profiler.dump_stats(output_file)
                    print(f"Profile data saved to {output_file}")
                else:
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats(s_by)
                    ps.print_stats(n)
                    print(s.getvalue())

                    if ProfileConfig.output_format == 'csv':
                        pass
                    elif ProfileConfig.output_format == 'html':
                        pass
        return wrapper
    return decorator


class BlockProfiler:
    """Enhanced block profiler that collects statistics over multiple runs."""

    def __init__(self, name="Block", collect_stats=False, silent=False):
        self.name = name
        self.collect_stats = collect_stats
        self.times = []
        self.running = False
        self.start_time = 0
        self.silent = silent

    def start(self):
        """Start timing the block."""
        if self.running:
            if not self.silent:
                warnings.warn(
                    f"Block '{self.name}' is already being profiled!")
            return

        self.running = True
        self.start_time = time.time()

        # Start memory tracking if available
        if MEMORY_PROFILER_AVAILABLE:
            self.tracemalloc_started = tracemalloc.is_tracing()
            if not self.tracemalloc_started:
                tracemalloc.start()
            self.start_mem = tracemalloc.get_traced_memory()[
                0] / (1024 * 1024)  # MB

    def stop(self):
        """Stop timing the block and report."""
        if not self.running:
            if not self.silent:
                warnings.warn(f"Block '{self.name}' was not being profiled!")
            return

        elapsed = time.time() - self.start_time

        # Memory tracking
        mem_info = ""
        if MEMORY_PROFILER_AVAILABLE:
            current, peak = tracemalloc.get_traced_memory()
            if not self.tracemalloc_started:
                tracemalloc.stop()
            current_mb = current / (1024 * 1024)  # MB
            mem_used = current_mb - self.start_mem
            mem_info = f" | Memory: +{mem_used:.2f}MB"

        if not self.silent:
            if self.collect_stats:
                self.times.append(elapsed)
                print(
                    f"{self.name} took {elapsed:.3f} seconds{mem_info} [Run {len(self.times)}]")
            else:
                print(f"{self.name} took {elapsed:.3f} seconds{mem_info}")
        else:
            # Even when silent, still record the time if collecting stats
            if self.collect_stats:
                self.times.append(elapsed)

        self.running = False

    def stats(self):
        """Report statistics if multiple runs were tracked."""
        if not self.times:
            print(f"No statistics available for '{self.name}'")
            return

        if len(self.times) > 1:
            avg = statistics.mean(self.times)
            if len(self.times) > 2:
                std = statistics.stdev(self.times)
                print(f"{self.name} statistics: runs={len(self.times)}, avg={avg:.3f}s, min={min(self.times):.3f}s, max={max(self.times):.3f}s, stdev={std:.3f}s")
            else:
                print(
                    f"{self.name} statistics: runs={len(self.times)}, avg={avg:.3f}s, min={min(self.times):.3f}s, max={max(self.times):.3f}s")


@contextmanager
def profile_block(name="Block", collect_stats=False):
    """Context manager for timing code blocks with optional statistics collection."""
    # Create profiler with silent mode when profiling is disabled
    profiler = BlockProfiler(name=name, collect_stats=collect_stats,
                             silent=not ProfileConfig.enabled)

    if ProfileConfig.enabled:
        profiler.start()
        try:
            yield profiler
        finally:
            profiler.stop()
            if collect_stats and len(profiler.times) > 1:
                profiler.stats()
    else:
        # Still yield the profiler object, but don't run any profiling
        try:
            yield profiler
        finally:
            pass  # Do nothing when profiling is disabled
