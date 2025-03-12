"""
Specialized profiling utilities for finite element code.
"""
# Add __all__ to explicitly export the necessary symbols
__all__ = ['fem_profiler', 'profile_assembly',
           'profile_solve', 'profile_fem_solution', 'FEMProfiler']

import time
import tracemalloc
from contextlib import contextmanager

import numpy as np

from .profiling import ProfileConfig, profile_block, profile_function


class FEMProfiler:
    """Profile specific phases of finite element computation."""

    def __init__(self):
        self.assembly_times = []
        self.solve_times = []
        self.postprocess_times = []
        self.total_times = []
        self.matrix_sizes = []
        self.nonzeros = []
        self.performance_metrics = {}

    def record_matrix_stats(self, matrix):
        """Record statistics about the system matrix."""
        if hasattr(matrix, 'shape'):
            self.matrix_sizes.append(matrix.shape)
        if hasattr(matrix, 'nnz'):
            self.nonzeros.append(matrix.nnz)
        elif hasattr(matrix, 'nonzero'):
            self.nonzeros.append(len(matrix.nonzero()[0]))

    def calculate_metrics(self):
        """Calculate performance metrics like MFLOPS, assembly efficiency, etc."""
        if self.matrix_sizes and self.solve_times:
            avg_size = np.mean([s[0] for s in self.matrix_sizes])
            avg_nnz = np.mean(self.nonzeros)
            avg_solve_time = np.mean(self.solve_times)

            # Theoretical FLOPS for typical sparse solver
            approx_flops = avg_nnz * np.log(avg_size)
            self.performance_metrics['MFLOPS'] = approx_flops / \
                (avg_solve_time * 1e6)

        if self.assembly_times and self.solve_times:
            total_compute = sum(self.assembly_times) + sum(self.solve_times)
            self.performance_metrics['assembly_ratio'] = sum(
                self.assembly_times) / total_compute
            self.performance_metrics['solve_ratio'] = sum(
                self.solve_times) / total_compute

    def report(self):
        """Generate a comprehensive performance report."""
        if not any([self.assembly_times, self.solve_times, self.postprocess_times]):
            print("No FEM profiling data available.")
            return

        print("\n===== FEM PROFILING REPORT =====")

        # Print time statistics for each phase
        if self.assembly_times:
            print(f"\nAssembly Times: {len(self.assembly_times)} runs")
            print(f"  Average: {np.mean(self.assembly_times):.4f}s")
            print(
                f"  Min/Max: {np.min(self.assembly_times):.4f}s / {np.max(self.assembly_times):.4f}s")

        if self.solve_times:
            print(f"\nSolver Times: {len(self.solve_times)} runs")
            print(f"  Average: {np.mean(self.solve_times):.4f}s")
            print(
                f"  Min/Max: {np.min(self.solve_times):.4f}s / {np.max(self.solve_times):.4f}s")

        if self.total_times:
            print(f"\nTotal Times: {len(self.total_times)} runs")
            print(f"  Average: {np.mean(self.total_times):.4f}s")
            print(
                f"  Min/Max: {np.min(self.total_times):.4f}s / {np.max(self.total_times):.4f}s")

        # Print matrix statistics
        if self.matrix_sizes:
            print(f"\nMatrix sizes: {self.matrix_sizes}")

        if self.nonzeros:
            print(f"\nNon-zeros: {self.nonzeros}")
            if self.matrix_sizes:
                sparsity = [nnz/(size[0]*size[1]) for nnz,
                            size in zip(self.nonzeros, self.matrix_sizes)]
                print(f"Sparsity: {[f'{s:.6f}' for s in sparsity]}")

        # Print performance metrics
        if self.performance_metrics:
            print("\nPerformance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"  {metric}: {value:.4f}")

        print("================================\n")

    def clear(self):
        """Reset all profiling data."""
        self.__init__()


fem_profiler = FEMProfiler()


@contextmanager
def profile_assembly(enabled=False):
    if not enabled:
        yield
        return

    """Profile the assembly phase of FEM computation."""
    with profile_block("Assembly") as p:
        yield
    fem_profiler.assembly_times.append(p.times[-1] if p.times else 0)


@contextmanager
def profile_solve(enabled=False, matrix=None):
    """Profile the solver phase of FEM computation."""
    if not enabled:
        yield
        return

    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        fem_profiler.solve_times.append(elapsed)
        print(f"Solve took {elapsed:.3f} seconds")
        if matrix is not None:
            fem_profiler.record_matrix_stats(matrix)


@contextmanager
def profile_fem_solution(enabled=False):
    """Profile the entire FEM solution process."""
    if not enabled:
        yield
        return

    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        fem_profiler.total_times.append(elapsed)
        if ProfileConfig.enabled:  # Only print if profiling is enabled
            print(f"Total FEM solution took {elapsed:.3f} seconds")
