"""Compute the Golomb-Dickman constant via parallel tanh-sinh quadrature.

The Golomb-Dickman constant:
    λ = ∫₀¹ e^{li(x)} dx ≈ 0.6243299885435508709929363831008372496989979704917...

where li(x) is the logarithmic integral Ei(ln x).
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any

import mpmath


def integrand(x: mpmath.mpf) -> mpmath.mpf:
    """Return exp(li(x)), the integrand of the Golomb-Dickman constant."""
    return mpmath.exp(mpmath.li(x))


def _init_worker(dps: int) -> None:
    """Set mpmath working precision in each worker process."""
    mpmath.mp.dps = dps


def _eval_node(args: tuple[mpmath.mpf, mpmath.mpf]) -> tuple[mpmath.mpf, mpmath.mpf]:
    """Evaluate the integrand at one quadrature node: (x, w) -> (w, f(x))."""
    x, w = args
    return w, integrand(x)


class ParallelTanhSinh(mpmath.calculus.quadrature.TanhSinh):
    """Tanh-Sinh quadrature that distributes node evaluations across a process pool."""

    def __init__(self, ctx: Any, pool: Any, chunksize: int = 10) -> None:
        super().__init__(ctx)
        self.pool = pool
        self.chunksize = chunksize

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False) -> mpmath.mpf:
        """Sum one refinement level of the tanh-sinh hierarchy in parallel."""
        print(f"Integrating degree {degree} with {len(nodes)} nodes...")
        results = self.pool.imap_unordered(_eval_node, nodes, chunksize=self.chunksize)
        total = self.ctx.mpf(0)
        for w, fx in results:
            total += w * fx
        return total


def compute_golomb_dickman(dps: int = 1000, processes: int = 12) -> mpmath.mpf:
    """Compute the Golomb-Dickman constant to *dps* decimal places.

    Uses parallel tanh-sinh quadrature with *processes* worker processes.
    """
    mpmath.mp.dps = dps
    original_tanhsinh = mpmath.mp.TanhSinh
    with multiprocessing.Pool(
        processes=processes, initializer=_init_worker, initargs=(dps,)
    ) as pool:
        rule = ParallelTanhSinh(mpmath.mp, pool)
        mpmath.mp.TanhSinh = lambda ctx: rule  # type: ignore[method-assign]
        try:
            result = mpmath.quad(integrand, [0, 1], method='tanh-sinh')
        finally:
            mpmath.mp.TanhSinh = original_tanhsinh  # type: ignore[method-assign]
    return result


if __name__ == '__main__':
    t0 = time.time()
    result = compute_golomb_dickman(dps=1000, processes=12)
    elapsed = time.time() - t0
    print("Parallel time:", elapsed)
    print("Result:", result)
