"""Isolate worker count and batching on one representative exit angular tile."""

import json
from time import perf_counter

import numpy as np
import vecdiff_general as o
from scipy.fft import ifft2
from vecdiff_general.propagation.surface_transfer import BandSynthesis


def batched(plan, coefficients, workers, batch):
    values = coefficients.reshape(-1, coefficients.shape[-1])[plan.ids].T
    total = np.zeros((coefficients.shape[-1],) + plan.shape, complex)
    for first in range(0, plan.rank, batch):
        terms = plan.terms[first : first + batch]
        polynomials = np.stack([term[0] for term in terms])
        factors = np.stack([term[1] for term in terms])
        workspace = np.zeros((len(terms),) + total.shape, complex)
        workspace.reshape(len(terms), len(values), -1)[:, :, plan.shifted_ids] = (
            factors[:, None, :] * values[None, :, :]
        )
        waves = ifft2(
            workspace,
            axes=(-2, -1),
            workers=workers,
            overwrite_x=True,
        )
        total += np.sum(polynomials[:, None] * waves, axis=0)
    total *= plan.carrier[None]
    return np.moveaxis(np.fft.fftshift(total, axes=(-2, -1)), 0, -1)


def run():
    count = 280
    axis = (np.arange(count) - count // 2) * 130 / count
    plane = o.CartesianGrid(axis, axis)
    q = o.SurfaceTransport(
        plane,
        o.SphericalCap(-100, plane.frame),
        o.HomogeneousRegion("glass", o.Dielectric(1.5)),
        1,
    )
    support = q.active & (abs(q.ku / q.k) < 0.3) & (abs(q.kv / q.k) < 0.3)
    plan = BandSynthesis(q, np.flatnonzero(support))
    rng = np.random.default_rng(47)
    coefficients = np.zeros(plane.shape + (2,), complex)
    coefficients[support] = rng.normal(size=(support.sum(), 2)) + 1j * rng.normal(
        size=(support.sum(), 2)
    )
    reference = q._synthesize(coefficients)
    rows = []
    for workers in [1, 2, 4, -1]:
        plan.workers = workers
        for batch in [0, 1, 2, 4, 8]:
            times = []
            for _ in range(4):
                tick = perf_counter()
                result = (
                    plan(coefficients)
                    if batch == 0
                    else batched(plan, coefficients, workers, batch)
                )
                times.append(perf_counter() - tick)
            row = dict(
                workers=workers,
                batch=batch,
                seconds=times,
                median_seconds=float(np.median(times[1:])),
                relative_error=float(
                    np.linalg.norm(result - reference) / np.linalg.norm(reference)
                ),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent / "reference_data/fresnel_kernel_benchmark.json"
    )
    path.write_text(
        json.dumps(dict(rank=plan.rank, original_rank=q.rank, rows=rows), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    run()
