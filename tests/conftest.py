from pathlib import Path

import numpy as np
import pytest

from pdfcoherence.envelopes import qdamp_envelope, sphere

DATA_DIR = Path(__file__).parents[1] / "data"


def make_pdf(diameter=60.0, qdamp=0.0, noise=0.05, seed=0, r_max=300.0):
    """Synthetic G(r): random sines with unit RMS, damped by a sphere
    envelope and Qdamp, plus white noise."""
    rng = np.random.default_rng(seed)
    r = np.arange(0.1, r_max, 0.1)
    q = rng.uniform(1.5, 8.0, 60)
    phase = rng.uniform(0, 2 * np.pi, 60)
    bulk = np.sqrt(2 / 60) * np.sin(np.outer(r, q) + phase).sum(axis=1)
    signal = bulk * sphere(r, diameter) * qdamp_envelope(r, qdamp)
    return r, signal + noise * rng.standard_normal(len(r))


@pytest.fixture
def synthetic_pdf():
    return make_pdf()
