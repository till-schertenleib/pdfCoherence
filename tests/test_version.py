"""Unit tests for __version__.py."""

import pdfcoherence  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(pdfcoherence, "__version__")
    assert pdfcoherence.__version__ != "0.0.0"
