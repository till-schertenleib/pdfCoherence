#!/usr/bin/env python
##############################################################################
#
# (c) 2026 Till Schertenleib.
# All rights reserved.
#
# File coded by: Till Schertenleib.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/till-schertenleib/pdfCoherence/graphs/contributors
#
# See LICENSE for license information.
#
##############################################################################
"""Estimate the real-space coherence length of a material from its
PDF."""

from pdfcoherence.analysis import CoherenceResult, analyze
from pdfcoherence.io import find_pdf_files, load_pdf
from pdfcoherence.plotting import plot_result
from pdfcoherence.version import __version__

__all__ = [
    "CoherenceResult",
    "__version__",
    "analyze",
    "find_pdf_files",
    "load_pdf",
    "plot_result",
]

# End of file
