#!/usr/bin/env python
##############################################################################
#
# (c) 2026 Till Schertenleib.
# All rights reserved.
#
# File coded by: Till Schertenleib.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/till-schertenleib/real-space-coherence-lenght-extractor/graphs/contributors  # noqa: E501
#
# See LICENSE for license information.
#
##############################################################################
"""Definition of __version__."""

#  We do not use the other three variables, but can be added back if needed.
#  __all__ = ["__date__", "__git_commit__", "__timestamp__", "__version__"]

# obtain version information
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pdfcoherence")
except PackageNotFoundError:
    __version__ = "unknown"
