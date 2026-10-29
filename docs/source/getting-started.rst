:tocdepth: -1

.. index:: getting-started

.. _getting-started:

================
Getting started
================

Install the package in a conda environment:

.. code-block:: bash

    conda create -n pdfcoherence_env -c conda-forge python=3.12 \
        --file requirements/conda.txt --file requirements/tests.txt
    conda activate pdfcoherence_env
    pip install -e . --no-deps

Analyse a single PDF, or every PDF in a folder, with a known instrumental
``Qdamp`` (in inverse Angstrom):

.. code-block:: bash

    pdfcoherence data/uio66_10nm_30percent_polydisp_noisy.cgr 0.01
    pdfcoherence data/ 0.01 --shape lognormal --polydispersity 0.3 -o processed/

The same analysis from Python:

.. code-block:: python

    from pdfcoherence.analysis import analyze
    from pdfcoherence.io import load_pdf

    r, gr = load_pdf("data/uio66_10nm_30percent_polydisp_noisy.cgr")
    result = analyze(r, gr, qdamp=0.01, shape="sphere")
    print(result.r_model, result.fit.param, result.status)

The README describes the method, the meaning of the outputs and the
limitations of the approach. ``r_model`` is a detection limit (the largest
distance with observable structural signal), not a particle size.
