import pytest
from conftest import DATA_DIR

from pdfcoherence.cli import run

EXAMPLE = DATA_DIR / "uio66_10nm_30percent_polydisp_noisy.cgr"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example data not found")
def test_cli_on_example_data(tmp_path):
    rows = run([str(EXAMPLE), "0.01", "-o", str(tmp_path)])
    assert (tmp_path / f"{EXAMPLE.stem}_coherence_analysis.png").exists()
    assert (tmp_path / "summary_coherence_report.csv").exists()
    assert (tmp_path / "summary_coherence_table.tex").exists()
    assert rows[0]["r_model"] == pytest.approx(103.0, abs=5.0)


def test_cli_missing_target(tmp_path):
    with pytest.raises(SystemExit, match="No PDF files"):
        run([str(tmp_path), "0.01"])
