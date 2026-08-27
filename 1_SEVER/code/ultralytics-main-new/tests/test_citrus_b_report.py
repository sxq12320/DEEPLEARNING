"""Small policy tests for the CitrusB result reporter."""

from report_citrus_b_results import decision, reference_family


def test_loss_and_architecture_references_do_not_mix():
    """Loss ablations compare with BL00, while structures compare with B00."""
    assert reference_family("B09_recall_balanced") == "B00_reference"
    assert reference_family("BL09_bq_vfl") == "BL00_none"


def test_reporter_uses_a_conservative_noise_floor():
    """Tiny positive changes remain inconclusive and clear negative changes are rejected."""
    assert decision(0.002, 0.01, 0.003) == "within one-run noise"
    assert decision(-0.004, 0.01, 0.003) == "provisionally harmful"
