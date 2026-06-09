"""Tests for the Gravity-Weighted DPO loss computation.

Tests the loss function in isolation using synthetic log-probability tensors,
not real models. Verifies margin scaling, gravity_alpha behavior, and
equivalence with standard DPO when margins are zero.
"""

import torch
import torch.nn.functional as F

from src.training.gw_dpo_trainer import compute_gw_dpo_loss


class TestGWDPOLoss:
    """Tests for the gravity-weighted DPO loss function."""

    def _standard_dpo_loss(
        self,
        beta: float,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Reference implementation of standard sigmoid DPO loss."""
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta = chosen_logratios - rejected_logratios
        return -F.logsigmoid(beta * delta)

    def test_zero_margin_matches_standard_dpo(self):
        """When margin=0 for all samples, GW-DPO equals standard DPO."""
        beta = 0.1
        chosen_logps = torch.tensor([-1.0, -2.0, -0.5])
        rejected_logps = torch.tensor([-3.0, -4.0, -2.5])
        ref_chosen_logps = torch.tensor([-1.5, -2.5, -1.0])
        ref_rejected_logps = torch.tensor([-3.5, -4.5, -3.0])
        margins = torch.tensor([0.0, 0.0, 0.0])

        gw_loss = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=margins,
        )
        std_loss = self._standard_dpo_loss(
            beta, chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )
        torch.testing.assert_close(gw_loss, std_loss)

    def test_positive_margin_increases_loss(self):
        """Larger margins make the loss harder to satisfy."""
        beta = 0.1
        chosen_logps = torch.tensor([-1.0])
        rejected_logps = torch.tensor([-3.0])
        ref_chosen_logps = torch.tensor([-1.5])
        ref_rejected_logps = torch.tensor([-3.5])

        loss_no_margin = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=torch.tensor([0.0]),
        )
        loss_with_margin = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=torch.tensor([2.0]),
        )
        assert loss_with_margin.item() > loss_no_margin.item()

    def test_larger_margin_means_larger_loss(self):
        """Gap=4 should produce higher loss than gap=1 for same logps."""
        beta = 0.1
        chosen_logps = torch.tensor([-1.0])
        rejected_logps = torch.tensor([-2.0])
        ref_chosen_logps = torch.tensor([-1.2])
        ref_rejected_logps = torch.tensor([-2.2])

        loss_gap1 = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=torch.tensor([1.0]),
        )
        loss_gap4 = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=torch.tensor([4.0]),
        )
        assert loss_gap4.item() > loss_gap1.item()

    def test_margin_scaling_by_gravity_alpha(self):
        """Doubling gravity_alpha doubles the effective margin."""
        beta = 0.1
        chosen_logps = torch.tensor([-1.0])
        rejected_logps = torch.tensor([-2.0])
        ref_chosen_logps = torch.tensor([-1.5])
        ref_rejected_logps = torch.tensor([-2.5])
        raw_gap = torch.tensor([2.0])

        # alpha=1.0, margin=2.0
        loss_a1 = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=raw_gap * 1.0,
        )
        # alpha=2.0, margin=4.0
        loss_a2 = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=raw_gap * 2.0,
        )
        # Manually verify: loss = -logsigmoid(beta * delta - margin)
        # With larger margin, loss should be strictly greater
        assert loss_a2.item() > loss_a1.item()

    def test_calibration_margin_zero_equals_standard(self):
        """Calibration pairs have margin=0, so they get standard DPO."""
        beta = 0.1
        chosen_logps = torch.tensor([-0.5, -1.0])
        rejected_logps = torch.tensor([-2.0, -3.0])
        ref_chosen_logps = torch.tensor([-0.8, -1.3])
        ref_rejected_logps = torch.tensor([-2.3, -3.3])
        # Calibration: margin=0 for both
        margins = torch.tensor([0.0, 0.0])

        gw_loss = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=margins,
        )
        std_loss = self._standard_dpo_loss(
            beta, chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )
        torch.testing.assert_close(gw_loss, std_loss)

    def test_loss_formula_exact(self):
        """Verify the exact formula: loss = -logsigmoid(beta * delta - margin)."""
        beta = 0.1
        chosen_logps = torch.tensor([-1.0])
        rejected_logps = torch.tensor([-3.0])
        ref_chosen_logps = torch.tensor([-1.5])
        ref_rejected_logps = torch.tensor([-3.5])
        margins = torch.tensor([2.0])

        gw_loss = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=margins,
        )

        # Manual computation
        delta = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
        expected = -F.logsigmoid(beta * delta - margins)
        torch.testing.assert_close(gw_loss, expected)

    def test_none_margins_matches_standard(self):
        """When margins is None, should equal standard DPO."""
        beta = 0.1
        chosen_logps = torch.tensor([-1.0, -2.0])
        rejected_logps = torch.tensor([-3.0, -4.0])
        ref_chosen_logps = torch.tensor([-1.5, -2.5])
        ref_rejected_logps = torch.tensor([-3.5, -4.5])

        gw_loss = compute_gw_dpo_loss(
            beta=beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=None,
        )
        std_loss = self._standard_dpo_loss(
            beta, chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )
        torch.testing.assert_close(gw_loss, std_loss)
