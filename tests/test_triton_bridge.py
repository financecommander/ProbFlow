"""Tests for probflow.integration.triton_bridge module."""

import pytest

from probflow.distributions.discrete import Bernoulli
from probflow.integration.triton_bridge import (
    BinaryDist,
    TernaryValue,
    prob_to_triton,
    triton_to_prob,
)


# ---------------------------------------------------------------------------
# triton_to_prob tests
# ---------------------------------------------------------------------------


class TestTritonToProb:
    def test_true_maps_to_high_probability(self) -> None:
        dist = triton_to_prob(TernaryValue.TRUE)
        assert isinstance(dist, Bernoulli)
        assert dist.p == pytest.approx(0.95)

    def test_false_maps_to_low_probability(self) -> None:
        dist = triton_to_prob(TernaryValue.FALSE)
        assert isinstance(dist, Bernoulli)
        assert dist.p == pytest.approx(0.05)

    def test_unknown_maps_to_half(self) -> None:
        dist = triton_to_prob(TernaryValue.UNKNOWN)
        assert isinstance(dist, Bernoulli)
        assert dist.p == pytest.approx(0.5)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected TernaryValue"):
            triton_to_prob("TRUE")

    def test_invalid_type_int_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected TernaryValue"):
            triton_to_prob(1)


# ---------------------------------------------------------------------------
# prob_to_triton tests
# ---------------------------------------------------------------------------


class TestProbToTriton:
    def test_high_probability_is_true(self) -> None:
        assert prob_to_triton(Bernoulli(0.95)) == TernaryValue.TRUE

    def test_low_probability_is_false(self) -> None:
        assert prob_to_triton(Bernoulli(0.05)) == TernaryValue.FALSE

    def test_mid_probability_is_unknown(self) -> None:
        assert prob_to_triton(Bernoulli(0.5)) == TernaryValue.UNKNOWN

    def test_exact_true_threshold(self) -> None:
        """p exactly at true_threshold classifies as TRUE."""
        assert prob_to_triton(Bernoulli(0.9)) == TernaryValue.TRUE

    def test_exact_false_threshold(self) -> None:
        """p exactly at false_threshold classifies as FALSE."""
        assert prob_to_triton(Bernoulli(0.1)) == TernaryValue.FALSE

    def test_just_below_true_threshold(self) -> None:
        assert prob_to_triton(Bernoulli(0.89)) == TernaryValue.UNKNOWN

    def test_just_above_false_threshold(self) -> None:
        assert prob_to_triton(Bernoulli(0.11)) == TernaryValue.UNKNOWN

    def test_custom_thresholds(self) -> None:
        """Custom thresholds for domain-specific calibration."""
        result = prob_to_triton(
            Bernoulli(0.7), true_threshold=0.7, false_threshold=0.3
        )
        assert result == TernaryValue.TRUE

    def test_custom_thresholds_false(self) -> None:
        result = prob_to_triton(
            Bernoulli(0.3), true_threshold=0.7, false_threshold=0.3
        )
        assert result == TernaryValue.FALSE

    def test_custom_thresholds_unknown(self) -> None:
        result = prob_to_triton(
            Bernoulli(0.5), true_threshold=0.7, false_threshold=0.3
        )
        assert result == TernaryValue.UNKNOWN

    def test_tight_thresholds(self) -> None:
        """Equal thresholds: only exact value is TRUE/FALSE, rest UNKNOWN."""
        result = prob_to_triton(
            Bernoulli(0.5), true_threshold=0.5, false_threshold=0.5
        )
        assert result == TernaryValue.TRUE

    def test_zero_probability(self) -> None:
        assert prob_to_triton(Bernoulli(0.0)) == TernaryValue.FALSE

    def test_one_probability(self) -> None:
        assert prob_to_triton(Bernoulli(1.0)) == TernaryValue.TRUE

    def test_invalid_dist_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected Bernoulli"):
            prob_to_triton(0.5)

    def test_true_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="true_threshold"):
            prob_to_triton(Bernoulli(0.5), true_threshold=1.5)

    def test_false_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="false_threshold"):
            prob_to_triton(Bernoulli(0.5), false_threshold=-0.1)

    def test_false_exceeds_true_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            prob_to_triton(
                Bernoulli(0.5), true_threshold=0.3, false_threshold=0.7
            )


# ---------------------------------------------------------------------------
# Round-trip consistency tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_true_round_trip(self) -> None:
        """TRUE → Bernoulli → TRUE."""
        dist = triton_to_prob(TernaryValue.TRUE)
        result = prob_to_triton(dist)
        assert result == TernaryValue.TRUE

    def test_false_round_trip(self) -> None:
        """FALSE → Bernoulli → FALSE."""
        dist = triton_to_prob(TernaryValue.FALSE)
        result = prob_to_triton(dist)
        assert result == TernaryValue.FALSE

    def test_unknown_round_trip(self) -> None:
        """UNKNOWN → Bernoulli → UNKNOWN."""
        dist = triton_to_prob(TernaryValue.UNKNOWN)
        result = prob_to_triton(dist)
        assert result == TernaryValue.UNKNOWN

    def test_round_trip_all_values(self) -> None:
        """All ternary values survive a round trip."""
        for tv in TernaryValue:
            dist = triton_to_prob(tv)
            assert prob_to_triton(dist) == tv

    def test_round_trip_custom_thresholds(self) -> None:
        """Round trip with custom thresholds still preserves identity."""
        for tv in TernaryValue:
            dist = triton_to_prob(tv)
            result = prob_to_triton(
                dist, true_threshold=0.8, false_threshold=0.2
            )
            assert result == tv


# ---------------------------------------------------------------------------
# BinaryDist tests
# ---------------------------------------------------------------------------


class TestBinaryDist:
    def test_create_from_bernoulli(self) -> None:
        bd = BinaryDist(Bernoulli(0.95))
        assert bd.p == pytest.approx(0.95)

    def test_ternary_property_true(self) -> None:
        bd = BinaryDist(Bernoulli(0.95))
        assert bd.ternary == TernaryValue.TRUE

    def test_ternary_property_false(self) -> None:
        bd = BinaryDist(Bernoulli(0.05))
        assert bd.ternary == TernaryValue.FALSE

    def test_ternary_property_unknown(self) -> None:
        bd = BinaryDist(Bernoulli(0.5))
        assert bd.ternary == TernaryValue.UNKNOWN

    def test_from_ternary_classmethod(self) -> None:
        bd = BinaryDist.from_ternary(TernaryValue.TRUE)
        assert bd.p == pytest.approx(0.95)
        assert bd.ternary == TernaryValue.TRUE

    def test_from_ternary_all_values(self) -> None:
        for tv in TernaryValue:
            bd = BinaryDist.from_ternary(tv)
            assert bd.ternary == tv

    def test_custom_thresholds(self) -> None:
        bd = BinaryDist(Bernoulli(0.7), true_threshold=0.7, false_threshold=0.3)
        assert bd.ternary == TernaryValue.TRUE

    def test_and_operator(self) -> None:
        a = BinaryDist(Bernoulli(0.9))
        b = BinaryDist(Bernoulli(0.8))
        result = a & b
        assert isinstance(result, BinaryDist)
        assert result.p == pytest.approx(0.9 * 0.8)

    def test_or_operator(self) -> None:
        a = BinaryDist(Bernoulli(0.3))
        b = BinaryDist(Bernoulli(0.4))
        result = a | b
        assert isinstance(result, BinaryDist)
        assert result.p == pytest.approx(0.3 + 0.4 - 0.3 * 0.4)

    def test_and_preserves_thresholds(self) -> None:
        a = BinaryDist(Bernoulli(0.9), true_threshold=0.8, false_threshold=0.2)
        b = BinaryDist(Bernoulli(0.8), true_threshold=0.7, false_threshold=0.3)
        result = a & b
        assert result.true_threshold == 0.8
        assert result.false_threshold == 0.2

    def test_invalid_dist_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected Bernoulli"):
            BinaryDist(0.5)

    def test_invalid_true_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="true_threshold"):
            BinaryDist(Bernoulli(0.5), true_threshold=1.5)

    def test_invalid_false_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="false_threshold"):
            BinaryDist(Bernoulli(0.5), false_threshold=-0.1)

    def test_false_exceeds_true_raises(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            BinaryDist(Bernoulli(0.5), true_threshold=0.3, false_threshold=0.7)

    def test_repr(self) -> None:
        bd = BinaryDist(Bernoulli(0.95))
        assert "BinaryDist" in repr(bd)
        assert "0.95" in repr(bd)
        assert "TRUE" in repr(bd)

    def test_repr_unknown(self) -> None:
        bd = BinaryDist(Bernoulli(0.5))
        assert "UNKNOWN" in repr(bd)


# ---------------------------------------------------------------------------
# Integration with Triton AST node patterns
# ---------------------------------------------------------------------------


class TestTritonASTIntegration:
    """Tests simulating integration with Triton AST nodes."""

    def test_ast_node_true_evaluation(self) -> None:
        """Simulate evaluating a Triton AST node that resolves to TRUE."""
        node_value = TernaryValue.TRUE
        dist = triton_to_prob(node_value)
        assert dist.p > 0.9
        assert prob_to_triton(dist) == TernaryValue.TRUE

    def test_ast_node_false_evaluation(self) -> None:
        """Simulate evaluating a Triton AST node that resolves to FALSE."""
        node_value = TernaryValue.FALSE
        dist = triton_to_prob(node_value)
        assert dist.p < 0.1
        assert prob_to_triton(dist) == TernaryValue.FALSE

    def test_ast_node_unknown_propagation(self) -> None:
        """UNKNOWN nodes should propagate uncertainty."""
        node_value = TernaryValue.UNKNOWN
        dist = triton_to_prob(node_value)
        assert 0.1 < dist.p < 0.9

    def test_conjunction_of_ast_nodes(self) -> None:
        """AND of two TRUE nodes should remain TRUE-like."""
        a = BinaryDist.from_ternary(TernaryValue.TRUE)
        b = BinaryDist.from_ternary(TernaryValue.TRUE)
        result = a & b
        assert result.p == pytest.approx(0.95 * 0.95)
        assert result.ternary == TernaryValue.TRUE

    def test_conjunction_with_unknown(self) -> None:
        """AND of TRUE and UNKNOWN should reduce to UNKNOWN."""
        a = BinaryDist.from_ternary(TernaryValue.TRUE)
        b = BinaryDist.from_ternary(TernaryValue.UNKNOWN)
        result = a & b
        assert result.p == pytest.approx(0.95 * 0.5)
        assert result.ternary == TernaryValue.UNKNOWN

    def test_disjunction_of_ast_nodes(self) -> None:
        """OR of two FALSE nodes should remain FALSE-like."""
        a = BinaryDist.from_ternary(TernaryValue.FALSE)
        b = BinaryDist.from_ternary(TernaryValue.FALSE)
        result = a | b
        expected_p = 0.05 + 0.05 - 0.05 * 0.05
        assert result.p == pytest.approx(expected_p)
        assert result.ternary == TernaryValue.FALSE

    def test_disjunction_with_true(self) -> None:
        """OR with TRUE should push toward TRUE."""
        a = BinaryDist.from_ternary(TernaryValue.FALSE)
        b = BinaryDist.from_ternary(TernaryValue.TRUE)
        result = a | b
        assert result.ternary == TernaryValue.TRUE

    def test_domain_specific_calibration(self) -> None:
        """Medical domain: stricter thresholds for safety."""
        dist = Bernoulli(0.85)
        # Standard thresholds: UNKNOWN
        assert prob_to_triton(dist) == TernaryValue.UNKNOWN
        # Relaxed thresholds for screening: TRUE
        assert prob_to_triton(
            dist, true_threshold=0.8, false_threshold=0.2
        ) == TernaryValue.TRUE
