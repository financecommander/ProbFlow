"""Tests for MarkovChain and HiddenMarkovModel."""

import numpy as np
import pytest

from probflow.temporal.markov import HiddenMarkovModel, MarkovChain


# ---------------------------------------------------------------------------
# MarkovChain tests
# ---------------------------------------------------------------------------


class TestMarkovChain:
    """Tests for the MarkovChain class."""

    def test_weather_model_forecast(self):
        """Weather model with known transition probabilities."""
        states = ["Sunny", "Rainy"]
        # P(Sunny->Sunny)=0.8, P(Sunny->Rainy)=0.2
        # P(Rainy->Sunny)=0.4, P(Rainy->Rainy)=0.6
        transition = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(states, transition)

        result = mc.forecast(3, initial_state="Sunny")
        assert result.shape == (4, 2)
        # At t=0, distribution is [1, 0]
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0])
        # At t=1, distribution is [0.8, 0.2]
        np.testing.assert_array_almost_equal(result[1], [0.8, 0.2])
        # Each row should sum to 1
        for row in result:
            assert abs(row.sum() - 1.0) < 1e-10

    def test_forecast_with_initial_dist(self):
        """Forecast starting from a probability distribution."""
        mc = MarkovChain(["A", "B"], [[0.7, 0.3], [0.4, 0.6]])
        result = mc.forecast(2, initial_dist=[0.5, 0.5])
        np.testing.assert_array_almost_equal(result[0], [0.5, 0.5])
        expected_t1 = np.array([0.5, 0.5]) @ np.array([[0.7, 0.3], [0.4, 0.6]])
        np.testing.assert_array_almost_equal(result[1], expected_t1)

    def test_forecast_default_uniform(self):
        """Forecast defaults to uniform initial distribution."""
        mc = MarkovChain(["A", "B"], [[0.7, 0.3], [0.4, 0.6]])
        result = mc.forecast(1)
        np.testing.assert_array_almost_equal(result[0], [0.5, 0.5])

    def test_stationary_distribution(self):
        """Stationary distribution via eigenvalue decomposition."""
        states = ["Sunny", "Rainy"]
        transition = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(states, transition)

        pi = mc.stationary_distribution()
        # Known stationary: pi = [2/3, 1/3]
        np.testing.assert_array_almost_equal(pi, [2 / 3, 1 / 3], decimal=6)
        # Verify it sums to 1
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_convergence_to_stationary(self):
        """Forecast over long horizon converges to stationary distribution."""
        states = ["Sunny", "Rainy"]
        transition = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(states, transition)

        pi = mc.stationary_distribution()
        result = mc.forecast(200, initial_state="Rainy")
        np.testing.assert_array_almost_equal(result[-1], pi, decimal=6)

    def test_three_state_chain(self):
        """Three-state Markov chain stationary distribution."""
        states = ["A", "B", "C"]
        transition = [
            [0.1, 0.6, 0.3],
            [0.4, 0.2, 0.4],
            [0.3, 0.3, 0.4],
        ]
        mc = MarkovChain(states, transition)
        pi = mc.stationary_distribution()
        # Verify stationarity: pi @ T = pi
        np.testing.assert_array_almost_equal(pi @ transition, pi, decimal=6)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_invalid_transition_matrix_shape(self):
        """Transition matrix with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            MarkovChain(["A", "B"], [[0.5, 0.5]])

    def test_invalid_row_sums(self):
        """Transition matrix rows not summing to 1 raises ValueError."""
        with pytest.raises(ValueError, match="sum to 1"):
            MarkovChain(["A", "B"], [[0.5, 0.3], [0.4, 0.6]])

    def test_negative_probabilities(self):
        """Negative probabilities raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            MarkovChain(["A", "B"], [[-0.1, 1.1], [0.4, 0.6]])

    def test_initial_state_and_dist_both_raises(self):
        """Providing both initial_state and initial_dist raises ValueError."""
        mc = MarkovChain(["A", "B"], [[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="not both"):
            mc.forecast(5, initial_state="A", initial_dist=[0.5, 0.5])


# ---------------------------------------------------------------------------
# HiddenMarkovModel tests
# ---------------------------------------------------------------------------


class TestHiddenMarkovModel:
    """Tests for the HiddenMarkovModel class."""

    def test_viterbi_decode_accuracy(self):
        """Viterbi decoding on a simple two-state HMM."""
        hidden = ["Healthy", "Fever"]
        obs = ["Normal", "Cold", "Dizzy"]
        trans = [[0.7, 0.3], [0.4, 0.6]]
        emit = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]
        init = [0.6, 0.4]

        hmm = HiddenMarkovModel(
            hidden, obs,
            transition_probs=trans,
            emission_probs=emit,
            initial_probs=init,
        )
        seq = hmm.infer_state(["Normal", "Cold", "Dizzy"])
        # Known solution: Healthy, Healthy, Fever
        assert seq == ["Healthy", "Healthy", "Fever"]

    def test_viterbi_all_same_observation(self):
        """Viterbi with all-same observations."""
        hidden = ["A", "B"]
        obs = ["X", "Y"]
        trans = [[0.9, 0.1], [0.2, 0.8]]
        emit = [[0.9, 0.1], [0.1, 0.9]]
        init = [0.5, 0.5]

        hmm = HiddenMarkovModel(hidden, obs,
                                transition_probs=trans,
                                emission_probs=emit,
                                initial_probs=init)
        # All X observations should decode to mostly state A
        seq = hmm.infer_state(["X", "X", "X", "X", "X"])
        assert all(s == "A" for s in seq)

    def test_baum_welch_fit(self):
        """Baum-Welch EM recovers reasonable parameters from generated data."""
        np.random.seed(42)

        # True model parameters
        hidden = ["H", "L"]
        obs = ["A", "B", "C"]
        true_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
        true_emit = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
        true_init = np.array([0.6, 0.4])

        # Generate data from the true model
        T = 500
        states_seq = np.zeros(T, dtype=int)
        obs_seq = []
        states_seq[0] = np.random.choice(2, p=true_init)
        obs_seq.append(obs[np.random.choice(3, p=true_emit[states_seq[0]])])
        for t in range(1, T):
            states_seq[t] = np.random.choice(2, p=true_trans[states_seq[t - 1]])
            obs_seq.append(obs[np.random.choice(3, p=true_emit[states_seq[t]])])

        # Fit model from uniform initialization
        hmm = HiddenMarkovModel(hidden, obs)
        hmm.fit(obs_seq, max_iter=100)

        # The learned emission probabilities should roughly separate
        # the observation distributions (order of hidden states may differ).
        # Check that rows of emission_probs are valid distributions
        for i in range(2):
            assert abs(hmm.emission_probs[i].sum() - 1.0) < 1e-6
            assert np.all(hmm.emission_probs[i] >= 0)

        # Check that transition matrix rows sum to 1
        for i in range(2):
            assert abs(hmm.transition_probs[i].sum() - 1.0) < 1e-6

    def test_hmm_decode_generated_sequence(self):
        """HMM with known parameters decodes a generated sequence with
        reasonable accuracy."""
        np.random.seed(123)

        hidden = ["S0", "S1"]
        obs = ["X", "Y"]
        trans = np.array([[0.9, 0.1], [0.2, 0.8]])
        emit = np.array([[0.9, 0.1], [0.1, 0.9]])
        init = np.array([0.5, 0.5])

        # Generate sequence
        T = 100
        true_states = np.zeros(T, dtype=int)
        observations = []
        true_states[0] = np.random.choice(2, p=init)
        observations.append(obs[np.random.choice(2, p=emit[true_states[0]])])
        for t in range(1, T):
            true_states[t] = np.random.choice(2, p=trans[true_states[t - 1]])
            observations.append(obs[np.random.choice(2, p=emit[true_states[t]])])

        hmm = HiddenMarkovModel(hidden, obs,
                                transition_probs=trans,
                                emission_probs=emit,
                                initial_probs=init)
        decoded = hmm.infer_state(observations)
        decoded_idx = [hidden.index(s) for s in decoded]

        accuracy = np.mean(np.array(decoded_idx) == true_states)
        # Well-separated emissions should give high accuracy
        assert accuracy > 0.8

    def test_fit_returns_self(self):
        """fit() returns the model instance for chaining."""
        hmm = HiddenMarkovModel(["A", "B"], ["X", "Y"])
        result = hmm.fit(["X", "Y", "X"], max_iter=5)
        assert result is hmm
