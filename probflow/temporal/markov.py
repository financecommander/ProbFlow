"""Markov chain and Hidden Markov Model implementations."""

import numpy as np
from scipy import linalg


class MarkovChain:
    """Discrete-time Markov chain.

    Parameters
    ----------
    states : list
        List of state labels.
    transition_matrix : array-like
        Row-stochastic transition matrix where ``transition_matrix[i][j]``
        is the probability of transitioning from state *i* to state *j*.
    """

    def __init__(self, states, transition_matrix):
        self.states = list(states)
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        n = len(self.states)
        if self.transition_matrix.shape != (n, n):
            raise ValueError(
                f"Transition matrix shape {self.transition_matrix.shape} "
                f"does not match number of states ({n})."
            )
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of the transition matrix must sum to 1.")
        if np.any(self.transition_matrix < 0):
            raise ValueError("Transition probabilities must be non-negative.")

    def forecast(self, horizon, initial_state=None, initial_dist=None):
        """Forecast state probability distributions over time.

        Parameters
        ----------
        horizon : int
            Number of time steps to forecast.
        initial_state : optional
            Starting state label.  Mutually exclusive with *initial_dist*.
        initial_dist : array-like, optional
            Starting probability distribution over states.  If neither
            *initial_state* nor *initial_dist* is given, a uniform
            distribution is used.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(horizon + 1, n_states)`` where each row is
            the probability distribution at that time step.
        """
        n = len(self.states)

        if initial_state is not None and initial_dist is not None:
            raise ValueError(
                "Provide either initial_state or initial_dist, not both."
            )

        if initial_state is not None:
            idx = self.states.index(initial_state)
            dist = np.zeros(n)
            dist[idx] = 1.0
        elif initial_dist is not None:
            dist = np.asarray(initial_dist, dtype=float)
        else:
            dist = np.ones(n) / n

        result = np.zeros((horizon + 1, n))
        result[0] = dist
        for t in range(1, horizon + 1):
            dist = dist @ self.transition_matrix
            result[t] = dist
        return result

    def stationary_distribution(self):
        """Find the stationary (equilibrium) distribution via eigenvalue decomposition.

        Returns
        -------
        numpy.ndarray
            The stationary probability distribution.
        """
        # The stationary distribution is the left eigenvector of the
        # transition matrix corresponding to eigenvalue 1.
        eigenvalues, eigenvectors = linalg.eig(self.transition_matrix, left=True, right=False)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)


class HiddenMarkovModel:
    """Hidden Markov Model with Baum-Welch fitting and Viterbi decoding.

    Parameters
    ----------
    hidden_states : list
        List of hidden state labels.
    observables : list
        List of observable symbol labels.
    transition_probs : array-like, optional
        Row-stochastic transition matrix for hidden states.  If ``None``,
        initialised uniformly.
    emission_probs : array-like, optional
        Emission probability matrix of shape ``(n_hidden, n_obs)`` where
        ``emission_probs[i][j]`` is the probability of emitting observable
        *j* when in hidden state *i*.  If ``None``, initialised uniformly.
    initial_probs : array-like, optional
        Initial hidden-state distribution.  Defaults to uniform.
    """

    def __init__(
        self,
        hidden_states,
        observables,
        transition_probs=None,
        emission_probs=None,
        initial_probs=None,
    ):
        self.hidden_states = list(hidden_states)
        self.observables = list(observables)
        n_hidden = len(self.hidden_states)
        n_obs = len(self.observables)

        if transition_probs is not None:
            self.transition_probs = np.asarray(transition_probs, dtype=float)
        else:
            self.transition_probs = np.ones((n_hidden, n_hidden)) / n_hidden

        if emission_probs is not None:
            self.emission_probs = np.asarray(emission_probs, dtype=float)
        else:
            self.emission_probs = np.ones((n_hidden, n_obs)) / n_obs

        if initial_probs is not None:
            self.initial_probs = np.asarray(initial_probs, dtype=float)
        else:
            self.initial_probs = np.ones(n_hidden) / n_hidden

    def _obs_indices(self, observations):
        """Convert observation labels to integer indices."""
        return [self.observables.index(o) for o in observations]

    def _forward(self, obs_idx):
        """Forward algorithm.  Returns alpha matrix and scaling factors."""
        T = len(obs_idx)
        n = len(self.hidden_states)
        alpha = np.zeros((T, n))
        scale = np.zeros(T)

        alpha[0] = self.initial_probs * self.emission_probs[:, obs_idx[0]]
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transition_probs) * self.emission_probs[:, obs_idx[t]]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        return alpha, scale

    def _backward(self, obs_idx, scale):
        """Backward algorithm.  Returns beta matrix."""
        T = len(obs_idx)
        n = len(self.hidden_states)
        beta = np.zeros((T, n))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.transition_probs @ (self.emission_probs[:, obs_idx[t + 1]] * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        return beta

    def fit(self, data, max_iter=100, tol=1e-6):
        """Fit the model parameters using the Baum-Welch EM algorithm.

        Parameters
        ----------
        data : list
            Sequence of observed symbols (labels).
        max_iter : int
            Maximum number of EM iterations.
        tol : float
            Convergence tolerance on log-likelihood.

        Returns
        -------
        self
        """
        obs_idx = self._obs_indices(data)
        T = len(obs_idx)
        n = len(self.hidden_states)
        m = len(self.observables)

        prev_ll = -np.inf
        for _ in range(max_iter):
            alpha, scale = self._forward(obs_idx)
            beta = self._backward(obs_idx, scale)

            # Log-likelihood from scaling factors
            ll = np.sum(np.log(scale[scale > 0]))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # Compute xi and gamma
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1.0
            gamma = gamma / gamma_sum

            xi = np.zeros((T - 1, n, n))
            for t in range(T - 1):
                numer = (
                    np.outer(alpha[t], self.emission_probs[:, obs_idx[t + 1]] * beta[t + 1])
                    * self.transition_probs
                )
                denom = numer.sum()
                if denom > 0:
                    xi[t] = numer / denom

            # Update parameters
            self.initial_probs = gamma[0]

            xi_sum = xi.sum(axis=0)
            gamma_sum_trans = gamma[:-1].sum(axis=0)
            for i in range(n):
                if gamma_sum_trans[i] > 0:
                    self.transition_probs[i] = xi_sum[i] / gamma_sum_trans[i]

            for j in range(m):
                mask = np.array([1.0 if obs_idx[t] == j else 0.0 for t in range(T)])
                for i in range(n):
                    g_sum = gamma[:, i].sum()
                    if g_sum > 0:
                        self.emission_probs[i, j] = (gamma[:, i] * mask).sum() / g_sum

        return self

    def infer_state(self, observations):
        """Decode the most likely hidden state sequence using the Viterbi algorithm.

        Parameters
        ----------
        observations : list
            Sequence of observed symbols (labels).

        Returns
        -------
        list
            Most likely sequence of hidden state labels.
        """
        obs_idx = self._obs_indices(observations)
        T = len(obs_idx)
        n = len(self.hidden_states)

        # Log probabilities to avoid underflow
        with np.errstate(divide="ignore"):
            log_trans = np.log(self.transition_probs)
            log_emit = np.log(self.emission_probs)
            log_init = np.log(self.initial_probs)

        V = np.zeros((T, n))
        path = np.zeros((T, n), dtype=int)

        V[0] = log_init + log_emit[:, obs_idx[0]]

        for t in range(1, T):
            for j in range(n):
                candidates = V[t - 1] + log_trans[:, j]
                path[t, j] = np.argmax(candidates)
                V[t, j] = candidates[path[t, j]] + log_emit[j, obs_idx[t]]

        # Backtrack
        states_seq = np.zeros(T, dtype=int)
        states_seq[T - 1] = np.argmax(V[T - 1])
        for t in range(T - 2, -1, -1):
            states_seq[t] = path[t + 1, states_seq[t + 1]]

        return [self.hidden_states[s] for s in states_seq]
