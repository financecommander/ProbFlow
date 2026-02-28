MCMC Convergence
================

Overview
--------

Markov Chain Monte Carlo (MCMC) methods generate samples from a target
distribution :math:`\pi(\mathbf{x})` by constructing a Markov chain whose
stationary distribution is :math:`\pi`. The key challenge is determining when
the chain has **converged** — that is, when the samples are representative of
the target distribution.

Markov Chain Fundamentals
-------------------------

A Markov chain with transition kernel :math:`T(\mathbf{x}' | \mathbf{x})` has
stationary distribution :math:`\pi` if:

.. math::

   \pi(\mathbf{x}') = \int T(\mathbf{x}' | \mathbf{x}) \, \pi(\mathbf{x}) \, d\mathbf{x}

The chain is **ergodic** (and therefore converges to :math:`\pi`) if it is:

1. **Irreducible**: Every state can be reached from every other state.
2. **Aperiodic**: The chain does not cycle deterministically.
3. **Positive recurrent**: Expected return time to any state is finite.

Metropolis-Hastings Algorithm
-----------------------------

The Metropolis-Hastings (MH) algorithm constructs a reversible Markov chain by
proposing moves from a proposal distribution :math:`q(\mathbf{x}' | \mathbf{x})`
and accepting them with probability:

.. math::

   \alpha(\mathbf{x}', \mathbf{x}) = \min\left(1,
   \frac{\pi(\mathbf{x}') \, q(\mathbf{x} | \mathbf{x}')}
        {\pi(\mathbf{x}) \, q(\mathbf{x}' | \mathbf{x})}\right)

This guarantees **detailed balance**:

.. math::

   \pi(\mathbf{x}) \, T(\mathbf{x}' | \mathbf{x})
   = \pi(\mathbf{x}') \, T(\mathbf{x} | \mathbf{x}')

Convergence Diagnostics
-----------------------

Several diagnostics are used in practice to assess MCMC convergence:

Gelman-Rubin Statistic (:math:`\hat{R}`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gelman-Rubin diagnostic compares between-chain and within-chain variance
across :math:`m` parallel chains, each of length :math:`n`:

.. math::

   \hat{R} = \sqrt{\frac{\hat{V}}{W}}

where:

- :math:`W` is the within-chain variance:

  .. math::

     W = \frac{1}{m} \sum_{j=1}^{m} s_j^2

- :math:`B` is the between-chain variance:

  .. math::

     B = \frac{n}{m-1} \sum_{j=1}^{m} (\bar{\theta}_j - \bar{\theta})^2

- :math:`\hat{V} = \frac{n-1}{n} W + \frac{1}{n} B`

Values of :math:`\hat{R} < 1.01` indicate approximate convergence.

Effective Sample Size (ESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The effective sample size accounts for autocorrelation in the chain:

.. math::

   \text{ESS} = \frac{n}{1 + 2 \sum_{k=1}^{\infty} \rho_k}

where :math:`\rho_k` is the autocorrelation at lag :math:`k`. Higher ESS
indicates more independent information content in the samples.

Trace Plots
~~~~~~~~~~~

Visual inspection of parameter traces across iterations can reveal
non-stationarity, multimodality, or poor mixing.

Burn-in and Thinning
--------------------

- **Burn-in**: Discarding initial samples to reduce the influence of the
  starting point. The burn-in period should be chosen so that the chain has
  reached its stationary distribution.
- **Thinning**: Retaining every :math:`k`-th sample to reduce autocorrelation.
  While thinning reduces storage, it discards information and is generally
  less preferred than simply running longer chains.

Connection to ProbFlow
----------------------

ProbFlow distributions provide the building blocks for defining target
distributions in MCMC sampling. The :meth:`~probflow.core.types.Dist.pdf`
method supplies the (unnormalized) target density needed by Metropolis-Hastings,
while :meth:`~probflow.core.types.Dist.sample` provides proposal distributions
and initialization. The :class:`~probflow.core.context.ProbFlow` context
manager organizes the model components that define the joint distribution to be
sampled.
