Belief Propagation Algorithm
============================

Overview
--------

Belief propagation (BP), also known as sum-product message passing, is an
algorithm for performing inference on graphical models. It computes marginal
distributions for each unobserved node, conditioned on observed nodes.

Factor Graphs
-------------

Belief propagation operates on **factor graphs**, bipartite graphs connecting
variable nodes :math:`x_i` to factor nodes :math:`f_a`. Each factor encodes a
local compatibility function:

.. math::

   p(\mathbf{x}) = \frac{1}{Z} \prod_{a} f_a(\mathbf{x}_a)

where :math:`\mathbf{x}_a` is the subset of variables connected to factor
:math:`f_a` and :math:`Z` is the partition function.

Message Passing
---------------

The algorithm proceeds by passing messages between variable nodes and factor
nodes:

**Variable-to-factor messages:**

.. math::

   \mu_{x_i \to f_a}(x_i) = \prod_{b \in N(i) \setminus a} \mu_{f_b \to x_i}(x_i)

**Factor-to-variable messages:**

.. math::

   \mu_{f_a \to x_i}(x_i) = \sum_{\mathbf{x}_a \setminus x_i}
   f_a(\mathbf{x}_a) \prod_{j \in N(a) \setminus i}
   \mu_{x_j \to f_a}(x_j)

where :math:`N(i)` denotes the set of factor nodes neighboring variable
:math:`x_i`, and :math:`N(a)` denotes the set of variable nodes neighboring
factor :math:`f_a`.

Marginal Computation
--------------------

After convergence, the approximate marginal for variable :math:`x_i` is:

.. math::

   p(x_i) \propto \prod_{a \in N(i)} \mu_{f_a \to x_i}(x_i)

Convergence Properties
----------------------

On **tree-structured** factor graphs, belief propagation converges in a finite
number of iterations and produces exact marginals. For graphs with cycles
(**loopy** belief propagation), the algorithm may not converge, but often
provides good approximations in practice.

Key convergence conditions:

- **Tree graphs**: Exact convergence guaranteed.
- **Loopy graphs**: Convergence is not guaranteed; damping or scheduling
  heuristics are often applied.
- **Gaussian models**: For jointly Gaussian variables, loopy BP converges to
  the correct means (though variances may be incorrect).

Connection to ProbFlow
----------------------

In ProbFlow, the :class:`~probflow.core.types.Dist` base class and the
:class:`~probflow.core.context.ProbFlow` context manager provide the building
blocks for constructing factor graphs. Distribution composition operators
(``+``, ``*``, ``&``) correspond to constructing factors that link variable
nodes, enabling belief propagation over composed probabilistic models.
