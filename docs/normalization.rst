*************
Normalization
*************

This page is the **canonical specification** of how NegMAS normalizes and scales
utility functions. It describes the single normalization *funnel* shared by every
ufun type, the fast analytic path used for linear ufuns, and the exact guarantees
of the applied scheme (*Normalization Method 3*).

.. contents::
    :local:
    :depth: 2


Why normalize?
==============

Two negotiators rarely express their preferences on the same numeric scale. One
may score outcomes in :math:`[0, 100]`, another in :math:`[-3, 4]`. Normalization
rewrites a utility function into a common, comparable range (usually
:math:`[0, 1]`) **without changing the preference order of outcomes**, so that
utilities are meaningful to compare, plot, and analyze (Pareto frontiers, Nash
and Kalai points, social welfare, ...).

NegMAS offers three related entry points on every
:class:`~negmas.preferences.BaseUtilityFunction`:

``ufun.normalize(to=(0.0, 1.0), ...)``
    Normalize over the ufun's **own** outcome space.

``ufun.normalize_for(to=(0.0, 1.0), outcome_space=..., ...)``
    Normalize over an **explicitly given** outcome space.

``UtilityFunctionClass.normalize_all_for(ufuns, to=(0.0, 1.0), ...)``
    Normalize a **collection** of ufuns together onto one *shared* scale so their
    values stay comparable across ufuns (the multi-agent / common-scale case).


The single funnel
=================

All three entry points ultimately run through **one implementation**,
:meth:`~negmas.preferences.BaseUtilityFunction.normalize_all_for`. ``normalize``
and ``normalize_for`` are thin wrappers that call it with a single-element tuple::

    # conceptually
    def normalize_for(self, to=(0.0, 1.0), outcome_space=None, **kw):
        return type(self).normalize_all_for((self,), to=to,
                                             outcome_space=outcome_space, **kw)[0]

Keeping one funnel means the degenerate-range handling, the epsilon policy, and
the scale/shift math live in exactly one place, so the behavior can never drift
between the single-ufun and multi-ufun code paths.

Linear ufuns override this with a faster **analytic** path (see
:ref:`analytic-linear` below), but they preserve the same contract.

Guarantees and defaults
------------------------

The ``to`` argument is a ``(min, max)`` target. Either bound may be ``None`` to
enforce only the other one. Two flags control how strictly the bounds are hit:

``guarantee_max`` (default ``True``)
    Ensure the maximum utility equals ``to[1]``.

``guarantee_min``
    Ensure the minimum utility equals ``to[0]``.

    * For the single-ufun ``normalize`` / ``normalize_for``, the default is
      ``True``: the result is stretched to *exactly* ``[to[0], to[1]]``.
    * For the multi-ufun ``normalize_all_for``, the default is ``False``: the
      shared scale guarantees the **maximum** across the collection maps to
      ``to[1]`` while each ufun's minimum is left wherever the shared affine map
      places it (``>= to[0]``). This is the ``"max = 1, min >= 0"`` convention
      used when comparing several negotiators.

When ``guarantee_min=False``, a ufun already lying inside ``to`` is left
untouched (an early-skip), rather than being stretched.

.. _degenerate:

Degenerate (constant) utilities
-------------------------------

If the range is smaller than ``NORMALIZE_EPS`` (:math:`10^{-10}`) the ufun is
treated as **constant** and cannot be stretched to a non-zero range. In that case
NegMAS returns a :class:`~negmas.preferences.ConstUtilityFunction` mapped to a
target **endpoint chosen by the reserved value**:

* if the constant beats its reservation (``value >= reserved_value``) it maps to
  ``to[1]`` (it reads as "best"), and the reserved value maps to ``to[0]``;
* otherwise it maps to ``to[0]`` and the reserved value maps to ``to[1]``.

This endpoint-by-reserved rule is used identically by the funnel and by the
analytic linear override.

Epsilon policy
--------------

Two tolerances are used throughout the stack (module constants in
``negmas.preferences.base_ufun``):

``NORMALIZE_EPS`` (:math:`10^{-10}`)
    The *degenerate-range* cutoff: below this, a range counts as constant. It is
    safely above floating-point noise in sampled min/max yet below any meaningful
    utility range.

``NORMALIZED_TOLERANCE`` (:math:`10^{-6}`)
    The *already-normalized* tolerance used by the early-skip and within-limits
    checks.


.. _analytic-linear:

The analytic path for linear ufuns
==================================

Sampling every outcome to find ``min``/``max`` is wasteful for
:class:`~negmas.preferences.AffineUtilityFunction`,
:class:`~negmas.preferences.LinearUtilityFunction`, and
:class:`~negmas.preferences.LinearAdditiveUtilityFunction`: their extrema and the
whole transform can be computed in closed form. These classes therefore
**override** ``normalize_for`` (and route ``normalize`` to it) with an analytic
algorithm that produces the same result as the funnel but without enumerating
outcomes.

For an affine ufun :math:`u(\omega) = b + \sum_i \alpha_i f_i(\omega_i)` normalized
to ``to`` with extrema ``mn, mx``, the transform is the affine map
:math:`\bar u = a\,u + c` with

.. math::

    a = \frac{to_1 - to_0}{mx - mn}, \qquad c = to_1 - a\cdot mx .

Crucially the **reserved value undergoes the same map**,
:math:`\bar r = a\,r + c`, so the relative position of the reservation with
respect to the outcomes is preserved (see :ref:`method3` guarantee #2).

.. note::

    **Normalizing an affine ufun to** :math:`[0, 1]` **returns the Method-3
    canonical form.** Because an affine ufun is a linear-additive ufun with
    implicit identity value functions, ``AffineUtilityFunction.normalize_for(to=(0,
    1))`` delegates to the :ref:`method3` algorithm and returns a
    :class:`~negmas.preferences.LinearAdditiveUtilityFunction` (weights summing to
    1, zero bias). The utility *values* and reserved value are identical to the
    direct affine map, but this canonical form round-trips faithfully to Genius
    XML (which has no bias field). Other target ranges keep the plain affine
    representation, and genuinely constant ufuns still follow the
    :ref:`degenerate` (endpoint-by-reserved) rule.

``LinearAdditiveUtilityFunction`` uses the full *Method 3* canonicalization
described next.


.. _method3:

Normalization Method 3 (the applied scheme)
===========================================

For linear-additive ufuns, NegMAS applies **Normalization Method 3**: an affine,
utility-preserving map that additionally makes every weight non-negative, every
value function all-positive with a maximum of 1, and the weights sum to 1. It is
the scheme that satisfies all ten conditions below.

The original ufun is

.. math::

    u(\omega) = b + \sum_{i=1}^{N} \alpha_i f_i(\omega_i), \qquad r \in \Re

where :math:`N` is the number of issues, :math:`r` is the reserved value,
:math:`\alpha_i` is a constant weight, :math:`f_i: I_i \rightarrow \Re` is a value
function on the domain :math:`I_i` of the :math:`i`-th issue, and :math:`b` is a
constant bias.

The normalized function :math:`\bar u(\omega) = \sum_{i=1}^{N} \bar\alpha_i
\bar f_i(\omega_i)` is guaranteed to satisfy:

1. :math:`\bar u(\omega) = a \times u(\omega) + c` where :math:`a, c \in \Re`
2. :math:`\bar r = a \times r + c`
3. :math:`0 \le \bar u(\omega) \le 1`
4. :math:`\exists\, \omega \in \Omega: \bar u(\omega) = 1`
5. :math:`0 \le \bar\alpha_i \le 1`
6. :math:`\sum_{i=1}^{N} \bar\alpha_i = 1`
7. :math:`0 \le \bar f_i(\omega_i) \le 1`
8. :math:`\exists\, \omega_i \in I_i: \bar f_i(\omega_i) = 1`
9. :math:`\exists\, \omega_j \in I_i: \bar f_i(\omega_j) = 0`
10. :math:`\exists\, \omega \in \Omega: \bar u(\omega) = 0`

Why conditions 1 & 2 matter
---------------------------

Conditions 1 and 2 say the whole ufun **and** its reserved value undergo the
**same affine map** :math:`\bar u = a\,u + c`. Geometrically, if you scatter one
negotiator's utility on the x-axis and another's on the y-axis, the picture does
not change except for per-axis scale: the shape of the outcome cloud, the Pareto
frontier, and the positions of the two reserved-value lines relative to the
outcomes are all preserved. A non-affine normalization would distort this picture,
so NegMAS requires an affine method.

The algorithm
-------------

1. **Make all weights positive.** For each issue, if :math:`\alpha_i \le 0`, flip
   the value function (:math:`\tilde f_i \leftarrow -f_i`) and negate the weight
   (:math:`\tilde\alpha_i \leftarrow |\alpha_i|`). This leaves :math:`u` unchanged.
2. **Make each value function non-negative.** With
   :math:`m_i = \min_{\omega_i} \tilde f_i(\omega_i)`, set
   :math:`\tilde f_i \leftarrow \tilde f_i - m_i` and fold the shift into the bias,
   :math:`\tilde b \leftarrow b + \sum_i \tilde\alpha_i m_i`. Now
   :math:`\min \tilde f_i = 0`.
3. **Find the weight sum** :math:`s \leftarrow \sum_i \tilde\alpha_i`.
4. **Normalize each value function to** :math:`[0, 1]`. With
   :math:`M_i = \max_{\omega_i} \tilde f_i(\omega_i)` and
   :math:`\beta_i = 1/M_i` (or 1 if :math:`M_i = 0`), scale
   :math:`\bar f_i = \beta_i \tilde f_i` and compensate in the weight
   :math:`\hat\alpha_i = \tilde\alpha_i / \beta_i`.
5. **Update the reserved value** for the bias/shift folded in during step 2:
   :math:`\hat r = r - \tilde b`, and set the bias to 0. Because both the value
   functions and the weights change together, :math:`u` itself is unchanged.
6. **Make the weights sum to one.** With :math:`s = \sum_i \hat\alpha_i`, set
   :math:`\bar\alpha_i = \hat\alpha_i / s` and :math:`\bar r = \hat r / s`.

This yields the affine map with :math:`a = 1/s` and
:math:`c = -\tilde b / s`, satisfying all ten conditions.

Assumptions and edge cases
--------------------------

Method 3 satisfies all ten conditions provided:

* the outcome space is a Cartesian product of **independent issues** (so per-issue
  extrema are jointly achievable — needed for conditions 3, 4 and 10);
* the ufun is **not globally constant** (:math:`s > 0`); a constant utility cannot
  be mapped to :math:`[0, 1]` with both a maximum of 1 and a minimum of 0 — see
  :ref:`degenerate`;
* **no value function is constant.** A constant value function is a single point
  and cannot satisfy both condition 8 (:math:`=1`) and condition 9 (:math:`=0`).

.. note::

    **Deliberate variation for constant value functions.** Unlike literal
    Method 3 (which sends a constant value function to 0 with weight 0), NegMAS
    maps a constant value function to **1** and keeps its weight, so the required
    "maximum is 1" guarantee (condition 8) holds for every issue. The consequence
    is that such an issue never reaches 0, so conditions 9 and 10 are not
    guaranteed when a constant issue is present. The map stays affine (conditions
    1 and 2 always hold).


Normalizing whole scenarios
===========================

:meth:`Scenario.normalize <negmas.inout.Scenario.normalize>` normalizes every ufun
in a scenario:

* ``independent=True`` normalizes each ufun on its own (each gets its own
  :math:`[0, 1]` range and the full per-issue guarantees, via ``normalize_for``);
* ``independent=False`` / ``common_range=True`` (the default) uses the shared-scale
  affine map from ``normalize_all_for`` so utilities stay comparable **across**
  ufuns.

``Scenario.normalize`` also exposes the reserved-value repair options
(``normalize_reserved_values=True``, ``reserved_value_penalty=...``) that correct
non-finite reserved values before normalizing.


Worked examples
===============

.. code-block:: python

    from negmas import make_issue
    from negmas.preferences import LinearAdditiveUtilityFunction
    from negmas.preferences.value_fun import AffineFun

    issues = (make_issue(10, "quantity"), make_issue(10, "quality"))
    # Mixed-sign weights and a non-[0,1] value function
    u = LinearAdditiveUtilityFunction(
        values=[AffineFun(1.0, -5.0), AffineFun(2.0, -3.0)],
        weights=[5.0, -3.0],
        issues=issues,
        reserved_value=2.0,
    )
    n = u.normalize()  # Method 3, delegates to normalize_for

    # All ten guarantees hold:
    assert abs(sum(n.weights) - 1.0) < 1e-6  # weights sum to one (cond. 6)
    assert all(w >= -1e-9 for w in n.weights)  # non-negative weights (cond. 5)
    us = [n(o) for o in n.outcome_space.enumerate_or_sample()]
    assert abs(max(us) - 1.0) < 1e-6  # reaches 1 (cond. 4)
    assert abs(min(us) - 0.0) < 1e-6  # reaches 0 (cond. 10)

.. code-block:: python

    from negmas import make_issue
    from negmas.preferences import AffineUtilityFunction

    issues = (make_issue(10),)
    # utilities in [10, 19], reserved value 15
    u = AffineUtilityFunction(weights=[1.0], bias=10.0, issues=issues, reserved_value=15.0)
    n = u.normalize_for(to=(0.0, 1.0))
    # the reserved value follows the same affine map: (15-10)/(19-10) = 5/9
    assert abs(n.reserved_value - 5.0 / 9.0) < 1e-6

.. code-block:: python

    from negmas import make_issue
    from negmas.preferences import LinearUtilityFunction

    issues = (make_issue([0, 5, 10], "x"), make_issue([5, 10, 15], "y"))
    u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)  # range [0, 10]
    u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)  # range [5, 15]
    # Shared scale so the two ufuns stay comparable:
    n1, n2 = LinearUtilityFunction.normalize_all_for((u1, u2), to=(0.0, 1.0))
