======
NegMAS
======


.. .. image:: https://img.shields.io/pypi/v/negmas/svg
..         :target: https://pypi.python.org/pypi/negmas

.. .. image:: https://img.shields.io/travis/yasserfarouk/negmas/svg
..         :target: https://travis-ci.org/yasserfarouk/negmas

.. .. image:: https://readthedocs.org/projects/negmas/badge/?version=latest
..         :target: https://negmas/readthedocs.io/en/latest/?badge=latest
..         :alt: Documentation Status




A python library for managing autonomous negotiation agents in simulation environments. The name ``negmas`` stands for
either NEGotiation MultiAgent System or NEGotiations Managed by Agent Simulations (your pick).

Introduction
============

This package was designed to help advance the state-of-art in negotiation research by providing an easy-to-use yet powerful platform for autonomous negotiation. It grew out of the NEC-AIST collaborative laboratory project.

The main purpose of this package is to provide the public interface to be used in the implementation of the negotiation platform. Because of that, no attempts were made to optimize the internal implementation of sample negotiators. Moreover, the package is trying to follow best-practice in Python library design but that does not constraint implementations if carried out in different languages from utilizing these languages’ best practices as well even by modifying the provided interface.

Main Features
=============

This platform was designed with both flexability and scalability in mind. The key features of the negmas package are:

1. The public API is decoupled from internal details allowing for scalable implementations of the same interaction
   protocols. Supports both bilateral and multilateral negotiations. Supports negotiators engaging in multiple concurrent
   negotiations. Provides support for inter-negotiation synchronization.

1. The package provides multiple levels of abstraction in the specifications of the computaitonal blocks required for
   negotiation allowing for gradual exposition to the subject.

1. There is always a single base class for every type of entity in the negotiation (e.g. Negotiator. Protocol, etc).
   Different options are implemented via Mixins increasing the flexibility of the system.

The package provides sample negotiators that can be used as templates for more complex negotiators.
The package allows for both mediated and unmediated negotiations.
Novel negotiation protocols can be added to the package as easily as adding novel negotiators.
Allows for non-traditional negotiation scenarios including dynamic entry/exit from the negotiation.
Has built-in support for experimenting with elicitation protocols for both single and multi-issue auctions.


What else is provided?
----------------------

* **Name Resolution** The package supports name resolution for finding partners using the `YellowPages` class.
* **Taking Initiative** Any `Negotiator` (including `Negotiator` s) can create `Protocol` s and
  invite other `Negotiator` s to it. The Moderator negotiator need not be around.
* **Elicitation** Utility elicitation is supported through the `elicitors` module and sample elicitors
  are available in the `sample.elicitors` module.


Basic Use cases
===============

To use negmas in a project::

    import negmas

The package was designed for many uses cases. On one extreme, it can be used by an end user who is interested in running
one of the built-in negotiation protocols. On the other extreme, it can be used to develop novel kinds of negotiation
negotiators and negotiation protocols. This section gives some examples of both kinds of usages. Please refer to the full
documentation for more concrete examples.

Running existing negotiators/negotiation protocols
--------------------------------------------------

Using the package for negotiation can be as simple as the following code snippet:

.. code-block:: python

    from negmas import SAOMechanism, AspirationNegotiator, MappingUtilityFunction
    neg = SAOMechanism(outcomes=10)
    agents = [AspirationNegotiator() for _ in range(5)]
    for agent in agents:
        neg.add(agent, ufun = MappingUtilityFunction(lambda x: rand() * x[0]))
    neg.run()

Developing a negotiator
------------------------

Developing a novel negotiator slightly more difficult by is still doable in few lines of code:

.. code-block:: python

    from negmas.negotiators import Negotiator
    class MyAwsomeNegotiator(Negotiator, MultiNegotiationsMixin):
        def __init__(self):
            # initialize the parents
            Negotiator.__init__(self)
            MultiNegotiationsMixin.__init__(self)

        def respond(self, offer, negotiation = None):
            # decide what to do when receiving an offer @ that negotiation
            pass

        def propose(self, n=1, negotiation=None):
            # proposed the required number of proposals (or less) @ that negotiation
            pass

By just implementing `respond()` and `propose()`. This negotiator is now capable of engaging in multiple
concurrent negotiations. It can access all the negotiations it is involved in using ``self.negotiations``
and can enter and leave negotiations (if allowed by the protocol) using `enter()` and `leave()`.
See the documentation of `Negotiator` for a full description of available functionality out of the box.

Developing a negotiation protocol
---------------------------------

Developing a novel negotiation protocol is actually even simpler:

.. code-block:: python

    from negmas.mechanisms import Mechanism

    class MyNovelProtocol(Mechanism):
        def __init__(self, n_steps, n_outcomes):
            super().__init__(n_steps=n_steps, n_outcomes=n_outcomes)

        def step(self):
            # one step of the protocol
            pass

By implementing the single `step()` function, a new protocol is created. New negotiators can be added to the
negotiation using `add()` and removed using `remove()`. See the documentation for a full description of
`Protocol` available functionality out of the box.
