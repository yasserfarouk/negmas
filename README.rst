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

This package was designed to help advance the state-of-art in negotiation research by providing an easy-to-use yet
powerful platform for autonomous negotiation targeting situated simultaneous negotiations.
It grew out of the NEC-AIST collaborative laboratory project.

By *situated* negotiations, we mean those for which utility functions are not pre-ordained by fiat but are a natural
result of a simulated business-like process.

By *simultaneous* negotiations, we mean sessions of dependent negotiations for which the utility value of an agreement
of one session is affected by what happens in other sessions.

Main Features
=============

This platform was designed with both flexibility and scalability in mind. The key features of the NegMAS package are:

#. The public API is decoupled from internal details allowing for scalable implementations of the same interaction
   protocols. Supports both bilateral and multilateral negotiations. Supports agents engaging in multiple concurrent
   negotiations. Provides support for inter-negotiation synchronization.
#. The package provides multiple levels of abstraction in the specifications of the computational blocks required for
   negotiation allowing for gradual exposition to the subject.
#. The package provides sample negotiators that can be used as templates for more complex negotiators.
#. The package supports both mediated and unmediated negotiations.
#. Novel negotiation protocols can be added to the package as easily as adding novel negotiators.
#. Allows for non-traditional negotiation scenarios including dynamic entry/exit from the negotiation.

Basic Use cases
===============

To use negmas in a project::

.. code-block:: python

    import negmas


The package was designed for many uses cases. On one extreme, it can be used by an end user who is interested in running
one of the built-in negotiation protocols. On the other extreme, it can be used to develop novel kinds of negotiation
agents and negotiation protocols.

Running existing negotiators/negotiation protocols
--------------------------------------------------

Using the package for negotiation can be as simple as the following code snippet:

.. code-block:: python

    from negmas import SAOMechanism, AspirationNegotiator, MappingUtilityFunction
    session = SAOMechanism(outcomes=10, n_steps=100)
    negotiators = [AspirationNegotiator(name=f'a{_}') for _ in range(5)]
    for negotiator in negotiators:
        session.add(negotiator, ufun=MappingUtilityFunction(lambda x: random.random() * x[0]))

    session.run()

Developing a negotiator
-----------------------

Developing a novel negotiator slightly more difficult by is still doable in few lines of code:

.. code-block:: python

    from negmas.negotiators import Negotiator
    class MyAwsomeNegotiator(Negotiator):
        def __init__(self):
            # initialize the parents
            Negotiator.__init__(self)
            MultiNegotiationsMixin.__init__(self)

        def respond_(self, offer, state):
            # decide what to do when receiving an offer @ that negotiation
            pass

        def propose_(self, state):
            # proposed the required number of proposals (or less) @ that negotiation
            pass

By just implementing `respond_()` and `propose_()`. This negotiator is now capable of engaging in alternating offers
negotiations. See the documentation of `Negotiator` for a full description of available functionality out of the box.

Developing a negotiation protocol
---------------------------------

Developing a novel negotiation protocol is actually even simpler:

.. code-block:: python

    from negmas.mechanisms import Mechanism

    class MyNovelProtocol(Mechanism):
        def __init__(self):
            super().__init__()

        def round(self):
            # one step of the protocol
            pass

By implementing the single `round()` function, a new protocol is created. New negotiators can be added to the
negotiation using `add()` and removed using `remove()`. See the documentation for a full description of
`Mechanism` available functionality out of the box [Alternatively you can use `Protocol` instead of `Mechanism`].


Running a world simulation
--------------------------

TBD
