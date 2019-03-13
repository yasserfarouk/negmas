.. image:: https://img.shields.io/pypi/pyversions/negmas.svg
        :target: https://pypi.python.org/pypi/negmas
        :alt: Python

.. image:: https://img.shields.io/pypi/status/negmas.svg
        :target: https://pypi.python.org/pypi/negmas
        :alt: Pypi

.. image:: https://img.shields.io/pypi/l/negmas.svg
        :target: https://pypi.python.org/pypi/negmas
        :alt: License

.. image:: https://img.shields.io/pypi/dm/negmas.svg
        :target: https://pypi.python.org/pypi/negmas
        :alt: Downloads

.. image:: https://img.shields.io/pypi/v/negmas.svg
        :target: https://pypi.python.org/pypi/negmas
        :alt: Pypi

.. image:: https://img.shields.io/travis/yasserfarouk/negmas.svg
        :target: https://travis-ci.org/yasserfarouk/negmas
        :alt: Build Status

.. image:: https://readthedocs.org/projects/negmas/badge/?version=latest
        :target: https://negmas/readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



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

The documentation is available at: documentation_

.. _documentation: http://www.yasserm.com/negmas/index.html

Main Features
=============

This platform was designed with both flexibility and scalability in mind. The key features of the NegMAS package are:

#. The public API is decoupled from internal details allowing for scalable implementations of the same interaction
   protocols.
#. Supports agents engaging in multiple concurrent negotiations.
#. Provides support for inter-negotiation synchronization either through coupled utility functions or through central
   *control* agents.
#. The package provides sample negotiators that can be used as templates for more complex negotiators.
#. The package supports both mediated and unmediated negotiations.
#. Supports both bilateral and multilateral negotiations.
#. Novel negotiation protocols and simulated *worlds* can be added to the package as easily as adding novel negotiators.
#. Allows for non-traditional negotiation scenarios including dynamic entry/exit from the negotiation.
#. A large variety of built in utility functions.
#. Utility functions can be active dynamic entities which allows the system to model a much wider range of dynamic ufuns
   compared with existing packages.
#. A distributed system with the same interface and industrial-strength implementation is being created allowing agents
   developed for NegMAS to be seemingly employed in real-world business operations.

To use negmas in a project

.. code-block:: python

    import negmas

The package was designed for many uses cases. On one extreme, it can be used by an end user who is interested in running
one of the built-in negotiation protocols. On the other extreme, it can be used to develop novel kinds of negotiation
agents, negotiation protocols, multi-agent simulations (usually involving situated negotiations), etc.

Running existing negotiators/negotiation protocols
==================================================

Using the package for negotiation can be as simple as the following code snippet:

.. code-block:: python

    from negmas import SAOMechanism, AspirationNegotiator, MappingUtilityFunction
    session = SAOMechanism(outcomes=10, n_steps=100)
    negotiators = [AspirationNegotiator(name=f'a{_}') for _ in range(5)]
    for negotiator in negotiators:
        session.add(negotiator, ufun=MappingUtilityFunction(lambda x: random.random() * x[0]))

    session.run()

In this snippet, we created a mechanism session with an outcome-space of *10* discrete outcomes that would run for *10*
steps. Five agents with random utility functions are then created and *added* to the session. Finally the session is
*run* to completion. The agreement (if any) can then be accessed through the *state* member of the session. The library
provides several analytic and visualization tools to inspect negotiations. See the first tutorial on
*Running a Negotiation* for more details.

Developing a negotiator
=======================

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
=================================

Developing a novel negotiation protocol is actually even simpler:

.. code-block:: python

    from negmas.mechanisms import Mechanism

    class MyNovelProtocol(Mechanism):
        def __init__(self):
            super().__init__()

        def step_(self):
            # one step of the protocol
            pass

By implementing the single `step_()` function, a new protocol is created. New negotiators can be added to the
negotiation using `add()` and removed using `remove()`. See the documentation for a full description of
`Mechanism` available functionality out of the box [Alternatively you can use `Protocol` instead of `Mechanism`].


Running a world simulation
==========================

The *raison d'être* for NegMAS is to allow you to develop negotiation agents capable of behaving in realistic
*business like* simulated environments. These simulations are called *worlds* in NegMAS. Agents interact with each other
within these simulated environments trying their maximize some intrinsic utility function of the agent through several
*possibly simultaneous* negotiations.

The `situated` module provides all that you need to create such worlds. An example can be found in the `scml` package.
This package implements a supply chain management system in which factory managers compete to maximize their profits in
a market with only negotiations as the means of securing contracts.


Acknowledgement
===============

.. _Genius: http://ii.tudelft.nl/genius

NegMAS tests use scenarios used in ANAC 2010 to ANAC 2018 competitions obtained from the Genius_ Platform. These domains
can be found in the tests/data and notebooks/data folders.


.. image:: https://api.codacy.com/project/badge/Grade/8ea7e86796c243f79d654de43458ed73
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/yasserfarouk/negmas?utm_source=github.com&utm_medium=referral&utm_content=yasserfarouk/negmas&utm_campaign=Badge_Grade_Dashboard