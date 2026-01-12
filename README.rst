NegMAS: Negotiation Multi-Agent System
======================================

.. start-badges

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

.. image:: https://img.shields.io/codacy/grade/1b204fe0a69e41a298a175ea225d7b81.svg
    :target: https://app.codacy.com/project/yasserfarouk/negmas/dashboard
    :alt: Code Quality

.. image:: https://img.shields.io/pypi/v/negmas.svg
    :target: https://pypi.python.org/pypi/negmas
    :alt: Pypi

.. image:: https://github.com/yasserfarouk/negmas/workflows/CI/badge.svg
    :target: https://www.github.com/yasserfarouk/negmas
    :alt: Build Status

.. image:: https://codecov.io/gh/yasserfarouk/negmas/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/yasserfarouk/negmas
    :alt: Coverage Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black
    :alt: Coding style black

.. image:: https://static.pepy.tech/personalized-badge/negmas?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads
 :target: https://pepy.tech/projects/negmas

.. end-badges

NegMAS is a Python library for developing autonomous negotiation agents embedded in simulation
environments. It supports bilateral and multilateral negotiations, multiple negotiation protocols,
and complex multi-agent simulations with interconnected negotiations.

**Documentation:** https://negmas.readthedocs.io/

Installation
------------

.. code-block:: bash

    pip install negmas

For additional features:

.. code-block:: bash

    # With Genius bridge support (Java-based agents)
    pip install negmas[genius]

    # With visualization support
    pip install negmas[plots]

    # All optional dependencies
    pip install negmas[all]

Quick Start
-----------

**Run a simple negotiation in 10 lines:**

.. code-block:: python

    from negmas import SAOMechanism, TimeBasedConcedingNegotiator, make_issue
    from negmas.preferences import LinearAdditiveUtilityFunction as LUFun

    # Define what we're negotiating about
    issues = [make_issue(name="price", values=100)]

    # Create negotiation session (Stacked Alternating Offers)
    session = SAOMechanism(issues=issues, n_steps=50)

    # Add buyer (prefers low price) and seller (prefers high price)
    session.add(
        TimeBasedConcedingNegotiator(name="buyer"),
        ufun=LUFun.random(issues, reserved_value=0.0),
    )
    session.add(
        TimeBasedConcedingNegotiator(name="seller"),
        ufun=LUFun.random(issues, reserved_value=0.0),
    )

    # Run and get result
    result = session.run()
    print(f"Agreement: {result.agreement}, Rounds: {result.step}")

**Multi-issue negotiation with custom preferences:**

.. code-block:: python

    from negmas import SAOMechanism, AspirationNegotiator, make_issue
    from negmas.preferences import LinearAdditiveUtilityFunction

    # Create a 2-issue negotiation domain
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=5),
    ]

    # Define utility functions
    buyer_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": lambda x: 1.0 - x / 10.0,  # lower price = better
            "quantity": lambda x: x / 5.0,  # more quantity = better
        },
        issues=issues,
    )
    seller_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": lambda x: x / 10.0,  # higher price = better
            "quantity": lambda x: 1.0 - x / 5.0,  # less quantity = better
        },
        issues=issues,
    )

    # Run negotiation
    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(AspirationNegotiator(name="buyer"), ufun=buyer_ufun)
    session.add(AspirationNegotiator(name="seller"), ufun=seller_ufun)
    session.run()

    # Visualize
    session.plot()

Command Line Interface
----------------------

NegMAS includes a ``negotiate`` CLI for quick experimentation:

.. code-block:: bash

    # Run with default negotiators
    negotiate -s 50

    # Specify negotiators and steps
    negotiate -n AspirationNegotiator -n NaiveTitForTatNegotiator -s 100

    # Use Python-native Genius agents (no Java required)
    negotiate -n GBoulware -n GConceder -s 50

    # Use custom BOA components
    negotiate -n "boa:offering=GTimeDependentOffering(e=0.2),acceptance=GACNext" -n AspirationNegotiator

    # Save results and plot
    negotiate -s 100 --save-path ./results

See ``negotiate --help`` for all options, or the `CLI documentation <https://negmas.readthedocs.io/en/latest/scripts.html>`_.

Architecture Overview
---------------------

NegMAS is built around four core concepts:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                           WORLD                                 │
    │  (Simulation environment where agents interact)                 │
    │                                                                 │
    │   ┌─────────┐     ┌─────────┐         ┌─────────────────────┐  │
    │   │  Agent  │     │  Agent  │   ...   │  BulletinBoard      │  │
    │   │         │     │         │         │  (Public info)      │  │
    │   └────┬────┘     └────┬────┘         └─────────────────────┘  │
    │        │               │                                        │
    │        │ creates       │ creates                                │
    │        ▼               ▼                                        │
    │   ┌─────────────────────────────────────────────────────────┐  │
    │   │                    MECHANISM                             │  │
    │   │  (Negotiation protocol: SAO, SingleText, Auction, etc.) │  │
    │   │                                                          │  │
    │   │   ┌────────────┐  ┌────────────┐  ┌────────────┐        │  │
    │   │   │ Negotiator │  │ Negotiator │  │ Negotiator │        │  │
    │   │   │  + UFun    │  │  + UFun    │  │  + UFun    │        │  │
    │   │   └────────────┘  └────────────┘  └────────────┘        │  │
    │   └─────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

**Core Components:**

1. **Outcome Space** (``outcomes`` module)
   - **Issues**: Variables being negotiated (price, quantity, delivery date, etc.)
   - **Outcomes**: Specific assignments of values to issues
   - Supports discrete, continuous, and categorical issues

2. **Preferences** (``preferences`` module)
   - **UtilityFunction**: Maps outcomes to utility values
   - Built-in types: ``LinearAdditiveUtilityFunction``, ``MappingUtilityFunction``, ``NonLinearAggregationUtilityFunction``, and more
   - Supports probabilistic and dynamic utility functions

3. **Negotiators** (``negotiators``, ``sao`` modules)
   - Implement negotiation strategies
   - Built-in: ``AspirationNegotiator``, ``TitForTatNegotiator``, ``NaiveTitForTatNegotiator``, ``BoulwareTBNegotiator``, etc.
   - Easy to create custom negotiators

4. **Mechanisms** (``mechanisms``, ``sao`` modules)
   - Implement negotiation protocols
   - ``SAOMechanism``: Stacked Alternating Offers (most common)
   - Also: Single-text protocols, auction mechanisms, etc.

**For Situated Negotiations (World Simulations):**

5. **Worlds** (``situated`` module)
   - Simulate environments where agents negotiate
   - Agents can run multiple concurrent negotiations
   - Example: Supply chain simulations (SCML)

6. **Controllers** (``sao.controllers`` module)
   - Coordinate multiple negotiators
   - Useful when negotiations are interdependent

Key Features
------------

- **Multiple Protocols**: SAO (Alternating Offers), Single-Text, Auctions, and custom protocols
- **Rich Utility Functions**: Linear, nonlinear, constraint-based, probabilistic, dynamic
- **Bilateral & Multilateral**: Support for 2+ party negotiations
- **Concurrent Negotiations**: Agents can participate in multiple negotiations simultaneously
- **World Simulations**: Build complex multi-agent simulations with situated negotiations
- **Genius Integration**: Run Java-based Genius agents via the built-in bridge
- **Visualization**: Built-in plotting for negotiation analysis
- **Extensible**: Easy to add new protocols, negotiators, and utility functions

Creating Custom Negotiators
---------------------------

**Minimal SAO negotiator:**

.. code-block:: python

    from negmas.sao import SAONegotiator, SAOResponse, ResponseType


    class MyNegotiator(SAONegotiator):
        def __call__(self, state, offer=None):
            # Accept any offer with utility > 0.8
            if offer is not None and self.ufun(offer) > 0.8:
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            # Otherwise, propose a random outcome
            return SAOResponse(ResponseType.REJECT_OFFER, self.nmi.random_outcome())

**Using the negotiator:**

.. code-block:: python

    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(MyNegotiator(name="custom"), ufun=my_ufun)
    session.add(AspirationNegotiator(name="opponent"), ufun=opponent_ufun)
    session.run()

Creating Custom Protocols
-------------------------

.. code-block:: python

    from negmas import Mechanism, MechanismStepResult


    class MyProtocol(Mechanism):
        def __call__(self, state, action=None):
            # Implement one round of your protocol
            # Return MechanismStepResult with updated state
            ...
            return MechanismStepResult(state=state)

Running World Simulations
-------------------------

For complex scenarios with multiple agents and concurrent negotiations:

.. code-block:: python

    from negmas.situated import World, Agent


    class MyAgent(Agent):
        def step(self):
            # Called each simulation step
            # Request negotiations, respond to events, etc.
            pass


    # See SCML package for a complete example
    # pip install scml

Citation
--------

If you use NegMAS in your research, please cite:

.. code-block:: bibtex

    @inproceedings{mohammad2021negmas,
      title={NegMAS: A Platform for Automated Negotiations},
      author={Mohammad, Yasser and Nakadai, Shinji and Greenwald, Amy},
      booktitle={PRIMA 2020: Principles and Practice of Multi-Agent Systems},
      pages={343--351},
      year={2021},
      publisher={Springer},
      doi={10.1007/978-3-030-69322-0_23}
    }

**Reference:**

    Mohammad, Y., Nakadai, S., Greenwald, A. (2021). NegMAS: A Platform for Automated Negotiations.
    In: *PRIMA 2020*. LNCS, vol 12568. Springer. https://doi.org/10.1007/978-3-030-69322-0_23

The NegMAS Ecosystem
--------------------

NegMAS is the core of a broader ecosystem for automated negotiation research:

**Competition Frameworks**

- `anl <https://github.com/autoneg/anl>`_ - Automated Negotiation League (ANAC negotiation track)
- `scml <https://github.com/yasserfarouk/scml>`_ - Supply Chain Management League

**Agent Repositories**

- `anl-agents <https://github.com/autoneg/anl-agents>`_ - ANL competition agents
- `scml-agents <https://github.com/yasserfarouk/scml-agents>`_ - SCML competition agents

**Bridges & Extensions**

- `negmas-geniusweb-bridge <https://github.com/autoneg/negmas-geniusweb-bridge>`_ - Run GeniusWeb agents
- `negmas-llm <https://github.com/autoneg/negmas-llm>`_ - LLM-powered negotiation agents
- `negmas-elicit <https://github.com/autoneg/negmas-elicit>`_ - Preference elicitation during negotiation
- `geniusbridge <https://github.com/yasserfarouk/geniusbridge>`_ - Java Genius bridge

**Visualization & Tools**

- `negmas-app <https://github.com/autoneg/negmas-app>`_ - Applications and interfaces for NegMAS
- `scml-vis <https://github.com/yasserfarouk/scml-vis>`_ - SCML visualization
- `jnegmas <https://github.com/yasserfarouk/jnegmas>`_ - Java interface

More Resources
--------------

- **Tutorials**: https://negmas.readthedocs.io/en/latest/tutorials.html
- **API Reference**: https://negmas.readthedocs.io/en/latest/api.html
- **YouTube Playlist**: https://www.youtube.com/playlist?list=PLqvs51K2Mb8IJe5Yz5jmYrRAwvIpGU2nF
- **Publications**: https://negmas.readthedocs.io/en/latest/publications.html

Papers Using NegMAS
-------------------

Selected papers (see `full list <https://negmas.readthedocs.io/en/latest/publications.html>`_):

**Competition & Benchmarks**

- Aydoğan et al. (2020). `Challenges and Main Results of ANAC 2019 <https://doi.org/10.1007/978-3-030-66412-1_23>`_. EUMAS/AT. *Cited by 51*
- Mohammad et al. (2019). `Supply Chain Management World <https://doi.org/10.1007/978-3-030-33792-6_10>`_. PRIMA. *Cited by 38*

**Negotiation Strategies**

- Sengupta et al. (2021). `RL-Based Negotiating Agent Framework <https://arxiv.org/abs/2102.03588>`_. arXiv. *Cited by 48*
- Higa et al. (2023). `Reward-based Negotiating Agent Strategies <https://ojs.aaai.org/index.php/AAAI/article/view/26831>`_. AAAI. *Cited by 16*

**Preference Elicitation**

- Mohammad, Y., Nakadai, S. (2019). `Optimal Value of Information Based Elicitation <https://dl.acm.org/doi/10.5555/3306127.3331698>`_. AAMAS.
- Mohammad, Y., Nakadai, S. (2018). `FastVOI: Efficient Utility Elicitation <https://doi.org/10.1007/978-3-030-03098-8_34>`_. PRIMA.

**Applications**

- Inotsume et al. (2020). `Path Negotiation for Multirobot Vehicles <https://doi.org/10.1109/IROS45743.2020.9340819>`_. IROS. *Cited by 17*

*Last updated: January 2026*

Contributing
------------

Contributions are welcome! Please see the `contributing guide <https://negmas.readthedocs.io/en/latest/contributing.html>`_.

License
-------

NegMAS is released under the BSD 3-Clause License.

AI Assistance Disclosure
------------------------

This project uses AI assistance for specific, limited tasks while remaining predominantly human-developed:

- **Publications list**: AI assisted in compiling and formatting the publications list
- **Documentation polishing**: AI assisted in proofreading and improving documentation clarity
- **gb.components.genius module**: AI assisted in reimplementing Genius BOA components in NegMAS
- **Registry feature**: AI assisted in developing the negotiator/mechanism registry system
- **Some tests**: AI assisted in writing tests, particularly for new features like the registry

All AI-assisted contributions are reviewed and approved by human maintainers. The core architecture,
algorithms, and research direction of NegMAS are human-driven and will remain so.

Acknowledgements
----------------

NegMAS was developed at the NEC-AIST collaborative laboratory. It uses scenarios from
ANAC 2010-2018 competitions obtained from the `Genius Platform <http://ii.tudelft.nl/genius>`_.
