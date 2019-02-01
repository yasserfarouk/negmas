=========
Use Cases
=========

This section provides a set of use cases showing the flexibility of the framework implemented by negmas and its
applicability to a wide variety of problems

.. toctree::
    :titlesonly:


Running a simple negotiation:
-----------------------------

Using the package for negotiation can be as simple as the following code snippet:

.. code-block:: python
    :linenos:

    from .sample.mechanisms import AlternatingOffersElicitingProtocol
    neg = AlternatingOffersElicitingProtocol(n_steps=100, elicitor_type=’optimal’, n_outcomes=10, r=0, costs=1
                , opponent_type=’limited_outcomes’, aspiration_type=’boulware’, max_aspiration=0.9)
    neg.run()


These three lines of code created a negotiation session with two negotiators. One agent had a list of predefined outcomes to
accept with some default probabilities while the other used optimal elicitation as described by
`Tim Baarslag <https://homepages.cwi.nl/~baarslag/>`_ in
`this 2015 paper <calendar.google.com/calendar/render?tab=mc#main_7>`_ in a more general scenario where the opponent
have different acceptanbe probabilities for different acceptable outcomes.

Yon can as easily bet two optimal eliciting negotiators against each other by a single modification:

.. code-block:: python
    :linenos:

    from .sample.mechanisms import AlternatingOffersElicitingProtocol
    neg = AlternatingOffersElicitingProtocol(n_steps=100, elicitor_type=’optimal’
        , n_outcomes=10, r=0, costs=1, opponent_type=’optimal’ , aspiration_type=’boulware’
        , max_aspiration=0.9)
    neg.run()

where we just modified the opponent_type parameter from ``'limited_outcomes'`` to ``'optimal'``

Developing a new agent
----------------------

Developing a novel neotiation agent slightly more difficult by is still accomplishable in few lines of code:

.. code-block:: python
    :linenos:

    from .negotiators import Negotiator
    class MyAwsomeAgent(Negotiator):
        def __init__(self):
            # initialize the parents
            Negotiator.__init__(self)
            MultiNegotiationsMixin.__init__(self)

        def respond_to(self, offer, info = None):
            # decide what to do when receiving an offer @ that negotiation
            pass

        def propose_for(self, n=1, info = None):
            # proposed the required number of proposals (or less) @ that negotiation
            pass

By just implementing `respond_to()` and `propose_for()`. This agent is now capable of engaging in multiple
concurrent negotiations. It can access all the negotiations it is involved in using ``self.negotiations``
and can enter and leave negotiations (if allowed by the protocol) using `enter()` and `leave()`.
See the documentation of `Negotiator` for a full description of available functionality out of the box.

Developing a negotiation protocol
---------------------------------

Developing a novel negotiation protocol is actually even simpler:

.. code-block:: python
    :linenos:

    from .mechanisms import Protocol

    class MyNovelProtocol(Protocol):
        def __init__(self, n_steps, n_outcomes):
            super().__init__(n_steps=n_steps, n_outcomes=n_outcomes)

        def round(self):
            # one step of the protocol
            pass

By implementing the single `round()` function, a new protocol is created. New negotiators can be added to the
negotiation using `add()` and removed using `remove()`. See the documentation for a full description of
`Protocol` available functionality out of the box.

Running two concurrent negotiations:
------------------------------------

something

Synchronized negotiations:
--------------------------

something

