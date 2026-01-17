.. code:: ipython3

    # This is to make the results reproducible if you are using the Jupyter notebook version.
    from rich import print
    from random import seed
    import warnings

    warnings.filterwarnings("ignore")

    import plotly.io as pio
    pio.renderers.default = "png"  # Use PNG for smaller docs

    from IPython.display import display, HTML

    display(HTML("<style>.container { width:95% !important; }</style>"))
    import random

    random.seed(203)
    import numpy as np

    np.random.seed(345)
    from rich import print
    from pathlib import Path



.. raw:: html

    <style>.container { width:95% !important; }</style>


Getting Started
===============

Running a negotiation
---------------------

NegMAS has several built-in negotiation ``Mechanisms``, negotiation
agents (``Negotiators``), and ``UtilityFunctions``. You can use these to
run negotiations as follows.

Imagine a buyer and a seller negotiating over the price of a single
object. First, we make an issue “price” with 50 discrete values. Note
here, it is possible to create multiple issues, but we will not include
that here. If you are interested, see the `NegMAS
documentation <https://negmas.readthedocs.io/en/latest/tutorials/01.running_simple_negotiation.html>`__
for a tutorial.

.. code:: ipython3

    from negmas import make_issue, SAOMechanism, TimeBasedConcedingNegotiator
    from negmas.sao.negotiators import BoulwareTBNegotiator as Boulware
    from negmas.sao.negotiators import LinearTBNegotiator as Linear
    from negmas.preferences import LinearAdditiveUtilityFunction as UFun
    from negmas.preferences.value_fun import IdentityFun, AffineFun

    # create negotiation agenda (issues)
    issues = [make_issue(name="price", values=50)]

    # create the mechanism
    mechanism = SAOMechanism(issues=issues, n_steps=20)

The negotiation protocol in NegMAS is handled by a ``Mechanism`` object.
Here we instantiate a\ ``SAOMechanism`` which implements the `Stacked
Alternating Offers
Protocol <https://ii.tudelft.nl/~catholijn/publications/sites/default/files/Aydogan2017_Chapter_AlternatingOffersProtocolsForM.pdf>`__.
In this protocol, negotiators exchange offers until an offer is accepted
by all negotiators (in this case 2), a negotiators leaves the table
ending the negotiation or a time-out condition is met. In the example
above, we use a limit on the number of rounds of ``20`` (a step of a
mechanism is an executed round).

Next, we define the utilities of the seller and the buyer. The utility
function of the seller is defined by the ``IdentityFun`` which means
that the higher the price, the higher the utility function. The buyer’s
utility function is reversed. The last two lines make sure that utility
is scaled between 0 and 1.

.. code:: ipython3

    seller_utility = UFun(values=[IdentityFun()], outcome_space=mechanism.outcome_space)

    buyer_utility = UFun(
        values=[AffineFun(slope=-1)], outcome_space=mechanism.outcome_space
    )

    seller_utility = seller_utility.normalize()
    buyer_utility = buyer_utility.normalize()

Then we add two agents with a boulware strategy. The negotiation ends
with status overview. For example, you can see if the negotiation
timed-out, what agreement was found, and how long the negotiation took.
Moreover, we output the full negotiation history. For a more visual
representation, we can plot the session. This shows the bidding curve,
but also the proximity to e.g. the Nash point.

.. code:: ipython3

    # create and add agent A and B
    mechanism.add(Boulware(name="seller"), ufun=seller_utility)
    mechanism.add(Linear(name="buyer"), ufun=buyer_utility)

    # run the negotiation and show the results
    print(mechanism.run())



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">SAOState</span><span style="font-weight: bold">(</span>
        <span style="color: #808000; text-decoration-color: #808000">running</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #808000; text-decoration-color: #808000">waiting</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #808000; text-decoration-color: #808000">started</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>,
        <span style="color: #808000; text-decoration-color: #808000">step</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,
        <span style="color: #808000; text-decoration-color: #808000">time</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0023606250033481047</span>,
        <span style="color: #808000; text-decoration-color: #808000">relative_time</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8095238095238095</span>,
        <span style="color: #808000; text-decoration-color: #808000">broken</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #808000; text-decoration-color: #808000">timedout</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #808000; text-decoration-color: #808000">agreement</span>=<span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">35</span>,<span style="font-weight: bold">)</span>,
        <span style="color: #808000; text-decoration-color: #808000">results</span>=<span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
        <span style="color: #808000; text-decoration-color: #808000">n_negotiators</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #808000; text-decoration-color: #808000">has_error</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #808000; text-decoration-color: #808000">error_details</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>,
        <span style="color: #808000; text-decoration-color: #808000">erred_negotiator</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>,
        <span style="color: #808000; text-decoration-color: #808000">erred_agent</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>,
        <span style="color: #808000; text-decoration-color: #808000">threads</span>=<span style="font-weight: bold">{}</span>,
        <span style="color: #808000; text-decoration-color: #808000">last_thread</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>,
        <span style="color: #808000; text-decoration-color: #808000">current_offer</span>=<span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">35</span>,<span style="font-weight: bold">)</span>,
        <span style="color: #808000; text-decoration-color: #808000">current_proposer</span>=<span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>,
        <span style="color: #808000; text-decoration-color: #808000">current_proposer_agent</span>=<span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
        <span style="color: #808000; text-decoration-color: #808000">n_acceptances</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #808000; text-decoration-color: #808000">new_offers</span>=<span style="font-weight: bold">[]</span>,
        <span style="color: #808000; text-decoration-color: #808000">new_offerer_agents</span>=<span style="font-weight: bold">[</span><span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>, <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="font-weight: bold">]</span>,
        <span style="color: #808000; text-decoration-color: #808000">last_negotiator</span>=<span style="color: #008000; text-decoration-color: #008000">'buyer'</span>,
        <span style="color: #808000; text-decoration-color: #808000">current_data</span>=<span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
        <span style="color: #808000; text-decoration-color: #808000">new_data</span>=<span style="font-weight: bold">[]</span>
    <span style="font-weight: bold">)</span>
    </pre>



In this case, the negotiation ended with an agreement which is indicated
by the ``agreement`` field of the
`SAOState <https://negmas.readthedocs.io/en/latest/api/negmas.sao.SAOState.html#saostate>`__.

We can see a trace of the negotiation giving the step number, agent-id
and its offer using the ``extended_trace`` property of the mechanism
(session):

.. code:: ipython3

    # negotiation history
    print(mechanism.extended_trace)



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">[</span>
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">48</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">18</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">48</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">21</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">47</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">23</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">46</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">25</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">44</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">28</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">42</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">30</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">40</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e94fa547-51ce-4361-adf9-560b963632e6'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">37</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-eda0a0c4-ddcb-4dc4-9b36-dd07ff30c3b7'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">35</span>,<span style="font-weight: bold">))</span>
    <span style="font-weight: bold">]</span>
    </pre>



We can also plot the negotiation.

.. code:: ipython3

    mechanism.plot(mark_max_welfare_points=False)



.. image:: getting_started_files/getting_started_10_0.png


The most commonly used method for visualizing a negotiation is to plot
the utility of one negotiator on the x-axis and the utility of the other
in the y-axis, offers of different negotiators are then displayed in
different colors. The agreement is marked by a black star and important
points like the `Nash Bargaining
Solution <https://en.wikipedia.org/wiki/Cooperative_bargaining#Nash_bargaining_solution>`__,
`Kalai/Egaliterian Bargaining
Solution <https://en.wikipedia.org/wiki/Cooperative_bargaining#Egalitarian_bargaining_solution>`__,
`Kalai-Smorodonisky Bargaining
Solution <https://en.wikipedia.org/wiki/Cooperative_bargaining#Kalai–Smorodinsky_bargaining_solution>`__
and points with maximum welfare. This kind of figure is shown in the
left-hand side of the previous graph and can be produced by calling
``plot()`` on the mechanism. Because our single-issue negotiation is a
zero-sum game, all points have the same welfare of ``1.0`` and lie on a
straight line.

Another type of graph represents time (i.e. relative-time ranging from 0
to 1, real time, or step number) on the x-axis and represents the
utility of one negotiator’s offer for itself with a bold color on the
y-axis. The utility of the offers from this negotiators for all other
negotiators are also shown using a lighter line with no marks. This kind
of representation is useful in understanding clearly the change of each
negotiator’s behavior over time (in terms of its own and its partners’
utilities). In the previous graph, we can clearly see the difference
between the seller’s (upper right) and buyer’s (lower right) offering
strategies.

The ``plot`` function is very customizable and you can learn about all
its parameters
`here <https://negmas.readthedocs.io/en/latest/api/negmas.sao.SAOMechanism.html#negmas.sao.SAOMechanism.plot>`__
