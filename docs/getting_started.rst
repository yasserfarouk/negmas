.. code:: ipython3

    from negmas import (
        make_issue,
        SAOMechanism,
       TimeBasedConcedingNegotiator,
    )
    from negmas.sao.negotiators import BoulwareTBNegotiator as Boulware
    from negmas.sao.negotiators import LinearTBNegotiator as Linear
    from negmas.preferences import LinearAdditiveUtilityFunction as UFun
    from negmas.preferences.value_fun import IdentityFun, AffineFun
    import matplotlib.pyplot as plt


    # create negotiation agenda (issues)
    issues = [
        make_issue(name="price", values=50),
    ]

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

    seller_utility = UFun(
        values=[IdentityFun()],
        outcome_space=mechanism.outcome_space,
    )

    buyer_utility = UFun(
        values=[AffineFun(slope=-1)],
        outcome_space=mechanism.outcome_space,
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
        <span style="color: #808000; text-decoration-color: #808000">time</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.003750874995603226</span>,
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
        <span style="color: #808000; text-decoration-color: #808000">current_proposer</span>=<span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>,
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
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">48</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">18</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">48</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">21</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">47</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">23</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">46</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">25</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">44</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">28</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">42</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">30</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">40</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>, <span style="color: #008000; text-decoration-color: #008000">'seller-e3614ef9-6d24-442a-9332-b15342ffa1e4'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">37</span>,<span style="font-weight: bold">))</span>,
        <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>, <span style="color: #008000; text-decoration-color: #008000">'buyer-953acaf8-4227-4d51-a652-8626f799fa76'</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">35</span>,<span style="font-weight: bold">))</span>
    <span style="font-weight: bold">]</span>
    </pre>



We can also plot the negotiation.

.. code:: ipython3

    mechanism.plot(mark_max_welfare_points=False)
    plt.show()



.. image:: getting_started_files/getting_started_8_0.png
