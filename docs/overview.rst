
Overview
========

negmas was designed mainly to support multi-strand multilateral
multi-issue negotiations with complex utility functions. This section
gives an introduction to the main concepts of the public interface.

In order to use the library you will need to import it as follows
(assuming that you followed the instructions in the installation section
of this document):

.. code:: ipython3

    # This is to make the results reproducible if you are using the Jupyter notebook version.
    from random import seed
    seed(0)
    import negmas

Organization
------------

The package is organized into a set of modules the combine together
related functionality. In general there are *base* modules that
implement the most general abstractions and concepts and then
*specialized* modules that implement the computational structures needed
for a specific application or research domain:

-  **Base Modules** These are the most general modules and all other
   *specialized* modules use the computational resources defined here.
   The base modules provided in this version are:

   1. **outcomes** This module represents issues, outcome and responses
      and provides basic functions and methods to operator with and on
      them.
   2. **utilities** This modules represents the base type of all
      utilities and different widely used utility function types
      including linear and nonlinear utilities and constraint-based
      utilities.
   3. **negotiators** This module represents basic negotiation agent
      implementation and provide basic interfaces to be overriden
      (implemented) by higher specialized modules
   4. **mechanisms** This module represents the most basic conceptual
      view of a negotiation protocol supporting both mediate and
      unmediated mechanisms. The term ``mechanism`` was used instead of
      the more common ``protocol`` to stress the fact that this
      mechanism need not be a standard negotiation protocol. For example
      auction mechanisms (like second-price auctions) can easily be
      implemented in this package.
   5. **opponent\_models** This module provides the basic interface for
      all opponent models.
   6. **situated** This module implements world simulations within which
      agents with intrinsic utilty functions can engage in simulataneous
      connected situated negotiations. It is the most important module
      for the goals of this library.
   7. **Helper Modules** These modules provide basic activities that is
      not directly related to the negotiation but that are relied upon
      by different base modules. The end user is not expected to
      interact directly with these modules.

      -  **common** Provides common interfaces that are used by all
         other modules.
      -  **generics** Provides a set of types and interfaces to increase
         the representation flexibility of different base modules.
      -  **helpers** Various helper functions and classes used
         throughout the library including mixins for logging.

-  **App Modules** This namespace provides the modules needed to run
   different *apps* that represent worlds within which negotiations take
   place.

   1. **scml** The Supply Chain Management App as defined for the SCM
      leage of ANAC 2019 competition.

To simplify the use of this library, all classes and functions from all
base modules are aliased in the root package (except generics and
helpers). This is an example of importing just ``Outcome``

.. code:: ipython3

    from negmas import Outcome

It is possible to just import everything in the package using:

.. code:: ipython3

    from negmas import *

As usual you can just import everything in a separate namespace using:

.. code:: ipython3

    import negmas

Issues, Outcomes, and Responses
-------------------------------

Negotiations are conducted between mutliple agents with the goal of
achieving an *agreement* (usually called a contract) on one of several
possible outcomes. Each *outcome* is in general an assignment to some
value to a set of issues. Each *issue* is a variable that can take one
of a -- probably infinit -- set of values from some predefined *domain*.

The classes and funtions supporting management of issues, outcomes and
responses are combined in the ``outcomes`` module.

To directly handle issues, outcomes and responses; you need to import
the ``outcomes`` modules. To simplify the code snippets in this
overview, we will just import everything in this module but you can of
course be selective

Issues
~~~~~~

Issues are represented in ``negmas`` using the issue class. An issue is
defined by a set of ``values`` and a ``name``. It can be created as
follows:

-  Using a set of strings:

.. code:: ipython3

    # an issue with randomly assigned name
    issue1 = Issue(values=['to be', 'not to be'])
    print(issue1)
    # an issue with given name:
    issue2 = Issue(values=['to be', 'not to be'], name='The Problem')
    print(issue2)


.. parsed-literal::

    2yW4Acq9GFz6Y1t9: ['to be', 'not to be']
    The Problem: ['to be', 'not to be']


-  Using a single integer to give an issue which takes any value from
   ``0`` to the given integer minus 1:

.. code:: ipython3

    issue3 = Issue(values=10, name='number of items')
    print(issue3)


.. parsed-literal::

    number of items: 10


-  Using a ``tuple`` with a lower and upper real-valued boundaries to
   give an issue with an infinite number of possibilities (all real
   numbers in between)

.. code:: ipython3

    issue4 = Issue(values=(0.0, 1.0), name='cost')
    print(issue4)


.. parsed-literal::

    cost: (0.0, 1.0)


The ``Issue`` class provides some useful functions. For example you can
find the ``cardinality`` of any issue using:

.. code:: ipython3

    [issue2.cardinality(), issue3.cardinality(), issue4.cardinality()]




.. parsed-literal::

    [2, 10, -1]



It is also possible to check the ``type`` of the issue and whether it is
discrete or continuous:

.. code:: ipython3

    [issue2.type, issue2.is_discrete(), issue2.is_continuous()]




.. parsed-literal::

    ['discrete', True, False]



It is possible to check the total cardinality for a set of issues (with
the usual ``-1`` encoding infinity):

.. code:: ipython3

    [Issue.n_outcomes([issue1, issue2, issue3, issue4]), # expected -1 because of issue4
     Issue.n_outcomes([issue1, issue2, issue3])] # expected 40 = 2 * 2 * 4




.. parsed-literal::

    [-1, 40]



You can pick random valid or invalid values for the issue:

.. code:: ipython3

    [
        [issue1.rand_valid(), issue1.rand_invalid()],
        [issue2.rand_valid(), issue2.rand_invalid()],
        [issue3.rand_valid(), issue3.rand_invalid()],
        [issue4.rand_valid(), issue4.rand_invalid()],
    ]




.. parsed-literal::

    [['not to be', '20190203-085645wL56nGisto be20190203-085645WgNZq6IT'],
     ['to be', '20190203-085645tgUe52Rvnot to be20190203-085645JgwBuNO6'],
     [3, 19],
     [0.47700977655271704, 1.86630992777164]]



You can also list all valid values for an issue using ``all``. Notice
that this property is a generator so it is memory efficient for the case
when an issue has many values.

.. code:: ipython3

    print(list(issue1.all))
    print(list(issue2.all))
    print(list(issue3.all))
    try:
        print(list(issue4.all))
    except ValueError as e:
        print(e)


.. parsed-literal::

    ['to be', 'not to be']
    ['to be', 'not to be']
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Cannot return all possibilities of a continuous issue


Outcomes
~~~~~~~~

Now that we know how to define issues, defining outcomes from a
negotiation is even simpler. An outcome can be any python ``mapping`` or
``iterable`` with a known length. That includes dictionaries, lists,
tuples among many other.

Here is how to define an outcome for the last three issues mentioned
above:

.. code:: ipython3

    valid_outcome = {'The Problem': 'to be', 'number of items': 5, 'cost': 0.15}
    invalid_outcome = {'The Problem': 'to be', 'number of items': 10, 'cost': 0.15}

Notice that the ``invalid_outcome`` is assigning a value of ``10`` to
the ``number of items`` issue which is not an acceptable value (``cost``
ranges between ``0`` and ``9``).

Because ``outcomes`` can be represented with many builtin collection
classes, the only common ancestor of all outcome objects is the
``object`` class. Nevertheless, the ``outcomes`` module provide a
type-alias ``Outcome`` that can be used for static type checking if
needed. The ``outcomes`` module also provides some functions for dealing
with ``outcome`` objects in relation to ``Issue``\ s. These are some
examples:

.. code:: ipython3

    [ outcome_is_valid(valid_outcome, [issue2, issue3, issue4])       # valid giving True
    , outcome_is_valid(invalid_outcome, [issue2, issue3, issue4])]    # invalid giving False




.. parsed-literal::

    [True, False]



It is not necessary for an outcome to assign a value for *all* issues to
be considered *valid*. For example the following outcomes are all valid
for the last three issues given above:

.. code:: ipython3

    [ outcome_is_valid({'The Problem': 'to be'}, [issue2, issue3, issue4])
    , outcome_is_valid({'The Problem': 'to be', 'number of items': 5}, [issue2, issue3, issue4])
    ]




.. parsed-literal::

    [True, True]



It is also important for some applications to check if an outcome is
``complete`` in the sense that it assigns a *valid* value to every issue
in the given set of issues. This can be done using the
``outcome_is_complete`` function:

.. code:: ipython3

    [ outcome_is_complete(valid_outcome, [issue2, issue3, issue4])       # complete -> True
    , outcome_is_complete(invalid_outcome, [issue2, issue3, issue4])  # invalid -> incomplete -> False
    , outcome_is_complete({'The Problem': 'to be'}, [issue2, issue3, issue4])  # incomplete -> False  
    ]




.. parsed-literal::

    [True, False, False]



It is sometimes tedius to keep track of issue names in dictionaries. For
this reason, the library provides a type called *OutcomeType*.
Inheriting your dataclass from an OutcomeType will allow it to act both
as a dict and a normal dot accessible object:

.. code:: ipython3

    from dataclasses import dataclass
    @dataclass
    class MyOutcome(OutcomeType):
        problem: bool
        price: float
        quantity: int

Now you can use objects of MyOutcome as normal outcomes

.. code:: ipython3

    issues = [Issue(['to be', 'not to be'], name='problem')
              , Issue((0.0, 3.0), name='price')
              , Issue(5, name='quantity')]

.. code:: ipython3

    outcomes = Issue.sample(issues, n_outcomes = 5, astype=MyOutcome)
    for _ in outcomes:
        print(_)



.. parsed-literal::

    MyOutcome(problem='to be', price=1.0848388916904823, quantity=0)
    MyOutcome(problem='to be', price=1.8906644944040263, quantity=0)
    MyOutcome(problem='not to be', price=1.2102407956353904, quantity=0)
    MyOutcome(problem='not to be', price=2.957644296190988, quantity=1)
    MyOutcome(problem='not to be', price=2.847064181581488, quantity=0)


The *sample* function created objects of type MyOutcome that can be
accessed using either the dot notation or as a dict

.. code:: ipython3

    print(outcomes[0].price)
    print(outcomes[0]['price'])
    print(outcomes[0].get('price', None))


.. parsed-literal::

    1.0848388916904823
    1.0848388916904823
    1.0848388916904823


OutcomeType is intended to be used as a syntactic sugar around your
outcome objects but it provides almost no functionality above a dict.

Outcome Ranges and constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, it is important to represent not only a single outcome but a
range of outcomes. This can be represented using an ``OutcomeRange``.
Again, an outcome range can be almost any ``mapping`` or ``iterable`` in
python including dictionaries, lists, tuples, etc with the only
exception that the values stored in it can be not only ``int``, ``str``,
``float`` but also ``tuple``\ s of two of any of them representing a
range. This is easier shown:

.. code:: ipython3

    range1 = {'The Problem': ['to be', 'not to be'], 'number of items': 5, 'cost': (0.1, 0.2)}

``range1`` represents the following range of outcomes:

-  **The Problem**: accepts both ``to be`` and ``not to be``

-  **number of items**: accepts only the value ``5``

-  **cost**: accepts any real number between ``0.1`` and ``0.2`` up to
   representation error

It is easy to check whether a specific outcome is within a given range:

.. code:: ipython3

    outcome1 = {'The Problem': 'to be', 'number of items': 5, 'cost': 0.15}
    outcome2 = {'The Problem': 'to be', 'number of items': 10, 'cost': 0.15}
    [ outcome_in_range(outcome1, range1)        # True
    , outcome_in_range(outcome2, range1)        # False
    ]       




.. parsed-literal::

    [True, False]



In general outcome ranges constraint outcomes depending on the type of
the constraint:

-  **tuple** The outcome must fall within the range specified by the
   first and second elements. Only valid for values that can be compared
   using ``__lt__`` (e.g. int, float, str).
-  **single value** The outcome must equal this given value.
-  **list of values** The outcome must be within the list.
-  **list of tuples** The outcome must fall within one of the ranges
   specified by the tuples.

Responses
~~~~~~~~~

When negotiations are run, agents are allowed to respond to given offers
for the final contract. An offer is simple an outcome (either complete
or incomplete depending on the protocol but it is always valid). Agents
can then respond with one of the values defined by the ``Response``
enumeration in the ``outcomes`` module. Currently these are:

-  **ACCEPT\_OFFER** Accepts the offer.
-  **REJECT\_OFFER** Rejects the offer.
-  **END\_NEGOTIATION** This implies rejection of the offer and further
   more indicates that the agent is not willing to continue with the
   negotiation. The protocol is free to handle this situation. It may
   just end the negotiation with no agreement, may just remove the agent
   from the negotiation and keep it running with the remaining agents
   (if that makes sense) or just gives the agent a second chance by
   treating it as just a ``REJECT_OFFER`` case. In most case the first
   response (just end the negotiation) is expected.
-  **NO\_RESPONSE** Making no response at all. This is usually not
   allowed by negotiation protocols and will be considered a protocol
   violation in most cases. Nevertheless, negotiation protocols are free
   to handle this response when it arise in any way.

Utilities
---------

Agents engage in negotiations to maximize their utility. That is the
central dogma in negotiation research. ``negmas`` allows the user to
define their own utility functions based on a set of predefined base
classes that can be found in the ``utilities`` module.

Utility Values
~~~~~~~~~~~~~~

In most applications, utility values can be represented by real numbers.
Nevertheless, some applications need a more complicated representation.
For example, during utility elicitation (the process of learning about
the utility function of the human being represented by the agent) or
opponent modeling (the process of learning about the utility function of
an opponent), the need may arise to represent a probability distribution
over utilities.

``negmas`` allows all functions that receive a utility value to receive
a utility distribution. This is achieved through the use of two basic
type definitions:

-  ``UtilityDistribution`` That is a probability ``Distribution`` class
   capable of representing probabilistic variables having both
   continuous and discrete distributions and applying basic operations
   on them (addition, subtraction and multiplication). Currently we use
   ``scipy.stats`` for modeling these distributions but this is an
   implementation detail that should not be relied upon as it is likely
   that the probabilistic framework will be changed in the future to
   enhance the flexibility of the package and its integration with other
   probabilistic modeling packages (e.g. PyMC3).

-  ``UtilityValue`` This is the input and output type used whenever a
   utility value is to be represented in the whole package. It is
   defined as a union of a real value and a ``UtilityDistribution``
   (``Union[float, UtilityDistribution]``). This way, it is possible to
   pass utility distributions to most functions expecting (or returning)
   a utility value including utility functions.

This means that both of the following are valid utility values

.. code:: ipython3

    u1 = 1.0
    u2 = UtilityDistribution(dtype='norm')   # standard normal distribution
    print(u1)
    print(u2)


.. parsed-literal::

    1.0
    norm(loc:0.0, scale:1.0)


Utility Functions
~~~~~~~~~~~~~~~~~

Utility functions are entities that take an ``Outcome`` and return its
``UtilityValue``. There are many types of utility functions defined in
the literature. In this package, the base of all utiliy functions is the
``UtilityFunction`` class which is defined in the ``utilities`` module.
It behaves like a standard python ``Callable`` which can be called with
a single ``Outcome`` object (i.e. a dictionary, list, tuple etc
representing an outcome) and returns a ``UtilityValue``. This allows
utility functions to return a distribution instead of a single utility
value.

Utility functions in ``negmas`` have a helper ``property`` called
``type`` which returns the type of the utility function and a helper
function ``eu`` for returning the expected utility of a given outcome
which is guaranteed to return a real number (``float``) even if the
utiliy function itself is returning a utility distribution.

To implement a specific utility function, you need to override the
single ``__call__`` function provided in the ``UtilityFunction``
abstract interface. This is a simple example:

.. code:: ipython3

    class ConstUtilityFunction(UtilityFunction):
       def __call__(self, offer):
            try:
                return 3.0 * offer['cost'] 
            except KeyError:  # No value was given to the cost
                return None
        
       def xml(self):
            return '<ufun const=True value=3.0></ufun>'
    
    f = ConstUtilityFunction()
    [f({'The Problem': 'to be'}), f({'cost': 10})]




.. parsed-literal::

    [None, 30.0]



Utility functions can store internal state and use it to return
different values for the same outcome over time allowing for dynamic
change or evolution of them during negotiations. For example this
*silly* utility function responds to the mood of the user:

.. code:: ipython3

    class MoodyUtilityFunction(UtilityFunction):
        def __init__(self, mood='good'):
            super().__init__()
            self.mood = mood
            
        def __call__(self, offer):
            return float(offer['cost']) if self.mood == 'good'\
                                else 0.1 * offer['cost'] if self.mood == 'bad' \
                                else None 
        def set_mood(self, mood):
            self.mood = mood
        
        def xml(self):
            pass
    
    offer = {'cost': 10.0}
    
    f = MoodyUtilityFunction()
    # I am in a good mode now
    print(f'Utility in good mood of {offer} is {f(offer)}')
    f.set_mood('bad')
    print(f'Utility in bad mood of {offer} is {f(offer)}')
    f.set_mood('undecided')
    print(f'Utility in good mood of {offer} is {f(offer)}')


.. parsed-literal::

    Utility in good mood of {'cost': 10.0} is 10.0
    Utility in bad mood of {'cost': 10.0} is 1.0
    Utility in good mood of {'cost': 10.0} is None


Notice that (as the last example shows) utility functions can return
``None`` to indicate that the utility value cannot be inferred for this
outcome/offer.

The package provides a set of predefined utility functions representing
most widely used types. The following subsections describe them briefly:

Linear Aggregation Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``LinearAggregationUtilityFunction`` class represents a function
that linearly aggregate utilities assigned to issues in the given
outcome which can be defined mathematically as follows:

.. raw:: latex

   \begin{equation}
   U(o) = \sum_{i=0}^{\left|o\right|}{w_i\times g_i(o_i)}
   \end{equation}

where :math:`o` is an outcome, :math:`w` is a real-valued weight vector
and :math:`g` is a vector of functions each mapping one issue of the
outcome to some real-valued number (utility of this issue).

Notice that despite the name, this type of utiliy functions can
represent nonlinear relation between issue values and utility values.
The linearity is in how these possibly nonlinear mappings are being
combind to generate a utility value for the outcome.

For example, the following utility function represents the utility of
``buyer`` who wants low cost, many items, and prefers delivery:

.. code:: ipython3

    buyer_utility = LinearUtilityAggregationFunction({'price': lambda x: - x
                               , 'number of items': lambda x: 0.5 * x
                               , 'delivery': {'delivered': 1.0, 'not delivered': 0.0}})

Given this definition of utility, we can easily calculate the utility of
different options:

.. code:: ipython3

    print(buyer_utility({'price': 1.0, 'number of items': 3, 'delivery': 'not delivered'}))


.. parsed-literal::

    0.5


Now what happens if we offer to deliver the items:

.. code:: ipython3

    print(buyer_utility({'price': 1.0, 'number of items': 3, 'delivery': 'delivered'}))


.. parsed-literal::

    1.5


And if delivery was accompanied with an increase in price

.. code:: ipython3

    print(buyer_utility({'price': 1.8, 'number of items': 3, 'delivery': 'delivered'}))


.. parsed-literal::

    0.7


It is clear that this buyer will still accept that increase of price
from ``'1.0'`` to ``'1.8``' if it is accompanied with the delivery
option.

Nonlinear Aggregation Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A direct generalization of the linear agggregation utility functions is
provided by the ``NonLinearAggregationUtilityFunction`` which represents
the following function:

.. raw:: latex

   \begin{equation}
   U(o) = f\left(\left\{{g_i(o_i)}\right\}\right)
   \end{equation}

where :math:`g` is a vector of functions defined as before and :math:`f`
is a mapping from a vector of real-values to a single real value.

For example, a seller's utility can be defined as:

.. code:: ipython3

    seller_utility =NonLinearUtilityAggregationFunction({
                                 'price': lambda x: x
                               , 'number of items': lambda x: 0.5 * x
                               , 'delivery': {'delivered': 1.0, 'not delivered': 0.0}}
                       , f=lambda x: x['price']/x['number of items'] - 0.5 * x['delivery'])

This utility will go up with the ``price`` and down with the
``number of items`` as expected but not in a linear fassion.

We can now evaluate different options similar to the case for the buyer:

.. code:: ipython3

    print(seller_utility({'price': 1.0, 'number of items': 3, 'delivery': 'not delivered'}))


.. parsed-literal::

    0.6666666666666666


.. code:: ipython3

    print(seller_utility({'price': 1.0, 'number of items': 3, 'delivery': 'delivered'}))


.. parsed-literal::

    0.16666666666666663


.. code:: ipython3

    print(seller_utility({'price': 1.8, 'number of items': 3, 'delivery': 'delivered'}))


.. parsed-literal::

    0.7




Hyper Rectangle Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many cases, it is not possible to define a utility mapping for every
issue independently. We provide the utility function
``HyperVolumeUtilityFunction`` to handle this situation by allowing for
representation of a set of nonlinear functions defined on arbitrary
hypervolumes of the space of outcomes.

The simplext example is a nonlinear-function that is defined over the
whole space but that nonlinearly combines several issues to calculate
the utility.

For example the previous ``NonLinearUtilityFunction`` for the ``seller``
can be represented as follows:

.. code:: ipython3

    seller_utility =HyperRectangleUtilityFunction(outcome_ranges= [None]
                               , utilities= [lambda x: 2.0*x['price']/x['number of items'] \
                                               - 0.5 * int(x['delivery'] == 'delivered')])
    print(seller_utility({'price': 1.0, 'number of items': 3, 'delivery': 'not delivered'}))
    print(seller_utility({'price': 1.0, 'number of items': 3, 'delivery': 'delivered'}))
    print(seller_utility({'price': 1.8, 'number of items': 3, 'delivery': 'delivered'}))


.. parsed-literal::

    0.6666666666666666
    0.16666666666666663
    0.7


This function recovered exactly the same values as the
``NonlinearUtilityFuction`` defined earlier by defining a single
hypervolume with the special value of ``None`` which applies the
function to the whole space and then defining a single nonlinear
function over the whole space to implement the required utiltiy mapping.

``HyperVolumeUtilityFunction`` was designed to a more complex situation
in which you can have multiple nonlinear functions defined over
different parts of the space of possible outcomes.

Here is an example in which we combine one global utility function and
two different local ones:

.. code:: ipython3

    f = HyperRectangleUtilityFunction(outcome_ranges=[None,
                                                {0: (1.0, 2.0), 1: (1.0, 2.0)},
                                                {0: (1.4, 2.0), 2: (2.0, 3.0)}]
                                           , utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]]
                                  , weights=[1,0.5,2.5])

There are three nonlinear functions in this example:

-  A global function which gives a utility of ``5.0`` everywhere
-  A local function which gives a utility of ``2.0`` to any outcome for
   which the first issue (issue ``0``) has a value between
   ``1.0 and``\ 2.0\ ``and the second issue (issue``\ 1\ ``) has a value between``\ 1.0\ ``and``\ 2.0\ ``which is represented as:``\ {0:
   (1.0, 2.0), 1: (1.0, 2.0)}\`\`
-  A second local function which gives a utility that depends on both
   the third and first issues ``(lambda x: 2 * x[2] + x[0]``) on the
   range ``{0: (1.4, 2.0), 2: (2.0, 3.0)}``.

You can also have weights for combining these functions linearly. The
default is just to sum all values from these functions to calculate the
final utility.

Here are some examples: \* An outcome that falls in the range of all
constraints:

.. code:: ipython3

    f([1.5, 1.5, 2.5])




.. parsed-literal::

    22.25



-  An outcome that falls in the range of the global and first local
   constraints only:

.. code:: ipython3

    f([1.5, 1.5, 1.0])




.. parsed-literal::

    6.0



-  An outcome that misses a value for some of the issues:

.. code:: ipython3

    print(f([1.5, 1.5]))


.. parsed-literal::

    None


Notice that in this case, no utility is calculated because we do not
know if the outcome falls within the range of the second local function
or not. To allow such cases, the initializer of
``HyperVolumeUtilityFunction`` allows you to ignore such cases:

.. code:: ipython3

    g = HyperRectangleUtilityFunction(outcome_ranges=[None,
                                                {0: (1.0, 2.0), 1: (1.0, 2.0)},
                                                {0: (1.4, 2.0), 2: (2.0, 3.0)}]
                                           , utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]]
                                   , ignore_failing_range_utilities=True
                                   , ignore_issues_not_in_input=True)
    print(g([1.5, 1.5]))


.. parsed-literal::

    7.0


Nonlinear Hyper Rectangle Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HyperVolumeUtilityFunction`` should be able to handle most complex
multi-issue utility evaluations but we provide a more general class
called ``NoneLinearHyperVolumeUtilityFunction`` which replaces the
simple weighted summation of local/global functions implemented in
``HyperVolumeUtilityFunction`` with a more general nonlinar mapping.

The relation between ``NoneLinearHyperVolumeUtilityFunction`` and
``HyperVolumeUtilityFunction`` is exactly the same as that between
``NonLinearUtilityAggregationFunction`` and
``LinearUtilityAggregationFunction``

Other utility function types
----------------------------

There are several other builtin utility function types in the utilities
module. Operations for utility function serialization to and from xml as
sell as normalization, finding pareto-frontier, generation of ufuns, etc
are also available. Please check the documentation of the utilities
module for more details

.. code:: ipython3

    from pprint import pprint
    pprint(negmas.utilities.__all__)


.. parsed-literal::

    ['UtilityDistribution',
     'UtilityValue',
     'UtilityFunction',
     'UtilityFunctionProxy',
     'ConstUFun',
     'LinDiscountedUFun',
     'ExpDiscountedUFun',
     'MappingUtilityFunction',
     'LinearUtilityAggregationFunction',
     'NonLinearUtilityAggregationFunction',
     'HyperRectangleUtilityFunction',
     'NonlinearHyperRectangleUtilityFunction',
     'ComplexWeightedUtilityFunction',
     'ComplexNonlinearUtilityFunction',
     'IPUtilityFunction',
     'pareto_frontier',
     'make_discounted_ufun',
     'normalize']


Negotiators
-----------

Negotiations are conducted by negotiators. We reserve the term ``Agent``
to more complex entities that can interact with a simulation or the real
world and spawn ``Negotiator`` objects as needed (see the situated
module documentation). The base ``Negotiator`` are implemented in the
``negotiators`` module. The design of this module tried to achieve
maximum flexibility by relying mostly on Mixins instead of inheretance
for adding functionality as will be described later.

Classes exposed in this module end with either ``Agent`` or ``Mixin``

.. code:: ipython3

    import negmas; negmas.negotiators.__all__




.. parsed-literal::

    ['Negotiator', 'NegotiatorProxy', 'AspirationMixin']



To build your negotiator, you need to inherit from one class ending with
``Negotiator``, implement its abstract functions and then add whatever
mixins you need implementing their abstract functions (if any) in turn.

Negotiators related to a specific negotiation mechanism are implemented
in that mechanism's module. For example, negotiators designed for the
Stacked Alternating Offers Mechanism are found in the ``sao`` module.

Agent (the base class of all negotiation agents)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class of all agents is ``Agent`` which has three abstract
methods that MUST be implemented by any agent you inherit from it:

.. code:: ipython3

    import negmas; negmas.sao.__all__




.. parsed-literal::

    ['SAOState',
     'SAOMechanism',
     'SAOMechanismProxy',
     'SAONegotiatorProxy',
     'SAONegotiator',
     'RandomNegotiator',
     'LimitedOutcomesNegotiator',
     'LimitedOutcomesAcceptor',
     'AspirationNegotiator',
     'ToughNegotiator',
     'OnlyBestNegotiator',
     'SimpleTitForTatNegotiator',
     'NiceNegotiator']



There is a speical type of negotiators called ``GeniusNegotiator``
implemented in the ``genius`` module that is capable of interacting with
negotiation sessions running in the genius platform (JVM). Please refer
to the documentation of this module for more information.

Mechanisms (Negotiations)
-------------------------

The base ``Mechanism`` class is implemented in the ``mechanisms``
module.

All protocols in the package inherit from the ``Protocol`` class and
provide the following basic functionalities:

-  checking ``capabilities`` of agents against ``requirements`` of the
   protocol
-  allowing agents to be join and leave the negotiation under the
   control of the underlying protocol. For example the protocol may
   allow or disallow agents from entering the negotiation once it
   started, it may allow or disallow modifying the issues being
   negotiated, may allow only a predefined maximum and minimum number of
   agents to engage in the negotiation. All of this is controlled
   through parameters to the protocol initializer.
-  provide the basic flow of protcols so that new protcols can be
   implemented by just overriding a single ``round()`` function.
-  provide basic callbacks that can be extended by new protocols.

   .. raw:: html

      <div class="alert alert-block alert-warning">

    Protocols must extend any callback (i.e. call the ``super()``
   version) instead of overriding them as they may do some actions to
   ensure correct processing.

   .. raw:: html

      </div>

The simplest way to use a protocol is to just run one of the already
provided protocols. This is an example of a full negotiation session:

.. code:: ipython3

    p = SAOMechanism(outcomes = 6, n_steps = 10)
    p.add(LimitedOutcomesNegotiator(name='seller', acceptable_outcomes=[(2,), (3,), (5,)]
                                       , outcomes=p.outcomes))
    p.add(LimitedOutcomesNegotiator(name='buyer', acceptable_outcomes=[(1,), (4,), (3,)]
                                       , outcomes=p.outcomes))
    state = p.run()
    p.state.agreement




.. parsed-literal::

    (3,)



You can create a new protocol by overriding a single function in the
``Protocol`` class. This is for example the full code of the
``AlternatingOffersProtcol`` for the multi-issue case.

.. code:: ipython3

    class MyAlternatingOffersProtocol(Mechanism):
        def __init__(self, issues=None, outcomes=None, n_steps=None, time_limit=None):
            super().__init__(issues=issues, outcomes=outcomes, n_steps=n_steps, time_limit=time_limit)
            self.current_offer = None
            self.current_offerer = None
            self.n_accepting_agents = 0
    
        def step_(self):
            end_negotiation = False
            n_agents = len(self.negotiators)
            accepted = False
            for i, agent in enumerate(self.negotiators):
                if self.current_offer is None:
                    response = ResponseType.NO_RESPONSE
                else:
                    response = agent.respond(state=self.state, offer=self.current_offer)
                if response == ResponseType.END_NEGOTIATION:
                    end_negotiation = True
                    self.current_offer = None
                else:
                    if response != ResponseType.ACCEPT_OFFER:
                        self.current_offer = agent.propose(state=self.state)
                        self.current_offerer = i
                        self.n_accepting_agents = 1
                    else:
                        self.n_accepting_agents += 1
                        if self.n_accepting_agents == n_agents:
                            accepted = True
                            break
                if end_negotiation:
                    break
            return MechanismRoundResult(broken=response == ResponseType.END_NEGOTIATION
                                        , timedout=False
                                        , agreement=self.current_offer if accepted else None)


Agents can now engage in interactions with this protocol as easily as
any built-in protocol:

.. code:: ipython3

    p = MyAlternatingOffersProtocol(outcomes = 6, n_steps = 10)
    p.add(LimitedOutcomesNegotiator(name='seller', acceptable_outcomes=[(2,), (3,), (5,)]
                                       , outcomes=p.outcomes))
    p.add(LimitedOutcomesNegotiator(name='buyer', acceptable_outcomes=[(1,), (4,), (3,)]
                                       , outcomes=p.outcomes))
    state = p.run()
    p.state.agreement




.. parsed-literal::

    (3,)



The negotiation ran with the expected results

