negmas.types
============

Base types used to construct different entities in NegMAS. This module provides
the fundamental building blocks that other NegMAS classes inherit from.

Overview
--------

The types module defines three core abstractions:

- :class:`~negmas.types.NamedObject` - Base class for all named entities with unique IDs
- :class:`~negmas.types.Rational` - Base class for entities that have preferences (negotiators, agents)
- :class:`~negmas.types.Runnable` - Protocol for objects that can be stepped/run

Class Hierarchy
---------------

.. code-block:: text

    NamedObject
        └── Rational
                ├── Negotiator
                ├── Agent
                └── Controller

NamedObject
-----------

.. autoclass:: negmas.types.NamedObject
   :members:
   :show-inheritance:

Rational
--------

.. autoclass:: negmas.types.Rational
   :members:
   :show-inheritance:

Runnable
--------

.. autoclass:: negmas.types.Runnable
   :members:
   :show-inheritance:

WithPath
--------

.. autoclass:: negmas.types.WithPath
   :members:
   :show-inheritance:

Module Contents
---------------

.. automodule:: negmas.types
   :members:
   :undoc-members:
   :show-inheritance:
