.. highlight:: shell

============
Installation
============


Stable release
--------------

To install negmas, run this command in your terminal:

.. code-block:: console

    $ pip install negmas

This is the preferred method to install negmas, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for negmas can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/yasserfarouk/negmas

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/yasserfarouk/negmas/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/yasserfarouk/negmas
.. _tarball: https://github.com/yasserfarouk/negmas/tarball/master


[Optional] Post Installation
----------------------------

After installation, two new commands will be added to your environment (hopefully it is a virtual environment): *scml*
and *rungenius*. To test your installation, run the following commands:

.. code-block:: console

    $ rungenius

This will start a service that allows NegMAS to use Genius_. After this process starts, you can run the tests normally
using:


.. code-block:: console

    $ python -m pytest --cov=negmas --pyargs negmas

Notice that this test will report coverage for test files as well. That is not ideal. To exclude such files from the
report you will need to use a .coveragerc file as described in Coverage_.

.. _Genius: http://ii.tudelft.nl/genius
.. _Coverage: https://pytest-cov.readthedocs.io/en/latest/config.html
