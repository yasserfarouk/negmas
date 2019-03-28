.. highlight:: shell

============
Installation
============

It is always a good idea to install your packages to a virtual environment. This is a reminder of how to create one
using the standard `venv` module in python 3 (inside a folder alled workspace)::

$ mkdir workspace; cd workspace
$ python -m venv venv
$ source venv/bin/activate

To check that you are in your newly created environment run the following commands depending on your OS.
In windows machines::

$ where python

In *nix/macos machines::

$ which python

In both cases you should find that python is running from venv/bin within your workspace folder.

After creating your venv, it is recommended to update pip::

$ pip install -U pip


Stable release
--------------

To install negmas, run this command in your terminal::

$ pip install negmas

This is the preferred method to install negmas, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Latest Commit
-------------

To installing the latest commit of negmas from github, you can run this command in your terminal::

    $ pip install git+https://github.com/yasserfarouk/negmas

Please note that this may not be a stable version. As always make sure that you are running the command within the
correct virtual environment.

From sources
------------

The sources for negmas can be downloaded from the `Github repo`_.

You can either clone the public repository::

$ git clone git://github.com/yasserfarouk/negmas

Or download the `tarball`_::

$ curl  -OL https://github.com/yasserfarouk/negmas/tarball/master

Once you have a copy of the source, you can install it with::

$ python setup.py install

or, better yet, if you do not have Poetry_, install it as explained in Poetry_install_, and then just run::

$ poetry install

We recommend the use of Poetry_.

.. _Github repo: https://github.com/yasserfarouk/negmas
.. _Poetry: https://poetry.eustace.io
.. _Poetry_install: https://poetry.eustace.io/docs/#installation
.. _tarball: https://github.com/yasserfarouk/negmas/tarball/master


[Optional] Post Installation
----------------------------

After installation, some new commands will be added to your environment (hopefully it is a virtual environment). Among
them there is a script called: *rungenius*.

To test your installation, run the following command (note that test_genius tests will be skipped)::

$ python -m pytest --pyargs negmas

If you want to test the Genius_  bridge, you need to download the :download:`Genius-NegMAS-Bridge<assets/genius-8.0.4-bridge.jar>`

Once you have the file save it to some path in your machine and run the following command (note that it will run in the
foreground until you press Ctrl-C to close it)::

$ negmas genius --path=path-to-genius-bridge.jar

This will start a service that allows NegMAS to use Genius_.

After this process starts, you can run the tests involving genius using::

$ python -m pytest --pyargs negmas/tests/test_genius


Notice that this test will report coverage for test files as well. That is not ideal. To exclude such files from the
report you will need to use a .coveragerc file as described in Coverage_.

.. _Genius: http://ii.tudelft.nl/genius
.. _Coverage: https://pytest-cov.readthedocs.io/en/latest/config.html
