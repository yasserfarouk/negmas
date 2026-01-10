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

In Unix/macOS machines::

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

.. _Github repo: https://github.com/yasserfarouk/negmas
.. _tarball: https://github.com/yasserfarouk/negmas/tarball/master
