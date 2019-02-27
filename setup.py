#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt', 'r') as f:
    requirements = [_.replace('\n', '') for _ in f.readlines() if not _.startswith('-')]

with open('negmas/__init__.py') as f:
    version = [_ for _ in f.readlines() if _.startswith('__version__')][0]
    version = version.split('\'')[-2]

setup_requirements = requirements

test_requirements = requirements
setup(
    author="Yasser Mohammad",
    author_email='yasserfarouk@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='A library for situated automated negotiations',
    #entry_points={
    #    'console_scripts': [
    #        'rungenius=scripts.rungenius.cli',
    #        'scml=scripts.scml.cli',
    #    ],
    #},
    scripts=['scripts/rungenius', 'scripts/scml', 'scripts/tournament'],
    install_requires=requirements,
    python_requires='>=3.6',
    license="GNU General Public License v2 (GPLv2)",
    long_description=readme,
    include_package_data=True,
    keywords='NegMAS negmas negotiate negotiation mas multi-agent simulation',
    name='negmas',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yasserfarouk/negmas',
    version='0.1.16',
    zip_safe=False,
)

