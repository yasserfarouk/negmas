negmas-app GUI
==============

The ``negmas-app`` provides a graphical user interface for running negotiations and tournaments
interactively without writing code. It's ideal for:

- **Education**: Teaching negotiation concepts with visual feedback
- **Rapid Prototyping**: Testing negotiation scenarios quickly
- **Experimentation**: Running tournaments and comparing strategies visually
- **Demonstrations**: Showing negotiation dynamics in real-time

Installation
------------

The GUI is distributed as a separate package to keep NegMAS lightweight:

.. code-block:: bash

    pip install negmas-app

This will install the GUI application along with all necessary dependencies.

Features
--------

The negmas-app GUI provides:

- **Interactive Negotiation Setup**: Configure negotiators, preferences, and mechanisms through a visual interface
- **Real-time Visualization**: Watch negotiations unfold step-by-step with live utility plots
- **Tournament Management**: Set up and run tournaments with multiple negotiators
- **Results Analysis**: View statistics, agreement outcomes, and performance metrics
- **Strategy Comparison**: Compare different negotiation strategies side-by-side
- **Export Capabilities**: Save negotiations and tournament results for further analysis

Getting Started
---------------

After installation, launch the application:

.. code-block:: bash

    negmas-app

Or from Python:

.. code-block:: python

    from negmas_app import run_app

    run_app()

Basic Workflow
--------------

1. **Create a Negotiation Session**

   - Select the negotiation protocol (SAO, Stacked Alternating Offers, etc.)
   - Choose or create negotiators from the available strategies
   - Define the negotiation domain (issues, outcomes, preferences)

2. **Configure Preferences**

   - Set utility functions for each negotiator
   - Define reservation values and constraints
   - Optionally add uncertainty or discounting

3. **Run the Negotiation**

   - Start the negotiation and watch it progress
   - Pause/resume at any time to examine the current state
   - Step through offers and responses one at a time

4. **Analyze Results**

   - View final agreements and utilities
   - Examine negotiation statistics (rounds, time, concessions)
   - Export results to JSON, CSV, or other formats

Running Tournaments
-------------------

The GUI also supports running tournaments to compare multiple negotiators:

1. Select negotiators to compete
2. Configure tournament settings (number of rounds, domains, scoring)
3. Run the tournament
4. View rankings, statistics, and detailed results

Integration with NegMAS Code
-----------------------------

The GUI is fully compatible with NegMAS code. You can:

- **Import Custom Negotiators**: Load negotiators defined in your Python code
- **Use Custom Domains**: Import outcome spaces and utility functions from files
- **Export for Code**: Save GUI-configured scenarios as Python code
- **Hybrid Workflow**: Design in GUI, refine in code, or vice versa

Example Use Cases
-----------------

**Teaching Negotiation Concepts**

Use the GUI to demonstrate negotiation strategies in a classroom setting:

- Show how different strategies (Boulware, Conceder, Tit-for-Tat) behave
- Visualize the effect of time pressure on agreements
- Demonstrate Nash bargaining solution vs. actual outcomes

**Testing New Strategies**

Before implementing complex negotiators in code:

- Test high-level strategy concepts with built-in negotiators
- Experiment with different opponent models
- Identify promising approaches worth full implementation

**Competition Preparation**

Prepare for ANAC or other negotiation competitions:

- Test your agents against known strategies
- Run practice tournaments with competition-like settings
- Analyze weaknesses and refine strategies

**Research Demonstrations**

Present research findings interactively:

- Show live examples of novel negotiation mechanisms
- Compare baseline vs. improved algorithms visually
- Create reproducible demonstration scenarios

Troubleshooting
---------------

**GUI doesn't start**

- Ensure you have a display environment (the GUI requires a graphical interface)
- Check that all dependencies are installed: ``pip install --upgrade negmas-app``
- On Linux, ensure you have required system libraries (PyQt5 dependencies)

**Negotiators not available**

- The GUI automatically discovers negotiators registered with NegMAS
- If your custom negotiators don't appear, ensure they're properly registered
- Check that the negotiator modules are in your Python path

**Performance issues with large tournaments**

- Reduce the number of concurrent negotiations
- Disable real-time visualization for large-scale tournaments
- Use the command-line tools for very large experiments (see :doc:`cli_negmas`)

Additional Resources
--------------------

- **GitHub Repository**: https://github.com/yasserfarouk/negmas-app
- **NegMAS CLI**: See :doc:`cli_negmas` for command-line alternatives
- **NegMAS Core Documentation**: Return to :doc:`overview` for core concepts
- **Tutorial Videos**: Check the repository for video demonstrations

.. note::
   The negmas-app GUI is actively developed. For the latest features and updates,
   see the `GitHub repository <https://github.com/yasserfarouk/negmas-app>`_.

See Also
--------

- :doc:`cli_negmas` - Command-line interface for automated workflows
- :doc:`cli_negotiate` - Quick negotiation runs from the terminal
- :doc:`negotiators` - Available negotiation strategies
- :doc:`negotiation_mechanisms` - Negotiation protocols and mechanisms
