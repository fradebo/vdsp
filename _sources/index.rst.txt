.. Loop Universal Compiler documentation master file, created by
   sphinx-quickstart on Tue Aug  1 11:40:44 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Loop Universal Compiler's documentation!
===================================================
The basic idea is to generate tree-like graph states, starting from the intuition presented in `here <https://github.com/thenerdmix/vdsp/blob/main/proof/main.pdf>`_, where we also present a formal proof of why the euristic strategy we are using is working.

The implementation is based on three different modules:

* The **Loop** module manages the actual physical photonic circuit.

* The **Graph** module the creation and extraction of classical graphs and trees.

* The **Quantum Tree** module maps a classical graph to the corresponding quantum graph state and his physical implementation.

This project is the fruit of a two months internship in June and July 2023 promoted by the `VDSP <https://vds-physics.univie.ac.at/>`_ and the `Walther group <https://walther.quantum.at/>`_ in Vienna.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
