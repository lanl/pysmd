Building the documentation
==========================

The documentation requires:

* Python 3.10 or newer;
* PySMD and its runtime dependencies;
* Sphinx 8.1 or newer.

From the project root, install PySMD with its documentation dependencies:

.. code-block:: console

   python -m pip install ".[docs]"

Then build the HTML documentation:

.. code-block:: console

   cd docs
   make html

The API reference is generated automatically. Generated API source files and
HTML output are not tracked by Git. The completed documentation is written to
``docs/build/html``.

The documentation sources are included in the PyPI source distribution. A
wheel contains the ``docs`` dependency metadata but does not contain this
source tree, so rebuilding the documentation from a release requires the
source distribution rather than an installed wheel.
