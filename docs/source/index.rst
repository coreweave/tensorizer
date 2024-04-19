CoreWeave Tensorizer
====================

CoreWeave Tensorizer is a PyTorch module, model, and tensor serializer
and deserializer that makes it possible to load models extremely quickly
from HTTP/HTTPS and S3 endpoints.
It enables both faster network load times,
as well as faster load times from local disk volumes.

This site hosts the API documentation for writing code with tensorizer.

External Links
--------------

An `overview of tensorizer`_ is available in CoreWeave's documentation.

The source code for tensorizer is hosted on `GitHub`_, along with examples,
quickstart code, and general usage patterns.

The package is available to install through `PyPI`_ using
``pip install tensorizer``.

View the `changelog here`_ for differences between tensorizer releases.

.. _overview of tensorizer: https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer
.. _GitHub: https://github.com/coreweave/tensorizer
.. _PyPI: https://pypi.org/project/tensorizer
.. _changelog here: https://github.com/coreweave/tensorizer/blob/main/CHANGELOG.md

Installation
------------


tensorizer can be installed through PyPI:

.. code-block:: bash

   pip install tensorizer

To use it in code, then ``import tensorizer``.

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tensorizer
   stream_io
   utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. * :ref:`search`
