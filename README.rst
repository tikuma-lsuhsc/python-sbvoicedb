`sbvoicedb`: Saarbrueken Voice Database Reader module
======================================================

|pypi| |status| |pyver| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/sbvoicedb
  :alt: PyPI
.. |status| image:: https://img.shields.io/pypi/status/sbvoicedb
  :alt: PyPI - Status
.. |pyver| image:: https://img.shields.io/pypi/pyversions/sbvoicedb
  :alt: PyPI - Python Version
.. |license| image:: https://img.shields.io/github/license/tikuma-lsuhsc/python-sbvoicedb
  :alt: GitHub

**Warning**
This Python package is not yet published and still under development.

This Python module provides functions to retrieve data and information easily from 
Saarbucken Voice Database: http://www.stimmdatenbank.coli.uni-saarland.de/

Install
-------

.. code-block:: bash

  pip install sbvoicedb


Examples
--------

.. code-block:: python

  from sbvoicedb import sbvoicedb

  # to create a database instance 
  sbvoicedb = sbvoicedb.SbVoiceDb('<path to the root directory of the extracted database>')
  # - if no downloaded database data found, it'll automatically download the database (not files)

  # to query the recording session entries which are pathological, female, between 50-69 yrs old
  df = sbvoicedb.query(T='p', G='w', A=[50,70])

  # to get a dataframe of WAV files and start and ending audio sample indices of 
  # all normal-pitch /a/ segments
  df = sbvoicedb.get_files('a_n')

  # to get the audio data of /a/ vowel at normal pitch from the recording session 2091
  fs, x = sbvoicedb.get_data(2091, 'a_n')

  # to iterate over 'a_n' acoustic data of male participants along with aux with age and pathologies
  for id, fs, x, auxdata in sbvoicedb.iter_data('a_n',
                                      auxdata_fields=["A","Pathologies"],
                                      G="m"):
    # run the acoustic data through your analysis function, get measurements
    params = my_analysis_function(fs, x)

    # log the measurements along with aux data
    my_logger.log_outcome(id, *auxdata, *params)

