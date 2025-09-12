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


This Python module provides capability to download and organize Saarbrücker Stimmdatenbank 
(Saarbrücken Voice Database, https://stimmdb.coli.uni-saarland.de/) with SQLAlchemy (sqlalchemy.org).

Features
--------

* Auto-download the database file at https://stimmdb.coli.uni-saarland.de
* Auto-download the associated datasets from Zonedo: https://zenodo.org/records/16874898
* Supports incremental, on-demand download per-pathology
* Stores database information as a local SQLite3 file
* Database and datasets are accessed via SQLAlchemy ORM (Object Relational Mapper)
  classes for ease of use
* Acoustic and EGG signals can be retrieved as NumPy arrays directly
* Supports filters to specify study conditions on pathologies, speaker's gender and age, 
  recording types, etc.
* Fixes known errors in the dataset (i.e., corrupted files and swapping of acoustic/EGG data)

Install
-------

.. code-block:: bash

  pip install sbvoicedb

If you prefer manually downloading the full dataset from Zonedo (`data.zip`, the 
full dataset, 17.9 GB) you may download the file first and unzip the content 
to a directory. Make sure that the zip file's internal structure is preserved.
If you're placing your downloaded database in ``my_svd`` folder, its directory
structure should appear like this:

.. code-block::

  .../my_svd/
  └── data/
      ├── 1/
      │   ├── sentnces
      │   │   ├── 1-phrase.nsp
      │   │   └── 1-phrase-egg.egg
      │   └── vowels
      │       ├── 1-a_h.nsp
      │       ├── 1-a_h.nsp
      │       ⋮
      │       └── 1-u_n-egg.egg
      ├── 2/
      │   │   ├── 2-phrase.nsp
      │   │   └── 2-phrase-egg.egg
      │   └── vowels
      │       ├── 2-a_h.nsp
      │       ├── 2-a_h.nsp
      │       ⋮
      │       └── 2-u_n-egg.egg
      ⋮

Examples
--------

.. code-block:: python

  from sbvoicedb import SbVoiceDb

  dbpath = '<path to the root directory of the extracted database>'

  # to create a database instance 
  db = SbVoiceDb(dbpath)
  # - if no downloaded database data found, it'll automatically download the database (not files)

This creates a new database instance. If ``dbpath`` does not contain the SQLite
database file, ``sbvoice.db``, it gets populated from the downloaded CSV file.

.. note::

  The ``sbvoice.db`` database file can be viewed using any SQLite database viewer
  such as DB Browser for SQLite (https://sqlitebrowser.org/)

If any portion of the dataset is already available in ``data`` subdirectory, it 
further populates the recordings table. These database population processes are
visualized with progress bars in the console.

By default, no dataset will be downloaded at this point. You can check how much
of the datasets are available by

.. code-block:: python

  print(f"{db.number_of_sessions_downloaded}/{db.number_of_all_sessions}")

The ``db.number_of_all_sessions`` property should always return 2043.

There are 4 tables to the SQLite database: ``pathologies``, ``speakers``, 
``recording_sessions``, and ``recordings``. The contents of these tables can be 
accessed by 

.. code-block:: python

  db.get_pathology_count()
  db.get_speaker_count()
  db.get_session_count()
  db.get_recording_count()

  db.iter_pathologies()
  db.iter_speakers()
  db.iter_sessions()
  db.iter_recordings()

Your study may not require all the recordings. In such case, you can set filters
on each table when creating the database object. For example, the following creates
a subset of the database which only consists of recordings of sustained /a/ or /i/
at normal pitch, uttered by women of age between 50 and 70 with normal voice or 
with a diagnosis of Laryngitis:

.. code-block:: python

  from sbvoicedb import Pathology, Speaker, RecordingSession, Recording, sql_expr

  db_laryngitis = database.SbVoiceDb(
      dbdir,
      pathology_filter=Pathology.name == "Laryngitis",
      include_healthy=True,
      speaker_filter=Speaker.gender == "w",
      session_filter=RecordingSession.speaker_age.between(50, 70),
      recording_filter=Recording.utterance.in_(("a_n", "i_n")),
  )
  print(f"number of pathologies found: {db_laryngitis.get_pathology_count()}")
  print(f"number of recording sessions found: {db_laryngitis.get_session_count()}")
  print(f"number of unique speakers: {db_laryngitis.get_speaker_count()}")
  print(f"number of recordings: {db_laryngitis.get_recording_count()}")

.. code-block::

  number of pathologies found: 1
  number of recording sessions found: 45
  number of unique speakers: 44
  number of recordings: 90

You can iterate over the rows of any of the tables:

.. code-block:: python

  # iterate over included pathologies
  for patho in db_laryngitis.iter_pathologies():
    print(f'{patho.id)}: {patho.name} ({patho.downloaded})'

  # iterate over included speakers
  for speaker in db_laryngitis.iter_speakers():
    print(f'{speaker.id)}: {speaker.gender}'

  # iterate over included recording sessions
  for session in db_laryngitis.iter_sessions():
    print(f'{session.id)}: speaker_id={session.speaker_id}, speaker_age={session.speaker_age}, speaker_health={session.type}'

  # iterate over included recordings
  for rec in db_laryngitis.iter_recordings():
    print(f'{rec.id)}: session_id={rec.session_id}, utterance={rec.utterance}, nspfile={rec.nspfile}, eggfile={rec.eggfile}'

To retrieve the acoustic and egg data, use ``Recording.nspdata`` and ``Recording.eggdata``:

.. code-block:: python

  import numpy as np
  from matplotlib import pyplot as plt

  rec = next(db_laryngitis.iter_recordings())

  t = np.arange(rec.length)/rec.rate

  fig, axes = plt.subplots(2, 1, sharex=True)
  axes[0].plot(t,rec.nspdata)
  axes[0].set_ylabel('acoustic data')
  axes[1].plot(t,rec.eggdata)
  axes[1].set_ylabel('EGG data')
  axes[1].set_xlabel('time (s)')
  plt.tight_layout()
  plt.show()

.. Data Modifications
.. ------------------

.. SVD Dataset has several 

.. 713-i_n/-iau - corrupted NSP/EGG files

.. 980-iau.wav/980-iau-egg.wav - acoustic and EGG waveforms were flipflopped at n = 583414

.. 980-phrase.wav/980-phrase-egg.wav - acoustic & Egg files were named backwards

.. 1697-iau.wav/1697-iau-egg.wav - acoustic & Egg files were named backwards
.. 1697-phrase.wav/1697-phrase-egg.wav - acoustic & Egg files were named backwards

.. 139-xxx, 141-xxx - acoustic & egg swapped

.. Downloaded vowel files
.. 1573 normal vowels (i-a-u) not cut correctly, recreated