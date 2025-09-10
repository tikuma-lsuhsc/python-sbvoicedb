"""Saarbrueken Voice Database Reader module"""

from __future__ import annotations

import logging
from datetime import datetime, date
import glob
from os import path, makedirs
import re
import unicodedata

from typing import Literal, List, Sequence, cast, Iterator, LiteralString, Any
import numpy as np

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    declarative_base,
    Session,
)

from sqlalchemy import (
    Engine,
    ForeignKey,
    Column,
    String,
    Date,
    create_engine,
    select,
    insert,
    update,
    Table,
    UniqueConstraint,
    Select,
    Result,
    text,
)

import sqlalchemy.sql.expression as sql_expr

from nspfile import read as nspread, NSPHeaderDict
import tqdm

from .download import download_data, download_database
from .utils import fix_incomplete_nsp, swap_nsp_egg

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Base = declarative_base()


class Setting(Base):
    """'settings' SQL table - general database key-value settings

    Currently only used to store `'healthy_downloaded'` key which has a value
    `'0'` if the healthy dataset has not been downloaded or
    `'1'` if the healthy dataset has been downloaded.
    """

    __tablename__ = "settings"
    __table_args__ = ()
    key: Mapped[str] = mapped_column(primary_key=True)
    """setting key"""
    value: Mapped[str]
    """setting value"""


recording_session_pathologies = Table(
    "recording_session_pathologies",
    Base.metadata,
    Column("session_id", ForeignKey("recording_sessions.id"), primary_key=True),
    Column("pathology_id", ForeignKey("pathologies.id"), primary_key=True),
)
"""'recording_session_pathologies' SQL table - many-to-many relationship table to 
specify pathologies to each recording session (for those not healthy).
"""


class Pathology(Base):
    """'pathologies' SQL table - list of all the voice pathologies given in Pathologien

    Each recording may be associated with one or more pathologies.

    Table Columns
    ^^^^^^^^^^^^^

    ==========  =====  ===================================
    name        type   desc
    ==========  =====  ===================================
    id          int    auto-assigned id
    name        str    pathology name in German
    downloaded  bool   True if dataset has been downloaded
    ==========  =====  ===================================

    Relationships
    ^^^^^^^^^^^^^

    ==========  ==========================  =====================================
    name        type                        desc
    ==========  ==========================  =====================================
    sessions    ``list[RecordingSession]``  recording sessions with the pathology
    ==========  ==========================  =====================================

    """

    __tablename__ = "pathologies"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    """auto-assigned id"""
    name: Mapped[str] = mapped_column(unique=True)
    """pathology name in German"""
    downloaded: Mapped[bool] = mapped_column(default=False)
    """True if pathology dataset has been downloaded"""

    sessions: Mapped[List["RecordingSession"]] = relationship(
        secondary=recording_session_pathologies,
        primaryjoin=id == recording_session_pathologies.c.pathology_id,
    )
    """recording sessions with the pathology"""


class Speaker(Base):
    """'speakers' SQL table - list of all speakers identified by SprecherID

    Table Columns
    ^^^^^^^^^^^^^

    ==========  ================  ======================================
    name        type   desc
    ==========  ================  ======================================
    id          int               SprecherID
    gender      Literal['m','w']  Geschlecht, gender of the speaker
    birthdate   date              Geburtsdatum, birthdate of the speaker
    ==========  ================  ======================================

    Relationships
    ^^^^^^^^^^^^^

    ==========  ==========================  =================================
    name        type                        desc
    ==========  ==========================  =================================
    sessions    ``list[RecordingSession]``  recording sessions of the speaker
    ==========  ==========================  =================================

    """

    __tablename__ = "speakers"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True)
    """speaker id"""
    gender: Mapped[Literal["m", "w"]] = mapped_column(String(1))
    """gender: (m)an or (w)oman"""
    birthdate: Mapped[date] = mapped_column(Date)
    """birthdate"""

    sessions: Mapped[List["RecordingSession"]] = relationship(
        back_populates="speaker",
        primaryjoin="Speaker.id==RecordingSession.speaker_id",
    )
    """sessions of this speaker"""


class RecordingSession(Base):
    """'recording_sessions' SQL table - list of all recording sessions identified by AufnahmeID

    Table Columns
    ^^^^^^^^^^^^^

    ===========  ================  ========================================
    name         type              desc
    ===========  ================  ========================================
    id           int               AufnahmeID
    date         date              AufnahmeDatum, date of the session
    speaker_id   int               id of the speaker
    speaker_age  int               age of the speaker
    type         Literal['n','p']  AufnahmeTyp, 'p' if having voice problem
    note         str               Diagnose, diagnostic comments in German
    ===========  ================  ========================================

    Relationships
    ^^^^^^^^^^^^^

    ==========  ==========================  ====================================
    name        type                        desc
    ==========  ==========================  ====================================
    speaker     ``Speaker``                 the speaker in session
    pathologies ``list[Pathology]``         list of the speaker's pathologies
    recordings  ``list[Recording]``         list of recordings from this session
    ==========  ==========================  ====================================

    """

    __tablename__ = "recording_sessions"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True)
    """recording session id"""
    date: Mapped[date] = mapped_column(Date)
    """session date"""
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id"))
    """id of the speaker in the session"""
    speaker_age: Mapped[int] = mapped_column()
    """age of the speaker"""
    type: Mapped[Literal["n", "p"]] = mapped_column(String(1))
    """speaker is (n)ormal or (p)athological"""
    note: Mapped[str] = mapped_column()
    """diagnostic comments in German"""

    speaker: Mapped[Speaker] = relationship(
        back_populates="sessions",
        primaryjoin="Speaker.id==RecordingSession.speaker_id",
    )
    """speaker's full data"""
    pathologies: Mapped[List[Pathology]] = relationship(
        secondary=recording_session_pathologies,
        overlaps="sessions",
        primaryjoin=id == recording_session_pathologies.c.session_id,
    )
    """list of speaker's pathologies (empty if healthy)"""
    recordings: Mapped[List["Recording"]] = relationship(
        back_populates="session",
        primaryjoin="Recording.session_id==RecordingSession.id",
    )
    """list of recordings from this session"""


class Recording(Base):
    """'recordings' SQL table - list of all recording datafiles

    Table Columns
    ^^^^^^^^^^^^^

    ===========  ================  ========================================
    name         type              desc
    ===========  ================  ========================================
    id           int               auto-assigned id number
    session_id   int               the recording session of this recording
    utterance    LiteralString     vocal task, see below for the full list
    rate         int               sampling rate in samples/second
    length       int               duration in the number of samples
    nspfile      str               path to the NSP file (acoustic)
    eggfile      str               path to the EGG file (electroglottogram)
    ===========  ================  ========================================

    Relationships
    ^^^^^^^^^^^^^

    ==========  ====================  =======================================
    name        type                  desc
    ==========  ====================  =======================================
    session     ``RecordingSession``  the recording session of this recording
    ==========  ====================  =======================================

    Utterance Types
    ^^^^^^^^^^^^^^^

    There are 14 utterance types (note: all recording sessions yielded in all 14).

    First, there are two types that are unique:

    =========  ===============================================================
    utterance  description
    =========  ===============================================================
    "iau"      full sequence of all the vowels, including onsets and offsets
    "phrase"   "Guten Morgen, wie geht es Ihnen?" (Good morning, how are you?)
    =========  ===============================================================

    The other 12 are sustained vowel segments of the vowel recording.
    These utterances vary by vowels and pitches as coded as follows:

    =====  ======  =====  =====  ============
    vowel  pitch
    -----  ----------------------------------
           normal  low    high   low-high-low
    =====  ======  =====  =====  ============
    /a/    "a_n"   "a_l"  "a_h"  "a_lhl"
    /i/    "a_n"   "a_l"  "a_h"  "a_lhl"
    /u/    "a_n"   "a_l"  "a_h"  "a_lhl"
    =====  ======  =====  =====  ============

    Acoustic and EGG data access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The acoustic and EGG data can be accessed easily via the dynamic properties

    +------------------+
    |dynamic properties|
    +==================+
    |nspdata           |
    |eggdata           |
    +------------------+

    Both are only returns valid data if the necessary dataset has been downloaded
    and the Recording object is obtained via ``SbVoiceDb.get_recordings_in_session()``,
    or ``SbVoiceDb.iter_recordings()`` with the argument ``full_file_paths=True``.
    This is the default behavior.

    Note
    ^^^^

    This table is populated as the per-pathology datasets are downloaded if
    the full dataset is not available locally when the database is created
    and the `SbVoiceDb` object is instantiated with `download_mode='incremental'`
    (default).

    """

    __tablename__ = "recordings"
    __table_args__ = (UniqueConstraint("session_id", "utterance"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    """auto-assigned id"""
    session_id: Mapped[int] = mapped_column(ForeignKey("recording_sessions.id"))
    """the recording session of this recording"""
    utterance: Mapped[LiteralString] = mapped_column(String(5))
    """the utterance type stored in this recording.
    
    =========  ===============================================================
    utterance  description
    =========  ===============================================================
    "iau"      full sequence of all the vowels, including onsets and offsets
    "phrase"   "Guten Morgen, wie geht es Ihnen?" (Good morning, how are you?)
    =========  ===============================================================

    There are also sustained vowel segments of the vowel recording. The utterances
    vary by vowels and pitches, and they are coded by the following utterance 
    types:

    =====  ======  =====  =====  ============
    vowel  pitch
    -----  ----------------------------------
           normal  low    high   low-high-low
    =====  ======  =====  =====  ============
    /a/    "a_n"   "a_l"  "a_h"  "a_lhl"
    /i/    "a_n"   "a_l"  "a_h"  "a_lhl"
    /u/    "a_n"   "a_l"  "a_h"  "a_lhl"
    =====  ======  =====  =====  ============
    
    """
    rate: Mapped[int] = mapped_column()
    """recording sampling rate in samples/second"""
    length: Mapped[int] = mapped_column()
    """recording duration in samples"""
    nspfile: Mapped[str] = mapped_column()
    """relative/absolute path to its nsp file (acoustic data file)"""
    eggfile: Mapped[str] = mapped_column()
    """relative/absolute path to its egg file (electroglottogram data file)"""

    session: Mapped[RecordingSession] = relationship(
        back_populates="recordings",
        primaryjoin="Recording.session_id==RecordingSession.id",
    )
    """associated recording session object"""

    @property
    def nspdata(self) -> np.ndarray[tuple[int], np.int16] | None:
        """acoustic data  (only if nspfile contains an absolute path)"""
        try:
            return nspread(self.nspfile)[1]
        except FileNotFoundError:
            logger.warning(
                "nspfile (%s) not found. Make sure to get recording info with `full_file_paths=True`",
                self.nspfile,
            )
            return None

    @property
    def eggdata(self) -> np.ndarray[tuple[int], np.int16] | None:
        """electroglottogram (EGG) data (only if eggfile contains an absolute path)"""
        try:
            return nspread(self.eggfile)[1]
        except FileNotFoundError:
            logger.warning(
                "eggfile (%s) not found. Make sure to get recording info with `full_file_paths=True`",
                self.eggfile,
            )
            return None


class SbVoiceDb:
    """Saarbrucken Voice Database Downloader and Reader

    :param dbdir: database directory, will be created if it does not exist.
    :param speaker_filter: SQL where clause specific to speakers table, defaults to None
    :param session_filter: SQL where clause specific to recording_sessions table, defaults to None
    :param recording_filter: SQL where clause specific to recordings table, defaults to None
    :param pathology_filter: SQL where clause specific to pathologies table, defaults to None
    :param include_healthy: True to include healthy samples if pathology_filter is set, defaults to None
    :param download_mode: Strategy for downloading acoustic/EGG datasets.
                          - 'lazy' (default) download when recordings table is first accessed
                          - 'immediate' to download at the instantiation
                          - 'off' to disable download feature.
    :param connect_kws: a dictionary of options which will be passed directly
                        to sqlite3.connect() as additional keyword arguments,
                        defaults to None
    :param echo: True to log all statements as well as a repr() of their
                 parameter lists to the default log handler, which defaults to
                 sys.stdout for output, defaults to False
    :param \\**create_engine_kws: Additional keywords to sqlalchemy.create_engine()
                                to open the SQLite database file.

    Database File Structure
    ^^^^^^^^^^^^^^^^^^^^^^^

    The specified ``dbdir`` directory will be organized with the following
    structure:

        <SbVoiceDb.dbdir>/
        ├── data/
        │   ├── 1/
        │   │   ├── sentnces
        │   │   │   ├── 1-phrase.nsp
        │   │   │   └── 1-phrase-egg.egg
        │   │   └── vowels
        │   │       ├── 1-a_h.nsp
        │   │       ├── 1-a_h.nsp
        │   │       ⋮
        │   │       └── 1-u_n-egg.egg
        │   ├── 2/
        │   │   ⋮
        |   ⋮
        |
        └── sbvoice.db

    `sbvoice.db` is the SQLite database file that `SbVoiceDb` creates and
    maintains. This file could be opened and viewed by any SQLite database
    viewer app (e.g., DB Browser for SQLite, https://sqlitebrowser.org/) although
    any alteration of the database file may disrupt the operation of `SbVoiceDb`.

    The ``data`` subdirectory is where the contents of the Zenodo dataset zip
    file (`data.zip` or the collection of all the other zip files) are unzipped
    to. The internal directory structure of the zip file is preserved as is.

    Automatic Dataset Downloading
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    `SbVoiceDb` will automatically download the dataset. The `download_mode`
    argument dictates

    All the subdirectories and files are created by ``SbVoiceDb`` unless the
    `data` subdirectory is pre-populated

    The ``data`` subdirectory will be created and populated immediately by
    ``SbVoiceDb` if instantiated with `download_mode='once'`.  or  as
    it will automatically download datasets as recordings are accessed.
    Alternately, ``SbVoiceDb.download_data()``

    may be pre-populated form the data.zip ZIP file
    downloaded from Zenodo (v2) by unzipping its content. Only a portion of
    dataset could be downloaded if so desired. Otherwise, this subdirectory

    Downloading

    """

    _datadir: str
    _dbdir: str
    _db: Engine
    _try_download: bool = False

    _speaker_filter: sql_expr.ColumnElement | None = None
    _session_filter: sql_expr.ColumnElement | None = None
    _pathology_filter: sql_expr.ColumnElement | None = None
    _recording_filter: sql_expr.ColumnElement | None = None
    _include_normal: bool = False  # True to include normals with pathology filter

    def __init__(
        self,
        dbdir: str,
        speaker_filter: sql_expr.ColumnElement | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
        pathology_filter: sql_expr.ColumnElement | None = None,
        include_healthy: bool | None = None,
        download_mode: Literal["immediate", "lazy", "off"] = "lazy",
        connect_kws: dict | None = None,
        echo: bool = False,
        **create_engine_kws,
    ):

        self._dbdir = dbdir
        self._datadir = path.join(self._dbdir, "data")

        if speaker_filter is not None:
            self._speaker_filter = speaker_filter
        if session_filter is not None:
            self._session_filter = session_filter
        if recording_filter is not None:
            self._recording_filter = recording_filter
        if pathology_filter is not None:
            self._pathology_filter = pathology_filter
            self._include_normal = include_healthy or False

        self._try_download = download_mode == "lazy"

        ### configure the SQLite database

        if not path.exists(self._datadir):
            makedirs(self._datadir)

        db_path = self.dbfile
        db_exists = path.exists(db_path)

        self._db = create_engine(
            f"sqlite+pysqlite:///{db_path}",
            connect_args=connect_kws or {},
            echo=echo,
            **create_engine_kws,
        )

        Base.metadata.create_all(self._db)  # -> fails if not in conformance

        if not db_exists:
            # if a new database file was just created, populate it (except for recordings)
            self._populate_db()
            # update downloaded column of pathologies table and settings.healthy_downloaded
            existing_pathos = self._mark_downloaded()

            # if any portion of dataset already exists, populate the DB with its recording info
            if len(existing_pathos):
                self._populate_recordings(pathology=existing_pathos)

        if download_mode == "immediate":
            # download full dataset if download_mode is 'immediate'
            self.download_data()

    @property
    def dbfile(self) -> str:
        """filepath of the SQLite database"""
        return path.join(self._dbdir, "sbvoice.db")

    @property
    def datadir(self) -> str:
        """filepath of the data subdirectory

        This is the base directory of nspfile and eggfile columns of
        the recordings SQLite table"""
        return self._datadir

    @property
    def lazy_download(self) -> bool:
        """True if lazy download is allowed but has not triggered yet"""
        return self._try_download

    @property
    def number_of_all_sessions(self) -> int:
        """total number of recording sessions in the database (not subject to the current filter settings)"""
        with Session(self._db) as session:
            return session.scalar(select(sql_expr.func.count(RecordingSession.id))) or 0

    @property
    def number_of_sessions_downloaded(self) -> int:
        """number of recording sessions already downloaded (not subject to the current filter settings)"""

        with Session(self._db) as session:
            count = sum(
                session.scalars(
                    select(sql_expr.func.count(RecordingSession.id)).where(
                        RecordingSession.pathologies.any(Pathology.downloaded)
                    )
                )
            )
            if (
                session.scalar(
                    select(Setting.value).where(Setting.key == "healthy_downloaded")
                )
                == "1"
            ):
                count += (
                    session.scalar(
                        select(sql_expr.func.count(RecordingSession.id)).where(
                            RecordingSession.type == "n"
                        )
                    )
                    or 0
                )
            return count

    @property
    def missing_datasets(self) -> list[str]:
        """the list of missing sub-dataset names (pathology name or 'healthy')"""
        return list(self._get_download_list())

    @property
    def speaker_filter(self) -> sql_expr.ColumnElement | None:
        """user-defined where clause for speakers table if set"""
        return self._speaker_filter

    @property
    def session_filter(self) -> sql_expr.ColumnElement | None:
        """user-defined where clause for recording_sessions table if set"""
        return self._session_filter

    @property
    def recording_filter(self) -> sql_expr.ColumnElement | None:
        """user-defined where clause for recordings table if set"""
        return self._recording_filter

    @property
    def pathology_filter(self) -> sql_expr.ColumnElement | None:
        """user-defined where clause for pathologies table if set"""
        return self._pathology_filter

    @property
    def include_healthy(self) -> bool | None:
        """user-defined where clause for speakers table if set"""
        return self._include_normal

    ############################
    ### PATHOLOGY ACCESS METHODS

    def _pathology_select(
        self, *args, exclude_related_filters=False, **kwargs
    ) -> Select:
        stmt = select(*args, **kwargs)

        if self._pathology_filter is not None:
            stmt = stmt.where(self._pathology_filter)

        if exclude_related_filters:
            return stmt

        session_filters = []
        if self._session_filter is not None:
            session_filters.append(self._session_filter)
        if self._speaker_filter is not None:
            session_filters.append(RecordingSession.speaker.has(self._speaker_filter))
        if self._recording_filter is not None:
            session_filters.append(
                RecordingSession.recordings.any(self._recording_filter)
            )
        nsfilt = len(session_filters)
        session_filter = (
            sql_expr.and_(*session_filters)
            if nsfilt > 1
            else session_filters[0] if nsfilt else None
        )
        if session_filter is not None:
            stmt = stmt.where(Pathology.sessions.any(session_filter))

        return stmt

    def get_pathology_count(self) -> int:
        """Returns the number of unique pathologies"""
        with Session(self._db) as session:
            return (
                session.scalar(
                    self._pathology_select(sql_expr.func.count(Pathology.id))
                )
                or 0
            )

    def iter_pathologies(self) -> Iterator[Pathology]:
        """Iterates over Pathology objects of unique pathologies"""
        with Session(self._db) as session:
            for patho in session.scalars(self._pathology_select(Pathology)):
                yield patho

    def get_pathology_ids(self) -> Sequence[int]:
        """Return the list of the ids of all the unique pathologies"""
        with Session(self._db) as session:
            return session.scalars(
                self._pathology_select(Pathology.id).order_by(Pathology.name)
            ).fetchall()

    def get_pathology_names(self) -> Sequence[str]:
        """Return the list of the names of all the unique pathologies"""
        with Session(self._db) as session:
            return session.scalars(
                self._pathology_select(Pathology.name).order_by(Pathology.name)
            ).fetchall()

    def get_pathology_name(self, pathology_id: int) -> str | None:
        """Return the name of the specified pathology id"""
        with Session(self._db) as session:
            return session.scalar(
                select(Pathology.name).where(Pathology.id == pathology_id)
            )

    def get_pathology_id(self, name: str) -> int | None:
        """Return the id of the specified pathology name"""
        with Session(self._db) as session:
            return session.scalar(select(Pathology.id).where(Pathology.name == name))

    def get_pathologies_in_session(self, session_id: int) -> Sequence[str]:
        """Return the list of pathology ids which are associated with the specified recording session."""
        with Session(self._db) as session:
            return session.scalars(
                self._pathology_select(
                    Pathology.name, exclude_related_filters=True
                ).where(Pathology.session.has(RecordingSession.id == session_id))
            ).fetchall()

    ##########################
    ### SPEAKER ACCESS METHODS

    def _speaker_select(
        self,
        *args,
        exclude_related_filters: bool = False,
        **kwargs,
    ) -> Select:
        stmt = select(*args, **kwargs)

        if self._speaker_filter is not None:
            stmt = stmt.where(self._speaker_filter)

        if exclude_related_filters:
            return stmt

        session_filters = []
        if not exclude_related_filters:
            if self._session_filter is not None:
                session_filters.append(self._session_filter)
        if self._pathology_filter is not None:
            f = RecordingSession.pathologies.any(self._pathology_filter)
            if self._include_normal is True:
                f = sql_expr.or_(RecordingSession.type == "n", f)
            session_filters.append(f)
        if self._recording_filter is not None:
            session_filters.append(
                RecordingSession.recordings.any(self._recording_filter)
            )
        nsfilt = len(session_filters)
        session_filter = (
            sql_expr.and_(*session_filters)
            if nsfilt > 1
            else session_filters[0] if nsfilt else None
        )
        if session_filter is not None:
            stmt = stmt.where(Speaker.sessions.any(session_filter))

        return stmt

    def get_speaker_count(self) -> int:
        """Return the number of unique speakers"""
        with Session(self._db) as session:
            return (
                session.scalar(self._speaker_select(sql_expr.func.count(Speaker.id)))
                or 0
            )

    def get_speaker_ids(self) -> Sequence[int]:
        """Return an id list of all the unique speakers"""
        with Session(self._db) as session:
            return session.scalars(
                self._speaker_select(Speaker.id).order_by(Speaker.id)
            ).fetchall()

    def get_speaker_id(self, n: int) -> int | None:
        """Return an id of the n-th speaker"""
        with Session(self._db) as session:
            return session.scalar(
                self._speaker_select(Speaker.id).order_by(Speaker.id).offset(n).limit(1)
            )

    def get_speaker_index(self, speaker_id: int) -> int | None:
        """Return the position of the specified speaker"""
        with Session(self._db) as session:
            try:
                return (
                    session.scalars(
                        self._speaker_select(Speaker.id).order_by(Speaker.id)
                    )
                    .all()
                    .index(speaker_id)
                )
            except ValueError:
                return None

    def get_speaker(self, speaker_id: int) -> Speaker | None:
        """get the full speaker info of the specified speaker"""
        with Session(self._db) as session:
            return session.scalar(select(Speaker).where(Speaker.id == speaker_id))

    def iter_speakers(self) -> Iterator[Speaker]:
        """iterate over all the speakers"""

        with Session(self._db) as session:
            for row in session.scalars(self._speaker_select(Speaker)):
                yield row

    ####################################
    ### RECORDING SESSION ACCESS METHODS

    def _session_select(
        self,
        *args,
        exclude_pathology_filter: bool = False,
        exclude_speaker_filter: bool = False,
        exclude_recording_filter: bool = False,
        **kwargs,
    ) -> Select:
        stmt = select(*args, **kwargs)

        if self._session_filter is not None:
            stmt = stmt.where(self._session_filter)
        if not exclude_speaker_filter and self._speaker_filter is not None:
            stmt = stmt.where(RecordingSession.speaker.has(self._speaker_filter))
        if self._pathology_filter is not None:
            if not exclude_pathology_filter:
                f = RecordingSession.pathologies.any(self._pathology_filter)
                if self._include_normal:
                    f = sql_expr.or_(f, RecordingSession.type == "n")
                stmt = stmt.where(f)
        if not exclude_recording_filter and self._recording_filter is not None:
            stmt = stmt.where(RecordingSession.recordings.any(self._recording_filter))

        return stmt

    def get_session_count(self, speaker_id: int | None = None) -> int:
        """Return the number of unique recording sessions

        :param speaker_id: specify speaker id, defaults to None to get
                           the number across all speakers
        """
        stmt = self._session_select(sql_expr.func.count(RecordingSession.id))
        if speaker_id is not None:
            stmt = stmt.where(RecordingSession.speaker_id == speaker_id)
        with Session(self._db) as session:
            return session.scalar(stmt) or 0

    def get_session_ids(self, speaker_id: int | None = None) -> Sequence[int]:
        """Return an id list of all the unique recording sessions

        :param speaker_id: specify speaker id, defaults to None to get
                           the number across all speakers
        """

        stmt = self._session_select(RecordingSession.id).order_by(RecordingSession.id)
        if speaker_id is not None:
            stmt = stmt.where(RecordingSession.speaker_id == speaker_id)
        with Session(self._db) as session:
            return session.scalars(stmt).fetchall()

    def iter_sessions(
        self, speaker_id: int | None = None
    ) -> Iterator[RecordingSession]:
        """iterate over all the sessions

        :param speaker_id: specify speaker id, defaults to None to get
                           the number across all speakers
        """

        stmt = self._session_select(RecordingSession)
        if speaker_id is not None:
            stmt = stmt.where(RecordingSession.speaker_id == speaker_id)
        with Session(self._db) as session:
            for sess in session.scalars(stmt):
                yield sess

    def get_session_id(self, n: int, speaker_id: int | None = None) -> int | None:
        """Returns the id of the n-th recording session

        :param n: the position of recording session in the default order
        :param speaker_id: specify speaker id, defaults to None to get
                           the number across all speakers
        """

        stmt = (
            self._session_select(RecordingSession.id)
            .order_by(RecordingSession.id)
            .offset(n)
            .limit(1)
        )
        if speaker_id is not None:
            stmt = stmt.where(RecordingSession.speaker_id == speaker_id)
        with Session(self._db) as session:
            return session.scalar(stmt)

    def get_session_index(
        self, session_id: int, speaker_id: int | None = None
    ) -> int | None:
        """Returns the position of the specified recording session

        :param session_id: recording session id
        :param speaker_id: specify speaker id, defaults to None to get
                           the number across all speakers
        """

        stmt = self._session_select(RecordingSession.id).order_by(RecordingSession.id)
        if speaker_id is not None:
            stmt = stmt.where(RecordingSession.speaker_id == speaker_id)

        with Session(self._db) as session:
            try:
                return session.scalars(stmt).all().index(session_id)
            except ValueError:
                return None

    def get_session(self, session_id: int) -> RecordingSession | None:
        """Retrun the RecordingSession object by its id"""
        with Session(self._db) as session:
            return session.scalar(
                select(RecordingSession).where(RecordingSession.id == session_id)
            )

    def get_session_ids_with_pathology(
        self, pathology: str | int | None
    ) -> Sequence[int]:
        """Returns the list of recording sessions with the specified pathology

        :param pathology: either the name or the id of the pathology
        """
        with Session(self._db) as session:
            if pathology is None:  # healthy samples
                if not self._include_normal:
                    return []
                stmt = self._session_select(
                    RecordingSession.id, exclude_pathology_filter=True
                ).where(RecordingSession.type == "n")
            else:  # pathology samples
                # if pathology name given, convert it to id
                pathology_id = (
                    pathology
                    if isinstance(pathology, int)
                    else session.scalar(
                        self._pathology_select(Pathology.id).where(
                            Pathology.name == pathology
                        )
                    )
                )
                if pathology_id is None:
                    raise ValueError(
                        f"{pathology=} is not a registered pathology name."
                    )

                stmt = self._session_select(
                    RecordingSession.id, exclude_pathology_filter=True
                ).where(RecordingSession.pathologies.any(Pathology.id == pathology_id))

            return session.scalars(stmt).fetchall()

    def get_session_ids_of_all_pathologies(
        self, use_name: bool = False
    ) -> dict[str | int | None, Sequence[int]]:
        """Returns a dict of session ids of all pathologies in the filtered database

        :param use_name: True to use pathology name as the keys, defaults to False
        """
        keys: dict[int | None, str | int | None] = (
            {patho.id: patho.name for patho in self.iter_pathologies()}
            if use_name
            else {patho_id: patho_id for patho_id in self.get_pathology_ids()}
        )
        if self._pathology_filter is None or self._include_normal is True:
            keys[None] = "healthy" if use_name else None

        return {v: self.get_session_ids_with_pathology(k) for k, v in keys.items()}

    ############################
    ### RECORDING ACCESS METHODS

    def _recording_select(
        self,
        *args,
        exclude_related_filters: bool = False,
        exclude_pathology_filter: bool = False,
        additional_session_filters: Sequence[sql_expr.ColumnElement] | None = None,
        **kwargs,
    ) -> Select:

        stmt = select(*args, **kwargs)

        if self._recording_filter is not None:
            stmt = stmt.where(self._recording_filter)

        if exclude_related_filters:
            return stmt

        session_filters = []

        if self._session_filter is not None:
            session_filters.append(self._session_filter)
        if self._speaker_filter is not None:
            session_filters.append(RecordingSession.speaker.has(self._speaker_filter))
        if not exclude_pathology_filter and self._pathology_filter is None:
            if self._pathology_filter is not None:
                f = RecordingSession.pathologies.any(self._pathology_filter)
                if self._include_normal:
                    f = sql_expr.or_(RecordingSession.type == "n", f)
                session_filters.append(f)
        if additional_session_filters is not None:
            session_filters.extend(additional_session_filters)

        nfilt = len(session_filters)
        session_filter = (
            sql_expr.and_(*session_filters)
            if nfilt > 1
            else session_filters[0] if nfilt else None
        )
        if session_filter is not None:
            stmt = stmt.where(Recording.session.has(session_filter))

        return stmt

    def get_recording_count(self, session_id: int | None = None) -> int:
        """Return the number of recordings

        :param session_id: specify recording session id, defaults to None to get
                           the number across all recording sessions
        """
        if self._try_download:
            self.download_data()

        stmt = self._recording_select(
            sql_expr.func.count(Recording.id),
            exclude_related_filters=session_id is not None,
        )

        if session_id is not None:
            stmt = stmt.where(Recording.session.has(RecordingSession.id == session_id))

        with Session(self._db) as session:
            return session.scalar(stmt) or 0

    def get_recording_ids(self, session_id: int | None = None) -> Sequence[int]:
        """Return an id list of all the unique recordings
        :param session_id: specify recording session id, defaults to None to get
                           the number across all recording sessions
        """
        if self._try_download:
            self.download_data()

        stmt = self._recording_select(Recording.id).order_by(Recording.id)
        if session_id is not None:
            stmt = stmt.where(Recording.session.has(RecordingSession.id == session_id))

        with Session(self._db) as recording:
            return recording.scalars(stmt).fetchall()

    def get_recording_id(self, n: int, session_id: int | None = None) -> int | None:
        """Return the recording id associated with the n-th recording

        :param n: position of the recording in the list
        :param session_id: specify recording session id, defaults to None to get
                           the number across all recording sessions
        """
        if self._try_download:
            self.download_data()

        stmt = (
            self._recording_select(Recording.id)
            .order_by(Recording.id)
            .offset(n)
            .limit(1)
        )
        if session_id is not None:
            stmt = stmt.where(Recording.session.has(RecordingSession.id == session_id))

        with Session(self._db) as session:
            rid = session.scalar(stmt)

        return rid

    def get_recording_index(
        self, recording_id: int, session_id: int | None = None
    ) -> int | None:
        """_summary_

        :param recording_id: _description_
        :param session_id: specify recording session id, defaults to None to get
                           the number across all recording sessions
        :return: _description_
        """
        if self._try_download:
            self.download_data()

        stmt = self._recording_select(Recording.id).all().index(recording_id)
        if session_id is not None:
            stmt = stmt.where(Recording.session.has(RecordingSession.id == session_id))

        with Session(self._db) as session:
            try:
                return session.scalars(stmt)
            except ValueError:
                return None

    def get_recording(
        self, recording_id: int, full_file_paths: bool = False
    ) -> Recording | None:
        if self._try_download:
            self.download_data()

        with Session(self._db) as session:
            rec = session.scalar(select(Recording).where(Recording.id == recording_id))

        if rec is not None and full_file_paths:
            rec.nspfile = path.join(self._datadir, rec.nspfile)
            rec.eggfile = path.join(self._datadir, rec.eggfile)
        return rec

    def iter_recordings(self, full_file_paths: bool = True) -> Iterator[Recording]:
        """iterate over recordings

        :param full_file_paths: True to expand the NSP and EGG file paths to full path, defaults to False
        :yield: ``Recording`` object
        """

        if self._try_download:
            self.download_data()

        with Session(self._db) as session:
            for rec in session.scalars(self._recording_select(Recording)):
                if full_file_paths:
                    rec.nspfile = path.join(self._datadir, rec.nspfile)
                    rec.eggfile = path.join(self._datadir, rec.eggfile)
                yield rec

    ###################
    ### GENERAL METHODS

    def download_data(self):
        """download minimal dataset required for current filter configurations"""

        session_counts = self._get_download_list()

        total_sessions = sum(session_counts.values())

        if total_sessions > 0:

            all_sessions = self.number_of_all_sessions

            if total_sessions > all_sessions:
                # duplicates in the pathologies are too large to warrant pathology-wise download
                download_data(self._datadir)
                patho_list = None

            else:
                # per-pathology downloads
                for patho_name in session_counts:
                    download_data(self._datadir, patho_name)
                patho_list = list(session_counts)

            # insert the new recordings to the database
            self._populate_recordings(patho_list)

        # downloading no longer needed
        self._try_download = False

    def query(
        self,
        sql_statement: str | sql_expr.Executable,
        params: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    ) -> Result:

        if isinstance(sql_statement, str):
            sql_statement = text(sql_statement)

        with Session(self._db) as session:
            return session.execute(sql_statement, params)

    def to_pandas(self) -> "pandas.DataFrame":
        import pandas as pd

        raise NotImplementedError()

    ###################
    ### PRIVATE METHODS

    def _populate_db(self):
        rexp = re.compile(r", ")
        with Session(self._db) as session:

            for row, _ in zip(
                download_database(),
                tqdm.tqdm(range(2225), desc="Populating SQLite database "),
            ):
                session_id = int(row["AufnahmeID"])
                if session_id > 2611:
                    # sessions with id>2611 do not have any data files
                    continue

                speaker_id = int(row["SprecherID"])
                birthdate = datetime.strptime(row["Geburtsdatum"], "%Y-%m-%d")
                session_date = datetime.strptime(row["AufnahmeDatum"], "%Y-%m-%d")

                # calculate speaker's age at the time of the recording session
                speaker_age = session_date.year - birthdate.year
                if (session_date.month, session_date.day) < (
                    birthdate.month,
                    birthdate.day,
                ):
                    speaker_age -= 1

                # set speaker
                session.execute(
                    insert(Speaker)
                    .prefix_with("OR IGNORE")
                    .values(
                        {
                            "id": speaker_id,
                            "birthdate": birthdate.date(),
                            "gender": row["Geschlecht"],
                        }
                    )
                )

                # set recording
                session.execute(
                    insert(RecordingSession).values(
                        {
                            "id": session_id,
                            "date": session_date,
                            "speaker_id": speaker_id,
                            "speaker_age": speaker_age,
                            "type": row["AufnahmeTyp"],
                            "note": row["Diagnose"],
                        }
                    )
                )

                # list pathologies
                pathologies = [
                    unicodedata.normalize("NFC", p)
                    for p in rexp.split(row["Pathologien"])
                    if p
                ]
                if len(pathologies):
                    session.execute(
                        insert(Pathology)
                        .prefix_with("OR IGNORE")
                        .values([{"name": name} for name in pathologies])
                    )

                    patho_ids = session.scalars(
                        select(Pathology.id).where(Pathology.name.in_(pathologies))
                    )
                    session.execute(
                        recording_session_pathologies.insert().values(
                            [
                                {"session_id": session_id, "pathology_id": id}
                                for id in patho_ids
                            ]
                        )
                    )

            # scan data availability
            session_id = session.scalar(
                select(RecordingSession.id).where(RecordingSession.type == "n").limit(1)
            )
            session.add(Setting(key="healthy_downloaded", value=session_id is not None))
            for pid in session.scalars(select(Pathology.id)):
                session_id = session.scalar(
                    select(recording_session_pathologies.c.session_id)
                    .where(recording_session_pathologies.c.pathology_id == pid)
                    .limit(1)
                )
                if path.exists(path.join(self._datadir, str(session_id))):
                    session.execute(
                        update(Pathology)
                        .where(Pathology.id == pid)
                        .values({"downloaded": True})
                    )

            # save
            session.commit()

    def _populate_recordings(self, pathology: Sequence[str] | None = None):
        """scan data directory and populate the recordings table

        :param pathology: populate only the recordings associated with the
                          specified pathologies, defaults to None to update all
                          recordings
        """

        # build SQL statement to retrieve recording session id's
        stmt = select(RecordingSession.id)
        if pathology is not None:
            # add where clause to specify pathologies or healthy
            try:
                i = pathology.index("healthy")
            except ValueError:
                has_healthy = False
            else:
                pathology = [patho for j, patho in enumerate(pathology) if i != j]
                has_healthy = True

            condition = RecordingSession.pathologies.any(
                Pathology.name == pathology
                if isinstance(pathology, str)
                else Pathology.name.in_(pathology)
            )
            if has_healthy:
                condition = sql_expr.or_(condition, RecordingSession.type == "n")
            stmt = stmt.where(condition)

        with Session(self._db) as session:
            # only if speaker is missing

            session_ids = list(session.scalars(stmt))
            nrecs = len(session_ids)

            for rid, _ in zip(
                session_ids,
                tqdm.tqdm(range(nrecs), desc="Populating recordings table "),
            ):
                dirpath = path.join(self._datadir, str(rid))
                nspfiles = [
                    nspfile
                    for nspfile in glob.glob(
                        "**/*.nsp", root_dir=dirpath, recursive=True
                    )
                    if not nspfile.endswith("-fixed.nsp")
                ]
                eggfiles = [
                    f"{path.splitext(nspfile)[0]}-egg.egg" for nspfile in nspfiles
                ]

                # TODO: compact dataset by deleting all vowel files except for iau and
                #       create tstart and tend columns on the recordings table

                # fix the known errors in dataset
                if rid == 713:
                    # corrupted nsp/egg files
                    for i, (nspfile, eggfile) in enumerate(zip(nspfiles, eggfiles)):
                        nspfiles[i] = fix_incomplete_nsp(nspfile, dirpath)
                        eggfiles[i] = fix_incomplete_nsp(eggfile, dirpath)
                elif rid == 980:
                    # most nsp/egg's swapped, partially in iau
                    for i, (nspfile, eggfile) in enumerate(zip(nspfiles, eggfiles)):
                        utype = nspfile[:-4].rsplit("-")[-1]
                        if utype == "iau":
                            nspfiles[i], eggfiles[i] = swap_nsp_egg(
                                nspfile, eggfile, dirpath, n=583414
                            )
                        elif utype not in ("a_n", "i_n", "u_n", "i_h"):
                            nspfiles[i], eggfiles[i] = swap_nsp_egg(
                                nspfile, eggfile, dirpath
                            )
                elif rid in (1697, 139, 141):
                    # all nsp/egg's swapped
                    for i, (nspfile, eggfile) in enumerate(zip(nspfiles, eggfiles)):
                        nspfiles[i], eggfiles[i] = swap_nsp_egg(
                            nspfile, eggfile, dirpath
                        )

                for nspfile, eggfile in zip(nspfiles, eggfiles):
                    utterance = path.basename(nspfile[:-4]).rsplit("-")[1]
                    hdr = cast(
                        NSPHeaderDict,
                        nspread(path.join(dirpath, nspfile), just_header=True),
                    )
                    entry = {
                        "session_id": rid,
                        "utterance": utterance,
                        "rate": hdr["rate"],
                        "length": hdr["length"],
                        "nspfile": nspfile.replace("\\", "/"),
                        "eggfile": eggfile.replace("\\", "/"),
                    }
                    session.execute(
                        insert(Recording).prefix_with("OR IGNORE").values(entry)
                    )
            session.commit()

    def _mark_downloaded(self) -> list[str]:
        """scan the data folder and mark downloaded flags

        :returns: list of existing per-pathology datasets
        """

        patho_list = []

        with Session(self._db) as session:
            for patho in session.scalars(select(Pathology)):

                sessions = patho.sessions
                downloaded = all(
                    path.exists(path.join(self._datadir, str(session)))
                    for session in sessions
                )
                session.execute(
                    update(Pathology)
                    .where(Pathology.id == patho.id)
                    .values({"downloaded": downloaded})
                )
                if downloaded:
                    patho_list.append(patho.name)

            sessions = session.scalars(
                select(RecordingSession.id).where(RecordingSession.type == "n")
            )
            downloaded = all(
                path.exists(path.join(self._datadir, str(session)))
                for session in sessions
            )
            session.execute(
                update(Setting)
                .where(Setting.key == "healthy_downloaded")
                .values({"value": downloaded})
            )
            if downloaded:
                patho_list.append("healthy")

            session.commit()

        return patho_list

    def _get_download_list(self) -> dict[str, int]:
        """returns dataset names (keys) and number of missing recording sessions"""

        # get all pathologies (and healthy) and their sessions
        patho_sessions = cast(
            dict[str, Sequence[int]],
            self.get_session_ids_of_all_pathologies(use_name=True),
        )

        with Session(self._db) as session:
            # exclude a pathology if at least one record is found with the pathology
            session_counts = {
                patho_name: len(session_ids)
                for patho_name, session_ids in patho_sessions.items()
                if session.scalar(
                    select(Recording.id).where(Recording.session_id.in_(session_ids))
                )
                is not None
            }
        return session_counts
