"""Saarbrueken Voice Database Reader module"""

from __future__ import annotations

import logging
from datetime import datetime, date
import glob
from os import path, makedirs
import re
import unicodedata
from shutil import rmtree
from contextlib import contextmanager

from tempfile import mkdtemp

from typing_extensions import (
    Literal,
    List,
    Sequence,
    cast,
    Iterator,
    Any,
    LiteralString,
    Generator,
)
import numpy as np

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    declarative_base,
    Session,
    joinedload,
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
    CursorResult,
    Row,
)

import sqlalchemy.sql.expression as sql_expr

from nspfile import read as nspread, NSPHeaderDict
import tqdm

from platformdirs import user_data_dir

from .download import download_data, download_database
from .utils import fix_incomplete_nsp, swap_nsp_egg

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


PathologyLiteral = LiteralString
# fmt:off
UtteranceLiteral = Literal[
    "a_n", "i_n", "u_n",
    "a_l", "i_l", "u_l",
    "a_h", "i_h", "u_h",
    "a_lhl", "i_lhl", "u_lhl",
    "aiu", "phrase",
]
# fmt:on


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
    utterance: Mapped[UtteranceLiteral] = mapped_column(String(5))
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
        """acoustic data  (only if nspfile contains absolute path)"""
        try:
            return nspread(self.nspfile)[1]
        except FileNotFoundError:
            logger.warning(
                "nspfile (%s) not found. Make sure to get the recording info with `full_file_paths=True`",
                self.nspfile,
            )
            return None

    @property
    def eggdata(self) -> np.ndarray[tuple[int], np.int16] | None:
        """electroglottogram (EGG) data (only if eggfile contains absolute path)"""
        try:
            return nspread(self.eggfile)[1]
        except FileNotFoundError:
            logger.warning(
                "eggfile (%s) not found. Make sure to get the recording info with `full_file_paths=True`",
                self.eggfile,
            )
            return None


PathologySummaryColumn = Literal["id", "name", "nb_speakers", "nb_sessions"]

RecordingSessionSummaryColumn = Literal[
    "speaker_id", "gender", "age", "session_id", "type", "nb_recordings"
]

RecordingSummaryColumn = Literal[
    "id",
    "speaker_id",
    "session_id",
    "gender",
    "age",
    "type",
    "utterance",
    "duration",
]


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

    _healthy_label: str = "[Healthy]"

    _datadir: str
    _dbdir: str | None
    _db: Engine
    _try_download: bool = False

    _speaker_filter: sql_expr.ColumnElement | None = None
    _session_filter: sql_expr.ColumnElement | None = None
    _pathology_filter: sql_expr.ColumnElement | None = None
    _recording_filter: sql_expr.ColumnElement | None = None
    _include_normal: bool = False  # True to include normals with pathology filter

    def __init__(
        self,
        dbdir: str | None = None,
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
        if dbdir == ":memory:":  # in-memory database
            self._dbdir = None
            self._datadir = mkdtemp()
        else:
            self._dbdir = user_data_dir("sbvoicedb") if dbdir is None else dbdir
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
        db_exists = self._dbdir is not None and path.exists(db_path)

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

            # create views
            self._create_pathology_summary()
            self._create_recording_session_summary()
            self._create_recording_summary()

        if download_mode == "immediate":
            # download full dataset if download_mode is 'immediate'
            self.download_data()

    def __del__(self):
        if self._dbdir is None:  # in-memory
            # remove the downloaded datasets
            rmtree(self._datadir)

    @property
    def dbfile(self) -> str:
        """filepath of the SQLite database"""
        return (
            ":memory:" if self._dbdir is None else path.join(self._dbdir, "sbvoice.db")
        )

    @property
    def datadir(self) -> str:
        """filepath of the data subdirectory

        This is the base directory of nspfile and eggfile columns of
        the recordings SQLite table"""
        return self._datadir

    @contextmanager
    def execute_sql(
        self,
        statement: str,
        parameters: dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        commit: bool = False,
    ) -> Generator[CursorResult[Any], None, None]:
        """execute an sql statement

        :param statement: SQL statement
        :param parameters: parameters which will be bound into the statement,
                           defaults to None. This may be either a dictionary of
                           parameter names to values, or a mutable sequence
                           (e.g. a list) of dictionaries. When a list of
                           dictionaries is passed, the underlying statement
                           execution will make use of the DBAPI cursor.
                           executemany() method. When a single dictionary is
                           passed,  the DBAPI cursor.execute() method will be
                           used.
        :yield: a Result object
        """
        with self._db.connect() as connection:
            yield connection.execute(text(statement), parameters)
            if commit:
                connection.commit()

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

        has_healthy = self.has_healthy_dataset()

        with Session(self._db) as session:
            count = sum(
                session.scalars(
                    select(sql_expr.func.count(RecordingSession.id)).where(
                        RecordingSession.pathologies.any(Pathology.downloaded)
                    )
                )
            )
            if has_healthy:
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
    def includes_healthy(self) -> bool | None:
        """user-defined where clause for speakers table if set"""
        return self._pathology_filter is None or self._include_normal is not False

    ############################
    ### PATHOLOGY ACCESS METHODS

    def _pathology_select(
        self, *args, downloaded: bool | None = None, **kwargs
    ) -> Select:
        stmt = select(*args, **kwargs)

        if self._pathology_filter is not None:
            stmt = stmt.where(self._pathology_filter)

        if downloaded is not None:
            stmt = stmt.where(
                Pathology.downloaded
                == (sql_expr.true() if downloaded else sql_expr.false())
            )
            return stmt

        session_filters = self._build_session_filters(exclude_pathology=True)
        nsfilt = len(session_filters)
        if nsfilt > 0:
            session_filter = (
                sql_expr.and_(*session_filters)
                if nsfilt > 1
                else session_filters[0] if nsfilt else None
            )
            stmt = stmt.where(Pathology.sessions.any(session_filter))

        return stmt

    def get_pathology_count(self, *, include_healthy: bool = False) -> int:
        """Returns the number of unique pathologies

        :param include_healthy: True to add vocally healthy subset to the count
                                if included, defaults to False
        """
        with Session(self._db) as session:
            n = (
                session.scalar(
                    self._pathology_select(sql_expr.func.count(Pathology.id))
                )
                or 0
            )
        if include_healthy and self.includes_healthy:
            n += 1
        return n

    def iter_pathologies(
        self, *, include_healthy: bool = False, downloaded: bool | None = None
    ) -> Iterator[Pathology]:
        """Iterates over Pathology objects of unique pathologies

        :param include_healthy: True to include vocally healthy, defaults to False
        :param downloaded: (Special iterator mode). True to iterate only already
                           downloaded pathologies, or False to iterate those
                           without their local datasets, defaults to None.

        """

        with Session(self._db) as session:

            if include_healthy and self.includes_healthy:
                yield Pathology(0, self._healthy_label, self.has_healthy_dataset())

            for patho in session.scalars(
                self._pathology_select(Pathology, downloaded=downloaded)
            ):
                yield patho

    def get_pathology_ids(self, *, include_healthy: bool = False) -> Sequence[int]:
        """Return the list of the ids of all the unique pathologies

        :param include_healthy: True to add 0 to the output list to represent
                                vocally healthy subset if included, defaults to False
        """
        with Session(self._db) as session:
            patho_ids = session.scalars(
                self._pathology_select(Pathology.id).order_by(Pathology.name)
            ).fetchall()

        if include_healthy and self.includes_healthy:
            patho_ids = [0, *patho_ids]

        return patho_ids

    def get_pathology_names(self, include_healthy: bool = False) -> Sequence[str]:
        """Return the list of the names of all the unique pathologies

        :param include_healthy: True to add vocally healthy subset's label
                                (`'[Healthy]'` by default) to the names if
                                included, defaults to False
        """

        with Session(self._db) as session:
            names = session.scalars(
                self._pathology_select(Pathology.name).order_by(Pathology.name)
            ).fetchall()

        if include_healthy and self.includes_healthy:
            names = [self._healthy_label, *names]
        
        return names

    def get_pathology_name(self, pathology_id: int) -> str | None:
        """Return the name of the specified pathology id"""

        if pathology_id == 0:
            return self._healthy_label

        with Session(self._db) as session:
            return session.scalar(
                select(Pathology.name).where(Pathology.id == pathology_id)
            )

    def get_pathology_id(self, name: PathologyLiteral) -> int | None:
        """Return the id of the specified pathology name"""

        if name == self._healthy_label:
            return 0
        with Session(self._db) as session:
            return session.scalar(select(Pathology.id).where(Pathology.name == name))

    def get_pathology(
        self,
        pathology_id: int,
        query_sessions: bool = False,
        query_speaker: bool = False,
        query_recordings: bool = False,
    ) -> Pathology | None:
        """retrieve the specified pathology info from `pathlogies` table

        :param pathology_id: id of the pathology
        :param query_sessions: True to populate `Pathology.sessions` attribute,
                              defaults to False
        :param query_speaker: True to populate `Pathology.sessions.speaker`
                              attribute, defaults to False
        :param query_recordings: True to populate `Pathology.sessions.recordings`
                                 attribute, defaults to False
        :return: queried output or None if `pathology_id` is not valid.
        """

        stmt = select(Pathology).where(Pathology.id == pathology_id)

        if query_sessions or query_speaker or query_recordings:
            opts = joinedload(Pathology.sessions)
            if query_recordings:
                opts = opts.subqueryload(RecordingSession.recordings)
            if query_speaker:
                opts = opts.subqueryload(RecordingSession.speaker)
            stmt = stmt.options(opts)

        with Session(self._db) as session:
            return session.scalar(stmt)

    ##########################
    ### SPEAKER ACCESS METHODS

    def _speaker_select(self, *args, **kwargs) -> Select:
        stmt = select(*args, **kwargs)

        if self._speaker_filter is not None:
            stmt = stmt.where(self._speaker_filter)

        session_filters = self._build_session_filters(exclude_speaker=True)
        nsfilt = len(session_filters)
        if nsfilt > 0:
            session_filter = (
                sql_expr.and_(*session_filters)
                if nsfilt > 1
                else session_filters[0] if nsfilt else None
            )
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

    def get_speaker(
        self,
        speaker_id: int,
        query_sessions: bool = False,
        query_recordings: bool = False,
    ) -> Speaker | None:
        """get the full speaker info of the specified speaker

        :param speaker_id: id of the speaker
        :param query_sessions: True to populate `Speaker.sessions` attribute,
                              defaults to False
        :param query_recordings: True to populate `Speaker.sessions.recordings`
                                 attribute, defaults to False
        :return: queried output or None if `speaker_id` is not valid.
        """

        stmt = select(Speaker).where(Speaker.id == speaker_id)

        if query_sessions or query_recordings:
            opts = joinedload(Speaker.sessions)
            if query_recordings:
                opts = opts.subqueryload(RecordingSession.recordings)
            stmt = stmt.options(opts)

        with Session(self._db) as session:
            return session.scalar(stmt)

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
        speaker_id: int | None = None,
        pathologies: (
            PathologyLiteral | int | Sequence[PathologyLiteral] | Sequence[int] | None
        ) = None,
        speaker_filter: sql_expr.ColumnElement | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
        **kwargs,
    ) -> Select:
        stmt = select(*args, **kwargs)

        if speaker_id is not None:
            stmt = stmt.where(RecordingSession.speaker_id == speaker_id)

        # sessions table filter
        if self._session_filter is not None:
            session_filter = (
                self._session_filter
                if session_filter is None
                else sql_expr.and_(self._session_filter, session_filter)
            )
        if session_filter is not None:
            stmt = stmt.where(session_filter)

        # speakers table filter
        if self._speaker_filter is not None:
            speaker_filter = (
                self._speaker_filter
                if speaker_filter is None
                else sql_expr.and_(self._speaker_filter, speaker_filter)
            )
        if speaker_filter is not None:
            stmt = stmt.where(RecordingSession.speaker.has(speaker_filter))

        # pathologies table filter
        if pathologies is not None:
            # use custom pathology filter
            patho_list = (
                [pathologies]
                if isinstance(pathologies, (str, int))
                else list(pathologies)
            )

            try:
                healthy_index = patho_list.index(0)
            except ValueError:
                try:
                    healthy_index = patho_list.index(self._healthy_label)
                except ValueError:
                    healthy_index = None

            if healthy_index is None:
                hf = None
            else:  # healthy included
                patho_list.pop(healthy_index)
                hf = RecordingSession.type == "n"

            npatho = len(patho_list)
            if npatho:

                patho_col = (
                    Pathology.id if isinstance(patho_list[0], int) else Pathology.name
                )

                pf = RecordingSession.pathologies.any(
                    patho_col.in_(patho_list)
                    if npatho > 1
                    else patho_col == patho_list[0]
                )
            else:
                pf = None

            if pf is not None:  # custom pathology filter
                if hf is not None:  # include healthy
                    # pathology + normal
                    pf = sql_expr.or_(hf, pf)
                stmt = stmt.where(pf)
            elif hf is not None:
                stmt = stmt.where(hf)

        else:
            # use the default pathology filter
            if self._pathology_filter is not None:
                f = RecordingSession.pathologies.any(self._pathology_filter)
                if self._include_normal:
                    f = sql_expr.or_(f, RecordingSession.type == "n")
                stmt = stmt.where(f)

        if self._recording_filter is not None:
            if recording_filter is None:
                recording_filter = self._recording_filter
            else:
                recording_filter = sql_expr.and_(
                    self._recording_filter, recording_filter
                )
        if recording_filter is not None:
            if self._try_download:
                self.download_data()
            stmt = stmt.where(RecordingSession.recordings.any(recording_filter))

        return stmt

    def get_session_count(
        self,
        speaker_id: int | None = None,
        pathologies: (
            PathologyLiteral | int | Sequence[PathologyLiteral] | Sequence[int] | None
        ) = None,
        speaker_filter: sql_expr.ColumnElement | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
    ) -> int:
        """Return the number of unique recording sessions

        :param speaker_id: only of the specified speaker_id, defaults to None
        :param pathologies: only sessions wtih the specified pathologies
                            either by their id or name. The healthy speaker_ids can be
                            selected either by using the ``id=0`` or ``name='[Healthy]'``,
                            defaults to None
        :param speaker_filter: additional filter on Speaker columns, defaults to None
        :param session_filter: additional filter on RecordingSession columns, defaults to None
        :param recording_filter: additional filter on Recording columns, defaults to None
        """

        stmt = self._session_select(
            sql_expr.func.count(RecordingSession.id),
            speaker_id=speaker_id,
            pathologies=pathologies,
            speaker_filter=speaker_filter,
            session_filter=session_filter,
            recording_filter=recording_filter,
        )
        with Session(self._db) as session:
            return session.scalar(stmt) or 0

    def get_session_ids(
        self,
        speaker_id: int | None = None,
        pathologies: (
            PathologyLiteral | int | Sequence[PathologyLiteral] | Sequence[int] | None
        ) = None,
        speaker_filter: sql_expr.ColumnElement | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
    ) -> Sequence[int]:
        """Return an id list of all the unique recording sessions

        :param speaker_id: only of the specified speaker_id, defaults to None
        :param pathologies: only sessions wtih the specified pathologies
                            either by their id or name. The healthy speaker_ids can be
                            selected either by using the ``id=0`` or ``name='[Healthy]'``,
                            defaults to None
        :param speaker_filter: additional filter on Speaker columns, defaults to None
        :param session_filter: additional filter on RecordingSession columns, defaults to None
        :param recording_filter: additional filter on Recording columns, defaults to None
        """

        stmt = self._session_select(
            RecordingSession.id,
            speaker_id=speaker_id,
            pathologies=pathologies,
            speaker_filter=speaker_filter,
            session_filter=session_filter,
            recording_filter=recording_filter,
        ).order_by(RecordingSession.id)
        with Session(self._db) as session:
            return session.scalars(stmt).fetchall()

    def iter_sessions(
        self,
        speaker_id: int | None = None,
        pathologies: (
            PathologyLiteral
            | Literal["healthy"]
            | int
            | Sequence[PathologyLiteral | Literal["healthy"]]
            | Sequence[int]
            | None
        ) = None,
        speaker_filter: sql_expr.ColumnElement | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
    ) -> Iterator[RecordingSession]:
        """iterate over all the sessions

        :param speaker_id: only of the specified speaker_id, defaults to None
        :param pathologies: only sessions wtih the specified pathologies
                            either by their id or name. The healthy speaker_ids can be
                            selected either by using the ``id=0`` or ``name='healthy'``,
                            defaults to None
        :param speaker_filter: additional filter on Speaker columns, defaults to None
        :param session_filter: additional filter on RecordingSession columns, defaults to None
        :param recording_filter: additional filter on Recording columns, defaults to None
        """

        stmt = self._session_select(
            RecordingSession,
            speaker_id=speaker_id,
            pathologies=pathologies,
            speaker_filter=speaker_filter,
            session_filter=session_filter,
            recording_filter=recording_filter,
        )

        with Session(self._db) as session:
            for sess in session.scalars(stmt):
                yield sess

    def get_session(
        self,
        session_id: int,
        query_speaker: bool = False,
        query_pathologies: bool = False,
        query_recordings: bool = False,
    ) -> RecordingSession | None:
        """Return a row of the `recording_sessions` table

        :param session_id: _description_
        :param query_speaker: True to populate `RecordingSession.speaker` attribute,
                              defaults to False
        :param query_pathologies: True to populate `RecordingSession.pathologies`
                                  attribute, defaults to False
        :param query_recordings: True to populate `RecordingSession.recordings`
                                 attribute, defaults to False
        :return: queried output or None if `session_id` is not valid.
        """

        stmt = select(RecordingSession).where(RecordingSession.id == session_id)
        if query_recordings:
            stmt = stmt.options(joinedload(RecordingSession.recordings))
        if query_pathologies:
            stmt = stmt.options(joinedload(RecordingSession.pathologies))
        if query_speaker:
            stmt = stmt.options(joinedload(RecordingSession.speaker))

        with Session(self._db) as session:
            return session.scalar(stmt)

    ############################
    ### RECORDING ACCESS METHODS

    def _recording_select(
        self,
        *args,
        session_id: int | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
        **kwargs,
    ) -> Select:

        if self._try_download:
            self.download_data()

        stmt = select(*args, **kwargs)

        if session_id is not None:
            stmt = stmt.where(Recording.session_id == session_id)

        if self._recording_filter is not None:
            recording_filter = (
                self._recording_filter
                if recording_filter is None
                else sql_expr.and_(self._recording_filter, recording_filter)
            )
        if recording_filter is not None:
            stmt = stmt.where(recording_filter)

        session_filters = self._build_session_filters(exclude_record=True)

        if session_filter is not None:
            session_filters.append(session_filter)
        if len(session_filters):
            stmt = stmt.where(Recording.session.has(sql_expr.and_(*session_filters)))

        return stmt

    def get_recording_count(
        self,
        session_id: int | Sequence[int] | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
    ) -> int:
        """Return the number of recordings

        :param session_id: only of the specified session_id, defaults to None
        :param session_filter: additional filter on RecordingSession columns, defaults to None
        :param recording_filter: additional filter on Recording columns, defaults to None
        """

        stmt = self._recording_select(
            sql_expr.func.count(Recording.id),
            session_id=session_id,
            session_filter=session_filter,
            recording_filter=recording_filter,
        )

        with Session(self._db) as session:
            return session.scalar(stmt) or 0

    def get_recording_ids(
        self,
        session_id: int | Sequence[int] | None = None,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
    ) -> Sequence[int]:
        """Return an id list of all the unique recordings

        :param session_id: only of the specified session_id, defaults to None
        :param session_filter: additional filter on RecordingSession columns, defaults to None
        :param recording_filter: additional filter on Recording columns, defaults to None
        """

        stmt = self._recording_select(
            Recording.id,
            session_id=session_id,
            session_filter=session_filter,
            recording_filter=recording_filter,
        ).order_by(Recording.id)

        with Session(self._db) as recording:
            return recording.scalars(stmt).fetchall()

    def get_recording(
        self,
        recording_id: int,
        full_file_paths: bool = False,
        query_session: bool = False,
        query_speaker: bool = False,
    ) -> Recording | None:
        """Return a row of recordings table as a Recording object

        :param recording_id: id of the recording row to retrieve
        :param full_file_paths: True for the returned `Recording.nspfile` and
                                `Recording.eggfile` to contain the full paths,
                                defaults to False
        :param query_session: True to populate `Recording.session` attribute,
                              defaults to False
        :param query_speaker: True to populate `Recording.session.speaker` attribute,
                              defaults to False
        :return: queried output or None if `recording_id` is not valid.
        """
        if self._try_download:
            self.download_data()

        stmt = select(Recording).where(Recording.id == recording_id)

        if query_session or query_speaker:
            opts = joinedload(Recording.session)
            if query_speaker:
                opts = opts.subqueryload(RecordingSession.speaker)
            stmt = stmt.options(opts)

        with Session(self._db) as session:
            rec = session.scalar(stmt)

        if rec is not None and full_file_paths:
            self._datafile_to_full_path(rec)
        return rec

    def iter_recordings(
        self,
        session_id: int | Sequence[int] | None = None,
        full_file_paths: bool = True,
        session_filter: sql_expr.ColumnElement | None = None,
        recording_filter: sql_expr.ColumnElement | None = None,
    ) -> Iterator[Recording]:
        """iterate over recordings

        :param session_id: only of the specified session_id, defaults to None
        :param full_file_paths: True to expand the NSP and EGG file paths to full path, defaults to False
        :param session_filter: additional filter on RecordingSession columns, defaults to None
        :param recording_filter: additional filter on Recording columns, defaults to None
        :yield: ``Recording`` object
        """

        with Session(self._db) as session:
            for rec in session.scalars(
                self._recording_select(
                    Recording,
                    session_id=session_id,
                    session_filter=session_filter,
                    recording_filter=recording_filter,
                )
            ):
                if full_file_paths:
                    self._datafile_to_full_path(rec)
                yield rec

    ###################
    ### SUMMARY VIEW

    def pathology_summary(
        self,
        columns: (
            PathologySummaryColumn | Literal["*"] | Sequence[PathologySummaryColumn]
        ) | dict[PathologySummaryColumn, str] = "*",
        *,
        minimum_speakers: int | None = None,
        maximum_speakers: int | None = None,
        minimum_sessions: int | None = None,
        maximum_sessions: int | None = None,
        order_by: (
            PathologySummaryColumn
            | list[
                PathologySummaryColumn
                | tuple[PathologySummaryColumn, Literal["asc", "desc"]]
            ]
            | None
        ) = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[Row[Any]]:
        """summary table of recording sessions

        :param columns: columns of `'recording_session_summary'` view to include,
                        defaults to "*" to choose all columns
        :param gender: specify gender ('m' or 'w'), defaults to None
        :param minimum_speakers: specify the minimum number of speakers required, defaults to None
        :param maximum_speakers: specify the maximum number of speakers required, defaults to None
        :param minimum_sessions: specify the minimum number of sessions required, defaults to None
        :param maximum_sessions: specify the maximum number of sessions required, defaults to None
        :param order_by: specify the recording sorting order by columns, defaults to None.
                         The specified columns are sorted in the ascending order by default.
                         To sort in a descending order, use `(column, 'desc')` where `column`
                         is the name of the column.
        :param limit: specify the maximum number of recordings to retrieve, defaults to None
        :param offset: specify the first recording to retrieve if `limit` is specified,
                       defaults to None (from the first recording)
        :return: sequence of the fetched rows
        """
        if isinstance(columns, str):
            columns = [columns]
        elif isinstance(columns, dict):
            columns = [f"{c} AS {alias}" for c, alias in columns.items()]

        columns_list = ",".join(columns)

        stmt = f"SELECT {columns_list} FROM pathology_summary"

        where_list = []
        if minimum_speakers is not None and maximum_speakers is not None:
            where_list.append(
                f"nb_speakers BETWEEN {minimum_speakers} AND {maximum_speakers}"
            )
        elif minimum_speakers is not None:
            where_list.append(f"nb_speakers>={minimum_speakers}")
        elif maximum_speakers is not None:
            where_list.append(f"nb_speakers<={maximum_speakers}")
        if minimum_sessions is not None and maximum_sessions is not None:
            where_list.append(
                f"nb_sessions BETWEEN {minimum_sessions} AND {maximum_sessions}"
            )
        elif minimum_sessions is not None:
            where_list.append(f"nb_sessions>={minimum_sessions}")
        elif maximum_sessions is not None:
            where_list.append(f"nb_sessions<={maximum_sessions}")
        if len(where_list) > 0:
            stmt += f" WHERE " + " AND ".join(where_list)

        if order_by is not None:
            if isinstance(order_by, str):
                order_by = [order_by]
            order_by_list = ",".join(
                o if isinstance(o, str) else f"{o[0]} {o[1].upper()}" for o in order_by
            )
            if len(order_by_list):
                stmt += " ORDER BY " + order_by_list

        if limit is not None:
            stmt += f" LIMIT {limit}"
            if offset is not None:
                stmt += f" OFFSET {offset}"

        logger.info("SbVoiceDb.recording_session_summary.SQL:\n%s  ", stmt)

        with self.execute_sql(stmt) as results:
            return results.fetchall()

    def recording_session_summary(
        self,
        columns: (
            RecordingSessionSummaryColumn
            | Literal["*"]
            | Sequence[RecordingSessionSummaryColumn]
        ) | dict[RecordingSessionSummaryColumn, str] = "*",
        *,
        gender: Literal["w", "m"] | None = None,
        minimum_age: int | None = None,
        maximum_age: int | None = None,
        pathologies: str | Sequence[str] | None = None,
        include_normal: bool = True,
        order_by: (
            RecordingSessionSummaryColumn
            | list[
                RecordingSessionSummaryColumn
                | tuple[RecordingSessionSummaryColumn, Literal["asc", "desc"]]
            ]
            | None
        ) = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[Row[Any]]:
        """summary table of recording sessions

        :param columns: columns of `'recording_session_summary'` view to include,
                        defaults to "*" to choose all columns
        :param gender: specify gender ('m' or 'w'), defaults to None
        :param minimum_age: specify the speaker's minimum age, defaults to None
        :param maximum_age: specify the speaker's maximum age, defaults to None
        :param pathologies: specify specific pathologies to include, defaults to None
                            (all pathologies)
        :param include_normal: True to include normal (pathology-free) speakers,
                               False to exclude them, defaults to True
        :param order_by: specify the recording sorting order by columns, defaults to None.
                         The specified columns are sorted in the ascending order by default.
                         To sort in a descending order, use `(column, 'desc')` where `column`
                         is the name of the column.
        :param limit: specify the maximum number of recordings to retrieve, defaults to None
        :param offset: specify the first recording to retrieve if `limit` is specified,
                       defaults to None (from the first recording)
        :return: sequence of the fetched rows
        """
        if isinstance(columns, str):
            columns = [columns]
        elif isinstance(columns, dict):
            columns = [f"{c} AS {alias}" for c, alias in columns.items()]

        columns_list = ",".join(columns)

        stmt = f"SELECT {columns_list} FROM recording_summary"

        where_list = []
        if gender is not None:
            where_list.append(f"gender='{gender}'")
        if minimum_age is not None and maximum_age is not None:
            where_list.append(f"age BETWEEN {minimum_age} AND {maximum_age}")
        elif minimum_age is not None:
            where_list.append(f"age>={minimum_age}")
        elif maximum_age is not None:
            where_list.append(f"age<={maximum_age}")
        if isinstance(pathologies, str):
            pathologies = [pathologies]
        if pathologies is not None:
            npatho = len(pathologies)
            if npatho == 0 and include_normal:
                where_list.append(f"type='n'")
            elif npatho > 0:
                patho_list = ",".join(f"'{p}'" for p in pathologies)
                patho_where = (
                    f"B.name={patho_list}"
                    if npatho == 1
                    else f"B.name IN ({patho_list})"
                )
                patho_select = f"SELECT A.session_id FROM recording_session_pathologies AS A INNER JOIN pathologies AS B ON A.pathology_id=B.id WHERE {patho_where}"
                if include_normal:
                    where_list.append(f"(session_id in ({patho_select}) OR type='n')")
                else:
                    where_list.append(f"session_id in ({patho_select})")
        elif not include_normal:
            # only pathological recordings
            where_list.append(f"type='p'")

        if len(where_list) > 0:
            stmt += f" WHERE " + " AND ".join(where_list)

        if order_by is not None:
            if isinstance(order_by, str):
                order_by = [order_by]
            order_by_list = ",".join(
                o if isinstance(o, str) else f"{o[0]} {o[1].upper()}" for o in order_by
            )
            if len(order_by_list):
                stmt += " ORDER BY " + order_by_list

        if limit is not None:
            stmt += f" LIMIT {limit}"
            if offset is not None:
                stmt += f" OFFSET {offset}"

        logger.info("SbVoiceDb.recording_session_summary.SQL:\n%s  ", stmt)

        with self.execute_sql(stmt) as results:
            return results.fetchall()

    def recording_summary(
        self,
        columns: (
            RecordingSummaryColumn | Literal["*"] | Sequence[RecordingSummaryColumn]
        ) | dict[RecordingSummaryColumn, str] = "*",
        *,
        gender: Literal["w", "m"] | None = None,
        minimum_age: int | None = None,
        maximum_age: int | None = None,
        pathologies: str | Sequence[str] | None = None,
        include_normal: bool = True,
        utterances: UtteranceLiteral | Sequence[UtteranceLiteral] | None = None,
        minimum_duration: float | None = None,
        maximum_duration: float | None = None,
        order_by: (
            RecordingSummaryColumn
            | list[
                RecordingSummaryColumn
                | tuple[RecordingSummaryColumn, Literal["asc", "desc"]]
            ]
            | None
        ) = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[Row[Any]]:
        """summary table of recordings

        :param columns: columns of `'recording_summary'` view to include,
                        defaults to "*" to choose all columns
        :param gender: specify gender ('m' or 'w'), defaults to None
        :param minimum_age: specify the speaker's minimum age, defaults to None
        :param maximum_age: specify the speaker's maximum age, defaults to None
        :param pathologies: specify specific pathologies to include, defaults to None
                            (all pathologies)
        :param include_normal: True to include normal (pathology-free) speakers,
                               False to exclude them, defaults to True
        :param utterances: specify utterances to include, defaults to None
        :param minimum_duration: specify the recordings' minimum duration in seconds, defaults to None
        :param maximum_duration: specify the recordings' maximum duration in seconds, defaults to None
        :param order_by: specify the recording sorting order by columns, defaults to None.
                         The specified columns are sorted in the ascending order by default.
                         To sort in a descending order, use `(column, 'desc')` where `column`
                         is the name of the column.
        :param limit: specify the maximum number of recordings to retrieve, defaults to None
        :param offset: specify the first recording to retrieve if `limit` is specified,
                       defaults to None (from the first recording)
        :return: sequence of the fetched rows
        """
        if isinstance(columns, str):
            columns = [columns]
        elif isinstance(columns, dict):
            columns = [f"{c} AS {alias}" for c, alias in columns.items()]

        columns_list = ",".join(columns)

        stmt = f"SELECT {columns_list} FROM recording_summary"

        where_list = []
        if gender is not None:
            where_list.append(f"gender='{gender}'")
        if minimum_age is not None and maximum_age is not None:
            where_list.append(f"age BETWEEN {minimum_age} AND {maximum_age}")
        elif minimum_age is not None:
            where_list.append(f"age>={minimum_age}")
        elif maximum_age is not None:
            where_list.append(f"age<={maximum_age}")
        if isinstance(pathologies, str):
            pathologies = [pathologies]
        if pathologies is not None:
            npatho = len(pathologies)
            if npatho == 0 and include_normal:
                where_list.append(f"type='n'")
            elif npatho > 0:
                patho_list = ",".join(f"'{p}'" for p in pathologies)
                patho_where = (
                    f"B.name={patho_list}"
                    if npatho == 1
                    else f"B.name IN ({patho_list})"
                )
                patho_select = f"SELECT A.session_id FROM recording_session_pathologies AS A INNER JOIN pathologies AS B ON A.pathology_id=B.id WHERE {patho_where}"
                if include_normal:
                    where_list.append(f"(session_id in ({patho_select}) OR type='n')")
                else:
                    where_list.append(f"session_id in ({patho_select})")
        elif not include_normal:
            # only pathological recordings
            where_list.append(f"type='p'")
        if utterances is not None and len(utterances) > 0:
            if isinstance(utterances, str):
                utterances = [utterances]
            utter_list = ",".join(f"'{u}'" for u in utterances)
            where_list.append(
                f"utterance={utter_list}"
                if len(utter_list) == 1
                else f"utterance IN ({utter_list})"
            )
        if minimum_duration is not None and maximum_duration is not None:
            where_list.append(
                f"duration BETWEEN {minimum_duration} AND {maximum_duration}"
            )
        elif minimum_duration is not None:
            where_list.append(f"duration>={minimum_duration}")
        elif maximum_duration is not None:
            where_list.append(f"duration<={maximum_duration}")

        if len(where_list) > 0:
            stmt += f" WHERE " + " AND ".join(where_list)

        if order_by is not None:
            if isinstance(order_by, str):
                order_by = [order_by]
            order_by_list = ",".join(
                o if isinstance(o, str) else f"{o[0]} {o[1].upper()}" for o in order_by
            )
            if len(order_by_list):
                stmt += " ORDER BY " + order_by_list

        if limit is not None:
            stmt += f" LIMIT {limit}"
            if offset is not None:
                stmt += f" OFFSET {offset}"

        logger.info("SbVoiceDb.recording_summary.SQL:\n%s  ", stmt)

        with self.execute_sql(stmt) as results:
            return results.fetchall()

    ###################
    ### GENERAL METHODS

    def has_healthy_dataset(self) -> bool:
        """Returns True if healthy dataset has been downloaded"""
        with Session(self._db) as session:
            return (
                session.scalar(
                    select(Setting.value).where(Setting.key == "healthy_downloaded")
                )
                == "1"
            )

    def download_data(self):
        """download minimal dataset required for current filter configurations"""

        # download is tried only once
        self._try_download = False

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
            # check pathological samples
            for patho in session.scalars(select(Pathology)):

                rsessions = patho.sessions
                downloaded = all(
                    path.exists(path.join(self._datadir, str(rsession.id)))
                    for rsession in rsessions
                )
                session.execute(
                    update(Pathology)
                    .where(Pathology.id == patho.id)
                    .values({"downloaded": downloaded})
                )
                if downloaded:
                    patho_list.append(patho.name)

            # check healthy samples
            session_ids = session.scalars(
                select(RecordingSession.id).where(RecordingSession.type == "n")
            )
            downloaded = all(
                path.exists(path.join(self._datadir, str(session_id)))
                for session_id in session_ids
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
        """returns dataset names (keys) and the number of recording sessions in each"""

        # get all pathologies (and healthy) and their sessions
        need_healthy = self.includes_healthy and not self.has_healthy_dataset()
        session_counts = {}
        stmt = select(Pathology).where(Pathology.downloaded == sql_expr.false())
        if self._pathology_filter is not None:
            stmt = stmt.where(self._pathology_filter)

        with Session(self._db) as session:

            for patho in session.scalars(stmt):
                session_counts[patho.name] = session.scalar(
                    select(sql_expr.func.count(RecordingSession.id)).where(
                        RecordingSession.pathologies.any(Pathology.id == patho.id)
                    )
                )

            if need_healthy:
                session_counts["healthy"] = session.scalar(
                    select(sql_expr.func.count(RecordingSession.id)).where(
                        RecordingSession.type == "n"
                    )
                )

        return session_counts

    def _datafile_to_full_path(self, rec: Recording):
        """modify Recording's nspfile and eggfile to have the full path of the file"""
        datadir = path.join(self._datadir, str(rec.session_id))
        rec.nspfile = path.join(datadir, rec.nspfile)
        rec.eggfile = path.join(datadir, rec.eggfile)

    def _build_session_filters(
        self,
        exclude_speaker: bool = False,
        exclude_pathology: bool = False,
        exclude_record: bool = False,
    ):
        session_filters = []
        if self._session_filter is not None:
            session_filters.append(self._session_filter)
        if not exclude_speaker and self._speaker_filter is not None:
            session_filters.append(RecordingSession.speaker.has(self._speaker_filter))
        if not exclude_pathology and self._pathology_filter is not None:
            f = RecordingSession.pathologies.any(self._pathology_filter)
            if self._include_normal:
                f = sql_expr.or_(f, RecordingSession.type == "n")
            session_filters.append(f)
        if not exclude_record and self._recording_filter is not None:
            if self._try_download:
                self.download_data()
            session_filters.append(
                RecordingSession.recordings.any(self._recording_filter)
            )

        return session_filters

    def _create_recording_summary(self):
        """add `recording_summary` to the database"""
        with self.execute_sql(
            """CREATE VIEW IF NOT EXISTS recording_summary 
                       (id, speaker_id, gender, age, session_id, type, utterance, duration) AS 
                    SELECT 
                      A.id,
                      B.speaker_id, 
                      C.gender,
                      (strftime('%Y', B.date) - strftime('%Y', C.birthdate)) - (strftime('%m-%d', B.date) < strftime('%m-%d', C.birthdate)), 
                      A.session_id, 
                      B.type,
                      A.utterance,
                      CAST(A.length AS REAL) / A.rate
                    FROM recordings AS A INNER JOIN recording_sessions AS B ON A.session_id=B.id INNER JOIN speakers AS C ON B.speaker_id=C.id
                    ORDER BY speaker_id, session_id, utterance"""
        ):
            ...

    def _create_recording_session_summary(self):
        """add `recording_session_summary` to the database"""
        with self.execute_sql(
            """CREATE VIEW IF NOT EXISTS recording_session_summary 
                       (speaker_id,gender,age,session_id,type,nb_recordings) AS 
                    SELECT 
                      A.speaker_id, 
                      C.gender,
                      (strftime('%Y', A.date) - strftime('%Y', C.birthdate)) - (strftime('%m-%d', A.date) < strftime('%m-%d', C.birthdate)), 
                      A.id, 
                      A.type,
                      QTY.nb_recordings
                    FROM recording_sessions AS A LEFT JOIN
                         (SELECT COUNT(B.session_id) AS nb_recordings, B.session_id FROM recordings AS B GROUP BY B.session_id) AS QTY
                         ON A.id = QTY.session_id
                         INNER JOIN speakers AS C ON A.speaker_id=C.id
                    ORDER BY speaker_id, session_id"""
        ):
            ...

    def _create_pathology_summary(self):
        """add `pathology_summary` to the database"""
        with self.execute_sql(
            """CREATE VIEW IF NOT EXISTS pathology_summary 
                       (id,name,nb_speakers,nb_sessions) AS 
                       SELECT
                        A.id,
                        A.name,
                        QTY.nb_speakers,
                        QTY.nb_sessions
                        FROM pathologies AS A LEFT JOIN
                                (SELECT B.pathology_id, COUNT(DISTINCT C.speaker_id) AS nb_speakers, COUNT(B.pathology_id) AS nb_sessions
                                FROM recording_session_pathologies AS B INNER JOIN recording_sessions AS C ON B.session_id=C.id
                                GROUP BY B.pathology_id) AS QTY
                                ON A.id = QTY.pathology_id
                        ORDER BY A.name"""
        ):
            ...
