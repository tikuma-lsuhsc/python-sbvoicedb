"""Saarbrueken Voice Database Reader module"""

from __future__ import annotations

import logging

# import pandas as pd
from os import path, makedirs
import re
import shutil
import numpy as np
import nspfile
import numpy as np
from tempfile import TemporaryDirectory
from typing import (
    Literal,
    List,
    Callable,
    Iterator,
    Tuple,
    Optional,
    get_args,
    Sequence,
)
from datetime import datetime
import glob


from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    declarative_base,
    Session,
    aliased,
)

# from sqlalchemy_utils import create_view
from sqlalchemy import (
    Engine,
    ForeignKey,
    Column,
    PrimaryKeyConstraint,
    String,
    Date,
    create_engine,
    bindparam,
    select,
    insert,
    update,
    delete,
    func,
    MetaData,
    Table,
    UniqueConstraint,
    Select,
    and_,
    or_,
)
from sqlalchemy.sql.expression import ColumnElement

import tqdm
from nspfile import read as nspread, NSPHeaderDict

from .download import download_data, download_database
from .common import fix_incomplete_nsp

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Base = declarative_base()


class Setting(Base):
    __tablename__ = "settings"
    __table_args__ = (PrimaryKeyConstraint("group", "key"),)
    group: Mapped[str]
    key: Mapped[str]
    value: Mapped[str]


pathology_in_recordings = Table(
    "pathology_in_recordings",
    Base.metadata,
    Column("session_id", ForeignKey("recording_sessions.id"), primary_key=True),
    Column("pathology_id", ForeignKey("pathologies.id"), primary_key=True),
)


class Pathology(Base):
    __tablename__ = "pathologies"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

    recording_sessions: Mapped[List["RecordingSession"]] = relationship(
        secondary=pathology_in_recordings
    )


class Speaker(Base):
    __tablename__ = "speakers"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True)
    gender: Mapped[Literal["m", "w"]] = mapped_column(String(1))
    birthdate: Mapped[str] = mapped_column(Date)

    recording_sessions: Mapped[List["RecordingSession"]] = relationship(
        back_populates="speaker"
    )


class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id"))
    date: Mapped[str] = mapped_column(Date)
    type: Mapped[Literal["n", "p"]] = mapped_column(String(1))
    note: Mapped[str] = mapped_column()

    speaker: Mapped[Speaker] = relationship(back_populates="recording_sessions")
    pathologies: Mapped[List[Pathology]] = relationship(
        secondary=pathology_in_recordings, overlaps="recording_sessions"
    )
    recordings: Mapped[List["Recording"]] = relationship(back_populates="session")


TaskType = Literal[
    # fmt: off
            "a_n", "i_n", "u_n",  "a_l", "i_l", "u_l", "a_h", "i_h", "u_h", 
            "a_lhl", "i_lhl", "u_lhl", "iau", "phase",
    # fmt: on
]


class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = (UniqueConstraint("session_id", "task"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("recording_sessions.id"))
    task: Mapped[TaskType] = mapped_column(
        String(max(len(s) for s in get_args(TaskType)))
    )
    rate: Mapped[int] = mapped_column()
    length: Mapped[int] = mapped_column()
    nspfile: Mapped[str] = mapped_column()
    eggfile: Mapped[str] = mapped_column()

    session: Mapped[RecordingSession] = relationship(back_populates="recordings")


class SbVoiceDb:
    """SbVoiceDb class

    Constructor Arguments

    :param dbdir: databse directory
    :param download_mode: True to use only cached files, defaults to False
    """

    _datadir: str
    _dbdir: str
    _db: Engine
    _download: bool = True

    _speaker_filter: ColumnElement | None = None
    _session_filter: ColumnElement | None = None
    _pathology_filter: ColumnElement | None = None
    _recording_filter: ColumnElement | None = None
    _include_normal: bool = True

    def __init__(
        self,
        dbdir: str,
        download_mode: Literal["once", "incremental", False] = "incremental",
        connect_kws=None,
        echo=False,
        **create_engine_kws,
    ):

        if not path.exists(dbdir):
            raise ValueError(f"invalid datbase path")

        self._dbdir = dbdir
        self._datadir = path.join(self._dbdir, "data")

        db_path = path.join(dbdir, "sbvoice.db")
        db_exists = path.exists(db_path)

        self._db = create_engine(
            f"sqlite+pysqlite:///{db_path}",
            connect_args=connect_kws or {},
            echo=echo,
            **create_engine_kws,
        )

        Base.metadata.create_all(self._db)

        if not db_exists:
            self._populate_db()

        if download_mode == "once" and path.exists(self._datadir):
            if not (self._download_data() or db_exists):
                self._populate_recordings()
        elif download_mode is False:
            self._download = False

    def _populate_db(self):
        rexp = re.compile(r", ")
        with Session(self._db) as session:
            for row, _ in zip(
                download_database(),
                tqdm.tqdm(range(2225), desc="Populating SQLite database "),
            ):
                speaker_id = int(row["SprecherID"])
                session_id = int(row["AufnahmeID"])

                if session_id > 2611:
                    # recording_sessions with id>2611 do not have any data files
                    continue

                # set speaker
                session.execute(
                    insert(Speaker)
                    .prefix_with("OR IGNORE")
                    .values(
                        {
                            "id": speaker_id,
                            "birthdate": datetime.strptime(
                                row["Geburtsdatum"], "%Y-%m-%d"
                            ).date(),
                            "gender": row["Geschlecht"],
                        }
                    )
                )

                # set recording
                session.execute(
                    insert(RecordingSession).values(
                        {
                            "id": session_id,
                            "speaker_id": speaker_id,
                            "date": datetime.strptime(
                                row["AufnahmeDatum"], "%Y-%m-%d"
                            ).date(),
                            "type": row["AufnahmeTyp"],
                            "note": row["Diagnose"],
                        }
                    )
                )

                # list pathologies
                pathologies = [p for p in rexp.split(row["Pathologien"]) if p]
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
                        pathology_in_recordings.insert().values(
                            [
                                {"session_id": session_id, "pathology_id": id}
                                for id in patho_ids
                            ]
                        )
                    )

                # save
                session.commit()

    def _download_data(self, pathology: str | None = None) -> bool:
        """download data from Zenodo if any data is missing

        :param pathology: specify the pathology to download,
                          defaults to download all the data (caution: large
                          download size, ~16 GB)
        :returns: True if downloadng took place
        """

        # download only if not all the speakers are present
        stmt = select(RecordingSession.id)
        if pathology is not None:
            stmt = stmt.where(
                RecordingSession.pathologies.any(Pathology.name == pathology)
            )
        with Session(self._db) as session:
            session_ids = list(session.scalars(stmt))

        do_download = not all(
            path.exists(path.join(self._datadir, str(rid))) for rid in session_ids
        )
        if do_download:
            download_data(self._datadir, pathology)
            self._populate_recordings(pathology)

        return do_download

    def _populate_recordings(self, pathology: str | None = None):
        """scan directory and populate tasks table

        :param pathology: populate only the tasks associated with the specified pathology, defaults to None
        """

        with Session(self._db) as session:
            # only if speaker is missing
            stmt = select(RecordingSession.id)
            if pathology is not None:
                stmt = stmt.where(
                    RecordingSession.pathologies.any(Pathology.name == (pathology))
                )

            session_ids = list(session.scalars(stmt))
            nrecs = len(session_ids)

            for rid, _ in zip(
                session_ids,
                tqdm.tqdm(range(nrecs), desc="Populating recordings table "),
            ):
                dirpath = path.join(self._datadir, str(rid))
                for nspfile in glob.glob("**/*.nsp", root_dir=dirpath, recursive=True):
                    task = path.basename(nspfile[:-4]).rsplit("-")[1]
                    eggfile = f"{path.splitext(nspfile)[0]}-egg.egg"
                    try:
                        hdr: NSPHeaderDict = nspread(
                            path.join(dirpath, nspfile), just_header=True
                        )
                    except RuntimeError:
                        nspfile, oldfile = (
                            fix_incomplete_nsp(path.join(dirpath, nspfile)),
                            nspfile,
                        )
                        hdr = nspread(nspfile, just_header=True)
                        nspfile = path.relpath(nspfile, dirpath)

                    entry = {
                        "session_id": rid,
                        "task": task,
                        "rate": hdr["rate"],
                        "length": hdr["length"],
                        "nspfile": nspfile.replace("\\", "/"),
                    }
                    if path.exists(path.join(dirpath, eggfile)):
                        try:
                            nspread(path.join(dirpath, eggfile), just_header=True)
                        except RuntimeError:
                            eggfile, oldfile = (
                                fix_incomplete_nsp(path.join(dirpath, eggfile)),
                                eggfile,
                            )
                            hdr = nspread(eggfile, just_header=True)
                            eggfile = path.relpath(eggfile, dirpath)
                        entry["eggfile"] = eggfile.replace("\\", "/")
                    session.execute(
                        insert(Recording).prefix_with("OR IGNORE").values(entry)
                    )
            session.commit()

    ################

    def set_speaker_filter(self, where_clause: ColumnElement | None):
        self._speaker_filter = where_clause

    def set_session_filter(self, where_clause: ColumnElement | None):
        self._session_filter = where_clause

    def set_pathology_filter(self, where_clause: ColumnElement | None):
        self._pathology_filter = where_clause

    def set_recording_filter(self, where_clause: ColumnElement | None):
        self._recording_filter = where_clause

    ################

    def _pathology_select(
        self, *args, exclude_session_filter=False, **kwargs
    ) -> Select:
        stmt = select(*args, **kwargs)

        if self._pathology_filter is not None:
            stmt = stmt.where(self._pathology_filter)

        if exclude_session_filter:
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
            and_(*session_filters)
            if nsfilt > 1
            else session_filters[0] if nsfilt else None
        )
        if session_filter is not None:
            stmt.where(Pathology.recording_sessions.any(session_filter))

        return stmt

    def get_pathology_count(self) -> int:
        """Return the number of unique pathologies"""
        with Session(self._db) as session:
            return session.scalar(self._pathology_select(func.count(Pathology.id))) or 0

    def get_pathology_ids(self) -> Sequence[int]:
        """Return an id list of all the unique speakers"""
        with Session(self._db) as session:
            return session.scalars(
                self._pathology_select(Pathology.id).order_by(Pathology.name)
            ).fetchall()

    def get_pathology_names(self) -> Sequence[str]:
        """Return an id list of all the unique speakers"""
        with Session(self._db) as session:
            return session.scalars(
                self._pathology_select(Pathology.name).order_by(Pathology.name)
            ).fetchall()

    def get_pathology_name(self, pathology_id: int) -> str | None:
        with Session(self._db) as session:
            return session.scalar(
                select(Pathology.name).where(Pathology.id == pathology_id)
            )

    def get_pathology_id(self, name: str) -> int | None:
        with Session(self._db) as session:
            return session.scalar(select(Pathology.id).where(Pathology.name == name))

    def get_pathologies_in_session(self, session_id: int) -> Sequence[str]:
        with Session(self._db) as session:
            return session.scalars(
                self._pathology_select(
                    Pathology.name, exclude_session_filter=True
                ).where(Pathology.session.has(RecordingSession.id == session_id))
            ).fetchall()

    ################

    def _speaker_select(self, *args, **kwargs) -> Select:
        stmt = select(*args, **kwargs)

        if self._speaker_filter is not None:
            stmt = stmt.where(self._speaker_filter)

        session_filters = []
        if self._session_filter is not None:
            session_filters.append(self._session_filter)
        if self._pathology_filter is not None:
            f = RecordingSession.pathologies.any(self._pathology_filter)
            if self._include_normal:
                f = or_(f, ~RecordingSession.pathologies.any())
            session_filters.append(f)
        if self._recording_filter is not None:
            session_filters.append(
                RecordingSession.recordings.any(self._recording_filter)
            )
        nsfilt = len(session_filters)
        session_filter = (
            and_(*session_filters)
            if nsfilt > 1
            else session_filters[0] if nsfilt else None
        )
        if session_filter is not None:
            stmt.where(Pathology.recording_sessions.any(session_filter))

        return stmt

    def get_speaker_count(self) -> int:
        """Return the number of unique speakers"""
        with Session(self._db) as session:
            return session.scalar(self._speaker_select(func.count(Speaker.id))) or 0

    def get_speaker_ids(self) -> Sequence[int]:
        """Return an id list of all the unique speakers"""
        with Session(self._db) as session:
            return session.scalars(self._speaker_select(Speaker.id)).fetchall()

    def get_speaker_id(self, index: int) -> int | None:
        with Session(self._db) as session:
            return session.scalar(
                self._speaker_select(Speaker.id)
                .order_by(Speaker.id)
                .offset(index)
                .limit(1)
            )

    def get_speaker_index(self, speaker_id: int) -> int | None:
        with Session(self._db) as session:
            try:
                return (
                    session.scalars(self._speaker_select(Speaker.id))
                    .all()
                    .index(speaker_id)
                )
            except ValueError:
                return None

    def get_speaker(self, speaker_id: int) -> Speaker | None:
        with Session(self._db) as session:
            return session.scalar(select(Speaker).where(Speaker.id == speaker_id))

    ################

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
        if not exclude_pathology_filter and self._pathology_filter is not None:
            f = RecordingSession.pathologies.any(self._pathology_filter)
            if self._include_normal:
                f = or_(f, ~RecordingSession.pathologies.any())
            stmt = stmt.where(f)
        if not exclude_recording_filter and self._recording_filter is not None:
            stmt = stmt.where(RecordingSession.recordings.any(self._recording_filter))

        return stmt

    def get_session_count(self) -> int:
        """Return the number of unique recording sessions"""
        with Session(self._db) as session:
            return (
                session.scalar(self._session_select(func.count(RecordingSession.id)))
                or 0
            )

    def get_session_ids(self) -> Sequence[int]:
        """Return an id list of all the unique sessions"""
        with Session(self._db) as session:
            return session.scalars(self._session_select(RecordingSession.id)).fetchall()

    def get_session_id(self, index: int) -> int | None:
        with Session(self._db) as session:
            return session.scalar(
                self._session_select(RecordingSession.id)
                .order_by(RecordingSession.id)
                .offset(index)
                .limit(1)
            )

    def get_session_index(self, session_id: int) -> int | None:
        with Session(self._db) as session:
            try:
                return (
                    session.scalars(self._session_select(RecordingSession.id))
                    .all()
                    .index(session_id)
                )
            except ValueError:
                return None

    def get_session(self, session_id: int) -> RecordingSession | None:
        with Session(self._db) as session:
            return session.scalar(
                self._session_select(RecordingSession.id).where(
                    RecordingSession.id == session_id
                )
            )

    def get_sessions_with_pathology(self, pathology: str) -> Sequence[RecordingSession]:
        with Session(self._db) as session:
            pathology_id = session.scalar(
                self._pathology_select(Pathology.id).where(Pathology.name == pathology)
            )

            if pathology_id is None:
                return []

            return session.scalars(
                self._session_select(
                    Pathology.name, exclude_pathology_filter=True
                ).where(RecordingSession.pathologies.any(Pathology.id == pathology_id))
            ).fetchall()

    ################

    def _recording_select(
        self, *args, exclude_session_filter: bool = False, **kwargs
    ) -> Select:
        stmt = select(*args, **kwargs)

        if self._recording_filter is not None:
            stmt = stmt.where(self._recording_filter)

        if exclude_session_filter:
            return stmt

        session_filters = []

        if self._session_filter is not None:
            session_filters.append(self._session_filter)
        if self._speaker_filter is not None:
            session_filters.append(RecordingSession.speaker.has(self._speaker_filter))
        if self._pathology_filter is not None:
            f = RecordingSession.pathologies.any(self._pathology_filter)
            if self._include_normal:
                f = or_(f, ~RecordingSession.pathologies.any())
            session_filters.append(f)

        nfilt = len(session_filters)
        session_filter = (
            and_(*session_filters)
            if nfilt > 1
            else session_filters[0] if nfilt else None
        )
        if session_filter is not None:
            stmt = stmt.where(Recording.session.has(session_filter))

        return stmt

    def get_recording_count(self) -> int:
        """Return the number of unique recording recordings"""
        with Session(self._db) as recording:
            return (
                recording.scalar(self._recording_select(func.count(Recording.id))) or 0
            )

    def get_recording_ids(self) -> Sequence[int]:
        """Return an id list of all the unique recordings"""
        with Session(self._db) as recording:
            return recording.scalars(self._recording_select(Recording.id)).fetchall()

    def get_recording_id(self, index: int) -> int | None:
        with Session(self._db) as recording:
            return recording.scalar(
                self._recording_select(Recording.id)
                .order_by(Recording.id)
                .offset(index)
                .limit(1)
            )

    def get_recording_index(self, recording_id: int) -> int | None:
        with Session(self._db) as recording:
            try:
                return (
                    recording.scalars(self._recording_select(Recording.id))
                    .all()
                    .index(recording_id)
                )
            except ValueError:
                return None

    def get_recording(self, recording_id: int) -> Recording | None:
        with Session(self._db) as recording:
            return recording.scalar(
                select(Recording).where(Recording.id == recording_id)
            )

    def get_recordings_in_session(self, session_id: int) -> Sequence[Recording]:
        with Session(self._db) as session:
            return session.scalars(
                self._recording_select(Recording, exclude_session_filter=True).where(
                    Recording.session_id == session_id
                )
            ).fetchall()

    ################

    @property
    def tasks(self) -> List[str]:
        """List of task types"""
        return list(task2key.keys())

    @property
    def diagnoses(self) -> List[str]:
        """List of diagnosis (in German)"""
        return list(self._df_dx["pathology"].unique())

    def query(
        self, columns: List[DataField] = None, include_missing: bool = False, **filters
    ):  # -> pd.DataFrame:
        """query database

        :param columns: database columns to return, defaults to None
        :type columns: sequence of str, optional
        :param include_missing: True to include recording recording_sessions with any missing file/timing
        :type include_missing: bool
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: query result
        :rtype: pandas.DataFrame

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values

        """

        # work on a copy of the dataframe
        df = self._df.copy(deep=True)

        if not include_missing:
            # remove any recording session missing files or timings
            n0 = self._df_timing["N0"].groupby("ID").min()
            missing_ids = self._mi_miss.get_level_values("ID").union(n0[n0 < 0].index)
            df = df.drop(index=missing_ids)

        df_dx = self._df_dx
        if "Pathologies" in filters:
            incl_dx = []
            excl_dx = []
            for dx in filters["Pathologies"]:
                if dx.startswith("-"):
                    excl_dx.append(dx)
                else:
                    incl_dx.append(dx[1:] if dx[0] == "+" else dx)

            # filter df_dx
            if len(incl_dx):
                ids = df_dx.loc[df_dx["pathology"].isin(incl_dx), "ID"].unique()
                if len(excl_dx):
                    df_dx = df_dx.loc[df_dx["ID"].isin(ids)]
            else:
                ids = df_dx["ID"].unique()
            if len(excl_dx):
                ids = df_dx.loc[~df_dx["pathology"].isin(excl_dx), "ID"].unique()

            if "Pathologies" in columns:
                df_dx = df_dx.loc[df_dx["ID"].isin(ids)]

            df = df.loc[ids]

        if columns and "Pathologies" in columns:
            # add diagnoses column to df
            df["Pathologies"] = self._df_dx.groupby("ID").transform(
                lambda x: ", ".join(x)
            )

        # apply the filters to reduce the rows
        for fcol, fcond in filters.items():
            if fcol == "Pathologies":
                continue

            try:
                if fcol == "ID":
                    s = df.index
                else:
                    s = df[fcol]
            except:
                raise ValueError(f"{fcol} is not a valid column label (check cases)")

            try:  # try range/multi-choices
                if s.dtype.kind in "iufcM":  # numeric/date
                    # 2-element range condition
                    df = df[(s >= fcond[0]) & (s < fcond[1])]
                else:  # non-numeric
                    df = df[s.isin(fcond)]  # choice condition
            except:
                # look for the exact match
                df = df[s == fcond]

        # return only the selected columns
        if columns is not None:
            try:
                df = df[columns]
            except:
                ValueError(
                    f'At least one label in the "columns" argument is invalid: {columns}'
                )

        return df

    def get_files(
        self,
        task: TaskType,
        egg: bool = False,
        cached_only: bool = False,
        paths_only: bool = False,
        auxdata_fields: List[DataField] = None,
        **filters,
    ):  # -> pd.DataFrame:
        """get audio files

        :param task: recorded task
        :type task: TaskType
        :param egg: True for EGG False for audio, defaults to False
        :type egg: bool, optional
        :param cached_only: True to disallow downloading, defaults to False
        :type cached_only: bool, optional
        :param paths_only: True to return only file paths without timing for vowels, defaults to False
        :type paths_only: bool, optional
        :param auxdata_fields: List of auxiliary data fields, defaults to None
        :type auxdata_fields: List[DataField], optional
        :return: data frame containing file path, start and end time marks, and auxdata
        :rtype: pandas.DataFrame

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values
        """

        if task not in tuple(task2key.keys()):
            raise ValueError(
                f"invalid task ('{task}'): must be one of {sorted(task2key.keys())}"
            )

        # retrieve auxdata and filter if conditions given
        filter_on = len(filters) or bool(auxdata_fields)
        df = (
            self.query(auxdata_fields, **filters)
            if filter_on or bool(auxdata_fields)
            else self._df
        )

        need_timing = task not in ("iau", "phrase")
        file = ("iau" if need_timing else task, egg)

        if not cached_only:
            # check for missing ids in cache
            ids = set(df.index)

            # remove the known missing files
            try:
                mi = self._mi_miss
                mi = mi[(mi.isin(file[0:1], 0) & mi.isin(file[1:2], 1))]
                ids -= set(mi)
            except:
                pass

            # remove already cached files
            try:
                # check for the file ids
                try:
                    ids_ok = set(self._df_files.loc[file].index)
                except KeyError:
                    # none avail
                    ids_ok = set()

                if need_timing and len(ids_ok):
                    # make sure id also in timing data
                    try:
                        # ok only if in timing
                        ids_ok = ids_ok & set(
                            self._df_timing.loc[
                                (slice(None), task), :
                            ].index.get_level_values(0)
                        )

                    except KeyError:
                        # none avail
                        ids_ok = set()

                ids -= ids_ok
            except:
                pass

            if len(ids):
                self._download_groups(
                    list(ids),
                    need_timing or task == "iau",
                    task == "phrase",
                    need_timing,
                    nsp=not egg,
                    egg=egg,
                )

        try:
            # get cached files
            df_files = self._df_files.loc[
                ("iau" if need_timing else task, egg)
            ].to_frame()
        except:
            raise RuntimeError(f"cannot find cached matching files")

        if paths_only:
            need_timing = False

        if need_timing:
            try:
                self._df_timing
                df_timing = self._df_timing.loc[(slice(None), task), :].droplevel(1)
            except:
                df_timing = pd.DataFrame(
                    index=self._df_timing.index.levels[0],
                    columns=self._df_timing.columns,
                    dtype="Int64",
                )
            df_files = df_files.join(df_timing, how="inner" if cached_only else "left")

        # if filtered, remove excluded ID's
        if filter_on:
            df_files = df_files[df_files.index.isin(df.index)]

        if bool(auxdata_fields):
            # return requested info along with the file & range
            df = df_files.join(df, how="inner" if cached_only else "right")
        elif cached_only:
            df = df_files[df_files.notna().all(axis=1)]
        else:
            # no extra info, incl entries needing downloading

            new_indices = df.index[~df.index.isin(df_files.index)]
            df = pd.concat(
                [
                    df_files,
                    pd.DataFrame(
                        [("", pd.NA, pd.NA)] if need_timing else [("",)],
                        index=new_indices,
                        columns=df_files.columns,
                    ).astype(df_files.dtypes),
                ]
            ).sort_index()

        # remove missing files
        mi = self._mi_miss
        mi = mi[(mi.isin(["iau" if need_timing else task], 0) & mi.isin([egg], 1))]
        df = df[~df.index.isin(mi.get_level_values(2))]

        # expand the filepaths
        df["File"] = df["File"].map(lambda v: v and path.join(self._dir, data_dir, v))

        return df

    def iter_data(
        self,
        task: TaskType = None,
        egg: bool = False,
        cached_only: bool = False,
        auxdata_fields: List[DataField] = None,
        normalize: bool = True,
        padding: float = None,
        **filters,
    ):  # -> Iterator[Tuple[int, int, np.array, Optional[pd.Series]]]:
        """iterate over queried recording_sessions (yielding data samples)

        :param task: vocal task type, defaults to None
        :type task: TaskType, optional
        :param egg: True for EGG, False for audio, defaults to False
        :type egg: bool, optional
        :param cached_only: True to block downloading, defaults to False
        :type cached_only: bool, optional
        :param auxdata_fields: Additional recording data to return, defaults to None
        :type auxdata_fields: List[DataField], optional
        :param normalize: True to convert sample values to float between [-1.0,1.0], defaults to True
        :type normalize: bool, optional
        :return: voice data namedtuple: id, fs, data array, (optional) aux info
        :rtype: Iterator[int, int, np.array, Optional[pd.Series]]
        """

        if not task:
            task = self.default_task

        df = self.get_files(task, egg, cached_only, True, auxdata_fields, **filters)

        for id, file, *auxdata in df.itertuples():
            timing = None if task in ("iau", "phrase") else self._df_timing.loc[id]

            framerate, x = self._read_file(file, task, timing, normalize, padding)

            yield (
                (id, framerate, x)
                if auxdata_fields is None
                else (
                    id,
                    framerate,
                    x,
                    auxdata,
                )
            )

    def read_data(
        self,
        id: int,
        task: TaskType = None,
        egg: bool = False,
        cached_only: bool = False,
        auxdata_fields: List[DataField] = None,
        normalize: bool = True,
        padding: float = None,
    ):  # -> Tuple[int, np.array, Optional[pd.Series]]:
        """read specific recording_sessions

        :param task: vocal task type, defaults to None
        :type task: TaskType, optional
        :param egg: True for EGG, False for audio, defaults to False
        :type egg: bool, optional
        :param cached_only: True to block downloading, defaults to False
        :type cached_only: bool, optional
        :param auxdata_fields: Additional recording data to return, defaults to None
        :type auxdata_fields: List[DataField], optional
        :param normalize: True to convert sample values to float between [-1.0,1.0], defaults to True
        :type normalize: bool, optional
        :return: voice data namedtuple: fs, data array, (optional) aux info
        :rtype: Iterator[int, np.array, Optional[pd.Series]]
        """
        if not task:
            task = self.default_task

        # get the file name
        try:
            file = self.get_files(
                task, egg, cached_only, True, auxdata_fields, ID=id
            ).loc[id]
            assert file[0]
        except:
            raise ValueError(f"{id}:{task} data is not available.")

        timing = None if task in ("iau", "phrase") else self._df_timing.loc[id]
        data = self._read_file(file.loc["File"], task, timing, normalize, padding)

        return data if auxdata_fields is None else (*data, file.iloc[1:])

    def _read_file(self, file, task, timing, normalize=True, padding=None):
        fs, x = nspfile.read(file)

        if timing is not None:
            if not padding:
                padding = self.default_padding

            if padding:
                # sort timing by N0
                ts = timing.sort_values("N0")
                i = ts.index.get_loc(task)

                padding = round(padding * fs)
                tstart, tend = ts.iloc[i]
                tstart -= padding
                tend += padding

                if padding > 0.0:
                    if i > 0:
                        i0 = i - 1
                        while ts.iloc[i, 0] < ts.iloc[i0, 1]:
                            i0 -= 1
                            if i0 < 0:
                                raise RuntimeError(
                                    f"something is wrong with timing data for id={id} ({task})"
                                )
                        talt = ts.iloc[i0, 1]
                        if tstart < talt:
                            tstart = talt

                    i0 = i + 1
                    if i0 < len(ts):
                        while ts.iloc[i, 1] > ts.iloc[i0, 0]:
                            i0 += 1
                            if i0 == len(ts):
                                raise RuntimeError(
                                    f"something is wrong with timing data for id={id} ({task})"
                                )
                        talt = ts.iloc[i0, 0]
                        if tend > talt:
                            tend = talt
            else:
                tstart, tend = timing.loc[task]

            x = x[tstart:tend]

        if normalize:
            x = x / 2.0**15

        return fs, x

    def __getitem__(self, key: int) -> Tuple[int, np.array]:
        return self.read_data(key)
