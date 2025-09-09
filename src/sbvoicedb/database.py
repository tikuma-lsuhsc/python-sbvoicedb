"""Saarbrueken Voice Database Reader module"""

from __future__ import annotations

import logging
from datetime import datetime
import glob
from os import path, makedirs
import re
import unicodedata

from typing import Literal, List, Sequence, cast, Iterator

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
    PrimaryKeyConstraint,
    String,
    Date,
    create_engine,
    select,
    insert,
    update,
    func,
    Table,
    UniqueConstraint,
    Select,
    and_,
    or_,
)
from sqlalchemy.sql.expression import ColumnElement

from nspfile import read as nspread, NSPHeaderDict
import tqdm

from .download import download_data, download_database
from .utils import fix_incomplete_nsp, swap_nsp_egg

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
    downloaded: Mapped[bool] = mapped_column(default=False)

    sessions: Mapped[List["RecordingSession"]] = relationship(
        secondary=pathology_in_recordings,
        primaryjoin=id == pathology_in_recordings.c.pathology_id,
    )


class Speaker(Base):
    """metadata of the `speakers` table

    :param Base: _description_
    """    
    __tablename__ = "speakers"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True)
    gender: Mapped[Literal["m", "w"]] = mapped_column(String(1))
    birthdate: Mapped[str] = mapped_column(Date)

    sessions: Mapped[List["RecordingSession"]] = relationship(
        back_populates="speaker",
        primaryjoin="Speaker.id==RecordingSession.speaker_id",
    )


class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    __table_args__ = ()
    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id"))
    speaker_age: Mapped[int] = mapped_column()
    type: Mapped[Literal["n", "p"]] = mapped_column(String(1))
    note: Mapped[str] = mapped_column()

    speaker: Mapped[Speaker] = relationship(
        back_populates="sessions",
        primaryjoin="Speaker.id==RecordingSession.speaker_id",
    )
    pathologies: Mapped[List[Pathology]] = relationship(
        secondary=pathology_in_recordings,
        overlaps="sessions",
        primaryjoin=id == pathology_in_recordings.c.session_id,
    )
    recordings: Mapped[List["Recording"]] = relationship(
        back_populates="session",
        primaryjoin="Recording.session_id==RecordingSession.id",
    )


class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = (UniqueConstraint("session_id", "utterance"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("recording_sessions.id"))
    utterance: Mapped[str] = mapped_column(String(5))
    rate: Mapped[int] = mapped_column()
    length: Mapped[int] = mapped_column()
    nspfile: Mapped[str] = mapped_column()
    eggfile: Mapped[str] = mapped_column()

    session: Mapped[RecordingSession] = relationship(
        back_populates="recordings",
        primaryjoin="Recording.session_id==RecordingSession.id",
    )


class SbVoiceDb:
    """SbVoiceDb class

    Constructor Arguments

    :param dbdir: databse directory
    :param download_mode: True to use only cached files, defaults to False
    """

    _datadir: str
    _dbdir: str
    _db: Engine
    _download_allowed: bool = True

    _speaker_filter: ColumnElement | None = None
    _session_filter: ColumnElement | None = None
    _pathology_filter: ColumnElement | None = None
    _recording_filter: ColumnElement | None = None
    _include_normal: bool = False  # True to include normals with pathology filter

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

        if not path.exists(self._datadir):
            makedirs(self._datadir)

        db_path = path.join(dbdir, "sbvoice.db")
        db_exists = path.exists(db_path)

        self._db = create_engine(
            f"sqlite+pysqlite:///{db_path}",
            connect_args=connect_kws or {},
            echo=echo,
            **create_engine_kws,
        )

        Base.metadata.create_all(self._db)

        self._download_allowed = download_mode is not False
        try_download = download_mode == "once"

        if not db_exists:
            # if new database file is just created, populate it (except for recordings)
            self._populate_db()
            incomplete, existing_pathos = self._mark_downloaded()
            if len(existing_pathos):
                # if any portion of dataset already exists, populate the DB with its recording info
                self._populate_recordings(pathology=existing_pathos)

            try_download = try_download and incomplete  # only download if data missing

        if try_download:
            # download full dataset if download_mode is 'once'
            self.download_data()

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
                        pathology_in_recordings.insert().values(
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
            session.add(
                Setting(
                    group="",
                    key="healthy_downloaded",
                    value=session_id is not None,
                )
            )
            for pid in session.scalars(select(Pathology.id)):
                session_id = session.scalar(
                    select(pathology_in_recordings.c.session_id)
                    .where(pathology_in_recordings.c.pathology_id == pid)
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

    def _populate_recordings(self, pathology: str | Sequence[str] | None = None):
        """scan directory and populate recordings table

        :param pathology: populate only the recordings associated with the specified pathology, defaults to None
        """

        # build SQL statement to retrieve recording session id's
        stmt = select(RecordingSession.id)
        if pathology is not None:
            # add where clasuse to specify pathologies or healthy
            has_healthy = pathology == "healthy"
            if has_healthy:
                pathology = []
            else:
                try:
                    i = pathology.index("healthy")
                except ValueError:
                    pass
                else:
                    pathology = [patho for j, patho in enumerate(pathology) if i != j]
                    has_healthy = True

            condition = RecordingSession.pathologies.any(
                Pathology.name == pathology
                if isinstance(pathology, str)
                else Pathology.name.in_(pathology)
            )
            if has_healthy:
                condition = or_(condition, RecordingSession.type == "n")
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

    @property
    def number_of_all_sessions(self) -> int:
        """total number of recording sessions in the database"""
        with Session(self._db) as session:
            return session.scalar(select(func.count(RecordingSession.id)))

    @property
    def number_of_sessions_downloaded(self) -> int:
        """number of recording sessions already downloaded"""

        with Session(self._db) as session:
            count = sum(
                session.scalars(
                    select(func.count(RecordingSession.id)).where(
                        RecordingSession.pathologies.any(Pathology.downloaded)
                    )
                )
            )
            if (
                session.scalar(
                    select(Setting.value).where(
                        and_(Setting.group == "", Setting.key == "healthy_downloaded")
                    )
                )
                == "1"
            ):
                count += (
                    session.scalar(
                        select(func.count(RecordingSession.id)).where(
                            RecordingSession.type == "n"
                        )
                    )
                    or 0
                )
            return count

    def download_data(self, session_id: int | None = None) -> bool:
        """download minimal dataset required for current filter configurations"""

        if session_id is None:
            sessions = cast(
                dict[str | None, Sequence[int]],
                self.get_session_ids_of_all_pathologies(use_name=True),
            )
        else:
            # if session specified, download its first pathology zip or healthy.zip
            pathos = self.get_pathologies_in_session(session_id)
            sessions = {pathos[0] if len(pathos) else "healthy": [session_id]}

        # find a number of missing recording sessions per pathology
        counts = {
            patho_name: len(session_ids)
            for patho_name, session_ids in sessions.items()
            if any(
                self._get_recording_count(session_id) == 0 for session_id in session_ids
            )
        }

        total_sessions = sum(counts.values())

        if total_sessions == 0:
            return False  # no download necessary

        all_sessions = self.number_of_all_sessions
        if self._download_allowed:
            if total_sessions > all_sessions:
                # duplicates in the pathologies are too large to warrant pathology-wise download
                download_data(self._datadir)
                self._populate_recordings()

            else:
                # per-pathology download
                for patho_name in counts:
                    name = patho_name
                    download_data(self._datadir, name)
                    self._populate_recordings(name)
            return True
        else:
            logger.warning(
                "At most %d recording sessions are missing from the current dataset. "
                'Instantiate SbVoiceDb object with `download_mode` argument set to either "once" or "incremental" '
                "to enable the automatic downloading feature.",
                min(total_sessions, all_sessions),
            )
            return False

    def _mark_downloaded(self) -> tuple[bool, list[str]]:
        """scan the data folder and mark downloaded flags

        :returns incomplete: True if dataset is incomplete
        :returns existing_pathologies: list of existing pathologies
        """

        patho_sessions = self.get_session_ids_of_all_pathologies()

        incomplete = False
        patho_list = []

        with Session(self._db) as session:
            for patho in session.scalars(select(Pathology)):

                sessions = patho_sessions[patho.id]
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
                else:
                    incomplete = True

            sessions = session.scalars(
                select(RecordingSession.id).where(RecordingSession.type == "n")
            )
            downloaded = all(
                path.exists(path.join(self._datadir, str(session)))
                for session in sessions
            )
            session.execute(
                update(Setting)
                .where(and_(Setting.group == "", Setting.key == "healthy_downloaded"))
                .values({"value": downloaded})
            )
            if downloaded:
                patho_list.append("healthy")
            else:
                incomplete = True

            session.commit()

        return incomplete, patho_list

    ################

    def set_speaker_filter(self, where_clause: ColumnElement | None):
        self._speaker_filter = where_clause

    def set_session_filter(self, where_clause: ColumnElement | None):
        self._session_filter = where_clause

    def set_pathology_filter(
        self, where_clause: ColumnElement | None, include_normal: bool = False
    ):
        self._pathology_filter = where_clause
        self._include_normal = include_normal

    def set_recording_filter(self, where_clause: ColumnElement | None):
        self._recording_filter = where_clause

    ################

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
            and_(*session_filters)
            if nsfilt > 1
            else session_filters[0] if nsfilt else None
        )
        if session_filter is not None:
            stmt = stmt.where(Pathology.sessions.any(session_filter))

        return stmt

    def get_pathology_count(self) -> int:
        """Return the number of unique pathologies"""
        with Session(self._db) as session:
            return session.scalar(self._pathology_select(func.count(Pathology.id))) or 0

    def get_pathologies(self) -> Sequence[Pathology]:
        with Session(self._db) as session:
            return session.scalars(self._pathology_select(Pathology)).fetchall()

    def iter_pathologies(self) -> Iterator[Pathology]:
        with Session(self._db) as session:
            for patho in session.scalars(self._pathology_select(Pathology)):
                yield patho

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
                    Pathology.name, exclude_related_filters=True
                ).where(Pathology.session.has(RecordingSession.id == session_id))
            ).fetchall()

    ################

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
                f = or_(RecordingSession.type == "n", f)
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
            stmt = stmt.where(Speaker.sessions.any(session_filter))

        return stmt

    def get_speaker_count(self) -> int:
        """Return the number of unique speakers"""
        with Session(self._db) as session:
            return session.scalar(self._speaker_select(func.count(Speaker.id))) or 0

    def get_speaker_ids(self) -> Sequence[int]:
        """Return an id list of all the unique speakers"""
        with Session(self._db) as session:
            return session.scalars(
                self._speaker_select(Speaker.id).order_by(Speaker.id)
            ).fetchall()

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
                    session.scalars(
                        self._speaker_select(Speaker.id).order_by(Speaker.id)
                    )
                    .all()
                    .index(speaker_id)
                )
            except ValueError:
                return None

    def get_speaker(self, speaker_id: int) -> Speaker | None:
        with Session(self._db) as session:
            return session.scalar(select(Speaker).where(Speaker.id == speaker_id))

    def iter_speakers(self) -> Iterator[Speaker]:
        """iterate over speakers"""

        with Session(self._db) as session:
            for row in session.scalars(self._speaker_select(Speaker)):
                yield row

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
        if self._pathology_filter is not None:
            if not exclude_pathology_filter:
                f = RecordingSession.pathologies.any(self._pathology_filter)
                if self._include_normal:
                    f = or_(f, RecordingSession.type == "n")
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
                select(RecordingSession).where(RecordingSession.id == session_id)
            )

    def get_session_ids_with_pathology(
        self, pathology: str | int | None
    ) -> Sequence[int]:
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

        keys: dict[int | None, str | int | None] = (
            {patho.id: patho.name for patho in self.get_pathologies()}
            if use_name
            else {patho_id: patho_id for patho_id in self.get_pathology_ids()}
        )
        if self._pathology_filter is None or self._include_normal is True:
            keys[None] = "healthy" if use_name else None

        return {v: self.get_session_ids_with_pathology(k) for k, v in keys.items()}

    def get_session_ids_of_speaker(self, speaker_id: int) -> Sequence[RecordingSession]:
        with Session(self._db) as session:
            return session.scalars(
                self._session_select(
                    RecordingSession.id, exclude_speaker_filter=True
                ).where(RecordingSession.speaker_id == speaker_id)
            ).fetchall()

    def iter_sessions(self) -> Iterator[RecordingSession]:
        with Session(self._db) as session:
            for sess in session.scalars(self._session_select(RecordingSession)):
                yield sess

    ################

    def _recording_select(
        self, *args, exclude_related_filters: bool = False, **kwargs
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

    def _get_recording_count(self, session_id: int | None = None) -> int:
        """Return the number of unique recording recordings (no download)"""

        stmt = self._recording_select(
            func.count(Recording.id),
            exclude_related_filters=session_id is not None,
        )

        if session_id is not None:
            stmt = stmt.where(Recording.session.has(RecordingSession.id == session_id))

        with Session(self._db) as session:
            return session.scalar(stmt) or 0

    def get_recording_count(self, session_id: int | None = None) -> int:
        """Return the number of unique recording recordings"""
        count = self._get_recording_count(session_id)

        if not count:
            self.download_data()
            count = self._get_recording_count(session_id)

        return count

    def get_recording_ids(self) -> Sequence[int]:
        """Return an id list of all the unique recordings"""
        with Session(self._db) as recording:
            return recording.scalars(self._recording_select(Recording.id)).fetchall()

    def get_recording_id(self, index: int) -> int | None:
        with Session(self._db) as session:
            rid = session.scalar(
                self._recording_select(Recording.id)
                .order_by(Recording.id)
                .offset(index)
                .limit(1)
            )

        return rid

    def get_recording_index(self, recording_id: int) -> int | None:
        with Session(self._db) as session:
            try:
                return (
                    session.scalars(self._recording_select(Recording.id))
                    .all()
                    .index(recording_id)
                )
            except ValueError:
                return None

    def get_recording(
        self, recording_id: int, full_file_paths: bool = False
    ) -> Recording | None:
        with Session(self._db) as session:
            rec = session.scalar(select(Recording).where(Recording.id == recording_id))

        if rec is not None and full_file_paths:
            rec.nspfile = path.join(self._datadir, rec.nspfile)
            rec.eggfile = path.join(self._datadir, rec.eggfile)
        return rec

    def get_recordings_in_session(self, session_id: int) -> Sequence[Recording]:
        with Session(self._db) as session:
            recs = session.scalars(
                self._recording_select(Recording, exclude_related_filters=True).where(
                    Recording.session_id == session_id
                )
            ).fetchall()

        return recs

    def iter_recordings(self, full_file_paths: bool = False) -> Iterator[Recording]:
        """iterate over recordings

        :param full_file_paths: True to expand the NSP and EGG file paths to full path, defaults to False
        :yield: ``Recording`` object
        """

        with Session(self._db) as session:
            for rec in session.scalars(self._recording_select(Recording)):
                if full_file_paths:
                    rec.nspfile = path.join(self._datadir, rec.nspfile)
                    rec.eggfile = path.join(self._datadir, rec.eggfile)
                yield rec
