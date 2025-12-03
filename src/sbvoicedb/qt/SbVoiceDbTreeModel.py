"""Qt tree model to interface with SbVoiceDb SQLite database"""

from __future__ import annotations

import logging

from typing_extensions import Any, Sequence, override, cast, Literal, Protocol

from qtpy.QtCore import QAbstractItemModel, QModelIndex, QObject, Qt

from .. import (
    SbVoiceDb,
    Recording,
    RecordingSession,
    Speaker,
    PathologyLiteral,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SbVoiceDbTreeLevelLiteral = Literal["pathology", "speaker", "session", "recording"]


class DbIdLister(Protocol):
    def __call__(self, db_id: int) -> Sequence[int]: ...


class DbItemGetter(Protocol):
    def __call__(self, db_id: int, **kwargs) -> Any: ...


class SbVoiceDbTreeModel(QAbstractItemModel):
    _db: SbVoiceDb
    _ids: list[tuple[int, int]]
    """[db id, parent gid], to find db id's from tree's unique gid (parent=-1 == root)"""
    _gids: dict[tuple[int, int], int]
    """dict key=((db id, parent gid), top->bottom level), value=gid, to find tree's gids (parent=-1 == root)"""
    _child_ids: dict[int, tuple[int, ...]]
    """dict key=parent gid (-1 for the root), value=list of db ids of its children"""

    _levels: tuple[SbVoiceDbTreeLevelLiteral]
    """tree structure, trunk->leaf"""
    _maxlevel: int
    """tree level of the stem"""
    _lister: list[DbIdLister]
    """per-level callback to return a function to list all nodes given a parent db id"""
    _item_getter: list[DbItemGetter]
    """per-level callback to return database ORM item"""

    # row index -> db table row id

    def __init__(
        self,
        db: SbVoiceDb,
        tree_levels: Sequence[SbVoiceDbTreeLevelLiteral] | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._db = db
        self._ids = []
        self._gids = {}
        self._child_ids = {}

        if tree_levels is None or len(tree_levels) == 0:
            # default levels
            # tree_levels = ["session", "recording"]
            # tree_levels = ["speaker", "session", "recording"]
            tree_levels = ["pathology", "session", "recording"]

        self._levels = tuple(tree_levels)
        self._lister = []
        self._item_getter = []
        for target, parent in zip(tree_levels, [None, *tree_levels[:-1]]):
            lister, getter = self._create_accessors(target, parent)
            self._lister.append(lister)
            self._item_getter.append(getter)
        self._maxlevel = len(tree_levels) - 1

    ### Index Conversion Methods ###

    @staticmethod
    def _tree_level(index: QModelIndex):
        level = -1
        while index.isValid():
            level += 1
            index = index.parent()
        return level

    def _db_id_to_gid(self, db_id: int, pid: int) -> int:
        key = (db_id, pid)
        try:
            return self._gids[key]
        except KeyError:
            # create a new entry
            gid = len(self._ids)
            self._ids.append(key)
            self._gids[key] = gid
            return gid

    def treeLevels(self) -> tuple[str, ...]:
        return self._levels

    def database(self) -> SbVoiceDb:
        return self._db

    @override
    def index(
        self, row: int, column: int = 0, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Returns the index of the item in the model specified by the given row, column and parent index

        :param row: _description_
        :param column: _description_, defaults to 0
        :param parent: _description_, defaults to QModelIndex()
        :return: _description_
        """

        if self._db is None or not self.hasIndex(row, column, parent):
            return QModelIndex()

        # get parent's child list (which must exist)
        level = self._tree_level(parent)
        pid = -1 if level < 0 else parent.internalId()
        children = self._child_ids[pid]  # assume prefilled earlier
        gid = self._db_id_to_gid(children[row], pid)

        # returns the Qt index object with a unique identifier (index to self._ids)
        return self.createIndex(row, column, gid)

    @override
    def parent(self, child: QModelIndex = QModelIndex()) -> QModelIndex:
        """Returns the parent of the child item with the given index. If the item has no parent, an invalid QModelIndex is returned.

        :param child: _description_, defaults to None
        :return: _description_
        """

        gid = child.internalId()
        pid = self._ids[gid][1]  # parent's gid
        if pid < 0:  # top-level
            return QModelIndex()

        # find the row of the parent
        dbpid, gpid = self._ids[pid]  # parent db ID & grand-parent GID
        row = self._child_ids[gpid].index(dbpid)

        return self.createIndex(row, 0, pid)

    @override
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Returns the number of rows under the given parent.

        When the parent is valid it means that rowCount is returning the number
        of children of parent.

        :param parent: _description_, defaults to QModelIndex()
        :return: _description_
        """

        if self._db is None:
            return 0

        gid = parent.internalId() if parent.isValid() else -1
        try:
            cids = self._child_ids[gid]
        except KeyError:
            # get parent's tree level
            level = self._tree_level(parent)
            try:
                # get the lister function for parent's child type
                lister = self._lister[level + 1]
            except IndexError:
                # parent is a leaf node, no children
                return 0
            else:
                # get the child id's and save them for later use
                try:
                    pid = self._ids[gid][0]
                except IndexError:
                    pid = 0  # root, pid ignored
                self._child_ids[gid] = cids = tuple(lister(pid))

        # return the length of the child list
        return len(cids)

    @override
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Returns the number of columns for the children of the given parent.

        :param parent: _description_, defaults to ...
        :return: _description_
        """
        return 1

    @override
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Returns the data stored under the given role for the item referred to by the index.

        :param index: _description_
        :param role: _description_
        """

        gid = index.internalId()
        dbid = self._ids[gid][0]
        level = self._tree_level(index)
        item = self._item_getter[level](dbid)

        if isinstance(item, Recording):
            # session row
            if role == Qt.ItemDataRole.DisplayRole:
                return f"{item.id:05d}:{item.utterance}:{item.length/item.rate:0.2f} s"
            # elif role == Qt.ItemDataRole.ToolTipRole:
            #     return f"{session.id}:{', '.join(session.pathologies)}\nspeaker={session.speaker_id}:
        elif isinstance(item, RecordingSession):
            # recordings row
            if role == Qt.ItemDataRole.DisplayRole:
                return f"{item.id:04d}:age={item.speaker_age:02d}:type={item.type}"
            # elif role == Qt.ItemDataRole.ToolTipRole:
            #     return self._db.get_segment_description(recording_id)
        elif isinstance(item, Speaker):
            # recordings row
            if role == Qt.ItemDataRole.DisplayRole:
                return f"{item.id:04d}:gender={item.gender}:birthdate={item.birthdate}"
            # elif role == Qt.ItemDataRole.ToolTipRole:
            #     return self._db.get_segment_description(recording_id)
        elif isinstance(item, str):  # pathology name in German
            # recordings row
            if role == Qt.ItemDataRole.DisplayRole:
                return item
            # elif role == Qt.ItemDataRole.ToolTipRole:
            #     return self._db.get_segment_description(recording_id)

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Returns the item flags for the given index.

        :param index: _description_
        :return: _description_
        """

        base_flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

        if self._tree_level(index) == self._maxlevel:  # leaf level
            return base_flags | Qt.ItemFlag.ItemNeverHasChildren

        return base_flags

    # @override
    # def headerData(
    #     self,
    #     section: int,
    #     orientation: Qt.Orientation,
    #     role: int = Qt.ItemDataRole.DisplayRole,
    # ) -> Any:
    #     """Returns the data for the given role and section in the header with the specified orientation.

    #     :param section: _description_
    #     :param orientation: _description_
    #     :param role: _description_, defaults to Qt.ItemDataRole.DisplayRole
    #     :return: _description_
    #     """

    #     return (
    #         "Name"
    #         if orientation == Qt.Orientation.Horizontal
    #         and role == Qt.ItemDataRole.DisplayRole
    #         else None
    #     )

    def _find_item(self, index: QModelIndex, level: int) -> int | None:
        gid = index.internalId()
        db_id, pid = self._ids[gid]
        i = self._tree_level(index)
        if i < level:
            # above the level of the requested item
            return None

        while i != level:
            i -= 1
            db_id, pid = self._ids[pid]
        return db_id

    def pathology(self, index: QModelIndex) -> str | None:
        """return the name of the pathology if pathology is included in the tree

        :param index: current index
        :return: pathology name in German
        """

        try:
            level = self._levels.index("pathology")
        except IndexError:
            return None

        db_id = self._find_item(index, level)
        return None if db_id is None else cast(str, self._item_getter[level](db_id))

    def speaker(self, index: QModelIndex) -> Speaker | None:

        try:
            level = self._levels.index("speaker")
        except IndexError:
            return None

        db_id = self._find_item(index, level)
        return None if db_id is None else cast(str, self._item_getter[level](db_id))

    def session(self, index: QModelIndex) -> RecordingSession | None:

        try:
            level = self._levels.index("session")
        except IndexError:
            return None

        db_id = self._find_item(index, level)
        return None if db_id is None else cast(str, self._item_getter[level](db_id))

    def recording(
        self, index: QModelIndex, full_file_paths: bool = True
    ) -> Recording | None:

        try:
            level = self._levels.index("recording")
        except IndexError:
            return None

        db_id = self._find_item(index, level)
        return (
            None
            if db_id is None
            else cast(
                str, self._item_getter[level](db_id, full_file_paths=full_file_paths)
            )
        )

    def _create_accessors(
        self,
        target: SbVoiceDbTreeLevelLiteral,
        parent: SbVoiceDbTreeLevelLiteral | None,
    ) -> tuple[DbIdLister, DbItemGetter]:
        db = self._db
        if target == "pathology":
            if parent is not None:
                raise ValueError("'pathology' level can only be the top level entry.")

            return (
                (lambda _: db.get_pathology_ids(include_healthy=True)),
                db.get_pathology_name,
            )
        elif target == "speaker":
            if parent is not None:
                raise ValueError("'speaker' level can only be the top level entry.")
            return ((lambda _: db.get_speaker_ids()), db.get_speaker)
        elif target == "session":
            if parent is None:
                return (
                    (lambda _: db.get_session_ids()),
                    db.get_session,
                )
            elif parent == "pathology":
                return (
                    (lambda pathology_id: db.get_session_ids(pathologies=pathology_id)),
                    db.get_session,
                )
            elif parent == "speaker":
                return (db.get_session_ids, db.get_session)
            else:
                raise ValueError(
                    "Only the 'pathology' or 'speaker' can be the parent of 'session'."
                )
        elif target == "recording":
            if parent is None:
                return ((lambda _: db.get_recording_ids()), db.get_recording)
            elif parent == "session":
                return (db.get_recording_ids, db.get_recording)
            else:
                raise ValueError("Only the 'session' can be the parent of 'recording'.")

        raise ValueError(f"{target=} is an unknown keyword for a SbVoiceDB tree level.")

    def pathologies(self) -> list[str]:
        return list(self._db.get_pathology_names())

    def getSpeakerIndex(self, db_id: int) -> QModelIndex:

        if "speaker" != self._levels[0]:
            raise ValueError("Speaker is not used in the model.")

        key = (db_id, -1)
        try:
            gid = self._ids.index(key)
        except ValueError:
            logger.error("Invalid Speaker ID: %d", db_id)
            return QModelIndex()

        pos = self._child_ids[-1].index(db_id)
        return self.createIndex(pos, 0, gid)

    def getPathologyIndex(self, name: PathologyLiteral) -> QModelIndex:

        if "pathology" != self._levels[0]:
            raise ValueError("Pathology is not used in the model.")

        db_id = self._db.get_pathology_id(name)
        if db_id is None:
            logger.error("Invalid pathology name: %s", name)
            return QModelIndex()

        key = (db_id, -1)
        try:
            gid = self._ids.index(key)
        except ValueError:
            logger.error("Invalid Speaker ID: %d", db_id)
            return QModelIndex()

        pos = self._child_ids[-1].index(db_id)
        return self.createIndex(pos, 0, gid)

    def getSessionIndex(self, db_id: int) -> QModelIndex:

        try:
            i = self._levels.index("session")
        except ValueError as e:
            raise ValueError("RecordingSession is not used in the model.") from e

        # get parent gid
        pid = -1
        if i > 0:  # 2nd level
            session = self._db.get_session(db_id)
            if session is None:
                logger.error("Invalid RecordingSession ID: %d (not in DB)", db_id)
                return QModelIndex()

            parent_ids = self._child_ids[-1]
            if self._levels[i - 1] == "speaker":
                pid = session.speaker_id
            else:
                # may be associated with multiple pathologies
                # return the first match
                pid = next(
                    parent_ids.index(patho.id)
                    for patho in session.pathologies
                    if patho.id in parent_ids
                )

            # convert db ID to GID
            pid = self._gids[(pid, -1)]

        key = (db_id, pid)
        try:
            gid = self._gids[key]
        except KeyError:
            logger.error("Invalid RecordingSession ID: %d (not in model)", db_id)
            return QModelIndex()

        pos = self._child_ids[pid].index(db_id)
        return self.createIndex(pos, 0, gid)

    def getRecordingIndex(self, db_id: int) -> QModelIndex:

        try:
            i = self._levels.index("recording")
        except ValueError as e:
            raise ValueError("Recording is not used in the model.") from e

        # get parent gid
        pid = -1
        if i > 0:  # stem/leaf level
            rec = self._db.get_recording(db_id)
            if rec is None:
                logger.error("Invalid Recording ID: %d (not in DB)", db_id)
                return QModelIndex()
            pid = self.getSessionIndex(rec.session_id).internalId()

        key = (db_id, pid)
        try:
            gid = self._gids[key]
        except KeyError:
            logger.error("Invalid Recording ID: %d (not in model)", db_id)
            return QModelIndex()

        pos = self._child_ids[pid].index(db_id)
        return self.createIndex(pos, 0, gid)

    def getDbDataIds(self, index: QModelIndex) -> dict[str, int]:
        indices = [index]
        while (index := index.parent()).isValid():
            indices.append(index)

        ids = self._ids
        return {
            level: ids[index.internalId()][0]
            for level, index in zip(self._levels, indices[::-1])
        }
