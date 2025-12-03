from __future__ import annotations

import logging
from bisect import bisect_left
from typing import Any, Callable, Literal, Sequence, get_args
from qtpy.QtCore import (
    QAbstractTableModel,
    QObject,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
)

from sqlalchemy import Row

from ..database import SbVoiceDb, RecordingSummaryColumn, UtteranceLiteral, Recording

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SummaryTableModel(QAbstractTableModel):

    __summary_name__: str

    _db: SbVoiceDb
    _db_method: Callable
    _sql_columns: Literal["*"] | list[str] | dict[str, str] = "*"
    _columns: list[str]
    _filter_kws: dict[str, Any]  # set by subclasses
    _sort_arg: list[str | tuple[str, Literal["asc", "desc"]]] | None = None

    _batch_size: int = 128
    _row_cache: dict[int, list[Row]]
    _row_count: int = 0
    _refetch: bool = True

    def __init__(
        self,
        db: SbVoiceDb,
        batch_size: int | None = None,
        parent: QObject | None = None,
    ):
        """Base class to display SbVoiceDb's summary views in Qt Framework

        :param db: database object
        :param batch_size: number of rows to fetch at a time from `db`, defaults to 128
        :param parent: parent object, defaults to None
        """
        super().__init__(parent)

        self.setDatabase(db)

        if batch_size is not None:
            self._batch_size = batch_size

        self._columns = []
        self._filter_kws = {}
        self._row_cache = {}  # TODO consider cachetools.LRUCache

    def database(self) -> SbVoiceDb:
        """Returns the model's SbVoiceDb database object"""
        return self._db

    def setDatabase(self, db: SbVoiceDb):
        """Sets the model's SbVoiceDb database object"""
        self.beginResetModel()
        self._db = db
        self._db_method = getattr(db, self.__summary_name__)
        self._refetch = True
        self.endResetModel()

    def columns(self) -> Literal["*"] | list[str] | dict[str, str]:
        """Returns the summary view's columns to be included in the model.

        "*" indicates all columns. A `dict` maps that column names of the view
        (keys) to those of the model (values).
        """
        return self._sql_columns

    def setColumns(self, columns: Literal["*"] | list[str] | dict[str, str]):
        """Sets the summary view's columns to be included in the model.

        To include all the columns of the view as is, use '*'.

        To use different column names for the model than the view's columns,
        set it to a `dict` with view's column names as the keys and the model's
        column names as the values.
        """

        self.beginResetModel()
        self._sql_columns = columns
        self._columns = []
        self._refetch = True
        self.endResetModel()

    def getColumnIndexLookup(self) -> dict[str, int]:
        """Returs a dict to map the view's column name to model's column index"""
        # true if table columns received new names
        lookup = isinstance(self._sql_columns, dict)

        return (
            {k: i for i, k in enumerate(self._sql_columns.keys())}
            if lookup
            else {k: i for i, k in enumerate(self._columns)}
        )

    def getColumnNameLookup(self) -> dict[str, str]:
        """Returns a dict to map the view's column name to model's column name"""

        # true if table columns received new names
        return (
            self._sql_columns
            if isinstance(self._sql_columns, dict)
            else {k: k for k in self._columns}
        )

    def rowCount(
        self, /, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return self._row_count

    def columnCount(
        self, /, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return len(self._columns)

    def canFetchMore(self, parent: QModelIndex | QPersistentModelIndex) -> bool:
        """Returns True if query has been changed"""
        return self._refetch

    def fetchMore(self, parent: QModelIndex | QPersistentModelIndex) -> None:
        """Called once to initialize/resize the table"""

        self.beginResetModel()

        # get columns if currently unknown
        if len(self._columns) == 0:
            self._columns = self._db_method(
                columns=self._sql_columns, order_by="id", limit=1
            )[0]._fields
            logger.info(
                "SummaryTableModel.fetchMore: setting columns (%d) = %s",
                len(self._columns),
                str(self._columns),
            )

        # get the row count
        self._row_count = self._db_method("COUNT(id)", **self._filter_kws)[0][0]
        logger.info(
            "SummaryTableModel.fetchMore: number of rows = %d",
            self._row_count,
        )

        # clear the cache
        self._row_cache.clear()
        self._refetch = False

        self.endResetModel()

    def data(
        self, index: QModelIndex | QPersistentModelIndex, /, role: int = Qt.DisplayRole
    ) -> Any:
        if index.isValid():
            if role == Qt.DisplayRole:
                return self.fetchRow(index.row())[index.column()]
            elif role == Qt.ToolTipRole:
                return self._fetch_tooltip(index.row())

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        /,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:

        if orientation == Qt.Orientation.Horizontal and role in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            if self.columnCount() == 0:
                self.fetchMore(QModelIndex())

            return self._columns[section]

        return super().headerData(section, orientation, role)

    def fetchRow(self, row: int) -> Row:
        """fetch a data for the specified row, query new batch as needed"""

        i, j = divmod(row, self._batch_size)

        try:
            batch = self._row_cache[i]
        except KeyError:
            # requested batch not available, get them from the database
            batch = self._db_method(
                self._sql_columns,
                **self._filter_kws,
                order_by=self._sort_arg,
                limit=self._batch_size,
                offset=i * self._batch_size,
            )
            self._row_cache[i] = batch

        return batch[j]

    def _fetch_tooltip(self, row: int) -> str | None:

        data: Row = self.fetchRow(row)
        lookup = isinstance(self._sql_columns, dict)
        lut = {v: k for k, v in self._sql_columns.items()} if lookup else None

        field_lut = {
            (lut.get(field, None) if lookup else field): field for field in data._fields
        }

        field_keys = {
            k: field_lut.get(k, None) for k in get_args(RecordingSummaryColumn)
        }
        field_values = {
            k: data._mapping[v] for k, v in field_keys.items() if v is not None
        }

        return self._compose_tooltip(field_values)

    def _compose_tooltip(self, field_values: dict[str, Any]) -> str | None:
        return None

    def reset(self):
        """clear the data to be repopulated"""
        self.beginResetModel()
        self._refetch = False
        self.endResetModel()

    def _set_or_drop_filter_key(self, key: str, value: Any):
        if value is not None:
            self._filter_kws[key] = value
        elif key in self._filter_kws:
            del self._filter_kws[key]
        self._refetch = True

    def findRow(self, view_id: int, view_column: str = "id") -> int:
        """find model's row which contains the specified row of the view's row

        :param view_id: primary key value
        :param view_column: primary key name, defaults to "id"
        :return: row index of the model containing the specified row
        """
        col = self.getColumnIndexLookup()[view_column]
        nrows = self.rowCount()

        i = bisect_left(
            range(nrows),
            view_id,
            key=lambda row: self.fetchRow(row)[col],
        )
        return i if i != nrows and self.fetchRow(i)[col] == view_id else -1


class RecordingSummaryTableModel(SummaryTableModel):

    __summary_name__: str = "recording_summary"

    def __init__(
        self,
        db: SbVoiceDb,
        *,
        batch_size: int | None = None,
        parent: QObject | None = None,
    ):

        super().__init__(db, batch_size, parent)

    def setGender(self, value: Literal["m", "w"] | None):
        """set a filter condition on speaker's gender"""
        self.beginResetModel()
        self._set_or_drop_filter_key("gender", value)
        self.endResetModel()

    def setAgeRange(self, min: int | None, max: int | None):
        """set filter conditions on speaker's age

        :param min: minimum age (inclusive)
        :param max: maximum age (inclusive)
        """
        self.beginResetModel()
        self._set_or_drop_filter_key("minimum_age", min)
        self._set_or_drop_filter_key("maximum_age", max)
        self.endResetModel()

    def setPathologies(
        self, pathologies: str | Sequence[str] | None, include_normal: bool = True
    ):
        """set filter conditions on speaker's voice disorders

        :param pathologies: a list of voice disorders to include
        :param include_normal: True to also include normal speakers, defaults to True

        To include all pathologies but no normals, provide `include_normal=False`
        without the pathologies argument.

        To include only normals, set `pathologies=[]`.
        """
        self.beginResetModel()
        self._set_or_drop_filter_key("pathologies", pathologies)
        self._set_or_drop_filter_key("include_normal", include_normal)
        self.endResetModel()

    def setUtterances(
        self, utterances: UtteranceLiteral | Sequence[UtteranceLiteral] | None
    ):
        """set filter condition on which utterance recordings to include"""
        self.beginResetModel()
        self._set_or_drop_filter_key("utterances", utterances)
        self.endResetModel()

    def setDurationRange(self, min: float | None, max: float | None):
        """set filter conditions on the duration of the recordings

        :param min: minimum duration in seconds
        :param max: maximum duration in seconds
        """
        self.beginResetModel()
        self._set_or_drop_filter_key("minimum_duration", min)
        self._set_or_drop_filter_key("maximum_duration", max)
        self.endResetModel()

    def _compose_tooltip(self, field_values: dict[str, Any]) -> str | None:
        # same tooltip for the entire row

        # field_values = {
        #     k: None if v is None else data._mapping[v] for k, v in field_keys.items()
        # }

        ttparts = {}
        for k, v in field_values.items():
            if k == "gender":
                ttparts["gender"] = {"m": "male", "w": "female"}[v]
            elif k == "age":
                ttparts["age"] = f"{v}-y/o"
            elif k == "type":
                ttparts["dis"] = {"n": "Normal", "p": "Disordered"}[v]
            elif k == "utterance":
                ttparts["utt"] = {
                    "a_n": "/a/ normal pitch",
                    "i_n": "/i/ normal pitch",
                    "u_n": "/u/ normal pitch",
                    "a_l": "/a/ low pitch",
                    "i_l": "/i/ low pitch",
                    "u_l": "/u/ low pitch",
                    "a_h": "/a/ high pitch",
                    "i_h": "/i/ high pitch",
                    "u_h": "/u/ high pitch",
                    "a_lhl": "/a/ varying pitch",
                    "i_lhl": "/i/ varying pitch",
                    "u_lhl": "/u/ varying pitch",
                    "aiu": "all vowels",
                    "phrase": "phrase",
                }[v]
            elif k == "duration":
                ttparts["dur"] = f"({v:0.2f} s)"

        tooltip = ", ".join(
            [
                " ".join(
                    [ttparts[part] for part in ("dis", "age", "gen") if part in ttparts]
                ),
                " ".join([ttparts[part] for part in ("utt", "dur") if part in ttparts]),
            ]
        )
        if len(tooltip):
            tooltip = tooltip.capitalize()

        return tooltip

    def getRecordingRow(self, recording_id: int) -> int:
        """return the model's row index of the specified recording

        :param recording_id: primary key value of the recording
        """
        return self.findRow(recording_id, "id")

    def getRecording(self, index: QModelIndex, **kwargs) -> Recording | None:
        """Fetch the database entry of the specified recording

        :param index: index of the table cell showing the target recording

        :param full_file_paths: True for the returned `Recording.nspfile` and
                                `Recording.eggfile` to contain the full paths,
                                defaults to False
        :param query_session: True to populate `Recording.session` attribute,
                              defaults to False
        :param query_speaker: True to populate `Recording.session.speaker` attribute,
                              defaults to False

        :return: _description_
        """

        return (
            self._db.get_recording(self.fetchRow(index.row())._mapping["id"], **kwargs)
            if index.isValid()
            else None
        )
