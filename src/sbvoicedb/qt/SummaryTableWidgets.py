from __future__ import annotations

import logging

from typing_extensions import cast, get_args

from qtpy.QtWidgets import QTableView, QWidget, QVBoxLayout
from qtpy.QtCore import QModelIndex, Signal, Slot, QItemSelectionModel, Qt

from .. import SbVoiceDb, Recording

from .SummaryTableModels import RecordingSummaryTableModel, RecordingSummaryColumn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RecordingSummaryTableWidget(QWidget):

    recordingSelected = Signal(Recording, name="recordingSelected")
    recordingUnselected = Signal(int, name="recordingUnselected")

    _shown_cols: list[str] | None = None
    _hidden_cols: list[str] | None = None

    def __init__(
        self,
        db: SbVoiceDb | None = None,
        column_names: dict[RecordingSummaryColumn, str] | None = None,
        show_columns: list[str] | None = None,
        hide_columns: list[str] | None = None,
        parent: QWidget | None = None,
    ):
        """Qt Widget to display the SbVoiceDb's recording_summary view

        :param db: database object, defaults to None
        :param column_names: recording_summary view's columns to display or
                             a dict selecting the columns as keys and their values
                             as the table column section headers, defaults to None
        :param show_columns: specify which of the recording_summary's columns to
                             show, defaults to None to show all columns.
        :param hide_columns: specify which of the recording_summary's columns to
                             hide, defaults to None to show all columns
        :param parent: parent widget, defaults to None

        Only one of the `show_columns` or `hide_columns` can be specified.
        """
        if show_columns is not None and hide_columns is not None:
            raise ValueError("Only show_columns or hide_columns can be specified.")

        if db is None:
            db = SbVoiceDb()

        super().__init__(parent)

        if show_columns is not None:
            self._shown_cols = show_columns
        if hide_columns is not None:
            self._hidden_cols = hide_columns

        self.view = view = QTableView(self)
        view.setSelectionMode(view.SelectionMode.SingleSelection)
        view.setSelectionBehavior(view.SelectionBehavior.SelectRows)
        hheader = view.horizontalHeader()
        hheader.setSectionResizeMode(hheader.ResizeMode.Stretch)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.verticalHeader().hide()
        view.setCurrentIndex(QModelIndex())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view)
        self.setLayout(layout)

        # assign the table model
        if column_names is None:
            columns = None
        else:
            columns = {
                c: column_names.get(c, c) for c in get_args(RecordingSummaryColumn)
            }
            if columns["id"] != "id":
                raise ValueError('column_names cannot modify the "id" column name.')

        model = RecordingSummaryTableModel(db, parent=self)
        self.setModel(model)

    def model(self) -> RecordingSummaryTableModel:
        """returns the underlying table model"""
        return cast(RecordingSummaryTableModel, self.view.model())

    def setModel(self, model: RecordingSummaryTableModel):
        """set the underlying table model and configure the widget"""

        # keep the same database
        old_model = self.model()
        if old_model == model:
            return

        if old_model is not None:
            old_model.modelReset.disconnect(self._on_model_reset)
            model.setDatabase(old_model.database())
            model.setColumns(old_model.columns())

        self.view.setModel(model)
        model.modelReset.connect(self._on_model_reset)

        # reconnect the selector to the model
        selector = self.selectionModel()
        selector.currentChanged.connect(self._on_current_changed)

    def selectionModel(self) -> QItemSelectionModel:
        return cast(QItemSelectionModel, self.view.selectionModel())

    @Slot()
    def _on_model_reset(self):
        # SummaryTable model emits modelReset signal after the first database fetch
        # Its columns and number of rows are not known until then

        # hide id & recording details (let tooltip show)
        model = self.model()

        view = self.view
        lut = model.getColumnIndexLookup()  # sql field to table column index lookup
        if self._shown_cols is not None:
            for c in get_args(RecordingSummaryColumn):
                i = lut.get(c, None)
                if i is not None:
                    view.setColumnHidden(i, c not in self._shown_cols)
        elif self._hidden_cols is not None:
            for c in get_args(RecordingSummaryColumn):
                i = lut.get(c, None)
                if i is not None:
                    view.setColumnHidden(i, c in self._hidden_cols)

    def selectRecording(self, recording_id: int):
        """select the specified recording row

        :param recording_id: database id of the recording to select
        """

        model = self.model()
        row = model.getRecordingRow(recording_id)
        self.view.setCurrentIndex(model.index(row, 0))

    @Slot(QModelIndex, QModelIndex)
    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex):

        model = self.model()

        if previous.isValid():
            self.recordingUnselected.emit(model.fetchRow(previous.row())._mapping["id"])

        if current.isValid():
            self.recordingSelected.emit(
                model.getRecording(current, full_file_paths=True)
            )

    ##############

    def currentRecording(self, **kwargs) -> Recording | None:
        """Fetch the database entry of the currently selected recording

        :param full_file_paths: True for the returned `Recording.nspfile` and
                                `Recording.eggfile` to contain the full paths,
                                defaults to False
        :param query_session: True to populate `Recording.session` attribute,
                              defaults to False
        :param query_speaker: True to populate `Recording.session.speaker` attribute,
                              defaults to False
        """
        model = self.model()

        return model.getRecording(
            self.view.currentIndex(), full_file_paths=True, **kwargs
        )
