from __future__ import annotations

from typing_extensions import cast, Sequence

import logging

from qtpy.QtWidgets import QTreeView, QWidget, QVBoxLayout
from qtpy.QtCore import (
    QModelIndex,
    Signal,
    Slot,
    QItemSelectionModel,
    Qt,
    QItemSelection,
)

from .. import SbVoiceDb, Speaker, RecordingSession, Recording

from .SbVoiceDbTreeModel import SbVoiceDbTreeModel, SbVoiceDbTreeLevelLiteral

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SbVoiceDbTreeWidget(QWidget):

    _last_dir: str | None = None
    _delete_db: bool = True

    _db: SbVoiceDb | None = None
    _SelectionFlag = QItemSelectionModel.Current | QItemSelectionModel.ClearAndSelect
    _keep_recording_selection: bool = False

    pathologySelected = Signal(str, name="pathologySelected")
    pathologyUnselected = Signal(int, name="pathologyUnselected")
    speakerSelected = Signal(Speaker, name="speakerSelected")
    speakerUnselected = Signal(int, name="speakerUnselected")
    sessionSelected = Signal(RecordingSession, name="sessionSelected")
    sessionUnselected = Signal(int, name="sessionUnselected")
    recordingSelected = Signal(Recording, name="recordingSelected")
    recordingUnselected = Signal(int, name="recordingUnselected")

    def __init__(
        self,
        db: SbVoiceDb,
        tree_levels: Sequence[SbVoiceDbTreeLevelLiteral] | None = None,
        parent: QWidget | None = None,
        *,
        ModelCls: type[SbVoiceDbTreeModel] = SbVoiceDbTreeModel,
        keep_recording_selection: bool = True,
    ):
        super().__init__(parent)

        if keep_recording_selection != self._keep_recording_selection:
            self._keep_recording_selection = keep_recording_selection

        self.view = view = QTreeView(self)
        view.setModel(ModelCls(db, tree_levels, parent=self))
        view.setSelectionMode(view.SelectionMode.NoSelection)
        view.setHeaderHidden(True)
        view.setTextElideMode(Qt.TextElideMode.ElideLeft)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view)
        self.setLayout(layout)

        self.setEnabled(False)

        # must reconnect selector
        selector = self.selectionModel()
        selector.currentChanged.connect(self._current_changed)
        selector.selectionChanged.connect(self._selection_changed)

        self.setEnabled(True)

        # default to select the first file
        # self.selectFile(row=0)

    def model(self) -> SbVoiceDbTreeModel:
        return cast(SbVoiceDbTreeModel, self.view.model())

    def selectionModel(self) -> QItemSelectionModel:
        return cast(QItemSelectionModel, self.view.selectionModel())

    def clearSelection(self):

        self.selectionModel().setCurrentIndex(QModelIndex(), self._SelectionFlag)

    def getPathologies(self) -> list[str]:
        return self.model().pathologies()

    def selectPathology(self, name: str):
        """select the named pathology row

        :param name: pathology name in German
        """

        index = self.model().getPathologyIndex(name)
        self.view.setCurrentIndex(index)

    def selectSpeaker(self, speaker_id: int):
        """select the specified speaker row

        :param speaker_id: database id of the speaker to select
        """

        index = self.model().getSpeakerIndex(speaker_id)
        self.view.setCurrentIndex(index)

    def selectSession(self, session_id: int):
        """select the specified recording session row

        :param session_id: database id of the recording session to select

        note: if session is listed under pathology, the same session may be listed
        multiple times. This function selects the first listing of the session id.
        """

        index = self.model().getSessionIndex(session_id)
        self.view.setCurrentIndex(index)

    def selectRecording(self, recording_id: int):
        """select the specified recording row

        :param recording_id: database id of the recording to select
        """

        index = self.model().getRecordingIndex(recording_id)
        self.view.setCurrentIndex(index)

    @Slot(QModelIndex, QModelIndex)
    def _current_changed(self, current: QModelIndex, previous: QModelIndex):

        model = self.model()
        db = model.database()
        levels = model.treeLevels()

        new_ids = model.getDbDataIds(current)
        old_ids = model.getDbDataIds(previous)

        logger.debug("._current_changed: %s -> %s", old_ids, new_ids)

        # issue unselected events (leaf -> root)
        for level in levels[::-1]:
            try:
                old_id = old_ids[level]
            except KeyError:
                # not previously selected
                continue

            new_id = new_ids.get(level, None)

            if new_id != old_id:  # something changed
                if level == "session":
                    self.sessionUnselected.emit(old_id)
                elif level == "speaker":
                    self.speakerUnselected.emit(old_id)
                elif level == "pathology":
                    self.pathologyUnselected.emit(old_id)
                elif not self._keep_recording_selection:
                    # only emit recording unselected signal if flag is False
                    self.recordingUnselected.emit(old_id)

        # issue selected events (root->leaf)
        for level in levels:
            try:
                new_id = new_ids[level]
            except KeyError:
                # not selected
                continue

            old_id = old_ids.get(level, None)

            if new_id != old_id:  # something changed
                if level == "pathology":
                    self.pathologySelected.emit(db.get_pathology_name(new_id))
                elif level == "speaker":
                    self.speakerSelected.emit(db.get_speaker(new_id))
                elif level == "session":
                    self.sessionSelected.emit(db.get_session(new_id))
                else:  # recording
                    if self._keep_recording_selection:
                        # also select the recording
                        selector = self.selectionModel()
                        selector.select(current, QItemSelectionModel.ClearAndSelect)
                    else:
                        self.recordingSelected.emit(
                            db.get_recording(new_id, full_file_paths=True)
                        )

    @Slot(QItemSelection, QItemSelection)
    def _selection_changed(self, selected: QItemSelection, deselected: QItemSelection):

        model = cast(SbVoiceDbTreeModel, self.view.model())
        db = model.database()
        levels = model.treeLevels()
        if levels[-1] != "recording":
            # only for the recording table
            return

        for previous in deselected.indexes():
            old_ids = model.getDbDataIds(previous)
            self.recordingUnselected.emit(old_ids["recording"])

        for current in selected.indexes():
            new_ids = model.getDbDataIds(current)
            self.recordingSelected.emit(
                db.get_recording(new_ids["recording"], full_file_paths=True)
            )

    ##############

    def currentPathology(self) -> str | None:
        model = cast(SbVoiceDbTreeModel, self.view.model())
        return model.pathology(self.view.currentIndex())

    def currentSpeaker(self) -> Speaker | None:
        model = cast(SbVoiceDbTreeModel, self.view.model())
        return model.speaker(self.view.currentIndex())

    def currentSession(self) -> RecordingSession | None:
        model = cast(SbVoiceDbTreeModel, self.view.model())
        return model.session(self.view.currentIndex())

    def currentRecording(self, **kwargs) -> Recording | None:
        model = cast(SbVoiceDbTreeModel, self.view.model())
        return model.recording(self.view.currentIndex(), full_file_paths=True)

    def expandAll(self):
        self.view.expandAll()
