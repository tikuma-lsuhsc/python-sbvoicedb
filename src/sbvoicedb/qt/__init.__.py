"""`sbvoicedb.qt` Subpackage to create list/table/tree in Qt Framework

To install the dependencies for this subpackage, use the pip install command:

```
pip install sbvoicedb[qt]
```

This however, does not install the core Qt packages to allow user to choose the
Qt flavor (PyQt6 or PySide6).

Currently there are 2 pairs of models and widgets:

==========================  ==========================  ===========================
Description                 Model Class                 Widget Class
==========================  ==========================  ===========================
`recording_summary` view    RecordingSummaryTableModel  RecordingSummaryTableWidget
Tree of database hierarchy  SbVoiceDbTreeModel          SbVoiceDbTreeWidget
==========================  ==========================  ===========================

"""

from __future__ import annotations

from .SbVoiceDbTreeModel import SbVoiceDbTreeModel, SbVoiceDbTreeLevelLiteral
from .SbVoiceDbTreeWidget import SbVoiceDbTreeWidget

from .SummaryTableModels import RecordingSummaryTableModel
from .SummaryTableWidgets import RecordingSummaryTableWidget

__all__ = [
    "RecordingSummaryTableModel",
    "RecordingSummaryTableWidget",
    "SbVoiceDbTreeModel",
    "SbVoiceDbTreeWidget",
    "SbVoiceDbTreeLevelLiteral",
]
