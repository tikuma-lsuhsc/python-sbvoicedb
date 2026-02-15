import logging
from os import path

import numpy as np
from matplotlib import pyplot as plt

from sbvoicedb import (
    Pathology,
    Recording,
    RecordingSession,
    Speaker,
    database,
)

logging.basicConfig(level=logging.WARNING)
db_laryngitis = database.SbVoiceDb(
    pathology_filter=Pathology.name == "Laryngitis",
    include_healthy=True,
    speaker_filter=Speaker.gender == "w",
    session_filter=RecordingSession.speaker_age.between(50, 70),
    recording_filter=Recording.utterance.in_(("a_n", "phrase")),
)

rec = next(db_laryngitis.iter_recordings())

t = np.arange(rec.length) / rec.rate

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t, rec.nspdata)
axes[0].set_ylabel("acoustic data")
axes[0].set_title(path.relpath(rec.nspfile, db_laryngitis.datadir))
axes[1].plot(t, rec.eggdata)
axes[1].set_ylabel("EGG data")
axes[1].set_title(path.relpath(rec.eggfile, db_laryngitis.datadir))
axes[1].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("example_rec_data.png")
plt.show()
