import logging
from sbvoicedb import database, Pathology, Speaker, RecordingSession, Recording
import nspfile
from pprint import pprint

logging.basicConfig(level=logging.WARNING)
db = database.SbVoiceDb(
    "/home/kesh/data/SVD",
    download_mode="lazy",
    # pathology_filter=Pathology.name == "Laryngitis",
    # include_healthy=True,
    # speaker_filter=Speaker.gender == "w",
    # session_filter=RecordingSession.speaker_age.between(50, 70),
    # recording_filter=Recording.utterance.in_(("a_n", "phrase")),
)

print(f"downloaded: {db.number_of_sessions_downloaded}/{db.number_of_all_sessions}")

# exit()

# db._mark_downloaded(("Spasmodische Dysphonie", "Stimmlippenpolyp", "healthy"), True)

single_case = [
    k for k in db.get_pathology_names() if db.get_session_count(pathologies=k) == 1
]

print(single_case)


db_laryngitis = db = database.SbVoiceDb(
    "/home/kesh/data/SVD",
    download_mode="lazy",
    pathology_filter=Pathology.name == "Laryngitis",
    include_healthy=True,
    speaker_filter=Speaker.gender == "w",
    session_filter=RecordingSession.speaker_age.between(50, 70),
    recording_filter=Recording.utterance.in_(("a_n", "phrase")),
)

# db.set_session_filter(~RecordingSession.pathologies.any())
# db.set_session_filter(RecordingSession.type=='n')
# db.set_speaker_filter(Speaker.gender=='m')
print(f"number of pathologies found: {db_laryngitis.get_pathology_count()}")
print(f"number of recording sessions found: {db_laryngitis.get_session_count()}")
print(f"number of unique speakers: {db_laryngitis.get_speaker_count()}")
print(f"number of recordings: {db_laryngitis.get_recording_count()}")

import numpy as np
from matplotlib import pyplot as plt

rec = next(db_laryngitis.iter_recordings())

t = np.arange(rec.length)/rec.rate

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t,rec.nspdata)
axes[0].set_ylabel('acoustic data')
axes[1].plot(t,rec.eggdata)
axes[1].set_ylabel('EGG data')
axes[1].set_xlabel('time (s)')
plt.tight_layout()
plt.show()

exit()

for session in db.iter_sessions():
    print(
        session.id,
        session.speaker.id,
        session.speaker_age,
        session.speaker.gender,
        [patho.name for patho in session.pathologies],
    )

for rec in db.iter_recordings(full_file_paths=True):
    print(
        rec.id,
        rec.nspfile,
        rec.eggfile,
        rec.session.id,
        rec.session.speaker.id,
        rec.session.speaker_age,
        rec.session.speaker.gender,
        [patho.name for patho in rec.session.pathologies],
    )

    assert len(rec.nspdata) == rec.length and len(rec.eggdata) == rec.length

for speaker in db.iter_speakers():
    print(
        speaker.id,
        speaker.birthdate,
        speaker.gender,
        [s.id for s in speaker.sessions],
    )

# pprint(db.get_session_ids_of_all_pathologies(use_name=True))
# for speaker in speakers:
#     print(f"{speaker=}")
#     print(db.get_session_ids_of_speaker(speaker))
