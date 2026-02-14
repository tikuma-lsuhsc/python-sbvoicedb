import logging

import pandas as pd

from sbvoicedb import Pathology, Recording, RecordingSession, Speaker, database

logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO)
db = database.SbVoiceDb(
    pathology_filter=Pathology.name.in_(["Laryngitis", "Dysphonie"]),
    include_healthy=False,
    speaker_filter=Speaker.gender == "w",
    session_filter=RecordingSession.speaker_age.between(50, 70),
    recording_filter=Recording.utterance.in_(("a_n", "i_n")),
)

print(f"number of pathologies found: {db.get_pathology_count()}")
print(f"number of recording sessions found: {db.get_session_count()}")
print(f"number of unique speakers: {db.get_speaker_count()}")
print(f"number of recordings: {db.get_recording_count()}")

print(pd.DataFrame(db.pathology_summary()))

db.pathology_summary(columns="*")
db.pathology_summary(columns="name")
db.pathology_summary(columns=["id", "name"])
db.pathology_summary(columns={"id": None, "name": "pathology name"})

db.pathology_summary(minimum_speakers=5)
db.pathology_summary(minimum_speakers=5, maximum_speakers=10)
db.pathology_summary(maximum_speakers=10)

db.pathology_summary(minimum_sessions=5)
db.pathology_summary(minimum_sessions=5, maximum_sessions=10)
db.pathology_summary(maximum_sessions=10)

db.pathology_summary(order_by="id")
db.pathology_summary(order_by=["nb_speakers", "nb_sessions"])
db.pathology_summary(order_by={"nb_speakers": "asc", "nb_sessions": "desc"})

db.pathology_summary(limit=5)
db.pathology_summary(offset=5)


print(pd.DataFrame(db.recording_session_summary()))
db.recording_session_summary(columns="*")
db.recording_session_summary(columns="speaker_id")
db.recording_session_summary(columns=["speaker_id", "session_id"])
db.recording_session_summary(columns={"speaker_id": None, "session_id": "session"})

db.recording_session_summary(gender="w")

db.recording_session_summary(minimum_age=30)
db.recording_session_summary(minimum_age=30, maximum_age=50)
db.recording_session_summary(maximum_age=50)

db.recording_session_summary(include_normal=False)
db.recording_session_summary(pathologies="Aryluxation")
db.recording_session_summary(pathologies="Aryluxation", include_normal=True)
db.recording_session_summary(pathologies=["Aryluxation", "Balbuties"])


db.recording_session_summary(order_by="session_id")
db.recording_session_summary(order_by=["age", "gender"])
db.recording_session_summary(order_by={"age": "desc", "speaker_id": "desc"})

db.recording_session_summary(limit=5)
db.recording_session_summary(offset=5)

print(pd.DataFrame(db.recording_summary()))

db.recording_summary(minimum_duration=0.5)
db.recording_summary(minimum_duration=0.5, maximum_duration=1.0)
db.recording_summary(maximum_duration=1.0)
