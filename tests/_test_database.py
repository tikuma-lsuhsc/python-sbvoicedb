import logging
from sbvoicedb import database, Pathology, Speaker, RecordingSession, Recording
import nspfile
from pprint import pprint

logging.basicConfig(level=logging.WARNING)

db = database.SbVoiceDb("/home/kesh/data/SVD", download_mode="incremental")
# db._mark_downloaded(("Spasmodische Dysphonie", "Stimmlippenpolyp", "healthy"), True)

# db.set_pathology_filter(Pathology.name == "Spasmodische Dysphonie", include_normal=True)
db.set_pathology_filter(
    Pathology.name.in_(("Spasmodische Dysphonie", "Stimmlippenpolyp")),
    include_normal=False,
)

db.download_data()

# db.set_session_filter(~RecordingSession.pathologies.any())
# db.set_session_filter(RecordingSession.type=='n')
# db.set_speaker_filter(Speaker.gender=='m')
pathologies = db.get_pathology_ids()
print(f"number of pathologies found: {len(pathologies)}")
sessions = db.get_session_ids()
print(f"number of sessions found: {len(sessions)}")
speakers = db.get_speaker_ids()
print(f"number of unique speakers: {len(speakers)}")
speakers = db.get_recording_ids()
print(f"number of recordings: {len(speakers)}")


print(f"{db.number_of_sessions_downloaded}/{db.number_of_all_sessions}")
# pprint(db.get_session_ids_of_all_pathologies(use_name=True))
# for speaker in speakers:
#     print(f"{speaker=}")
#     print(db.get_session_ids_of_speaker(speaker))
