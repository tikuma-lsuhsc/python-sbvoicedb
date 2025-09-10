import logging
from sbvoicedb import database, Pathology, Speaker, RecordingSession, Recording
import nspfile
from pprint import pprint

logging.basicConfig(level=logging.WARNING)
db = database.SbVoiceDb(
    "/home/kesh/data/SVD",
    download_mode="lazy",
    pathology_filter=Pathology.name == "Laryngitis",
    include_healthy=True,
    speaker_filter=Speaker.gender == "w",
    session_filter=RecordingSession.speaker_age.between(50, 70),
    recording_filter=Recording.utterance.in_(("a_n", "phrase")),
)

print(f"downloaded: {db.number_of_sessions_downloaded}/{db.number_of_all_sessions}")

# exit()

# db._mark_downloaded(("Spasmodische Dysphonie", "Stimmlippenpolyp", "healthy"), True)

single_case = [
    k
    for k, v in db.get_session_ids_of_all_pathologies(use_name=True).items()
    if len(v) == 1
]

print(single_case)


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
