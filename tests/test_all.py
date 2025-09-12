from sbvoicedb import SbVoiceDb, Pathology, Speaker, RecordingSession, Recording
import pytest
from os import remove
from shutil import rmtree


def load_db():
    return SbVoiceDb(
        ":memory:",
        download_mode="lazy",
        pathology_filter=Pathology.name.in_(
            ["Carcinoma in situ", "Dysplastischer Kehlkopf"]
        ),
        include_healthy=False,
        # speaker_filter=Speaker.gender == "m",
        # session_filter=RecordingSession.speaker_age.between(60, 75),
        # recording_filter=Recording.utterance.in_(("a_n", "phrase")),
    )


@pytest.fixture(scope="module")
def sbvoicedb():
    yield load_db()


def test_pathology(sbvoicedb):

    assert sbvoicedb.get_pathology_count() > 0
    ids = sbvoicedb.get_pathology_ids()
    names = sbvoicedb.get_pathology_names()
    for patho in sbvoicedb.iter_pathologies():
        assert patho.id in ids
        assert patho.name in names


def test_speaker(sbvoicedb):
    assert sbvoicedb.get_speaker_count() > 0
    ids = sbvoicedb.get_speaker_ids()
    for speaker in sbvoicedb.iter_speakers():
        assert speaker.id in ids


def test_session(sbvoicedb):
    assert sbvoicedb.get_session_count() > 0
    ids = sbvoicedb.get_session_ids()
    for session in sbvoicedb.iter_sessions():
        assert session.id in ids


def test_recording(sbvoicedb):
    assert sbvoicedb.get_recording_count() > 0  # dataset downloading occurs here
    ids = sbvoicedb.get_recording_ids()
    for rec in sbvoicedb.iter_recordings():
        assert rec.id in ids
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
