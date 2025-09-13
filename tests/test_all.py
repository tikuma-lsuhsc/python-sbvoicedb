from sbvoicedb import (
    SbVoiceDb,
    Pathology,
    Speaker,
    RecordingSession,
    Recording,
    sql_expr,
)
import pytest


def load_db():
    return SbVoiceDb(
        ":memory:",
        download_mode="lazy",
        pathology_filter=Pathology.name.in_(
            ["Carcinoma in situ", "Dysplastischer Kehlkopf"]
        ),
        include_healthy=False,
        speaker_filter=Speaker.gender == "m",
        session_filter=RecordingSession.speaker_age.between(50, 75),
        recording_filter=Recording.utterance.in_(("a_n", "phrase")),
    )


@pytest.fixture(scope="module")
def sbvoicedb():
    yield load_db()


def test_pathology(sbvoicedb):

    assert sbvoicedb.get_pathology_count() ==2
    ids = sbvoicedb.get_pathology_ids()
    names = sbvoicedb.get_pathology_names()
    for patho in sbvoicedb.iter_pathologies():
        assert patho.id in ids
        assert patho.name in names


def test_speaker(sbvoicedb):
    assert sbvoicedb.get_speaker_count() ==2
    ids = sbvoicedb.get_speaker_ids()
    for speaker in sbvoicedb.iter_speakers():
        assert speaker.id in ids


def test_session(sbvoicedb):
    assert sbvoicedb.get_session_count() ==2
    ids = sbvoicedb.get_session_ids()
    for session in sbvoicedb.iter_sessions():
        assert session.id in ids

    assert sbvoicedb.get_session(820) is not None
    assert sbvoicedb.get_session_count(speaker_id=1432) == 1
    assert sbvoicedb.get_session_count(pathologies="Carcinoma in situ") == 1
    assert (
        sbvoicedb.get_session_count(
            pathologies=["Carcinoma in situ", "Dysplastischer Kehlkopf"]
        )
        == 2
    )
    assert sbvoicedb.get_session_count(pathologies=36) == 1
    assert sbvoicedb.get_session_count(pathologies=[64, 36]) == 2
    assert (
        sbvoicedb.get_session_count(session_filter=RecordingSession.speaker_age > 70)
        == 1
    )
    assert (
        sbvoicedb.get_session_count(
            recording_filter=sql_expr.and_(
                Recording.length > 100000, Recording.utterance == "phrase"
            )
        )
        == 1
    )


def test_recording(sbvoicedb):
    assert sbvoicedb.get_recording_count() == 4  # dataset downloading occurs here
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
    assert sbvoicedb.get_recording_count(session_id=820) == 2
    assert (
        sbvoicedb.get_recording_count(session_filter=RecordingSession.speaker_age > 70)
        == 2
    )
