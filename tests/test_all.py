from sbvoicedb import SbVoiceDb
import pandas as pd
from os import path
import numpy as np
from glob import glob
import re
import pytest


def load_db():
    return SbVoiceDb("", _dev=True)


@pytest.fixture(scope="module")
def sbvoicedb():
    return load_db()


def test_query(sbvoicedb):

    df = sbvoicedb.query()
    print(df)


def test_files(sbvoicedb):
    print(
        sbvoicedb.get_files(
            "i_n",
            G="m",
        )
    )
    print(sbvoicedb.get_files("a_n"))
    # print(sbvoicedb.get_files("phrase"))


def test_iter_data(sbvoicedb):
    for id, fs, x in sbvoicedb.iter_data("a_n"):
        pass
    for id, fs, x, info in sbvoicedb.iter_data("a_n", auxdata_fields=["G", "A"]):
        pass


def test_read_data(sbvoicedb):
    id = 1
    task = "a_n"
    sbvoicedb.read_data(id, task, padding=0.3)  # full data file
    types = sbvoicedb.tasks  # audio segment types
    for t in types:
        sbvoicedb.read_data(id, t)
