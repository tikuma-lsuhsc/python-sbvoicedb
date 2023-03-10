"""
conftest.py
"""

import pytest
import winreg
from os import path

from contextlib import suppress
import itertools

from sbvoicedb import SbVoiceDb


def find_onedrive():
    reg_base = "Software\Microsoft\OneDrive\Accounts"
    found = False
    with suppress(WindowsError), winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, reg_base
    ) as key:
        for i in itertools.count():
            with winreg.OpenKey(key, winreg.EnumKey(key, i)) as acct:
                drive_dir = winreg.QueryValueEx(acct, "UserFolder")[0]
                found = "LSUHSC" in drive_dir
                if found:
                    break
    assert found
    return drive_dir


def load_db():
    dbdir = path.join(find_onedrive(), "data", "PVQD")
    return SbVoiceDb(dbdir)


@pytest.fixture(scope="module")
def sbvoicedb():
    return load_db()
