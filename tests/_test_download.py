from pprint import pprint
import zipfile
import io
from os import path
from shutil import copyfileobj
import unicodedata

import requests

from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from sbvoicedb import download as dl


def test_download_database():
    for row in dl.download_database():
        print(row)


patho = "Bulbärparalyse"
dl.download_data("tests", patho, False)
dl.download_data("tests", patho, True)
dl.download_data("tests", patho, None)
