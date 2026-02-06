import csv
import io
import unicodedata
import zipfile
from os import path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

import requests
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from typing_extensions import IO, TypedDict

url = "https://stimmdb.coli.uni-saarland.de/data/voice_data.csv"
zenodo_url = "https://zenodo.org/api/records/16874898"


class CsvDict(TypedDict):
    AufnahmeID: str
    AufnahmeTyp: str
    AufnahmeDatum: str
    Diagnose: str
    SprecherID: str
    Geburtsdatum: str
    Geschlecht: str
    Pathologien: str


def download_database(timeout: float = 10.0) -> csv.DictReader:

    # site-packages/sbvoicedb/voice_data.csv
    cached_file = path.join(path.dirname(__file__), "summary.csv")

    try:
        with open(cached_file, "rt", encoding="utf8") as f:
            f = io.StringIO(f.read())
    except FileNotFoundError:
        r = requests.get(url, timeout=timeout)
        f = io.StringIO(r.text)
        with open(cached_file, "wt", encoding="utf8") as fcache:
            copyfileobj(f, fcache)
        f.seek(0)
    return csv.DictReader(f)


def download_data(
    dstdir: str,
    pathology: str | None = None,
    use_memory: bool | None = None,
    data_zip_file: str | None = None,
):
    """download nsp/egg data from Zenodo

    :param dstdir: destination folder (will create subfolders with recording id's as the names)
    :param pathology: specify the pathology to download (or 'healthy' for normals), defaults to None
    :param use_memory: specify if download only to memory, defaults to None to use memory for the download files < 127 MB
    :param data_zip_file: specify if data.zip file is available locally to skip
                          downloading
    """

    if pathology is None or data_zip_file is not None:
        # download full dataset at once
        key = "data.zip"
        dstdir, datadir = path.split(dstdir)
        if datadir != "data":
            raise ValueError("dstdir be a 'data' subfolder")
    else:
        # specific dataset
        key = f"{unicodedata.normalize('NFC', pathology)}.zip"

    if data_zip_file is not None:
        nbytes = path.getsize(data_zip_file)
        extract_data(dstdir, "data.zip", nbytes, data_zip_file)
        return

    repo = requests.get(zenodo_url).json()

    try:
        file = next(
            file
            for file in repo["files"]
            if unicodedata.normalize("NFC", file["key"]) == key
        )
    except StopIteration as e:
        raise ValueError(
            f"{pathology=} is not a valid pathology name used by the database."
        ) from e

    nbytes = file["size"]
    if use_memory is None:
        use_memory = nbytes < 2**27
    with (
        TemporaryDirectory() as tmpdir,
        (
            io.BytesIO() if use_memory else open(path.join(tmpdir, "data.zip"), "+bw")
        ) as buf,
    ):
        # Streaming, so we can iterate over the response.
        response = requests.get(file["links"]["self"], stream=True)

        # Sizes in bytes.
        block_size = 1024

        with tqdm(
            desc=f"downloading {key}", total=nbytes, unit=" bytes", leave=True
        ) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                buf.write(data)

        if buf.seekable():
            buf.seek(0)

        extract_data(dstdir, key, nbytes, buf)


def extract_data(dstdir: str, key: str, nbytes: int, buf: str | IO[bytes]):
    """extract downloaded data file

    :param dstdir: data directory
    :param key: file name display for tqdm
    :param nbytes: number of bytes of the data
    :param buf: zipped byte buffer or path of the downloaded zip file
    """

    with (
        zipfile.ZipFile(buf, "r") as f,
        tqdm(
            desc=f"unzipping {key}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=nbytes,
            leave=True,
        ) as progress,
    ):
        for i in f.infolist():
            if not getattr(i, "file_size", 0):  # directory
                f.extract(i, dstdir)
            else:
                with (
                    f.open(i) as fi,
                    open(path.join(dstdir, i.filename), "wb") as fo,
                ):
                    copyfileobj(CallbackIOWrapper(progress.update, fi), fo)
