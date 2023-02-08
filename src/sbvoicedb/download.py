import logging
import re
from os import path, scandir, makedirs
import requests
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd
from time import sleep
from requests.adapters import HTTPAdapter
from tempfile import TemporaryDirectory
from typing import Tuple

from .common import *
from .process import extract

url_root = "http://stimmdb.coli.uni-saarland.de"
url = "/".join((url_root, "index.php4#target"))


def _parse_result_item(tr):
    tds = tr.find_all("td")
    return pd.Series(
        {
            "ID": int(tds[1].text),
            "T": tds[2].text,
            "D": date.fromisoformat(tds[3].text),
            "S": int(tds[4].text),
            "G": tds[5].text,
            "A": int(tds[6].text),
            "Pathologies": tds[7].text,
            "Remark w.r.t. diagnosis": tds[8].text,
            "B": tds[9].text,
        }
    )


def _parse_result_page(soup):
    """Parse database query post response"""
    re_labels = re.compile(r"^labels_(\d+)$")
    {
        int(re_labels.match(t.attrs["name"])[1]): t.parent.parent
        for t in soup.find_all("input", {"name": re_labels})
    }

    return pd.DataFrame(
        [
            _parse_result_item(input.parent.parent)
            for input in soup.find_all("input", {"name": re_labels})
        ],
    ).astype(db_dtypes)


def _post_page(s, pg, pause=0.01):
    sleep(pause)
    r = s.post(url, data={"hits_page1": pg})
    soup = BeautifulSoup(r.text, "html.parser")
    return _parse_result_page(soup)


def download_database(_dev=False):
    with requests.Session() as s:
        # set language
        s.post(url, data={"sb_lang": "English"})
        # start database request session
        s.post(url, data={"sb_search": "Database+request"})
        # retrieve the first 10
        r = s.post(url, data={"sb_sent": "Accept"})
        soup = BeautifulSoup(r.text, "html.parser")
        df = _parse_result_page(soup)  # form the first dataframe

        if not _dev:
            # for development, only download the first page

            # get number of pages, and retrieve
            nb_pages = int(
                soup.find("select", id="hits_page0").find_all("option")[-1].text
            )
            df = pd.concat([df, *(_post_page(s, pg) for pg in range(1, nb_pages))])

        return df.set_index("ID").sort_index()


def load_db(
    dbdir: str, no_download: bool, _dev: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.MultiIndex]:
    """load Saarbrueken voice database dataframe

    :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
    :type dbdir: str
    :param no_download: True to use only cached files
    :type no_download: bool
    :param _dev: True to be in devmode (download only the first 10)
    :type _dev: bool
    :return: Panda dataframes---main database, diagnoses, files, & task timings--- and
             Panda multiindex of missing files
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.MultiIndex]

    """

    if not path.isdir(dbdir):
        raise ValueError(f'Database dir ["{dbdir}"] does not exist')

    # load the main dataframe
    csvfile = path.join(dbdir, "db.csv")

    try:
        df = (
            pd.read_csv(
                csvfile,
                dtype={k: v for k, v in db_dtypes.items() if k != "D"},
                parse_dates=["D"],
                keep_default_na=False,
            )
            .set_index("ID")
            .sort_index()
        )
    except:
        if no_download:
            raise RuntimeError(f"Database CSV not found at {csvfile}")
        print(
            "Retrieve the Saarbrueken Voice Database. This will take a while to complete..."
        )
        df = download_database(_dev)
        df.reset_index().to_csv(csvfile, index=False)
        print(
            "Successfuly retrieved the database. Audio data will be downloaded and cached later on demand."
        )

    # create session_id <-> pathology table
    re_tok = re.compile(r"\s*,\s*")
    df_dx = pd.concat(
        [
            pd.DataFrame(
                {"ID": id, "pathology": pd.Series(re_tok.split(s), dtype="string")}
            )
            for id, s in df["Pathologies"].items()
        ]
    )

    # load the vowel timing dataframe
    csvfile = path.join(dbdir, "vowel_timings.csv")

    try:
        df_timing = (
            pd.read_csv(csvfile)
            .astype(dtypes_timing)
            .set_index(["ID", "F"])
            .sort_index()
        )
    except:
        # none created yet, create an empty dataframe
        df_timing = TimingDataFrame()

    # regexp to id downloaded files
    re_wname = re.compile(r"(\d{1,4})-(phrase|iau)(-egg)?.(?:nsp|egg)$")

    # get downloaded file list
    wavdir = path.join(dbdir, data_dir)
    try:
        with scandir(path=wavdir) as it:
            df_files = FileSeries(
                [
                    {
                        "ID": int(m[1]),
                        "Task": m[2],
                        "IsEgg": bool(m[3]),
                        "File": entry.name,
                    }
                    for entry in it
                    if entry.is_file() and (m := re_wname.match(entry.name))
                ]
            )
    except:
        df_files = FileSeries()

    # get missing/invalid file list
    csvfile = path.join(dbdir, "missing_data.csv")
    try:
        mi_miss = pd.read_csv(csvfile)
    except:
        mi_miss = pd.DataFrame(columns=["Task", "IsEgg", "ID"])
    mi_miss = pd.MultiIndex.from_frame(
        mi_miss.astype({k: v for k, v in dtypes_files.items() if k != "File"})
    )

    return df, df_dx, df_files, df_timing, mi_miss


def download_data(dst, ids, tasks=None, exp_nsp=True, exp_egg=False, progress=None):

    if not (exp_nsp or exp_egg):
        raise ValueError("both exp_nsp and exp_egg are False, nothing to download")

    if not path.isdir(dst):
        raise ValueError("dst must be an existing directory path")

    if len(ids) > 20:
        raise ValueError("ids cannot have more than 20 elements")

    if tasks is None:
        tasks or list(task2key.keys())  # download all
    data = {k: 1 for f, k in task2key.items() if f in tasks}

    if len(data) != len(tasks):
        raise ValueError("invalid task assigned in tasks")

    if exp_nsp:
        data["s_exp_nspnsp"] = 1
    if exp_egg:
        data["s_exp_eggegg"] = 1
    data["sb_sent"] = "Accept"

    s = requests.Session()
    s.post(url, data={"sb_lang": "English"})  # set language
    s.post(
        url, data={"sb_search": "Database+request"}
    )  # start database request session

    s.post(
        url, data={"s_sess_id": ",".join(str(id) for id in ids), "sb_sent": "Accept"}
    )  # get specific recording
    s.post(url, data={"sb_export": "Export"})  # retrieve the first 10

    r = s.post(url, data=data)  # export
    soup = BeautifulSoup(r.text, "html.parser")

    re_a_href = re.compile(r"export/sbvoicedb_\S{6}\.zip$")
    re_pct = re.compile(r"\d+%")
    while not (a := soup.find("a", href=re_a_href)):
        try:
            prog_pct = int(soup.find(string=re_pct).text[:-1])
            print(f"waiting... ({prog_pct}% complete)")

            assert prog_pct < 100 and not soup.find(string="The zip file is empty.")

        except:
            # invalid file
            logging.warn("no matching file found.")
            return []

        sleep(4)
        r = s.post(url)
        soup = BeautifulSoup(r.text, "html.parser")

    # download the zip file
    print("downloading...")
    s.mount("http://", HTTPAdapter(max_retries=5))
    r = s.get(
        "/".join([url_root, a.attrs["href"]]),
        stream=True,
    )

    with TemporaryDirectory() as tdir:
        zippath = path.join(tdir, "test.zip")

        nbytes = int(r.headers.get("Content-Length", 0))
        blksz = nbytes // 32 or 1024 * 1024
        with open(zippath, "wb") as f:
            nread = 0
            for b in r.iter_content(chunk_size=blksz):
                if not b:
                    break
                f.write(b)
                nread += len(b)
                # if progress:
                #     progress.update(nread)

        print("extracting...")
        return extract(zippath, dst, progress)
