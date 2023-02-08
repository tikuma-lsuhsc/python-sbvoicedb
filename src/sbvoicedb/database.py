"""Saarbrueken Voice Database Reader module
"""

import logging
import pandas as pd
from os import path, makedirs
import shutil
import numpy as np
import nspfile
import numpy as np
from tempfile import TemporaryDirectory
from typing import Literal, List, Callable, NamedTuple, Iterator

from .common import *
from .download import download_data, load_db
from .process import align_data

TaskType = Literal[
    # fmt: off
            "a_n", "i_n", "u_n", 
            "a_l", "i_l", "u_l",
            "a_h", "i_h", "u_h", 
            "a_lhl", "i_lhl", "u_lhl",
            "iau", "phase",
    # fmt: on
]

DataField = Literal[
    "ID",  # recording ID
    "T",  # Voice Type ['n' or 'p']
    "D",  # date of recording
    "S",  # speaker ID
    "G",  # gender ['w' or 'm']
    "A",  # age
    "Pathologies",
    "Remark w.r.t. diagnosis",
]


class VoiceData(NamedTuple):
    id: int
    fs: int
    x: np.array


class VoiceDataPlusInfo(NamedTuple):
    id: int
    fs: int
    x: np.array
    info: pd.Series


class SbVoiceDb:
    """SbVoiceDb class

    Constructor Arguments

    :param dbdir: databse directory
    :type dbdir: str
    :param no_download: True to use only cached files, defaults to False
    :type no_download: bool, optional
    :param default_task: set default task type to get, defaults to "a_n"
    :type default_task: Literal[ &quot;a_n&quot;, &quot;i_n&quot;, &quot;u_n&quot;, &quot;a_l&quot;, &quot;i_l&quot;, &quot;u_l&quot;, &quot;a_h&quot;, &quot;i_h&quot;, &quot;u_h&quot;, &quot;a_lhl&quot;, &quot;i_lhl&quot;, &quot;u_lhl&quot;, &quot;iau&quot;, &quot;phase&quot;, ], optional
    :param default_padding: set default padding in seconds (negative to crop), defaults to 0.3
    :type default_padding: float, optional
    :param _dev: _description_, defaults to False
    :type _dev: bool, optional
    """

    def __init__(
        self,
        dbdir: str,
        no_download: bool = False,
        default_task: TaskType = "a_n",
        default_padding: float = 0.3,
        _dev=False,
    ):
        self.default_task = default_task
        self.default_padding = default_padding

        if _dev:
            # devmode
            dbdir = path.join("tests", "db")
            if not path.exists(dbdir):
                makedirs(dbdir)

        # load the database
        (
            self._df,  # main database table
            self._df_dx,  # patient pathology table
            self._df_files,  # list of audio file
            self._df_timing,  # list of audio task start/end timestamps
            self._mi_miss,  # list of missing files
        ) = load_db(dbdir, no_download, _dev)
        self._dir = dbdir  # database dir

    def _add_files(self, new_files):
        # update the dataframe
        df_files = pd.concat([self._df_files, new_files]).sort_index()
        df_files = df_files[~df_files.index.duplicated(keep="last")]

        # finalize dataframe
        self._df_files = df_files

    def _add_missing(self, new_missings):
        # update the dataframe
        df = pd.concat([self._mi_miss.to_frame(), new_missings.to_frame()]).sort_index()
        df = df[~df.index.duplicated()]

        # update the csv file
        csvfile = path.join(self._dir, "missing_data.csv")
        df.to_csv(csvfile, index=False)

        # finalize dataframe
        self._mi_miss = pd.MultiIndex.from_frame(df)

    def _add_timings(self, new_timings):
        # update the dataframe
        df_timing = pd.concat([self._df_timing, new_timings]).sort_index()
        df_timing = df_timing[~df_timing.index.duplicated(keep="last")]

        # update the csv file
        csvfile = path.join(self._dir, "vowel_timings.csv")
        df_timing.reset_index().to_csv(csvfile, index=False)

        # finalize dataframe
        self._df_timing = df_timing

    @property
    def tasks(self) -> List[str]:
        """List of task types"""
        return list(task2key.keys())

    @property
    def diagnoses(self) -> List[str]:
        """List of diagnosis (in German)"""
        return list(self._df_dx["pathology"].unique())

    def query(self, columns: List[DataField] = None, **filters) -> pd.DataFrame:
        """query database

        :param columns: database columns to return, defaults to None
        :type columns: sequence of str, optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: query result
        :rtype: pandas.DataFrame

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values

        """

        # work on a copy of the dataframe
        df = self._df.copy(deep=True)

        df_dx = self._df_dx
        if "Pathologies" in filters:
            incl_dx = []
            excl_dx = []
            for dx in filters["Pathologies"]:
                if dx.startswith("-"):
                    excl_dx.append(dx)
                else:
                    incl_dx.append(dx[1:] if dx[0] == "+" else dx)

            # filter df_dx
            if len(incl_dx):
                ids = df_dx.loc[df_dx["pathology"].isin(incl_dx), "ID"].unique()
                if len(excl_dx):
                    df_dx = df_dx.loc[df_dx["ID"].isin(ids)]
            else:
                ids = df_dx["ID"].unique()
            if len(excl_dx):
                ids = df_dx.loc[~df_dx["pathology"].isin(excl_dx), "ID"].unique()

            if "Pathologies" in columns:
                df_dx = df_dx.loc[df_dx["ID"].isin(ids)]

            df = df.loc[ids]

        if columns and "Pathologies" in columns:
            # add diagnoses column to df
            df["Pathologies"] = self._df_dx.groupby("ID").transform(
                lambda x: ", ".join(x)
            )

        # apply the filters to reduce the rows
        for fcol, fcond in filters.items():

            if fcol == "Pathologies":
                continue

            try:
                if fcol == "ID":
                    s = df.index
                else:
                    s = df[fcol]
            except:
                raise ValueError(f"{fcol} is not a valid column label (check cases)")

            try:  # try range/multi-choices
                if s.dtype.kind in "iufcM":  # numeric/date
                    # 2-element range condition
                    df = df[(s >= fcond[0]) & (s < fcond[1])]
                else:  # non-numeric
                    df = df[s.isin(fcond)]  # choice condition
            except:
                # look for the exact match
                df = df[s == fcond]

        # return only the selected columns
        if columns is not None:
            try:
                df = df[columns]
            except:
                ValueError(
                    f'At least one label in the "columns" argument is invalid: {columns}'
                )

        return df

    def get_files(
        self,
        task: TaskType,
        egg: bool = False,
        cached_only: bool = False,
        paths_only: bool = False,
        auxdata_fields: List[DataField] = None,
        **filters,
    ) -> pd.DataFrame:
        """get audio files

        :param task: utterance task
        :type task: TaskType
        :param egg: True for EGG False for audio, defaults to False
        :type egg: bool, optional
        :param cached_only: True to disallow downloading, defaults to False
        :type cached_only: bool, optional
        :param paths_only: True to return only file paths without timing for vowels, defaults to False
        :type paths_only: bool, optional
        :param auxdata_fields: List of auxiliary data fields, defaults to None
        :type auxdata_fields: List[DataField], optional
        :return: data frame containing file path, start and end time marks, and auxdata
        :rtype: pandas.DataFrame

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values
        """

        if task not in tuple(task2key.keys()):
            raise ValueError(
                f"invalid task ('{task}'): must be one of {sorted(task2key.keys())}"
            )

        # retrieve auxdata and filter if conditions given
        filter_on = len(filters) or bool(auxdata_fields)
        df = (
            self.query(auxdata_fields, **filters)
            if filter_on or bool(auxdata_fields)
            else self._df
        )

        need_timing = task not in ("iau", "phrase")
        file = ("iau" if need_timing else task, egg)

        if not cached_only:
            # check for missing ids in cache
            ids = set(df.index)

            # remove the known missing files
            try:
                mi = self._mi_miss
                mi = mi[(mi.isin(file[0:1], 0) & mi.isin(file[1:2], 1))]
                ids -= set(mi)
            except:
                pass

            # remove already cached files
            try:

                # check for the file ids
                try:
                    ids_ok = set(self._df_files.loc[file].index)
                except KeyError:
                    # none avail
                    ids_ok = set()

                if need_timing and len(ids_ok):
                    # make sure id also in timing data
                    try:
                        # ok only if in timing
                        ids_ok = ids_ok & set(
                            self._df_timing.loc[
                                (slice(None), task), :
                            ].index.get_level_values(0)
                        )

                    except KeyError:
                        # none avail
                        ids_ok = set()

                ids -= ids_ok
            except:
                pass

            if len(ids):
                self._download_groups(
                    list(ids),
                    need_timing or task == "iau",
                    task == "phrase",
                    need_timing,
                    nsp=not egg,
                    egg=egg,
                )

        try:
            # get cached files
            df_files = self._df_files.loc[
                ("iau" if need_timing else task, egg)
            ].to_frame()
        except:
            raise RuntimeError(f"cannot find cached matching files")

        if paths_only:
            need_timing = False

        if need_timing:
            try:
                self._df_timing
                df_timing = self._df_timing.loc[(slice(None), task), :].droplevel(1)
            except:
                df_timing = pd.DataFrame(
                    index=self._df_timing.index.levels[0],
                    columns=self._df_timing.columns,
                    dtype="Int64",
                )
            df_files = df_files.join(df_timing, how="inner" if cached_only else "left")

        # if filtered, remove excluded ID's
        if filter_on:
            df_files = df_files[df_files.index.isin(df.index)]

        if bool(auxdata_fields):
            # return requested info along with the file & range
            df = df_files.join(df, how="inner" if cached_only else "right")
        elif cached_only:
            df = df_files[df_files.notna().all(axis=1)]
        else:
            # no extra info, incl entries needing downloading

            new_indices = df.index[~df.index.isin(df_files.index)]
            df = pd.concat(
                [
                    df_files,
                    pd.DataFrame(
                        [("", pd.NA, pd.NA)] if need_timing else [("",)],
                        index=new_indices,
                        columns=df_files.columns,
                    ).astype(df_files.dtypes),
                ]
            ).sort_index()

        # remove missing files
        mi = self._mi_miss
        mi = mi[(mi.isin(["iau" if need_timing else task], 0) & mi.isin([egg], 1))]
        df = df[~df.index.isin(mi.get_level_values(2))]

        # expand the filepaths
        df["File"] = df["File"].map(lambda v: v and path.join(self._dir, data_dir, v))

        return df

    def in_cache(self, id: int, task: TaskType, egg: bool = False) -> bool:
        """Check if data has been cached

        :param id: recording ID
        :type id: int
        :param task: vocal task
        :type task: TaskType
        :param egg: True for EGG, False for audio, defaults to False
        :type egg: bool, optional
        :return: True if data has been cached and is available locally
        :rtype: bool
        """
        _task = "phrase" if task == "phrase" else "iau"
        file_idx = (_task, egg, id)
        if file_idx in self._mi_miss:
            return True
        has_file = file_idx in self._df_files.index
        return (
            (id, task) in self._df_timing.index
            if has_file and task != _task
            else has_file
        )

    def download_task(
        self, id: int, task: TaskType, egg: bool = False, progress: Callable = None
    ):
        """download a vocal task from a recording

        :param id: recording id
        :type id: int
        :param task: vocal task
        :type task: TaskType
        :param egg: True to download EGG, False to download audio, defaults to False
        :type egg: bool, optional
        :param progress: progress callback, defaults to None (TODO)
        :type progress: Callable, optional
        """
        if self.in_cache(id, task, egg):
            return

        # make sure id & task are valid:
        if id not in self._df.index:
            raise ValueError(f"invalid id: {id}")

        if task not in task2key:
            raise ValueError(f"invalid task: {task}")

        phrase = task == "phrase"
        vowels = task == "iau"
        timings = not (phrase or vowels)
        nsp = not egg

        if timings and not self.in_cache(id, "iau", False):
            vowels = True
            nsp = True

        # if file has not been downloaded, grab it
        self._download_once([id], vowels, phrase, timings, nsp, egg, progress)

    def download_full(
        self,
        vowels: bool,
        phrase: bool,
        timings: bool,
        nsp: bool = True,
        egg: bool = False,
        progress: Callable = None,
        **filters,
    ):
        """download all recording data of given voice & data types

        :param vowels: True to download vowel tasks
        :type vowels: bool
        :param phrase: True to download phrase task
        :type phrase: bool
        :param timings: True to download vowel task timings
        :type timings: bool
        :param nsp: True to download audio data, defaults to True
        :type nsp: bool, optional
        :param egg: True to download EGG data, defaults to False
        :type egg: bool, optional
        :param progress: progress callback function, defaults to None (TODO)
        :type progress: Callable, optional
        """
        egg = bool(egg)

        if not (nsp or egg or timings):
            raise ValueError("Both nsp and egg are False. Nothing to download.")

        # get all matching indices
        ids = self.query(**filters).index

        # download
        self.download_batch(ids, vowels, phrase, timings, nsp, egg, progress)

    def download_batch(
        self,
        ids: List[int],
        vowels: bool,
        phrase: bool,
        timings: bool,
        nsp: bool = True,
        egg: bool = False,
        progress: Callable = None,
    ):
        """download a batch of specified recording sessions in a given voice & data types

        :param ids: ids of recording sessions to download
        :param vowels: True to download vowel tasks
        :type vowels: bool
        :param phrase: True to download phrase task
        :type phrase: bool
        :param timings: True to download vowel task timings
        :type timings: bool
        :param nsp: True to download audio data, defaults to True
        :type nsp: bool, optional
        :param egg: True to download EGG data, defaults to False
        :type egg: bool, optional
        :param progress: progress callback function, defaults to None (TODO)
        :type progress: Callable, optional
        """

        # separate ids into groups with different downloading needs
        files = self._df_files
        mi = self._mi_miss
        tf = pd.DataFrame(index=ids)
        id = pd.Index(list(ids))
        if vowels or timings:
            mi1 = mi[mi.isin(["iau"], 0)]
            tf_miss = id.isin(mi1[mi1.isin([egg], 1)].get_level_values(2))
            tf[(True, False, False, not egg, egg)] = ~(
                id.isin(files.loc[("iau", egg)].index) | tf_miss
            )
            if egg and (nsp or timings):
                tf[(True, False, False, True, False)] = ~(
                    id.isin(files.loc[("iau", False)].index)
                    | id.isin(mi1[mi1.isin([False], 1)].get_level_values(2))
                )
        if phrase:
            mi1 = mi[mi.isin(["phrase"], 0)]
            tf[(False, True, False, not egg, egg)] = ~(
                id.isin(files.loc[("phrase", egg)].index)
                | id.isin(mi1[mi1.isin([egg], 1)].get_level_values(2))
            )
            if egg and nsp:
                tf[(False, True, False, True, False)] = ~(
                    id.isin(files.loc[("phrase", False)].index)
                    | id.isin(mi1[mi1.isin([False], 1)].get_level_values(2))
                )

        if timings:
            tf[(False, False, True, False, False)] = ~(
                tf_miss | id.isin(self._df_timing.index.get_level_values(0).unique())
            )

        def download_groups(grp):

            tf = grp.iloc[0]  # guaranteed to have at least 1 element
            if not tf.any(axis=0):
                return  # nothing to download

            args = tuple(pd.DataFrame(list(grp.columns.values[tf])).any(axis=0))
            self._download_groups(grp.index, *args, progress=progress)

        tf.groupby(list(tf.columns.values)).apply(download_groups)

    def _download_groups(self, ids, *args, ngroup=20, **kwargs):
        k = len(ids) // ngroup

        for i in range(k):
            self._download_once(ids[i * ngroup : (i + 1) * ngroup], *args, **kwargs)
        if ngroup * k < len(ids):
            self._download_once(ids[k * ngroup :], *args, **kwargs)

    def _download_once(
        self, ids, vowels, phrase, timings, nsp=True, egg=False, progress=None
    ):

        dir = path.join(self._dir, data_dir)
        makedirs(dir, exist_ok=True)

        tasks = []
        if vowels:
            tasks.append("iau")
        if phrase:
            tasks.append("phrase")
        if timings:
            tasks.extend((t for t in task2key.keys() if t not in ("iau", "phrase")))
            if not nsp:  # must download acoustic data
                nsp = True

        files = []
        for id in ids:
            if nsp:
                if vowels:
                    files.append(("iau", False, id, f"{id}-iau.nsp"))
                if phrase:
                    files.append(("phrase", False, id, f"{id}-phrase.nsp"))
            if egg:
                if vowels:
                    files.append(("iau", True, id, f"{id}-iau-egg.egg"))
                if phrase:
                    files.append(("phrase", True, id, f"{id}-phrase-egg.egg"))

        full_data = []
        if not vowels and timings:
            # iau files are expected to be already present for all ids
            for id in ids:
                try:
                    file = path.join(self._dir, data_dir, f"{id}-iau.nsp")
                    _, x = nspfile.read(file)
                    full_data.append((id, x))
                except:
                    raise ValueError(
                        f"The iau task nsp data not found. Set vowels=True"
                        if nsp
                        else f"Need the iau task nsp data to resolve vowel timings: missing {file}"
                    )

        with TemporaryDirectory() as tdir:
            # download & extract files to tdir
            dl_files = download_data(tdir, ids, tasks, nsp, egg, progress)

            # validate iau & phrase files
            save_files = []
            miss_files = []
            for entry in files:
                file = path.join(tdir, entry[-1])
                if file in dl_files:
                    try:
                        _, x = nspfile.read(file)
                        file_ok = True
                    except:
                        file_ok = False
                        miss_files.append(entry[:-1])

                    if file_ok:
                        try:
                            shutil.move(file, dir)
                        except Exception as e:
                            if not path.isfile(path.join(dir, entry[-1])):
                                raise e
                        save_files.append(entry)
                        if timings:
                            full_data.append((entry[-2], x))

            if len(files):
                self._add_files(FileSeries(save_files))
            if len(miss_files):
                self._add_missing(MissingDataIndex(miss_files))

            # if timing data is needed for the task, download individual segments and compute
            if timings:
                timing_data = []
                for id, x in full_data:
                    df = TimingDataFrame(
                        {"ID": id, "F": vowel_tasks, "N0": -1, "N1": -1}
                    )
                    for task in vowel_tasks:
                        f = path.join(tdir, f"{id}-{task}.nsp")
                        if path.isfile(f):
                            try:
                                df.loc[(id, task)] = align_data(x, f)
                            except:
                                pass
                    timing_data.append(df)
                self._add_timings(pd.concat(timing_data))

    def iter_data(
        self,
        task: TaskType = None,
        egg: bool = False,
        cached_only: bool = False,
        auxdata_fields: List[DataField] = None,
        normalize: bool = True,
        padding: float = None,
        **filters,
    ) -> Iterator[VoiceData | VoiceDataPlusInfo]:
        """iterate over queried recordings (yielding data samples)

        :param task: vocal task type, defaults to None
        :type task: TaskType, optional
        :param egg: True for EGG, False for audio, defaults to False
        :type egg: bool, optional
        :param cached_only: True to block downloading, defaults to False
        :type cached_only: bool, optional
        :param auxdata_fields: Additional recording data to return, defaults to None
        :type auxdata_fields: List[DataField], optional
        :param normalize: True to convert sample values to float between [-1.0,1.0], defaults to True
        :type normalize: bool, optional
        :return: voice data namedtuple: id, fs, data array, (optional) aux info
        :rtype: Iterator[VoiceData|VoiceDataPlusInfo]

        Iterates over all the DataFrame columns, returning a tuple with the column name and the content as a Series.

        Yields

            labelobject

                The column names for the DataFrame being iterated over.
            contentSeries

                The column entries belonging to each label, as a Series.



        Valid values of `auxdata_fields` argument
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "pathology" and "#")
        * "Pathologies" - A list containing all the original "pathology" associated with the subject
        * "NORM" - True if normal data, False if pathological data
        * "MDVP" - Short-hand notation to include all the MDVP parameter measurements: from "Fo" to "PER"

        Valid `filters` keyword arguments
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "pathology" and "#")
        * "Pathologies" - A list containing all the original "pathology" associated with the subject
        * "NORM" - True if normal data, False if pathological data

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values
        """

        if not task:
            task = self.default_task

        df = self.get_files(task, egg, cached_only, True, auxdata_fields, **filters)

        for id, file, *auxdata in df.itertuples():

            timing = None if task in ("iau", "phrase") else self._df_timing.loc[id]

            framerate, x = self._read_file(file, task, timing, normalize, padding)

            yield VoiceData(
                id, framerate, x
            ) if auxdata_fields is None else VoiceDataPlusInfo(
                id, framerate, x, auxdata
            )

    def read_data(
        self,
        id: int,
        task: TaskType = None,
        cached_only: bool = False,
        egg: bool = False,
        auxdata_fields: List[DataField] = None,
        normalize: bool = True,
        padding: float = None,
    ) -> tuple[int, np.array] | tuple[int, np.array, pd.Series]:

        if not task:
            task = self.default_task

        # get the file name
        file = self.get_files(task, egg, cached_only, True, auxdata_fields, ID=id).loc[
            id
        ]

        timing = None if task in ("iau", "phrase") else self._df_timing.loc[id]
        data = self._read_file(file.loc["File"], task, timing, normalize, padding)

        return data if auxdata_fields is None else (*data, file.iloc[1:])

    def _read_file(self, file, task, timing, normalize=True, padding=None):

        fs, x = nspfile.read(file)

        if timing is not None:

            if not padding:
                padding = self.default_padding

            if padding:
                # sort timing by N0
                ts = timing.sort_values("N0")
                i = ts.index.get_loc(task)

                padding = round(padding * fs)
                tstart, tend = ts.iloc[i]
                tstart -= padding
                tend += padding

                if padding > 0.0:
                    talt = ts.iloc[i - 1, 1] if i > 0 else 0
                    if tstart < talt:
                        tstart = talt

                    if i + 1 < len(ts):
                        talt = ts.iloc[i + 1, 0]
                        if tend > talt:
                            tend = talt
            else:
                tstart, tend = timing.loc[task]

            x = x[tstart:tend]

        if normalize:
            x = x / 2.0**15

        return fs, x

    def __getitem__(self, key):
        return self.read_data(key)
