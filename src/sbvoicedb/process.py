from .common import *

import re
from os import path, remove
from glob import glob
import nspfile
import numpy as np
import zipfile
from shutil import copyfileobj
from os import path


# unzip
def extract(zippath, dst, progress=None):

    files = []
    with zipfile.ZipFile(zippath, "r") as f:
        for i in f.infolist():
            if getattr(i, "file_size", 0):  # file
                try:
                    filepath = path.join(dst, path.basename(i.filename))
                    with f.open(i) as fi, open(filepath, "xb") as fo:
                        copyfileobj(progress.io_wrapper(fi) if progress else fi, fo)
                    files.append(filepath)
                except FileExistsError:
                    pass

    return files


def align_data(x, segm_file):
    _, y = nspfile.read(segm_file)
    try:
        n0 = x.tobytes().index(y.tobytes()) // x.itemsize
        return n0, n0 + len(y)
    except:
        nx = len(x)
        ny = len(y)
        d0 = ny // 2
        d1 = ny - d0
        i = np.where(x == y[d0])[0]
        ix0 = [(ii - d0, ii + d1) for ii in i]
        ix = [(max(ii0, 0), min(ii1, nx)) for ii0, ii1 in ix0]
        iy = [(ii[0] - ii0[0], ny - ii0[1] + ii[1]) for ii, ii0 in zip(ix, ix0)]
        s = [np.sum(x[i0:i1] == y[j0:j1]) for (i0, i1), (j0, j1) in zip(ix, iy)]
        return ix[np.argmax(s)]


def validate_file(file):
    egg = file.endswith(".egg")
    m = re.search(
        r"(\d+)-(iau|phrase)-egg.egg$" if egg else r"(\d+)-(iau|phrase).nsp$",
        file,
    )
    if not m:
        raise ValueError(f"Invalid file name: {file}")

    idx = (m[2], egg, int(m[1]))
    try:
        nspfile.read(file)
        return *idx, path.basename(file)
    except:
        remove(file)
        return *idx, ""


def align_vowels(id, file, segm_dir):

    re_pattern = re.compile(rf"(?:^|\D){id}-([aiu]_(?:[nlh]|lhl)).nsp$")
    glob_pattern = path.join(segm_dir, f"{id}-*.nsp")
    segm_files = glob(glob_pattern)

    if not len(segm_files):
        return TimingDataFrame()

    _, x = nspfile.read(file)

    def align_data(x, segm_file):
        _, y = nspfile.read(segm_file)
        try:
            n0 = x.tobytes().index(y.tobytes()) // x.itemsize
            return n0, n0 + len(y)
        except:
            nx = len(x)
            ny = len(y)
            d0 = ny // 2
            d1 = ny - d0
            i = np.where(x == y[d0])[0]
            ix0 = [(ii - d0, ii + d1) for ii in i]
            ix = [(max(ii0, 0), min(ii1, nx)) for ii0, ii1 in ix0]
            iy = [(ii[0] - ii0[0], ny - ii0[1] + ii[1]) for ii, ii0 in zip(ix, ix0)]
            s = [np.sum(x[i0:i1] == y[j0:j1]) for (i0, i1), (j0, j1) in zip(ix, iy)]
            return ix[np.argmax(s)]

    return TimingDataFrame(
        [
            (id, m[1], *align_data(x, f))
            for f in segm_files
            if (m := re_pattern.search(f))
        ]
    )


def pad_timing(timing, task, fs, padding=0.0):

    if padding:
        # sort timing by N0
        ts = timing.sort_values("N0")
        i = ts.index.get_loc(task)

        padding = round(padding * fs)
        tstart, tend = ts.iloc[i]
        tstart -= padding
        tend += padding

        if padding > 0.0:
            if i > 0:
                i0 = i - 1
                while ts.iloc[i, 0] < ts.iloc[i0, 1]:
                    i0 -= 1
                    if i0 < 0:
                        raise RuntimeError(
                            f"something is wrong with timing data for id={id} ({task})"
                        )
                talt = ts.iloc[i0, 1]
                if tstart < talt:
                    tstart = talt

            i0 = i + 1
            if i0 < len(ts):
                while ts.iloc[i, 1] > ts.iloc[i0, 0]:
                    i0 += 1
                    if i0 == len(ts):
                        raise RuntimeError(
                            f"something is wrong with timing data for id={id} ({task})"
                        )
                talt = ts.iloc[i0, 0]
                if tend > talt:
                    tend = talt
                    
        return tstart, tend
    else:
        return timing.loc[task]
