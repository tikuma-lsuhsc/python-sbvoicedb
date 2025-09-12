from os import path
from shutil import copyfile
from .utils import *

import re
from os import path
from glob import glob
import nspfile
import numpy as np
from os import path


def fix_incomplete_nsp(file: str, root_dir: str) -> str:
    """fix an incomplete NSP file without data size information

    :param file: path of the nsp file
    :returns: fixed nsp file (with -fixed postfix)
    """

    with open(path.join(root_dir, file), "rb") as f:
        b = bytearray(f.read())

    sz = len(b) - 12
    b[8:12] = sz.to_bytes(4, "little", signed=False)

    i = b.find("SDA_".encode("utf8")) + 4
    nbytes = len(b) - i - 4
    b[i : i + 4] = nbytes.to_bytes(4, "little", signed=False)

    nsamples = nbytes // 2  # int16_t data
    b[0x2C:0x30] = nsamples.to_bytes(4, "little", signed=False)

    fileparts = path.splitext(file)
    outfile = "".join([fileparts[0], "-fixed", fileparts[1]])
    with open(path.join(root_dir, outfile), "wb") as f:
        f.write(b)

    return outfile


def swap_nsp_egg(
    nspfile: str, eggfile: str, root_dir: str, n: int | None = None
) -> tuple[str, str]:

    def rename(file):
        fileparts = path.splitext(file)
        return "".join([fileparts[0], "-fixed", fileparts[1]])

    newnsp = rename(nspfile)
    newegg = rename(eggfile)

    if n is None:
        copyfile(path.join(root_dir, nspfile), path.join(root_dir, newegg))
        copyfile(path.join(root_dir, eggfile), path.join(root_dir, newnsp))
    else:
        with open(path.join(root_dir, nspfile), "rb") as f:
            b_nsp = bytearray(f.read())

        with open(path.join(root_dir, eggfile), "rb") as f:
            b_egg = bytearray(f.read())

        i = b_nsp.find("SDA_".encode("utf8")) + 8  # SDA_+SIZE
        i += n * 2
        b_nsp[i:], b_egg[i:] = b_egg[i:], b_nsp[i:]

        with open(path.join(root_dir, newnsp), "wb") as f:
            f.write(b_nsp)
        with open(path.join(root_dir, newegg), "wb") as f:
            f.write(b_egg)

    return newnsp, newegg


data_dir = "data"

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
