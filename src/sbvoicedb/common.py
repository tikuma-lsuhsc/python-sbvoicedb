from os import path
from shutil import copyfile

# import pandas as pd


def fix_incomplete_nsp(file: str) -> str:
    """fix an incomplete NSP file without data size information

    :param file: path of the nsp file
    :returns: fixed nsp file (with -fixed postfix)
    """

    with open(file, "rb") as f:
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
    with open(outfile, "wb") as f:
        f.write(b)

    return outfile


def swap_nsp_egg(nspfile: str, eggfile: str, n: int | None = None) -> tuple[str, str]:

    def rename(file):
        fileparts = path.splitext(file)
        return "".join([fileparts[0], "-fixed", fileparts[1]])

    newnsp = rename(nspfile)
    newegg = rename(eggfile)

    if n is None:
        copyfile(nspfile, newegg)
        copyfile(eggfile, newnsp)
    else:
        with open(nspfile, "rb") as f:
            b_nsp = bytearray(f.read())

        with open(eggfile, "rb") as f:
            b_egg = bytearray(f.read())

        i = b_nsp.find("SDA_".encode("utf8")) + 4
        i += 8  # beginning of the data
        i += n * 2
        b_nsp[i:], b_egg[i:] = b_egg[i:], b_nsp[i:]

        with open(newnsp, "wb") as f:
            f.write(b_nsp)
        with open(newegg, "wb") as f:
            f.write(b_egg)

    return newnsp, newegg


db_dtypes = {
    "T": "string",
    "D": "datetime64[ns]",
    "G": "string",
    "Pathologies": "string",
    "Remark w.r.t. diagnosis": "string",
    "B": "string",
}
data_dir = "files"

task2key = {
    "iau": "s_export_vowels",
    "i_n": "s_export_i_n",
    "i_h": "s_export_i_h",
    "i_l": "s_export_i_l",
    "i_lhl": "s_export_i_lhl",
    "a_n": "s_export_a_n",
    "a_h": "s_export_a_h",
    "a_l": "s_export_a_l",
    "a_lhl": "s_export_a_lhl",
    "u_n": "s_export_u_n",
    "u_h": "s_export_u_h",
    "u_l": "s_export_u_l",
    "u_lhl": "s_export_u_lhl",
    "phrase": "s_export_phrase",
}

vowel_tasks = tuple(t for t in task2key if t not in ("iau", "phrase"))

dtypes_files = {"Task": "string", "IsEgg": "bool", "ID": "Int64", "File": "string"}
dtypes_timing = {"ID": "Int64", "F": "string", "N0": "Int64", "N1": "Int64"}


def FileSeries(data=None):
    columns = None if isinstance(data, dict) else list(dtypes_files.keys())
    return (
        pd.DataFrame(data, columns=columns)
        .astype(dtypes_files)
        .set_index(["Task", "IsEgg", "ID"])
        .sort_index()["File"]
    )


def MissingDataIndex(data=None):
    columns = None if isinstance(data, dict) else list(dtypes_files.keys())[:-1]
    return (
        pd.DataFrame(data, columns=columns)
        .astype({k: v for k, v in dtypes_files.items() if k != "File"})
        .set_index(["Task", "IsEgg", "ID"])
        .sort_index()
        .index
    )


def TimingDataFrame(data=None):
    return (
        pd.DataFrame(data, columns=list(dtypes_timing.keys()))
        .astype(dtypes_timing)
        .set_index(["ID", "F"])
    )
