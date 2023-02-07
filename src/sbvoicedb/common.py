from os import path
import pandas as pd

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
