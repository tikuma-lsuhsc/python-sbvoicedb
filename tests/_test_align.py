from sbvoicedb import process as proc
import pandas as pd

id = 74
file = "tests/db/files/74-iau.nsp"
segm_dir = "tests/db/files"

out = proc.align_vowels(id, file, segm_dir)

out = proc.pad_timing(out.loc[id], 'a_n', 50000, 0.3)
print(out)
