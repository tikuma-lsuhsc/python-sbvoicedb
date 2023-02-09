from sbvoicedb import process as proc

id = 74
file = "tests/db/files/74-iau.nsp"
segm_dir = "tests/db/files"

out = proc.align_vowels(id, file, segm_dir)
print(out)
