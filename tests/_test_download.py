from sbvoicedb import download as dl

dst = "./tests/db/files"
id = 74
dl.download_data(dst, [id], tasks=["iau", "a_n"])
