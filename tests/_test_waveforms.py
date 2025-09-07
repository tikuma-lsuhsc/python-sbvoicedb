from sbvoicedb import database
import nspfile
from os import path
from matplotlib import pyplot as plt

fs,x  = nspfile.read("/home/kesh/data/SVD/data/713/vowels/713-i_n-fixed.nsp")
fs,yt  = nspfile.read("/home/kesh/data/SVD/data/713/vowels/713-i_n-fixed.nsp")

plt.plot(x)
plt.show()
# db = database.SbVoiceDb("/home/kesh/data/SVD", download_mode="once")
