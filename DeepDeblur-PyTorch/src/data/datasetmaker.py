import os
import glob
import pandas as pd

files = glob.glob("../../../../data/publaynet/train/*.jpg")
if not os.path.exists("../../../../data/publaynet/val"):
    os.mkdir("../../../../data/publaynet/val")
files = pd.DataFrame(pd.Series(files))
files.columns = ["name"]

print(files.columns, "HW")
files.loc[:,"indexer"] = files.name.str.split("_")

files.loc[:,'indexer'] = files.loc[:,'indexer'].apply(lambda x: int(x[-1].split(".")[0]))

val_files = files[~files.indexer.isin(range(1,15))].name.to_list()


for file in val_files:
    os.replace(file,file.replace("train","val"))
