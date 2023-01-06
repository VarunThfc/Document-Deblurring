# Split dataset into train and eval
import os
import glob
import pandas as pd

files = glob.glob("publaynet/train/*.jpg")
if not os.path.exists("publaynet/val"):
    os.mkdir("publaynet/val")
files = pd.DataFrame(pd.Series(files))
files.columns = ["name"]

files.loc[:,"indexer"] = files.name.str.split("_")

files.loc[:,'indexer'] = files.loc[:,'indexer'].apply(lambda x: int(x[-1].split(".")[0]))

val_files = files[~files.indexer.isin(range(1,15))].name.to_list()

for file in val_files:
    os.replace(file,file.replace("train","val"))