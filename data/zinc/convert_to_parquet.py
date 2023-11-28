import pandas as pd
import os.path as osp
import os
from tqdm import tqdm
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import shutil
cwd = osp.abspath(osp.dirname(__file__))
zinc_path = os.path.join(cwd, "zinc_complete")
alls_dirs = [
    osp.join(zinc_path, f)
    for f in os.listdir(zinc_path)
    if osp.isdir(osp.join(zinc_path, f))
]


print("Number of dirs: ", len(alls_dirs))
all_dfs = []
for d in alls_dirs:
    print(f"Read: {d    }")
    df = dd.read_csv(
        os.path.join(cwd, "zinc_complete", f"{d}/*.txt"),
        sep="\t",
        usecols=["smiles"],
    )
    all_dfs.append(df)

concatenated_df = dd.concat(all_dfs)
# res = df["logp"].map_partitions(lambda d, bins: pd.cut(d, bins), 25).compute()
# print(res)

print("Writing")
# print(df)
# name_function = lambda x: f"zincfull-{x}.parquet"
concatenated_df = concatenated_df.repartition(npartitions=1)
concatenated_df = concatenated_df.reset_index(drop=True)
concatenated_df.to_parquet(
    os.path.join(cwd, "zinc_processed"),
)
print("Done Writing")
print(len(concatenated_df))
shutil.copy(
    os.path.join(cwd, "zinc_processed", "part.0.parquet"),
    os.path.join(cwd, "zinc_processed.parquet")
)

# df = None
# for d in tqdm(alls_dirs):
#     if df is not None:
#         print(len(df))
#     files = [osp.join(d,f) for f in os.listdir(d)]
#     for f in files:
#         try:
#             df_extra = pd.read_csv(f,sep="\t")
#         except Exception as e:
#             print(f"Got error {f}: {e}")
#             continue
#         # print(df)
#         if df is None:
#             df = df_extra

#         else:
#             df = df.append(df_extra)


# df.to_parquet(osp.join(cwd, "zinc_combined.parquet"))