import pandas as pd
from rdkit import Chem
import multiprocessing
import numpy as np
from tqdm.contrib.concurrent import process_map

def cal_canonoical():
    df = pd.read_parquet("OrganiX13_orig.parquet")

    print("Loaded dataset")
    def convert_to_canonical(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("No")
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    # Use multiprocessing to parallelize the conversion process
    workers = int(multiprocessing.cpu_count() * 0.5)
    print("Using", workers)
    df["smiles"] = process_map(convert_to_canonical, df["smiles"], max_workers=workers, chunksize = 2000)
    df.to_parquet("OrganiX13.parquet")
    mask = ~df["canonical_smiles"].isna()
    print("Number non nan: ", np.count_nonzero(mask), "from", len(df))
    df["smiles"] = df["smiles"][mask]

    # Count the number of duplicate canonical SMILES
    duplicate_count = df.duplicated(subset="smiles").sum()

    print("Number of duplicate canonical SMILES:", duplicate_count)

# cal_canonoical()

df = pd.read_parquet("CanSMI_OrganiX13.parquet")
print(len(df))
df["smiles"] = df["canonical_smiles"]
df = df.drop(columns=["canonical_smiles"])
df = df.drop_duplicates(subset=['smiles'])
df.reset_index(inplace=True, drop=True)
print(df.head())
print(len(df))
df.to_parquet("OrganiX13.parquet")

# duplicate_counts = df["canonical_smiles"].value_counts()

# # Create a dictionary with the duplicate SMILES and their frequencies
# duplicate_dict = duplicate_counts[duplicate_counts > 1].to_dict()

# # Sort the dictionary by the frequency count in descending order
# sorted_duplicate_dict = dict(sorted(duplicate_dict.items(), key=lambda x: x[1], reverse=False))

# # Print the sorted duplicate dictionary
# for smile, count in list(sorted_duplicate_dict.items())[-100:]:
#     print(smile, ":", count)

# print(len(sorted_duplicate_dict))

