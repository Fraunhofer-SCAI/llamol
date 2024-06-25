import pandas as pd
from rdkit import Chem
import numpy as np
df = pd.read_parquet("OrganiX13.parquet")
df = df.sample(n=100_000)
print("Calculating...")
context_smarts = '[#16]1:[#6]:[#6]:[#6]:[#6]:1'
mols = df["smiles"].map(Chem.MolFromSmiles)
matches = mols.map(lambda m: m.HasSubstructMatch(Chem.MolFromSmarts(context_smarts)))

print("Matches", np.count_nonzero(matches), "from", len(mols))
print("Result smiles", df["smiles"][matches])

