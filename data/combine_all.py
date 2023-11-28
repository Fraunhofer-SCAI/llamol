import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import multiprocessing

from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

np.random.seed(42)

def calcLogPIfMol(smi):
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return Descriptors.MolLogP(m)
    else:
        return None

def calcMol(smi):
    return Chem.MolFromSmiles(smi)

def calcMolWeight(smi):
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.ExactMolWt(mol)

def calcSascore(smi):
    mol = Chem.MolFromSmiles(smi)
    
    return sascorer.calculateScore(mol)

def calculateValues(smi: pd.Series):
    
    
    with multiprocessing.Pool(8) as pool:
        print("Starting logps")
        logps = pool.map(calcLogPIfMol, smi)
        print("Done logps")        
        valid_mols = ~pd.isna(logps)
        logps = pd.Series(logps)[valid_mols]
        smi = pd.Series(smi)[valid_mols]
        logps.reset_index(drop=True,inplace=True)
        smi.reset_index(drop=True,inplace=True)
        print("Starting mol weights")
        mol_weights = pool.map(calcMolWeight, smi)  
        print("Done mol weights")
        print("Starting sascores")
        sascores = pool.map(calcSascore, smi) 
        print("Done sascores")
        
    return smi, logps, mol_weights,sascores

def calculateProperties(df):
    
    smi, logps, mol_weights,sascores = calculateValues(df["smiles"])
    out_df = pd.DataFrame({"smiles": smi, "logp":logps, "mol_weight":mol_weights, "sascore":sascores })
        
    return out_df

if __name__ == "__main__":
    
    cwd = os.path.dirname(__file__)
    
    print("df_pc9")
    df_pc9 = pd.read_parquet(os.path.join(cwd, "Full_PC9_GAP.parquet"))
    df_pc9 = calculateProperties(df_pc9)


    print("df_zinc_full")
    
    df_zinc_full = pd.read_parquet(
        os.path.join(cwd, "zinc", "zinc_processed.parquet")
    )
    df_zinc_full = df_zinc_full.sample(n=5_000_000)
    df_zinc_full = calculateProperties(df_zinc_full)

    
    print("df_zinc_qm9")
    df_zinc_qm9 = pd.read_parquet(os.path.join(cwd,"qm9_zinc250k_cep", "qm9_zinc250_cep.parquet"))
    df_zinc_qm9 = calculateProperties(df_zinc_qm9)

    print("df_opv")
    df_opv = pd.read_parquet(os.path.join(cwd,"opv", "opv.parquet"))
    df_opv = calculateProperties(df_opv)


    print("df_reddb")
    # Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F3QFSQ
    df_reddb = pd.read_parquet(os.path.join(cwd,"RedDB_Full.parquet"))
    df_reddb = calculateProperties(df_reddb)

    print("df_chembl")
    df_chembl = pd.read_parquet(
        os.path.join(cwd, "chembl_log_sascore.parquet")
    )
    df_chembl = calculateProperties(df_chembl)

    
    print("df_pubchemqc_2017")
    df_pubchemqc_2017 = pd.read_parquet(
        os.path.join(cwd, "pubchemqc_energy.parquet")
    )
    df_pubchemqc_2017 = calculateProperties(df_pubchemqc_2017)


    print("df_pubchemqc_2020")
    
    df_pubchemqc_2020 = pd.read_parquet(
        os.path.join(cwd, "pubchemqc2020_energy.parquet")
    )
    df_pubchemqc_2020 = calculateProperties(df_pubchemqc_2020)


    
    df_list = [
        df_zinc_qm9,
        df_opv,
        df_pubchemqc_2017,
        df_pubchemqc_2020,
        df_zinc_full,
        df_reddb,
        df_pc9,
        df_chembl,
    ]
    
    print(f"ZINC QM9 {len(df_zinc_qm9)}")
    print(f"df_opv {len(df_opv)}")
    print(f"df_pubchemqc_2017 {len(df_pubchemqc_2017)}")
    print(f"df_pubchemqc_2020 {len(df_pubchemqc_2020)}")
    print(f"df_zinc_full {len(df_zinc_full)}")
    print(f"df_reddb {len(df_reddb)}")
    print(f"df_pc9 {len(df_pc9)}")
    print(f"df_chembl {len(df_chembl)}")
    
    
    


    all_columns = [
        "smiles",
        "logp",
        "sascore",
        "mol_weight"
    ]  # set([*df_zinc_qm9.columns.tolist(),*df_pubchemqc_2017.columns.tolist(),*df_pubchemqc_2020.columns.tolist(),*df_zinc_full.columns.tolist()] )
    print("concatenting")
    df = pd.concat(
        df_list, axis=0, ignore_index=True
    )  # pd.DataFrame(columns=all_columns)
    df = df[all_columns]  # .fillna(0)
    # df = df.sample(n=7_500_000)
    df.reset_index(drop=True, inplace=True)
    df["mol_weight"] = df["mol_weight"] / 100.0 
    
    print(df.head())
    print("saving")
    print("Combined len:", len(df))
    df.to_parquet(
        os.path.join(cwd, "OrganiX13.parquet")
    )
