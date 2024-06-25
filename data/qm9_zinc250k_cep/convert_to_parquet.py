import pandas as pd
import requests
import hashlib
import os
# Download and read zinc_properties file
zinc_url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
zinc_md5 = "b59078b2b04c6e9431280e3dc42048d5"
zinc_filename = "zinc_properties.csv"

response = requests.get(zinc_url)
downloaded_data = response.content

downloaded_md5 = hashlib.md5(downloaded_data).hexdigest()
if zinc_md5 == downloaded_md5:
    with open(zinc_filename, 'wb') as f:
        f.write(downloaded_data)
    print(f"File '{zinc_filename}' downloaded and saved.")
else:
    raise ValueError("MD5 checksum does not match")

zinc_df = pd.read_csv(zinc_filename)
zinc_df = zinc_df[["smiles"]]

cwd = os.path.dirname(__file__)

qm9_filename = os.path.join(cwd,"QM9IsoFull.csv")
cep_filename = os.path.join(cwd,"cep-processed.csv")

qm9_df = pd.read_csv(qm9_filename, sep="|")
qm9_df = qm9_df[["smiles"]]

cep_df = pd.read_csv(cep_filename)
cep_df = cep_df[["smiles"]]

# Combine the dataframes into one large dataframe
combined_df = pd.concat([zinc_df, qm9_df, cep_df], axis=0)

# Save the combined dataframe to a Parquet file
output_filename = "qm9_zinc250_cep.parquet"
combined_df.to_parquet(output_filename, index=False)
print(f"Combined dataframe saved to '{output_filename}' as Parquet file.")