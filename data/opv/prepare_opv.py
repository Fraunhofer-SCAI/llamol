import math
import os
import struct
import logging
from tqdm import tqdm
import csv
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

# Taken from here https://torchdrug.ai/docs/_modules/torchdrug/utils/file.html#download
def download(url, path, save_file=None, md5=None):
    """
    Download a file from the specified url.
    Skip the downloading step if there exists a file satisfying the given MD5.

    Parameters:
        url (str): URL to download
        path (str): path to store the downloaded file
        save_file (str, optional): name of save file. If not specified, infer the file name from the URL.
        md5 (str, optional): MD5 of the file
    """
    from six.moves.urllib.request import urlretrieve

    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file) or compute_md5(save_file) != md5:
        logger.info("Downloading %s to %s" % (url, save_file))
        urlretrieve(url, save_file)
    return save_file



def smart_open(file_name, mode="rb"):
    """
    Open a regular file or a zipped file.

    This function can be used as drop-in replacement of the builtin function `open()`.

    Parameters:
        file_name (str): file name
        mode (str, optional): open mode for the file stream
    """
    import bz2
    import gzip

    extension = os.path.splitext(file_name)[1]
    if extension == '.bz2':
        return bz2.BZ2File(file_name, mode)
    elif extension == '.gz':
        return gzip.GzipFile(file_name, mode)
    else:
        return open(file_name, mode)


def extract(zip_file, member=None):
    """
    Extract files from a zip file. Currently, ``zip``, ``gz``, ``tar.gz``, ``tar`` file types are supported.

    Parameters:
        zip_file (str): file name
        member (str, optional): extract specific member from the zip file.
            If not specified, extract all members.
    """
    import gzip
    import shutil
    import zipfile
    import tarfile

    zip_name, extension = os.path.splitext(zip_file)
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]
    save_path = os.path.dirname(zip_file)

    if extension == ".gz":
        member = os.path.basename(zip_name)
        members = [member]
        save_files = [os.path.join(save_path, member)]
        for _member, save_file in zip(members, save_files):
            with open(zip_file, "rb") as fin:
                fin.seek(-4, 2)
                file_size = struct.unpack("<I", fin.read())[0]
            with gzip.open(zip_file, "rb") as fin:
                if not os.path.exists(save_file) or file_size != os.path.getsize(save_file):
                    logger.info("Extracting %s to %s" % (zip_file, save_file))
                    with open(save_file, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
    elif extension in [".tar.gz", ".tgz", ".tar"]:
        tar = tarfile.open(zip_file, "r")
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            logger.info("Extracting %s from %s to %s" % (member, zip_file, save_files[0]))
        else:
            members = tar.getnames()
            save_files = [os.path.join(save_path, _member) for _member in members]
            logger.info("Extracting %s to %s" % (zip_file, save_path))
        for _member, save_file in zip(members, save_files):
            if tar.getmember(_member).isdir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or tar.getmember(_member).size != os.path.getsize(save_file):
                with tar.extractfile(_member) as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    elif extension == ".zip":
        zipped = zipfile.ZipFile(zip_file)
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            logger.info("Extracting %s from %s to %s" % (member, zip_file, save_files[0]))
        else:
            members = zipped.namelist()
            save_files = [os.path.join(save_path, _member) for _member in members]
            logger.info("Extracting %s to %s" % (zip_file, save_path))
        for _member, save_file in zip(members, save_files):
            if zipped.getinfo(_member).is_dir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or zipped.getinfo(_member).file_size != os.path.getsize(save_file):
                with zipped.open(_member, "r") as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    else:
        raise ValueError("Unknown file extension `%s`" % extension)

    if len(save_files) == 1:
        return save_files[0]
    else:
        return save_path



def compute_md5(file_name, chunk_size=65536):
    """
    Compute MD5 of the file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    import hashlib

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()



def get_line_count(file_name, chunk_size=8192*1024):
    """
    Get the number of lines in a file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    count = 0
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            count += chunk.count(b"\n")
            chunk = fin.read(chunk_size)
    return count


class OPV:
    """
    Quantum mechanical calculations on organic photovoltaic candidate molecules.

    Statistics:
        - #Molecule: 94,576
        - #Regression task: 8

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    train_url = "https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e433d5750d6/download/" \
                "b69cf9a5-e7e0-405b-88cb-40df8007242e"
    valid_url = "https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e433d5750d6/download/" \
                "1c8e7379-3071-4360-ba8e-0c6481c33d2c"
    test_url = "https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e433d5750d6/download/" \
               "4ef40592-0080-4f00-9bb7-34b25f94962a"
    train_md5 = "16e439b7411ea0a8d3a56ba4802b61b1"
    valid_md5 = "3aa2ac62015932ca84661feb5d29adda"
    test_md5 = "bad072224f0755478f0729476ca99a33"
    target_fields = ["gap", "homo", "lumo", "spectral_overlap", "gap_extrapolated", "homo_extrapolated",
                     "lumo_extrapolated", "optical_lumo_extrapolated"]

    def read_csv(self, csv_file, smiles_field="smiles", target_fields=None, verbose=0):
        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, get_line_count(csv_file)))
            fields = next(reader)
            smiles = []
            targets = defaultdict(list)
            for i, values in enumerate(reader):
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        pass
                        # value = eval(value)
                        # if value == "":
                        #     value = math.nan
                        # targets[field].append(value)

        return smiles, targets

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_zip_file = download(self.train_url, path, save_file="mol_train.csv.gz", md5=self.train_md5)
        valid_zip_file = download(self.valid_url, path, save_file="mol_valid.csv.gz", md5=self.valid_md5)
        test_zip_file = download(self.test_url, path, save_file="mol_test.csv.gz", md5=self.test_md5)
        train_file = extract(train_zip_file)
        valid_file = extract(valid_zip_file)
        test_file = extract(test_zip_file)

        train_smiles, train_targets = self.read_csv(train_file, smiles_field="smile", target_fields=self.target_fields)
        valid_smiles, valid_targets = self.read_csv(valid_file, smiles_field="smile", target_fields=self.target_fields)
        test_smiles, test_targets = self.read_csv(test_file, smiles_field="smile", target_fields=self.target_fields)
        self.num_train = len(train_smiles)
        self.num_valid = len(valid_smiles)
        self.num_test = len(test_smiles)

        smiles = train_smiles + valid_smiles + test_smiles
        targets = {k: train_targets[k] + valid_targets[k] + test_targets[k] for k in train_targets}

        # self.load_smiles(smiles, targets, verbose=verbose, **kwargs)
        print(smiles[:10])
        df_out = pd.DataFrame({"smiles": smiles})
        df_out.to_parquet(os.path.join(os.path.dirname(__file__), "opv.parquet"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cwd = os.path.join(os.path.dirname(__file__), "download")
    os.makedirs(cwd,exist_ok=True)
    d = OPV(cwd)
    