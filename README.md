# Llamol

<p align="center">
  <img src="assets/llamol.png" width="300" height="300" alt="LLamol">
</p>

This is the official repository for the paper ["LLamol: A Dynamic Multi-Conditional Generative Transformer for De Novo Molecular Design"](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00863-8).
In this repository are the weights for LLamol (`out/llama2-M-Full-RSS-Canonical-Canonical.pt`) and the dataset OrganiX13.

Image made with [Hotpot.ai](https://hotpot.ai/art-generator) 
## Installation
Install using Mamba to be fast: https://mamba.readthedocs.io/en/latest/micromamba-installation.html
 

```bash
$ "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
$ micromamba env create -f torch2-env.yaml
$ micromamba activate torch2-llamol
$ python sample.py
```
# Download and preprocess the OrganiX13 dataset:
If you want to train with the full 12.5 Million dataset do the following steps. These are *not* necessary if you just want to use the model for inference:
1. Download and preprocess the OPV dataset by running `/data/opv/prepare_opv.py`
2. Download and preprocess the ZINC dataset by running `/data/zinc/zinc_complete/run_download.py` followed by `/data/zinc/convert_to_parquet.py`
(we recommend at least 16GB RAM for this)
3. Download and preprocess the ZINC dataset by running `/data/qm9_zinc250k_cep/convert_to_parquet.py`

4. Run `data/combine_all.py` to combine the dataset to `data/OrganiX13.parquet` (this can take a while, especially on the zinc dataset. In total it took ~2 hours when using my Laptop, which has 16 GB ram and an Intel i7 10th Gen)
5. Run `preprocess_dataset.py` which should create the file `.cache/processed_dataset_None.pkl`

Now you can use that in the training of the model by specifing the file under the `processed_dataset_ckpt` of the training .yaml files.



# Interactive Demo

After installation you can play around with the model using the `demonstrator.ipynb` file. Just run all and scroll down to the last cell. 
After a short time there should be a UI where you can play around with the model.


## Training

First the env needs to be activated so:
```bash
$ conda activate torch2-llamol # When installed with conda instead of micromamba
OR
$ micromamba activate torch2-llamol
``````

To train locally you can run:
```bash
# To set the config that you want to train with
$ python train.py train=llama2-M-Full-RSS-Canonical
```

Parameters can also be overriden by using the following, for example:
```bash
$ python train.py train=llama2-M-Full-RSS-Canonical train.model.dim=1024
```
For more information look at [Hydra](https://hydra.cc/docs/1.3/intro/)

To start a job on a SLURM cluster use the following script:
```bash
$ sbatch trainLLamaMol.sh 
``````

## Training Multi-GPU on 1 Node with multiple GPUS (nproc_per_node)
```bash
torchrun --standalone --max_restarts=3  --nnodes=1 --nproc_per_node=2 --rdzv-backend=c10d  --rdzv-endpoint="$localhost:12345" train.py train=llama2-M-Full-RSS-Canonical > "train_runs/run_MultiGPU.out" 
```
## Training Multi-GPU on 1 Node with multiple GPUS on a Cluster
Currently there is only one script to train with DDP. To change the number of GPUS in that script you have to change the bash script itself.
TODO: Make it more dynamic, with allowing console commands to change the number of GPUS etc.
```bash
sbatch trainLLamaMolDDPSingleNode.sh
```

## Sampling
Sampling can be changed by the OPTIONAL parameters as shown below.
```bash
$ python sample.py --help

$ python sample.py --num_samples 2000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt"  --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --seed 4312 --context_cols logp sascore mol_weight --temperature 0.8
```


## Using own dataset

Use the `preprocess_dataset.py` file to tokenize the dataset. The dataset should be either in the parquet or csv format.
The SMILES used for training should be in the `smiles` column in the dataset. All conditions, should be given to the pretokenize function.
After the preprocessing is done a file should be stored in the .cache directory with the name `processed_dataset_{limit}.pkl`. 
You could also rename this file to not overwrite it every time you run the preprocessing. 

The `.cache/processed_dataset_{limit}.pkl` can then be set in the `config/train/llama2-M-Full-RSS-Canonical.yaml file` to change the training with the new dataset in the `processed_dataset_ckpt` field in the yaml file. 

# Training methods

The training method we used and described in the paper is here called RSS for "Random Smiles Sampling" which was the method then described in the "Stochastic Context Learning" as taking a random subsequence from the current SMILES while training and feeding that into the model as a token sequence condition. So the model we used in the paper was the `out/llama2-M-Full-RSS-Canonical.pt`.

We also tried other approached for including the token sequence. 
One was using murcko scaffolds as they were used in the MolGPT paper, but this approach did not yield great results for our purposes. 
The other was using BRICKS decomposition, which also did not yield very good results.

The different methods are implemented in the `fragment_creator.py` file.
Each of the models were trained with their respective configurations in the `config/train` folder.

# Thanks


- [Karpathy](https://github.com/karpathy/llama2.c) for the implementation of the Llama 2 architecture and training code 

- [DeepChem](https://github.com/deepchem/deepchem) for the SmilesTokenizer

- [TorchDrug](https://github.com/DeepGraphLearning/torchdrug/) for the downloads scripts for the OPV and CEP datasets

- Zinc 15 dataset (Teague Sterling and John J. Irwin. ZINC 15 – ligand discovery for everyone. Journal of Chemical Information
and Modeling, 55(11):2324–2337, November 2015.)

- QM9 dataset (
  Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. Quantum chemistry
  structures and properties of 134 kilo molecules. Scientific Data, 1(1), aug 2014.)

- PC9 dataset (Marta Glavatskikh, Jules Leguy, Gilles Hunault, Thomas Cauchy, and Benoit Da Mota. Dataset’s chemical
diversity limits the generalizability of machine learning predictions. Journal of Cheminformatics, 11(1), nov 2019)

- ZINC 250k (Rafael Gó mez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín
Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán
Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. ACS
Central Science, 4(2):268–276, jan 2018.)

- RedDB (Elif Sorkun, Qi Zhang, Abhishek Khetan, Murat Cihan Sorkun, and Süleyman Er. RedDB, a computational
database of electroactive molecules for aqueous redox flow batteries. Scientific Data, 9(1), nov 2022.)

- OPV (Peter C. St. John, Caleb Phillips, Travis W. Kemper, A. Nolan Wilson, Yanfei Guan, Michael F. Crowley, Mark R.
Nimlos, and Ross E. Larsen. Message-passing neural networks for high-throughput polymer screening. The
Journal of Chemical Physics, 150(23):234111, jun 2019.)

- PubchemQC 2020 (Maho Nakata, Tomomi Shimazaki, Masatomo Hashimoto, and Toshiyuki Maeda. PubChemQC PM6: Data sets
of 221 million molecules with optimized molecular geometries and electronic properties. Journal of Chemical
Information and Modeling, 60(12):5891–5899, oct 2020.)

- PubchemQC 2017 (Maho Nakata and Tomomi Shimazaki. PubChemQC project: A large-scale first-principles electronic structure
database for data-driven chemistry. Journal of Chemical Information and Modeling, 57(6):1300–1308, may 2017.)

- CEP (Johannes Hachmann, Roberto Olivares-Amaya, Sule Atahan-Evrenk, Carlos Amador-Bedolla, Roel S. Sánchez-
Carrera, Aryeh Gold-Parker, Leslie Vogt, Anna M. Brockway, and Alán Aspuru-Guzik. The Harvard clean energy
project: Large-scale computational screening and design of organic photovoltaics on the world community grid.
The Journal of Physical Chemistry Letters, 2(17):2241–2251, aug 2011.) subset ( David Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gómez-Bombarelli, Timothy Hirzel,
Alán Aspuru-Guzik, and Ryan P. Adams. Convolutional networks on graphs for learning molecular fingerprints,
2015.)
- ChEMBL (James Blackshaw, Anna Gaulton, A. Patrícia Bento, Marleen De Veij, David Mendez Lopez, Nicolas Bosc, Juan
Felipe Mosquera Morales, María Paula Margariños, Andrew Leach, Emma Manners, Barbara Zdrazil, Harris
Ioannidis, Fiona Hunter, Eloy Félix, and Ricardo Arcila Toro. CHEMBL database release 31, September 2009.)

# Funding disclaimer
	            
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under Grant Agreement no. 875489. 

This website reflects only the author’s view. The funding agency is not responsible for any use made of the information it contains.

# License
 <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">LLamol is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>