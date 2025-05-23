# ESNet: A Dual-Modal Joint Framework of Elemental Composition and Crystal Structure for Material Properties Prediction


![cover](assets/architecture2.png)

## Overview
We propose a dual-modal joint framework - ESNet. Specifically, ESNet builds on existing models based on crystal structure graph, to provide an in-depth analysis of how material elemental composition and crystal structure work together to influence material properties. To obtain the elemental composition information from the compound, we pose the elemental knowledge graph embedding technique to obtain the useful knowledge. To capture key information and potential connections between the two modes of elemental composition and crystal structure, we use a content-directed attention mechanism to dynamically focus on important regions in both features.It is worth noting that, despite its simplicity, ESNet outperforms existing methods in various material property prediction tasks on the Materials Project and Jarvis datasets.

## System Requirements
### Hardware Requirements
GPU ：Tesla V100S-PCIE-32GB, 1

### Software Requirements
Development version is tested on Linux operating systems. 

Linux：Ubuntu 22.04

CUDA version: 11.8 or 12.4

## Dataset

### The Materials Project Dataset

For tasks in The Materials Project, we follow Matformer (Yan et al.) and use the same training, validation, and test sets.
For bulk and shear datasets, the datasets are avaliable at https://figshare.com/projects/Bulk_and_shear_datasets/165430

### JARVIS dataset

JARVIS is a newly released database proposed by Choudhary et al.. For JARVIS dataset, we follow ALIGNN and use the same training, validation, and test set. We evaluate our ComFormer on five important crystal property tasks, including formation energy, bandgap(OPT), bandgap(MBJ), total energy, and Ehull. The training, validation, and test set contains 44578, 5572, and 5572 crystals for tasks of formation energy, total energy, and bandgap(OPT). The numbers are 44296, 5537, 5537 for Ehull, and 14537, 1817, 1817 for bandgap(MBJ). The used metric is test MAE. 


## Enviroment

```bash
conda create -n esnet python=3.10
conda activate esnet
pip install torch==2.1.0
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install pandarallel
pip install pydantic_settings
pip install e3nn
pip install numpy==1.26.4
pip install jarvis-tools==2022.9.16
pip install einops
pip install pymatgen
pip install pytorch-ignite
pip install scikit-learn
```

## Demo

1. Preparing the dataset

Megnet dataset download: https://ndownloader.figshare.com/files/26724977

Bulk and Shear datasets download: https://figshare.com/projects/Bulk_and_shear_datasets/165430

2. Download esnet
Go to the root of your ESNet project and run setup.py
```bash
cd ESNet
python setup.py install
```

3. Modifying the path
Change the project path to your local execution path as follows:

（1）Modify data.py
```bash
cd esnet/
vim data.py
kgembedding_path = "/yourpath/ESNet/graphs/RotatE_128_64.pkl"
```

For the bulk modulus and shear modulus data sets, modify the corresponding paths, for example:
```bash
 with open('/yourpath/ESNet/data/bulk_megnet_train.pkl', 'rb') as f # line 457-470
```


（2）Modify graphs.py
```bash
cd esnet/
vim graphs.py
with open("/yourpath/ESNet/graphs/atom_init.json", "r") as f  # line 146 and line 276 
```

（3）Modify load_triples.py
```bash
cd esnet/
vim load_triples.py
data_dir="/yourpath/ESNet/graphs/triples.txt" 
```

(4) If you want to reproduce based on the weights we provided(currently only bulk modulus weights are provided), you'll need to modify train.py:
```bash
cd esnet/
vim train.py
checkpoint_tmp = torch.load('/yourpath/ESNet/checkpoints/checkpoint_500.pt') # bulk modulus checkpoint
```

4. Execute the training script
Make sure to change the path to your own local execution path.
```bash
cd esnet/scripts
vim train_mp.py
```
The changes are as follows:
```bash
dataset_path="/yourpath/ESNet/data/megnet.json"
output_dir="/yourpath/ESNet/results"
```
Execute the script:
```bash
python train_mp.py
```

## Results

### The Materials Project Dataset
![cover](assets/MP.png)
### JARVIS dataset
![cover](assets/Jarvis.png)


## Acknowledgement

This work was supported by the "Kechuang Yongjiang 2035" key technology breakthrough plan of Zhejiang Ningbo((grant nos. 2024Z119).

## Contact

If you have any question, please contact me at chuang@ict.ac.cn.
