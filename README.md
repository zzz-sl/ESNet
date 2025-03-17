# ESNet: A Dual-Modal Joint Framework of Elemental Composition and Crystal Structure for Material Properties Prediction


![cover](assets/architecture2.png)

## Overview
We propose a dual-modal joint framework - ESNet. Specifically, ESNet builds on existing models based on crystal structure graph, to provide an in-depth analysis of how material elemental composition and crystal structure work together to influence material properties. To obtain the elemental composition information from the compound, we pose the elemental knowledge graph embedding technique to obtain the useful knowledge. To capture key information and potential connections between the two modes of elemental composition and crystal structure, we use a content-directed attention mechanism to dynamically focus on important regions in both features.It is worth noting that, despite its simplicity, ESNet outperforms existing methods in various material property prediction tasks on the Materials Project and Jarvis datasets.

## System Requirements

### Hardware Requirements

### Software Requirements

## Dataset

### The Materials Project Dataset

For tasks in The Materials Project, we follow Matformer (Yan et al.) and use the same training, validation, and test sets.
For bulk and shear datasets, the datasets are avaliable at https://figshare.com/projects/Bulk_and_shear_datasets/165430

### JARVIS dataset

JARVIS is a newly released database proposed by Choudhary et al.. For JARVIS dataset, we follow ALIGNN and use the same training, validation, and test set. We evaluate our ComFormer on five important crystal property tasks, including formation energy, bandgap(OPT), bandgap(MBJ), total energy, and Ehull. The training, validation, and test set contains 44578, 5572, and 5572 crystals for tasks of formation energy, total energy, and bandgap(OPT). The numbers are 44296, 5537, 5537 for Ehull, and 14537, 1817, 1817 for bandgap(MBJ). The used metric is test MAE. 


## Benchmarked results

### The Materials Project Dataset
![cover](assets/MP.png)
### JARVIS dataset
![cover](assets/Jarvis.png)


## Enviroment

```bash
conda create -n esnet python=3.10
conda activate esnet
pip install torch==2.1.0
pip install torch_geometric
pip install pandarallel
pip install pydantic_settings
pip install torch_scatter
pip install e3nn
pip install torch_sparse
pip install numpy==1.26.4
pip install jarvis-tools==2022.9.16
pip install einops
python setup.py
```

## Demo

1. Preparing the dataset
```bash
cd ESNet/data
unzip jdft_3d_ehull.zip
```

2. Download esnet
Go to the root of your ESNet project and run setup.py
```bash
cd ESNet
python setup.py install
```

3. Execute the training script
```bash
cd esnet/scripts
python train_jarvis.py # for jarvis
python train_mp.py # for the materials project
```

## Acknowledgement

This work was supported by the "Kechuang Yongjiang 2035" key technology breakthrough plan of Zhejiang Ningbo((grant nos. 2024Z119).

## Contact

If you have any question, please contact me at chuang@ict.ac.cn.
