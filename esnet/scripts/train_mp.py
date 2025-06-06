import sys
sys.path.append("/mnt/public/bleschen/code/ESNet/esnet")
from esnet.train_props import train_prop_model
props = [
    "formation_energy_per_atom",
    "gap pbe",
    "bulk modulus",
    "shear modulus",
]
train_prop_model(learning_rate=0.0001,
                 name="iComformer",
                 dataset="megnet",
                 dataset_path="/mnt/public/bleschen/code/ESNet/graphs/MP2022_formation_energy_6000.json",
                 prop=props[0],
                 pyg_input=True,
                 n_epochs=500,
                 max_neighbors=25,
                 cutoff=4.0,
                 batch_size=64,
                 use_lattice=True,
                 output_dir="/mnt/public/bleschen/code/ESNet/test",
                 use_angle=True,
                #  file_name="bulk",
                #  mp_id_list="bulk",
                 atom_features="atomic_number",
                 save_dataloader=False,
                 test_only=False)
