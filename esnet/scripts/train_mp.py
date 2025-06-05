import sys
sys.path.append("D:/code/material/ESNet/esnet")
from esnet.train_props import train_prop_model
props = [
    "e_form",
    "gap pbe",
    "bulk modulus",
    "shear modulus",
]
train_prop_model(learning_rate=0.0001,
                 name="iComformer",
                 dataset="megnet",
                 dataset_path="../../graphs/megnet.json",
                 prop=props[0],
                 pyg_input=True,
                 n_epochs=500,
                 max_neighbors=25,
                 cutoff=4.0,
                 batch_size=64,
                 use_lattice=True,
                 output_dir="../../test",
                 use_angle=True,
                 file_name="bulk",
                 mp_id_list="bulk",
                 atom_features="atomic_number",
                 save_dataloader=False,
                 test_only=True)
