import sys
sys.path.append("/yourpath/esnet")
from esnet.train_props import train_prop_model
props = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]

# train_prop_model(learning_rate=0.001, name="iComformer", criterion="mse", prop=props[0], pyg_input=True, n_epochs=700, max_neighbors=25, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="/root/autodl-tmp/ComFormer/data", use_angle=True, save_dataloader=True)

train_prop_model(learning_rate=0.001, name="iComformer", dataset_path="/root/autodl-tmp/graphs/jdft_3d-8-18-2021.json", prop=props[1], pyg_input=True, n_epochs=700, max_neighbors=25, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="/root/autodl-tmp/results/jdft_eform", use_angle=True, save_dataloader=False, test_only=False)
