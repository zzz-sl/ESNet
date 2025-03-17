import sys
sys.path.append("/home/nbuser/SL/ESNet/esnet")
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


train_prop_model(learning_rate=0.001, name="iComformer", dataset_path="/home/nbuser/SL/ESNet/data/jdft_3d_ehull.json", scheduler="onecycle", prop=props[27], pyg_input=True, n_epochs=500, max_neighbors=25, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="/home/nbuser/SL/ESNet/results", use_angle=True, save_dataloader=False, test_only=True)
