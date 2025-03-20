"""Implementation based on the template of ALIGNN."""

import imp
import random
from pathlib import Path
from typing import Optional

# from typing import Dict, List, Optional, Set, Tuple

import os
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from esnet.graphs import PygGraph, PygStructureDataset, PygKnowledgeAndStructureDataset
from jarvis.db.figshare import data as jdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson
#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=False)
# from sklearn.pipeline import Pipeline
import pickle as pk
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure
# import warnings
# from pymatgen.io.jarvis import JarvisAtomsAdaptor
# adaptor = JarvisAtomsAdaptor()
# warnings.filterwarnings("ignore")
# use pandas progress_apply
tqdm.pandas()

device = "cpu"
if torch.cuda.is_available():
   device = torch.device("cuda")

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(
    name: str = "dft_3d",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na" and not math.isnan(i[target]):
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    d = pd.DataFrame(d)
    return d


def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def load_pyg_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """
    
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)

        return PygGraph.atom_dgl_multigraph(
            structure,
            neighbor_strategy=neighbor_strategy,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )
    
    tqdm.pandas()     
    #graphs = df["atoms"].parallel_apply(atoms_to_graph).values
    graphs = df["atoms"].apply(atoms_to_graph).values

    return graphs


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_pyg_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    use_lattice=False,
    use_angle=False,
    data_from='Jarvis',
    use_save=False,
    mean_train=None,
    std_train=None,
    now=False, # for test
):
    """Get pyg Dataset."""
    df = pd.DataFrame(dataset)
    vals = df[target].values
    if target == "shear modulus" or target == "bulk modulus":
        val_list = [vals[i].item() for i in range(len(vals))]
        vals = val_list
    # output_dir = "./saved_data/" + tmp_name + "test_graph_angle.pkl" # for fast test use
    print("data range", np.max(vals), np.min(vals))
    # print(output_dir)
    # print('graphs not saved')
    graphs_path = output_dir + "/" + tmp_name + "_graph_angle.pkl"
    if not os.path.exists(graphs_path):
        graphs = load_pyg_graphs(
            df,
            name=name,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )
        with open(graphs_path, 'wb') as pf:
            pk.dump(graphs, pf)
        print('save graphs to ', graphs_path)
    else:
        print('loading graphs from ', graphs_path)
        with open(graphs_path, 'rb') as pf:
            graphs = pk.load(pf)

    # KG Embedding
    # kgembedding_path = r"/root/autodl-tmp/graphs/ele2emb.pkl"
    kgembedding_path = "/home/nbuser/SL/ESNet/graphs/RotatE_128_64.pkl"
    print('loading KGE from ', kgembedding_path)
    ele2emb = pk.load(open(kgembedding_path, 'rb'))
    if mean_train == None:
        print('loading KGE !!!')
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        data = PygKnowledgeAndStructureDataset(
            df,
            graphs,
            ele2emb,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    else:
        print('loading KGE !!!')
        data = PygKnowledgeAndStructureDataset(
            df,
            graphs,
            ele2emb,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    '''
    if mean_train == None:
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    else:
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    '''
    return data, mean_train, std_train


def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=[],
    dataset_path = ".",
    target: str = "formation_energy_peratom",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    matrix_input=False,
    pyg_input=False,
    use_lattice=False,
    use_angle=False,
    use_save=True,
    mp_id_list=None,
):
    """Help function to set up JARVIS train and val dataloaders."""
    # data loading
    mean_train=None
    std_train=None
    assert (matrix_input and pyg_input) == False
    
    # train_sample = filename + "_train.data"
    # val_sample = filename + "_val.data"
    # test_sample = filename + "_test.data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_sample = output_dir + "/" + filename + "_train.data"
    val_sample = output_dir + "/" + filename + "_val.data"
    test_sample = output_dir + "/" + filename + "_test.data"
    if (
        os.path.exists(train_sample)
        and os.path.exists(val_sample)
        and os.path.exists(test_sample)
        and save_dataloader
    ):
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)
        if train_loader.pin_memory != pin_memory:
            train_loader.pin_memory = pin_memory
        if test_loader.pin_memory != pin_memory:
            test_loader.pin_memory = pin_memory
        if val_loader.pin_memory != pin_memory:
            val_loader.pin_memory = pin_memory
        if train_loader.num_workers != workers:
            train_loader.num_workers = workers
        if test_loader.num_workers != workers:
            test_loader.num_workers = workers
        if val_loader.num_workers != workers:
            val_loader.num_workers = workers
        print("train", len(train_loader.dataset))
        print("val", len(val_loader.dataset))
        print("test", len(test_loader.dataset))
        return (
            train_loader,
            val_loader,
            test_loader,
            train_loader.dataset.prepare_batch,
            None,
            None
        )
    else:
        if not dataset_array:
            # d = jdata(dataset)
            import json
            # dataset_path = "/root/autodl-tmp/graphs/jdft_3d_ehull.json"
            print("Loading dataset: ", dataset_path)
            f = open(dataset_path, "r", encoding="utf-8")
            d = json.load(f)
            
        else:
            d = dataset_array
            # for ii, i in enumerate(pc_y):
            #    d[ii][target] = pc_y[ii].tolist()

        dat = []
        if classification_threshold is not None:
            print(
                "Using ",
                classification_threshold,
                " for classifying ",
                target,
                " data.",
            )
            print("Converting target data into 1 and 0.")
        all_targets = []

        # TODO:make an all key in qm9_dgl
        if dataset == "qm9_dgl" and target == "all":
            print("Making all qm9_dgl")
            tmp = []
            for ii in d:
                ii["all"] = [
                    ii["mu"],
                    ii["alpha"],
                    ii["homo"],
                    ii["lumo"],
                    ii["gap"],
                    ii["r2"],
                    ii["zpve"],
                    ii["U0"],
                    ii["U"],
                    ii["H"],
                    ii["G"],
                    ii["Cv"],
                ]
                tmp.append(ii)
            print("Made all qm9_dgl")
            d = tmp
        for i in d:
            if isinstance(i[target], list):  # multioutput target
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)

            elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
            ):
                if target_multiplication_factor is not None:
                    i[target] = i[target] * target_multiplication_factor
                if classification_threshold is not None:
                    if i[target] <= classification_threshold:
                        i[target] = 0
                    elif i[target] > classification_threshold:
                        i[target] = 1
                    else:
                        raise ValueError(
                            "Check classification data type.",
                            i[target],
                            type(i[target]),
                        )
                dat.append(i)
                all_targets.append(i[target])
    
    if mp_id_list is not None:
        if mp_id_list == 'bulk':
            print('using mp bulk dataset')
            with open('/yourpath/ESNet/data/bulk_megnet_train.pkl', 'rb') as f:
                dataset_train = pk.load(f)
            with open('/yourpath/ESNet/data/bulk_megnet_val.pkl', 'rb') as f:
                dataset_val = pk.load(f)
            with open('/yourpath/ESNet/data/bulk_megnet_test.pkl', 'rb') as f:
                dataset_test = pk.load(f)
        
        if mp_id_list == 'shear':
            print('using mp shear dataset')
            with open('/yourpath/ESNet/data/shear_megnet_train.pkl', 'rb') as f:
                dataset_train = pk.load(f)
            with open('/yourpath/ESNet/data/shear_megnet_val.pkl', 'rb') as f:
                dataset_val = pk.load(f)
            with open('/yourpath/ESNet/data/shear_megnet_test.pkl', 'rb') as f:
                dataset_test = pk.load(f)

    else:
        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(output_dir, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

    train_data, mean_train, std_train = get_pyg_dataset(
        dataset=dataset_train,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="train_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
    )
    
    val_data,_,_ = get_pyg_dataset(
        dataset=dataset_val,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="val_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
    )
    test_data,_,_ = get_pyg_dataset(
        dataset=dataset_test,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="test_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
    )

    
    collate_fn = train_data.collate
    if line_graph:
        collate_fn = train_data.collate_line_graph

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if save_dataloader:
        torch.save(train_loader, train_sample)
        torch.save(val_loader, val_sample)
        torch.save(test_loader, test_sample)
    
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
        mean_train,
        std_train,
    )
    
