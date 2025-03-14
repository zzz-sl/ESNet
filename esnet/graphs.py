"""Module to generate networkx graphs."""
"""Implementation based on the template of Matformer."""
# from multiprocessing.context import ForkContext
from re import X
import numpy as np
import pandas as pd
import json
from collections import Counter
from jarvis.core.specie import chem_data, get_node_attributes, Specie

# from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
import itertools
from .load_triples import Triples
import pickle
from pymatgen.core import Structure

try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass


def angle_from_array(a, b, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    assert a_new.shape == a.shape
    value = sum(a_new * b_new)
    length = (sum(a_new ** 2) ** 0.5) * (sum(b_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0

def correct_coord_sys(a, b, c, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    assert a_new.shape == a.shape
    plane_vec = np.cross(a_new, b_new)
    value = sum(plane_vec * c_new)
    length = (sum(plane_vec ** 2) ** 0.5) * (sum(c_new ** 2) ** 0.5)
    cos = value / length
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos)
    return (angle / np.pi * 180.0 <= 90.0)

def same_line(a, b):
    a_new = a / (sum(a ** 2) ** 0.5)
    b_new = b / (sum(b ** 2) ** 0.5)
    flag = False
    if abs(sum(a_new * b_new) - 1.0) < 1e-5:
        flag = True
    elif abs(sum(a_new * b_new) + 1.0) < 1e-5:
        flag = True
    else:
        flag = False
    return flag

def same_plane(a, b, c):
    flag = False
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        flag = True
    return flag


# pyg dataset
class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[Data],
        target: str,
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        classification=False,
        id_tag="jid",
        neighbor_strategy="",
        nolinegraph=False,
        mean_train=None,
        std_train=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.df = df
        self.graphs = graphs
        self.target = target
        self.line_graph = line_graph

        self.ids = self.df[id_tag]
        # self.atoms = self.df['atoms']
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )
        print("mean %f std %f"%(self.labels.mean(), self.labels.std()))
        if mean_train == None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train
            print("normalize using training mean %f and std %f" % (mean_train, std_train))

        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f

        self.prepare_batch = prepare_pyg_batch
        if line_graph:
            self.prepare_batch = prepare_pyg_line_graph_batch
        

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        # template = get_node_attributes("C", atom_features)
        template_key = str(Specie("C").element_property("Z"))
        with open("/home/nbuser/SL/ESNet/graphs/atom_init.json", "r") as f:
            atom_init = json.load(f)
        template = atom_init[template_key]

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            # x = get_node_attributes(element, atom_features)
            key = str(Specie(element).element_property("Z"))
            x = atom_init[key]

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.line_graph:
            return g, g, g, label

        return g, label

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[Data, Data, torch.Tensor, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattice, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, batched_line_graph, torch.tensor(labels)


# Introduce the knowledge graph

class PygKnowledgeAndStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
            self,
            df: pd.DataFrame,
            graphs: Sequence[Data],
            kges: dict,
            target: str,
            atom_features="atomic_number",
            transform=None,
            line_graph=False,
            classification=False,
            id_tag="jid",
            neighbor_strategy="",
            nolinegraph=False,
            mean_train=None,
            std_train=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.df = df
        self.graphs = graphs
        self.target = target
        self.line_graph = line_graph

        self.ids = self.df[id_tag]
        # self.atoms = self.df['atoms']
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )
        print("mean %f std %f" % (self.labels.mean(), self.labels.std()))
        if mean_train == None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train
            print("normalize using training mean %f and std %f" % (mean_train, std_train))

        self.transform = transform

        self.kge_features = load_pyq_knowledge(df, kges)

        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            # print("g: ", g)
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f


        self.prepare_batch = prepare_pyg_batch
        if line_graph:
            self.prepare_batch = prepare_pyg_line_graph_batch

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        # template = get_node_attributes("C", atom_features)
        template_key = str(Specie("C").element_property("Z"))
        with open("/home/nbuser/SL/ESNet/graphs/atom_init.json", "r") as f:
            atom_init = json.load(f)
        template = atom_init[template_key]

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            # x = get_node_attributes(element, atom_features)
            key = str(Specie(element).element_property("Z"))
            x = atom_init[key]

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        # kge = torch.tensor(self.kge_features[idx]).type(torch.get_default_dtype())
        kge = self.kge_features[idx]
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.line_graph:
            return g, g, g, kge, label

        return g, label

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
            samples: List[Tuple[Data, Data, torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattice, kge, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        batched_kge = Batch.from_data_list(kge)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, batched_line_graph, batched_kge, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, batched_line_graph, batched_kge, torch.tensor(labels)


def load_pyq_knowledge(df: pd.DataFrame, rotatE_emb: dict) -> Sequence[Data]:

    data = Triples()
    knowledge_features = []

    id2entity = rotatE_emb["id2entity"]
    entity = rotatE_emb["entity"]

    for mat in df["atoms"].values:
        #structure = Structure.from_str(s, fmt="cif")
        #elements = [site.specie.symbol for site in structure.sites]
        elements = mat["elements"]
        element_counts = Counter(elements)
        total_elements = len(elements)
        element_ratios = {element: count / total_elements for element, count in element_counts.items()}
        #element_ratios = {element: count for element, count in element_counts.items()}
        template = entity[0]
        features = np.zeros((len(element_counts), len(np.array(template))))
        z = 0
        attributes = []
        for element, ratios in element_ratios.items():
            if element in data.entities:
                attribute_id = [h for (r, h) in data.t2rh[data.entity2id[element]]]
                attributes.extend(attribute_id)

            key = data.entity2id[element]
            x = entity[int(key)]
            if x is not None:
                features[z, :] = x * ratios
            z += 1

        attributes = sorted(list(set(attributes)))
        attr_features = np.zeros((len(attributes), len(np.array(template))))
        t = 0
        for key in attributes:
            attr_features[t, :] = entity[int(key)]
            t += 1

        f = torch.tensor(features).type(torch.get_default_dtype())
        attr_f = torch.tensor(attr_features).type(torch.get_default_dtype())
        node_emb = torch.cat((f, attr_f), 0)
        knowledge_features.append(Data(x=node_emb))

    return knowledge_features


# def load_pyq_knowledge(df: pd.DataFrame, ele2emb: dict) -> Sequence[Data]:
#     # kgembedding_path = r"D:\Projects\CodeFold\ComFormer\data\graphs\ele2emb.pkl"
#     # print("Loading KGE from ", kgembedding_path)
#     # ele2emb = pickle.load(open(kgembedding_path, 'rb'))
#     knowledge_features = []
#     for mat in df["atoms"].values:
#         elements = mat["elements"]
#         element_counts = Counter(elements)
#         total_elements = len(elements)
#         element_ratios = {element: count / total_elements for element, count in element_counts.items()}
#         template = ele2emb[0]
#         # features = np.zeros((len(element_counts), len(np.array(template))))
#         features = np.zeros((len(elements), len(np.array(template))))
#         z = 0
#         # for element, ratios in element_ratios.items():
#         for element in elements:
#             key = str(Specie(element).element_property("Z"))
#             x = ele2emb[int(key)]
#             if x is not None:
#                 # features[z, :] = x * ratios
#                 features[z, :] = x
#             z += 1
#         f = torch.tensor(features).type(torch.get_default_dtype())
#         knowledge_features.append(Data(x=f))

#     return knowledge_features


# def load_pyq_knowledge(df: pd.DataFrame, rotatE_emb: dict) -> Sequence[Data]:

#     data = Triples()
#     knowledge_features = []

#     id2entity = rotatE_emb["id2entity"]
#     # id2relation = rotatE_emb["id2relation"]
#     entity = rotatE_emb["entity"]
#     # relation = rotatE_emb["relation"]

#     for mat in df["atoms"].values:
#         elements = mat["elements"]
#         element_counts = Counter(elements)
#         total_elements = len(elements)
#         # element_ratios = {element: count / total_elements for element, count in element_counts.items()}
#         template = entity[0]

#         # features = np.zeros((len(elements), len(np.array(template))))
#         # z = 0

#         all_node_emb_list = []
#         for element in elements:
#             for item in id2entity.items():
#                 atom_id = []
#                 if item[1] == element:
#                     atom_id.append(item[0])
#                     break

#             atom_emb = entity[atom_id]
#             if element in data.entities:
#                 attribute_id = [h for (r, h) in data.t2rh[data.entity2id[element]]]

#             attribute_emb = entity[attribute_id]

#             node_emb = torch.cat((atom_emb, attribute_emb), 0)
#             all_node_emb_list.append(node_emb)

#         final_all_node_emb = torch.cat(all_node_emb_list, dim=0)
#         # f = torch.tensor(features).type(torch.get_default_dtype())
#         knowledge_features.append(Data(x=final_all_node_emb))

#     # print(knowledge_features)
#     return knowledge_features


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
            use_lattice=use_lattice,
        )
    
    edges = defaultdict(set)
    # lattice correction process
    r_cut = max(lat.a, lat.b, lat.c) + 1e-2
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    neighborlist = all_neighbors[0]
    neighborlist = sorted(neighborlist, key=lambda x: x[2])
    ids = np.array([nbr[1] for nbr in neighborlist])
    images = np.array([nbr[3] for nbr in neighborlist])
    images = images[ids == 0]
    lat1 = images[0]
    # finding lat2
    start = 1
    for i in range(start, len(images)):
        lat2 = images[i]
        if not same_line(lat1, lat2):
            start = i
            break
    # finding lat3
    for i in range(start, len(images)):
        lat3 = images[i]
        if not same_plane(lat1, lat2, lat3):
            break
    # find the invariant corner
    if angle_from_array(lat1,lat2,lat.matrix) > 90.0:
        lat2 = - lat2
    if angle_from_array(lat1,lat3,lat.matrix) > 90.0:
        lat3 = - lat3
    # find the invariant coord system
    if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
        lat1 = - lat1
        lat2 = - lat2
        lat3 = - lat3
        
    # if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
    #     print(lat1, lat2, lat3)
    # lattice correction end
    for site_idx, neighborlist in enumerate(all_neighbors_now):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(lat1))
            edges[(site_idx, site_idx)].add(tuple(lat2))
            edges[(site_idx, site_idx)].add(tuple(lat3))
            
    return edges, lat1, lat2, lat3


def compute_bond_cosine(v1, v2):
    """Compute bond angle cosines from bond displacement vectors."""
    v1 = torch.tensor(v1).type(torch.get_default_dtype())
    v2 = torch.tensor(v2).type(torch.get_default_dtype())
    bond_cosine = torch.sum(v1 * v2) / (
        torch.norm(v1) * torch.norm(v2)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


def build_undirected_edgedata(
    atoms=None,
    edges={},
    a=None,
    b=None,
    c=None,
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r, l, nei, angle, atom_lat = [], [], [], [], [], [], []
    v1, v2, v3 = atoms.lattice.cart_coords(a), atoms.lattice.cart_coords(b), atoms.lattice.cart_coords(c)
    # atom_lat.append([v1, v2, v3, -v1, -v2, -v3])
    atom_lat.append([v1, v2, v3])
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                # nei.append([v1, v2, v3, -v1, -v2, -v3])
                nei.append([v1, v2, v3])
                # angle.append([compute_bond_cosine(dd, v1), compute_bond_cosine(dd, v2), compute_bond_cosine(dd, v3)])

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    l = torch.tensor(l).type(torch.int)
    nei = torch.tensor(np.array(nei)).type(torch.get_default_dtype())
    atom_lat = torch.tensor(np.array(atom_lat)).type(torch.get_default_dtype())
    # nei_angles = torch.tensor(angle).type(torch.get_default_dtype())
    return u, v, r, l, nei, atom_lat


class PygGraph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=4.0, 
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = False,
        use_angle: bool = False,
    ):
        if neighbor_strategy == "k-nearest":
            edges, a, b, c = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r, l, nei, atom_lat = build_undirected_edgedata(atoms, edges, a, b, c)
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        atom_lat = atom_lat.repeat(node_features.shape[0],1,1)
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
        g = Data(x=node_features, edge_index=edge_index, edge_attr=r, edge_type=l, edge_nei=nei, atom_lat=atom_lat)
        
        return g


def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


def prepare_pyg_line_graph_batch(
    batch: Tuple[Tuple[Data, Data, Data, torch.Tensor], torch.Tensor],
    device=None,
    non_blocking=False,
    ):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, lattice, kg, t = batch
    batch = (
        (
            g.to(device),
            lg.to(device),
            kg.to(device),
            lattice.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch

