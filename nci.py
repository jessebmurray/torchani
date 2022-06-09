import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qcelemental as qcel
from os import listdir
from torchani.nn import SpeciesConverter
from torch.utils.data import DataLoader
from typing import Collection, Dict, List, Optional, Tuple, Union
from collections import Counter


dataset_elements = {'NCIA_HB375x10': ['H', 'C', 'N', 'O'],
    'NCIA_IHB100x10': ['H', 'C', 'N', 'O'],
    'NCIA_HB300SPXx10': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
    'NCIA_R739x5': ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                    'P', 'S', 'Cl', 'Ar', 'Br', 'Kr', 'I', 'Xe'],
    'NCIA_SH250x10': ['H', 'C', 'N', 'O', 'F',
                    'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I'],
    'NCIA_D442x10': ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                    'P', 'S', 'Cl', 'Ar', 'Se', 'Br', 'Kr', 'I', 'Xe'],
    'NCIA_D1200': ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                    'P', 'S', 'Cl', 'Ar', 'Se', 'Br', 'Kr', 'I', 'Xe']}

ncia_elements = set(sum(list(dataset_elements.values()), []))

unique_elements = []
for l in dataset_elements.values():
    unique_elements += l

unique_elements = list(set(unique_elements))

to_Z = np.vectorize(qcel.periodictable.to_Z)

def header_to_dict(header):
    fields = header.split()

    d = {}
    for field in fields:
        key, value = field.split("=")
        if key == 'selection_a':
            try:
                value = int(value)
            except ValueError:

                # try:
                _, stop = value.split("-")

                # Create slice
                # Shift to 0-based indexing
                # Stop is not included
                value = stop

                # except ValueError:
                #     pass
        elif key == 'benchmark_Eint':
            value = float(value)

        d[key] = value

    return d

def read_xyz(fname):
    with open(fname, "r") as f:
        _ = f.readline()
        header = f.readline()

    elements = np.loadtxt(fname, dtype="U3", skiprows=2, usecols=(0,))
    coordinates = np.loadtxt(fname, skiprows=2, usecols=(1, 2, 3))

    # Convert elements to atomic numbers
    anums = to_Z(elements)

    info = header_to_dict(header)
    info['element_counts'] = dict(Counter(elements))

    return anums, coordinates, info

def get_constants(radial_cutoff: float, angular_cutoff: float,
                    radial_eta: float, angular_eta: float,
                    radial_dist_divisions: int, angular_dist_divisions: int,
                    zeta: float, angle_sections: int, num_species: int,
                    angular_start: float = 0.9, radial_start: float = 0.9):
    r""" Provides a convenient way to linearly fill cutoffs
    This is a user friendly constructor that builds an
    :class:`torchani.AEVComputer` where the subdivisions along the the
    distance dimension for the angular and radial sub-AEVs, and the angle
    sections for the angular sub-AEV, are linearly covered with shifts. By
    default the distance shifts start at 0.9 Angstroms.
    To reproduce the ANI-1x AEV's the signature ``(5.2, 3.5, 16.0, 8.0, 16, 4, 32.0, 8, 4)``
    can be used.
    """
    # This is intended to be self documenting code that explains the way
    # the AEV parameters for ANI1x were chosen. This is not necessarily the
    # best or most optimal way but it is a relatively sensible default.
    Rcr = radial_cutoff
    Rca = angular_cutoff
    EtaR = torch.tensor([float(radial_eta)])
    EtaA = torch.tensor([float(angular_eta)])
    Zeta = torch.tensor([float(zeta)])

    ShfR = torch.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]
    ShfA = torch.linspace(angular_start, angular_cutoff, angular_dist_divisions + 1)[:-1]
    angle_start = math.pi / (2 * angle_sections)

    ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]

    return {'Rcr': Rcr, 'Rca': Rca, 'EtaR': EtaR, 'ShfR': ShfR,
            'EtaA': EtaA, 'Zeta': Zeta, 'ShfA': ShfA, 'ShfZ': ShfZ, 'num_species': num_species}


def get_entry(fpath, species_converter):
    anums, coords, info = read_xyz(fpath)
    anums = torch.from_numpy(anums) #.unsqueeze(0)
    coords = torch.from_numpy(coords).float() #.unsqueeze(0).float()
    species, coordinates = species_converter((anums, coords))
    energies = np.array([info['benchmark_Eint']])
    index_diff = torch.from_numpy(np.array([info['selection_a']], dtype='int'))
    scaling = float(info['scaling'])

    entry = {'species': species, 'coordinates': coordinates,
            'index_diff': index_diff, 'energies': energies,
            'scaling': scaling, 'reverse': False}
    for element in ncia_elements:
        entry[element] = info['element_counts'].get(element, 0)
    return entry

def get_reverse_entry(entry):
    """Reverse an entry."""
    species = entry['species']
    coordinates = entry['coordinates']
    index_diff = entry['index_diff']
    energies = entry['energies']
    scaling = entry['scaling']

    r_species = species.flip(0)
    r_coordinates = coordinates.flip(0)
    r_index_diff = len(species) - index_diff

    r_entry = {'species': r_species, 'coordinates': r_coordinates,
        'index_diff': r_index_diff, 'energies': energies, 'reverse': True,
        'scaling': scaling}
    for element in ncia_elements:
        r_entry[element] = entry[element]
    return r_entry

def get_fnames(datapath):
    return sorted(listdir(datapath))

def load_data(dataset):
    elements = dataset_elements[dataset]
    species_converter = SpeciesConverter(elements)

    datapath = f"../NCIAtlas/geometries/{dataset}/"
    data_fnames = get_fnames(datapath)
    data = []
    for i, fname in enumerate(data_fnames):

        fpath = datapath + fname

        entry = get_entry(fpath, species_converter)
        entry.update({'id': i})
        data.append(entry)

        reverse_entry = get_reverse_entry(entry)
        reverse_entry.update({'id': i})
        data.append(reverse_entry)

    return data

def pad_collate(
    batch,
    species_pad_value=-1,
    coords_pad_value=0,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """
    Collate function to pad batches.
    Parameters
    ----------
    batch:
        Batch
    species_pad_value:
        Padding value for species vector
    coords_pad_value:
        Padding value for coordinates
    device: Optional[Union[str, torch.device]]
        Computation device
    Returns
    -------
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    Notes
    -----
    :code:`torch.utils.data.Dataset`'s :code:`__getitem__` returns a batch that need
    to be padded. This function can be passed to :code:`torch.utils.data.DataLoader`
    as :code:`collate_fn` argument in order to pad species and coordinates
    """

    ids, labels, species_and_coordinates, index_diff = zip(*batch)

    species, coordinates = zip(*species_and_coordinates)

    pad_species = torch.nn.utils.rnn.pad_sequence(
        species, batch_first=True, padding_value=species_pad_value
    )
    pad_coordinates = torch.nn.utils.rnn.pad_sequence(
        coordinates, batch_first=True, padding_value=coords_pad_value
    )
    labels = torch.tensor(np.array(labels)).reshape(1, -1).squeeze(0)
    index_diff = torch.tensor(index_diff).unsqueeze(1)

    return np.array(ids), labels, (pad_species, pad_coordinates), index_diff


class Data(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        # TODO: Better way to avoid mypy complaints?
        self.n: int = -1
        self.ids: List[str] = []
        self.labels: List[float] = []
        self.species: List[torch.Tensor] = []
        self.coordinates: List[torch.Tensor] = []
        self.species_are_indices: bool = False
        self.index_diff: List[int] = []

    def __len__(self) -> int:
        """
        Number of protein-ligand complexes in the dataset.
        Returns
        -------
        int
            Dataset length
        """
        return self.n

    def __getitem__(
        self, idx: int
    ):  # -> Tuple[str, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get item from dataset.
        Parameters
        ----------
        idx: int
            Item index within the dataset
        Returns
        -------
        Tuple[str, float, Tuple[torch.Tensor, torch.Tensor]]
            Item from the dataset (PDB IDs, labels, species, coordinates)
        """
        return (
            self.ids[idx],
            self.labels[idx],
            (self.species[idx], self.coordinates[idx]),
            self.index_diff[idx])

    def load(self, data):

        self.n = len(data)
        for entry in data:
            self.ids.append((entry['dataset'], entry['id']))
            self.species.append(entry['species'])
            self.coordinates.append(entry['coordinates'])
            self.labels.append(entry['energies'])
            self.index_diff.append(entry['index_diff'])


def get_data_loader(dataset, batch_size=40, shuffle=True):
    out = Data()
    out.load(dataset)
    return DataLoader(out, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)


def get_data_loaders(training, validation, batch_size=40):
    trainloader = get_data_loader(training, batch_size=batch_size)
    validloader = get_data_loader(validation, batch_size=batch_size)
    return trainloader, validloader


def get_predictions(loader, model, AEVC, device=None, return_info=False):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model.to(device)

    # Model in evaluation mode
    model.eval()

    true = []
    predictions = []
    identifiers = []

    if return_info:
        species_all = []
        coordinates_all = []
        index_diff_all = []
        aevs_all = []
        species_aevs_all = []

    for ids, labels, species_coordinates_ligmasks, index_diff in loader:

        # Move data to device
        labels = labels.to(device).float()
        species = species_coordinates_ligmasks[0].to(device)
        coordinates = species_coordinates_ligmasks[1].to(device).float()
        index_diff = index_diff.to(device)

        if len(species_coordinates_ligmasks) == 2:
            ligmasks = None
        else:
            ligmasks = species_coordinates_ligmasks[2].to(device)

        # return species, coordinates, index_diff

        species_aevs, aevs = AEVC.forward((species, coordinates), index_diff)

        output = model(species, aevs, ligmasks)
        output = output.cpu().detach().numpy()
        labels = labels.cpu().numpy()

        # Store true and predicted values
        predictions += output.tolist()
        true += labels.tolist()
        identifiers += ids.tolist()

        if return_info:
            species_all.append(species)
            coordinates_all.append(coordinates)
            index_diff_all.append(index_diff)
            aevs_all.append(aevs)
            species_aevs_all.append(species_aevs)

    identifiers = np.array(identifiers)
    true = np.array(true)
    predictions = np.array(predictions)

    if return_info:
        return identifiers, true, predictions, species_all, coordinates_all, index_diff_all, aevs_all, species_aevs_all

    return identifiers, true, predictions


def analyze_pairs(pair1, pair2, alpha=0.2):
    print(np.corrcoef(pair1, pair2)[0, 1])
    plt.scatter(pair1, pair2, alpha=alpha, facecolors='none', edgecolors='C0')
    plt.show()
    plt.hist(pair1 - pair2, bins=50);
    plt.show()


def get_system_names(dataset):
    with open(f"../NCIAtlas/tables/{dataset}/{dataset}_system_names.txt") as f:
        lines = f.readlines()
    lines = lines[1:]
    for i, line in enumerate(lines):
        lines[i] = line[line.find('\t')+1:].rstrip('\n').split(' ... ')
    system_names = []
    for entry in lines:
        system_names.append(entry)
        system_names.append(entry[::-1])
    return system_names

def get_metadata(dataset):
    with open(f"../NCIAtlas/tables/{dataset}/{dataset}_metadata.txt") as f:
        lines = f.readlines()
    metadata = []
    for line in lines:
        try:
            int(line[0])
        except:
            continue
        # lines[i] = line[line.find('\t')+1:].rstrip('\n').split(' ... ')
        entry = line.split('\t')
        if dataset == 'NCIA_HB375x10':
            interaction_type = 'HB'
            interaction = entry[1]
            if interaction == 'noHB':
                interaction_type = 'VDW'
                interaction = float('NaN')
        elif dataset == 'NCIA_IHB100x10':
            interaction_type = 'IHB'
            interaction = entry[1]
        elif dataset == 'NCIA_HB300SPXx10':
            interaction_type = 'HB'
            interaction = entry[2].split(',')[0].lstrip('"')
        else:
            interaction_type = 'VDW'
            interaction = float('NaN')
        metadata.append({'interaction_type': interaction_type, 'interaction': interaction})
        metadata.append({'interaction_type': interaction_type, 'interaction': interaction})
    return metadata

def load_df(dataset):
    data = load_data(dataset)
    dfd = pd.DataFrame(data)

    system_names = get_system_names(dataset)
    dfs = pd.DataFrame(system_names, columns=['MoleculeA', 'MoleculeB'])
    metadata = get_metadata(dataset)
    dfm = pd.DataFrame(metadata)
    df = dfs.join(dfm).join(dfd)
    df['dataset'] = np.repeat([dataset], df.shape[0])
    return df

def load_dfs(datasets: list):
    dfs = []
    for dataset in datasets:
        df = load_df(dataset)
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    return df

def exclude_elements(df):
    noble_gasses = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
    exclude = noble_gasses + ['As']
    df = df.loc[df[exclude].sum(axis=1) == 0]
    df = df.drop(columns=exclude)
    return df


def exclude_molecule(df, molecule):
    return df.loc[(df.MoleculeA != molecule) & (df.MoleculeB != molecule)]

def include_molecule(df, molecule):
    return df.loc[(df.MoleculeA == molecule) | (df.MoleculeB == molecule)]
