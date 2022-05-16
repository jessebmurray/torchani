import torch
import math
import numpy as np
import qcelemental as qcel
from os import listdir
from torchani.nn import SpeciesConverter
from torch.utils.data import DataLoader
import torchani
from typing import Collection, Dict, List, Optional, Tuple, Union


dataset_elements = {'NCIA_HB375x10': ['H', 'C', 'N', 'O']}

to_Z = np.vectorize(qcel.periodictable.to_Z)

def header_to_dict(header):
    fields = header.split()

    d = {}
    for field in fields:
        key, value = field.split("=")

        try:
            value = float(value)
        except ValueError:

            try:
                start, stop = value.split("-")

                # Create slice
                # Shift to 0-based indexing
                # Stop is not included
                value = slice(int(start) - 1, int(stop))

            except ValueError:
                pass

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

    return anums, coordinates, header_to_dict(header)

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
    index_diff = np.array([info['selection_a'].stop], dtype='int')

    entry = {'species': species, 'coordinates': coordinates,
            'index_diff': index_diff, 'energies': energies}
    return entry

def get_reverse_entry(entry):
    """Reverse an entry."""
    species = entry['species']
    coordinates = entry['coordinates']
    index_diff = entry['index_diff']
    energies = entry['energies']

    r_species = species.flip(0)
    r_coordinates = coordinates.flip(0)
    r_index_diff = len(species) - index_diff

    r_entry = {'species': r_species, 'coordinates': r_coordinates,
        'index_diff': r_index_diff, 'energies': energies}
    return r_entry

def get_fnames(datapath):
    return sorted(listdir(datapath))

def load_data(dataset):
    elements = dataset_elements[dataset]
    species_converter = SpeciesConverter(elements)
    datapath = f"./{dataset}/geometries/"
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

    return np.array(ids), torch.tensor(labels), (pad_species, pad_coordinates), torch.tensor(index_diff)


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
            self.ids.append(entry['id'])
            self.species.append(entry['species'])
            self.coordinates.append(entry['coordinates'])
            self.labels.append(entry['energies'])
            self.index_diff.append(entry['index_diff'])


def get_data_loaders(training, validation, batch_size=40):
    traindata = Data()
    validdata = Data()

    traindata.load(training)
    validdata.load(validation)

    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    validloader = DataLoader(validdata, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    return trainloader, validloader
