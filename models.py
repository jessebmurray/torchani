from collections import OrderedDict
from typing import List, Optional

import torch
from torch import nn


class AtomicNN(nn.Module):
    """
    Atomic Neural Network (ANN)

    Parameters
    ----------
    n_inputs: int
        Input size (AEVs length)
    layers_sizes: List[int]
        List with the size of fully connected layers, excluding firs
    dropp: Optional[float]
        Dropout probability
    """

    def __init__(
        self,
        n_inputs: int,
        layers_sizes: Optional[List[int]] = None,
        dropp: Optional[float] = None,
    ):

        super().__init__()

        if layers_sizes is None:
            # Default values from TorchANI turorial
            self.layers_sizes: List[int] = [160, 128, 96, 1]
        else:
            self.layers_sizes = layers_sizes.copy()

        # Prepend input size to other layer sizes
        self.layers_sizes.insert(0, n_inputs)

        self.layers = nn.ModuleList()

        for in_size, out_size in zip(self.layers_sizes[:-2], self.layers_sizes[1:-1]):
            self.layers.append(nn.Linear(in_size, out_size))
            self.layers.append(nn.ReLU())

            if dropp is not None:
                self.layers.append(nn.Dropout(dropp))

        # Last linear layer
        self.layers.append(nn.Linear(self.layers_sizes[-2], self.layers_sizes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class AffinityModel(nn.ModuleDict):
    """
    Affinity prediction from AEVs.

    Parameters
    ----------
    n_species: int
        Number of species
    aev_length: int
        Length of the atomic environment vectors
    layers_sizes: Optional[List[int]] = None
        Layers' dimensions for each atomic NN
    dropp: Optional[float]
        Dropout probability

    """

    def __init__(
        self,
        n_species: int,
        aev_length: int,
        layers_sizes: Optional[List[int]] = None,
        dropp: Optional[float] = None,
    ):

        assert n_species > 0

        modules = n_species * [
            AtomicNN(aev_length, layers_sizes=layers_sizes, dropp=dropp)
        ]

        super().__init__(self.ensureOrderedDict(modules))

        # Store values
        self.aev_length = aev_length
        self.n_species = n_species
        self.dropp = dropp
        self.layers_sizes = modules[0].layers_sizes

    def _forward_atomic(self, species, aevs, ligmasks=None):
        """
        Forward pass for individual atomic environments.

        Parameters
        ----------
        species: torch.Tensor
            Species
        aevs: torch.Tensor
            Atomic environment vectors
        ligmasks: torch.Tensor
            Masks for ligand atoms

        Returns
        -------
        torch.Tensor
            Atomic contributions (unmasked)

        Notes
        -----

        This is extracted from the original code and computes
        forward pass without aggregation.

        Atomic contributions are not masked by ligand atoms. However, when a ligand
        mask is used non-ligand contributions are set to zero and therefore they do
        not contribute to the final sum.
        """
        if ligmasks is not None:
            species_ = species.clone()
            species_[~ligmasks] = -1
        else:
            species_ = species

        species_ = species_.flatten()
        aevs = aevs.flatten(0, 1)

        # size of species_ but same dtype and device of aevs
        output = aevs.new_zeros(species_.shape)

        for i, (_, m) in enumerate(self.items()):
            mask = species_ == i
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aevs.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)

        return output

    def forward(self, species, aevs, ligmasks=None):
        """
        Parameters
        ----------
        species: torch.Tensor
            Species
        aevs: torch.Tensor
            Atomic environment vectors
        ligmasks: torch.Tensor
            Masks for ligand atoms

        Returns
        -------
        torch.Tensor
            Model output (affinity predictions)
        """
        output = self._forward_atomic(species, aevs, ligmasks)
        return torch.sum(output, dim=1)

    @staticmethod
    def ensureOrderedDict(modules):
        """
        Ensure ordered dictionary (for old-ish Python versions)
        """
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od
