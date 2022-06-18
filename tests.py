import torch
import nci
import unittest
import utils
from torchani_reg.aev import AEVComputer as AEVComputer_reg

class TestAEVs(unittest.TestCase):
    def __init__(self):

        dataset = 'NCIA_HB375x10'
        data = nci.load_data(dataset)
        loader0 = nci.get_data_loader(data[:2], batch_size=2, shuffle=False)
        _, _, _, species, coordinates, index_diff, aevs, species_aevs = get_predictions(
        loader0, model, AEVC, return_info=True)

        self.species, self.coordinates, self.index_diff, self.aevs, self.species_aevs = \
                species[0], coordinates[0], index_diff[0], aevs[0], species_aevs[0]

    def test_species_aev_mask(self):
        self.assertTrue((self.species_aevs[0][:int(self.index_diff[0])] >= 0).all())
        self.assertTrue((self.species_aevs[1][:int(self.index_diff[1])] >= 0).all())
        self.assertTrue((self.species_aevs[1][int(self.index_diff[1]):] == -1).all())
        self.assertTrue((self.species_aevs[0][int(self.index_diff[0]):] == -1).all())

    def test_aev_mask(self):
        self.assertTrue((self.aevs[0].sum(axis=1)[:int(self.index_diff[0])] >= 0).all())
        self.assertTrue((self.aevs[1].sum(axis=1)[:int(self.index_diff[1])] >= 0).all())
        self.assertTrue((self.aevs[1].sum(axis=1)[int(self.index_diff[1]):] == 0).all())
        self.assertTrue((self.aevs[0].sum(axis=1)[int(self.index_diff[0]):] == 0).all())

    def test_aev_pairs(self):
        path_aevc = './out/aevc.pth'
        AEVC = utils.loadAEVC(path=path_aevc)
        AEVC_reg = AEVComputer_reg(AEVC.Rcr, AEVC.Rca, AEVC.EtaR,
                        AEVC.ShfR, AEVC.EtaA, AEVC.Zeta, AEVC.ShfA, AEVC.ShfZ, AEVC.num_species)

        species0 = torch.concat((self.species[0][0].unsqueeze(0), self.species[0][int(self.index_diff[0]):]))
        coordinates0 = torch.vstack((self.coordinates[0][0], self.coordinates[0][int(self.index_diff[0]):]))
        _, aev0out = AEVC_reg.forward((species0.unsqueeze(0), coordinates0.unsqueeze(0)))
        self.assertTrue((aev0out.squeeze(0)[0] == self.aevs[0][0]).all())

        species1 = torch.concat((self.species[1][1].unsqueeze(0), self.species[1][int(self.index_diff[1]):]))
        coordinates1 = torch.vstack((self.coordinates[1][1], self.coordinates[1][int(self.index_diff[1]):]))
        _, aev1out = AEVC_reg.forward((species1.unsqueeze(0), coordinates1.unsqueeze(0)))
        self.assertTrue((aev1out.squeeze(0)[0] == self.aevs[1][1]).all())

    def test_entry_flips(self):
        entry0 = torch.vstack((self.coordinates[0][int(self.index_diff[0]):].flip(0), \
                self.coordinates[0][:int(self.index_diff[0])].flip(0)))
        entry1 = torch.vstack((self.coordinates[1][:int(self.index_diff[1])], \
                self.coordinates[1][int(self.index_diff[1]):]))

        self.assertTrue((entry0 == entry1).all())
        self.assertTrue((self.species[0] == self.species[1].flip(0)).all())
