import torch
import numpy as np

from ._base import register_transform


@register_transform('add_atom_noise')
class AddAtomNoise(object):

    def __init__(self, noise_std=0.02):
        super().__init__()
        self.noise_std = noise_std

    def __call__(self, data):
        pos_atoms = data['pos_atoms']   # (L, A, 3)
        mask_atoms = data['mask_atoms'] # (L, A)
        noise = (torch.randn_like(pos_atoms) * self.noise_std) * mask_atoms[:, :, None]
        pos_noisy = pos_atoms + noise
        data['pos_atoms'] = pos_noisy
        return data


@register_transform('add_chi_angle_noise')
class AddChiAngleNoise(object):

    def __init__(self, noise_std=0.02):
        super().__init__()
        self.noise_std = noise_std

    def _normalize_angles(self, angles):
        angles = angles % (2*np.pi)
        return torch.where(angles > np.pi, angles - 2*np.pi, angles)

    def __call__(self, data):
        chi, chi_alt = data['chi'], data['chi_alt'] # (L, 4)
        chi_mask = data['chi_mask'] # (L, 4)

        _get_noise = lambda: ((torch.randn_like(chi) * self.noise_std) * chi_mask)
        data['chi'] = self._normalize_angles( chi + _get_noise() )
        data['chi_alt'] = self._normalize_angles( chi_alt + _get_noise() )
        return data
