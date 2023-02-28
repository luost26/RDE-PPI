import math
import random
import torch
import torch.nn.functional as F

from ._base import register_transform


def _extend_mask(mask, chain_nb):
    """
    Args:
        mask, chain_nb: (L, ).
    """
    # Shift right
    mask_sr = torch.logical_and(
        F.pad(mask[:-1], pad=(1,0), value=0),
        (F.pad(chain_nb[:-1], pad=(1,0), value=-1) == chain_nb)
    )
    # Shift left
    mask_sl = torch.logical_and(
        F.pad(mask[1:], pad=(0,1), value=0),
        (F.pad(chain_nb[1:], pad=(0,1), value=-1) == chain_nb)
    )
    return torch.logical_or(mask, torch.logical_or(mask_sr, mask_sl))


def _mask_sidechains(pos_atoms, mask_atoms, mask_idx):
    """
    Args:
        pos_atoms:  (L, A, 3)
        mask_atoms: (L, A)
    """
    pos_atoms = pos_atoms.clone()
    pos_atoms[mask_idx, 4:] = 0.0

    mask_atoms = mask_atoms.clone()
    mask_atoms[mask_idx, 4:] = False
    return pos_atoms, mask_atoms


@register_transform('random_mask_amino_acids')
class RandomMaskAminoAcids(object):

    def __init__(
        self, 
        mask_ratio_in_all=0.05, 
        ratio_in_maskable_limit=0.5, 
        mask_token=20, 
        maskable_flag_attr='core_flag', 
        extend_maskable_flag=False,
        mask_ratio_mode='constant',
    ):
        super().__init__()
        self.mask_ratio_in_all = mask_ratio_in_all
        self.ratio_in_maskable_limit = ratio_in_maskable_limit
        self.mask_token = mask_token
        self.maskable_flag_attr = maskable_flag_attr
        self.extend_maskable_flag = extend_maskable_flag
        assert mask_ratio_mode in ('constant', 'random')
        self.mask_ratio_mode = mask_ratio_mode

    def __call__(self, data):
        if self.maskable_flag_attr is None:
            maskable_flag = torch.ones([data['aa'].size(0),], dtype=torch.bool)
        else:
            maskable_flag = data[self.maskable_flag_attr]
            if self.extend_maskable_flag:
                maskable_flag = _extend_mask(maskable_flag, data['chain_nb'])
        
        num_masked_max = math.ceil(self.mask_ratio_in_all * data['aa'].size(0))
        if self.mask_ratio_mode == 'random':
            num_masked = random.randint(1, num_masked_max)
        else:
            num_masked = num_masked_max
        mask_idx = torch.multinomial(
            maskable_flag.float() / maskable_flag.sum(), 
            num_samples = num_masked,
        )
        mask_idx = mask_idx[:math.ceil(self.ratio_in_maskable_limit * maskable_flag.sum().item())]
        
        aa_masked = data['aa'].clone()
        aa_masked[mask_idx] = self.mask_token
        data['aa_true'] = data['aa']
        data['aa_masked'] = aa_masked

        data['pos_atoms'], data['mask_atoms'] = _mask_sidechains(
            data['pos_atoms'], data['mask_atoms'], mask_idx
        )

        return data


@register_transform('mask_selected_amino_acids')
class MaskSelectedAminoAcids(object):

    def __init__(self, select_attr, mask_token=20):
        super().__init__()
        self.select_attr = select_attr
        self.mask_token = mask_token

    def __call__(self, data):
        mask_flag = (data[self.select_attr] > 0)

        aa_masked = data['aa'].clone()
        aa_masked[mask_flag] = self.mask_token
        data['aa_true'] = data['aa']
        data['aa_masked'] = aa_masked

        data['pos_atoms'], data['mask_atoms'] = _mask_sidechains(
            data['pos_atoms'], data['mask_atoms'], mask_flag
        )

        return data
