import random
import torch

from ._base import _mask_select_data, register_transform


@register_transform('random_interacting_chain')
class RandomInteractingChain(object):

    def __init__(self, interaction_attr):
        super().__init__()
        self.interaction_attr = interaction_attr

    def __call__(self, data):
        interact_flag = (data[self.interaction_attr] > 0)    # (L, )
        if interact_flag.sum() == 0:
            # If there is no active residues, randomly pick one.
            interact_flag[random.randint(0, interact_flag.size(0)-1)] = True
        seed_idx = torch.multinomial(interact_flag.float(), num_samples=1).item()

        chain_nb_selected = data['chain_nb'][seed_idx].item()
        mask_chain = (data['chain_nb'] == chain_nb_selected)
        return _mask_select_data(data, mask_chain)


@register_transform('select_focused')
class SelectFocused(object):

    def __init__(self, focus_attr):
        super().__init__()
        self.focus_attr = focus_attr

    def __call__(self, data):
        mask_focus = (data[self.focus_attr] > 0)
        return _mask_select_data(data, mask_focus)

