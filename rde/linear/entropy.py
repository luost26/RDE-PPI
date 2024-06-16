import os
import pickle
import copy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index, index_to_one
from tqdm.auto import tqdm

from rde.utils.protein.constants import num_chi_angles
from rde.utils.protein.parsers import parse_biopython_structure
from rde.utils.transforms import *
from rde.utils.train import recursive_to
from rde.models.rde import CircularSplineRotamerDensityEstimator


class SkempiStructureRepo(object):

    def __init__(
        self, 
        root='./data/SKEMPI_v2/PDBs', 
        cache_path='./data/SKEMPI_v2_cache.pkl', 
        reset=False
    ):
        super().__init__()
        self.root = root
        self.cache_path = cache_path
    
        self.data_dict = None
        if not os.path.exists(cache_path) or reset:
            self.data_dict = self._preprocess()
        else:
            with open(cache_path, 'rb') as f:
                self.data_dict = pickle.load(f)

    def _preprocess(self):
        pdbcode_list = []
        for fname in os.listdir(self.root):
            if fname.endswith('.pdb') and fname[0] != ".":
                pdbcode_list.append(fname[:4].lower())

        data_dict = {}
        for pdbcode in tqdm(pdbcode_list, desc='Preprocess SKEMPI'):
            parser = PDBParser(QUIET=True)
            pdb_path = os.path.join(self.root, '{}.pdb'.format(pdbcode.upper()))
            model = parser.get_structure(pdbcode, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model)
            data_dict[pdbcode] = (data, seq_map)
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data_dict, f)
        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, pdbcode):
        item = copy.deepcopy(self.data_dict[pdbcode.lower()])
        return item

    def __contains__(self, pdbcode):
        return pdbcode.lower() in self.data_dict


class SkempiMutationDataset(object):

    def __init__(
        self, 
        structure_repo, 
        csv_path='./data/SKEMPI_v2/skempi_v2.csv',
        entry_filters=[],
        patch_size=128,
    ):
        super().__init__()
        self.repo = structure_repo
        self.csv_path = csv_path
        self.entry_filters = entry_filters
        self.patch_size = patch_size
        self.pre_transform = Compose([
            SelectAtom('backbone+CB')
        ])

        self.entries = None
        self._load_entries_from_csv()

    @staticmethod
    def single_mutation_filter(entry):
        return len(entry['mutations']) == 1

    @staticmethod
    def multiple_mutations_filter(entry):
        return len(entry['mutations']) > 1

    def _entry_filter(self, entry):
        for f in self.entry_filters:
            if not f(entry):
                return False
        return True

    def _load_entries_from_csv(self):
        df = pd.read_csv(self.csv_path, sep=';')
        df['dG_wt'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
        df['dG_mut'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
        df['ddG'] = df['dG_mut'] - df['dG_wt']

        def _parse_mut(mut_name):
            wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
            mutseq = int(mut_name[2:-1])
            return {
                'wt': wt_type,
                'mt': mt_type,
                'chain': mutchain,
                'resseq': mutseq,
                'icode': ' ',
            }

        entries = []
        for i, row in df.iterrows():
            pdbcode, group1, group2 = row['#Pdb'].split('_')
            if pdbcode not in self.repo:
                continue
            muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
            if muts[0]['chain'] in group1:
                group_ligand, group_receptor = group1, group2
            else:
                group_ligand, group_receptor = group2, group1

            entry = {
                'id': i,
                'complex': row['#Pdb'],
                'mutstr': row['Mutation(s)_cleaned'],
                'num_muts': len(muts),
                'pdbcode': pdbcode,
                'group_ligand': list(group_ligand),
                'group_receptor': list(group_receptor),
                'mutations': muts,
                'ddG': row['ddG'],
            }
            if self._entry_filter(entry):
                entries.append(entry)

        self.entries = entries

    @staticmethod
    def _get_Cbeta_positions(pos_atoms, mask_atoms):
        """
        Args:
            pos_atoms:  (L, A, 3)
            mask_atoms: (L, A)
        """
        from rde.utils.protein.constants import BBHeavyAtom
        L = pos_atoms.size(0)
        pos_CA = pos_atoms[:, BBHeavyAtom.CA]   # (L, 3)
        if pos_atoms.size(1) < 5:
            return pos_CA
        pos_CB = pos_atoms[:, BBHeavyAtom.CB]
        mask_CB = mask_atoms[:, BBHeavyAtom.CB, None].expand(L, 3)
        return torch.where(mask_CB, pos_CB, pos_CA)

    @staticmethod
    def _mask_select_data(data, mask):
        def _mask_select(v, mask):
            if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
                return v[mask]
            elif isinstance(v, list) and len(v) == mask.size(0):
                return [v[i] for i, b in enumerate(mask) if b]
            else:
                return v
        return {
            k: _mask_select(v, mask)
            for k, v in data.items()
        }

    @staticmethod
    def _index_select_data(data, index):
        def _index_select(v, index, n):
            if isinstance(v, torch.Tensor) and v.size(0) == n:
                return v[index]
            elif isinstance(v, list) and len(v) == n:
                return [v[i] for i in index]
            else:
                return v
        return {
            k: _index_select(v, index, data['aa'].size(0))
            for k, v in data.items()
        }

    def _pad_data(self, data):
        def _pad_last(x, n, value=0):
            if isinstance(x, torch.Tensor):
                assert x.size(0) <= n
                if x.size(0) == n:
                    return x
                pad_size = [n - x.size(0)] + list(x.shape[1:])
                pad = torch.full(pad_size, fill_value=value).to(x)
                return torch.cat([x, pad], dim=0)
            elif isinstance(x, list):
                pad = [value] * (n - len(x))
                return x + pad
            else:
                return x
        
        ref_length = data['aa'].shape[0]
        if ref_length >= self.patch_size:
            return data

        data_padded = {}
        for k, v in data.items():
            if len(v) == ref_length:
                data_padded[k] = _pad_last(v, self.patch_size)
            else:
                data_padded[k] = v
        return data_padded        

    def __len__(self):
        return len(self.entries)

    def get(self, idx, group, state):
        assert group in ('ligand', 'receptor', 'complex')
        assert state in ('mt', 'wt')
        entry = self.entries[idx]
        data, seq_map = self.repo[entry['pdbcode']]
        data = self.pre_transform(data)

        mutation_flag = torch.zeros((data['aa'].shape[0]), dtype=torch.bool)
        chi_corrupt = data['chi'].clone()
        mut_beta_positions = []
        for mutation in entry['mutations']:
            position = (mutation['chain'], mutation['resseq'], mutation['icode'])
            seq_idx = seq_map[position]
            mutation_flag[seq_idx] = True
            chi_corrupt[seq_idx] = 0.0

            # Mutate the protein
            if state == 'mt':
                mtype = one_to_index(mutation['mt'])
                data['aa'][seq_idx] = mtype
                data['chi'][seq_idx] = 0.0
                data['chi_alt'][seq_idx] = 0.0
                data['chi_mask'][seq_idx] = False
                data['chi_mask'][seq_idx, :num_chi_angles[mtype]] = True

            pos_atom = data['pos_heavyatom'][seq_idx, :5]   # (5, 3)
            msk_atom = data['mask_heavyatom'][seq_idx, :5]  # (5,)
            beta_pos = pos_atom[4] if msk_atom[4].item() else pos_atom[1]
            mut_beta_positions.append(beta_pos)
            
        mut_beta_positions = torch.stack(mut_beta_positions)    # (M, 3)
        data['chi_masked_flag'] = mutation_flag
        data['chi_corrupt'] = chi_corrupt

        # For each residue, compute the distance to the closest mutated residue
        beta_pos = self._get_Cbeta_positions(data['pos_heavyatom'], data['mask_heavyatom'])
        pw_dist = torch.cdist(beta_pos, mut_beta_positions) # (N, M)
        dist_to_mut = pw_dist.min(dim=1)[0] # (N, )
        data['dist_to_mut'] = dist_to_mut

        # Flags
        receptor_flag = torch.BoolTensor([
            (c in entry['group_receptor']) for c in data['chain_id']
        ])
        ligand_flag = torch.BoolTensor([
            (c in entry['group_ligand']) for c in data['chain_id']
        ])
        data['receptor_flag'] = receptor_flag
        data['ligand_flag'] = ligand_flag

        # Add the information of closest residues in the receptor
        receptors = []
        rec_idx = torch.logical_and(
            dist_to_mut <= 8.0,
            receptor_flag
        ).nonzero().flatten()
        for idx in rec_idx:
            receptors.append({
                'chain': data['chain_id'][idx],
                'resseq': data['resseq'][idx].item(),
                'icode': data['icode'][idx],
                'type': index_to_one(data['aa'][idx].item()),
                'distance': dist_to_mut[idx].item(),
            })
        entry['receptors'] = receptors

        # Add the information of closest residues in the ligand
        lignbrs = []
        lig_idx = torch.logical_and(
            dist_to_mut <= 8.0,
            ligand_flag
        ).nonzero().flatten()
        for idx in lig_idx:
            lignbrs.append({
                'chain': data['chain_id'][idx],
                'resseq': data['resseq'][idx].item(),
                'icode': data['icode'][idx],
                'type': index_to_one(data['aa'][idx].item()),
                'distance': dist_to_mut[idx].item(),
            })
        entry['lignbrs'] = lignbrs

        # Select the chain group 
        if group == 'ligand':
            group_mask = torch.BoolTensor([
                (c in entry['group_ligand']) for c in data['chain_id']
            ])
        elif group == 'receptor':
            group_mask = torch.BoolTensor([
                (c in entry['group_receptor']) for c in data['chain_id']
            ])
        else:
            group_mask = torch.ones((data['aa'].shape[0]), dtype=torch.bool)
        data = self._mask_select_data(data, group_mask)

        # Patch or pad
        patch_idx = data['dist_to_mut'].argsort()[:self.patch_size]
        data = self._index_select_data(data, patch_idx)
        data = self._pad_data(data)

        # Add tags
        data['entry'] = entry
        data['group'] = group
        data['state'] = state
        return data


class Expand(object):

    def __init__(self, mutation_dataset):
        super().__init__()
        self.mutation_dataset = mutation_dataset
        self.variants = [
            ('ligand', 'wt'), ('ligand', 'mt'),     # Unbound ligand
            ('receptor', 'wt'),                     # Unbound receptor
            ('complex', 'wt'), ('complex', 'mt'),   # Bound
        ]

    def __len__(self):
        return len(self.mutation_dataset) * len(self.variants)

    def __getitem__(self, idx):
        idx, variant = divmod(idx, len(self.variants))
        group, state = self.variants[variant]
        return self.mutation_dataset.get(idx, group, state)


def collate(data_list):
    batch = {}
    for k in data_list[0].keys():
        merge = [data[k] for data in data_list]
        if isinstance(data_list[0][k], torch.Tensor):
            batch[k] = torch.stack(merge, dim=0)
        else:
            batch[k] = merge
    return batch


def batch_to_data_list(batch):
    data_list = []
    for i in range(batch['aa'].shape[0]):
        data = {}
        for k, v in batch.items():
            data[k] = v[i]
        data_list.append(data)
    return data_list


class ResultAccumulator(object):

    def __init__(self):
        super().__init__()
        self.results = {}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.results = pickle.load(f)

    @staticmethod
    def _position_to_idx(data, chain, resseq, icode):
        for i, (ch, rs, ic) in enumerate(zip(data['chain_id'], data['resseq'], data['icode'])):
            if ch == chain and rs == resseq and ic == icode:
                return i
        return None

    def _add_unbound_ligand(self, data, state):
        assert state in ('wt', 'mt')
        result = self.results[data['entry']['id']]

        for mutation in result['mutations']:
            idx = self._position_to_idx(
                data,
                mutation['chain'], mutation['resseq'], mutation['icode']
            )
            if idx is None:
                continue
            mutation['H_ub_%s' % state] = data['entropy'][idx].item()

        for lignbr in result['lignbrs']:
            idx = self._position_to_idx(
                data,
                lignbr['chain'], lignbr['resseq'], lignbr['icode']
            )
            if idx is None:
                continue
            lignbr['H_ub_%s' % state] = data['entropy'][idx].item()

    def _add_unbound_receptor(self, data):
        result = self.results[data['entry']['id']]

        for receptor in result['receptors']:
            idx = self._position_to_idx(
                data,
                receptor['chain'], receptor['resseq'], receptor['icode']
            )
            if idx is None:
                continue
            receptor['H_ub'] = data['entropy'][idx].item()

    def _add_bound_complex(self, data, state):
        result = self.results[data['entry']['id']]

        for mutation in result['mutations']:
            idx = self._position_to_idx(
                data,
                mutation['chain'], mutation['resseq'], mutation['icode']
            )
            if idx is None:
                continue
            mutation['H_b_%s' % state] = data['entropy'][idx].item()

        for lignbr in result['lignbrs']:
            idx = self._position_to_idx(
                data,
                lignbr['chain'], lignbr['resseq'], lignbr['icode']
            )
            if idx is None:
                continue
            lignbr['H_b_%s' % state] = data['entropy'][idx].item()

        for receptor in result['receptors']:
            idx = self._position_to_idx(
                data,
                receptor['chain'], receptor['resseq'], receptor['icode']
            )
            if idx is None:
                continue
            receptor['H_b_%s' % state] = data['entropy'][idx].item()

    def add(self, data):
        entry = data['entry']
        if entry['id'] not in self.results:
            self.results[entry['id']] = {
                'complex': entry['complex'],
                'mutstr': entry['mutstr'],
                'pdbcode': entry['pdbcode'],
                'mutations': entry['mutations'],
                'lignbrs': entry['lignbrs'],
                'group_ligand': entry['group_ligand'],
                'group_receptor': entry['group_receptor'],
                'receptors': entry['receptors'],
                'ddG': entry['ddG'],
            }
        if data['group'] == 'ligand' and data['state'] == 'mt':
            self._add_unbound_ligand(data, state='mt')
        elif data['group'] == 'ligand' and data['state'] == 'wt':
            self._add_unbound_ligand(data, state='wt')
        elif data['group'] == 'receptor':
            self._add_unbound_receptor(data)
        elif data['group'] == 'complex' and data['state'] == 'mt':
            self._add_bound_complex(data, state='mt')
        elif data['group'] == 'complex' and data['state'] == 'wt':
            self._add_bound_complex(data, state='wt')
        else:
            raise ValueError('Unknown group/state combination')


def get_entropy(
    ckpt_path, 
    device='cuda', 
    skempi_dir='./data/SKEMPI_v2/PDBs',
    skempi_cache_path='./data/SKEMPI_v2_cache.pkl',
):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = CircularSplineRotamerDensityEstimator(ckpt['config']['model']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    torch.set_grad_enabled(False)

    mut_dataset = Expand(SkempiMutationDataset(
        SkempiStructureRepo(
            root=skempi_dir,
            cache_path=skempi_cache_path,
        ),
        entry_filters=[
            # SkempiMutationDataset.single_mutation_filter
        ],
    ))
    results = ResultAccumulator()
    loader = DataLoader(mut_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)
    for i, batch in enumerate(tqdm(loader, desc='Computing entropy')):
        batch = recursive_to(batch, device)
        xs, logprobs = model.sample(batch, n_samples=200)
        entropys = -logprobs.mean(dim=0)    # (B, L)
        batch['entropy'] = entropys

        data_list = batch_to_data_list(recursive_to(batch, 'cpu'))
        for data in data_list:
            results.add(data)

    return results.results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, default='./trained_models/RDE.pt')
    parser.add_argument('-o', '--output', type=str, default='./RDE_skempi_entropy.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    results = get_entropy(args.ckpt, args.device)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
