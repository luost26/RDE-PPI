import os
import copy
import argparse
import pandas as pd
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader, Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm

from rde.utils.misc import load_config, seed_all
from rde.utils.data import PaddingCollate
from rde.utils.train import *
from rde.utils.transforms import Compose, SelectAtom, SelectedRegionFixedSizePatch
from rde.utils.protein.parsers import parse_biopython_structure
from rde.models.rde_ddg import DDG_RDE_Network


class PMDataset(Dataset):

    def __init__(self, pdb_path, mutations):
        super().__init__()
        self.pdb_path = pdb_path

        self.data = None
        self.seq_map = None
        self._load_structure()

        self.mutations = self._parse_mutations(mutations)
        self.transform = Compose([
            SelectAtom('backbone+CB'),
            SelectedRegionFixedSizePatch('mut_flag', 128)
        ])

    
    def clone_data(self):
        return copy.deepcopy(self.data)

    def _load_structure(self):
        if self.pdb_path.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        elif self.pdb_path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError('Unknown file type.')

        structure = parser.get_structure(None, self.pdb_path)
        data, seq_map = parse_biopython_structure(structure[0])
        self.data = data
        self.seq_map = seq_map

    def _parse_mutations(self, mutations):
        parsed = []
        for m in mutations:
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])
            pos = (ch, seq, ' ')
            if pos not in self.seq_map: continue

            if mt == '*':
                for mt_idx in range(20):
                    mt = index_to_one(mt_idx)
                    if mt == wt: continue
                    parsed.append({
                        'position': pos,
                        'wt': wt,
                        'mt': mt,
                    })
            else:
                parsed.append({
                    'position': pos,
                    'wt': wt,
                    'mt': mt,
                })
        return parsed

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, index):
        data = self.clone_data()
        mut = self.mutations[index]
        mut_pos_idx = self.seq_map[mut['position']]

        data['mut_flag'] = torch.zeros(size=data['aa'].shape, dtype=torch.bool)
        data['mut_flag'][mut_pos_idx] = True
        data['aa_mut'] = data['aa'].clone()
        data['aa_mut'][mut_pos_idx] = one_to_index(mut['mt'])
        data = self.transform(data)
        data['ddG'] = 0
        data['mutstr'] = '{}{}{}{}'.format(
            mut['wt'],
            mut['position'][0],
            mut['position'][1],
            mut['mt']
        )
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-o', '--output', type=str, default='pm_results.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    config, _ = load_config(args.config)

    # Model
    ckpt = torch.load(config.checkpoint, map_location='cpu')
    cv_mgr = CrossValidation(model_factory=DDG_RDE_Network, config=ckpt['config'], num_cvfolds=3)
    cv_mgr.load_state_dict(ckpt['model'])
    cv_mgr.to(args.device)
    model = cv_mgr.models[0]

    # Data
    dataset = PMDataset(
        pdb_path = config.pdb,
        mutations = config.mutations,
    )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=PaddingCollate(), 
    )

    result = []
    for batch in tqdm(loader):
        batch = recursive_to(batch, args.device)
        for fold in range(cv_mgr.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()
            with torch.no_grad():
                _, out_dict = model(batch)
            for mutstr, ddG_pred in zip(batch['mutstr'], out_dict['ddG_pred'].cpu().tolist()):
                result.append({
                    'mutstr': mutstr,
                    'ddG_pred': ddG_pred,
                })
    result = pd.DataFrame(result)
    result = result.groupby('mutstr').mean().reset_index()
    result['rank'] = result['ddG_pred'].rank() / len(result)
    print(result)
    print(f'Results saved to {args.output}.')
    result.to_csv(args.output)

    if 'interest' in config and config.interest:
        print(result[result['mutstr'].isin(config.interest)])
