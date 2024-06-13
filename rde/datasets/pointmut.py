import copy
import torch
from torch.utils.data import Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index

from rde.utils.transforms import Compose, SelectAtom
from rde.utils.protein.parsers import parse_biopython_structure


class PMDataset(Dataset):

    def __init__(self, pdb_path, mutations, receptor_chains, ligand_chains, extra_transform_list=None):
        super().__init__()
        self.pdb_path = pdb_path

        self.data = None
        self.seq_map = None
        self._load_structure()

        self.receptor_chains = receptor_chains
        self.ligand_chains = ligand_chains
        self.mutations = self._parse_mutations(mutations)

        self.transform = Compose([SelectAtom('backbone+CB')] + (extra_transform_list or []))
    
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
            if ch not in self.ligand_chains:
                print(f'Chain {ch} not in ligand chains. Skipping mutation {m}.')
                continue
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
        data['ligand_flag'] = torch.tensor([c in self.ligand_chains for c in data['chain_id']], dtype=torch.bool)
        data = self.transform(data)
        data['ddG'] = 0
        data['mutstr'] = '{}{}{}{}'.format(
            mut['wt'],
            mut['position'][0],
            mut['position'][1],
            mut['mt']
        )
        return data
