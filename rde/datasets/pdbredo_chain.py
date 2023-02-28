import os
import math
import random
import pickle
import collections
import torch
import lmdb
from typing import Mapping, List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from rde.utils.protein.parsers import parse_biopython_structure


ClusterIdType = str
PdbCodeType = str
ChainIdType = str


def _process_structure(cif_path, structure_id) -> Optional[Dict]:
    parser = MMCIFParser(QUIET=True)
    try:
        model = parser.get_structure(structure_id, cif_path)[0]
    except PDBConstructionException:
        print(f'[INFO] Failed to load structure using MMCIFParser: {cif_path}.')
        return None

    data, _ = parse_biopython_structure(model)
    if data is None:
        print(f'[INFO] Failed to parse structure. Too few valid residues: {cif_path}')
        return None
    data['id'] = structure_id
    return data


class PDBRedoChainDataset(Dataset):

    MAP_SIZE = 384*(1024*1024*1024) # 384GB

    def __init__(
        self, 
        split,
        pdbredo_dir = './data/PDB_REDO', 
        clusters_path = './data/pdbredo_clusters.txt',
        splits_path = './data/pdbredo_splits.txt',
        processed_dir = './data/PDB_REDO_processed',
        num_preprocess_jobs = math.floor(cpu_count() * 0.8),
        transform = None,
        reset = False,
    ):
        super().__init__()
        self.pdbredo_dir = pdbredo_dir
        self.clusters_path = clusters_path
        self.splits_path = splits_path
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        self.num_preprocess_jobs = num_preprocess_jobs
        self.transform = transform

        self.clusters: Mapping[
            ClusterIdType, List[Tuple[PdbCodeType, ChainIdType]]
        ] = collections.defaultdict(list)
        self.splits: Mapping[
            str, List[ClusterIdType]
        ] = collections.defaultdict(list)
        self._load_clusters()
        self._load_splits()

        # Structure cache
        self.db_conn = None
        self.db_keys: Optional[List[PdbCodeType]] = None
        self._preprocess_structures(reset)

        # Sanitize clusters
        self._sanitize_clusters(reset)

        # Select clusters of the split
        self._clusters_of_split = [
            c
            for c in self.splits[split]
            if c in self.clusters
        ]

    @property
    def lmdb_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')
    
    @property
    def keys_path(self):
        return os.path.join(self.processed_dir, 'keys.pkl')
    
    @property
    def sanitized_clusters_path(self):
        return os.path.join(self.processed_dir, 'sanitized_clusters.pkl')

    def _load_clusters(self):
        with open(self.clusters_path, 'r') as f:
            lines = f.readlines()
        current_cluster = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                if word[0] == '[' and word[-1] == ']':
                    current_cluster = word[1:-1]
                else:
                    pdbcode, chain_id = word.split(':')
                    self.clusters[current_cluster].append( (pdbcode, chain_id) )
    
    def _load_splits(self):
        with open(self.splits_path, 'r') as f:
            lines = f.readlines()
        current_split = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                if word[0] == '[' and word[-1] == ']':
                    current_split = word[1:-1]
                else:
                    self.splits[current_split].append( word )

    def get_all_pdbcodes(self):
        pdbcodes = set()
        for _, pdbchain_list in self.clusters.items():
            for pdbcode, _ in pdbchain_list:
                pdbcodes.add(pdbcode)
        return pdbcodes

    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path) and not reset:
            return
        pdbcodes = self.get_all_pdbcodes()
        tasks = []
        for pdbcode in pdbcodes:
            cif_path = os.path.join(
                self.pdbredo_dir, pdbcode[1:3], pdbcode, f"{pdbcode}_final.cif"
            )
            if not os.path.exists(cif_path):
                print(f'[WARNING] CIF not found: {cif_path}.')
                continue
            tasks.append(
                delayed(_process_structure)(cif_path, pdbcode)
            )
        
        # Split data into chunks
        chunk_size = 8192
        task_chunks = [
            tasks[i*chunk_size:(i+1)*chunk_size] 
            for i in range(math.ceil(len(tasks)/chunk_size))
        ]

        # Establish database connection
        db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )

        keys = []
        for i, task_chunk in enumerate(task_chunks):
            with db_conn.begin(write=True, buffers=True) as txn:
                processed = Parallel(n_jobs=self.num_preprocess_jobs)(
                    task
                    for task in tqdm(task_chunk, desc=f"Chunk {i+1}/{len(task_chunks)}")
                )
                stored = 0
                for data in processed:
                    if data is None:
                        continue
                    key = data['id']
                    keys.append(key)
                    txn.put(key=key.encode(), value=pickle.dumps(data))
                    stored += 1
                print(f"[INFO] {stored} processed for chunk#{i+1}")
        db_conn.close()

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)

    def _connect_db(self):
        assert self.db_conn is None
        self.db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with open(self.keys_path, 'rb') as f:
            self.db_keys = pickle.load(f)

    def _close_db(self):
        self.db_conn.close()
        self.db_conn = None
        self.db_keys = None

    def _get_from_db(self, pdbcode):
        if self.db_conn is None:
            self._connect_db()
        data = pickle.loads(self.db_conn.begin().get(pdbcode.encode()))   # Made a copy
        return data

    def _sanitize_clusters(self, reset):
        if os.path.exists(self.sanitized_clusters_path) and not reset:
            with open(self.sanitized_clusters_path, 'rb') as f:
                self.clusters = pickle.load(f)
            return

        # Step 1: Find structures and chains that do not exist in PDB_REDO
        clusters_raw = self.clusters
        pdbcode_to_chains: Dict[PdbCodeType, List[ChainIdType]] = collections.defaultdict(list)
        for pdbcode, pdbchain_list in clusters_raw.items():
            for pdbcode, chain in pdbchain_list:
                pdbcode_to_chains[pdbcode].append(chain)
        
        pdb_removed, chain_removed = 0, 0
        pdbcode_to_chains_ok = {}
        self._connect_db()
        for pdbcode, chain_list in tqdm(pdbcode_to_chains.items(), desc='Sanitize'):
            if pdbcode not in self.db_keys: 
                pdb_removed += 1
                continue
            data = self._get_from_db(pdbcode)
            ch_exists = []
            for ch in chain_list:
                if ch in data['chain_id']: 
                    ch_exists.append(ch)
                else:
                    chain_removed += 1
            if len(ch_exists) > 0:
                pdbcode_to_chains_ok[pdbcode] = ch_exists
            else:
                pdb_removed += 1
        
        print(f'[INFO] Structures removed: {pdb_removed}. Chains removed: {chain_removed}.')
        pdbchains_allowed = set(
            (p, c) 
            for p, clist in pdbcode_to_chains_ok.items() 
            for c in clist
        )

        # Step 2: Rebuild the clusters according to the allowed chains.
        pdbchain_to_clust = {}
        for clust_name, pdbchain_list in clusters_raw.items():
            for pdbchain in pdbchain_list:
                if pdbchain in pdbchains_allowed:
                    pdbchain_to_clust[pdbchain] = clust_name

        clusters_sanitized = collections.defaultdict(list)
        for pdbchain, clust_name in pdbchain_to_clust.items():
            clusters_sanitized[clust_name].append(pdbchain)

        print('[INFO] %d clusters after sanitization (from %d).' % (len(clusters_sanitized), len(clusters_raw)))

        with open(self.sanitized_clusters_path, 'wb') as f:
            pickle.dump(clusters_sanitized, f)
        self.clusters = clusters_sanitized

    def __len__(self):
        return len(self._clusters_of_split)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index, None)
        
        # Select cluster
        clust = self._clusters_of_split[index[0]]
        pdbchain_list = self.clusters[clust]

        # Select a pdb-chain from the cluster and retrieve the data point
        if index[1] is None:
            pdbcode, chain = random.choice(pdbchain_list)
        else:
            pdbcode, chain = pdbchain_list[index[1]]
        data = self._get_from_db(pdbcode)   # Made a copy

        # Focus on the chain
        focus_flag = torch.zeros(len(data['chain_id']), dtype=torch.bool)
        for i, ch in enumerate(data['chain_id']):
            if ch == chain: focus_flag[i] = True
        data['focus_flag'] = focus_flag
        data['focus_chain'] = chain

        if self.transform is not None:
            data = self.transform(data)

        return data


def get_pdbredo_chain_dataset(cfg):
    from rde.utils.transforms import get_transform
    return PDBRedoChainDataset(
        split = cfg.split,
        pdbredo_dir = cfg.pdbredo_dir,
        clusters_path = cfg.clusters_path,
        splits_path = cfg.splits_path,
        processed_dir = cfg.processed_dir,
        transform = get_transform(cfg.transform),
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()

    dataset = PDBRedoChainDataset(args.split)

    for data in tqdm(dataset, desc='Iterating'):
        pass
    print(data)
    print(f'[INFO] {len(dataset.clusters)} clusters in the entire dataset.')
    print(f'[INFO] {len(dataset)} samples in the split.')
