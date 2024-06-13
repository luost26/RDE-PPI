import os
import copy
import argparse
import pandas as pd
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rde.datasets.pointmut import PMDataset
from rde.utils.misc import load_config
from rde.utils.data import PaddingCollate
from rde.utils.train import CrossValidation, recursive_to
from rde.utils.transforms import SelectedRegionFixedSizePatch
from rde.models.rde_ddg import DDG_RDE_Network


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
        receptor_chains = config.receptor_chains,
        ligand_chains = config.ligand_chains,
        extra_transform_list=[SelectedRegionFixedSizePatch('mut_flag', 128)],
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
