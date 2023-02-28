import os
import argparse
import torch

from rde.utils.skempi import eval_skempi_three_modes
from .entropy import get_entropy
from .calibrate import run_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./trained_models/RDE.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skempi_dir', type=str, default='./data/SKEMPI_v2/PDBs')
    parser.add_argument('--skempi_cache_path', type=str, default='./data/SKEMPI_v2_cache.pkl')
    parser.add_argument('--output_dir', type=str, default='./RDE_linear_skempi')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    entropy_cache_path = os.path.join(args.output_dir, 'entropy.pt')
    if os.path.exists(entropy_cache_path):
        entropy = torch.load(entropy_cache_path)
        print(f'[INFO] Using entropy cache: {entropy_cache_path}.')
    else:
        entropy = get_entropy(
            ckpt_path = args.ckpt,
            device = args.device,
            skempi_dir = args.skempi_dir,
            skempi_cache_path = args.skempi_cache_path,
        )
        torch.save(entropy, os.path.join(args.output_dir, 'entropy.pt'))

    results = run_calibration(entropy)
    results.to_csv(os.path.join(args.output_dir, 'results.csv'))
    results['method'] = 'RDE-Linear'

    metrics = eval_skempi_three_modes(results)
    metrics.to_csv(os.path.join(args.output_dir, 'metrics.csv'))
    print(metrics)


if __name__ == '__main__':
    main()
    