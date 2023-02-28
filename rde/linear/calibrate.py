import os
import argparse
import pickle
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats
from Bio.PDB.Polypeptide import one_to_index


def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def convert_results_to_table(result):
    # Filter
    result_filtered = {}
    for index, item in result.items():
        mutations = item['mutations']
        chains_having_muts = set(map(lambda m: m['chain'], mutations))
        # if len(mutations) > 1:
        #     continue
        if len(chains_having_muts) > 1:
            # print('Ignore', index)
            continue
        result_filtered[index] = item
    result = result_filtered

    max_mutations = max([len(r['mutations']) for r in result.values()])
    max_receptors = max([len(r['receptors']) for r in result.values()])
    max_lignbrs = max([len(r['lignbrs']) for r in result.values()])

    table = []
    for index, item in result.items():
        row = {
            'index': index,
            'pdbcode': item['pdbcode'],
            # 'complex': '{}_{}_{}'.format(
            #     item['pdbcode'],
            #     ''.join(sorted(item['group_ligand'])),
            #     ''.join(sorted(item['group_receptor'])),
            # ),
            'complex': item['complex'],
            'mutstr': item['mutstr'],
            'num_muts': len(item['mutations']),
            'ddG': item['ddG']
        }

        mutations = item['mutations']
        for i, mutation in enumerate(mutations):
            row['wt_%d' % i] = one_to_index(mutation['wt'])
            row['mt_%d' % i] = one_to_index(mutation['mt'])
            row['H_lig_ub_wt_%d' % i] = mutation['H_ub_wt']
            row['H_lig_ub_mt_%d' % i] = mutation['H_ub_mt']
            row['H_lig_b_wt_%d' % i] = mutation['H_b_wt']
            row['H_lig_b_mt_%d' % i] = mutation['H_b_mt']

        lignbrs = item['lignbrs']
        lignbrs.sort(key=lambda x: x['distance'])
        for i, lignbr in enumerate(lignbrs):
            row['lignbr_aa_%d' % i] = one_to_index(lignbr['type'])
            row['lignbr_distance_%d' % i] = lignbr['distance']
            row['H_lignbr_ub_wt_%d' % i] = lignbr['H_ub_wt']
            row['H_lignbr_ub_mt_%d' % i] = lignbr['H_ub_mt']
            row['H_lignbr_b_wt_%d' % i] = lignbr['H_b_wt']
            row['H_lignbr_b_mt_%d' % i] = lignbr['H_b_mt']

        receptors = item['receptors']
        receptors.sort(key=lambda m: m['distance'])
        for i, receptor in enumerate(receptors):
            row['rec_aa_%d' % i] = one_to_index(receptor['type'])
            row['rec_distance_%d' % i] = receptor['distance']
            row['H_rec_ub_%d' % i] = receptor['H_ub']
            row['H_rec_b_wt_%d' % i] = receptor['H_b_wt']
            row['H_rec_b_mt_%d' % i] = receptor['H_b_mt']

        table.append(row)

    table = pd.DataFrame(table)
    for i in range(max_mutations):
        table = table.astype({
            'wt_%d' % i: 'Int64',
            'mt_%d' % i: 'Int64',
        })
    for i in range(max_lignbrs):
        table = table.astype({
            'lignbr_aa_%d' % i: 'Int64',
        })
    for i in range(max_receptors):
        table = table.astype({
            'rec_aa_%d' % i: 'Int64',
        })
    
    return table


def pearson_loss(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost


def ccc_loss(y_hat, y_true):
    eps = 1e-6
    y_true_mean = torch.mean(y_true)
    y_hat_mean = torch.mean(y_hat)
    y_true_var = torch.var(y_true)
    y_hat_var = torch.var(y_hat)
    y_true_std = torch.std(y_true)
    y_hat_std = torch.std(y_hat)
    vx = y_true - torch.mean(y_true)
    vy = y_hat - torch.mean(y_hat)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + eps) * torch.sqrt(torch.sum(vy ** 2) + eps))
    ccc = (2 * pcc * y_true_std * y_hat_std) / \
          (y_true_var + y_hat_var + (y_hat_mean - y_true_mean) ** 2)
    ccc = 1 - ccc
    return ccc


def grouped_ccc_loss(y_hat, y_true, group, min_points=10):
    losses = []
    for i in range(group.max().item()):
        mask = (group == i)
        if mask.sum().item() < min_points:
            continue
        losses.append(ccc_loss(
            y_hat = y_hat[mask],
            y_true = y_true[mask],
        ))
    losses = torch.stack(losses)
    loss = losses.mean()
    return loss


def split_table(table, num_folds=3, fold_id=0, seed=2022):
    assert fold_id < num_folds
    pdbcodes = table['pdbcode'].unique()
    random.Random(seed).shuffle(pdbcodes)

    fold_size = math.ceil( len(pdbcodes) / num_folds )
    train_pdbcodes, val_pdbcodes, test_pdbcodes = [], [], []
    for i in range(num_folds):
        fold = pdbcodes[i * fold_size: (i + 1) * fold_size]
        if i == fold_id:
            train_pdbcodes.extend(fold)
        elif i == (fold_id+1) % num_folds:
            val_pdbcodes.extend(fold)
        else:
            test_pdbcodes.extend(fold)

    train_mask = table['pdbcode'].isin(train_pdbcodes)
    val_mask = table['pdbcode'].isin(val_pdbcodes)
    test_mask = table['pdbcode'].isin(test_pdbcodes)
    return table[train_mask], table[val_mask], table[test_mask]


class TableToArrays(object):

    def __init__(self, table):
        super().__init__()
        self.table = table

        max_muts, max_nbrs, max_recs = 0, 0, 0
        for col in table.columns:
            if col.startswith('H_lig_ub_wt_'):
                max_muts = max(max_muts, int(col.split('_')[-1])+1)
            elif col.startswith('H_lignbr_ub_wt_'):
                max_nbrs = max(max_nbrs, int(col.split('_')[-1])+1)
            elif col.startswith('H_rec_ub_'):
                max_recs = max(max_recs, int(col.split('_')[-1])+1)
        self.max_muts = max_muts
        self.max_nbrs = max_nbrs
        self.max_recs = max_recs

    def create_ligand_arrays(self, n):
        table = self.table
        H_ub_wt, H_ub_mt, H_b_wt, H_b_mt = [], [], [], []
        T_wt, T_mt = [], []
        for i in range(n):
            try:
                H_ub_wt.append(table['H_lig_ub_wt_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_ub_mt.append(table['H_lig_ub_mt_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_b_wt.append(table['H_lig_b_wt_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_b_mt.append(table['H_lig_b_mt_%d' % i].fillna(0).astype(np.float32).to_numpy())

                T_wt.append(table['wt_%d' % i].fillna(20).astype(np.int64).to_numpy())
                T_mt.append(table['mt_%d' % i].fillna(20).astype(np.int64).to_numpy())
            except KeyError:
                break

        arrays = [H_ub_wt, H_ub_mt, H_b_wt, H_b_mt, T_wt, T_mt]
        arrays = [np.stack(a, axis=0) for a in arrays]
        H_ub_wt, H_ub_mt, H_b_wt, H_b_mt, T_wt, T_mt = arrays
        
        H_arr = np.stack([H_ub_wt, H_ub_mt, H_b_wt, H_b_mt], axis=0)    # (num_terms, num_residues, num_data)
        T_arr = np.stack([T_wt, T_mt, T_wt, T_mt], axis=0)
        labels = ['H_lig_ub_wt', 'H_lig_ub_mt', 'H_lig_b_wt', 'H_lig_b_mt']
        return H_arr, T_arr, labels

    def create_lignbr_arrays(self, n):
        table = self.table
        H_ub_wt, H_ub_mt, H_b_wt, H_b_mt = [], [], [], []
        T, D = [], []
        for i in range(1, n+1):
            try:
                H_ub_wt.append(table['H_lignbr_ub_wt_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_ub_mt.append(table['H_lignbr_ub_mt_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_b_wt.append(table['H_lignbr_b_wt_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_b_mt.append(table['H_lignbr_b_mt_%d' % i].fillna(0).astype(np.float32).to_numpy())

                T.append(table['lignbr_aa_%d' % i].fillna(20).astype(np.int64).to_numpy())
                D.append(table['lignbr_distance_%d' % i].fillna(float('+inf')).astype(np.float32).to_numpy())
            except KeyError:
                break

        arrays = [H_ub_wt, H_ub_mt, H_b_wt, H_b_mt, T, D]
        arrays = [np.stack(a, axis=0) for a in arrays]
        H_ub_wt, H_ub_mt, H_b_wt, H_b_mt, T, D = arrays
        
        H_arr = np.stack([H_ub_wt, H_ub_mt, H_b_wt, H_b_mt], axis=0)
        T_arr = np.stack([T] * 4, axis=0)
        D_arr = np.stack([D] * 4, axis=0)
        labels = ['H_lignbr_ub_wt', 'H_lignbr_ub_mt', 'H_lignbr_b_wt', 'H_lignbr_b_mt']
        
        return H_arr, T_arr, D_arr, labels

    def create_receptor_arrays(self, n):
        table = self.table
        H_ub, H_b_wt, H_b_mt = [], [], []
        T, D = [], []
        for i in range(n):
            try:
                H_ub.append(table['H_rec_ub_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_b_wt.append(table['H_rec_b_wt_%d' % i].fillna(0).astype(np.float32).to_numpy())
                H_b_mt.append(table['H_rec_b_mt_%d' % i].fillna(0).astype(np.float32).to_numpy())

                T.append(table['rec_aa_%d' % i].fillna(20).astype(np.int64).to_numpy())
                D.append(table['rec_distance_%d' % i].fillna(float('+inf')).astype(np.float32).to_numpy())
            except KeyError:
                break

        arrays = [H_ub, H_b_wt, H_b_mt, T, D]
        arrays = [np.stack(a, axis=0) for a in arrays]
        H_ub, H_b_wt, H_b_mt, T, D = arrays
        
        H_arr = np.stack([H_ub, H_b_wt, H_b_mt], axis=0)
        T_arr = np.stack([T] * 3, axis=0)
        D_arr = np.stack([D] * 3, axis=0)
        labels = ['H_rec_ub', 'H_rec_b_wt', 'H_rec_b_mt']
        return H_arr, T_arr, D_arr, labels

    def create_ddG_array(self):
        table = self.table
        ddG = table['ddG'].fillna(0).astype(np.float32).to_numpy()
        return ddG

    def create_group_list(self):
        return self.table['complex'].tolist()

    def create_group_tensor(self):
        group_unique = self.table['complex'].unique().tolist()
        group_list = self.table['complex'].tolist()
        return torch.LongTensor([group_unique.index(g) for g in group_list])

    def create_mutstr_list(self):
        return self.table['mutstr'].tolist()

    def create_num_muts_list(self):
        return self.table['num_muts'].tolist()

    def to_tensor(self, device):
        H_lig, T_lig, labels_lig = self.create_ligand_arrays(n=32)
        H_rec, T_rec, D_rec, labels_rec = self.create_receptor_arrays(n=16)
        # H_nbr, T_nbr, D_nbr, labels_nbr = self.create_lignbr_arrays(n=32)
        ddG = self.create_ddG_array()

        numpy_to_device = lambda x: torch.from_numpy(x).to(device)

        Hs = list(map(
            numpy_to_device, 
            [H_lig, H_rec, ]
        ))
        Ts = list(map(
            numpy_to_device, 
            [T_lig, T_rec, ]
        ))
        labels = labels_lig + labels_rec 

        ddG = torch.from_numpy(ddG).to(device)
        groups = self.create_group_list()
        grp_tensor = self.create_group_tensor().to(device)

        return Hs, Ts, ddG, labels, groups, grp_tensor

class Regression(nn.Module):

    def __init__(self, num_terms, labels=None):
        super().__init__()
        self.aa_ref = nn.Embedding(21, embedding_dim=1, padding_idx=20)
        self.aa_ref.weight.data.zero_()

        self.regr_coef_ = nn.Parameter(torch.randn(num_terms) * 0.01)

        self.regr_bias = nn.Parameter(torch.randn([1, ]) * 0.01)
        if labels is None:
            labels = ['c%d' % i for i in range(num_terms)]
        self.labels = labels

    @property
    def regr_coef(self):
        return self.regr_coef_

    def set_aa_ref_trainable(self, trainable):
        self.aa_ref.weight.requires_grad_(trainable)

    def set_regr_trainable(self, trainable):
        self.regr_coef.requires_grad_(trainable)
        self.regr_bias.requires_grad_(trainable)

    def add_ref(self, x, t):
        r = self.aa_ref(t)[..., 0]
        return x + r

    def print_weights(self):
        print('Weights')
        for label, w in zip(self.labels, self.regr_coef):
            print(' > {}: {:.6f}'.format(label, w.item()))

    def forward(self, Hs, Ts, y_true, grp_tensor, aa_ref_trainable=True, regr_trainable=True, loss_mode='mix'):
        """
        Args:
            Hs: [(n_terms_i, n_residues_i, n_data), ...]
            Ts: [(n_terms_i, n_residues_i, n_data), ...]
            grp_tensor: (n_data, )
            y_true: (n_data, )
        Returns:
            (n_data)
        """
        self.set_aa_ref_trainable(aa_ref_trainable)
        self.set_regr_trainable(regr_trainable)

        Hs = [self.add_ref(H, T) for H, T in zip(Hs, Ts)]
        Hs = [H.sum(dim=1) for H in Hs]     # [(n_terms_i, n_data), ...]
        X = torch.cat(Hs, dim=0)           # (n_terms, n_data)

        y_pred = torch.matmul(self.regr_coef, X) + self.regr_bias    # (n_data)

        if y_true is not None:
            if loss_mode == 'mix':
                loss = ccc_loss(y_pred, y_true) + 0.5*F.mse_loss(y_pred, y_true)
            elif loss_mode == 'mse':
                loss = F.mse_loss(y_pred, y_true)
            # loss = grouped_ccc_loss(y_pred, y_true, grp_tensor)
            return y_pred, loss
        return y_pred


def correlation(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    pearson = np.corrcoef(y_pred, y_true)[0, 1]

    spearman = scipy.stats.spearmanr(y_pred, y_true)[0]
    return pearson, spearman


def grouped_correlation(y_pred, y_true, group):
    df = pd.DataFrame({
        'group': group,
        'y_pred': y_pred.detach().cpu().numpy(),
        'y_true': y_true.detach().cpu().numpy(),
    })
    corrs = []
    for grp in df['group'].unique():
        df_grp = df[df['group'] == grp]
        if len(df_grp) < 10: continue
        pearson = df_grp[['y_pred', 'y_true']].corr('pearson').iloc[0, 1]
        spearman = df_grp[['y_pred', 'y_true']].corr('spearman').iloc[0, 1]
        corrs.append({
            'group': grp,
            'count': len(df_grp),
            'pearson': pearson,
            'spearman': spearman,
        })
    
    df_corr = pd.DataFrame(corrs)
    return df_corr


def fit(table_train, table_val, table_test, device, iters, alternate=None):
    torch.set_grad_enabled(True)
    arrayer_train = TableToArrays(table_train)
    Hs_train, Ts_train, ddG_train, labels, _, grp_tensor_train = arrayer_train.to_tensor(device)

    arrayer_val = TableToArrays(table_val)
    Hs_val, Ts_val, ddG_val, _, groups_val, grp_tensor_val = arrayer_val.to_tensor(device)
    
    arrayer_test = TableToArrays(table_test)
    Hs_test, Ts_test, ddG_test, _, groups_test, grp_tensor_test = arrayer_test.to_tensor(device)
    
    model = Regression(num_terms=len(labels), labels=labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    for i in range(1, iters+1):
        aa_ref_trainable, regr_trainable = True, True
        if alternate is not None and alternate > 0:
            regr_trainable   = ((i // alternate) % 2) == 0
            aa_ref_trainable = ((i // alternate) % 2) == 1

        model.train()
        y_pred, loss = model(
            Hs_train, 
            Ts_train, 
            ddG_train,
            grp_tensor_train,
            aa_ref_trainable = aa_ref_trainable,
            regr_trainable = regr_trainable,
        )
        train_loss = loss.clone().detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if optimizer.param_groups[0]['lr'] > 0.01:
            scheduler.step()

        if i % 100 == 0:
            with torch.no_grad():
                model.eval()
                y_pred, loss = model(
                    Hs_test, 
                    Ts_test, 
                    ddG_test,
                    grp_tensor_test,
                )
                _, loss_val = model(
                    Hs_val,
                    Ts_val,
                    ddG_val,
                    grp_tensor_val,
                )
                y_pred, loss = y_pred.cpu(), loss.cpu()
                pearson, spearman = correlation(y_pred, ddG_test)
                df_corr = grouped_correlation(y_pred, ddG_test, groups_test)
                corr_mean = df_corr[['pearson', 'spearman']].mean()
                print('Iteration {}, LR: {:.6f}, Loss(Train) {:.6f}, Loss(Val) {:.6f}, Loss(Test) {:.6f}, Pearson {:.4f}, Spearman {:.4f}, Pearson(G) {:.4f}, Spearman(G) {:.4f}'.format(
                    i, optimizer.param_groups[0]['lr'], train_loss, loss_val, loss.item(), pearson, spearman,
                    corr_mean['pearson'], corr_mean['spearman']
                ))

    df_raw = pd.DataFrame({
        'complex': arrayer_test.create_group_list(),
        'mutstr': arrayer_test.create_mutstr_list(),
        'num_muts': arrayer_test.create_num_muts_list(),
        'ddG': ddG_test.detach().cpu().numpy(),
        'ddG_pred': y_pred.detach().cpu().numpy(),
    })

    return df_corr, df_raw, model, loss_val.item()


def run_calibration(
    result, 
    seed=2020, 
    num_trials=10, 
    iters=2000, 
    num_folds=3, 
    device='cuda', 
    alternate=100,  # Block descend
    output_dir='./RDE_linear_skempi',
):
    seed_all(seed)
    os.makedirs(output_dir, exist_ok=True)

    block_list = {'1KBH'}   # This structure in SKEMPI is problematic
    result = {
        k: v 
        for k, v in result.items() 
        if v['pdbcode'] not in block_list
    }
    table = convert_results_to_table(result)

    seeds = [np.random.randint(0, 1000000) for _ in range(num_trials)]
    df_result = []
    for i, seed in enumerate(seeds):
        for fold_id in range(num_folds):
            print(f'---------- ROUND {i} SEED {seed} CV {fold_id} ----------')
            table_train, table_val, table_test = split_table(
                table, 
                seed=seed, 
                num_folds=num_folds,
                fold_id=fold_id,
            )

            df_corr, df_raw, model, loss = fit(
                table_train = table_train,
                table_val = table_val,
                table_test = table_test,
                iters = iters,
                device = device,
                alternate = alternate,
            )
            if loss <= 3:   # Selecting models according to VALIDATION loss (not test loss)
                df_result.append(df_raw)
                print(df_corr)
                model.print_weights()
            else:
                print(f'[INFO] Validation loss too high, skipping {i}:{fold_id}')
            df_corr.to_csv(os.path.join(output_dir, f'corr_{i}_{fold_id}.csv'))
            df_raw.to_csv(os.path.join(output_dir, f'raw_{i}_{fold_id}.csv'))
            torch.save({
                'state_dict': model.state_dict(),
                'labels': model.labels,
            }, os.path.join(output_dir, f'params_{i}_{fold_id}.pt'))

    df_result = pd.concat(df_result)
    df_result = df_result.groupby(['complex', 'mutstr', 'num_muts']).mean().reset_index()
    return df_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default='./RDE_linear_skempi')
    parser.add_argument('-s', '--seed', type=int, default=2020)
    parser.add_argument('-t', '--num_trials', type=int, default=10)
    parser.add_argument('-i', '--iters', type=int, default=2000)
    parser.add_argument('-n', '--num_folds', type=int, default=3)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-a', '--alternate', type=int, default=None)
    parser.add_argument('--single', action='store_true', default=False)    
    args = parser.parse_args()

    with open(args.result, 'rb') as f:
        result = pickle.load(f)
    results = run_calibration(result)
    results.to_csv(os.path.join(args.output_dir, 'results.csv'))
