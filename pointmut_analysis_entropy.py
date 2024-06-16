import argparse
import copy
import torch
import itertools

import pandas as pd
from Bio.PDB.Polypeptide import one_to_index, index_to_one
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from tqdm.auto import tqdm

from rde.utils.protein.constants import num_chi_angles
from rde.utils.protein.parsers import parse_biopython_structure
from rde.utils.misc import load_config
from rde.utils.train import CrossValidation
from rde.models.rde_ddg import DDG_RDE_Network
from rde.utils.transforms import SelectAtom


def _mask_select_data(data, mask):
    def _mask_select(v, mask):
        if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
            return v[mask]
        elif isinstance(v, list) and len(v) == mask.size(0):
            return [v[i] for i, b in enumerate(mask) if b]
        else:
            return v

    return {k: _mask_select(v, mask) for k, v in data.items()}


def _index_select_data(data, index):
    def _index_select(v, index, n):
        if isinstance(v, torch.Tensor) and v.size(0) == n:
            return v[index]
        elif isinstance(v, list) and len(v) == n:
            return [v[i] for i in index]
        else:
            return v

    return {k: _index_select(v, index, data["aa"].size(0)) for k, v in data.items()}


def _pad_data(data, patch_size=128):
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

    ref_length = data["aa"].shape[0]
    if ref_length >= patch_size:
        return data

    data_padded = {}
    for k, v in data.items():
        if len(v) == ref_length:
            data_padded[k] = _pad_last(v, patch_size)
        else:
            data_padded[k] = v
    return data_padded


def _get_Cbeta_positions(pos_atoms, mask_atoms):
    """
    Args:
        pos_atoms:  (L, A, 3)
        mask_atoms: (L, A)
    """
    from rde.utils.protein.constants import BBHeavyAtom

    L = pos_atoms.size(0)
    pos_CA = pos_atoms[:, BBHeavyAtom.CA]  # (L, 3)
    if pos_atoms.size(1) < 5:
        return pos_CA
    pos_CB = pos_atoms[:, BBHeavyAtom.CB]
    mask_CB = mask_atoms[:, BBHeavyAtom.CB, None].expand(L, 3)
    return torch.where(mask_CB, pos_CB, pos_CA)


def _load_structure(pdb_path):
    if pdb_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif pdb_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unknown file type.")

    structure = parser.get_structure(None, pdb_path)
    data, seq_map = parse_biopython_structure(structure[0])
    return data, seq_map


def _parse_mutations(ligand_chains, seq_map, mutations):
    parsed = []
    for m in mutations:
        wt, ch, mt = m[0], m[1], m[-1]
        if ch not in ligand_chains:
            print(f"Chain {ch} not in ligand chains. Skipping mutation {m}.")
            continue
        seq = int(m[2:-1])
        pos = (ch, seq, " ")
        if pos not in seq_map:
            continue

        if mt == "*":
            for mt_idx in range(20):
                mt = index_to_one(mt_idx)
                if mt == wt:
                    continue
                parsed.append(
                    {
                        "position": pos,
                        "wt": wt,
                        "mt": mt,
                    }
                )
        else:
            parsed.append(
                {
                    "position": pos,
                    "wt": wt,
                    "mt": mt,
                }
            )
    return parsed


def make_mutstr(mut):
    return '{}{}{}{}'.format(
        mut['wt'],
        mut['position'][0],
        mut['position'][1],
        mut['mt']
    )


def get(
    data,
    seq_map,
    mutation,
    receptor_group,
    ligand_group,
    group,
    state,
    patch_size=128,
):
    assert group in ("ligand", "receptor", "complex")
    assert state in ("mt", "wt")
    data = SelectAtom('backbone+CB')(copy.deepcopy(data))

    mutation_flag = torch.zeros((data["aa"].shape[0]), dtype=torch.bool)
    chi_corrupt = data["chi"].clone()
    mut_beta_positions = []

    position = mutation["position"]
    seq_idx = seq_map[position]
    mutation_flag[seq_idx] = True
    chi_corrupt[seq_idx] = 0.0
    data["mutation_flag"] = mutation_flag

    # Mutate the protein
    if state == "mt":
        mtype = one_to_index(mutation["mt"])
        data["aa"][seq_idx] = mtype
        data["chi"][seq_idx] = 0.0
        data["chi_alt"][seq_idx] = 0.0
        data["chi_mask"][seq_idx] = False
        data["chi_mask"][seq_idx, : num_chi_angles[mtype]] = True

    pos_atom = data["pos_heavyatom"][seq_idx, :5]  # (5, 3)
    msk_atom = data["mask_heavyatom"][seq_idx, :5]  # (5,)
    beta_pos = pos_atom[4] if msk_atom[4].item() else pos_atom[1]
    mut_beta_positions.append(beta_pos)

    mut_beta_positions = torch.stack(mut_beta_positions)  # (M, 3)
    data["chi_masked_flag"] = mutation_flag
    data["chi_corrupt"] = chi_corrupt

    # For each residue, compute the distance to the closest mutated residue
    beta_pos = _get_Cbeta_positions(data["pos_heavyatom"], data["mask_heavyatom"])
    pw_dist = torch.cdist(beta_pos, mut_beta_positions)  # (N, M)
    dist_to_mut = pw_dist.min(dim=1)[0]  # (N, )
    data["dist_to_mut"] = dist_to_mut

    # Flags
    receptor_flag = torch.BoolTensor([(c in receptor_group) for c in data["chain_id"]])
    ligand_flag = torch.BoolTensor([(c in ligand_group) for c in data["chain_id"]])
    data["receptor_flag"] = receptor_flag
    data["ligand_flag"] = ligand_flag

    # Add the information of closest residues in the receptor
    rec_idx = torch.logical_and(dist_to_mut <= 8.0, receptor_flag).nonzero().flatten()
    nbr_rec_flag = torch.zeros((data["aa"].shape[0]), dtype=torch.bool)
    nbr_rec_flag[rec_idx] = True
    data["nbr_rec_flag"] = nbr_rec_flag

    # Add the information of closest residues in the ligand
    lig_idx = torch.logical_and(dist_to_mut <= 8.0, ligand_flag).nonzero().flatten()
    nbr_lig_flag = torch.zeros((data["aa"].shape[0]), dtype=torch.bool)
    nbr_lig_flag[lig_idx] = True
    data["nbr_lig_flag"] = nbr_lig_flag

    # Select the chain group
    if group == "ligand":
        group_mask = torch.BoolTensor([(c in ligand_group) for c in data["chain_id"]])
    elif group == "receptor":
        group_mask = torch.BoolTensor([(c in receptor_group) for c in data["chain_id"]])
    else:
        group_mask = torch.ones((data["aa"].shape[0]), dtype=torch.bool)
    data = _mask_select_data(data, group_mask)

    # Patch or pad
    patch_idx = data["dist_to_mut"].argsort()[:patch_size]
    data = _index_select_data(data, patch_idx)
    data = _pad_data(data, patch_size=patch_size)

    # Add tags
    data["group"] = group
    data["state"] = state
    return data


def _batchify(data, device):
    batch = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, list):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def load_rdelinear_params(path, device):

    def _parse_list_str(s):
        return [float(x.strip()) for x in s.strip("[]").split(",")]

    df = pd.read_csv(path).sort_values("grouped_spearman", ascending=False).reset_index(drop=True)
    row = df.iloc[0]

    params = {
        "aa_coef": torch.tensor(_parse_list_str(row["aa_coef"]), dtype=torch.float32, device=device),
        "aa_bias": torch.tensor(_parse_list_str(row["aa_bias"]), dtype=torch.float32, device=device),
        "H_lig_ub_wt": row["H_lig_ub_wt"],
        "H_lig_ub_mt": row["H_lig_ub_mt"],
        "H_lig_b_wt": row["H_lig_b_wt"],
        "H_lig_b_mt": row["H_lig_b_mt"],
        "H_rec_ub": row["H_rec_ub"],
        "H_rec_b_wt": row["H_rec_b_wt"],
        "H_rec_b_mt": row["H_rec_b_mt"],
        "bias": row["bias"],
    }
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("-o", "--output", type=str, default="pm_results_entropy.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--params", type=str, default="data/rdelinear_params.csv")
    args = parser.parse_args()
    config, _ = load_config(args.config)

    print("[NOTE] This is a prototype script for entropy-based ddG prediction. It is not optimized for speed.")

    data, seq_map = _load_structure(config.pdb)
    mutations = _parse_mutations(
        ligand_chains=config.ligand_chains,
        seq_map=seq_map,
        mutations=config.mutations,
    )

    ckpt = torch.load(config.checkpoint, map_location='cpu')
    cv_mgr = CrossValidation(model_factory=DDG_RDE_Network, config=ckpt['config'], num_cvfolds=3)
    cv_mgr.load_state_dict(ckpt['model'])
    cv_mgr.to(args.device)
    # Extract the RDE network inside the DDG_RDE_Network
    model = cv_mgr.models[0].rde

    params = load_rdelinear_params(args.params, device=args.device)

    results = []
    try:
        for mutation in tqdm(mutations):
            row = {}
            for group, state in itertools.product(["ligand", "receptor", "complex"], ["wt", "mt"]):
                mutstr = make_mutstr(mutation)

                batch = _batchify(get(
                    data=data,
                    seq_map=seq_map,
                    mutation=mutation,
                    receptor_group=config.receptor_chains,
                    ligand_group=config.ligand_chains,
                    group=group,
                    state=state,
                ), args.device)
                entropy_original = model.entropy(batch, n_samples=200)
                ent_coef = params["aa_coef"][batch["aa"]]
                ent_bias = params["aa_bias"][batch["aa"]]
                entropy = (ent_coef * entropy_original + ent_bias)[0]

                # Consider only the mutated residues as ligand part, consistent with the calibration procedure
                entropy_ligand = entropy[batch["mutation_flag"][0]].sum().item()
                entropy_receptor = entropy[batch["nbr_rec_flag"][0]].sum().item()

                if group == "ligand" and state == "wt":
                    row["H_lig_ub_wt"] = entropy_ligand
                elif group == "ligand" and state == "mt":
                    row["H_lig_ub_mt"] = entropy_ligand
                elif group == "receptor":
                    row["H_rec_ub"] = entropy_receptor
                elif group == "complex" and state == "wt":
                    row["H_lig_b_wt"] = entropy_ligand
                    row["H_rec_b_wt"] = entropy_receptor
                elif group == "complex" and state == "mt":
                    row["H_lig_b_mt"] = entropy_ligand
                    row["H_rec_b_mt"] = entropy_receptor

            row["ddG_pred"] = sum(row[k] * params[k] for k in row.keys()) + params["bias"]
            results.append({"mutstr": mutstr, **row})
    except KeyboardInterrupt:
        print("Interrupted. Saving results.")

    df = pd.DataFrame(results)
    df['rank'] = df['ddG_pred'].rank() / len(df)
    print(df)
    df.to_csv(args.output, float_format="%.4f", index=False)

    if 'interest' in config and config.interest:
        print(df[df['mutstr'].isin(config.interest)])
