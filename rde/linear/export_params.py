import os
import argparse
import pandas as pd
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./RDE_linear_skempi')
    parser.add_argument("--output", type=str, default="data/rdelinear_params.csv")
    args = parser.parse_args()

    results = []

    for fname in os.listdir(args.result_dir):
        if fname.startswith("params_") and fname.endswith('.pt'):
            ckpt = torch.load(os.path.join(args.result_dir, fname), map_location='cpu')
            pdict = ckpt["state_dict"]
            row = {"name": fname}
            for label, coef in zip(ckpt["labels"], pdict["regr_coef_"].tolist()):
                row[label] = coef
            row["bias"] = pdict["regr_bias"].item()
            row["aa_coef"] = (torch.nn.functional.softplus(pdict["aa_coef.weight"]) + 0.5)[..., 0].tolist()
            row["aa_bias"] = pdict["aa_ref.weight"][..., 0].tolist()

            postfix = fname.split(".")[0][7:]
            corr_df = pd.read_csv(os.path.join(args.result_dir, f"corr_{postfix}.csv"), index_col=0)
            row["grouped_pearson"] = corr_df["pearson"].mean()
            row["grouped_spearman"] = corr_df["spearman"].mean()
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(args.output, float_format="%.4f", index=False)
    print(df)
