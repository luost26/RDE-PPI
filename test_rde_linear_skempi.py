from rde.linear import __main__

if __name__ == '__main__':
    """
    Commandline Arguments:
    --ckpt:     Trained RDE model weight path, default: ./trained_models/RDE.pt.
    --device:   Default: cuda.
    --skempi_dir:   SKEMPI dataset path: default: ./data/SKEMPI_v2/PDBs
    --output_dir:   Intermediate and final results directory path.
                    Default: ./RDE_linear_skempi
    """
    __main__.main()
