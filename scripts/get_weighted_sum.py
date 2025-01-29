import ROOT
import argparse
import yaml

def main(args):
    # Read and parse the YAML config file
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    # Extract relevant parameters based on the mode
    mode = args.mode
    tree_name = dataset_params.tree_name
    b_mass_branch = dataset_params.b_mass_branch
    ll_mass_branch = dataset_params.ll_mass_branch
    score_branch = dataset_params.score_branch
    weight_branch = dataset_params.mc_weight_branch
    full_mass_range = fit_params.full_mass_range
    bdt_score_cut = fit_params.bdt_score_cut
    ll_mass_range = fit_params.regions[mode]['ll_mass_range']

    # Read the ROOT file and TTree into an RDataFrame
    root_file = args.input
    df = ROOT.RDataFrame(tree_name, root_file)

    # Apply filters based on full_mass_range, bdt_score_cut, and ll_mass_range
    df_filtered = df.Filter(f"{b_mass_branch} >= {full_mass_range[0]} && {b_mass_branch} <= {full_mass_range[1]}")
    df_filtered = df_filtered.Filter(f"{score_branch} >= {bdt_score_cut}")
    df_filtered = df_filtered.Filter(f"{ll_mass_branch} >= {ll_mass_range[0]} && {ll_mass_branch} <= {ll_mass_range[1]}")
    df_filtered = df_filtered.Filter(f"KLmassD0 > 2.")

    # Sum the entries in the specified branch
    sum_weights = round(df_filtered.Count().GetValue())
    #sum_weights = round(df_filtered.Sum(weight_branch).GetValue())

    # Print the sum
    print(f'Sum of {weight_branch} in {tree_name}: {sum_weights}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file to count')
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, help='which region')
    args = parser.parse_args()

    main(args)
