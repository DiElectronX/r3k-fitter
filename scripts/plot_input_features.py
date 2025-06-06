import os
import sys
import yaml
import argparse
import pickle
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
import ROOT
import ROOT.RooFit as rf

sys.path.insert(1, str(Path('..').resolve()))
from fit_models import FitModel
from utils import *
from do_fit import *
from do_mc_yields_and_effs import *


plot_cfgs = {
    'L1pt'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,40,50), 'label' : r'$p_{T}(e_{1})$ [GeV]'},
    'L2pt'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,40,50), 'label' : r'$p_{T}(e_{2})$ [GeV]'},
    'Bprob'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,1,25),},
    'BsLxy'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,250,50),},
    'L2iso/L2pt'     : {'norm' : True, 'logy' : True, 'bins' : np.linspace(0,100,50),},
    'Bcos'           : {'norm' : True, 'logy' : True, 'bins' : np.linspace(.9,1,50),},
    'Kiso/Kpt'       : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,60,50),},
    'LKdz'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,1,50),},
    'LKdr'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,3,50),},
    'Passymetry'     : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-1,1,50),},
    'Kip3d/Kip3dErr' : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-6,6,50),},
    'L1id'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-6,6,25),},
    'L2id'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-6,6,25),},
    'L1iso'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,100,50),},
    'L2iso'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,100,50),},
    'bdt_score'      : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-20,20,50),},
    'default'        : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,100,50),},
}


def plot_inputs(dataset_params,output_params,fit_params,args):
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

    # Validate the branches given in args.branches
    """
    available_branches = list(df.GetColumnNames().)
    for branch in args.branches:
        assert branch in available_branches, f"Branch {branch} not found. Available branches: {available_branches}"
    """

    # Apply filters based on full_mass_range, bdt_score_cut, and ll_mass_range
    df_filtered = df.Filter(f"{b_mass_branch} >= {full_mass_range[0]} && {b_mass_branch} <= {full_mass_range[1]}")
    df_filtered = df_filtered.Filter(f"{ll_mass_branch} >= {ll_mass_range[0]} && {ll_mass_branch} <= {ll_mass_range[1]}")
    df_filtered = df_filtered.Filter(f"KLmassD0 > 2.")
    df_filtered_wBDT = df_filtered.Filter(f"{score_branch} >= {bdt_score_cut}")
    
    # Extract branches to plot
    branches_to_plot = args.branches
    data = {branch: np.array(df_filtered.AsNumpy(columns=[branch])[branch]) for branch in branches_to_plot}
    data_wBDT = {branch: np.array(df_filtered_wBDT.AsNumpy(columns=[branch])[branch]) for branch in branches_to_plot}
    weights = np.array(df_filtered.AsNumpy(columns=[weight_branch])[weight_branch])
    weights_wBDT = np.array(df_filtered_wBDT.AsNumpy(columns=[weight_branch])[weight_branch])

    # Create output directory for plots
    output_dir = "input_feature_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save each branch
    for branch in branches_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_cfg = plot_cfgs[branch]

        norm = weights / weights.sum() if plot_cfg['norm'] else weights
        norm_wBDT = weights_wBDT / weights_wBDT.sum() if plot_cfg['norm'] else weights_wBDT
        hist, bin_edges = np.histogram(data[branch], bins=plot_cfg['bins'], weights=norm)
        hist_wBDT, _ = np.histogram(data_wBDT[branch], bins=plot_cfg['bins'], weights=norm_wBDT)

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax.errorbar(bin_centers, hist, yerr=0, marker='', drawstyle='steps-mid', label='No BDT Cut')
        ax.errorbar(bin_centers, hist_wBDT, yerr=0, marker='', drawstyle='steps-mid', label='W/ BDT Cut')

        ax.set_xlabel(plot_cfg.get('label', branch), loc='right')
        ax.set_ylabel('nEntries [A.U.]', loc='top')
        ax.legend()
        output_path = os.path.join(output_dir, f"{mode}_{branch}_distribution.png")
        fig.savefig(output_path)
        fig.savefig(output_path.replace('.png','.pdf'))
        plt.close()


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    if args.mode=='all':
        args.mode = 'jpsi'
        set_mode(dataset_params,output_params,fit_params,args)
        plot_inputs(dataset_params,output_params,fit_params,args)

        args.mode = 'psi2s'
        set_mode(dataset_params,output_params,fit_params,args)
        plot_inputs(dataset_params,output_params,fit_params,args)

        args.mode = 'lowq2'
        set_mode(dataset_params,output_params,fit_params,args)
        plot_inputs(dataset_params,output_params,fit_params,args)

    elif args.mode=='jpsi':
        set_mode(dataset_params,output_params,fit_params,args)
        plot_inputs(dataset_params,output_params,fit_params,args)
    elif args.mode=='psi2s':
        set_mode(dataset_params,output_params,fit_params,args)
        plot_inputs(dataset_params,output_params,fit_params,args)
    elif args.mode=='lowq2':
        set_mode(dataset_params,output_params,fit_params,args)
        plot_inputs(dataset_params,output_params,fit_params,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='../fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file to plot')
    parser.add_argument('-b', '--branches', type=str, nargs='+', required=True, help='branches to plot')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which scan to perform')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    args = parser.parse_args()

    main(args)
