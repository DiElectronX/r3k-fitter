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


def plot_mass_sculpting(dataset_params,output_params,fit_params,args):
    # Extract relevant parameters based on the mode
    mode = args.mode
    tree_name = dataset_params.tree_name
    b_mass_branch = dataset_params.b_mass_branch
    ll_mass_branch = dataset_params.ll_mass_branch
    score_branch = dataset_params.score_branch
    weight_branch = dataset_params.mc_weight_branch
    full_mass_range = fit_params.full_mass_range
    nom_bdt_cut = fit_params.bdt_score_cut
    ll_mass_range = fit_params.regions[mode]['ll_mass_range']

    # Read the ROOT file and TTree into an RDataFrame
    df = ROOT.RDataFrame(tree_name, dataset_params.samesign_data_file)

    # Apply filters based on full_mass_range and ll_mass_range
    df_filtered = df.Filter(f"{b_mass_branch} >= {full_mass_range[0]} && {b_mass_branch} <= {full_mass_range[1]}")
    df_filtered = df_filtered.Filter(f"{ll_mass_branch} >= {ll_mass_range[0]} && {ll_mass_branch} <= {ll_mass_range[1]}")
    df_filtered = df_filtered.Filter(f"KLmassD0 > 2.")
    df_plot = df_filtered

    # Create output directory for plots
    output_dir = Path('.') / 'mass_sculpting_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save each branch
    bdt_cut_range = np.linspace(0,4,5)
    bdt_cut_range = np.sort(np.append(bdt_cut_range,nom_bdt_cut))

    fig, ax = plt.subplots(figsize=(8, 6))
    binning = np.linspace(4.7, 5.7, 30)
    for bdt_cut in loop_wrapper(bdt_cut_range,args):
        df_plot =  df_plot.Filter(f"{score_branch} >= {bdt_cut}")
        data = np.array(df_plot.AsNumpy(columns=[b_mass_branch])[b_mass_branch], dtype=float)
        
        hist, bin_edges = np.histogram(data, bins=binning)
        hist_norm = hist / len(data)
        #hist_err = np.zeros_like(hist)
        hist_err = np.divide(np.ones_like(hist, dtype=float), np.sqrt(hist), out=np.zeros_like(hist_norm), where=hist_norm!=0) / data.size

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plot_label = f'Nominal Analysis BDT Cut (>{round(bdt_cut,2)})' if bdt_cut==nom_bdt_cut else f'BDT Cut >{round(bdt_cut,2)}'

        ax.errorbar(
            bin_centers, 
            hist_norm, 
            yerr=hist_err, 
            marker='', 
            drawstyle='steps-mid', 
            label=plot_label
        )

    ax.set_xlabel('B Candidate Mass [GeV]', loc='right')
    ax.set_ylabel('nCandidates [A.U.]', loc='top')
    ax.set_ylim(bottom=0)
    ax.legend()
    output_path = output_dir / f'{mode}_samesign_mass_sculpting_distribution.pdf'
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight')
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
        plot_mass_sculpting(dataset_params,output_params,fit_params,args)

        args.mode = 'psi2s'
        set_mode(dataset_params,output_params,fit_params,args)
        plot_mass_sculpting(dataset_params,output_params,fit_params,args)

        args.mode = 'lowq2'
        set_mode(dataset_params,output_params,fit_params,args)
        plot_mass_sculpting(dataset_params,output_params,fit_params,args)

    elif args.mode=='jpsi':
        set_mode(dataset_params,output_params,fit_params,args)
        plot_mass_sculpting(dataset_params,output_params,fit_params,args)
    elif args.mode=='psi2s':
        set_mode(dataset_params,output_params,fit_params,args)
        plot_mass_sculpting(dataset_params,output_params,fit_params,args)
    elif args.mode=='lowq2':
        set_mode(dataset_params,output_params,fit_params,args)
        plot_mass_sculpting(dataset_params,output_params,fit_params,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='../fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which scan to perform')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    args = parser.parse_args()

    main(args)
